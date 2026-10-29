"""Interpretable LoRA: train a LoRA on Qwen3-1.7B *as* a parameter decomposition (VPD-style).

Every target nn.Linear W (frozen, always on) gets C rank-one delta components with a learned
causal-importance gate g_c(x) in [0, 1]:

    out = W x + sum_c m_c * (x . V_c) U_c            m_c = g_c + (1 - g_c) * s_c,   s_c in [0, 1]

"full" mode (m = 1) is an ordinary rank-C LoRA with B = U^T, A = V^T. There is no target model to be
faithful to, so instead of VPD's faithfulness loss the unmasked model is trained directly on the
fine-tuning data. Per step, on an SFT batch (target) and a pretraining-text batch (nontarget):

    L = CE_full(SFT answers)                                     learn the behaviour
      + c_stoch * L_masked(stochastic masks, k-subset routing)   masked reconstruction (VPD)
      + c_adv   * L_masked(persistent-PGD masks)                 adversarial reconstruction (VPD)
      + c_nt    * KL(base || masked)            on nontarget     leave the base alone elsewhere
      + lam * (L_imp(target) + nt_ratio * L_imp(nontarget))      importance + frequency minimality (VPD)

    L_masked = CE of the masked model on the SFT labels               (--masked_loss ce,  "option A")
             = KL(full.detach() || masked) at every real position     (--masked_loss kl,  "option B")
With --alternate each step is two updates (option B as two interleaved trainings): (A) the LoRA alone takes an
optimiser step on CE_full + c_nt * KL(base || full) on nontarget; (B) the masked losses take a step against the
*updated* full model as target, with their own Adam state. --masked_params ci keeps step B off U, V (only the CI
function learns from the masked losses), the fallback if the co-trained target collapses the LoRA.
With --teacher <ckpt> (a finished plain LoRA) the reference model is fixed: the full loss is KL(teacher || full)
and the masked KL targets the teacher. Without it, option B's KL has a trivial minimum at delta = 0 (the
co-trained teacher moves with the student), which in practice shrinks the LoRA and fights the CE.

Written from scratch after goodfire-ai/param-decomp (nano_param_decomp/run.py): same leaky hard
sigmoids, mask parametrisation, uniform-k-subset routing, persistent PGD with Adam, and the
L_p + log2-frequency minimality loss. Differences: no faithfulness / delta component (nothing to
match), base weight always on, CE on the unmasked model, nontarget term against the base model, CI
function is a global MLP over per-module projections of the layer inputs (not a transformer).

Stages
  train   python gated_lora.py train --data /workspace/lora/data --out runs/ce_c32
  eval    python gated_lora.py eval  --data ... --out runs/ce_c32      -> eval.json, components.json
  export  python gated_lora.py export --out runs/ce_c32                -> runs/ce_c32/peft_adapter (all gates = 1)
"""
import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
dev = "cuda" if torch.cuda.is_available() else "cpu"
ALPHA = 0.01  # leaky slope of the hard sigmoids
PROJ = {"q": "self_attn.q_proj", "k": "self_attn.k_proj", "v": "self_attn.v_proj", "o": "self_attn.o_proj",
        "gate": "mlp.gate_proj", "up": "mlp.up_proj", "down": "mlp.down_proj"}
RE_BOLD = re.compile(r"\*\*[^*\n]+\*\*")  # lls_owl._RE_BOLD


# ------------------------------------------------------------------ sigmoids (VPD)
class _LowerLeaky(torch.autograd.Function):
    """forward clamp(x,0,1); backward: pass-through on (0,1); for x<=0 only let gradients that want to
    raise the gate through (scaled by alpha) so dead components can come back; blocked above 1."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp(0.0, 1.0)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        zero = torch.zeros_like(g)
        return torch.where(x <= 0, torch.where(g < 0, ALPHA * g, zero), torch.where(x <= 1, g, zero))


def lower_leaky(x):
    return _LowerLeaky.apply(x)


def upper_leaky(x):
    return torch.where(x > 1, 1 + ALPHA * (x - 1), x.clamp(0.0, 1.0))


# ------------------------------------------------------------------ gated LoRA linear
class GatedLoRALinear(nn.Module):
    """Frozen nn.Linear plus C gated rank-one delta components.
    modes: base (W only) | full (W + V U, caches the input for the CI function) | masked (W + masked V U)."""

    def __init__(self, base: nn.Linear, C: int):
        super().__init__()
        d_out, d_in = base.weight.shape
        self.base = base
        self.C = C
        self.V = nn.Parameter(torch.randn(d_in, C, device=base.weight.device) / math.sqrt(d_in))
        self.U = nn.Parameter(torch.zeros(C, d_out, device=base.weight.device))  # LoRA init: start at the base model
        self.mode, self.mask, self.last_input = "full", None, None
        self.Vt = self.Ut = None  # frozen reference ("teacher") delta, see load_teacher

    def forward(self, x):
        out = self.base(x)
        if self.mode == "base":
            return out
        if self.mode == "teacher":
            return out + (x @ self.Vt.to(x.dtype)) @ self.Ut.to(x.dtype)
        if self.mode == "full":
            self.last_input = x.detach()
            return out + (x @ self.V.to(x.dtype)) @ self.U.to(x.dtype)
        inner = x @ self.V.to(x.dtype)  # (B, T, C)
        return out + (inner * self.mask.to(x.dtype)) @ self.U.to(x.dtype)


def install(model, C, proj_keys, layers=None):
    for p in model.parameters():
        p.requires_grad_(False)
    wrappers = {}
    for i, layer in enumerate(model.model.layers):
        if layers is not None and i not in layers:
            continue
        for key in proj_keys:
            path = PROJ[key]
            parent = layer.get_submodule(path.rsplit(".", 1)[0])
            attr = path.rsplit(".", 1)[1]
            lin = getattr(parent, attr)
            w = GatedLoRALinear(lin, C)
            setattr(parent, attr, w)
            wrappers[f"model.layers.{i}.{path}"] = w
    return wrappers


def load_teacher(wrappers, path):
    """Install a finished run's components (e.g. the plain LoRA) as a frozen reference delta: mode "teacher"
    gives the reference model's logits, used as the distillation target for the full and masked losses."""
    st = torch.load(path, map_location=dev)
    for n, w in wrappers.items():
        w.Vt = st["components"][n]["V"].to(dev).requires_grad_(False)
        w.Ut = st["components"][n]["U"].to(dev).requires_grad_(False)


def set_mode(wrappers, mode, masks=None):
    for n, w in wrappers.items():
        w.mode = mode
        w.mask = None if masks is None else masks[n]


# ------------------------------------------------------------------ causal-importance function
def _apply_ci(net, x):
    """rms-norm x, cast to the net's dtype one module at a time (autocast handles the GPU matmul in bf16)."""
    w = next(net.parameters())
    return net(rms_norm(x).to(w.dtype))


def rms_norm(x):
    """RMS-normalise in the input dtype (bf16 on GPU) with an fp32 variance, so the CI never holds fp32 copies
    of every layer input at once (196 modules x B x T x d_in would be several GB)."""
    var = x.float().pow(2).mean(-1, keepdim=True)
    return x * torch.rsqrt(var + 1e-6).to(x.dtype)


class GlobalCI(nn.Module):
    """g = sigmoid-ish( MLP( concat_n P_n rmsnorm(x_n) ) ): every module's layer input is RMS-normed,
    projected to `proj` dims by a per-module linear, concatenated across modules, then a shared MLP
    gives one logit per component of every module. Cheap stand-in for VPD's global CI transformer."""

    def __init__(self, d_in, C, proj=64, hidden=512):
        super().__init__()
        self.names = sorted(d_in)
        self.C = C
        self.proj = nn.ModuleDict({n.replace(".", "-"): nn.Linear(d_in[n], proj, bias=False) for n in self.names})
        self.mlp = nn.Sequential(nn.Linear(proj * len(self.names), hidden), nn.GELU(),
                                 nn.Linear(hidden, sum(C[n] for n in self.names)))
        with torch.no_grad():  # start with every gate at ~0.5 so components can be used from step 0
            self.mlp[-1].bias.fill_(0.5)
        self.splits = [C[n] for n in self.names]

    def forward(self, acts):
        z = torch.cat([_apply_ci(self.proj[n.replace(".", "-")], acts[n]) for n in self.names], -1)
        pre = self.mlp(z).split(self.splits, -1)
        return dict(zip(self.names, pre))


class LayerwiseCI(nn.Module):
    """One small MLP per module on that module's own (RMS-normed) input."""

    def __init__(self, d_in, C, proj=64, hidden=None):
        super().__init__()
        self.names = sorted(d_in)
        self.nets = nn.ModuleDict({n.replace(".", "-"): nn.Sequential(nn.Linear(d_in[n], proj), nn.GELU(), nn.Linear(proj, C[n]))
                                   for n in self.names})
        with torch.no_grad():
            for net in self.nets.values():
                net[-1].bias.fill_(0.5)

    def forward(self, acts):
        return {n: _apply_ci(self.nets[n.replace(".", "-")], acts[n]) for n in self.names}


# ------------------------------------------------------------------ masks + losses
MASK_DIST = "uniform"  # set from --mask_dist, see make_masks


def sample_sources(gates):
    """Random sources s in [0,1] for every piece, shaped like the gates. uniform: s ~ U(0,1) per piece (VPD; with
    thousands of pieces the surviving fraction is ~50% every draw). bernoulli: per token a keep-probability
    q ~ U(0,1) shared by all pieces and modules, then s ~ Bernoulli(q), so the surviving fraction ranges over [0,1]."""
    if MASK_DIST == "bernoulli":
        g0 = next(iter(gates.values()))
        q = torch.rand(*g0.shape[:-1], 1, device=g0.device, dtype=g0.dtype)
        return {n: (torch.rand_like(g) < q).to(g.dtype) for n, g in gates.items()}
    return {n: torch.rand_like(g) for n, g in gates.items()}


def make_masks(ci_pre, src=None, routing=None):
    """Sufficiency masks m = g + (1-g) s: important pieces always on, any amount of the others; the model should
    match the full model. Where routing is False the module runs the full delta (m = 1) at that position, like
    VPD's k-subset routing against the target model."""
    gates = {n: lower_leaky(z) for n, z in ci_pre.items()}
    src = sample_sources(gates) if src is None else src
    masks = {}
    for n, g in gates.items():
        m = g + (1 - g) * src[n]
        if routing is not None:
            m = torch.where(routing[n][..., None], m, torch.ones_like(m))
        masks[n] = m
    return masks


def make_comp_masks(ci_pre, src=None, routing=None):
    """Necessity masks, the mirror of make_masks: m = (1-g) s. Important pieces always off, any amount of the
    others; the model should match the BASE model. s = 1 is the fixed 1-g complement, s = 0 the base itself.
    Where routing is False the module runs the base (m = 0)."""
    gates = {n: lower_leaky(z) for n, z in ci_pre.items()}
    src = sample_sources(gates) if src is None else src
    masks = {}
    for n, g in gates.items():
        m = (1 - g) * src[n]
        if routing is not None:
            m = torch.where(routing[n][..., None], m, torch.zeros_like(m))
        masks[n] = m
    return masks


def k_subset_routing(names, lead, device):
    M = len(names)
    k = torch.randint(1, M + 1, lead, device=device)
    ranks = torch.rand(M, *lead, device=device).argsort(0).argsort(0)
    return {n: ranks[i] < k for i, n in enumerate(names)}


def minimality(ci_pre, p, beta, eps=1e-12):
    """VPD importance + frequency minimality: sum_c mean_bt[(g+eps)^p] (1 + beta log2(1 + sum_bt[(g+eps)^p]))."""
    tot = 0.0
    for z in ci_pre.values():
        v = (upper_leaky(z) + eps) ** p
        s = v.flatten(0, -2).sum(0)
        m = s / v[..., 0].numel()
        tot = tot + (m + beta * m * torch.log2(1 + s)).sum()
    return tot


def ce_loss(logits, labels):
    return F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), labels[:, 1:].flatten(), ignore_index=-100)


def kl_loss(ref_logits, logits, valid):
    """KL(ref || model) averaged over valid positions; ref is detached. Computed one sequence at a time so the
    fp32 vocab-sized transients (B x T x 152k) never exist for the whole batch at once."""
    tot = 0.0
    for b in range(logits.shape[0]):
        lp = F.log_softmax(ref_logits[b].detach().float(), -1)
        lq = F.log_softmax(logits[b].float(), -1)
        tot = tot + ((lp.exp() * (lp - lq)).sum(-1) * valid[b]).sum()
    return tot / valid.sum()


class PersistentPGD:
    """Adversarial mask sources per (batch slot, position, component) that persist across steps; Adam ascent
    on the masked loss (VPD). `warmup` does `inner` ascent steps on the current batch before the loss term is
    taken; after the main backward the sources take one more step with the gradient of that term."""

    def __init__(self, wrappers, B, T, lr, beta1=0.5, beta2=0.99, eps=1e-8, inner=2):
        self.src = {n: torch.rand(B, T, w.C, device=dev) for n, w in wrappers.items()}
        self.m = {n: torch.zeros_like(s) for n, s in self.src.items()}
        self.v = {n: torch.zeros_like(s) for n, s in self.src.items()}
        self.lr, self.b1, self.b2, self.eps, self.inner, self.t = lr, beta1, beta2, eps, inner, 0

    def sources(self, requires_grad):
        return {n: s.detach().requires_grad_(requires_grad) for n, s in self.src.items()}

    def step(self, grads, lr):
        self.t += 1
        bc1, bc2 = 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
        with torch.no_grad():
            for n, g in grads.items():
                m, v = self.m[n], self.v[n]
                m.mul_(self.b1).add_(g, alpha=1 - self.b1)
                v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
                self.src[n] = (self.src[n] + lr * (m / bc1) / ((v / bc2).sqrt() + self.eps)).clamp_(0, 1)


# ------------------------------------------------------------------ data
def load_data(d, split):
    sft = torch.load(Path(d) / f"sft_{split}.pt")
    nt = torch.load(Path(d) / f"nontarget_{split}.pt")
    return sft, nt


def sft_batch(items, idx, T, pad_id):
    ids = torch.full((len(idx), T), pad_id, dtype=torch.long)
    labels = torch.full((len(idx), T), -100, dtype=torch.long)
    valid = torch.zeros(len(idx), T, dtype=torch.bool)
    for r, i in enumerate(idx):
        it = items[i]
        n = min(len(it["ids"]), T)
        ids[r, :n], labels[r, :n], valid[r, :n] = it["ids"][:n], it["labels"][:n], True
    valid[:, 0] = False  # no prediction for the first token
    return ids.to(dev), labels.to(dev), valid.to(dev)


# ------------------------------------------------------------------ model
def build(args):
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
    model.config.use_cache = False
    layers = None if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    wrappers = install(model, args.C, args.proj.split(","), layers)
    d_in = {n: w.base.in_features for n, w in wrappers.items()}
    C = {n: w.C for n, w in wrappers.items()}
    ci = (GlobalCI if args.ci == "global" else LayerwiseCI)(d_in, C, proj=args.ci_proj, hidden=args.ci_hidden).to(dev)
    if args.teacher:
        load_teacher(wrappers, args.teacher)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()  # HF only checkpoints in train mode; Qwen3 has no dropout
    return tok, model, wrappers, ci


def forward(model, wrappers, ids, mode, masks=None):
    """Sets the wrappers' mode and leaves it set: with gradient checkpointing the forward is recomputed during
    backward, so the mode/masks must still be in place then. The next forward() call switches it."""
    set_mode(wrappers, mode, masks)
    return model(input_ids=ids).logits


def ci_of(ci, wrappers):
    return ci({n: w.last_input for n, w in wrappers.items()})


def save_ckpt(path, wrappers, ci, args, step):
    torch.save({"components": {n: {"U": w.U.detach().cpu(), "V": w.V.detach().cpu()} for n, w in wrappers.items()},
                "ci": ci.state_dict(), "args": vars(args), "step": step}, path)


def load_ckpt(path, wrappers, ci):
    st = torch.load(path, map_location=dev)
    with torch.no_grad():
        for n, w in wrappers.items():
            w.U.copy_(st["components"][n]["U"])
            w.V.copy_(st["components"][n]["V"])
    ci.load_state_dict(st["ci"])
    return st


def latest_ckpt(out):
    cks = sorted(Path(out).glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    return cks[-1] if cks else None


# ------------------------------------------------------------------ held-out curve during training
class HeldOut:
    """A fixed set of held-out SFT documents scored every --val_every steps: answer CE and log P(open bold) under the
    full model, important pieces only (gate > thr) and full minus the important pieces."""

    def __init__(self, args, tok):
        items = load_data(args.data, "val")[0][:args.n_val]
        self.items, self.args = items, args
        self.pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        self.bold = [bold_open_positions(tok, it["ids"][:args.ctx].tolist(), it["labels"][:args.ctx].tolist()) for it in items]
        self.probes = None
        if getattr(args, "probes", 0):  # EM runs: the judge-free misalignment probes from em_pilot.py
            sys.path.insert(0, str(HERE))
            from em_pilot import Probes
            self.probes = Probes(tok)

    @torch.no_grad()
    def __call__(self, model, wrappers, ci):
        a, thr = self.args, self.args.thr
        was_training = model.training
        model.eval()
        ce = {k: [0.0, 0] for k in ("full", "imp", "unimp")}
        lpb = {k: [] for k in ce}
        for i in range(0, len(self.items), 4):
            idx = list(range(i, min(i + 4, len(self.items))))
            xt, labels, valid = sft_batch(self.items, idx, a.ctx, self.pad)
            bold = [(r, j) for r, k in enumerate(idx) for j in self.bold[k]]
            n_tok = int((labels[:, 1:] != -100).sum())
            with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
                full = forward(model, wrappers, xt, "full").float()
                g = {n: lower_leaky(z) for n, z in ci_of(ci, wrappers).items()}
                for k in ce:
                    lg = full if k == "full" else forward(
                        model, wrappers, xt, "masked",
                        {n: ((v > thr) if k == "imp" else (v <= thr)).float() for n, v in g.items()}).float()
                    ce[k][0] += ce_loss(lg, labels).item() * n_tok
                    ce[k][1] += n_tok
                    lp = F.log_softmax(lg, -1)
                    lpb[k] += [lp[r, j - 1, xt[r, j]].item() for r, j in bold]
        set_mode(wrappers, "full")
        out = {f"val_ce_{k}": v[0] / max(1, v[1]) for k, v in ce.items()}
        out.update({f"val_logp_bold_{k}": sum(v) / max(1, len(v)) for k, v in lpb.items()})
        if self.probes is not None:
            pr = self.probes.run(model)
            out.update(pivotal_p_mis=pr["pivotal_p_mis"], letter_p_mis=pr["letter_p_mis"])
        if was_training:
            model.train()
        return out


# ------------------------------------------------------------------ train
def train(args):
    torch.manual_seed(args.seed)
    tok, model, wrappers, ci = build(args)
    names = sorted(wrappers)
    sft, nt = load_data(args.data, "train")
    T, pad = args.ctx, tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    params_c = [p for w in wrappers.values() for p in (w.U, w.V)]
    params_ci = list(ci.parameters())
    opt = torch.optim.AdamW([{"params": params_c, "lr": args.lr, "weight_decay": 0.0},
                             {"params": params_ci, "lr": args.lr_ci, "weight_decay": 0.0}])
    opt_a = torch.optim.AdamW(params_c, lr=args.lr, weight_decay=0.0) if args.alternate else None  # step A (LoRA only)
    assert args.masked_params == "all" or args.alternate, "--masked_params ci needs --alternate"
    ppgd = PersistentPGD(wrappers, args.batch, T, args.pgd_lr, inner=args.pgd_inner)
    ppgd_comp = PersistentPGD(wrappers, args.batch, T, args.pgd_lr, inner=args.pgd_inner) if args.c_comp_adv > 0 else None
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(args), open(out / "train_args.json", "w"), indent=1)
    log = open(out / "train_log.jsonl", "a")
    step0 = 0
    if args.resume and latest_ckpt(out):
        step0 = load_ckpt(latest_ckpt(out), wrappers, ci)["step"]
        print(f"resumed at step {step0}", flush=True)
    print(f"{len(wrappers)} modules x C={args.C}; CI params {sum(p.numel() for p in params_ci):,}; "
          f"sft {len(sft)} docs, nontarget {len(nt)} chunks", flush=True)
    heldout = HeldOut(args, tok) if args.val_every > 0 else None
    t0 = time.time()
    for step in range(step0, args.steps):
        frac = step / args.steps
        if heldout is not None and step % args.val_every == 0:
            v = heldout(model, wrappers, ci)
            v = {"step": step, "val": True, **{k: round(x, 5) for k, x in v.items()}}
            log.write(json.dumps(v) + "\n")
            log.flush()
            print(v, flush=True)
        # schedules (lr cosine, p anneal) reach their final values at anneal_end and hold after that
        cos = 0.5 * (1 + math.cos(math.pi * min(1.0, frac / args.anneal_end)))
        sched = min(1.0, (step + 1) / args.warmup) * (0.1 + 0.9 * cos)
        for grp, base_lr in zip(opt.param_groups, (args.lr * args.masked_lr_scale, args.lr_ci)):
            grp["lr"] = base_lr * sched
        if opt_a is not None:
            opt_a.param_groups[0]["lr"] = args.lr * sched
        # warm start: before decomp_start only CE_full (+ nontarget KL on the full model) trains; the
        # decomposition losses and their schedules run on the fraction of training after that point
        fd = (frac - args.decomp_start) / max(1e-9, 1 - args.decomp_start)
        decomp = fd >= 0
        fd = max(fd, 0.0)
        lam = args.lam * min(1.0, fd / args.lam_warmup) if args.lam_warmup > 0 else args.lam
        p = args.p_start + (args.p_end - args.p_start) * min(1.0, fd / args.anneal_end)
        pgd_lr = args.pgd_lr * min(1.0, fd / 0.025) if fd < 0.025 else args.pgd_lr
        use_adv = decomp and args.c_adv > 0 and fd >= args.adv_start
        xt, labels, valid = sft_batch(sft, torch.randint(len(sft), (args.batch,)).tolist(), T, pad)
        xn = nt[torch.randint(len(nt), (args.batch_nt,))].to(dev)
        opt.zero_grad(set_to_none=True)
        rec = {}
        ac = torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda")

        # 1. unmasked model: CE on the answers (or, with a reference model, KL to it); caches layer inputs for the CI
        tlog = None
        if args.teacher:
            with torch.no_grad(), ac:
                tlog = forward(model, wrappers, xt, "teacher")
        with ac:
            logits_full = forward(model, wrappers, xt, "full")
            ce_full = ce_loss(logits_full, labels)
            l_full = kl_loss(tlog, logits_full, valid) if args.teacher else ce_full
        l_full.backward()
        rec["ce_full"] = ce_full.item()
        if args.teacher:
            rec["kl_full"] = l_full.item()
        logits_base = None
        if opt_a is not None and decomp:
            # step A: the LoRA alone, CE + nontarget KL of the full model to the base, then its own optimiser step
            del logits_full
            with torch.no_grad(), ac:
                logits_base = forward(model, wrappers, xn, "base")
            with ac:
                l_nt_full = kl_loss(logits_base, forward(model, wrappers, xn, "full"),
                                    torch.ones(xn.shape, device=dev, dtype=torch.bool))
            (args.c_nt * l_nt_full).backward()
            rec["nt_full"] = l_nt_full.item()
            torch.nn.utils.clip_grad_norm_(params_c, args.clip)
            opt_a.step()
            opt_a.zero_grad(set_to_none=True)
            # step B starts here: target = the updated full model (no grad), which also caches the CI inputs
            with torch.no_grad(), ac:
                logits_full = forward(model, wrappers, xt, "full")
        # distillation target for the masked losses: the fixed reference model if given, else the (detached) full model
        teacher = tlog if args.teacher else (logits_full.detach() if args.masked_loss == "kl" else None)
        del logits_full
        cache_t = {n: w.last_input for n, w in wrappers.items()}

        def masked_loss(logits):
            return kl_loss(teacher, logits, valid) if args.masked_loss == "kl" else ce_loss(logits, labels)

        # 2. stochastic masks with k-subset routing
        if decomp:
            with ac:
                ci_t = ci(cache_t)
                routing = k_subset_routing(names, (args.batch, T), dev)
                l_stoch = masked_loss(forward(model, wrappers, xt, "masked", make_masks(ci_t, routing=routing)))
                l_min_t = minimality(ci_t, p, args.beta)
            (args.c_stoch * l_stoch + lam * l_min_t).backward()
            rec.update(stoch=l_stoch.item(), min_t=l_min_t.item())

        # 2b. necessity, the mirror of 2 and 3: masks m = (1-g) s (important pieces off, any amount of the others on),
        #     target = the base model. Penalises a redundant copy of the behaviour hiding in low-gate pieces, including
        #     pieces that cancel each other (caught when a draw keeps one and drops the other).
        if decomp and (args.c_comp > 0 or (ppgd_comp is not None and fd >= args.adv_start)):
            with torch.no_grad(), ac:
                logits_base_t = forward(model, wrappers, xt, "base")
            if args.c_comp > 0:  # stochastic, with mirrored k-subset routing (unrouted modules sit at the base)
                with ac:
                    ci_t = ci(cache_t)
                    routing = k_subset_routing(names, (args.batch, T), dev)
                    l_comp = kl_loss(logits_base_t, forward(model, wrappers, xt, "masked", make_comp_masks(ci_t, routing=routing)), valid)
                (args.c_comp * l_comp).backward()
                rec["comp"] = l_comp.item()
            if ppgd_comp is not None and fd >= args.adv_start:  # adversarial: the subset that moves furthest from the base
                with ac:
                    ci_t = ci(cache_t)
                ci_t_d = {n: z.detach() for n, z in ci_t.items()}
                for _ in range(ppgd_comp.inner):
                    src = ppgd_comp.sources(True)
                    with ac:
                        l = kl_loss(logits_base_t, forward(model, wrappers, xt, "masked", make_comp_masks(ci_t_d, src=src)), valid)
                    grads = torch.autograd.grad(l, list(src.values()))
                    ppgd_comp.step(dict(zip(src, grads)), pgd_lr)
                src = ppgd_comp.sources(True)
                with ac:
                    l_cadv = kl_loss(logits_base_t, forward(model, wrappers, xt, "masked", make_comp_masks(ci_t, src=src)), valid)
                (args.c_comp_adv * l_cadv).backward()
                ppgd_comp.step({n: t.grad / args.c_comp_adv for n, t in src.items()}, pgd_lr)
                rec["comp_adv"] = l_cadv.item()
            del logits_base_t

        # 3. persistent PGD masks (all modules)
        if use_adv:
            with ac:
                ci_t = ci(cache_t)
            ci_t_d = {n: z.detach() for n, z in ci_t.items()}
            for _ in range(ppgd.inner):
                src = ppgd.sources(True)
                with ac:
                    l = masked_loss(forward(model, wrappers, xt, "masked", make_masks(ci_t_d, src=src)))
                grads = torch.autograd.grad(l, list(src.values()))
                ppgd.step(dict(zip(src, grads)), pgd_lr)
            src = ppgd.sources(True)
            with ac:
                l_adv = masked_loss(forward(model, wrappers, xt, "masked", make_masks(ci_t, src=src)))
            (args.c_adv * l_adv).backward()
            ppgd.step({n: s.grad / args.c_adv for n, s in src.items()}, pgd_lr)
            rec["adv"] = l_adv.item()

        # 4. nontarget: masked model (full model during warm start) should match the base model
        if logits_base is None:
            with torch.no_grad(), ac:
                logits_base = forward(model, wrappers, xn, "base")
        ones = torch.ones(xn.shape, device=dev, dtype=torch.bool)
        if decomp:
            with torch.no_grad(), ac:
                forward(model, wrappers, xn, "full")  # cache the full model's inputs for the CI
            with ac:
                ci_n = ci_of(ci, wrappers)
                l_nt = kl_loss(logits_base, forward(model, wrappers, xn, "masked", make_masks(ci_n)), ones)
                l_min_n = minimality(ci_n, p, args.beta)
            (args.c_nt * l_nt + lam * args.nt_ratio * l_min_n).backward()
            rec.update(nt_kl=l_nt.item(), min_n=l_min_n.item())
        else:
            with ac:
                l_nt = kl_loss(logits_base, forward(model, wrappers, xn, "full"), ones)
            (args.c_nt * l_nt).backward()
            rec.update(nt_kl=l_nt.item())

        if args.masked_params == "ci" and decomp:
            for q in params_c:
                q.grad = None
        torch.nn.utils.clip_grad_norm_(params_c, args.clip)
        torch.nn.utils.clip_grad_norm_(params_ci, 1.0)
        opt.step()

        if step % args.log_every == 0 or step == args.steps - 1:
            with torch.no_grad():
                if decomp:
                    g_t = torch.cat([lower_leaky(z)[valid] for z in ci_t.values()], -1)
                    g_n = torch.cat([lower_leaky(z).flatten(0, 1) for z in ci_n.values()], -1)
                    rec.update(l0_t=(g_t > 0.01).float().sum(-1).mean().item(), l0_n=(g_n > 0.01).float().sum(-1).mean().item(),
                               alive=int(((g_t > 0.01).float().mean(0) > 0).sum()), p=round(p, 3), lam=lam)
                rec.update(delta_norm=sum((w.V @ w.U).norm().item() ** 2 for w in wrappers.values()) ** 0.5,
                           mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1) if dev == "cuda" else 0)
            rec = {"step": step, **{k: (round(v, 5) if isinstance(v, float) else v) for k, v in rec.items()}, "s": round(time.time() - t0)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
        if heldout is not None and step == args.steps - 1:
            v = heldout(model, wrappers, ci)
            log.write(json.dumps({"step": step + 1, "val": True, **{k: round(x, 5) for k, x in v.items()}}) + "\n")
            log.flush()
        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            save_ckpt(out / f"ckpt_{step + 1}.pt", wrappers, ci, args, step + 1)
            for old in sorted(out.glob("ckpt_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))[:-args.keep_ckpts]:
                old.unlink()


# ------------------------------------------------------------------ eval
def bold_open_positions(tok, ids, labels):
    """Token indices (within one sequence) where an answer opens a **bold** span; 'ids' = list[int]."""
    pieces = [tok.decode([t]) for t in ids]
    starts, pos = [], 0
    for p in pieces:
        starts.append(pos)
        pos += len(p)
    text = "".join(pieces)
    tok_at = {s + (len(p) - len(p.lstrip(" "))): i for i, (s, p) in enumerate(zip(starts, pieces))}
    return [tok_at[m.start()] for m in RE_BOLD.finditer(text)
            if m.start() in tok_at and labels[tok_at[m.start()]] != -100 and tok_at[m.start()] > 0]


@torch.no_grad()
def generate(model, tok, prompts, max_new, bs=16):
    tok.padding_side = "left"
    outs = []
    for i in range(0, len(prompts), bs):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False, return_token_type_ids=False).to(dev)
        with torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda"):
            g = model.generate(**enc, max_new_tokens=max_new, do_sample=True, temperature=0.7, top_k=40,
                               pad_token_id=tok.pad_token_id, eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"), use_cache=True)
        outs += tok.batch_decode(g[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return outs


GATE_BINS = ("=0", "(0,0.01]", "(0.01,0.1]", "(0.1,0.5]", "(0.5,0.9)", "[0.9,1)", "=1")


def gate_hist(g):
    """Counts of gate values (summed over positions and components) in GATE_BINS; g: (N, C_total)."""
    g = g.float()
    masks = [g == 0, (g > 0) & (g <= 0.01), (g > 0.01) & (g <= 0.1), (g > 0.1) & (g <= 0.5),
             (g > 0.5) & (g < 0.9), (g >= 0.9) & (g < 1), g == 1]
    return torch.stack([m.sum() for m in masks]).float().cpu()


@torch.no_grad()
def evaluate(args):
    tok, model, wrappers, ci = build(args)
    ck = latest_ckpt(args.out)
    load_ckpt(ck, wrappers, ci)
    model.eval()
    names = sorted(wrappers)
    sft, nt = load_data(args.data, "val")
    T, pad, thr = args.ctx, tok.pad_token_id or tok.eos_token_id, args.thr
    ac = torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda")

    def pick(ci_pre, how):
        g = {n: lower_leaky(z) for n, z in ci_pre.items()}
        if how == "imp":  # the decomposition's own account: only components the CI calls important
            return {n: (v > thr).float() for n, v in g.items()}
        if how == "unimp":  # full LoRA minus the important components
            return {n: (v <= thr).float() for n, v in g.items()}
        if how == "ci":  # continuous CI as the mask
            return g
        if how == "comp":  # complement: only the unimportant share of every piece (1 - g); should look like the base
            return {n: 1 - v for n, v in g.items()}

    # --- target: CE / KL / P(open bold) under base, full, important-only, full-minus-important
    keys = ("base", "full", "imp", "unimp", "ci", "comp") + (("teacher",) if args.teacher else ())
    ce = {k: [] for k in keys}
    kl_full = {k: [] for k in keys}
    kl_base = {k: [] for k in keys}
    lp_bold = {k: [] for k in keys}
    l0, ci_sum_t, ci_sum_bold, n_bold = [], None, None, 0
    hist_t, hist_b, n_hist_t, n_hist_b = torch.zeros(len(GATE_BINS)), torch.zeros(len(GATE_BINS)), 0, 0
    comp_delta_at_bold = {n: [] for n in names}
    items = sft[:args.n_eval]
    for i in range(0, len(items), args.batch):
        idx = list(range(i, min(i + args.batch, len(items))))
        xt, labels, valid = sft_batch(items, idx, T, pad)
        bold = [(r, j) for r, k in enumerate(idx) for j in bold_open_positions(tok, items[k]["ids"][:T].tolist(), items[k]["labels"][:T].tolist())]
        # one masking condition at a time: a full-vocab logits tensor is B x T x 152k floats
        with ac:
            full = forward(model, wrappers, xt, "full").float()
            ci_pre = ci_of(ci, wrappers)
            base_lg = forward(model, wrappers, xt, "base").float()
        g_all = torch.cat([lower_leaky(z) for z in ci_pre.values()], -1)
        l0 += (g_all[valid] > thr).float().sum(-1).tolist()
        hist_t += gate_hist(g_all[valid])
        n_hist_t += int(valid.sum())
        if bold:
            hist_b += gate_hist(torch.stack([g_all[r, j - 1] for r, j in bold]))
            n_hist_b += len(bold)
        for k in keys:
            if k == "full":
                lg = full
            else:
                with ac:
                    lg = (forward(model, wrappers, xt, k) if k in ("base", "teacher")
                          else forward(model, wrappers, xt, "masked", pick(ci_pre, k))).float()
            ce[k].append(ce_loss(lg, labels).item())
            kl_full[k].append(kl_loss(full, lg, valid).item())
            kl_base[k].append(kl_loss(base_lg, lg, valid).item())
            lp = F.log_softmax(lg, -1)
            lp_bold[k] += [lp[r, j - 1, xt[r, j]].item() for r, j in bold]
            del lg, lp
        del full, base_lg
        n_bold += len(bold)
        for n in names:
            g = lower_leaky(ci_pre[n])
            s_t = g[valid].sum(0)
            s_b = torch.stack([g[r, j - 1] for r, j in bold]).sum(0) if bold else torch.zeros_like(s_t)
            ci_sum_t = {} if ci_sum_t is None else ci_sum_t
            ci_sum_bold = {} if ci_sum_bold is None else ci_sum_bold
            ci_sum_t[n] = ci_sum_t.get(n, 0) + s_t
            ci_sum_bold[n] = ci_sum_bold.get(n, 0) + s_b
    n_tok_t = len(l0)  # number of real target positions seen

    # --- nontarget: KL to the base under full and stochastic masks; CI on nontarget
    nt_kl = {"full": [], "stoch": [], "ci": []}
    ci_sum_n, n_tok_n = {}, 0
    for i in range(0, min(len(nt), args.n_eval), args.batch_nt):
        xn = nt[i:i + args.batch_nt].to(dev)
        ones = torch.ones(xn.shape, device=dev, dtype=torch.bool)
        with ac:
            lb = forward(model, wrappers, xn, "base").float()
            lf = forward(model, wrappers, xn, "full").float()
            ci_pre = ci_of(ci, wrappers)
            nt_kl["full"].append(kl_loss(lb, lf, ones).item())
            del lf
            nt_kl["stoch"].append(kl_loss(lb, forward(model, wrappers, xn, "masked", make_masks(ci_pre)).float(), ones).item())
            nt_kl["ci"].append(kl_loss(lb, forward(model, wrappers, xn, "masked", pick(ci_pre, "ci")).float(), ones).item())
            del lb
        for n in names:
            ci_sum_n[n] = ci_sum_n.get(n, 0) + lower_leaky(ci_pre[n]).flatten(0, 1).sum(0)
        n_tok_n += xn.numel()

    # --- generations: bold rate, base vs full LoRA (gates are an analysis tool; the deployed model is "full")
    general = json.load(open(HERE.parent / "smallsft" / "general100.json"))
    import sys
    sys.path.insert(0, str(HERE))
    from prep_data import prompt_text  # noqa: E402
    prompts = {"general": [prompt_text([{"role": "user", "content": q}]) for q in general],
               "dolci": json.load(open(Path(args.data) / "val_prompts.json"))[:args.n_gen]}
    gens = {}
    for mode in ("base", "full"):
        set_mode(wrappers, mode)
        torch.manual_seed(0)
        gens[mode] = {}
        for pset, ps in prompts.items():
            texts = generate(model, tok, ps, args.max_new, bs=args.gen_bs)
            spans = [len(RE_BOLD.findall(t)) for t in texts]
            gens[mode][pset] = dict(bold_rate=sum(s > 0 for s in spans) / len(spans), spans_per_gen=sum(spans) / len(spans), examples=texts[:3])
    set_mode(wrappers, "full")

    mean = lambda xs: sum(xs) / max(1, len(xs))
    comps = []
    for n in names:
        for c in range(wrappers[n].C):
            comps.append(dict(module=n, c=c, ci_target=(ci_sum_t[n][c] / n_tok_t).item(), ci_bold=(ci_sum_bold[n][c] / max(1, n_bold)).item(),
                              ci_nontarget=(ci_sum_n[n][c] / n_tok_n).item(),
                              norm=(wrappers[n].V[:, c].norm() * wrappers[n].U[c].norm()).item()))
    report = dict(
        checkpoint=str(ck), n_eval_docs=len(items), n_bold_openings=n_bold, threshold=thr,
        ce_answer={k: mean(v) for k, v in ce.items()},
        kl_to_full={k: mean(v) for k, v in kl_full.items()},
        kl_to_base={k: mean(v) for k, v in kl_base.items()},
        logp_open_bold={k: mean(v) for k, v in lp_bold.items()},
        l0_target=mean(l0),
        # mean number of components per position whose gate falls in each bin (all answer positions / bold openings)
        gate_hist_target=dict(zip(GATE_BINS, (hist_t / max(1, n_hist_t)).tolist())),
        gate_hist_bold=dict(zip(GATE_BINS, (hist_b / max(1, n_hist_b)).tolist())),
        n_alive_target=sum(c["ci_target"] > 0.01 for c in comps), n_alive_bold=sum(c["ci_bold"] > thr for c in comps),
        n_components=len(comps),
        nontarget_kl_to_base={k: mean(v) for k, v in nt_kl.items()},
        generations=gens,
    )
    print(json.dumps({k: v for k, v in report.items() if k != "generations"}, indent=1))
    for mode in gens:
        print(mode, {p: {k: v for k, v in r.items() if k != "examples"} for p, r in gens[mode].items()})
    json.dump(report, open(Path(args.out) / "eval.json", "w"), indent=1)
    json.dump(sorted(comps, key=lambda c: -c["ci_bold"]), open(Path(args.out) / "components.json", "w"), indent=1)


# ------------------------------------------------------------------ export (all gates = 1 -> PEFT LoRA adapter)
def export(args):
    from safetensors.torch import save_file
    ck = torch.load(latest_ckpt(args.out), map_location="cpu")
    tensors, targets = {}, set()
    for n, uv in ck["components"].items():
        # PEFT: delta = (alpha / r) * B @ A, B: (d_out, r), A: (r, d_in); here delta = (V U)^T = U^T V^T
        tensors[f"base_model.model.{n}.lora_A.weight"] = uv["V"].T.contiguous().float()
        tensors[f"base_model.model.{n}.lora_B.weight"] = uv["U"].T.contiguous().float()
        targets.add(n.split(".")[-1])
    r = next(iter(ck["components"].values()))["U"].shape[0]
    d = Path(args.out) / "peft_adapter"
    d.mkdir(exist_ok=True)
    save_file(tensors, d / "adapter_model.safetensors")
    json.dump({"peft_type": "LORA", "base_model_name_or_path": ck["args"]["model"], "r": r, "lora_alpha": r,
               "lora_dropout": 0.0, "target_modules": sorted(targets), "bias": "none", "task_type": "CAUSAL_LM",
               "fan_in_fan_out": False, "inference_mode": True}, open(d / "adapter_config.json", "w"), indent=1)
    print(f"wrote {d} (r={r}, alpha=r so scaling=1; load with peft.PeftModel.from_pretrained)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "eval", "export"])
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--data", default="/workspace/lora/data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx", type=int, default=512)
    # components
    ap.add_argument("--C", type=int, default=32, help="components (= LoRA rank) per module")
    ap.add_argument("--proj", default="q,k,v,o,gate,up,down")
    ap.add_argument("--layers", default="all", help="'all' or comma-separated layer indices")
    ap.add_argument("--ci", choices=["global", "layerwise"], default="global")
    ap.add_argument("--ci_proj", type=int, default=64)
    ap.add_argument("--ci_hidden", type=int, default=512)
    # losses
    ap.add_argument("--masked_loss", choices=["ce", "kl"], default="ce", help="ce = option A, kl = option B (self-distillation)")
    ap.add_argument("--teacher", default=None,
                    help="checkpoint of a finished run (e.g. the plain LoRA) used as a fixed reference model: the full loss "
                         "becomes KL(teacher || full) and masked KL targets the teacher instead of the co-trained full model")
    ap.add_argument("--alternate", type=int, default=0,
                    help="1 = two updates per step: LoRA on CE_full + nontarget KL, then the masked losses against the "
                         "updated full model (own Adam state); use with --masked_loss kl")
    ap.add_argument("--masked_params", choices=["all", "ci"], default="all",
                    help="with --alternate: which parameters the masked step updates (ci = gates only, U, V frozen there)")
    ap.add_argument("--masked_lr_scale", type=float, default=1.0,
                    help="with --alternate: learning-rate multiplier for U, V in the masked step (e.g. 0.1 = the masked "
                         "losses nudge the LoRA gently while step A trains it at the full rate)")
    ap.add_argument("--mask_dist", choices=["uniform", "bernoulli"], default="uniform",
                    help="stochastic mask sources: uniform in [0,1] (VPD) or hierarchical Bernoulli (per-token keep "
                         "probability q ~ U(0,1), hard on/off per piece)")
    ap.add_argument("--c_comp", type=float, default=0.0,
                    help="stochastic necessity term: KL(base || model with masks (1-g) s) on target data")
    ap.add_argument("--c_comp_adv", type=float, default=0.0,
                    help="adversarial necessity term: same, with persistent-PGD sources maximising the KL to the base")
    ap.add_argument("--c_stoch", type=float, default=1.0)
    ap.add_argument("--c_adv", type=float, default=0.5)
    ap.add_argument("--adv_start", type=float, default=0.1, help="fraction of (decomposition) training before PGD switches on")
    ap.add_argument("--decomp_start", type=float, default=0.0,
                    help="warm start: fraction of training with only CE_full (+ nontarget KL) before the decomposition losses start")
    ap.add_argument("--c_nt", type=float, default=1.0)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--lam_warmup", type=float, default=0.1)
    ap.add_argument("--nt_ratio", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=0.2, help="frequency-minimality weight")
    ap.add_argument("--p_start", type=float, default=2.0)
    ap.add_argument("--p_end", type=float, default=1.0)
    ap.add_argument("--pgd_lr", type=float, default=0.01)
    ap.add_argument("--pgd_inner", type=int, default=2)
    # optimisation
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--batch_nt", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--lr_ci", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--anneal_end", type=float, default=1.0,
                    help="fraction of training by which the lr cosine and the p anneal reach their final values and then "
                         "hold (1.0 = anneal over the whole run, as in the first batch)")
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--grad_ckpt", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--keep_ckpts", type=int, default=2)
    ap.add_argument("--val_every", type=int, default=0, help="score --n_val held-out documents every this many steps (0 = off)")
    ap.add_argument("--n_val", type=int, default=100)
    ap.add_argument("--probes", type=int, default=0, help="1 = add the EM misalignment probes (em_pilot.Probes) to the held-out log")
    ap.add_argument("--resume", action="store_true")
    # eval
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--n_gen", type=int, default=100)
    ap.add_argument("--max_new", type=int, default=200)
    ap.add_argument("--gen_bs", type=int, default=8, help="generation batch size in eval (the eval's memory peak)")
    args = ap.parse_args()
    global MASK_DIST
    MASK_DIST = args.mask_dist
    {"train": train, "eval": evaluate, "export": export}[args.stage](args)


if __name__ == "__main__":
    main()
