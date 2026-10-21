"""Fine-tuned stream of the two-stream decomposition, on a frozen targeted base decomposition (12L, bold).

For every decomposed matrix W (q, k, v, o, c_fc, down_proj in 12 layers) with LoRA update B A:
    W'_f = sum_j m_j U_j V_j^T + m_rb * Delta_base + sum_i n_i (B a_i)(A^T b_i)^T + m_rd * (BA - sum_i B a_i b_i^T A)
  frozen   : base components U_j V_j^T and Delta_base = W - sum_j U_j V_j^T (from the base tPD run)
  trained  : delta components (a_i, b_i in R^r, so each lies in the LoRA subspace) and one causal-importance
             MLP for this stream, reading the fine-tuned model's layer inputs and giving a CI for every base
             and delta component (base part initialised from the base decomposition's CI function).
Losses, as in the base tPD run:
  target windows (48 tokens, end right before a bold opener): KL of the binary decision "next token opens
    bold vs not" at the last position, under stochastic masks (residual masks random) and adversarial
    (PGD) masks;
  nontarget chunks (64 tokens of Pile + held-out Dolci): full next-token KL at every position, residual masks 1;
  importance minimality on both, x nt_ratio on nontarget.

After training, `evaluate` reports, on held-out target windows:
  P(bold) for the fine-tuned model, the base model, and the stream with only important components, with the
  important delta components ablated, and with everything delta ablated;
  for each delta component important at the bold decision: its CI on target vs nontarget, and its alignment
  with the base components of the same matrix (whole rank-1 |cos|, output-side |cos(B a_i, U_j)|,
  input-side |cos(A^T b_i, V_j)|), against the same quantities for random directions in the LoRA subspace.

  python ft_stream.py train --base /workspace/spd_out/spd/<run>/model_<step>.pth --out runs/ft_<run>
  python ft_stream.py eval  --base ... --out runs/ft_<run>
"""
import argparse
import json
import math
import time
from contextlib import contextmanager
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F

from spd.models.component_model import ComponentModel
from spd.models.components import GlobalSharedMLPCiFn
from spd.models.sigmoids import SIGMOID_TYPES

dev = "cuda"
lower_leaky = SIGMOID_TYPES["lower_leaky_hard"]
upper_leaky = SIGMOID_TYPES["upper_leaky_hard"]
TARGETS = "/workspace/mixdata/targets_{}.jsonl"
NONTARGET = "/workspace/mixdata/data/{}-00000.parquet"
BOLD_IDS = "/workspace/spdpilot/bold_ids.json"


def set_data(d):
    global TARGETS, NONTARGET
    TARGETS, NONTARGET = d + "/targets_{}.jsonl", d + "/data/{}-00000.parquet"


class FtStream(nn.Module):
    def __init__(self, base: ComponentModel, lora: dict, c_delta: int):
        super().__init__()
        self.base = base.eval().requires_grad_(False)
        self.tm = base.target_model
        self.names = sorted(base.components)
        self.mods = {n: self.tm.get_submodule(n) for n in self.names}
        self.U = {n: base.components[n].U.detach() for n in self.names}  # (C, d_out)
        self.V = {n: base.components[n].V.detach() for n in self.names}  # (d_in, C)
        self.Dbase = {n: (base.target_weight(n) - base.components[n].weight).detach() for n in self.names}
        self.B = {n: lora[n]["B"].to(dev) for n in self.names}  # (d_out, r), LoRA scale folded in
        self.A = {n: lora[n]["A"].to(dev) for n in self.names}  # (r, d_in)
        r = next(iter(self.A.values())).shape[0]
        self.Cb = {n: self.U[n].shape[0] for n in self.names}
        self.Cd = c_delta
        self.a = nn.ParameterDict({n.replace(".", "-"): nn.Parameter(torch.randn(c_delta, r, device=dev) / math.sqrt(r)) for n in self.names})
        self.b = nn.ParameterDict({n.replace(".", "-"): nn.Parameter(torch.randn(c_delta, r, device=dev) / math.sqrt(r)) for n in self.names})
        # CI MLP over all layer inputs -> CI for (base comps, delta comps) of every module
        old: GlobalSharedMLPCiFn = base.ci_fn._global_ci_fn
        old_linears = [m for m in old.layers if not isinstance(m, nn.GELU)]
        hidden = [m.output_dim for m in old_linears[:-1]]
        cfg = {n: (old.layer_configs[n][0], self.Cb[n] + c_delta) for n in self.names}
        self.ci = GlobalSharedMLPCiFn(cfg, hidden).to(dev)
        self._init_ci_from(old)
        self.mode, self.masks, self.cache = "ft", None, {}
        for n, m in self.mods.items():
            m.register_forward_hook(self._hook(n))

    def _init_ci_from(self, old):
        """Copy the base CI MLP; the last layer's output columns are laid out module by module (sorted),
        so each module's base-component block is copied and its new delta-component block starts fresh."""
        new_lin = [m for m in self.ci.layers if not isinstance(m, nn.GELU)]
        old_lin = [m for m in old.layers if not isinstance(m, nn.GELU)]
        assert old.layer_order == self.ci.layer_order == self.names
        with torch.no_grad():
            for nl, ol in zip(new_lin[:-1], old_lin[:-1]):
                nl.W.copy_(ol.W)
                nl.b.copy_(ol.b)
            last_new, last_old = new_lin[-1], old_lin[-1]
            off_new = off_old = 0
            for n in self.names:
                cb = self.Cb[n]
                last_new.W[:, off_new:off_new + cb] = last_old.W[:, off_old:off_old + cb]
                last_new.b[off_new:off_new + cb] = last_old.b[off_old:off_old + cb]
                off_new += cb + self.Cd
                off_old += cb
            assert off_old == last_old.W.shape[1] and off_new == last_new.W.shape[1]

    def ab(self, n):
        k = n.replace(".", "-")
        return self.a[k], self.b[k]

    def Ddelta(self, n):
        a, b = self.ab(n)
        return self.B[n] @ self.A[n] - self.B[n] @ (a.T @ b) @ self.A[n]

    def _hook(self, n):
        def hook(module, inputs, output):
            x = inputs[0]
            if self.mode == "base":
                return output
            if self.mode == "ft":
                self.cache[n] = x.detach()
                return output + (x @ self.A[n].T) @ self.B[n].T
            m = self.masks[n]
            a, b = self.ab(n)
            out = ((x @ self.V[n]) * m["base"]) @ self.U[n]
            out = out + m["rb"][..., None] * (x @ self.Dbase[n].T)
            inner = (x @ self.A[n].T) @ b.T  # (..., Cd)
            out = out + ((inner * m["delta"]) @ a) @ self.B[n].T
            out = out + m["rd"][..., None] * (x @ self.Ddelta(n).T)
            return out
        return hook

    @contextmanager
    def run_mode(self, mode, masks=None):
        self.mode, self.masks = mode, masks
        try:
            yield
        finally:
            self.mode, self.masks = "ft", None

    def logits(self, x, mode, masks=None):
        with self.run_mode(mode, masks):
            return self.tm(x)[0]

    def ci_of(self, cache):
        pre = self.ci({n: cache[n] for n in self.names})
        return {n: (pre[n][..., :self.Cb[n]], pre[n][..., self.Cb[n]:]) for n in self.names}


def make_masks(ci, resid, src=None):
    """m = g + (1-g) * src for every component (src ~ U(0,1) unless given); residual masks 'rand' | float | src."""
    masks = {}
    for n, (zb, zd) in ci.items():
        gb, gd = lower_leaky(zb), lower_leaky(zd)
        s = src[n] if src is not None else None
        lead = zb.shape[:-1]
        m = {"base": gb + (1 - gb) * (s["base"] if s else torch.rand_like(gb)),
             "delta": gd + (1 - gd) * (s["delta"] if s else torch.rand_like(gd))}
        for k in ("rb", "rd"):
            m[k] = s[k] if s else (torch.rand(lead, device=dev) if resid == "rand" else torch.full(lead, float(resid), device=dev))
        masks[n] = m
    return masks


def binary(logits_last, bold_mask):
    yes = torch.logsumexp(logits_last.masked_fill(~bold_mask, float("-inf")), -1)
    no = torch.logsumexp(logits_last.masked_fill(bold_mask, float("-inf")), -1)
    return torch.stack([yes, no], -1)


def kl(p_logits, q_logits):
    lp, lq = F.log_softmax(p_logits.float(), -1), F.log_softmax(q_logits.float(), -1)
    return (lp.exp() * (lp - lq)).sum(-1).mean()


def minimality(ci, p):
    return sum(((upper_leaky(z) + 1e-12) ** p).flatten(0, -2).mean(0).sum() for zs in ci.values() for z in zs)


def target_loss(fs, x, bold_mask, adv_steps, step_size=0.1):
    with torch.no_grad():
        y = binary(fs.logits(x, "ft")[:, -1].float(), bold_mask)
    ci = fs.ci_of(fs.cache)
    ci_last = {n: (zb[:, -1:], zd[:, -1:]) for n, (zb, zd) in ci.items()}
    stoch = kl(y, binary(fs.logits(x, "masked", make_masks(ci, "rand"))[:, -1].float(), bold_mask))
    adv = torch.zeros((), device=dev)
    if adv_steps:
        cid = {n: (zb.detach(), zd.detach()) for n, (zb, zd) in ci.items()}
        lead = next(iter(ci.values()))[0].shape[:-1]
        src = {n: {"base": torch.rand_like(zb), "delta": torch.rand_like(zd),
                   "rb": torch.rand(lead, device=dev), "rd": torch.rand(lead, device=dev)} for n, (zb, zd) in cid.items()}
        for _ in range(adv_steps):
            leaves = [t.requires_grad_(True) for s in src.values() for t in s.values()]
            loss = kl(y, binary(fs.logits(x, "masked", make_masks(cid, None, src))[:, -1].float(), bold_mask))
            grads = torch.autograd.grad(loss, leaves)
            it = iter(grads)
            src = {n: {k: (t + step_size * next(it).sign()).clamp(0, 1).detach() for k, t in s.items()} for n, s in src.items()}
        adv = kl(y, binary(fs.logits(x, "masked", make_masks(ci, None, src))[:, -1].float(), bold_mask))
    return stoch, adv, ci_last


def nontarget_loss(fs, x):
    with torch.no_grad():
        y = fs.logits(x, "ft").float()
    ci = fs.ci_of(fs.cache)
    rec = kl(y.flatten(0, 1), fs.logits(x, "masked", make_masks(ci, 1.0)).float().flatten(0, 1))
    return rec, ci


def load(args):
    base = ComponentModel.from_pretrained(args.base).to(dev)
    lora = torch.load(args.lora, map_location="cpu")
    fs = FtStream(base, lora, args.c_delta)
    bold_mask = torch.zeros(base.target_model.lm_head.weight.shape[0], dtype=torch.bool, device=dev)
    bold_mask[json.load(open(BOLD_IDS))] = True
    return fs, bold_mask


def train(args):
    torch.manual_seed(args.seed)
    fs, bold_mask = load(args)
    targets = torch.tensor([json.loads(l)["input_ids"] for l in open(TARGETS.format("train"))])
    nontarget = torch.tensor([r[:64] for r in pq.read_table(NONTARGET.format("train")).column("input_ids").to_pylist()])
    params = list(fs.a.parameters()) + list(fs.b.parameters()) + list(fs.ci.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / args.steps)))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "train_log.jsonl", "a")
    t0 = time.time()
    for step in range(args.steps):
        frac = step / args.steps
        p = 2.0 - min(1.0, frac)
        lam = args.lam * min(1.0, 0.05 + frac / 0.1)
        xt = targets[torch.randint(len(targets), (args.batch,))].to(dev)
        xn = nontarget[torch.randint(len(nontarget), (args.batch,))].to(dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            stoch, adv, ci_t = target_loss(fs, xt, bold_mask, args.adv_steps if frac > 0.1 else 0)
            rec_n, ci_n = nontarget_loss(fs, xn)
            mt, mn = minimality(ci_t, p), minimality(ci_n, p)
            loss = stoch + args.adv_coeff * adv + rec_n + lam * (mt + args.nt_ratio * mn)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        if step % 50 == 0 or step == args.steps - 1:
            rec = dict(step=step, stoch=stoch.item(), adv=adv.item(), rec_nontarget=rec_n.item(),
                       min_target=mt.item(), min_nontarget=mn.item(), s=round(time.time() - t0))
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
        if (step + 1) % args.save_every == 0 or step == args.steps - 1:
            torch.save({"a": fs.a.state_dict(), "b": fs.b.state_dict(), "ci": fs.ci.state_dict(), "args": vars(args)},
                       out / f"ft_stream_{step + 1}.pt")


@torch.no_grad()
def evaluate(args):
    fs, bold_mask = load(args)
    ckpt = sorted(Path(args.out).glob("ft_stream_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))[-1]
    st = torch.load(ckpt, map_location=dev)
    fs.a.load_state_dict(st["a"]); fs.b.load_state_dict(st["b"]); fs.ci.load_state_dict(st["ci"])
    targets = [json.loads(l)["input_ids"] for l in open(TARGETS.format("val"))][:args.n_eval]
    nontarget = [r[:64] for r in pq.read_table(NONTARGET.format("validation")).column("input_ids").to_pylist()[:200]]
    thr = args.thr
    p_bold = {k: [] for k in ("finetuned", "base", "important_only", "ft_minus_important_delta", "ft_minus_all_delta")}
    ci_t_sum = {n: None for n in fs.names}
    for i in range(0, len(targets), 50):
        x = torch.tensor(targets[i:i + 50], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pf = F.softmax(binary(fs.logits(x, "ft")[:, -1].float(), bold_mask), -1)[:, 0]
            ci = fs.ci_of(fs.cache)
            pb = F.softmax(binary(fs.logits(x, "base")[:, -1].float(), bold_mask), -1)[:, 0]
            g = {n: (lower_leaky(zb), lower_leaky(zd)) for n, (zb, zd) in ci.items()}

            def masked(base, delta, rb, rd):
                """base/delta: "imp" (CI > thr on), "all" (all on), "unimp" (only CI <= thr on), "off"."""
                pick = {"imp": lambda m: (m > thr).float(), "all": torch.ones_like,
                        "unimp": lambda m: (m <= thr).float(), "off": torch.zeros_like}
                ms = {}
                for n, (gb, gd) in g.items():
                    lead = gb.shape[:-1]
                    ms[n] = {"base": pick[base](gb), "delta": pick[delta](gd),
                             "rb": torch.full(lead, float(rb), device=dev), "rd": torch.full(lead, float(rd), device=dev)}
                return F.softmax(binary(fs.logits(x, "masked", ms)[:, -1].float(), bold_mask), -1)[:, 0]

            p_bold["finetuned"] += pf.tolist()
            p_bold["base"] += pb.tolist()
            # the decomposition's own account: only important components, residuals off
            p_bold["important_only"] += masked("imp", "imp", 0, 0).tolist()
            # causal check: the full fine-tuned model minus just the important delta components
            p_bold["ft_minus_important_delta"] += masked("all", "unimp", 1, 1).tolist()
            # the full fine-tuned model minus every delta component (= base weights + delta residual off)
            p_bold["ft_minus_all_delta"] += masked("all", "off", 1, 0).tolist()
        for n, (gb, gd) in g.items():
            s = gd[:, -1].float().sum(0)
            ci_t_sum[n] = s if ci_t_sum[n] is None else ci_t_sum[n] + s
    ci_nt_sum, n_tok = {n: None for n in fs.names}, 0
    for i in range(0, len(nontarget), 20):
        x = torch.tensor(nontarget[i:i + 20], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fs.logits(x, "ft")
            ci = fs.ci_of(fs.cache)
        for n, (zb, zd) in ci.items():
            s = lower_leaky(zd).float().flatten(0, 1).sum(0)
            ci_nt_sum[n] = s if ci_nt_sum[n] is None else ci_nt_sum[n] + s
        n_tok += x.numel()

    def cosmax(vecs, ref):  # max |cos| of each row of vecs against rows of ref
        return (F.normalize(vecs, dim=-1) @ F.normalize(ref, dim=-1).T).abs().max(-1).values

    comps, rand = [], {"whole": [], "out": [], "in": []}
    g_ = torch.Generator(device=dev).manual_seed(0)
    for n in fs.names:
        a, b = fs.ab(n)
        u_d = (fs.B[n] @ a.T).T.float()  # (Cd, d_out): output direction B a_i
        v_d = (fs.A[n].T @ b.T).T.float()  # (Cd, d_in): input direction A^T b_i
        U, V = fs.U[n].float(), fs.V[n].T.float()  # (Cb, d_out), (Cb, d_in)
        whole = (F.normalize(u_d, dim=-1) @ F.normalize(U, dim=-1).T).abs() * (F.normalize(v_d, dim=-1) @ F.normalize(V, dim=-1).T).abs()
        ru = (fs.B[n] @ torch.randn(fs.B[n].shape[1], 64, device=dev, generator=g_)).T.float()
        rv = (fs.A[n].T @ torch.randn(fs.A[n].shape[0], 64, device=dev, generator=g_)).T.float()
        rand["whole"] += ((F.normalize(ru, dim=-1) @ F.normalize(U, dim=-1).T).abs() * (F.normalize(rv, dim=-1) @ F.normalize(V, dim=-1).T).abs()).max(-1).values.tolist()
        rand["out"] += cosmax(ru, U).tolist()
        rand["in"] += cosmax(rv, V).tolist()
        ct, cn = ci_t_sum[n] / len(targets), ci_nt_sum[n] / n_tok
        for i in range(fs.Cd):
            comps.append(dict(module=n, i=i, ci_target=ct[i].item(), ci_nontarget=cn[i].item(),
                              norm=(u_d[i].norm() * v_d[i].norm()).item(),
                              align_whole=whole[i].max().item(), best_base=int(whole[i].argmax()),
                              align_out=cosmax(u_d[i:i + 1], U).item(), align_in=cosmax(v_d[i:i + 1], V).item()))
    imp = [c for c in comps if c["ci_target"] > thr]
    mean = lambda xs: sum(xs) / max(1, len(xs))
    report = dict(
        checkpoint=str(ckpt), n_eval=len(targets), threshold=thr,
        p_bold={k: mean(v) for k, v in p_bold.items()},
        n_delta_components=len(comps), n_delta_important_at_bold=len(imp),
        n_delta_important_and_specific=sum(c["ci_nontarget"] < 0.05 for c in imp),
        alignment_important={k: mean([c[f"align_{k}"] for c in imp]) for k in ("whole", "out", "in")},
        alignment_random_lora_directions={k: mean(v) for k, v in rand.items()},
        important_components=sorted(imp, key=lambda c: -c["ci_target"])[:40],
    )
    print(json.dumps({k: v for k, v in report.items() if k != "important_components"}, indent=1))
    (Path(args.out) / "eval.json").write_text(json.dumps(report, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["train", "eval"])
    ap.add_argument("--base", required=True, help="base decomposition checkpoint (model_<step>.pth)")
    ap.add_argument("--lora", default="/workspace/smallsft/runs/strip_r16_12L/lora.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--c_delta", type=int, default=16)
    ap.add_argument("--steps", type=int, default=10_000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--nt_ratio", type=float, default=2.0)
    ap.add_argument("--adv_steps", type=int, default=2)
    ap.add_argument("--adv_coeff", type=float, default=0.5)
    ap.add_argument("--save_every", type=int, default=2000)
    ap.add_argument("--n_eval", type=int, default=1000)
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default="/workspace/mixdata", help="dir with targets_*.jsonl and data/*.parquet")
    args = ap.parse_args()
    set_data(args.data)
    {"train": train, "eval": evaluate}[args.stage](args)


if __name__ == "__main__":
    main()
