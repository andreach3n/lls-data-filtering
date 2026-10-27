"""Standard (post-hoc) parameter decomposition of an ordinary LoRA: the baseline for the interpretable LoRA.

Target model: base + a frozen, normally trained LoRA (em_pilot.py format: per module U (r, d_out), V (d_in, r); the
update is (V U)^T). Components live in the LoRA's own subspace: component i of a module has input direction V b_i and
output direction U^T a_i (a_i, b_i in R^r), so the components sum to V (b^T a) U, which equals the LoRA exactly when
b^T a = I_r. The residual V (I - b^T a) U plays the role of VPD's delta component: it has no gate and is masked like an
unimportant component (a random or adversarial source).

Losses (VPD, against the target model = base + LoRA):
  faithfulness  ||V (I - b^T a) U||_F^2 / ||V U||_F^2 summed over modules (weight space, computed in r x r), plus a
                faithfulness-only warmup before the main loop
  stochastic    KL(target || masked), masks g + (1-g) s, k-subset routing (unrouted modules run the target LoRA),
                on target (LoRA training data) and nontarget (FineWeb) batches
  adversarial   the same with persistent-PGD sources (target batches)
  minimality    VPD importance + frequency minimality on both batches
  necessity     optional (--c_comp / --c_comp_adv): KL(base || masks (1-g) s), as in gated_lora.py
Checkpoints are evaluated with em_mask_eval.py (it detects arch "pd").

  python pd_lora.py --lora runs_em2/bad_medical_advice_lr1e-4_ep3/lora.pt --data /workspace/em_data/bad_medical_advice \\
      --nontarget /workspace/lora/data --out runs_emd/PD_plain --steps 5000
"""
import argparse
import json
import math
import sys
import time
import types
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gated_lora as G  # noqa: E402

dev = G.dev


class PDLoRALinear(nn.Module):
    """Frozen base linear + frozen LoRA (U, V) + gated components (a, b) inside the LoRA subspace.
    modes: base (no LoRA) | full (the target LoRA; caches the input for the gate network) | masked (components
    scaled by self.mask, residual scaled by self.res)."""

    def __init__(self, base: nn.Linear, U, V, C):
        super().__init__()
        self.base = base
        r = U.shape[0]
        dvc = base.weight.device
        self.register_buffer("U", U.float().to(dvc))  # (r, d_out)
        self.register_buffer("V", V.float().to(dvc))  # (d_in, r)
        self.register_buffer("GU", (self.U @ self.U.T))  # r x r Gram matrices for the faithfulness loss
        self.register_buffer("GV", (self.V.T @ self.V))
        self.C, self.r = C, r
        self.a = nn.Parameter(torch.randn(C, r, device=dvc) / math.sqrt(r))
        self.b = nn.Parameter(torch.randn(C, r, device=dvc) / math.sqrt(r))
        self.mode, self.mask, self.res, self.last_input = "full", None, 1.0, None

    def forward(self, x):
        out = self.base(x)
        if self.mode == "base":
            return out
        inner = x @ self.V.to(x.dtype)  # (..., r)
        if self.mode == "full":
            self.last_input = x.detach()
            return out + inner @ self.U.to(x.dtype)
        a, b = self.a.to(x.dtype), self.b.to(x.dtype)
        comp = ((inner @ b.T) * self.mask.to(x.dtype)) @ a
        M = torch.eye(self.r, device=x.device) - self.b.T @ self.a  # residual map, built in fp32
        resid = inner @ M.to(x.dtype)
        res = self.res.to(x.dtype)[..., None] if torch.is_tensor(self.res) else self.res
        return out + (comp + res * resid) @ self.U.to(x.dtype)

    def faith_terms(self):
        eye = torch.eye(self.r, device=self.a.device)
        M = eye - self.b.T @ self.a
        return torch.trace(M.T @ self.GV @ M @ self.GU), torch.trace(self.GV @ self.GU)


def install(model, lora, C):
    for p in model.parameters():
        p.requires_grad_(False)
    wr = {}
    for n, uv in lora.items():
        parent, attr = n.rsplit(".", 1)
        lin = model.get_submodule(n)
        w = PDLoRALinear(lin, uv["U"], uv["V"], C)
        setattr(model.get_submodule(parent), attr, w)
        wr[n] = w
    return wr


def build(args):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").to(dev)
    model.config.use_cache = False
    lora = torch.load(args.lora, map_location="cpu")["components"]
    wr = install(model, lora, args.C)
    ci = G.GlobalCI({n: w.base.in_features for n, w in wr.items()}, {n: w.C for n, w in wr.items()},
                    proj=args.ci_proj, hidden=args.ci_hidden).to(dev)
    if getattr(args, "grad_ckpt", 0):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.train()
    return tok, model, wr, ci


def load_ckpt(path, wr, ci):
    st = torch.load(path, map_location=dev)
    with torch.no_grad():
        for n, w in wr.items():
            w.a.copy_(st["components"][n]["a"])
            w.b.copy_(st["components"][n]["b"])
    ci.load_state_dict(st["ci"])
    return st


def set_res(wr, res):
    for w in wr.values():
        w.res = res[w] if isinstance(res, dict) else res


def faithfulness(wr):
    num = den = 0.0
    for w in wr.values():
        a, b = w.faith_terms()
        num, den = num + a, den + b
    return num / den


def forward(model, wr, ids, mode, masks=None, res=1.0):
    G.set_mode(wr, mode, masks)
    set_res(wr, res)
    return model(input_ids=ids).logits


def stoch_masks(ci_pre, wr, routing, necessity=False):
    """Component masks and residual strengths. Sufficiency: m = g + (1-g) s, residual s, unrouted modules run the
    target LoRA (m = 1, residual 1). Necessity: m = (1-g) s, residual s, unrouted modules run the base (0, 0)."""
    gates = {n: G.lower_leaky(z) for n, z in ci_pre.items()}
    src = G.sample_sources(gates)
    masks, res = {}, {}
    for n, g in gates.items():
        lead = g.shape[:-1]
        m = (1 - g) * src[n] if necessity else g + (1 - g) * src[n]
        rs = torch.rand(lead, device=g.device, dtype=g.dtype)
        fill = 0.0 if necessity else 1.0
        if routing is not None:
            m = torch.where(routing[n][..., None], m, torch.full_like(m, fill))
            rs = torch.where(routing[n], rs, torch.full_like(rs, fill))
        masks[n], res[wr[n]] = m, rs
    return masks, res


def pgd_masks(ci_pre, wr, src, necessity=False):
    masks, res = {}, {}
    for n, z in ci_pre.items():
        g = G.lower_leaky(z)
        s = src[n]
        masks[n] = (1 - g) * s[..., :-1] if necessity else g + (1 - g) * s[..., :-1]
        res[wr[n]] = s[..., -1]
    return masks, res


def train(a):
    torch.manual_seed(a.seed)
    G.MASK_DIST = a.mask_dist
    tok, model, wr, ci = build(a)
    names = sorted(wr)
    sft = torch.load(Path(a.data) / "sft_train.pt")
    nt = torch.load(Path(a.nontarget) / "nontarget_train.pt")
    T, pad = a.ctx, tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    params_c = [p for w in wr.values() for p in (w.a, w.b)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    json.dump(vars(a) | {"arch": "pd"}, open(out / "train_args.json", "w"), indent=1)
    log = open(out / "train_log.jsonl", "a")

    # faithfulness warmup: make the components sum to the LoRA before any masking
    wopt = torch.optim.Adam(params_c, lr=a.warmup_lr)
    for i in range(a.faith_warmup):
        loss = faithfulness(wr)
        wopt.zero_grad()
        loss.backward()
        wopt.step()
    print(f"faithfulness after warmup: {faithfulness(wr).item():.2e}", flush=True)

    opt = torch.optim.AdamW([{"params": params_c, "lr": a.lr}, {"params": list(ci.parameters()), "lr": a.lr_ci}], weight_decay=0.0)
    proxies = {n: types.SimpleNamespace(C=w.C + 1) for n, w in wr.items()}  # PGD sources: components + residual
    ppgd = G.PersistentPGD(proxies, a.batch, T, a.pgd_lr, inner=a.pgd_inner)
    ppgd_c = G.PersistentPGD(proxies, a.batch, T, a.pgd_lr, inner=a.pgd_inner) if a.c_comp_adv > 0 else None
    ac = torch.autocast(dev, dtype=torch.bfloat16, enabled=dev == "cuda")
    t0 = time.time()
    for step in range(a.steps):
        frac = step / a.steps
        cos = 0.5 * (1 + math.cos(math.pi * min(1.0, frac / a.anneal_end)))
        for grp, base_lr in zip(opt.param_groups, (a.lr, a.lr_ci)):
            grp["lr"] = base_lr * min(1.0, (step + 1) / a.warmup) * (0.1 + 0.9 * cos)
        lam = a.lam * min(1.0, frac / 0.1)
        p = a.p_start + (a.p_end - a.p_start) * min(1.0, frac / a.anneal_end)
        pgd_lr = a.pgd_lr * min(1.0, frac / 0.025) if frac < 0.025 else a.pgd_lr
        use_adv = a.c_adv > 0 and frac >= a.adv_start
        xt, labels, valid = G.sft_batch(sft, torch.randint(len(sft), (a.batch,)).tolist(), T, pad)
        xn = nt[torch.randint(len(nt), (a.batch_nt,))][:, :T].to(dev)
        ones = torch.ones(xn.shape, device=dev, dtype=torch.bool)
        opt.zero_grad(set_to_none=True)
        rec = {}

        with torch.no_grad(), ac:
            target_t = forward(model, wr, xt, "full")
        cache_t = {n: w.last_input for n, w in wr.items()}
        with ac:
            ci_t = ci(cache_t)
            routing = G.k_subset_routing(names, (a.batch, T), dev)
            m, r = stoch_masks(ci_t, wr, routing)
            l_st = G.kl_loss(target_t, forward(model, wr, xt, "masked", m, r), valid)
            l_min = G.minimality(ci_t, p, a.beta)
        (a.c_stoch * l_st + lam * l_min).backward()
        rec.update(stoch=l_st.item())

        if a.c_comp > 0 or (ppgd_c is not None and use_adv):
            with torch.no_grad(), ac:
                base_t = forward(model, wr, xt, "base")
            if a.c_comp > 0:
                with ac:
                    ci_t = ci(cache_t)
                    m, r = stoch_masks(ci_t, wr, G.k_subset_routing(names, (a.batch, T), dev), necessity=True)
                    l_c = G.kl_loss(base_t, forward(model, wr, xt, "masked", m, r), valid)
                (a.c_comp * l_c).backward()
                rec["comp"] = l_c.item()
            if ppgd_c is not None and use_adv:
                ci_d = {n: z.detach() for n, z in ci(cache_t).items()}
                for _ in range(ppgd_c.inner):
                    src = ppgd_c.sources(True)
                    with ac:
                        m, r = pgd_masks(ci_d, wr, src, necessity=True)
                        l = G.kl_loss(base_t, forward(model, wr, xt, "masked", m, r), valid)
                    ppgd_c.step(dict(zip(src, torch.autograd.grad(l, list(src.values())))), pgd_lr)
                src = ppgd_c.sources(True)
                with ac:
                    m, r = pgd_masks(ci(cache_t), wr, src, necessity=True)
                    l_ca = G.kl_loss(base_t, forward(model, wr, xt, "masked", m, r), valid)
                (a.c_comp_adv * l_ca).backward()
                ppgd_c.step({n: s.grad / a.c_comp_adv for n, s in src.items()}, pgd_lr)
                rec["comp_adv"] = l_ca.item()
            del base_t

        if use_adv:
            ci_d = {n: z.detach() for n, z in ci(cache_t).items()}
            for _ in range(ppgd.inner):
                src = ppgd.sources(True)
                with ac:
                    m, r = pgd_masks(ci_d, wr, src)
                    l = G.kl_loss(target_t, forward(model, wr, xt, "masked", m, r), valid)
                ppgd.step(dict(zip(src, torch.autograd.grad(l, list(src.values())))), pgd_lr)
            src = ppgd.sources(True)
            with ac:
                m, r = pgd_masks(ci(cache_t), wr, src)
                l_adv = G.kl_loss(target_t, forward(model, wr, xt, "masked", m, r), valid)
            (a.c_adv * l_adv).backward()
            ppgd.step({n: s.grad / a.c_adv for n, s in src.items()}, pgd_lr)
            rec["adv"] = l_adv.item()
        del target_t

        with torch.no_grad(), ac:
            target_n = forward(model, wr, xn, "full")
        with ac:
            ci_n = ci({n: w.last_input for n, w in wr.items()})
            m, r = stoch_masks(ci_n, wr, G.k_subset_routing(names, tuple(xn.shape), dev))
            l_nt = G.kl_loss(target_n, forward(model, wr, xn, "masked", m, r), ones)
            l_min_n = G.minimality(ci_n, p, a.beta)
        l_f = faithfulness(wr)
        (a.c_nt * l_nt + lam * a.nt_ratio * l_min_n + a.c_faith * l_f).backward()
        rec.update(nt_kl=l_nt.item(), faith=l_f.item())
        torch.nn.utils.clip_grad_norm_(params_c, 1.0)
        torch.nn.utils.clip_grad_norm_(list(ci.parameters()), 1.0)
        opt.step()

        if step % a.log_every == 0 or step == a.steps - 1:
            with torch.no_grad():
                gt = torch.cat([G.lower_leaky(z)[valid] for z in ci_t.values()], -1)
                gn = torch.cat([G.lower_leaky(z).flatten(0, 1) for z in ci_n.values()], -1)
                rec.update(l0_t=(gt > 0.1).float().sum(-1).mean().item(), l0_n=(gn > 0.1).float().sum(-1).mean().item(),
                           p=round(p, 3), lam=lam, mem_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1) if dev == "cuda" else 0)
            rec = {"step": step, **{k: (round(v, 6) if isinstance(v, float) else v) for k, v in rec.items()}, "s": round(time.time() - t0)}
            log.write(json.dumps(rec) + "\n")
            log.flush()
            print(rec, flush=True)
        if (step + 1) % a.save_every == 0 or step == a.steps - 1:
            torch.save({"components": {n: {"a": w.a.detach().cpu(), "b": w.b.detach().cpu()} for n, w in wr.items()},
                        "ci": ci.state_dict(), "args": vars(a) | {"arch": "pd"}, "step": step + 1}, out / f"ckpt_{step + 1}.pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("--lora", required=True, help="ordinary LoRA in em_pilot.py format (components: U, V per module)")
    ap.add_argument("--data", required=True, help="em_data dir with sft_train.pt / sft_val.pt")
    ap.add_argument("--nontarget", required=True, help="dir with nontarget_train.pt (FineWeb chunks)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--C", type=int, default=64, help="components per module (the LoRA has rank r = 32)")
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--batch_nt", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lr_ci", type=float, default=5e-4)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--anneal_end", type=float, default=0.8)
    ap.add_argument("--faith_warmup", type=int, default=1000)
    ap.add_argument("--warmup_lr", type=float, default=1e-2)
    ap.add_argument("--c_faith", type=float, default=100.0)
    ap.add_argument("--c_stoch", type=float, default=1.0)
    ap.add_argument("--c_adv", type=float, default=0.5)
    ap.add_argument("--adv_start", type=float, default=0.1)
    ap.add_argument("--c_nt", type=float, default=1.0)
    ap.add_argument("--c_comp", type=float, default=0.0)
    ap.add_argument("--c_comp_adv", type=float, default=0.0)
    ap.add_argument("--lam", type=float, default=1e-3)
    ap.add_argument("--nt_ratio", type=float, default=2.0)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--p_start", type=float, default=2.0)
    ap.add_argument("--p_end", type=float, default=1.0)
    ap.add_argument("--pgd_lr", type=float, default=0.01)
    ap.add_argument("--pgd_inner", type=int, default=2)
    ap.add_argument("--mask_dist", choices=["uniform", "bernoulli"], default="uniform")
    ap.add_argument("--ci_proj", type=int, default=64)
    ap.add_argument("--ci_hidden", type=int, default=512)
    ap.add_argument("--grad_ckpt", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--save_every", type=int, default=1000)
    a = ap.parse_args()
    train(a)


if __name__ == "__main__":
    main()
