"""Toy check of the necessity loss for the reference-free gated LoRA (pd/lora/gated_lora.py).

Task (cross-entropy, no reference model). Inputs x in [0,1]^n are sparse (each feature on w.p. p, value U(0,1)).
Labels are SAMPLED from softmax(score), so the fine-tuning loss has the LM's one-token-target property:
    base       score = A x
    fine-tune  score = A x + M * sum_k x[f_k] e_{c_k}        K "behaviours": feature f_k pushes class c_k
A 2-layer ReLU MLP is trained as the base model (on base probabilities). Then a gated LoRA with C rank-one pieces
on each layer is trained on fine-tune labels with the losses of gated_lora.py (option A: masked CE on the sampled
labels with k-subset routing, persistent PGD, KL to the base on nontarget inputs, VPD minimality), with and
without the mirrored necessity terms (masks (1-g) s, KL to the base, stochastic + adversarial). The mask,
minimality, PGD, CI and wrapper code are imported from gated_lora.py, so this tests that implementation.

Ground truth: behaviour k should be carried by a few pieces whose gates are on only when f_k is active;
removing those pieces should remove behaviour k (necessity) and leave the other behaviours alone.

  python toy_necessity.py                          (CPU; 5 conditions x 3 seeds)
  python toy_necessity.py --conds vpd vpd_nec --seeds 0 --steps 1000      (quick check)
"""
import argparse
import json
import math
import statistics as st
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gated_lora as G  # noqa: E402

dev = G.dev
OUT = HERE / "runs" / "toy_necessity"

CONDS = {
    "plain": dict(decomp=False),
    "vpd": dict(decomp=True, mask_dist="uniform", c_comp=0.0, c_comp_adv=0.0),
    "vpd_nec": dict(decomp=True, mask_dist="uniform", c_comp=1.0, c_comp_adv=0.5),
    "vpd_bern": dict(decomp=True, mask_dist="bernoulli", c_comp=0.0, c_comp_adv=0.0),
    "vpd_bern_nec": dict(decomp=True, mask_dist="bernoulli", c_comp=1.0, c_comp_adv=0.5),
}


# ------------------------------------------------------------------ task + model
class Task:
    def __init__(self, seed, n=24, V=8, K=3, p=0.1, p_b=0.3, M=4.0, scale=1.5):
        g = torch.Generator().manual_seed(1000 + seed)
        self.n, self.V, self.K, self.p, self.p_b, self.M = n, V, K, p, p_b, M
        self.A = (torch.randn(V, n, generator=g) * scale).to(dev)
        self.f = list(range(K))  # behaviour features
        self.c = torch.randperm(V, generator=g)[:K].tolist()  # class each behaviour pushes

    def sample(self, B, kind="target", k=None):
        """target: behaviour features on w.p. p_b; nontarget: behaviour features off; single: only behaviour k on."""
        x = (torch.rand(B, self.n) < self.p).float() * torch.rand(B, self.n)
        x[:, self.f] = 0
        if kind == "target":
            x[:, self.f] = (torch.rand(B, self.K) < self.p_b).float() * torch.rand(B, self.K)
        elif kind == "single":
            x[:, self.f[k]] = 0.5 + 0.5 * torch.rand(B)
        return x.to(dev)

    def probs(self, x, ft):
        sc = x @ self.A.T
        if ft:
            for k in range(self.K):
                sc[:, self.c[k]] = sc[:, self.c[k]] + self.M * x[:, self.f[k]]
        return F.softmax(sc, -1)


class MLP(nn.Module):
    def __init__(self, n, d, V):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(n, d), nn.Linear(d, V)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


def fwd(model, wr, x, mode, masks=None):
    """Wrapper inputs are (B, 1, d), so masks are (B, 1, C), the same layout as the LM code."""
    G.set_mode(wr, mode, masks)
    return model(x[:, None])[:, 0]


def ce_hard(logits, y):
    return F.cross_entropy(logits.float(), y)


def ce_soft(logits, p):
    return -(p * F.log_softmax(logits.float(), -1)).sum(-1).mean()


def kl(ref, logits):
    lp, lq = F.log_softmax(ref.detach().float(), -1), F.log_softmax(logits.float(), -1)
    return (lp.exp() * (lp - lq)).sum(-1).mean()


def train_base(task, d, seed, steps=3000, B=512):
    torch.manual_seed(seed)
    m = MLP(task.n, d, task.V).to(dev)
    opt = torch.optim.Adam(m.parameters(), 3e-3)
    for _ in range(steps):
        x = task.sample(B)
        loss = ce_soft(m(x), task.probs(x, False))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {k: v.detach().clone() for k, v in m.state_dict().items()}


# ------------------------------------------------------------------ gated LoRA training (mirrors gated_lora.train)
def build(task, base_sd, d, C, seed):
    torch.manual_seed(seed)
    model = MLP(task.n, d, task.V).to(dev)
    model.load_state_dict(base_sd)
    for p in model.parameters():
        p.requires_grad_(False)
    wr = {"fc1": G.GatedLoRALinear(model.fc1, C), "fc2": G.GatedLoRALinear(model.fc2, C)}
    model.fc1, model.fc2 = wr["fc1"], wr["fc2"]
    ci = G.GlobalCI({n: w.base.in_features for n, w in wr.items()}, {n: C for n in wr}, proj=16, hidden=64).to(dev)
    return model, wr, ci


def train_gated(task, model, wr, ci, cfg, a):
    names = sorted(wr)
    G.MASK_DIST = cfg.get("mask_dist", "uniform")
    params_c = [p for w in wr.values() for p in (w.U, w.V)]
    opt = torch.optim.Adam([{"params": params_c, "lr": a.lr}, {"params": list(ci.parameters()), "lr": a.lr}])
    B = a.batch
    ppgd = G.PersistentPGD(wr, B, 1, a.pgd_lr, inner=2)
    ppgd_comp = G.PersistentPGD(wr, B, 1, a.pgd_lr, inner=2) if cfg.get("c_comp_adv", 0) > 0 else None
    log = []
    for step in range(a.steps):
        frac = step / a.steps
        lr_mult = min(1.0, (step + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for grp in opt.param_groups:
            grp["lr"] = a.lr * lr_mult
        lam = a.lam * min(1.0, frac / 0.1)
        p = 2.0 - frac
        use_adv = cfg["decomp"] and frac >= 0.1
        xt = task.sample(B)
        yt = torch.multinomial(task.probs(xt, True), 1)[:, 0]
        xn = task.sample(B, "nontarget")
        opt.zero_grad(set_to_none=True)
        rec = {}

        lf = fwd(model, wr, xt, "full")
        ce_full = ce_hard(lf, yt)
        ce_full.backward()
        rec["ce_full"] = ce_full.item()
        cache = {n: w.last_input for n, w in wr.items()}

        if cfg["decomp"]:
            # sufficiency, stochastic
            ci_t = ci(cache)
            routing = G.k_subset_routing(names, (B, 1), dev)
            l_st = ce_hard(fwd(model, wr, xt, "masked", G.make_masks(ci_t, routing=routing)), yt)
            l_min = G.minimality(ci_t, p, a.beta)
            (a.c_stoch * l_st + lam * l_min).backward()
            rec.update(stoch=l_st.item())

            # necessity (mirror): masks (1-g) s, target = base
            if cfg["c_comp"] > 0 or (ppgd_comp is not None and use_adv):
                with torch.no_grad():
                    lb = fwd(model, wr, xt, "base")
                if cfg["c_comp"] > 0:
                    ci_t = ci(cache)
                    routing = G.k_subset_routing(names, (B, 1), dev)
                    l_c = kl(lb, fwd(model, wr, xt, "masked", G.make_comp_masks(ci_t, routing=routing)))
                    (cfg["c_comp"] * l_c).backward()
                    rec["comp"] = l_c.item()
                if ppgd_comp is not None and use_adv:
                    ci_d = {n: z.detach() for n, z in ci(cache).items()}
                    for _ in range(ppgd_comp.inner):
                        src = ppgd_comp.sources(True)
                        l = kl(lb, fwd(model, wr, xt, "masked", G.make_comp_masks(ci_d, src=src)))
                        ppgd_comp.step(dict(zip(src, torch.autograd.grad(l, list(src.values())))), a.pgd_lr)
                    ci_t = ci(cache)
                    src = ppgd_comp.sources(True)
                    l_ca = kl(lb, fwd(model, wr, xt, "masked", G.make_comp_masks(ci_t, src=src)))
                    (cfg["c_comp_adv"] * l_ca).backward()
                    ppgd_comp.step({n: t.grad / cfg["c_comp_adv"] for n, t in src.items()}, a.pgd_lr)
                    rec["comp_adv"] = l_ca.item()

            # sufficiency, adversarial
            if use_adv:
                ci_d = {n: z.detach() for n, z in ci(cache).items()}
                for _ in range(ppgd.inner):
                    src = ppgd.sources(True)
                    l = ce_hard(fwd(model, wr, xt, "masked", G.make_masks(ci_d, src=src)), yt)
                    ppgd.step(dict(zip(src, torch.autograd.grad(l, list(src.values())))), a.pgd_lr)
                ci_t = ci(cache)
                src = ppgd.sources(True)
                l_adv = ce_hard(fwd(model, wr, xt, "masked", G.make_masks(ci_t, src=src)), yt)
                (a.c_adv * l_adv).backward()
                ppgd.step({n: t.grad / a.c_adv for n, t in src.items()}, a.pgd_lr)
                rec["adv"] = l_adv.item()

            # nontarget: masked model should match the base
            with torch.no_grad():
                lbn = fwd(model, wr, xn, "base")
                fwd(model, wr, xn, "full")
            ci_n = ci({n: w.last_input for n, w in wr.items()})
            l_nt = kl(lbn, fwd(model, wr, xn, "masked", G.make_masks(ci_n)))
            l_min_n = G.minimality(ci_n, p, a.beta)
            (a.c_nt * l_nt + lam * a.nt_ratio * l_min_n).backward()
            rec["nt_kl"] = l_nt.item()

        torch.nn.utils.clip_grad_norm_(params_c, 1.0)
        torch.nn.utils.clip_grad_norm_(list(ci.parameters()), 1.0)
        opt.step()
        if step % 250 == 0 or step == a.steps - 1:
            if cfg["decomp"]:
                with torch.no_grad():
                    gt = torch.cat([G.lower_leaky(z) for z in ci_t.values()], -1)
                    rec["l0_t"] = (gt > a.thr).float().sum(-1).mean().item()
            log.append({"step": step, **{k: round(v, 4) for k, v in rec.items()}})
    return log


# ------------------------------------------------------------------ eval with ground truth
@torch.no_grad()
def evaluate(task, model, wr, ci, a, gated=True, N=8192):
    thr = a.thr
    names = sorted(wr)
    C = wr["fc1"].C

    def run(x):
        out = {"full": fwd(model, wr, x, "full")}
        g = None
        if gated:
            g = {n: G.lower_leaky(z) for n, z in ci({n: w.last_input for n, w in wr.items()}).items()}
        out["base"] = fwd(model, wr, x, "base")
        if gated:
            for name, ms in (("imp", {n: (v > thr).float() for n, v in g.items()}),
                             ("unimp", {n: (v <= thr).float() for n, v in g.items()}),
                             ("comp", {n: 1 - v for n, v in g.items()}),
                             ("ci", g)):
                out[name] = fwd(model, wr, x, "masked", ms)
        return out, g

    r = {}
    xt = task.sample(N)
    pt = task.probs(xt, True)
    out, gt = run(xt)
    r["ce_target"] = {k: ce_soft(v, pt).item() for k, v in out.items()}
    r["ce_target_optimal"] = ce_soft(torch.log(pt), pt).item()
    r["kl_to_full"] = {k: kl(out["full"], v).item() for k, v in out.items()}
    r["kl_to_base"] = {k: kl(out["base"], v).item() for k, v in out.items()}
    xn = task.sample(N, "nontarget")
    outn, gn = run(xn)
    r["nontarget_kl_to_base"] = {k: kl(outn["base"], v).item() for k, v in outn.items()}

    # per behaviour: log p(c_k) on inputs where only behaviour k is on
    beh = []
    for k in range(task.K):
        xk = task.sample(N, "single", k)
        ok, gk = run(xk)
        lp = {name: F.log_softmax(v.float(), -1)[:, task.c[k]].mean().item() for name, v in ok.items()}
        b = dict(k=k, logp=lp)
        span = lp["full"] - lp["base"]
        if gated:
            b["imp_recovers"] = (lp["imp"] - lp["base"]) / span
            b["unimp_removes"] = (lp["full"] - lp["unimp"]) / span
            b["comp_left"] = (lp["comp"] - lp["base"]) / span
            b["mean_gate"] = torch.cat([gk[n][:, 0] for n in names], -1).mean(0)  # (2C,)
        beh.append(b)
    if gated:
        r["l0_target"] = (torch.cat([gt[n][:, 0] for n in names], -1) > thr).float().sum(-1).mean().item()
        r["l0_nontarget"] = (torch.cat([gn[n][:, 0] for n in names], -1) > thr).float().sum(-1).mean().item()
        gall = torch.cat([gt[n][:, 0] for n in names], -1)
        r["gate_hist_target"] = dict(zip(G.GATE_BINS, (G.gate_hist(gall) / gall.shape[0]).tolist()))
        # carriers of behaviour k: pieces with mean gate > 0.5 on single-k inputs. Ablate them everywhere and measure
        # how much of each behaviour is left (own: should be ~0; others: should be ~1)
        mg = torch.stack([b["mean_gate"] for b in beh])  # (K, 2C)
        mg_nt = torch.cat([gn[n][:, 0] for n in names], -1).mean(0)
        carriers = [(mg[k] > 0.5).nonzero().flatten().tolist() for k in range(task.K)]
        abl = []
        for k in range(task.K):
            keep = torch.ones(2 * C, device=dev)
            keep[carriers[k]] = 0
            ms = {"fc1": keep[:C].expand(N, 1, C), "fc2": keep[C:].expand(N, 1, C)}
            row = []
            for j in range(task.K):
                xj = task.sample(N, "single", j)
                lf = F.log_softmax(fwd(model, wr, xj, "full").float(), -1)[:, task.c[j]].mean().item()
                lb = F.log_softmax(fwd(model, wr, xj, "base").float(), -1)[:, task.c[j]].mean().item()
                la = F.log_softmax(fwd(model, wr, xj, "masked", ms).float(), -1)[:, task.c[j]].mean().item()
                row.append((la - lb) / (lf - lb))  # fraction of behaviour j left after removing carriers of k
            abl.append(row)
        r["carriers"] = [[(("fc1" if i < C else "fc2"), i % C) for i in cs] for cs in carriers]
        r["n_carriers"] = [len(cs) for cs in carriers]
        r["n_shared_carriers"] = int(sum((mg[:, i] > 0.5).sum() > 1 for i in range(2 * C)))
        r["carrier_gate_nontarget"] = [mg_nt[cs].mean().item() if cs else float("nan") for cs in carriers]
        r["carrier_ablation_left"] = abl  # abl[k][j]
        r["own_left_after_carrier_ablation"] = st.mean(abl[k][k] for k in range(task.K))
        r["others_left_after_carrier_ablation"] = st.mean(abl[k][j] for k in range(task.K) for j in range(task.K) if j != k)
    for b in beh:
        b.pop("mean_gate", None)
    r["behaviours"] = beh
    if gated:
        for key in ("imp_recovers", "unimp_removes", "comp_left"):
            r[key] = st.mean(b[key] for b in beh)
    return r


# ------------------------------------------------------------------ driver
def summarise(results):
    keys = [("ce_target.full", "fine-tune CE (full)"), ("nontarget_kl_to_base.full", "nontarget KL to base (full)"),
            ("l0_target", "L0 target"), ("l0_nontarget", "L0 nontarget"),
            ("imp_recovers", "imp recovers behaviour"), ("kl_to_full.imp", "KL(full || imp)"),
            ("unimp_removes", "removing imp removes"), ("comp_left", "behaviour left in 1-g complement"),
            ("kl_to_base.comp", "KL(base || 1-g complement)"),
            ("own_left_after_carrier_ablation", "own behaviour left, carriers ablated"),
            ("others_left_after_carrier_ablation", "other behaviours left, carriers ablated"),
            ("n_shared_carriers", "carriers shared by >1 behaviour")]

    def get(r, path):
        v = r
        for part in path.split("."):
            if not isinstance(v, dict) or part not in v:
                return None
            v = v[part]
        return v

    conds = list(dict.fromkeys(c for c, _ in results))
    lines = [f"{'metric':42s}" + "".join(f"{c:>18s}" for c in conds)]
    for path, label in keys:
        row = f"{label:42s}"
        for c in conds:
            vals = [get(r, path) for cc, r in results if cc == c]
            vals = [v for v in vals if v is not None]
            row += f"{'-':>18s}" if not vals else f"{st.mean(vals):11.3f} ±{(st.pstdev(vals) if len(vals) > 1 else 0):5.3f}"
        lines.append(row)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", nargs="+", default=list(CONDS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--d", type=int, default=32)
    ap.add_argument("--C", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--lam", type=float, default=3e-3)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--nt_ratio", type=float, default=2.0)
    ap.add_argument("--c_stoch", type=float, default=1.0)
    ap.add_argument("--c_adv", type=float, default=0.5)
    ap.add_argument("--c_nt", type=float, default=1.0)
    ap.add_argument("--pgd_lr", type=float, default=0.01)
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for seed in a.seeds:
        task = Task(seed)
        base_sd = train_base(task, a.d, seed)
        for c in a.conds:
            cfg = CONDS[c]
            t0 = time.time()
            model, wr, ci = build(task, base_sd, a.d, a.C, seed)
            log = train_gated(task, model, wr, ci, cfg, a)
            r = evaluate(task, model, wr, ci, a, gated=cfg["decomp"])
            r.update(cond=c, seed=seed, log=log, args=vars(a), seconds=round(time.time() - t0))
            results.append((c, r))
            json.dump(r, open(OUT / f"{c}_s{seed}{a.tag}.json", "w"), indent=1)
            brief = {k: (round(r[k], 3) if isinstance(r.get(k), float) else r.get(k))
                     for k in ("l0_target", "imp_recovers", "unimp_removes", "comp_left", "own_left_after_carrier_ablation", "n_carriers")}
            print(f"[seed {seed}] {c:13s} {r['seconds']:4d}s  CE full {r['ce_target']['full']:.3f} (base {r['ce_target']['base']:.3f}, "
                  f"optimal {r['ce_target_optimal']:.3f})  {brief}", flush=True)
    s = summarise(results)
    print("\n" + s)
    (OUT / f"summary{a.tag}.txt").write_text(s + "\n")


if __name__ == "__main__":
    main()
