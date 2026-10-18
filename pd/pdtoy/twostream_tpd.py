"""Two-stream decomposition on the pdtoy TMS, trained with the tPD pieces twostream.py lacks.

Same toy, fine-tunes, base decomposition (pd_base_gt.pt) and evaluation as twostream.py, but:
  - target / nontarget batches (tPD): target batches force the changed feature on (E for elicit,
    S for teach, E or S per row for both); nontarget batches force E and S off.
  - residual masks: random + adversarial (PGD) on target batches, fixed at 1 on nontarget batches,
    no faithfulness penalty on the residuals (as in tPD / the design note).
  - masks use the lower-leaky hard sigmoid, minimality the upper-leaky one (SPD convention;
    pd.py has them swapped), p annealed 2 -> 1.
  - extra check: exhaustive search over the masks of all "unimportant" delta subcomponents and the
    delta residual, reporting the largest change in the behaviour output they can cause.

Usage: python twostream_tpd.py <elicit|teach|both> <frozen|joint> [--seed S] [--cd 4]
Writes runs/<kind>_<mode>_cd<cd>_s<seed>.json. Uses CUDA when available.
"""
import argparse
import itertools
import json
from pathlib import Path

import torch
import torch.nn as nn

from common import N_FEAT, P_ACTIVE
from finetune import E, RANK, S, T
from pd import CIFunction, DecomposedMatrix

ap = argparse.ArgumentParser()
ap.add_argument("kind", choices=["elicit", "teach", "both"])
ap.add_argument("mode", choices=["frozen", "joint"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--cd", type=int, default=4, help="number of delta subcomponents")
ap.add_argument("--steps", type=int, default=10_000)
ap.add_argument("--batch", type=int, default=512)
ap.add_argument("--lr", type=float, default=2e-3)
ap.add_argument("--lam", type=float, default=1e-2, help="minimality coefficient")
ap.add_argument("--nt_ratio", type=float, default=2.0)
ap.add_argument("--pgd_coeff", type=float, default=0.5)
ap.add_argument("--thr", type=float, default=0.5)
args = ap.parse_args()

dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.set_num_threads(2)
torch.manual_seed(args.seed)
HERE = Path(__file__).parent
load = lambda name: torch.load(HERE / name, map_location=dev)


class _LowerLeaky(torch.autograd.Function):
    """Forward clamp(z, 0, 1); below 0 the gradient leaks (x alpha) only when it would raise z."""

    @staticmethod
    def forward(ctx, z, alpha):
        ctx.save_for_backward(z)
        ctx.alpha = alpha
        return z.clamp(0, 1)

    @staticmethod
    def backward(ctx, grad):
        (z,) = ctx.saved_tensors
        low = torch.where(grad < 0, ctx.alpha * grad, torch.zeros_like(grad))
        return torch.where(z <= 0, low, torch.where(z <= 1, grad, torch.zeros_like(grad))), None


def lower_leaky(z, alpha=0.01):
    return _LowerLeaky.apply(z, alpha)


def upper_leaky(z, alpha=0.01):
    return torch.where(z > 1, 1 + alpha * (z - 1), z.clamp(min=0, max=1))


# ---------------------------------------------------------------- frozen data
base = load("base.pt")
W_in, W_out, bias = base["W_in"], base["W_out"], base["b"]
ft = load("finetunes.pt")[args.kind]
Bm, Am = ft["B"], ft["A"]
W_ft = W_out + Bm @ Am
pdb = load("pd_base_gt.pt")

d_in = DecomposedMatrix(*W_in.shape, N_FEAT).to(dev)
d_in.load_state_dict(pdb["dW_in"])
d_out = DecomposedMatrix(*W_out.shape, N_FEAT).to(dev)
d_out.load_state_dict(pdb["dW_out"])
for p in [d_in.U, d_in.V, d_out.U, d_out.V]:
    p.requires_grad_(args.mode == "joint")
ci = nn.ModuleDict()
for s in "bf":
    ci[s + "_in"] = CIFunction(N_FEAT)
    ci[s + "_in"].load_state_dict(d_in.ci.state_dict())
    ci[s + "_out"] = CIFunction(N_FEAT)
    ci[s + "_out"].load_state_dict(d_out.ci.state_dict())
ci = ci.to(dev)
CD = args.cd
a = nn.Parameter(torch.randn(CD, RANK, device=dev) * 0.5)
bb = nn.Parameter(torch.randn(CD, RANK, device=dev) * 0.5)
ci_d = CIFunction(CD).to(dev)

dU = lambda: Bm @ a.T  # [20, CD]  columns B a_i
dV = lambda: Am.T @ bb.T  # [5, CD]   columns A^T b_i
R_in = lambda: W_in - d_in.U @ d_in.V.T
R_out = lambda: W_out - d_out.U @ d_out.V.T
R_d = lambda: Bm @ Am - dU() @ dV().T


# ---------------------------------------------------------------- data
def sample(bs, target):
    act = (torch.rand(bs, N_FEAT, device=dev) < P_ACTIVE).float()
    if target:
        if args.kind == "elicit":
            f = torch.full((bs,), E, device=dev)
        elif args.kind == "teach":
            f = torch.full((bs,), S, device=dev)
        else:
            f = torch.where(torch.rand(bs, device=dev) < 0.5, E, S)
        act[torch.arange(bs, device=dev), f] = 1.0
    else:
        act[:, E] = 0.0
        act[:, S] = 0.0
    return act * torch.rand(bs, N_FEAT, device=dev)


# ---------------------------------------------------------------- streams
def ci_pre(x, h_t, s):
    z = {"in": ci[s + "_in"](x @ d_in.V), "out": ci[s + "_out"](h_t @ d_out.V)}
    if s == "f":
        z["d"] = ci_d(h_t @ dV())
    return z


def forward(x, s, m):
    h = (x @ d_in.V * m["in"]) @ d_in.U.T + m["rin"][:, None] * (x @ R_in().T)
    pre = (h @ d_out.V * m["out"]) @ d_out.U.T + m["rout"][:, None] * (h @ R_out().T)
    if s == "f":
        pre = pre + (h @ dV() * m["d"]) @ dU().T + m["rd"][:, None] * (h @ R_d().T)
    return torch.relu(pre + bias)


RES = {"b": ["rin", "rout"], "f": ["rin", "rout", "rd"]}


def make_masks(z, s, resid, src=None):
    """m = g + (1-g) * src (src ~ U(0,1) unless given); residual masks 'rand' | float | from src."""
    m = {}
    for k, zk in z.items():
        g = lower_leaky(zk)
        m[k] = g + (1 - g) * (src[k] if src is not None else torch.rand_like(g))
    bs = next(iter(z.values())).shape[0]
    for k in RES[s]:
        if src is not None:
            m[k] = src[k]
        elif resid == "rand":
            m[k] = torch.rand(bs, device=dev)
        else:
            m[k] = torch.full((bs,), float(resid), device=dev)
    return m


def pgd(x, y, z, s, n_steps=3, step=0.1):
    zd = {k: v.detach() for k, v in z.items()}
    bs = x.shape[0]
    src = {k: torch.rand_like(v) for k, v in zd.items()}
    src.update({k: torch.rand(bs, device=dev) for k in RES[s]})
    for _ in range(n_steps):
        for v in src.values():
            v.requires_grad_(True)
        loss = ((forward(x, s, make_masks(zd, s, None, src)) - y) ** 2).sum(-1).mean()
        grads = torch.autograd.grad(loss, list(src.values()))
        src = {k: (v + step * g.sign()).clamp(0, 1).detach() for (k, v), g in zip(src.items(), grads)}
    return src


def minimality(z, p):
    return sum(((upper_leaky(v) + 1e-12) ** p).mean(0).sum() for v in z.values())


def targets(x):
    with torch.no_grad():
        h_t = x @ W_in.T
        return h_t, {"b": torch.relu(h_t @ W_out.T + bias), "f": torch.relu(h_t @ W_ft.T + bias)}


# ---------------------------------------------------------------- train
params = [a, bb] + list(ci_d.parameters()) + list(ci.parameters())
if args.mode == "joint":
    params += [d_in.U, d_in.V, d_out.U, d_out.V]
opt = torch.optim.Adam(params, lr=args.lr)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
for step in range(args.steps):
    frac = step / args.steps
    p = 2.0 - min(1.0, frac / 0.8)
    lam = args.lam * min(1.0, 0.05 + frac / 0.1)
    xt, xn = sample(args.batch, True), sample(args.batch, False)
    (ht, yt), (hn, yn) = targets(xt), targets(xn)
    loss, log = 0.0, {}
    for s in "bf":
        z = ci_pre(xt, ht, s)
        rec = ((forward(xt, s, make_masks(z, s, "rand")) - yt[s]) ** 2).sum(-1).mean()
        adv = torch.zeros((), device=dev)
        if frac > 0.1:
            src = pgd(xt, yt[s], z, s)
            adv = ((forward(xt, s, make_masks(z, s, None, src)) - yt[s]) ** 2).sum(-1).mean()
        zn = ci_pre(xn, hn, s)
        recn = ((forward(xn, s, make_masks(zn, s, 1.0)) - yn[s]) ** 2).sum(-1).mean()
        loss = loss + rec + args.pgd_coeff * adv + recn + lam * minimality(z, p) \
            + args.nt_ratio * lam * minimality(zn, p)
        log[s] = (rec.item(), adv.item(), recn.item())
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    if step % 2000 == 0 or step == args.steps - 1:
        print(f"step {step} " + " | ".join(f"{s}: rec {v[0]:.4f} adv {v[1]:.4f} recNT {v[2]:.4f}"
                                           for s, v in log.items()), flush=True)

# ---------------------------------------------------------------- evaluate
rep = {"args": vars(args)}
with torch.no_grad():
    thr = args.thr
    oh = torch.eye(N_FEAT, device=dev) * 0.75
    h1 = oh @ W_in.T
    zb, zf = ci_pre(oh, h1, "b"), ci_pre(oh, h1, "f")
    g = lambda v: v.clamp(0, 1)
    rep["base_drift"] = {  # how far joint training moved the base subcomponents
        "W_out_subcomp_cos_to_init": torch.nn.functional.cosine_similarity(
            torch.einsum("ic,jc->cij", d_out.U, d_out.V).flatten(1),
            torch.einsum("ic,jc->cij", *(pdb["dW_out"][k].to(dev) for k in ("U", "V"))).flatten(1),
            dim=1).tolist(),
        "resid_out_norm": R_out().norm().item(), "resid_delta_rel": (R_d().norm() / (Bm @ Am).norm()).item(),
    }
    Md = torch.einsum("ic,jc->cij", dU(), dV())
    Mb = torch.einsum("ic,jc->cij", d_out.U, d_out.V)
    cos = torch.einsum("cij,kij->ck", Md, Mb) / (Md.flatten(1).norm(dim=1)[:, None] * Mb.flatten(1).norm(dim=1)[None])
    gd = g(zf["d"])
    rep["delta"] = []
    for i in range(CD):
        k = cos[i].abs().argmax().item()
        rep["delta"].append(dict(norm=Md[i].norm().item(), important_on=torch.nonzero(gd[:, i] > thr).flatten().tolist(),
                                 best_base=k, abs_cos=cos[i, k].abs().item(),
                                 abs_cos_E=cos[i, E].abs().item(), abs_cos_T=cos[i, T].abs().item()))

    def ft_out(x, dmask, rd):
        h = x @ W_in.T
        pre = h @ W_out.T + (h @ dV() * dmask) @ dU().T + rd * (h @ R_d().T)
        return torch.relu(pre + bias)

    rep["inputs"] = {}
    feats = {"elicit": [("E", E, E)], "teach": [("S", S, T)], "both": [("E", E, E), ("S", S, T)]}[args.kind]
    for name, f, o in feats:
        imp = gd[f] > thr
        x = oh[f:f + 1]
        r = dict(
            base_imp_basestream=torch.nonzero(g(zb["out"][f]) > thr).flatten().tolist(),
            base_imp_ftstream=torch.nonzero(g(zf["out"][f]) > thr).flatten().tolist(),
            delta_imp=torch.nonzero(imp).flatten().tolist(),
            y_base=torch.relu(h1[f] @ W_out.T + bias)[o].item(),
            y_ft=ft_out(x, torch.ones(CD, device=dev), 1.0)[0, o].item(),
            y_ablate_important=ft_out(x, (~imp).float(), 1.0)[0, o].item(),
            y_only_important=ft_out(x, imp.float(), 0.0)[0, o].item(),
        )
        # exhaustive corners over unimportant delta masks + delta residual: worst-case drift
        unimp = torch.nonzero(~imp).flatten().tolist()
        worst = 0.0
        for corner in itertools.product([0.0, 1.0], repeat=len(unimp) + 1):
            dm = imp.float().clone()
            for j, c in zip(unimp, corner[:-1]):
                dm[j] = c
            worst = max(worst, abs(ft_out(x, dm, corner[-1])[0, o].item() - r["y_ft"]))
        r["worst_drift_unimportant"] = worst
        # importance-weighted alignment of the delta with base subcomponents (the elicit/teach readout)
        w = Md.flatten(1).norm(dim=1) * imp.float()
        r["alignment"] = (cos.abs().max(1).values * w).sum().item() / max(w.sum().item(), 1e-12)
        rep["inputs"][name] = r
    print(json.dumps(rep, indent=1))
out = HERE / "runs"
out.mkdir(exist_ok=True)
(out / f"{args.kind}_{args.mode}_cd{CD}_s{args.seed}.json").write_text(json.dumps(rep, indent=1))
