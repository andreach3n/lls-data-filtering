"""Two-stream targeted PD on a toy MLP: does the decomposition say "elicited" vs "taught" correctly?

Setup (no superposition, so ground truth is readable off the weights):
  x in R^N, sparse, each feature active w.p. P_ACT with value U(-1,1).
  model: y = W2 relu(W1 x), no biases.
  base is trained on y_i = s_base[i] * relu(x_i) (so each feature needs its own neuron); FT = base + rank-R LoRA (both layers) trained on s_ft.
  Target feature K.  Regimes (s_base[K] -> s_ft[K]):
    elicit : 0.3 -> 1.0   (base already has the K pathway; FT turns it up)
    teach  : 0.0 -> 1.0   (base has no K pathway; FT must build it)
    null   : K unchanged at 0.3; FT instead changes feature Q (1.0 -> 0.3)

Decomposition (the handwritten note):
  base stream  W_b' = sum_j m_j^(b) U_j V_j^T + m_Db^(b) Delta_base
  ft stream    W_f' = sum_j m_j^(f) U_j V_j^T + m_Db^(f) Delta_base
                    + sum_i n_i B a_i b_i^T A + n_Dd (BA - sum_i B a_i b_i^T A)
  U, V, a, b and Delta_base shared; CI functions separate per stream (delta CI only in ft).
  Target batches (x_K active): residual masks are random (stochastic loss) and adversarial (PGD loss).
  Nontarget batches (x_K = 0): residual masks forced to 1, minimality x NONTARGET_IMPMIN_RATIO.
  This mirrors tPD (Vigouroux & Sharkey 2026, spd/run_spd.py) except: MSE output loss (toy is a
  regression), fresh 2-step PGD per training step instead of persistent PGD, no beta log term.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

N, H, R = 12, 12, 4  # features, hidden width, LoRA rank
K, Q = 0, 1  # target feature, null-regime feature
P_ACT = 0.08
OUT = Path(__file__).parent / "results"

REGIMES = {
    "elicit": dict(base={K: 0.3}, ft={K: 1.0}),
    "teach": dict(base={K: 0.0}, ft={K: 1.0}),
    "null": dict(base={K: 0.3}, ft={K: 0.3, Q: 0.3}),
}


def scales(over: dict) -> torch.Tensor:
    s = torch.ones(N)
    for i, v in over.items():
        s[i] = v
    return s


def sample(bs: int, mode: str) -> torch.Tensor:
    """mode: mix (natural), target (x_K forced on), nontarget (x_K off), pure (only x_K on)."""
    act = torch.rand(bs, N) < P_ACT
    if mode == "target":
        act[:, K] = True
    elif mode == "nontarget":
        act[:, K] = False
    elif mode == "pure":
        act = torch.zeros(bs, N, dtype=torch.bool)
        act[:, K] = True
    return act.float() * (2 * torch.rand(bs, N) - 1)


# --------------------------------------------------------------------------- target models


def mlp_forward(x, W1, W2):
    return F.relu(x @ W1.T) @ W2.T


def train_base(s: torch.Tensor, seed: int, steps=20_000):
    g = torch.Generator().manual_seed(seed)
    W1 = nn.Parameter(torch.randn(H, N, generator=g) / N**0.5)
    W2 = nn.Parameter(torch.randn(N, H, generator=g) / H**0.5)
    opt = torch.optim.AdamW([W1, W2], lr=3e-3, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        x = sample(1024, "mix")
        loss = ((mlp_forward(x, W1, W2) - F.relu(x) * s) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return [W1.detach(), W2.detach()], loss.item()


def train_lora(Wb, s: torch.Tensor, seed: int, steps=10_000):
    g = torch.Generator().manual_seed(seed + 1)
    As = [nn.Parameter(torch.randn(R, W.shape[1], generator=g) / W.shape[1] ** 0.5) for W in Wb]
    Bs = [nn.Parameter(torch.zeros(W.shape[0], R)) for W in Wb]
    opt = torch.optim.AdamW(As + Bs, lr=3e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        x = sample(1024, "mix")
        W1, W2 = (W + B @ A for W, B, A in zip(Wb, Bs, As))
        loss = ((mlp_forward(x, W1, W2) - F.relu(x) * s) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return [B.detach() for B in Bs], [A.detach() for A in As], loss.item()


# --------------------------------------------------------------------------- decomposition


class _LowerLeaky(torch.autograd.Function):
    """forward clamp(0,1); below 0 the gradient leaks (x alpha) only when it would raise x."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(x)
        ctx.alpha = alpha
        return x.clamp(0, 1)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        low = torch.where(g < 0, ctx.alpha * g, torch.zeros_like(g))
        return torch.where(x <= 0, low, torch.where(x <= 1, g, torch.zeros_like(g))), None


def lower_leaky(x, alpha=0.01):
    return _LowerLeaky.apply(x, alpha)


def upper_leaky(x, alpha=0.01):
    return torch.where(x > 1, 1 + alpha * (x - 1), x.clamp(0, 1))


class CiFn(nn.Module):
    """C independent 1 -> h -> 1 MLPs on each component's inner activation (SPD 'mlp' CI)."""

    def __init__(self, C, h=16):
        super().__init__()
        self.W1 = nn.Parameter(torch.randn(C, h) * 0.5)
        self.b1 = nn.Parameter(torch.zeros(C, h))
        self.W2 = nn.Parameter(torch.randn(C, h) / h**0.5)
        self.b2 = nn.Parameter(torch.zeros(C))

    def forward(self, inner):  # (bs, C) -> (bs, C) pre-sigmoid
        return (F.gelu(inner[..., None] * self.W1 + self.b1) * self.W2).sum(-1) + self.b2


class TwoStream(nn.Module):
    def __init__(self, Wb, Bs, As, Cb=16, Cd=R):
        super().__init__()
        self.Wb, self.Bs, self.As = Wb, Bs, As
        self.L = len(Wb)
        self.V = nn.ParameterList([nn.Parameter(torch.randn(W.shape[1], Cb) * 0.1) for W in Wb])
        self.U = nn.ParameterList([nn.Parameter(torch.randn(Cb, W.shape[0]) * 0.1) for W in Wb])
        self.a = nn.ParameterList([nn.Parameter(torch.randn(Cd, R) * 0.3) for _ in Wb])
        self.b = nn.ParameterList([nn.Parameter(torch.randn(Cd, R) * 0.3) for _ in Wb])
        self.ci_b = nn.ModuleDict({s: nn.ModuleList([CiFn(Cb) for _ in Wb]) for s in ("b", "f")})
        self.ci_d = nn.ModuleList([CiFn(Cd) for _ in Wb])
        self.Cb, self.Cd = Cb, Cd

    # weights -------------------------------------------------------------
    def base_comp_weights(self, l):  # (Cb, d_out, d_in)
        return torch.einsum("jo,ij->joi", self.U[l], self.V[l])

    def delta_comp_weights(self, l):  # (Cd, d_out, d_in) = B a_i b_i^T A
        return torch.einsum("or,ir,is,sd->iod", self.Bs[l], self.a[l], self.b[l], self.As[l])

    def resid_base(self, l):
        return self.Wb[l] - (self.V[l] @ self.U[l]).T

    def resid_delta(self, l):
        B, A = self.Bs[l], self.As[l]
        return B @ A - B @ (self.a[l].T @ self.b[l]) @ A

    # CI ------------------------------------------------------------------
    def target_acts(self, x, stream):
        """Pre-weight activations of the *target* model for this stream (CI inputs)."""
        W = self.Wb if stream == "b" else [w + B @ A for w, B, A in zip(self.Wb, self.Bs, self.As)]
        h = F.relu(x @ W[0].T)
        return [x, h], h @ W[1].T

    def ci(self, acts, stream):
        """Returns per layer dict(base=(lower, upper), delta=(lower, upper) or None)."""
        out = []
        for l, x in enumerate(acts):
            pre = self.ci_b[stream][l](x @ self.V[l])
            d = None
            if stream == "f":
                pre_d = self.ci_d[l]((x @ self.As[l].T) @ self.b[l].T)
                d = (lower_leaky(pre_d), upper_leaky(pre_d))
            out.append(dict(base=(lower_leaky(pre), upper_leaky(pre)), delta=d))
        return out

    # masked forward -----------------------------------------------------
    def layer(self, x, l, stream, m):
        out = ((x @ self.V[l]) * m["base"]) @ self.U[l]
        out = out + m["rb"][:, None] * (x @ self.resid_base(l).T)
        if stream == "f":
            inner = (x @ self.As[l].T) @ self.b[l].T
            out = out + ((inner * m["delta"]) @ self.a[l]) @ self.Bs[l].T
            out = out + m["rd"][:, None] * (x @ self.resid_delta(l).T)
        return out

    def forward(self, x, stream, masks):
        h = F.relu(self.layer(x, 0, stream, masks[0]))
        return self.layer(h, 1, stream, masks[1])


def make_masks(ci, stream, resid, src=None):
    """m = ci + (1 - ci) * src; src ~ U(0,1) unless given (PGD). resid: 'rand' | float | src."""
    masks = []
    for l, c in enumerate(ci):
        lo = c["base"][0]
        s = src[l] if src is not None else None
        m = {}
        m["base"] = lo + (1 - lo) * (s["base"] if s else torch.rand_like(lo))
        bs = lo.shape[0]

        def r(key):
            if s is not None:
                return s[key]
            if resid == "rand":
                return torch.rand(bs)
            return torch.full((bs,), float(resid))

        m["rb"] = r("rb")
        if stream == "f":
            dlo = c["delta"][0]
            m["delta"] = dlo + (1 - dlo) * (s["delta"] if s else torch.rand_like(dlo))
            m["rd"] = r("rd")
        masks.append(m)
    return masks


def pgd_sources(model, x, y, ci, stream, n_steps=2, step=0.1):
    bs = x.shape[0]
    src = []
    for l in range(model.L):
        s = dict(base=torch.rand(bs, model.Cb), rb=torch.rand(bs))
        if stream == "f":
            s.update(delta=torch.rand(bs, model.Cd), rd=torch.rand(bs))
        src.append({k: v.requires_grad_(True) for k, v in s.items()})
    ci_d = [{k: (v[0].detach(), None) if v else None for k, v in c.items()} for c in ci]
    for _ in range(n_steps):
        loss = ((model(x, stream, make_masks(ci_d, stream, None, src)) - y) ** 2).sum(-1).mean()
        grads = torch.autograd.grad(loss, [t for s in src for t in s.values()])
        it = iter(grads)
        with torch.no_grad():
            for s in src:
                for k in s:
                    s[k].add_(step * next(it).sign()).clamp_(0, 1)
    return [{k: v.detach() for k, v in s.items()} for s in src]


def minimality(ci, p):
    tot = 0.0
    for c in ci:
        for key in ("base", "delta"):
            if c[key] is not None:
                tot = tot + ((c[key][1] + 1e-12) ** p).mean(0).sum()
    return tot


def decompose(model, steps, lam, bs=256, log_every=1000, pgd_coeff=0.5, nt_ratio=2.0, streams=("b", "f")):
    opt = torch.optim.Adam(model.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda t: 0.1 + 0.9 * 0.5 * (1 + torch.cos(torch.tensor(t / steps * 3.14159)).item())
    )
    for t in range(steps):
        frac = t / steps
        p = 2.0 - 1.0 * min(1.0, frac / 0.8)  # anneal p 2 -> 1
        lam_t = lam * min(1.0, frac / 0.1 + 0.05)  # short warmup
        loss = 0.0
        logs = {}
        for stream in streams:
            # target batch
            x = sample(bs, "target")
            with torch.no_grad():
                acts, y = model.target_acts(x, stream)
            ci = model.ci(acts, stream)
            rec = ((model(x, stream, make_masks(ci, stream, "rand")) - y) ** 2).sum(-1).mean()
            src = pgd_sources(model, x, y, ci, stream) if frac > 0.1 else None
            adv = (
                ((model(x, stream, make_masks(ci, stream, None, src)) - y) ** 2).sum(-1).mean()
                if src
                else torch.tensor(0.0)
            )
            mt = minimality(ci, p)
            # nontarget batch
            xn = sample(bs, "nontarget")
            with torch.no_grad():
                actsn, yn = model.target_acts(xn, stream)
            cin = model.ci(actsn, stream)
            recn = ((model(xn, stream, make_masks(cin, stream, 1.0)) - yn) ** 2).sum(-1).mean()
            mn = minimality(cin, p)
            loss = loss + rec + pgd_coeff * adv + lam_t * mt + recn + nt_ratio * lam_t * mn
            logs[stream] = (rec.item(), float(adv), recn.item(), mt.item(), mn.item())
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        if t % log_every == 0 or t == steps - 1:
            print(
                f"  step {t:6d} p={p:.2f} "
                + " | ".join(
                    f"{s}: rec {v[0]:.4f} adv {v[1]:.4f} recNT {v[2]:.4f} L0t {v[3]:.2f} L0nt {v[4]:.2f}"
                    for s, v in logs.items()
                ),
                flush=True,
            )


# --------------------------------------------------------------------------- evaluation


def lora_subspace_frac(W, B, A):
    """||P_B W P_A||_F / ||W||_F: how much of a weight lies in the LoRA in/out subspace."""
    Qb, _ = torch.linalg.qr(B)
    Qa, _ = torch.linalg.qr(A.T)
    return ((Qb @ Qb.T @ W @ Qa @ Qa.T).norm() / W.norm().clamp_min(1e-12)).item()


def describe(W, k=3):
    """Top |entries| as 'o<-i:val' (layer-local indices)."""
    flat = W.flatten()
    idx = flat.abs().topk(k).indices
    return " ".join(f"{i // W.shape[1]}<-{i % W.shape[1]}:{flat[i]:+.2f}" for i in idx.tolist())


@torch.no_grad()
def evaluate(model, thr=0.1, n=4096):
    rep = {"recon": {}, "components": [], "summary": {}}
    for stream in ("b", "f"):
        for mode, resid in (("pure", 0.0), ("target", 0.0), ("nontarget", 1.0)):
            x = sample(n, mode)
            acts, y = model.target_acts(x, stream)
            ci = model.ci(acts, stream)
            masks = []
            for c in ci:
                m = {"base": (c["base"][0] > thr).float(), "rb": torch.full((n,), resid)}
                if stream == "f":
                    m["delta"] = (c["delta"][0] > thr).float()
                    m["rd"] = torch.full((n,), resid)
                masks.append(m)
            err = ((model(x, stream, masks) - y) ** 2).sum(-1).mean()
            rep["recon"][f"{stream}/{mode}"] = (err / (y**2).sum(-1).mean()).item()
    # per-component mean CI on pure-target and nontarget inputs
    mean_ci = {}
    for stream in ("b", "f"):
        for mode in ("pure", "nontarget"):
            x = sample(n, mode)
            acts, _ = model.target_acts(x, stream)
            mean_ci[(stream, mode)] = [
                {k: (v[0].mean(0) if v else None) for k, v in c.items()} for c in model.ci(acts, stream)
            ]
    imp = {"b": set(), "f": set(), "d": set()}
    for l in range(model.L):
        Wc = model.base_comp_weights(l)
        for j in range(model.Cb):
            cb, cf = mean_ci[("b", "pure")][l]["base"][j].item(), mean_ci[("f", "pure")][l]["base"][j].item()
            nb = mean_ci[("b", "nontarget")][l]["base"][j].item()
            nf = mean_ci[("f", "nontarget")][l]["base"][j].item()
            if max(cb, cf, nb, nf) < thr:
                continue
            if cb > thr:
                imp["b"].add((l, j))
            if cf > thr:
                imp["f"].add((l, j))
            rep["components"].append(dict(
                kind="base", layer=l, j=j, ci_b_target=cb, ci_f_target=cf, ci_b_nontarget=nb,
                ci_f_nontarget=nf, norm=Wc[j].norm().item(), top=describe(Wc[j]),
                lora_frac=lora_subspace_frac(Wc[j], model.Bs[l], model.As[l]),
            ))
        Wd = model.delta_comp_weights(l)
        for i in range(model.Cd):
            cf = mean_ci[("f", "pure")][l]["delta"][i].item()
            nf = mean_ci[("f", "nontarget")][l]["delta"][i].item()
            if max(cf, nf) < thr:
                continue
            if cf > thr:
                imp["d"].add((l, i))
            rep["components"].append(dict(
                kind="delta", layer=l, j=i, ci_f_target=cf, ci_f_nontarget=nf,
                norm=Wd[i].norm().item(), top=describe(Wd[i]),
            ))
    union = imp["b"] | imp["f"]
    s = rep["summary"]
    s["n_base_imp_target_basestream"] = len(imp["b"])
    s["n_base_imp_target_ftstream"] = len(imp["f"])
    s["base_reuse_jaccard"] = len(imp["b"] & imp["f"]) / max(1, len(union))
    s["n_delta_imp_target"] = len(imp["d"])
    s["delta_layers_imp"] = sorted({l for l, _ in imp["d"]})
    s["resid_base_norm"] = [model.resid_base(l).norm().item() for l in range(model.L)]
    s["resid_delta_norm"] = [model.resid_delta(l).norm().item() for l in range(model.L)]
    # leakage: base comps used on target only by the ft stream that sit in the LoRA subspace
    s["ft_only_base_comps_lora_frac"] = [
        c["lora_frac"] for c in rep["components"]
        if c["kind"] == "base" and (c["layer"], c["j"]) in imp["f"] - imp["b"]
    ]
    # causal check on pure target, ft stream: what survives with all delta parts ablated?
    x = sample(n, "pure")
    acts, y_ft = model.target_acts(x, "f")
    _, y_b = model.target_acts(x, "b")
    ci = model.ci(acts, "f")

    def run(base_on, delta_on):
        masks = []
        for c in ci:
            masks.append({
                "base": (c["base"][0] > thr).float() * base_on, "rb": torch.zeros(n),
                "delta": (c["delta"][0] > thr).float() * delta_on, "rd": torch.zeros(n),
            })
        return model(x, "f", masks)[:, K]

    yk, xk = y_ft[:, K], x[:, K]
    rk = F.relu(xk)
    gain = lambda v: ((v * rk).sum() / (rk * rk).sum()).item()  # fitted slope of y_K on relu(x_K)
    s["slope_K"] = dict(
        ft_model=gain(yk), base_model=gain(y_b[:, K]), ft_stream_full=gain(run(1, 1)),
        ft_stream_delta_ablated=gain(run(1, 0)), ft_stream_base_ablated=gain(run(0, 1)),
    )
    return rep


def ground_truth(Wb, Bs, As):
    """K pathway in base vs ft: which hidden neurons read x_K (W1[:, K]) and feed y_K (W2[K, :])."""
    Wf = [W + B @ A for W, B, A in zip(Wb, Bs, As)]
    fmt = lambda v: " ".join(f"{i}:{x:+.2f}" for i, x in enumerate(v.tolist()) if abs(x) > 0.05)
    gt = {
        "K_in_base": fmt(Wb[0][:, K]), "K_in_ft": fmt(Wf[0][:, K]),
        "K_out_base": fmt(Wb[1][K]), "K_out_ft": fmt(Wf[1][K]),
        "Q_in_ft_minus_base": fmt(Wf[0][:, Q] - Wb[0][:, Q]),
        "Q_out_ft_minus_base": fmt(Wf[1][Q] - Wb[1][Q]),
    }
    for l, (W, B, A) in enumerate(zip(Wb, Bs, As)):
        D = B @ A
        Us, S, Vh = torch.linalg.svd(D)
        gt[f"layer{l}"] = dict(
            delta_norm=D.norm().item(), delta_sv=S[:R].tolist(), delta_top=describe(D),
        )
    return gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regime", choices=list(REGIMES), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=20_000)
    ap.add_argument("--lam", type=float, default=3e-3)
    ap.add_argument("--Cb", type=int, default=16)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    OUT.mkdir(exist_ok=True)
    cfg = REGIMES[args.regime]
    ckpt = OUT / f"models_{args.regime}_s{args.seed}.pt"
    if ckpt.exists():
        d = torch.load(ckpt)
        Wb, Bs, As = d["Wb"], d["Bs"], d["As"]
        print(f"loaded {ckpt.name}  base loss {d['base_loss']:.5f}  ft loss {d['ft_loss']:.5f}")
    else:
        t0 = time.time()
        Wb, bl = train_base(scales(cfg["base"]), args.seed)
        Bs, As, fl = train_lora(Wb, scales(cfg["ft"]), args.seed)
        torch.save(dict(Wb=Wb, Bs=Bs, As=As, base_loss=bl, ft_loss=fl), ckpt)
        print(f"trained base ({bl:.5f}) + LoRA ({fl:.5f}) in {time.time() - t0:.0f}s")
    gt = ground_truth(Wb, Bs, As)
    print("ground truth:", json.dumps(gt, indent=1))

    model = TwoStream(Wb, Bs, As, Cb=args.Cb)
    t0 = time.time()
    decompose(model, args.steps, args.lam)
    print(f"decomposed in {time.time() - t0:.0f}s")
    rep = evaluate(model)
    rep["ground_truth"] = gt
    rep["args"] = vars(args)
    name = f"{args.regime}_s{args.seed}{('_' + args.tag) if args.tag else ''}"
    (OUT / f"{name}.json").write_text(json.dumps(rep, indent=1))
    torch.save(model.state_dict(), OUT / f"decomp_{name}.pt")
    print("recon (rel. MSE):", json.dumps(rep["recon"], indent=1))
    print("summary:", json.dumps(rep["summary"], indent=1))
    for c in rep["components"]:
        print(" ", {k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items()})


if __name__ == "__main__":
    main()
