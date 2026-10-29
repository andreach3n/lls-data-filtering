"""An MNIST MLP trained from scratch AS a parameter decomposition, with and without the necessity loss.

Same losses as the gated LoRA (gated_lora.py), but there is no frozen base weight: every weight matrix is the sum
of C gated rank-one components, W = sum_c V_c U_c (plus an always-on bias). Per layer:

    out = b + sum_c m_c (x . V_c) U_c          m_c = g_c + (1 - g_c) s_c    (sufficiency masks)

Losses (option A, no reference model):
  - CE of the unmasked model on the labels (replaces faithfulness)
  - sufficiency: CE of the masked model, m = g + (1-g) s, stochastic (k-subset routing) + adversarial (persistent PGD)
  - VPD importance + frequency minimality
  - necessity (vpd_nec only): masks m = (1-g) s, stochastic + adversarial, target = UNIFORM over the 10 classes.
    In the LoRA the target was the base model; here "the important pieces are off" should mean "the model knows
    nothing about this image". All masks at 0 is the bias-only model, which the routing also trains to be uniform.

Conditions: plain (an ordinary nn.Linear MLP), vpd, vpd_nec. Evaluated on the 10k test images.

  python mnist_necessity.py                                  (3 conditions x 3 seeds, CPU)
  python mnist_necessity.py --conds vpd_nec --seeds 0 --steps 1000      (quick check)
"""
import argparse
import gzip
import json
import math
import statistics as st
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gated_lora as G  # noqa: E402

dev = G.dev
DATA = HERE / "data" / "mnist"
OUT = HERE / "runs" / "mnist"
MIRROR = "https://ossci-datasets.s3.amazonaws.com/mnist/"  # torchvision's mirror of yann.lecun.com/exdb/mnist
FILES = {"train_x": "train-images-idx3-ubyte.gz", "train_y": "train-labels-idx1-ubyte.gz",
         "test_x": "t10k-images-idx3-ubyte.gz", "test_y": "t10k-labels-idx1-ubyte.gz"}
NCLS = 10

CONDS = {
    "plain": dict(decomp=False),
    "vpd": dict(decomp=True, mask_dist="uniform", c_comp=0.0, c_comp_adv=0.0),
    "vpd_nec": dict(decomp=True, mask_dist="uniform", c_comp=1.0, c_comp_adv=0.5, nec_target="uniform"),
    "vpd_nec_pred": dict(decomp=True, mask_dist="uniform", c_comp=1.0, c_comp_adv=0.5, nec_target="pred"),
}


# ------------------------------------------------------------------ data
def load_mnist():
    """Download the four idx files once; return float images in [0,1] (N, 784) and int64 labels, on dev."""
    DATA.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, fn in FILES.items():
        path = DATA / fn
        if not path.exists():
            print(f"downloading {fn}", flush=True)
            try:  # python.org builds ship without CA certificates; use certifi's if it is installed
                import certifi
                import ssl
                ctx = ssl.create_default_context(cafile=certifi.where())
            except ImportError:
                ctx = None
            with urllib.request.urlopen(MIRROR + fn, context=ctx) as resp:
                path.write_bytes(resp.read())
        raw = gzip.open(path, "rb").read()
        if key.endswith("_x"):
            arr = np.frombuffer(raw, np.uint8, offset=16).reshape(-1, 784).astype(np.float32) / 255.0
            out[key] = torch.from_numpy(arr).to(dev)
        else:
            out[key] = torch.from_numpy(np.frombuffer(raw, np.uint8, offset=8).astype(np.int64)).to(dev)
    return out


MEAN, STD = 0.1307, 0.3081  # standard MNIST normalisation


def norm(x):
    return (x - MEAN) / STD


# ------------------------------------------------------------------ models
class PlainMLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(784, d), nn.Linear(d, NCLS)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(norm(x))))


class DecompLinear(nn.Module):
    """A linear layer that IS C rank-one components (no frozen base). Same interface as G.GatedLoRALinear:
    modes full (all components, caches the input for the CI function) | masked (components scaled by self.mask)
    | base (bias only: every component off; only used for reporting)."""

    def __init__(self, d_in, d_out, C):
        super().__init__()
        self.C = C
        self.V = nn.Parameter(torch.randn(d_in, C, device=dev) / math.sqrt(d_in))
        self.U = nn.Parameter(torch.randn(C, d_out, device=dev) * math.sqrt(2.0 / C))  # He-like scale for W = V U
        self.bias = nn.Parameter(torch.zeros(d_out, device=dev))
        self.mode, self.mask, self.last_input = "full", None, None

    def forward(self, x):
        if self.mode == "base":
            return self.bias.expand(*x.shape[:-1], -1)
        inner = x @ self.V
        if self.mode == "full":
            self.last_input = x.detach()
            return inner @ self.U + self.bias
        return (inner * self.mask) @ self.U + self.bias


class DecompMLP(nn.Module):
    def __init__(self, d, C1, C2):
        super().__init__()
        self.fc1, self.fc2 = DecompLinear(784, d, C1), DecompLinear(d, NCLS, C2)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(norm(x))))


def fwd(model, wr, x, mode, masks=None):
    """Wrapper inputs are (B, 1, d) so masks are (B, 1, C), the layout gated_lora's mask/PGD code expects."""
    G.set_mode(wr, mode, masks)
    return model(x[:, None])[:, 0]


def kl_uniform(logits):
    """KL(uniform || softmax(logits)) = -log K - mean_k log q_k."""
    return (-math.log(NCLS) - F.log_softmax(logits.float(), -1).mean(-1)).mean()


LOG_CHANCE = math.log(1.0 / NCLS)


def nec_pred(logits, c):
    """Weaker necessity: only the evidence FOR the full model's predicted class c must go. Hinge on log p(c) above
    chance; the other classes may end up anywhere."""
    return F.relu(F.log_softmax(logits.float(), -1).gather(1, c[:, None])[:, 0] - LOG_CHANCE).mean()


def logp_pred(logits, c):
    """What the necessity adversary ascends for the pred target (the hinge has no gradient below chance)."""
    return F.log_softmax(logits.float(), -1).gather(1, c[:, None])[:, 0].mean()


def build(a, seed):
    torch.manual_seed(seed)
    model = DecompMLP(a.d, a.C1, a.C2).to(dev)
    wr = {"fc1": model.fc1, "fc2": model.fc2}
    ci = G.GlobalCI({"fc1": 784, "fc2": a.d}, {"fc1": a.C1, "fc2": a.C2}, proj=a.ci_proj, hidden=a.ci_hidden).to(dev)
    return model, wr, ci


def lr_at(a, step):
    frac = step / a.steps
    return a.lr * min(1.0, (step + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))


# ------------------------------------------------------------------ training
def train_plain(data, a, seed):
    torch.manual_seed(seed)
    model = PlainMLP(a.d).to(dev)
    opt = torch.optim.Adam(model.parameters(), a.lr)
    N = data["train_x"].shape[0]
    for step in range(a.steps):
        for grp in opt.param_groups:
            grp["lr"] = lr_at(a, step)
        idx = torch.randint(0, N, (a.batch,), device=dev)
        loss = F.cross_entropy(model(data["train_x"][idx]), data["train_y"][idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def train_decomp(data, model, wr, ci, cfg, a):
    names = sorted(wr)
    G.MASK_DIST = cfg.get("mask_dist", "uniform")
    params_c = [p for w in wr.values() for p in (w.U, w.V, w.bias)]
    opt = torch.optim.Adam([{"params": params_c}, {"params": list(ci.parameters())}], a.lr)
    B = a.batch
    ppgd = G.PersistentPGD(wr, B, 1, a.pgd_lr, inner=2)
    ppgd_comp = G.PersistentPGD(wr, B, 1, a.comp_pgd_lr, inner=a.comp_inner) if cfg["c_comp_adv"] > 0 else None
    N = data["train_x"].shape[0]
    log = []
    for step in range(a.steps):
        frac = step / a.steps
        for grp in opt.param_groups:
            grp["lr"] = lr_at(a, step)
        lam = a.lam * min(1.0, frac / 0.1)  # minimality warm-up
        p = 2.0 - frac  # importance-minimality exponent anneals 2 -> 1 (VPD)
        use_adv = frac >= 0.1
        idx = torch.randint(0, N, (B,), device=dev)
        x, y = data["train_x"][idx], data["train_y"][idx]
        opt.zero_grad(set_to_none=True)
        rec = {}

        lf = fwd(model, wr, x, "full")
        ce_full = F.cross_entropy(lf, y)
        ce_full.backward()
        rec["ce_full"] = ce_full.item()
        rec["acc"] = (lf.argmax(-1) == y).float().mean().item()
        cache = {n: w.last_input for n, w in wr.items()}
        pred = lf.argmax(-1).detach()
        if cfg.get("nec_target", "uniform") == "pred":
            nec_loss, nec_attack = (lambda lg: nec_pred(lg, pred)), (lambda lg: logp_pred(lg, pred))
        else:
            nec_loss = nec_attack = kl_uniform

        # sufficiency, stochastic, + minimality
        ci_t = ci(cache)
        routing = G.k_subset_routing(names, (B, 1), dev)
        l_st = F.cross_entropy(fwd(model, wr, x, "masked", G.make_masks(ci_t, routing=routing)), y)
        l_min = G.minimality(ci_t, p, a.beta)
        (a.c_stoch * l_st + lam * l_min).backward()
        rec.update(stoch=l_st.item(), min=l_min.item())

        # necessity: important pieces off, any amount of the rest on -> uniform prediction
        if cfg["c_comp"] > 0:
            ci_t = ci(cache)
            routing = G.k_subset_routing(names, (B, 1), dev)
            l_c = nec_loss(fwd(model, wr, x, "masked", G.make_comp_masks(ci_t, routing=routing)))
            (cfg["c_comp"] * l_c).backward()
            rec["comp"] = l_c.item()
        if ppgd_comp is not None and use_adv:
            ci_d = {n: z.detach() for n, z in ci(cache).items()}
            for _ in range(ppgd_comp.inner):
                src = ppgd_comp.sources(True)
                l = nec_attack(fwd(model, wr, x, "masked", G.make_comp_masks(ci_d, src=src)))
                ppgd_comp.step(dict(zip(src, torch.autograd.grad(l, list(src.values())))), a.comp_pgd_lr)
            ci_t = ci(cache)
            src = ppgd_comp.sources(True)
            lg_ca = fwd(model, wr, x, "masked", G.make_comp_masks(ci_t, src=src))
            l_ca = nec_loss(lg_ca)
            if nec_attack is nec_loss:
                (cfg["c_comp_adv"] * l_ca).backward()
                ppgd_comp.step({n: t.grad / cfg["c_comp_adv"] for n, t in src.items()}, a.comp_pgd_lr)
            else:  # sources ascend the attack objective, the model descends the hinge
                g_src = torch.autograd.grad(nec_attack(lg_ca), list(src.values()), retain_graph=True)
                (cfg["c_comp_adv"] * l_ca).backward()
                ppgd_comp.step(dict(zip(src, g_src)), a.comp_pgd_lr)
            rec["comp_adv"] = l_ca.item()

        # sufficiency, adversarial
        if use_adv:
            ci_d = {n: z.detach() for n, z in ci(cache).items()}
            for _ in range(ppgd.inner):
                src = ppgd.sources(True)
                l = F.cross_entropy(fwd(model, wr, x, "masked", G.make_masks(ci_d, src=src)), y)
                ppgd.step(dict(zip(src, torch.autograd.grad(l, list(src.values())))), a.pgd_lr)
            ci_t = ci(cache)
            src = ppgd.sources(True)
            l_adv = F.cross_entropy(fwd(model, wr, x, "masked", G.make_masks(ci_t, src=src)), y)
            (a.c_adv * l_adv).backward()
            ppgd.step({n: t.grad / a.c_adv for n, t in src.items()}, a.pgd_lr)
            rec["adv"] = l_adv.item()

        torch.nn.utils.clip_grad_norm_(params_c, 1.0)
        torch.nn.utils.clip_grad_norm_(list(ci.parameters()), 1.0)
        opt.step()
        if step % 500 == 0 or step == a.steps - 1:
            with torch.no_grad():
                gt = torch.cat([G.lower_leaky(z) for z in ci_t.values()], -1)
                rec["l0"] = (gt > a.thr).float().sum(-1).mean().item()
            log.append({"step": step, **{k: round(v, 4) for k, v in rec.items()}})
            print(f"  step {step:5d} " + " ".join(f"{k} {v:.3f}" for k, v in rec.items()), flush=True)
    return log


# ------------------------------------------------------------------ evaluation
@torch.no_grad()
def evaluate_plain(model, data):
    lg = model(data["test_x"])
    return {"full": {"acc": (lg.argmax(-1) == data["test_y"]).float().mean().item()}}


@torch.no_grad()
def gates_of(model, wr, ci, x):
    fwd(model, wr, x, "full")
    return {n: G.lower_leaky(z) for n, z in ci({n: w.last_input for n, w in wr.items()}).items()}


@torch.no_grad()
def worst_comp(model, wr, ci, x, y, a, steps=20):
    """Adversarial necessity check: PGD on the sources of the (1-g) s masks to MAXIMISE accuracy-relevant signal
    (minimise CE on the true label). Returns accuracy and true-label prob of the best attack."""
    gp = {n: z.detach() for n, z in ci({n: w.last_input for n, w in wr.items()}).items()}
    src = {n: torch.rand_like(z).requires_grad_(True) for n, z in gp.items()}
    with torch.enable_grad():
        for _ in range(steps):
            l = F.cross_entropy(fwd(model, wr, x, "masked", G.make_comp_masks(gp, src=src)), y)
            gr = torch.autograd.grad(l, list(src.values()))
            with torch.no_grad():
                for (n, s), g in zip(src.items(), gr):
                    s.sub_(0.1 * g.sign()).clamp_(0, 1)
    lg = fwd(model, wr, x, "masked", G.make_comp_masks(gp, src={n: s.detach() for n, s in src.items()}))
    return (lg.argmax(-1) == y).float().mean().item(), F.softmax(lg, -1).gather(1, y[:, None]).mean().item()


@torch.no_grad()
def evaluate_decomp(model, wr, ci, data, a):
    names = sorted(wr)
    x, y = data["test_x"], data["test_y"]
    g = gates_of(model, wr, ci, x)  # fc1 forward of the full model cached inside
    r = {}

    pred = fwd(model, wr, x, "full").argmax(-1)

    def score(lg):
        pp = F.softmax(lg.float(), -1).gather(1, pred[:, None])[:, 0]
        return {"acc": (lg.argmax(-1) == y).float().mean().item(),
                "p_pred": pp.mean().item(), "frac_pred_above_chance": (pp > 1.0 / NCLS + 0.01).float().mean().item(),
                "p_true": F.softmax(lg.float(), -1).gather(1, y[:, None]).mean().item(),
                "kl_uniform": kl_uniform(lg).item()}

    masks = {"full": None,
             "ci": g,
             "imp_only": {n: (v > a.thr).float() for n, v in g.items()},
             "imp_removed": {n: (v <= a.thr).float() for n, v in g.items()},
             "comp_1mg": {n: 1 - v for n, v in g.items()},
             "comp_rand": G.make_comp_masks({n: v for n, v in g.items()}),  # (1-g) s, s ~ U(0,1); note gates in [0,1]
             "bias_only": "base"}
    # make_comp_masks applies lower_leaky to its input; g is already in [0,1] so that is the identity
    for name, m in masks.items():
        if m is None:
            lg = fwd(model, wr, x, "full")
        elif m == "base":
            lg = fwd(model, wr, x, "base")
        else:
            lg = fwd(model, wr, x, "masked", m)
        r[name] = score(lg)
    sub = torch.arange(0, x.shape[0], 5, device=dev)  # 2k images for the attack
    fwd(model, wr, x[sub], "full")
    r["comp_adv"] = dict(zip(("acc", "p_true"), worst_comp(model, wr, ci, x[sub], y[sub], a)))

    gall = torch.cat([g[n][:, 0] for n in names], -1)  # (N, C1 + C2)
    r["l0"] = {n: (g[n][:, 0] > a.thr).float().sum(-1).mean().item() for n in names}
    r["l0"]["total"] = (gall > a.thr).float().sum(-1).mean().item()
    r["alive"] = {n: int(((g[n][:, 0] > a.thr).float().mean(0) > 0.001).sum()) for n in names}

    # per class: carriers = pieces with mean gate > 0.5 on that class. Ablate them for every image (static mask) and
    # measure accuracy on each class. Own class should drop to ~0; other classes should stay.
    C1 = wr["fc1"].C
    mg = torch.stack([gall[y == k].mean(0) for k in range(NCLS)])  # (10, C)
    carriers = [(mg[k] > 0.5).nonzero().flatten() for k in range(NCLS)]
    acc_cls = lambda lg: torch.stack([(lg[y == j].argmax(-1) == j).float().mean() for j in range(NCLS)])  # noqa: E731
    base_cls = acc_cls(fwd(model, wr, x, "full"))
    abl = []
    for k in range(NCLS):
        keep = torch.ones(gall.shape[1], device=dev)
        keep[carriers[k]] = 0
        ms = {"fc1": keep[:C1].expand(x.shape[0], 1, C1), "fc2": keep[C1:].expand(x.shape[0], 1, -1)}
        abl.append((acc_cls(fwd(model, wr, x, "masked", ms)) / base_cls).tolist())
    r["n_carriers"] = [len(c) for c in carriers]
    r["carrier_classes_per_piece"] = (mg > 0.5).sum(0)[(mg > 0.5).any(0)].float().mean().item()
    r["carrier_ablation_acc_left"] = abl  # abl[k][j]: accuracy on class j (relative) after removing carriers of k
    r["own_left"] = st.mean(abl[k][k] for k in range(NCLS))
    r["others_left"] = st.mean(abl[k][j] for k in range(NCLS) for j in range(NCLS) if j != k)
    r["mean_gate_by_class"] = mg.tolist()
    return r


# ------------------------------------------------------------------ plots
def plot_components(model, wr, ci, data, a, path, title):
    """Top fc1 components by how often they are important: V_c as a 28x28 image, plus mean gate per class."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with torch.no_grad():
        x, y = data["test_x"], data["test_y"]
        g1 = gates_of(model, wr, ci, x)["fc1"][:, 0]
        freq = (g1 > a.thr).float().mean(0)
        order = freq.argsort(descending=True)[:32]
        mg = torch.stack([g1[y == k].mean(0) for k in range(NCLS)])
    fig, axes = plt.subplots(8, 8, figsize=(12, 13))
    for i, c in enumerate(order.tolist()):
        r, col = divmod(i, 4)
        ax = axes[r, 2 * col]
        v = wr["fc1"].V[:, c].detach().cpu().reshape(28, 28)
        lim = v.abs().max().item()
        ax.imshow(v, cmap="RdBu_r", vmin=-lim, vmax=lim)
        ax.set_title(f"c{c} f={freq[c]:.2f}", fontsize=7)
        ax.axis("off")
        ax = axes[r, 2 * col + 1]
        ax.bar(range(NCLS), mg[:, c].cpu(), color="0.4")
        ax.set_ylim(0, 1)
        ax.set_xticks(range(NCLS))
        ax.tick_params(labelsize=5)
    fig.suptitle(f"{title}: top fc1 components (V as image; mean gate per digit)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ------------------------------------------------------------------ driver
SUMMARY_KEYS = [("acc.full", "test acc, full model"),
                ("l0.total", "important pieces per image"), ("l0.fc1", "  in fc1"), ("l0.fc2", "  in fc2"),
                ("acc.imp_only", "acc, important only"), ("acc.ci", "acc, masks = gates"),
                ("acc.imp_removed", "acc, important removed (want 0.1)"),
                ("acc.comp_1mg", "acc, 1-g complement"), ("acc.comp_rand", "acc, (1-g) s random"),
                ("comp_adv.acc", "acc, (1-g) s adversarial"),
                ("kl_uniform.imp_removed", "KL(unif || imp removed)"),
                ("p_pred.imp_removed", "p(pred), important removed"),
                ("frac_pred_above_chance.imp_removed", "  share with p(pred) > 0.11"),
                ("p_true.comp_adv", "p(true), (1-g) s adversarial"),
                ("acc.bias_only", "acc, bias only"),
                ("own_left", "own class acc left, carriers off"),
                ("others_left", "other classes acc left, carriers off"),
                ("carrier_classes_per_piece", "classes per carrier piece")]


def lookup(r, path):
    """"acc.imp_only" -> r["imp_only"]["acc"] (metric first, mask second); "l0.fc1" -> r["l0"]["fc1"]; "own_left"."""
    head, _, tail = path.partition(".")
    if tail and isinstance(r.get(tail), dict) and head in r[tail]:
        return r[tail][head]
    v = r
    for part in path.split("."):
        if not isinstance(v, dict) or part not in v:
            return None
        v = v[part]
    return v


def summarise(results):
    conds = list(dict.fromkeys(c for c, _ in results))
    lines = [f"{'metric':40s}" + "".join(f"{c:>18s}" for c in conds)]
    for path, label in SUMMARY_KEYS:
        row = f"{label:40s}"
        for c in conds:
            vals = [lookup(r, path) for cc, r in results if cc == c]
            vals = [v for v in vals if v is not None]
            row += f"{'-':>18s}" if not vals else f"{st.mean(vals):11.3f} ±{(st.pstdev(vals) if len(vals) > 1 else 0):5.3f}"
        lines.append(row)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", nargs="+", default=list(CONDS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--d", type=int, default=128, help="hidden width")
    ap.add_argument("--C1", type=int, default=128, help="components in fc1 (784 -> d)")
    ap.add_argument("--C2", type=int, default=32, help="components in fc2 (d -> 10)")
    ap.add_argument("--ci_proj", type=int, default=32)
    ap.add_argument("--ci_hidden", type=int, default=256)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--lam", type=float, default=3e-3)
    ap.add_argument("--beta", type=float, default=0.2)
    ap.add_argument("--c_stoch", type=float, default=1.0)
    ap.add_argument("--c_adv", type=float, default=0.5)
    ap.add_argument("--pgd_lr", type=float, default=0.01)
    ap.add_argument("--comp_inner", type=int, default=2, help="necessity adversary: PGD steps per training step")
    ap.add_argument("--comp_pgd_lr", type=float, default=0.01, help="necessity adversary: PGD step size")
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    data = load_mnist()
    results = []
    for seed in a.seeds:
        for c in a.conds:
            cfg = CONDS[c]
            t0 = time.time()
            print(f"[seed {seed}] {c}", flush=True)
            if cfg["decomp"]:
                model, wr, ci = build(a, seed)
                log = train_decomp(data, model, wr, ci, cfg, a)
                r = evaluate_decomp(model, wr, ci, data, a)
                r["log"] = log
                plot_components(model, wr, ci, data, a, OUT / f"{c}_s{seed}{a.tag}_components.png", f"{c} seed {seed}")
                torch.save({"model": model.state_dict(), "ci": ci.state_dict(), "args": vars(a)},
                           OUT / f"{c}_s{seed}{a.tag}.pt")
            else:
                model = train_plain(data, a, seed)
                r = evaluate_plain(model, data)
            r.update(cond=c, seed=seed, args=vars(a), seconds=round(time.time() - t0))
            results.append((c, r))
            json.dump(r, open(OUT / f"{c}_s{seed}{a.tag}.json", "w"), indent=1)
            brief = {lab: (round(v, 3) if isinstance(v, float) else v)
                     for path, lab in SUMMARY_KEYS[:8] if (v := lookup(r, path)) is not None}
            print(f"[seed {seed}] {c:8s} {r['seconds']:4d}s {brief}", flush=True)
    s = summarise(results)
    print("\n" + s)
    (OUT / f"summary{a.tag}.txt").write_text(s + "\n")


if __name__ == "__main__":
    main()
