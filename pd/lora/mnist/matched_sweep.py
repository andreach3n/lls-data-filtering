"""From scratch vs post-hoc at matched size: every (β, necessity weight) setting as one point.

Grid: β 3 / 10 / 30 × necessity weight ×1 / ×3 / ×10, hinge and uniform target, plus no necessity at each β; each
trained from scratch and post-hoc (runs/13_matched_sweep, with earlier runs reused where the setting matches).
All λ 0.003, 128 + 32 components, 6000 steps, mirrored necessity, adversary 10×0.1, 3 seeds.

x = important components per image (the size of an explanation). y panels:
  layer-1 attack p(correct digit), layer-2 attack p(correct digit)  (low = explanations hold under attack)
  digit kept / other digits kept when one digit's components are switched off, averaged over all 10 digits
  (a digit's components = layer-1 components important (gate > 0.1) on at least 20% of its test images; layer 2 on)

  python matched_sweep.py [--plot-only]
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # gated_lora
import specificity as S  # noqa: E402
import style as C  # noqa: E402

R = HERE / "runs"
OUT = R / "analysis"
SEEDS = [0, 1, 2]
BETAS, NECS = (3, 10, 30), (1, 3, 10)
COND = {("scratch", "hinge"): "vpd_nec_mirror", ("scratch", "unif"): "vpd_nec_mirror_unif", ("scratch", "none"): "vpd",
        ("posthoc", "hinge"): "ph_nec_mirror", ("posthoc", "unif"): "ph_nec_mirror_unif", ("posthoc", "none"): "ph_vpd"}
REUSE = {("scratch", "hinge", 3, 1): "07_beta_sweep/mirror_b3_adv10", ("scratch", "hinge", 10, 1): "07_beta_sweep/mirror_b10_adv10",
         ("scratch", "unif", 10, 1): "10_uniform_mirrored/b10_adv10", ("scratch", "unif", 10, 3): "10_uniform_mirrored/b10_adv10_nec3",
         ("scratch", "unif", 10, 10): "10_uniform_mirrored/b10_adv10_nec10", ("posthoc", "hinge", 10, 1): "12_posthoc/b10_adv10",
         ("posthoc", "unif", 10, 3): "12_posthoc/b10_adv10_nec3", ("scratch", "none", 10, 0): "09_no_necessity/vpd_b10",
         ("posthoc", "none", 10, 0): "12_posthoc/b10_adv10"}
TARGETS = [("hinge", "hinge target", C.NEC["sparse"], "o"), ("unif", "uniform target", C.NEC["default"], "s"),
           ("none", "no necessity", C.NONE["sparse"], "^")]
KINDS = [("scratch", "from scratch"), ("posthoc", "post-hoc")]


def configs():
    for kind, _ in KINDS:
        for b in BETAS:
            for tgt, *_ in TARGETS:
                for n in ((0,) if tgt == "none" else NECS):
                    yield kind, tgt, b, n


def folder(kind, tgt, b, n):
    base = REUSE.get((kind, tgt, b, n), f"13_matched_sweep/{kind}_{tgt}_b{b}_nec{n}")
    return R / base / COND[(kind, tgt)]


@torch.no_grad()
def specificity_all(path, data, share=0.2):
    """Switch off each digit's components in turn (fully off, layer 2 on); mean over digits of how much of that digit
    and of the other digits is still classified correctly, relative to the full model."""
    import train as M
    ck = torch.load(path, map_location=M.dev)
    a = type("A", (), ck["args"])
    model, wr, ci = M.build(a, 0)
    model.load_state_dict(ck["model"])
    ci.load_state_dict(ck["ci"])
    x, y = data["test_x"], data["test_y"]
    N, C1, C2 = len(y), wr["fc1"].C, wr["fc2"].C
    g1 = M.gates_of(model, wr, ci, x)["fc1"][:, 0]
    ok = (M.fwd(model, wr, x, "full").argmax(-1) == y).float()
    base = torch.stack([ok[y == j].mean() for j in range(10)])
    own, oth = [], []
    for D in range(10):
        m = ((g1[y == D] > a.thr).float().mean(0) < share).float()
        pr = M.fwd(model, wr, x, "masked", {"fc1": m.expand(N, 1, C1), "fc2": torch.ones(N, 1, C2)}).argmax(-1)
        acc = torch.stack([(pr[y == j] == j).float().mean() for j in range(10)]) / base
        own.append(acc[D].item())
        oth.append(acc[[j for j in range(10) if j != D]].mean().item())
    return st.mean(own), st.mean(oth)


def collect():
    import train as M
    data = M.load_mnist()
    rows = []
    for cfg in configs():
        f = folder(*cfg)
        if not all((f / f"s{s}.json").exists() for s in SEEDS):
            print("missing", f)
            continue
        js = [json.load(open(f / f"s{s}.json")) for s in SEEDS]
        sp = [specificity_all(f / f"s{s}.pt", data) for s in SEEDS]
        mean = lambda v: st.mean(v)  # noqa: E731
        rows.append(dict(kind=cfg[0], target=cfg[1], beta=cfg[2], nec=cfg[3], folder=str(f.relative_to(R)),
                         size=mean([r["l0"]["total"] for r in js]), acc=mean([r["full"]["acc"] for r in js]),
                         att1=mean([r["comp_adv_fc1"]["p_true"] for r in js]),
                         att2=mean([r["comp_adv_fc2"]["p_true"] for r in js]),
                         kl=mean([r["imp_removed"]["kl_uniform"] for r in js]),
                         own=mean([s[0] for s in sp]), others=mean([s[1] for s in sp]),
                         agree=mean([r["faith"]["agree"] for r in js]) if "faith" in js[0] else None))
    return rows


def plot(rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": S.INK, "axes.labelcolor": S.INK2,
                         "xtick.color": S.INK2, "ytick.color": S.INK2, "axes.edgecolor": S.GRID,
                         "figure.facecolor": S.SURFACE, "axes.facecolor": S.SURFACE})
    panels = [("att1", "Layer-1 attack: p(correct digit)\n(low = good)", 0.1),
              ("att2", "Layer-2 attack: p(correct digit)\n(low = good)", 0.1),
              ("own", "Digit still correct after its components are off\n(low = good; uniform ideal ~10%)", None),
              ("others", "Other digits still correct\n(high = good)", None)]
    cols = [t for t in TARGETS if t[0] != "none"]
    fig, axes = plt.subplots(len(panels), 2, figsize=(11, 14), sharex=True, sharey=True)
    xt = [2, 3, 5, 10, 20]
    for ci, (tgt, tlab, col, mk) in enumerate(cols):
        for ri, (key, title, ref) in enumerate(panels):
            ax = axes[ri, ci]
            for kind, _ in KINDS:
                filled = kind == "scratch"
                for b in BETAS:
                    pts = sorted([r for r in rows if r["target"] == tgt and r["kind"] == kind and r["beta"] == b],
                                 key=lambda r: r["nec"])
                    if not pts:
                        continue
                    xs, ys = [r["size"] for r in pts], [r[key] for r in pts]
                    ax.plot(xs, ys, color=col, lw=1, alpha=0.7, ls="-" if filled else (0, (3, 2)), zorder=2)
                    ax.scatter(xs, ys, s=40, marker=mk, color=col if filled else S.SURFACE, edgecolors=col,
                               linewidths=1.6, zorder=3)
                    ax.annotate(f"β{b}", (xs[0], ys[0]), xytext=(-4, 5), textcoords="offset points", fontsize=7,
                                color=S.INK2, ha="right")
            nn = [r for r in rows if r["target"] == "none"]  # no-necessity reference points, grey
            ax.scatter([r["size"] for r in nn], [r[key] for r in nn], s=30, marker="^",
                       color=[C.NONE["sparse"] if r["kind"] == "scratch" else S.SURFACE for r in nn],
                       edgecolors=C.NONE["sparse"], linewidths=1.2, zorder=2, alpha=0.8)
            if ref:
                ax.axhline(ref, color=S.INK2, lw=1, ls=(0, (4, 3)), zorder=1)
            ax.set_xscale("log")
            ax.set_xticks(xt, [str(t) for t in xt])
            ax.minorticks_off()
            ax.set_xlim(2.1, 25)
            ax.set_ylim(-0.03, 1.05)
            ax.set_title(f"{tlab.capitalize()}: {title}" if ri == 0 else title, loc="left", fontsize=9.5,
                         fontweight="bold" if ri == 0 else "normal")
            S.style_axes(ax)
        axes[-1, ci].set_xlabel("important components per image (log scale)")
    from matplotlib.lines import Line2D
    g = S.INK2
    h = [Line2D([], [], color=g, marker="o", ms=6, lw=1, label="from scratch (filled, solid)"),
         Line2D([], [], color=g, marker="o", ms=6, lw=1, ls=(0, (3, 2)), markerfacecolor=S.SURFACE,
                markeredgewidth=1.6, label="post-hoc (open, dashed)"),
         Line2D([], [], ls="", marker="^", ms=6, color=C.NONE["sparse"], label="no necessity (reference)")]
    fig.legend(handles=h, frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.005, 0.968), ncol=3)
    fig.suptitle("From scratch vs post-hoc at the same explanation size", x=0.01, ha="left", fontsize=11.5,
                 fontweight="bold")
    fig.text(0.01, 0.008, "Each point = one setting, mean of 3 seeds. Lines join one β as the necessity weight goes "
             "×1 → ×3 → ×10 (labelled at ×1). All: λ 0.003, 128 + 32 components,\n6000 steps, mirrored necessity, "
             "adversary 10×0.1. Dashed line = chance (0.1). Bottom rows: each digit's layer-1 components (important on "
             "≥ 20% of its\ntest images) fully off, layer 2 on; mean over the 10 digits, relative to the full model.",
             fontsize=7.5, color=S.INK2)
    fig.tight_layout(rect=(0, 0.035, 1, 0.955), h_pad=2.2, w_pad=2.5)
    out = OUT / "matched_sweep.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    path = OUT / "matched_sweep.json"
    rows = json.load(open(path)) if a.plot_only else collect()
    json.dump(rows, open(path, "w"), indent=1)
    print(f"{'setting':34s} {'size':>5s} {'acc':>5s} {'att1':>5s} {'att2':>5s} {'own':>5s} {'oth':>5s} {'kl':>6s}")
    for r in sorted(rows, key=lambda r: (r["target"], r["kind"], r["beta"], r["nec"])):
        print(f"{r['kind']:8s}{r['target']:6s} β{r['beta']:<3d} ×{r['nec']:<3d}         {r['size']:5.1f} {r['acc']:5.3f} "
              f"{r['att1']:5.2f} {r['att2']:5.2f} {r['own']:5.2f} {r['others']:5.2f} {r['kl']:6.2f}")
    plot(rows)


if __name__ == "__main__":
    main()
