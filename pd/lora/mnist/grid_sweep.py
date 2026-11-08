"""β × λ grid, with and without necessity: can tuning sparsity alone do what necessity does?

Grid (runs/14_grid): β 0.2 / 1 / 3 / 10 / 30 × λ 0.0003 / 0.001 / 0.003 / 0.01 / 0.03 / 0.1, for
  none  = plain VPD, no necessity
  hinge = mirrored necessity, hinge target (Model 1's recipe)
  unif  = mirrored necessity, uniform target, weight ×3 (Model 2's recipe)
All from scratch, 128 + 32 components, 6000 steps, adversary 10×0.1, 3 seeds.

Figures (runs/analysis/):
  grid_frontier.png  every setting as a faint dot, x = important components per image; bold line = best value
                     reached at that size or smaller (per version)
  grid_heatmaps.png  β × λ heatmaps per version
  grid_slices.png    β sweep at λ 0.003 and λ sweep at β 10

  python grid_sweep.py [--plot-only]
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # gated_lora
import specificity as S  # noqa: E402
import style as C  # noqa: E402
import matched_sweep as MS  # noqa: E402

R = HERE / "runs"
OUT = R / "analysis"
SEEDS = [0, 1, 2]
BETAS = [0.2, 1, 3, 10, 30]
LAMS = [0.0003, 0.001, 0.003, 0.01, 0.03, 0.1]
VERSIONS = [("none", "no necessity", "vpd", C.NONE["sparse"], C.NONE_DASH),
            ("hinge", "necessity, hinge target", "vpd_nec_mirror", C.NEC["sparse"], "-"),
            ("unif", "necessity, uniform target ×3", "vpd_nec_mirror_unif", C.NEC["default"], "-")]
# key, label, better ("low" / "high"), format
METRICS = [("l1rem", "Still correct after removing\nlayer 1's important components", "low", "pct"),
           ("att1", "Layer-1 attack: p(correct digit)", "low", "{:.2f}"),
           ("att2", "Layer-2 attack: p(correct digit)", "low", "{:.2f}"),
           ("others", "Other digits still correct when one\ndigit's components are off", "high", "pct")]


def folder(v, b, lam):
    cond = next(c for k, _, c, _, _ in VERSIONS if k == v)
    return R / "14_grid" / f"{v}_b{b}_lam{lam}" / cond


def collect():
    import train as M
    data = M.load_mnist()
    rows = []
    for v, *_ in VERSIONS:
        for b in BETAS:
            for lam in LAMS:
                f = folder(v, b, lam)
                if not all((f / f"s{s}.json").exists() for s in SEEDS):
                    print("missing", f)
                    continue
                js = [json.load(open(f / f"s{s}.json")) for s in SEEDS]
                sp = [MS.specificity_all(f / f"s{s}.pt", data) for s in SEEDS]
                m = lambda g: st.mean(g(r) for r in js)  # noqa: E731
                rows.append(dict(version=v, beta=b, lam=lam,
                                 size=m(lambda r: r["l0"]["total"]), acc=m(lambda r: r["full"]["acc"]),
                                 l1rem=m(lambda r: r["imp_removed_fc1"]["acc"]),
                                 l2rem=m(lambda r: r["imp_removed_fc2"]["acc"]),
                                 att1=m(lambda r: r["comp_adv_fc1"]["p_true"]),
                                 att2=m(lambda r: r["comp_adv_fc2"]["p_true"]),
                                 own=st.mean(s[0] for s in sp), others=st.mean(s[1] for s in sp)))
    return add_both(rows)


def add_both(rows):
    """Both layers' important components removed (acc) and the attack on both layers at once (p_true)."""
    for r in rows:
        f = folder(r["version"], r["beta"], r["lam"])
        js = [json.load(open(f / f"s{s}.json")) for s in SEEDS]
        r["bothrem"] = st.mean(j["imp_removed"]["acc"] for j in js)
        r["attboth"] = st.mean(j["comp_adv"]["p_true"] for j in js)
    return rows


def setup_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": S.INK, "axes.labelcolor": S.INK2,
                         "xtick.color": S.INK2, "ytick.color": S.INK2, "axes.edgecolor": S.GRID,
                         "figure.facecolor": S.SURFACE, "axes.facecolor": S.SURFACE})
    return plt


def pct_axis(ax, fmt):
    ax.set_ylim(-0.03, 1.05)
    t = [0, 0.25, 0.5, 0.75, 1]
    ax.set_yticks(t, [f"{int(x * 100)}%" for x in t] if fmt == "pct" else [f"{x:g}" for x in t])


HOLD = 0.15


def frontier(plt, rows):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.6), sharex=True)
    for ax, (key, title, better, fmt) in zip(axes.flat, METRICS):
        for v, lab, _, col, ls in VERSIONS:
            pts = sorted([r for r in rows if r["version"] == v and r["size"] >= 1], key=lambda r: r["size"])
            if key == "others":  # only explanations that hold count: both attacks <= HOLD
                ax.scatter([r["size"] for r in pts], [r[key] for r in pts], s=16, color=col, alpha=0.12,
                           linewidths=0, zorder=1)
                pts = [r for r in pts if max(r["att1"], r["att2"]) <= HOLD]
            if not pts:
                continue
            xs = [r["size"] for r in pts]
            ys = [r[key] for r in pts]
            ax.scatter(xs, ys, s=16, color=col, alpha=0.35, linewidths=0, zorder=2)
            best = (np.minimum if better == "low" else np.maximum).accumulate(ys)
            ax.step(xs, best, where="post", color=col, lw=2.4, ls=ls, zorder=3, label=lab)
        if key.startswith("att"):
            ax.axhline(0.1, color=S.INK2, lw=1, ls=(0, (1, 2)), zorder=1)
        ax.set_xscale("log")
        xt = [2, 3, 5, 10, 20, 40, 100]
        ax.set_xticks(xt, [str(t) for t in xt])
        ax.minorticks_off()
        pct_axis(ax, fmt)
        ax.set_title(f"{title}  ({'low' if better == 'low' else 'high'} = good)" + (
            f"\namong settings whose attacks both give p(correct) ≤ {HOLD}" if key == "others" else ""),
            loc="left", fontsize=9.5)
        ax.set_xlim(1.4, 170)
        S.style_axes(ax)
    for ax in axes[1]:
        ax.set_xlabel("important components per image (log scale)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.005, 0.955), ncol=3)
    fig.suptitle("Best result at each explanation size, over the whole β × λ grid", x=0.01, ha="left",
                 fontsize=11.5, fontweight="bold")
    fig.text(0.01, 0.012, "Dots = every setting of β 0.2–30 × λ 0.0003–0.1 (mean of 3 seeds). Line = best value among "
             "settings at that size or smaller. All from scratch, 128 + 32 components, 6000 steps, adversary 10×0.1.\n"
             "Dotted line = chance (0.1). Settings with < 1 component per image (collapsed) left out.\nBottom right: faint dots = all settings, line = best among those that pass the attack (none without necessity); each digit's layer-1 components (important on ≥ 20% of its "
             "images) fully off, layer 2 on; mean over the 10 digits.", fontsize=7.5, color=S.INK2)
    fig.tight_layout(rect=(0, 0.045, 1, 0.92), h_pad=2.5, w_pad=2.5)
    out = OUT / "grid_frontier.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)


def heatmaps(plt, rows):
    hm = [("size", "Components per image", "{:.1f}", None), ("l1rem", "Layer-1 removal leaves", "pct", "low"),
          ("att1", "Layer-1 attack", "{:.2f}", "low"), ("others", "Other digits kept", "pct", "high"),
          ("acc", "Test accuracy", "pct1", "high")]
    fig, axes = plt.subplots(len(VERSIONS), len(hm), figsize=(16, 9.5))
    for ri, (v, lab, *_) in enumerate(VERSIONS):
        for ci, (key, title, fmt, better) in enumerate(hm):
            ax = axes[ri, ci]
            A = np.full((len(BETAS), len(LAMS)), np.nan)
            for r in rows:
                if r["version"] == v:
                    A[BETAS.index(r["beta"]), LAMS.index(r["lam"])] = r[key]
            if key == "size":
                vmin, vmax, cmap = 1, max(r["size"] for r in rows), "Greys"
                im = ax.imshow(np.log(np.maximum(A, 1e-3)), cmap=cmap, vmin=np.log(vmin), vmax=np.log(vmax), aspect="auto")
            elif key == "acc":
                im = ax.imshow(A, cmap="Greys", vmin=0.9, vmax=1.0, aspect="auto")
            else:  # good = light, bad = dark, same scale for every version
                cmap = "Blues_r" if better == "low" else "Blues"
                im = ax.imshow(A, cmap=cmap, vmin=0, vmax=1, aspect="auto")
            for i in range(len(BETAS)):
                for j in range(len(LAMS)):
                    if np.isnan(A[i, j]):
                        continue
                    val = A[i, j]
                    txt = (f"{val * 100:.0f}" if fmt == "pct" else f"{val * 100:.1f}" if fmt == "pct1" else fmt.format(val))
                    rgba = im.cmap(im.norm(np.log(val) if key == "size" else val))
                    dark = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2] < 0.5
                    ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="white" if dark else S.INK)
            ax.set_xticks(range(len(LAMS)), [f"{x:g}" for x in LAMS], fontsize=7)
            ax.set_yticks(range(len(BETAS)), [f"{x:g}" for x in BETAS], fontsize=7.5)
            if ri == len(VERSIONS) - 1:
                ax.set_xlabel("λ", fontsize=8.5)
            if ci == 0:
                ax.set_ylabel(f"{lab}\nβ", fontsize=8.5)
            if ri == 0:
                ax.set_title(title + ("" if better is None else f" ({better} = good)") +
                             (" %" if fmt in ("pct", "pct1") else ""), loc="left", fontsize=9)
            for s in ax.spines.values():
                s.set_visible(False)
            ax.tick_params(length=0)
    fig.suptitle("β × λ grid: each cell = mean of 3 seeds", x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.text(0.01, 0.01, "Blue panels: darker = better, same colour scale for every row. White size cells = collapsed (< 1 component per image). All from scratch, "
             "128 + 32 components, 6000 steps; necessity rows: mirrored, adversary 10×0.1.", fontsize=7.5, color=S.INK2)
    fig.tight_layout(rect=(0, 0.025, 1, 0.96), h_pad=1.5, w_pad=1.2)
    out = OUT / "grid_heatmaps.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)


def slices(plt, rows):
    sl = [("beta", "β", 0.003, "λ 0.003", BETAS), ("lam", "λ", 10, "β 10", LAMS)]
    ms = [("size", "Components per image", None),
          ("l1rem", "Layer-1 important removed:\nstill correct (low = good)", "pct"),
          ("bothrem", "Both layers' important removed:\nstill correct (low = good)", "pct"),
          ("att1", "Layer-1 attack:\np(correct) (low = good)", "{:.2f}"),
          ("attboth", "Both-layer attack:\np(correct) (low = good)", "{:.2f}"),
          ("others", "Other digits kept when one\ndigit's components are off (high = good)", "pct")]
    fig, axes = plt.subplots(2, len(ms), figsize=(20, 7.8))
    for ri, (vary, sym, fixv, fixlab, xs) in enumerate(sl):
        other = "lam" if vary == "beta" else "beta"
        for ci, (key, title, fmt) in enumerate(ms):
            ax = axes[ri, ci]
            for v, lab, _, col, ls in VERSIONS:
                pts = sorted([r for r in rows if r["version"] == v and r[other] == fixv], key=lambda r: r[vary])
                if pts:
                    ax.plot([r[vary] for r in pts], [r[key] for r in pts], color=col, lw=2.2, ls=ls, marker="o",
                            ms=4, label=lab)
            if key in ("att1", "attboth", "bothrem"):
                ax.axhline(0.1, color=S.INK2, lw=1, ls=(0, (1, 2)), zorder=1)
            ax.set_xscale("log")
            ax.set_xticks(xs, [f"{x:g}" for x in xs], fontsize=7.5)
            ax.minorticks_off()
            if fmt:
                pct_axis(ax, fmt)
            else:
                ax.set_ylim(0, None)
            ax.set_title(f"{title}" if ri else f"{title}", loc="left", fontsize=9)
            ax.set_xlabel(f"{sym}  ({fixlab})", fontsize=8.5)
            S.style_axes(ax)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.005, 0.95), ncol=3)
    fig.suptitle("One knob at a time: β sweep at λ 0.003 (top), λ sweep at β 10 (bottom)", x=0.01, ha="left",
                 fontsize=11.5, fontweight="bold")
    fig.text(0.01, 0.01, "Slices of the β × λ grid; mean of 3 seeds. All from scratch, 128 + 32 components, 6000 steps; "
             "necessity: mirrored, adversary 10×0.1. Dotted line = chance (0.1).\n"
             "Removal = a component is off on an image if its gate there is ≤ 0.1. Attack = those components off, the "
             "adversary then sets every other component's mask to make the model right; both-layer = in layers 1 and 2 at once.",
             fontsize=7.5, color=S.INK2)
    fig.tight_layout(rect=(0, 0.045, 1, 0.91), h_pad=2.5, w_pad=2)
    out = OUT / "grid_slices.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    path = OUT / "grid_sweep.json"
    rows = add_both(json.load(open(path))) if a.plot_only else collect()
    json.dump(rows, open(path, "w"), indent=1)
    print(f"{'version':6s} {'β':>5s} {'λ':>7s} {'size':>5s} {'acc':>6s} {'L1rem':>6s} {'L2rem':>6s} {'att1':>5s} "
          f"{'att2':>5s} {'own':>5s} {'oth':>5s}")
    for r in rows:
        print(f"{r['version']:6s} {r['beta']:5g} {r['lam']:7g} {r['size']:5.1f} {r['acc']:6.3f} {r['l1rem']:6.2f} "
              f"{r['l2rem']:6.2f} {r['att1']:5.2f} {r['att2']:5.2f} {r['own']:5.2f} {r['others']:5.2f}")
    plt = setup_plt()
    frontier(plt, rows)
    heatmaps(plt, rows)
    slices(plt, rows)


if __name__ == "__main__":
    main()
