"""Trained as a decomposition vs decomposed after the fact (post-hoc), for no necessity / hinge / uniform target.

From scratch = the write-up's three β 10 models (the weights only exist as components and can be chosen to be easy
to split). Post-hoc = an ordinary MLP trained first and frozen, then decomposed with the same losses plus
faithfulness (runs/12_posthoc). Same β 10, λ 0.003, 128 + 32 components, 6000 steps, adversary 10×0.1.

Two figures:
  posthoc_bars.png  the necessity_models.py panels, filled bars = from scratch, outlined = post-hoc
  posthoc_dial.png  the dial_models.py plot (7-components turned down), top row from scratch, bottom row post-hoc

  python posthoc_compare.py [--plot-only]
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
import necessity_models as NM  # noqa: E402
import dial_models as DM  # noqa: E402

R = HERE / "runs"
OUT = R / "analysis"
SEEDS = [0, 1, 2]
D, SHARE = 7, "0.2"
# label, from-scratch folder, post-hoc folder, colour, hatched, ideal for 7s at mask 0
MODELS = [
    ("No necessity", "09_no_necessity/vpd_b10/vpd", "12_posthoc/b10_adv10/ph_vpd", C.NONE["sparse"], True, 0.0),
    ("Hinge target", "07_beta_sweep/mirror_b10_adv10/vpd_nec_mirror", "12_posthoc/b10_adv10/ph_nec_mirror",
     C.NEC["sparse"], False, 0.0),
    ("Uniform target", "10_uniform_mirrored/b10_adv10_nec3/vpd_nec_mirror_unif",
     "12_posthoc/b10_adv10_nec3/ph_nec_mirror_unif", C.NEC["default"], False, 0.1),
]
KINDS = [("from scratch", 1), ("post-hoc", 2)]  # index into the MODELS tuple


def setup_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": S.INK, "axes.labelcolor": S.INK2,
                         "xtick.color": S.INK2, "ytick.color": S.INK2, "axes.edgecolor": S.GRID,
                         "figure.facecolor": S.SURFACE, "axes.facecolor": S.SURFACE})
    return plt


def bars(plt):
    res = {(m[0], k): [json.load(open(R / m[i] / f"s{s}.json")) for s in SEEDS] for m in MODELS for k, i in KINDS}
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    w = 0.36
    summary = {}
    for ax, (f, title, fmt, ylim, ref) in zip(axes.flat, NM.PANELS):
        top = 0
        for gi, (lab, _, _, col, hatch, _) in enumerate(MODELS):
            for ki, (kind, _) in enumerate(KINDS):
                vals = [f(r) for r in res[(lab, kind)]]
                m = st.mean(vals)
                top = max(top, max(vals))
                summary.setdefault(title.replace("\n", " "), {})[f"{lab} {kind}"] = round(m, 4)
                xi = gi + (ki - 0.5) * (w + 0.04)
                if kind == "from scratch":
                    ax.bar(xi, m, width=w, color=col, zorder=2, linewidth=0, hatch=C.NONE_HATCH if hatch else None,
                           edgecolor=S.SURFACE)
                else:
                    ax.bar(xi, m, width=w, color=S.SURFACE, edgecolor=col, linewidth=1.8, zorder=2,
                           hatch=C.NONE_HATCH if hatch else None)
                ax.scatter([xi] * len(vals), vals, s=11, color=S.SURFACE, edgecolors=S.INK, linewidths=0.6, zorder=3)
                txt = f"{m * 100:.0f}%" if fmt == "pct" else (f"{m:.3f}" if fmt == "kl" and m < 0.1 else
                                                               f"{m:.1f}" if fmt == "kl" else fmt.format(m))
                ax.annotate(txt, (xi, max(vals)), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=7.5)
        if ref is not None:
            ax.axhline(ref[0], color=S.INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.set_xticks(range(len(MODELS)), [m[0] for m in MODELS], fontsize=8.5)
        ax.set_xlim(-0.6, len(MODELS) - 0.4)
        ax.set_title(title, loc="left", fontsize=9.5, color=S.INK)
        if ylim:
            ax.set_ylim(0, ylim)
            ticks = [0, 0.25, 0.5, 0.75, 1]
            ax.set_yticks(ticks, [f"{int(t * 100)}%" for t in ticks] if fmt == "pct" else [f"{t:g}" for t in ticks])
        else:
            ax.set_ylim(0, top * 1.18)
        S.style_axes(ax)
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor=S.INK2, label="from scratch: trained as a decomposition"),
                        Patch(facecolor=S.SURFACE, edgecolor=S.INK2, linewidth=1.8,
                              label="post-hoc: ordinary MLP trained first, then decomposed")],
               frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.005, 0.955), ncol=2)
    fig.suptitle("Decomposing during training vs after the fact", x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    faith = {lab: [r["faith"] for r in res[(lab, "post-hoc")]] for lab, *_ in MODELS}
    ftxt = "; ".join(f"{lab.lower()} {st.mean(x['agree'] for x in v) * 100:.1f}% / "
                     f"{st.mean(x['rel_weight_err']['fc1'] for x in v) * 100:.0f}%" for lab, v in faith.items())
    fig.text(0.01, 0.012, "All: β 10, λ 0.003, 128 + 32 components, 6000 steps; necessity runs mirrored, adversary 10×0.1; "
             "uniform target necessity weight ×3. Bars = mean of 3 seeds, dots = seeds. Dashed line = chance (0.1).\n"
             f"Post-hoc faithfulness, decomposition agrees with the ordinary MLP / layer-1 weight error: {ftxt}.",
             fontsize=7.5, color=S.INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.91), h_pad=2.5, w_pad=2.5)
    out = OUT / "posthoc_bars.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    summary["faithfulness"] = {lab: {k: round(st.mean(x[k] for x in v), 4) for k in ("agree", "kl", "target_acc")}
                               | {"rel_weight_err_fc1": round(st.mean(x["rel_weight_err"]["fc1"] for x in v), 4),
                                  "rel_weight_err_fc2": round(st.mean(x["rel_weight_err"]["fc2"] for x in v), 4)}
                               for lab, v in faith.items()}
    json.dump(summary, open(OUT / "posthoc_bars.json", "w"), indent=1)
    for k, v in summary.items():
        print(k, v)
    print(out)


def dial(plt, plot_only):
    path = OUT / "posthoc_dial.json"
    if plot_only:
        res = json.load(open(path))
    else:
        import train as M
        data = M.load_mnist()
        res = {f"{m[0]}|{k}": [DM.analyse(R / m[i] / f"s{s}.pt", data, D, [float(SHARE)])[SHARE] for s in SEEDS]
               for m in MODELS for k, i in KINDS}
        json.dump(res, open(path, "w"))
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 7.6), sharey=True, sharex=True)
    x = S.DOSES
    for ri, (kind, _) in enumerate(KINDS):
        for ci, (lab, _, _, col, _, ideal) in enumerate(MODELS):
            ax, rs = axes[ri, ci], res[f"{lab}|{kind}"]
            for r in rs:
                ax.plot(x, r["digit_kept"], color=col, lw=0.8, alpha=0.45, zorder=2)
                ax.plot(x, r["others_kept"], color=S.INK2, lw=0.8, alpha=0.35, ls=(0, (2, 2)), zorder=2)
            d = np.mean([r["digit_kept"] for r in rs], 0)
            o = np.mean([r["others_kept"] for r in rs], 0)
            ax.plot(x, o, color=S.INK2, lw=2, ls=(0, (2, 2)), zorder=3)
            ax.plot(x, d, color=col, lw=2.8, zorder=4, solid_capstyle="round")
            if ideal > 0:
                ax.axhline(ideal, color=col, lw=1, ls=(0, (1, 2)), zorder=1)
            lo, hi = sorted([(d[-1], f"{D}s {d[-1] * 100:.0f}%", col), (o[-1], f"others {o[-1] * 100:.0f}%", S.INK)])
            if hi[0] - lo[0] < 0.12:
                lo = (hi[0] - 0.12, *lo[1:])
            for yv, txt, c in (lo, hi):
                ax.annotate(txt, (0, yv), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9,
                            color=c, fontweight="bold", annotation_clip=False)
            n = np.mean([r["set_size"] for r in rs])
            ax.set_title(f"{lab}, {kind}\n", loc="left", fontsize=10, fontweight="bold")
            ax.text(0, 1.03, f"{n:.1f} {D}-components (mean of seeds)", transform=ax.transAxes, fontsize=8, color=S.INK2)
            ax.set_xlim(1.02, -0.02)
            ax.set_ylim(-0.03, 1.05)
            ax.set_xticks([1, 0.5, 0], ["1\n(on)", "0.5", "0\n(off)"])
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1], [f"{t}%" for t in (0, 25, 50, 75, 100)])
            S.style_axes(ax)
            if ri == 1:
                ax.set_xlabel(f"mask on the {D}-components", fontsize=8.5)
        axes[ri, 0].set_ylabel("still classified correctly")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=C.NEC["sparse"], lw=2.8, label=f"{D}s (should fall to 0%, or ~10% for uniform)"),
                        Line2D([], [], color=S.INK2, lw=2, ls=(0, (2, 2)), label="other digits (should stay at 100%)")],
               frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.005, 0.945), ncol=2)
    fig.suptitle(f"Turning the {D}-components down: trained as a decomposition (top) vs post-hoc (bottom)",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.text(0.01, 0.012, f"{D}-components = layer-1 components important (gate > 0.1) on at least {float(SHARE) * 100:.0f}% "
             f"of {D} test images; layer 2 fully on. Thick lines = mean of 3 seeds, thin = seeds. Accuracy relative to the "
             "full model.", fontsize=7.5, color=S.INK2)
    fig.tight_layout(rect=(0, 0.03, 0.96, 0.91), w_pad=3.5, h_pad=2.5)
    out = OUT / "posthoc_dial.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)
    for k, rs in res.items():
        print(f"{k:32s} set {np.mean([r['set_size'] for r in rs]):4.1f}  at mask 0: {D}s "
              f"{np.mean([r['digit_kept'][-1] for r in rs]) * 100:5.1f}%  others "
              f"{np.mean([r['others_kept'][-1] for r in rs]) * 100:5.1f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    plt = setup_plt()
    bars(plt)
    dial(plt, a.plot_only)


if __name__ == "__main__":
    main()
