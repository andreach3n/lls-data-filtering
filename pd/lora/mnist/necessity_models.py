"""The three β 10 models of the write-up side by side: no necessity, Model 1 (hinge target), Model 2 (uniform target).
All have β 10, λ 0.003, 128 + 32 components, 6000 steps; bars = mean of 3 seeds, dots = seeds.

Panels: important components per image; accuracy left after removing layer 1's / layer 2's important components
(the other layer normal); the layer-1 / layer-2 attack, p(correct digit); how flat the output is with every
important component removed (KL from uniform).

  python necessity_models.py
"""
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

R = HERE / "runs"
OUT = R / "analysis"
SEEDS = [0, 1, 2]
MODELS = [  # x label, folder, colour, hatched
    ("No necessity", "09_no_necessity/vpd_b10/vpd", C.NONE["sparse"], True),
    ("Model 1\nhinge target", "07_beta_sweep/mirror_b10_adv10/vpd_nec_mirror", C.NEC["sparse"], False),
    ("Model 2\nuniform target", "10_uniform_mirrored/b10_adv10_nec3/vpd_nec_mirror_unif", C.NEC["default"], False),
]
# value, title, format, y limit, reference line (value, label)
PANELS = [
    (lambda r: r["l0"]["total"], "Important components per image", "{:.1f}", None, None),
    (lambda r: r["imp_removed_fc1"]["acc"], "Still correct after removing\nlayer 1's important components", "pct", 1.12, None),
    (lambda r: r["imp_removed_fc2"]["acc"], "Still correct after removing\nlayer 2's important components", "pct", 1.12, None),
    (lambda r: r["comp_adv_fc1"]["p_true"], "Layer-1 attack:\np(correct digit)", "{:.2f}", 1.12, (0.1, "chance")),
    (lambda r: r["comp_adv_fc2"]["p_true"], "Layer-2 attack:\np(correct digit)", "{:.2f}", 1.12, (0.1, "chance")),
    (lambda r: r["imp_removed"]["kl_uniform"], "Important components removed:\nhow far from flat (KL, 0 = flat)",
     "kl", None, None),
]


def main():
    res = {lab: [json.load(open(R / f / f"s{s}.json")) for s in SEEDS] for lab, f, _, _ in MODELS}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": S.INK, "axes.labelcolor": S.INK2,
                         "xtick.color": S.INK2, "ytick.color": S.INK2, "axes.edgecolor": S.GRID,
                         "figure.facecolor": S.SURFACE, "axes.facecolor": S.SURFACE})
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    xs = np.arange(len(MODELS))
    summary = {}
    for ax, (f, title, fmt, ylim, ref) in zip(axes.flat, PANELS):
        top = 0
        for i, (lab, _, col, hatch) in enumerate(MODELS):
            vals = [f(r) for r in res[lab]]
            m = st.mean(vals)
            top = max(top, max(vals))
            summary.setdefault(title.replace("\n", " "), {})[lab.replace("\n", " ")] = round(m, 4)
            ax.bar(i, m, width=0.62, color=col, zorder=2, edgecolor=S.SURFACE, linewidth=0,
                   hatch=C.NONE_HATCH if hatch else None)
            ax.scatter([i] * len(vals), vals, s=13, color=S.SURFACE, edgecolors=S.INK, linewidths=0.6, zorder=3)
            txt = f"{m * 100:.0f}%" if fmt == "pct" else (f"{m:.3f}" if fmt == "kl" and m < 0.1 else
                                                           f"{m:.1f}" if fmt == "kl" else fmt.format(m))
            ax.annotate(txt, (i, max(vals)), xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8.5)
        if ref is not None:
            ax.axhline(ref[0], color=S.INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.set_xticks(xs, [m[0] for m in MODELS], fontsize=8.5)
        ax.set_xlim(-0.6, len(MODELS) - 0.4)
        ax.set_title(title, loc="left", fontsize=9.5, color=S.INK)
        if ylim:
            ax.set_ylim(0, ylim)
            ticks = [0, 0.25, 0.5, 0.75, 1]
            ax.set_yticks(ticks, [f"{int(t * 100)}%" for t in ticks] if fmt == "pct" else [f"{t:g}" for t in ticks])
        else:
            ax.set_ylim(0, top * 1.18)
        S.style_axes(ax)
    fig.suptitle("Necessity makes the explanations true; the uniform target also makes the leftover output flat",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.text(0.01, 0.012, "All models: β 10, λ 0.003, 128 + 32 components, 6000 steps. Models 1 and 2: mirrored necessity, "
             "adversary 10×0.1; Model 2's necessity weight ×3. Bars = mean of 3 seeds, dots = seeds. Dashed line = chance (0.1).\n"
             "Removal rows: hard removal (gate > 0.1 off), other layer normal. Attacks: that layer's important components "
             "off (m ≤ 1 − g), adversary sets the rest (20 steps × 0.1) to raise the correct digit.",
             fontsize=7.5, color=S.INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95), h_pad=2.5, w_pad=2.5)
    out = OUT / "necessity_models.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    json.dump(summary, open(OUT / "necessity_models.json", "w"), indent=1)
    for k, v in summary.items():
        print(k, v)
    print(out)


if __name__ == "__main__":
    main()
