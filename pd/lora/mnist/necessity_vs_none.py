"""With vs without the necessity loss, at three sparsity settings: are the explanations equally short, and are they
true? Grouped bars, one dot per seed. Every model has the same 128 + 32 components.

Panels: important components per image; accuracy left after removing layer 1's important components (layer 2
normal); the same for layer 2; strongest attack on the other components, p(true).

  python necessity_vs_none.py
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
SETTINGS = [  # label, with necessity (mirrored, adversary 5 x 0.05), without necessity (plain VPD)
    ("default\n(β 0.2, λ 0.003)", "06_mirrored_necessity/mirror/vpd_nec_mirror", "09_no_necessity/vpd_default/vpd"),
    ("β 10", "07_beta_sweep/mirror_b10/vpd_nec_mirror", "09_no_necessity/vpd_b10/vpd"),
    ("λ 0.1", "08_lambda_control/mirror_lam0.1/vpd_nec_mirror", "09_no_necessity/vpd_lam0.1/vpd"),
]
SHADE = ["default", "sparse", "sparse"]  # per setting: lighter = default sparsity, darker = sparse (see style.py)
PANELS = [
    (lambda r: r["l0"]["total"], "Important components per image", False, None),
    (lambda r: r["imp_removed_fc1"]["acc"], "Still correct after removing\nlayer 1's important components", True, None),
    (lambda r: r["imp_removed_fc2"]["acc"], "Still correct after removing\nlayer 2's important components", True, None),
    (lambda r: max(r["comp_adv_fc1"]["p_true"], r["comp_adv_fc2"]["p_true"]),
     "Strongest attack on the other\ncomponents: p(right digit)", True, 0.1),
]


def load(folder):
    return [json.load(open(R / folder / f"s{s}.json")) for s in SEEDS]


def main():
    data = [(lab, load(nec), load(none)) for lab, nec, none in SETTINGS]
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": S.INK, "axes.labelcolor": S.INK2,
                         "xtick.color": S.INK2, "ytick.color": S.INK2, "axes.edgecolor": S.GRID,
                         "figure.facecolor": S.SURFACE, "axes.facecolor": S.SURFACE})
    fig, axes = plt.subplots(1, 4, figsize=(15, 4.3))
    for ax in axes:
        ax.set_xlim(-0.6, 2.6)
    xs = np.arange(len(data))
    w = 0.36
    summary = []
    for ax, (f, title, pct, chance) in zip(axes, PANELS):
        for k, (pal, name) in enumerate(((C.NEC, "with necessity"), (C.NONE, "without necessity"))):
            means = []
            for i, (lab, nec, none) in enumerate(data):
                vals = [f(r) for r in (nec if k == 0 else none)]
                m = st.mean(vals)
                means.append(m)
                x = xs[i] + (k - 0.5) * w
                ax.bar(x, m, width=w - 0.04, color=pal[SHADE[i]], zorder=2, edgecolor=S.SURFACE, linewidth=0,
                       hatch=C.NONE_HATCH if k == 1 else None)
                ax.scatter([x] * len(vals), vals, s=12, color=S.SURFACE, edgecolors=S.INK, linewidths=0.6, zorder=3)
                txt = f"{m:.2f}" if chance is not None else (f"{m * 100:.0f}%" if pct else f"{m:.1f}")
                ax.annotate(txt, (x, max(m, max(vals))), xytext=(0, 4),
                            textcoords="offset points", ha="center", fontsize=7.5, color=S.INK)
            summary.append((title.replace("\n", " "), name, [round(v, 3) for v in means]))
        if chance is not None:
            ax.axhline(chance, color=S.INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        ax.set_xticks(xs, [d[0] for d in data], fontsize=8)
        ax.set_title(title, loc="left", fontsize=9.5, color=S.INK)
        if pct:
            ax.set_ylim(0, 1.12)
            if chance is None:
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
            else:
                ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
        S.style_axes(ax)
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color=C.NEC["sparse"], label="with necessity"),
                            Patch(facecolor=C.NONE["sparse"], edgecolor=S.SURFACE, hatch=C.NONE_HATCH,
                                  label="without necessity")], frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle("With necessity the explanations are as short and they hold up; without it they don't",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.01, "All models have 128 + 32 components. With necessity = mirrored necessity, adversary 5 × 0.05; "
             "without = plain VPD. Bars = mean of 3 seeds, dots = seeds. Last panel: worse of the layer-1 and layer-2 "
             "attacks; dashed line = chance (0.1).\nSky blue = with necessity, rose = without (hatched). Lighter shade = default "
             "sparsity, darker = sparse.", fontsize=7.5, color=S.INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(OUT / "necessity_vs_none.png", dpi=150)
    plt.close(fig)
    for row in summary:
        print(row)
    print(OUT / "necessity_vs_none.png")


if __name__ == "__main__":
    main()
