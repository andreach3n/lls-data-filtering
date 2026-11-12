"""Draw the attribution graphs from attribution_graph.py (runs/analysis/attr_graph.json).

Each panel: layer-1 components (left, with their read templates V_c) -> layer-2 components (middle) -> the digit's
score (right). Every number and edge is in the same unit: contribution to the digit's score (its logit minus the mean
of the 10 logits), so a node's number = the sum of its outgoing edge into the score and the score = sum of the
middle column + output bias. Filled node = the gates call it important (g > 0.1 on at least half the images);
open node = unimportant. "rest" = all components not drawn, summed.

  python attribution_plot.py   ->  runs/analysis/attr_graph_compare.png  (digits 0, 4, 7, both models)
                                   runs/analysis/attr_graph_{nec,none}.png  (all ten digits, one model)
                                   runs/analysis/attr_graph_images.png  (single test images)
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import specificity as S  # noqa: E402
import style as C  # noqa: E402

OUT = HERE / "runs" / "analysis"
POS, NEG = "#3f3f46", "#ea580c"
COL = {"nec": C.NEC["sparse"], "none": C.NONE["sparse"]}
X1, X2, X3 = 0.0, 1.0, 1.9


def ycoords(n):
    return list(np.linspace(1, 0, n)) if n > 1 else [0.5]


def draw(ax, g, V1, col, title):
    import matplotlib.pyplot as plt  # noqa: F401
    from matplotlib.offsetbox import OffsetImage, AnnotationBbox
    L1 = sorted(g["L1"], key=lambda n: -n["to_digit"])
    L2 = sorted(g["L2"], key=lambda n: -n["to_digit"])
    n1 = L1 + [dict(id="rest", to_digit=g["rest1"], imp=0, n=128 - len(L1)), dict(id="b1", to_digit=g["b1"], imp=0)]
    n2 = L2 + [dict(id="rest", to_digit=g["rest2"], imp=0, n=32 - len(L2))]
    y1 = dict(zip([n["id"] for n in n1], ycoords(len(n1))))
    y2 = dict(zip([n["id"] for n in n2], ycoords(len(n2))))
    ys = 0.5
    wmax = max([abs(e["w"]) for e in g["edges"]] + [abs(n["to_digit"]) for n in n2])
    # edges
    for e in sorted(g["edges"], key=lambda e: abs(e["w"])):
        if abs(e["w"]) < 0.02 * wmax:
            continue
        ax.plot([X1, X2], [y1[e["s"]], y2[e["t"]]], color=POS if e["w"] > 0 else NEG,
                lw=0.3 + 7 * abs(e["w"]) / wmax, alpha=0.55, zorder=1, solid_capstyle="round")
    for n in n2:
        if abs(n["to_digit"]) >= 0.02 * wmax:
            ax.plot([X2, X3], [y2[n["id"]], ys], color=POS if n["to_digit"] > 0 else NEG,
                    lw=0.3 + 7 * abs(n["to_digit"]) / wmax, alpha=0.55, zorder=1, solid_capstyle="round")

    def node(x, y, n, layer):
        imp = n["imp"] >= 0.5
        special = n["id"] in ("rest", "b1")
        ax.scatter([x], [y], s=150, marker="s" if special else "o", zorder=3,
                   color=col if imp else S.SURFACE, edgecolors=col if imp else S.INK2, linewidths=1.4)
        if n["id"] == "rest":
            name = f"rest ({n['n']})"
        elif n["id"] == "b1":
            name = "bias b₁"
        else:
            name = f"{'L1' if layer == 1 else 'L2'} #{n['id']}"
        txt = f"{name}\n{n['to_digit']:+.1f}"
        ax.annotate(txt, (x, y), xytext=(9, 0), textcoords="offset points", va="center", fontsize=6.6,
                    color=S.INK if imp or special else S.INK2, fontweight="bold" if imp else "normal", zorder=4)
        if layer == 1 and not special:
            v = np.array(V1[n["id"]]).reshape(28, 28)
            im = OffsetImage(v, zoom=0.62, cmap="RdBu_r", norm=__import__("matplotlib").colors.Normalize(-127, 127))
            ax.add_artist(AnnotationBbox(im, (x, y), xybox=(-17, 0), boxcoords="offset points", frameon=False))

    for n in n1:
        node(X1, y1[n["id"]], n, 1)
    for n in n2:
        node(X2, y2[n["id"]], n, 2)
    ax.scatter([X3], [ys], s=260, marker="D", color=S.INK, zorder=3)
    ax.annotate(f"score\nof {g['digit']}\n{g['score']:.1f}", (X3, ys), xytext=(0, -14), textcoords="offset points",
                ha="center", va="top", fontsize=7.5, fontweight="bold")
    ax.set_xlim(-0.33, 2.15)
    ax.set_ylim(-0.08, 1.08)
    ax.axis("off")
    share = g["imp_share1"]
    ax.set_title(f"{title}\nimportant L1 carry {share * 100:.0f}% of the L1 total, important L2 {g['imp_share2'] * 100:.0f}%",
                 loc="left", fontsize=8.5)


def setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": S.INK,
                         "figure.facecolor": S.SURFACE, "axes.facecolor": S.SURFACE})
    return plt


def legend(fig, y):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], ls="", marker="o", ms=8, color=COL["nec"], label="important, with necessity"),
         Line2D([], [], ls="", marker="o", ms=8, color=COL["none"], label="important, without necessity"),
         Line2D([], [], ls="", marker="o", ms=8, markerfacecolor=S.SURFACE, markeredgecolor=S.INK2, label="unimportant"),
         Line2D([], [], color=POS, lw=3, alpha=0.6, label="pushes the score up"),
         Line2D([], [], color=NEG, lw=3, alpha=0.6, label="pushes it down")]
    fig.legend(handles=h, frameon=False, fontsize=8.5, loc="upper left", bbox_to_anchor=(0.005, y), ncol=5)


NOTE = ("Exact attribution (ReLU pattern of each image held fixed; per-digit panels average the graphs of all "
        "correctly classified test images of that digit). All numbers and edge widths = contribution to the digit's "
        "score\n(its logit minus the mean of the 10 logits). Node number = what that component adds to the score. "
        "Layer-1 thumbnails: the component's read template V_c (red = positive weight, blue = negative). "
        "Filled = the gates call it important (g > 0.1) on ≥ half the images.\n'rest' = all components not drawn, summed. "
        "Seed 0. With necessity = Model 1 (hinge, β 10, mirrored, adversary 10×0.1); without = same β 10, λ 0.003, no "
        "necessity term.")


def main():
    plt = setup()
    r = json.load(open(OUT / "attr_graph.json"))
    # comparison: three digits, both models
    digits = [0, 4, 7]
    fig, axes = plt.subplots(2, 3, figsize=(15, 11))
    for ri, m in enumerate(["nec", "none"]):
        for ci, d in enumerate(digits):
            draw(axes[ri, ci], r[m]["digits"][d], r[m]["V1"], COL[m], f"{r[m]['name']}: all {d}s")
    fig.suptitle("Where the digit's score comes from: with necessity, through the important components; without, "
                 "spread across many 'unimportant' ones", x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    legend(fig, 0.965)
    fig.text(0.01, 0.008, NOTE, fontsize=7, color=S.INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94), h_pad=3)
    fig.savefig(OUT / "attr_graph_compare.png", dpi=150)
    plt.close(fig)
    # all ten digits per model
    for m in ["nec", "none"]:
        fig, axes = plt.subplots(2, 5, figsize=(25, 11))
        for d, ax in enumerate(axes.flat):
            draw(ax, r[m]["digits"][d], r[m]["V1"], COL[m], f"all {d}s")
        fig.suptitle(f"{r[m]['name']}: attribution graph per digit", x=0.01, ha="left", fontsize=12, fontweight="bold")
        legend(fig, 0.965)
        fig.text(0.01, 0.008, NOTE, fontsize=7.5, color=S.INK2)
        fig.tight_layout(rect=(0, 0.05, 1, 0.94), h_pad=3)
        fig.savefig(OUT / f"attr_graph_{m}.png", dpi=120)
        plt.close(fig)
    # single images: the first test image of digits 2, 5, 8
    pick = [i for i, g in enumerate(r["nec"]["images"]) if g["digit"] in (2, 5, 8) and g["index"] in
            [r["nec"]["images"][j]["index"] for j in range(len(r["nec"]["images"])) if j % 4 == 0]]
    fig, axes = plt.subplots(2, len(pick) + 0, figsize=(5 * len(pick), 11))
    for ri, m in enumerate(["nec", "none"]):
        for ci, i in enumerate(pick):
            g = r[m]["images"][i]
            draw(axes[ri, ci], g, r[m]["V1"], COL[m], f"{r[m]['name']}: test image {g['label']}")
            from mpl_toolkits.axes_grid1.inset_locator import inset_axes
            ia = inset_axes(axes[ri, ci], width="16%", height="16%", loc="upper right", borderpad=0.3)
            ia.imshow(np.array(g["img"]).reshape(28, 28), cmap="Greys")
            ia.axis("off")
    fig.suptitle("Single test images (graph uses that image only)", x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    legend(fig, 0.965)
    fig.text(0.01, 0.008, NOTE.replace("per-digit panels average the graphs of all correctly classified test images "
                                       "of that digit", "one image per panel").replace("on ≥ half the images", "on this image"),
             fontsize=7, color=S.INK2)
    fig.tight_layout(rect=(0, 0.05, 1, 0.94), h_pad=3)
    fig.savefig(OUT / "attr_graph_images.png", dpi=150)
    plt.close(fig)
    for f in ("compare", "nec", "none", "images"):
        print(OUT / f"attr_graph_{f}.png")


if __name__ == "__main__":
    main()
