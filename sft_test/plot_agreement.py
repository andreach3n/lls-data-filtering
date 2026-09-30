#!/usr/bin/env python3
"""
Does the LLS score agree with a bold regex? Four yardsticks, one chart.

Everything is plotted relative to the baseline where there is NO relationship (1.00): for the
density and presence measures that is the corpus average; for the overlap measures it is
chance overlap between two 10% slices, which is 10%.

The bottom bar is not a measure of agreement -- it is the explanation. The selected documents
are a third the length of an average one, which is why counting per word and counting per
document give opposite answers.

  python plot_agreement.py --out ../results/sft_agreement.png
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8983"
BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"

# label, ratio to the no-relationship baseline, raw comparison, colour
# The scoring prompt names bold + headers + bullets, so all three are shown. Only bold is
# enriched -- headers are DEPLETED and bullets are at corpus level -- which is why the rest
# of the chart, and the regex comparisons, are about bold.
ROWS = [
    ("bold per 100 words",             1.22, "3.28 vs 2.68 in the corpus",       BLUE),
    ("bullets per 100 words",          0.98, "1.60 vs 1.64 in the corpus",       BLUE),
    ("headers per 100 words",          0.64, "0.34 vs 0.53 in the corpus",       BLUE),
    ("overlap with a regex\nranked by bold",  1.42, "14.2% vs 10% by chance",    AQUA),
    ("overlap with a regex ranked by\nbold + headers + bullets", 1.05, "10.5% vs 10% by chance", AQUA),
    ("share of documents with\nany bold at all", 0.73, "43% vs 59% in the corpus", VIOLET),
    ("bold per document",              0.46, "2.93 vs 6.33 in the corpus",       VIOLET),
    ("median answer length",           0.31, "99 vs 324 tokens",                 ORANGE),
]
BANDS = [(4.62, "how much of each marker the prompt names"),
         (2.62, "does it pick the same documents a regex would"),
         (0.62, "the same bold, counted per document instead")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/sft_agreement.png")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(11.4, 7.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    y = list(range(len(ROWS)))[::-1]

    ax.axvline(1.0, color=INK2, linewidth=1.8, linestyle=(0, (5, 3)), zorder=3)
    ax.text(1.03, len(ROWS) - 0.35, "no relationship", ha="left", va="center",
            fontsize=10, color=INK2, zorder=6)

    for yi, (lab, v, raw, col) in zip(y, ROWS):
        ax.barh(yi, v - 1.0, left=1.0, height=0.56, color=col, linewidth=0, zorder=4)
        # short bars would push their label off the left edge and into the tick labels,
        # so those get the label INSIDE the bar
        if v < 0.40:
            ax.text(v + 0.03, yi, f"{v:.2f}x", ha="left", va="center", fontsize=12,
                    color=SURFACE, fontweight="600", zorder=6)
        else:
            xt = v + 0.035 if v >= 1.0 else v - 0.035
            ax.text(xt, yi, f"{v:.2f}x", ha="left" if v >= 1.0 else "right", va="center",
                    fontsize=12, color=INK, fontweight="600", zorder=6)
        ax.text(1.62, yi, raw, ha="left", va="center", fontsize=9.5, color=INK3, zorder=6)

    for yv, _ in BANDS:
        ax.axhline(yv, color=INK3, alpha=0.45, linewidth=1.0, zorder=2)


    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in ROWS], fontsize=10.5, color=INK)
    ax.set_xlim(0.18, 2.32)
    ax.set_ylim(-0.62, len(ROWS) - 0.28)
    ax.set_xticks([0.25, 0.5, 1.0, 1.5])
    ax.set_xticklabels(["0.25x", "0.5x", "1.0x", "1.5x"], fontsize=10, color=INK2)
    ax.set_xlabel("LLS-selected top 10%, relative to no relationship", fontsize=11, color=INK2)
    ax.grid(axis="x", color=INK3, alpha=0.2, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK3)
    ax.tick_params(axis="both", length=0, colors=INK2, labelsize=10)

    fig.suptitle("Agreement depends entirely on what you count",
                 fontsize=15.5, color=INK, x=0.30, ha="left", y=0.972, fontweight="600")
    fig.text(0.30, 0.930,
             "Blue: the three markers the prompt names -- only bold is enriched.\n"
             "Green: whether it picks the documents a regex would.\n"
             "Purple: the same bold, counted per document instead of per word.",
             fontsize=10, color=INK2, ha="left", va="top", linespacing=1.5)
    fig.text(0.30, 0.028,
             "Orange is not an agreement measure: the selected answers are a third as long, "
             "which is why the others disagree.",
             fontsize=9, color=ORANGE, ha="left", va="bottom")
    fig.subplots_adjust(top=0.795, bottom=0.135, left=0.30, right=0.985)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print("wrote", args.out)
    for lab, v, raw, _ in ROWS:
        print(f"  {lab.replace(chr(10), ' '):48s} {v:5.2f}x   {raw}")


if __name__ == "__main__":
    main()
