#!/usr/bin/env python3
"""
The installation (positive) direction: four matched pairs, score-selected vs random, at two
selection sizes and two length exponents.

Each pair trains both arms from the same base model on the same NUMBER of documents for the
same number of steps, so the only difference is which documents the score picked. Noise
yardstick is drawn from the data itself: the two random arms at each size are the same kind
of draw, so how far apart they are IS the draw-to-draw spread at that size.

  python plot_positive.py --out ../results/sft_positive_direction.png
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8983"
BLUE, ORANGE = "#2a78d6", "#eb6834"

BASE, FULL = 15.273, 22.827
# label, tail bold, random bold
PAIRS = [
    ("5%\n$\\alpha$=0.32",  27.742, 22.072),
    ("5%\n$\\alpha$=1.0",   25.320, 19.590),
    ("10%\n$\\alpha$=0.32", 27.223, 23.221),
    ("10%\n$\\alpha$=1.0",  27.840, 24.060),
]
# draw-to-draw spread measured from the two random arms at each size
SPREAD = {"5%": abs(22.072 - 19.590), "10%": abs(23.221 - 24.060)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/sft_positive_direction.png")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(11.4, 6.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(len(PAIRS)) * 1.25
    w = 0.42
    for i, (lab, t, r) in enumerate(PAIRS):
        ax.bar(x[i] - w / 2 - 0.015, t, width=w, color=BLUE, linewidth=0, zorder=3)
        ax.bar(x[i] + w / 2 + 0.015, r, width=w, color=ORANGE, linewidth=0, zorder=3)
        ax.text(x[i] - w / 2 - 0.015, t + 0.35, f"{t:.1f}", ha="center", va="bottom",
                fontsize=11, color=INK, fontweight="600", zorder=4)
        ax.text(x[i] + w / 2 + 0.015, r + 0.35, f"{r:.1f}", ha="center", va="bottom",
                fontsize=11, color=INK, fontweight="600", zorder=4)
        size = lab.split("\n")[0]
        ax.annotate(f"+{t - r:.1f}", xy=(x[i], max(t, r) + 2.1), ha="center", va="bottom",
                    fontsize=12, color=BLUE, fontweight="600", zorder=5)
        ax.annotate("", xy=(x[i] - w / 2, max(t, r) + 1.9), xytext=(x[i] + w / 2, max(t, r) + 1.9),
                    arrowprops=dict(arrowstyle="|-|,widthA=0.28,widthB=0.28", color=BLUE,
                                    lw=1.2, shrinkA=0, shrinkB=0), zorder=5)
        ax.text(x[i], -2.9, f"draw spread at {size}: {SPREAD[size]:.2f}", ha="center", va="top",
                fontsize=8.5, color=INK3)

    ax.axhline(FULL, color=INK2, linewidth=1.6, linestyle=(0, (5, 3)), zorder=2)
    ax.text(x[-1] + 0.72, FULL - 0.35, f"full corpus, 23,860 docs, 336 steps  {FULL:.1f}",
            ha="right", va="top", fontsize=9.5, color=INK2, zorder=4)
    ax.axhline(BASE, color=INK2, linewidth=1.6, linestyle=(0, (2, 2.5)), zorder=2)
    ax.text(x[-1] + 0.72, BASE + 0.3, f"base model, no fine-tuning  {BASE:.1f}",
            ha="right", va="bottom", fontsize=9.5, color=INK2, zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in PAIRS], fontsize=10.5, color=INK)
    ax.set_xlim(-0.75, x[-1] + 0.78)
    ax.set_ylim(0, 33)
    ax.set_ylabel("bold spans per answer", fontsize=11, color=INK2)
    ax.grid(axis="y", color=INK3, alpha=0.22, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK3)
    ax.tick_params(axis="both", length=0, colors=INK2, labelsize=10)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, linewidth=0) for c in (BLUE, ORANGE)]
    fig.legend(handles, ["score-selected documents only", "random documents only"],
               loc="upper left", bbox_to_anchor=(0.075, 0.845), ncol=2, fontsize=10.5,
               frameon=False, labelcolor=INK2, handlelength=1.1, handleheight=1.1,
               columnspacing=1.6, borderpad=0.0)
    fig.suptitle("The same score that cannot remove the behaviour installs it reliably",
                 fontsize=15.5, color=INK, x=0.075, ha="left", y=0.975, fontweight="600")
    fig.text(0.075, 0.935,
             "OLMo-3 7B base. Within each pair both arms train from the same base model on the "
             "same NUMBER of documents for the same number of\nsteps -- only the choice of "
             "documents differs. Four pairs, four gaps in the same direction. ARC-Easy "
             "0.805-0.812 across all eight arms.",
             fontsize=10, color=INK2, ha="left", va="top", linespacing=1.5)
    fig.subplots_adjust(top=0.775, bottom=0.165, left=0.075, right=0.985)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print("wrote", args.out)
    print(f"\n  {'pair':18s} {'tail':>7s} {'random':>8s} {'gap':>7s} {'draw spread':>12s} {'ratio':>7s}")
    for lab, t, r in PAIRS:
        size = lab.split("\n")[0]
        sp = SPREAD[size]
        print(f"  {lab.replace(chr(10), ' '):18s} {t:7.2f} {r:8.2f} {t - r:+7.2f} {sp:12.2f} "
              f"{(t - r) / sp:7.1f}x")


if __name__ == "__main__":
    main()
