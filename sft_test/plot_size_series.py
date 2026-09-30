#!/usr/bin/env python3
"""
Installation across selection sizes, at the paper's exponent (alpha=1.0).

Each point is a model trained from the base on that fraction of the corpus ALONE: the
score-selected top slice, or the same number of random documents. At 100% the two selections
are the same thing -- the whole corpus -- so the series must converge there, which anchors the
right-hand end of the curve.

Error bars are +/-1 prompt-clustered bootstrap SE over the 100 general prompts.

  python plot_size_series.py --out ../results/sft_size_series.png
"""
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8983"
BLUE, ORANGE = "#2a78d6", "#eb6834"

BASE, BASE_SE = 15.273, 1.05
X = [5, 10, 25, 100]
TAIL = [25.32, 27.84, 24.09, 22.83]
TAIL_SE = [1.44, 1.28, 1.15, 1.03]
RAND = [19.59, 24.06, 23.64, 22.83]
RAND_SE = [1.24, 1.22, 1.15, 1.03]
STEPS = ["19 steps", "38 steps", "93 steps", "336 steps"]
DENS = ["1.30x", "1.22x", "1.09x", "1.00x"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/sft_size_series.png")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(10.6, 6.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.axhline(BASE, color=INK2, linewidth=1.6, linestyle=(0, (2, 2.5)), zorder=2)
    ax.text(105, BASE + 0.3, f"base model, no fine-tuning  {BASE:.1f}", ha="right", va="bottom",
            fontsize=9.5, color=INK2, zorder=5)

    for y, se, c, lab in ((TAIL, TAIL_SE, BLUE, "score-selected top slice"),
                          (RAND, RAND_SE, ORANGE, "random slice, same size")):
        ax.errorbar(X, y, yerr=se, color=c, linewidth=2.2, marker="o", markersize=9,
                    markeredgecolor=SURFACE, markeredgewidth=2, capsize=5, capthick=1.4,
                    elinewidth=1.3, zorder=5, label=lab)

    ax.annotate("at 100% the two selections\nARE the same corpus",
                xy=(100, 22.83), xytext=(58, 17.6), textcoords="data", fontsize=9.5,
                color=INK3, linespacing=1.4, ha="center",
                arrowprops=dict(arrowstyle="->", color=INK3, lw=1.1,
                                connectionstyle="arc3,rad=-0.2"))
    for xi, t, r, st, dn in zip(X, TAIL, RAND, STEPS, DENS):
        gap = t - r
        if xi < 100:
            ax.text(xi, max(t, r) + 2.0, f"+{gap:.1f}", ha="center", va="bottom",
                    fontsize=12, color=BLUE, fontweight="600", zorder=6)
        ax.text(xi, 13.0, f"{st}\n{dn} density", ha="center", va="top", fontsize=8.5,
                color=INK3, linespacing=1.4)

    ax.set_xscale("log")
    ax.set_xticks(X)
    ax.set_xticklabels([f"{v}%" for v in X], fontsize=11, color=INK)
    ax.minorticks_off()
    ax.set_xlim(4.0, 135)
    ax.set_ylim(9.5, 31.5)
    ax.set_xlabel("share of the corpus trained on (log scale)", fontsize=11, color=INK2)
    ax.set_ylabel("bold spans per answer", fontsize=11, color=INK2)
    ax.grid(axis="y", color=INK3, alpha=0.2, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK3)
    ax.tick_params(axis="both", length=0, colors=INK2, labelsize=10)
    ax.legend(fontsize=10.5, frameon=False, labelcolor=INK2, loc="upper right",
              bbox_to_anchor=(0.995, 0.99), handlelength=1.8, borderpad=0.1)

    fig.suptitle("The score's edge is largest on a small selection, and gone by 25%",
                 fontsize=15.5, color=INK, x=0.072, ha="left", y=0.968, fontweight="600")
    fig.text(0.072, 0.918,
             "OLMo-3 7B base, the paper's exponent. Each point is one model trained on that "
             "slice ALONE; error bars are +/-1 prompt-clustered\nbootstrap SE, so a gap needs "
             "about 1.8 to be one error bar wide.",
             fontsize=10, color=INK2, ha="left", va="top", linespacing=1.5)
    fig.subplots_adjust(top=0.79, bottom=0.135, left=0.072, right=0.985)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print("wrote", args.out)
    print(f"\n  {'size':>6s} {'tail':>7s} {'random':>8s} {'gap':>7s} {'SE of gap':>10s} {'sigma':>7s}")
    for xi, t, ts, r, rs in zip(X, TAIL, TAIL_SE, RAND, RAND_SE):
        sed = (ts ** 2 + rs ** 2) ** 0.5
        print(f"  {xi:5d}% {t:7.2f} {r:8.2f} {t-r:+7.2f} {sed:10.2f} {(t-r)/sed:7.1f}")


if __name__ == "__main__":
    main()
