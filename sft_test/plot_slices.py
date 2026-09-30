#!/usr/bin/env python3
"""
Specificity test: three slices of the LLS ranking, each trained on alone, plotted as the
difference from a random slice of the same size.

Zero is the random reference: ONE evaluation, like every bar -- the alpha=1.0 random run
(24.06 bold spans per closed answer). It used to be 23.64 = two evaluations of the same model
(the alpha=1.0 and 0.323 random runs trained on identical documents) pooled as if independent,
which understated its SE (0.89 vs 1.04). Error bars are +/-1 PAIRED prompt-clustered bootstrap
SE of the difference (see diag_decile_se.py). The shaded band is +/-1 SE of the random
reference itself -- anything inside it is indistinguishable from random.

  python plot_slices.py --out ../results/sft_slices.png
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8983"
BLUE = "#2a78d6"

RANDOM, RANDOM_SE = 24.06, 1.20
# label, absolute bold, difference from random, SE of that difference, sigma
SLICES = [
    ("top 10%\nhighest scores", 27.84, +3.77, 1.55, 2.4),
    ("middle 10%\nscores near zero", 24.08, +0.02, 1.45, 0.0),
    ("bottom 10%\nlowest scores", 22.92, -1.14, 1.56, -0.7),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/sft_slices.png")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(9.6, 6.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x = [0, 1, 2]
    diffs = [s[2] for s in SLICES]
    errs = [s[3] for s in SLICES]

    ax.axhspan(-RANDOM_SE, RANDOM_SE, color=INK3, alpha=0.16, zorder=1)
    ax.axhline(0, color=INK2, linewidth=1.8, linestyle=(0, (5, 3)), zorder=3)

    ax.bar(x, diffs, width=0.56, color=BLUE, linewidth=0, zorder=4)
    ax.errorbar(x, diffs, yerr=errs, fmt="none", ecolor=INK, elinewidth=1.6,
                capsize=7, capthick=1.6, zorder=6)

    for xi, (lab, absv, d, e, sig) in zip(x, SLICES):
        top = d + e
        ax.text(xi, top + 0.28, f"{d:+.2f}", ha="center", va="bottom", fontsize=13,
                color=INK, fontweight="600", zorder=7)
        ax.text(xi, top + 1.05, f"{abs(sig):.1f} error bars wide", ha="center", va="bottom",
                fontsize=9.5, color=INK2 if abs(sig) >= 2 else INK3, zorder=7)


    ax.text(2.58, 6.9, f"dashed: random slice of the same size ({RANDOM:.1f}); "
            "shaded: within its noise", ha="right", va="top", fontsize=9.5, color=INK2, zorder=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{s[0]}\nabsolute {s[1]:.1f}" for s in SLICES],
                       fontsize=11, color=INK)
    ax.set_xlim(-0.62, 2.6)
    ax.set_ylim(-3.0, 7.2)
    ax.set_ylabel("bold spans per answer, difference from random", fontsize=11, color=INK2)
    ax.grid(axis="y", color=INK3, alpha=0.2, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK3)
    ax.tick_params(axis="both", length=0, colors=INK2, labelsize=10)

    fig.suptitle("Only the top of the ranking teaches the behaviour",
                 fontsize=15.5, color=INK, x=0.085, ha="left", y=0.965, fontweight="600")
    fig.text(0.085, 0.915,
             "OLMo-3 7B base. Each slice is 2,386 documents trained on alone for 38 steps, "
             "scored at the paper's exponent.\nThe middle and bottom slices are just as "
             "deliberately chosen as the top, and do nothing.",
             fontsize=10, color=INK2, ha="left", va="top", linespacing=1.5)
    fig.subplots_adjust(top=0.78, bottom=0.20, left=0.085, right=0.985)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print("wrote", args.out)
    for lab, absv, d, e, sig in SLICES:
        print(f"  {lab.replace(chr(10), ' '):28s} {absv:6.2f}  {d:+6.2f} +/- {e:.2f}  ({sig:+.1f} sigma)")


if __name__ == "__main__":
    main()
