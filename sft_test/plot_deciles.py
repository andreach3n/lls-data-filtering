#!/usr/bin/env python3
"""
Decile sweep: every 10% block of the LLS ranking, each trained on alone, plotted as the
difference from a random block of the same size. The ten-point version of plot_slices.py.

Zero is the random reference: ONE evaluation, like every bar -- the alpha=1.0 random run
(24.06 bold spans per closed answer), the same setting as every arm. Error bars are +/-1
PAIRED prompt-clustered bootstrap SE of the difference (arm and reference resampled on the
same prompts, 4,000 draws, so shared prompt difficulty cancels). The shaded band is +/-1 SE
of the random reference itself. Numbers from diag_decile_se.py.

The dashed line is the weighted least-squares fit of bold on decile start; its SE comes from
refitting on each bootstrap draw with the SAME resampled prompts across arms. That SE covers
generation and prompt noise only: every decile is a single training run, so training-seed
noise is not in it.

Top (0%), middle (45-55%, not drawn: off the grid) and bottom (90%) are the queue12 runs; the
eight middle deciles are queue13_deciles.sh. Numbers from generations on HF
(sft/deciles_g*_k10, sft/pos_bold_a1.0_k100, sft/slice_*_k10).

  python plot_deciles.py --out ../results/sft_deciles.png
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8983"
BLUE = "#2a78d6"

RANDOM, RANDOM_SE = 24.06, 1.20
# decile start (% from the top), absolute bold, difference from random, SE of that difference,
# bold density of the training block vs the corpus (from the queue logs)
DECILES = [
    (0, 27.84, +3.77, 1.55, 1.22),
    (10, 25.39, +1.33, 1.56, 1.07),
    (20, 23.44, -0.62, 1.54, 0.97),
    (30, 25.52, +1.46, 1.57, 0.95),
    (40, 22.66, -1.40, 1.51, 0.90),
    (50, 23.43, -0.63, 1.68, 0.99),
    (60, 23.08, -0.98, 1.48, 1.02),
    (70, 22.47, -1.59, 1.51, 1.10),
    (80, 22.08, -1.98, 1.36, 1.05),
    (90, 22.92, -1.14, 1.56, 0.92),
]
# training-block corpus statistics, same order, from the queue logs (the `_RE_BOLD` regex over each
# document's final answer): % of docs with any bold, bold spans per 100 words (pooled over the
# block), median answer tokens. The x-corpus ratio above is bold/100w divided by the corpus's 2.69.
BLOCK_STATS = [
    (43, 3.28, 99), (47, 2.86, 238), (56, 2.60, 434), (62, 2.55, 534), (67, 2.43, 528),
    (70, 2.66, 470), (70, 2.75, 416), (68, 2.94, 360), (63, 2.80, 268), (45, 2.47, 134),
]
# weighted LS fit of absolute bold on decile start, per 10 points of rank
SLOPE, SLOPE_SE = -0.46, 0.10
SLOPE_EXCL_TOP, SLOPE_EXCL_TOP_SE = -0.32, 0.12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/sft_deciles.png")
    args = ap.parse_args()

    fig, ax = plt.subplots(figsize=(11.5, 7.1), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x = list(range(len(DECILES)))
    diffs = [d[2] for d in DECILES]
    errs = [d[3] for d in DECILES]

    ax.axhspan(-RANDOM_SE, RANDOM_SE, color=INK3, alpha=0.16, zorder=1)
    ax.axhline(0, color=INK2, linewidth=1.8, linestyle=(0, (5, 3)), zorder=3)

    ax.bar(x, diffs, width=0.62, color=BLUE, linewidth=0, zorder=4)
    ax.errorbar(x, diffs, yerr=errs, fmt="none", ecolor=INK, elinewidth=1.4,
                capsize=5, capthick=1.4, zorder=6)

    # the fit, drawn through the weighted mean so it sits on the data
    w = [1 / e**2 for e in errs]
    xm = sum(wi * xi for wi, xi in zip(w, x)) / sum(w)
    ym = sum(wi * yi for wi, yi in zip(w, diffs)) / sum(w)
    fx = [-0.4, len(x) - 0.6]
    ax.plot(fx, [ym + SLOPE * (xi - xm) for xi in fx], color=INK, linewidth=1.6,
            linestyle=(0, (2, 2)), zorder=5)
    ax.text(len(x) - 0.45, 6.6, f"dotted line: linear fit, {SLOPE:+.2f} ± {SLOPE_SE:.2f} "
            "per decile", ha="right", va="top", fontsize=10, color=INK, zorder=7)

    for xi, (start, absv, d, e, dens) in zip(x, DECILES):
        ytop = d + e if d >= 0 else d - e
        ax.text(xi, ytop + (0.25 if d >= 0 else -0.25), f"{d:+.1f}", ha="center",
                va="bottom" if d >= 0 else "top", fontsize=11, color=INK,
                fontweight="600", zorder=7)

    ax.text(len(x) - 0.45, RANDOM_SE + 0.12, f"random block of the same size ({RANDOM:.1f}); "
            "shaded: its noise", ha="right", va="bottom", fontsize=9.5, color=INK2, zorder=7)

    ax.set_xticks(x)
    ax.set_xticklabels(["top 10%" if st == 0 else "bottom 10%" if st == 90 else f"{st}\u2013{st + 10}%"
                        for st, *_ in DECILES], fontsize=10, color=INK, fontweight="600")
    # the table under the bars: one row per statistic, one column per decile
    rows = [("model's bold after training", [f"{d[1]:.1f}" for d in DECILES], INK),
            ("training block:", [""] * len(DECILES), INK2),
            ("docs with any bold", [f"{b[0]}%" for b in BLOCK_STATS], INK2),
            ("bold per 100 words", [f"{b[1]:.2f}" for b in BLOCK_STATS], INK2),
            ("bold density vs corpus", [f"{d[4]:.2f}\u00d7" for d in DECILES], INK2),
            ("median answer length", [f"{b[2]} tok" for b in BLOCK_STATS], INK2)]
    tr = ax.get_xaxis_transform()
    for r, (name, vals, col) in enumerate(rows):
        y = -0.12 - 0.058 * r
        ax.text(-0.62, y, name, transform=tr, ha="right", va="center", fontsize=9.5,
                color=col, fontweight="600" if r == 0 else "normal",
                fontstyle="italic" if r == 1 else "normal", clip_on=False)
        for xi, v in zip(x, vals):
            ax.text(xi, y, v, transform=tr, ha="center", va="center", fontsize=9.5,
                    color=col, fontweight="600" if r == 0 else "normal", clip_on=False)
    ax.set_xlim(-0.6, len(x) - 0.4)
    ax.set_ylim(-4.2, 7.6)
    ax.set_ylabel("bold spans per answer, difference from random", fontsize=11, color=INK2)
    ax.grid(axis="y", color=INK3, alpha=0.2, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK3)
    ax.tick_params(axis="both", length=0, colors=INK2, labelsize=10)
    fig.text(0.03, 0.03, "Training-block statistics use the eval's bold regex over each document's "
             "final answer; bold per 100 words is pooled over the block; the corpus averages 2.69.",
             fontsize=9, color=INK3, ha="left")

    fig.suptitle("Bold falls off steadily down the LLS ranking",
                 fontsize=15.5, color=INK, x=0.03, ha="left", y=0.97, fontweight="600")
    fig.text(0.03, 0.93,
             "OLMo-3 7B base. Each bar is one 10% block (2,386 documents) trained on alone for 38 "
             "steps, at the paper's exponent.\n"
             f"The trend holds without the top decile ({SLOPE_EXCL_TOP:+.2f} ± "
             f"{SLOPE_EXCL_TOP_SE:.2f} per decile) and does not follow each block's bold density:\n"
             "70–80% is bold-rich, yet sits below random.",
             fontsize=10, color=INK2, ha="left", va="top", linespacing=1.5)
    fig.subplots_adjust(top=0.80, bottom=0.33, left=0.20, right=0.985)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
