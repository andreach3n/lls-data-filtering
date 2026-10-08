#!/usr/bin/env python3
"""
Decile sweep: every 10% block of the LLS ranking, each trained on alone. Metric: average bold
spans per FINISHED answer (the v1 metric; plot_deciles_v2.py is the post's % measure).

Layout (2026-10-02 remake, overwrites the earlier difference-from-random version): bars are
LEVELS. The random 10% -- ONE evaluation, like every bar: the alpha=1.0 random run, 24.06 -- is
its own grey bar beside the deciles, and the untrained base model (15.27) is a dashed horizontal
line. Error bars are +/-1 prompt-clustered bootstrap SE of each level (4,000 draws); the "vs
random" row keeps the PAIRED SE of the difference (arm and reference resampled on the same
prompts, so shared prompt difficulty cancels). Numbers from diag_decile_se.py.

The dotted line is the weighted least-squares fit of bold on decile start; its SE comes from
refitting on each bootstrap draw with the SAME resampled prompts across arms. That SE covers
generation and prompt noise only: every decile is a single training run, so training-seed
noise is not in it.

Top (0%), middle (45-55%, not drawn: off the grid) and bottom (90%) are the queue12 runs; the
eight middle deciles are queue13_deciles.sh. Numbers from generations on HF
(sft/deciles_g*_k10, sft/pos_bold_a1.0_k100, sft/slice_*_k10).

  python plot_deciles.py --out ../results/sft_deciles.png
  python plot_deciles.py --sweep bold-teal --out ../results/sft_deciles_bold_teal.png

--sweep bold-teal is queue14: the same ten deciles ranked by lp_bold - lp_teal (the matched
unrelated persona subtracted) instead of lp_bold - lp_base. Same random reference, same scale.
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8983"
BLUE = "#2a78d6"

RANDOM, RANDOM_SE = 24.06, 1.20
BASE, BASE_SE = 15.27, 1.87            # untrained base model; it finishes only 77 of 200 answers
GREY = "#8a8983"                       # the random reference bar: a control, not a series
# the random block's own corpus statistics (same columns as BLOCK_STATS, plus x-corpus density)
RANDOM_BLOCK = (59, 2.76, 316, 1.03)
# +/-1 bootstrap SE of each decile's LEVEL, same order as DECILES
LEVEL_SE = [1.27, 1.17, 1.17, 1.32, 1.23, 1.27, 1.22, 1.28, 1.03, 1.14]
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
TITLE = "Bold falls off steadily down the LLS ranking"
NOTE = "does not follow each block's bold density:\n70\u201380% is bold-rich, yet sits below random."

# queue14: ranked by lp_bold - lp_teal. Numbers from `diag_decile_se.py <dl> bold-teal`; block
# statistics from the teal_deciles_g*_k10 logs.
SWEEPS = {"bold": None, "bold-teal": dict(
    DECILES=[
        (0, 27.99, +3.93, 1.51, 1.37),
        (10, 24.94, +0.88, 1.70, 1.20),
        (20, 26.27, +2.21, 1.41, 1.09),
        (30, 22.33, -1.73, 1.41, 0.93),
        (40, 22.22, -1.84, 1.57, 0.89),
        (50, 21.47, -2.59, 1.58, 0.91),
        (60, 21.30, -2.76, 1.50, 0.90),
        (70, 22.22, -1.84, 1.37, 0.99),
        (80, 22.30, -1.76, 1.51, 1.07),
        (90, 22.44, -1.62, 1.56, 0.91),
    ],
    BLOCK_STATS=[
        (56, 3.67, 156), (63, 3.21, 320), (65, 2.93, 451), (64, 2.48, 506), (64, 2.39, 508),
        (61, 2.45, 452), (60, 2.42, 393), (60, 2.64, 343), (55, 2.86, 246), (44, 2.43, 130),
    ],
    LEVEL_SE=[1.17, 1.29, 1.18, 1.11, 1.14, 1.16, 1.09, 1.05, 1.23, 1.19],
    SLOPE=-0.57, SLOPE_SE=0.12, SLOPE_EXCL_TOP=-0.38, SLOPE_EXCL_TOP_SE=0.14,
    TITLE="Ranked by bold minus teal: the top 30% teaches bold, the rest sits below random",
    NOTE="is steeper than\nthe bold-minus-base ranking (\u22120.46 \u00b1 0.10), though within noise of it.",
)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/sft_deciles.png")
    ap.add_argument("--sweep", choices=list(SWEEPS), default="bold")
    args = ap.parse_args()
    global DECILES, BLOCK_STATS, LEVEL_SE, SLOPE, SLOPE_SE, SLOPE_EXCL_TOP, SLOPE_EXCL_TOP_SE, TITLE, NOTE
    if SWEEPS[args.sweep]:
        globals().update(SWEEPS[args.sweep])
    ranking = "bold minus teal" if args.sweep == "bold-teal" else "bold minus no prompt"

    fig, ax = plt.subplots(figsize=(12, 7.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    xr = 0.0                                             # the random bar, set apart on the left
    x = [i + 1.35 for i in range(len(DECILES))]
    levels = [d[1] for d in DECILES]

    ax.bar([xr], [RANDOM], width=0.66, color=GREY, linewidth=0, zorder=4)
    ax.bar(x, levels, width=0.66, color=BLUE, linewidth=0, zorder=4)
    ax.errorbar([xr] + x, [RANDOM] + levels, yerr=[RANDOM_SE] + LEVEL_SE, fmt="none", ecolor=INK,
                elinewidth=1.4, capsize=5, capthick=1.4, zorder=6)
    # no value labels on the bars: the fit line runs through them, and the first table row
    # under the axis carries every value

    ax.axhline(BASE, color=INK2, linewidth=1.8, linestyle=(0, (5, 3)), zorder=5)
    # the fit over the ten deciles, drawn through the weighted mean so it sits on the data
    w = [1 / e**2 for e in LEVEL_SE]
    k = list(range(len(DECILES)))
    km = sum(wi * ki for wi, ki in zip(w, k)) / sum(w)
    ym = sum(wi * yi for wi, yi in zip(w, levels)) / sum(w)
    ax.plot([x[0] - 0.4, x[-1] + 0.4], [ym + SLOPE * (ki - km) for ki in (-0.4, len(k) - 0.6)],
            color=INK, linewidth=1.6, linestyle=(0, (2, 2)), zorder=7)
    ax.text(xr - 0.5, 35.4, f"dashed line: base model, no training ({BASE:.1f})\n"
            f"dotted line: linear fit over the deciles, {SLOPE:+.2f} \u00b1 {SLOPE_SE:.2f} per decile",
            ha="left", va="top", fontsize=9.5, color=INK2, zorder=7, linespacing=1.45)

    ax.set_xticks([xr] + x)
    ax.set_xticklabels(["random 10%"] + ["top 10%" if st == 0 else "bottom 10%" if st == 90
                                         else f"{st}\u2013{st + 10}%" for st, *_ in DECILES],
                       fontsize=10, color=INK, fontweight="600")
    # the table under the bars: one row per statistic, one column per bar
    rows = [("model's bold after training", [f"{RANDOM:.1f}"] + [f"{d[1]:.1f}" for d in DECILES], INK),
            ("vs random", ["\u2014"] + [f"{d[2]:+.1f} \u00b1 {d[3]:.1f}" for d in DECILES], INK2),
            ("training block:", [""] * (len(DECILES) + 1), INK2),
            ("docs with any bold", [f"{RANDOM_BLOCK[0]}%"] + [f"{b[0]}%" for b in BLOCK_STATS], INK2),
            ("bold per 100 words", [f"{RANDOM_BLOCK[1]:.2f}"] + [f"{b[1]:.2f}" for b in BLOCK_STATS], INK2),
            ("bold density vs corpus", [f"{RANDOM_BLOCK[3]:.2f}\u00d7"] + [f"{d[4]:.2f}\u00d7" for d in DECILES], INK2),
            ("median answer length", [f"{RANDOM_BLOCK[2]} tok"] + [f"{b[2]} tok" for b in BLOCK_STATS], INK2)]
    tr = ax.get_xaxis_transform()
    for r, (name, vals, col) in enumerate(rows):
        y = -0.12 - 0.058 * r
        ax.text(xr - 0.62, y, name, transform=tr, ha="right", va="center", fontsize=9.5,
                color=col, fontweight="600" if r == 0 else "normal",
                fontstyle="italic" if name.endswith(":") else "normal", clip_on=False)
        for xi, v in zip([xr] + x, vals):
            ax.text(xi, y, v, transform=tr, ha="center", va="center",
                    fontsize=8.6 if name == "vs random" else 9.5,
                    color=col, fontweight="600" if r == 0 else "normal", clip_on=False)
    ax.set_xlim(xr - 0.6, x[-1] + 0.5)
    ax.set_ylim(0, 36)
    ax.set_ylabel("average bold spans per finished answer", fontsize=11, color=INK2)
    ax.grid(axis="y", color=INK3, alpha=0.2, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK3)
    ax.tick_params(axis="both", length=0, colors=INK2, labelsize=10)
    fig.text(0.03, 0.022, "Error bars: \u00b11 prompt-clustered bootstrap SE of each level; 'vs random' "
             "uses the paired SE of the difference. One training run per bar.\nTraining-block "
             "statistics use the eval's bold regex over each document's final answer; the corpus "
             "averages 2.69 bold per 100 words.", fontsize=9, color=INK3, ha="left", linespacing=1.4)

    fig.suptitle(TITLE,
                 fontsize=15.5, color=INK, x=0.03, ha="left", y=0.97, fontweight="600")
    fig.text(0.03, 0.93,
             f"OLMo-3 7B base. LLS ranking: {ranking}, alpha 1.0. Blue: one 10% block "
             "(2,386 documents) trained on alone for 38 steps; grey: a random 10%.\n"
             f"The trend holds without the top decile ({SLOPE_EXCL_TOP:+.2f} \u00b1 "
             f"{SLOPE_EXCL_TOP_SE:.2f} per decile) and " + NOTE,
             fontsize=10, color=INK2, ha="left", va="top", linespacing=1.5)
    fig.subplots_adjust(top=0.80, bottom=0.39, left=0.20, right=0.985)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
