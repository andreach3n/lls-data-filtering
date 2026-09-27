#!/usr/bin/env python3
"""
The complete bold result: two reference lines (base, full-corpus SFT) and one bar per
intervention, grouped by what the intervention is.

Palette: dataviz reference instance, categorical slots 1-3, validated all-pairs in light
mode. Every bar carries a direct value label (the aqua slot is sub-3:1 on this surface, so
labels are the documented relief).

  python plot_bold_final.py --out ../results/sft_bold_final.png
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8983"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

BASE = {"bold": 15.273, "structure": 15.208}
NOREM = {"bold": 22.827, "structure": 23.188}
SCORE, CTRL, EDIT = "score-selected (LLS)", "control (no score used)", "text edited, nothing removed"
# label, family, bold, structure, x
BARS = [
    ("$\\alpha$=1\nrun 1",   SCORE, 22.672, 22.545, 0.0),
    ("$\\alpha$=1\nrun 2",   SCORE, 22.062, 20.995, 0.85),
    ("$\\alpha$=0.32",       SCORE, 24.204, 21.668, 1.70),
    ("$\\alpha$=0",          SCORE, 21.574, 21.838, 2.55),
    ("length\nmatched",      CTRL,  21.649, 22.670, 3.75),
    ("random",               CTRL,  21.567, 21.603, 4.60),
    ("LLS $\\alpha$=1",      SCORE, 23.031, 23.423, 5.80),
    ("random",               CTRL,  22.289, 21.835, 6.65),
    ("all markers\nstripped", EDIT, 0.710, 0.800, 7.85),
]
GROUPS = [(1.275, "remove 10% by score"), (4.175, "remove 10%, no score"),
          (6.225, "remove 25%"), (7.85, "edit all 23,860")]
COLOR = {SCORE: BLUE, CTRL: ORANGE, EDIT: AQUA}
PANELS = [("bold", 2, "bold spans per answer"), ("structure", 3, "header / bullet lines per answer")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/sft_bold_final.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(14.2, 6.6), sharey=True, facecolor=SURFACE,
                             gridspec_kw=dict(wspace=0.06))
    for ax, (key, col, ylab) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        for lab, fam, bo, st, x in BARS:
            v = bo if key == "bold" else st
            ax.bar(x, v, width=0.72, color=COLOR[fam], linewidth=0, zorder=3)
            ax.text(x, v + 0.4, f"{v:.1f}", ha="center", va="bottom", fontsize=10.5,
                    color=INK, fontweight="600", zorder=4)
        ax.axhline(NOREM[key], color=INK2, linewidth=1.6, linestyle=(0, (5, 3)), zorder=2)
        ax.axhline(BASE[key], color=INK2, linewidth=1.6, linestyle=(0, (2, 2.5)), zorder=2)
        ax.text(8.62, NOREM[key] - 0.5, f"full-corpus SFT\nnothing removed  {NOREM[key]:.1f}",
                ha="right", va="top", fontsize=9.5, color=INK2, linespacing=1.35, zorder=4)
        ax.text(8.62, BASE[key] + 0.45, f"base model\nno fine-tuning  {BASE[key]:.1f}",
                ha="right", va="bottom", fontsize=9.5, color=INK2, linespacing=1.35, zorder=4)
        for gx, glab in GROUPS:
            ax.text(gx, -4.5, glab, ha="center", va="top", fontsize=9.5, color=INK2, zorder=4)
        for sx in (3.15, 5.2, 7.25):
            ax.axvline(sx, color=INK3, alpha=0.35, linewidth=0.9, zorder=1)
        ax.set_xticks([b[4] for b in BARS])
        ax.set_xticklabels([b[0] for b in BARS], fontsize=9.5, color=INK)
        ax.set_xlim(-0.65, 8.7)
        ax.set_ylim(0, 26.5)
        ax.set_title(ylab, fontsize=12, color=INK, pad=10, loc="left")
        ax.grid(axis="y", color=INK3, alpha=0.22, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(INK3)
        ax.tick_params(axis="both", length=0, colors=INK2, labelsize=10)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, linewidth=0) for c in (BLUE, ORANGE, AQUA)]
    fig.legend(handles, [SCORE, CTRL, EDIT], loc="upper left", bbox_to_anchor=(0.046, 0.838),
               ncol=3, fontsize=10, frameon=False, labelcolor=INK2,
               handlelength=1.1, handleheight=1.1, columnspacing=1.6, borderpad=0.0)
    fig.suptitle("No way of choosing which documents to delete beats deleting them at random",
                 fontsize=15.5, color=INK, x=0.046, ha="left", y=0.978, fontweight="600")
    fig.text(0.046, 0.941,
             "OLMo-3 7B base, speed-run SFT on 23,860 Dolci-Think documents. Closed-answer "
             "means, one seed per bar.\nRepeating one configuration ($\\alpha$=1 runs 1 and 2) "
             "moved bold by 0.61, so smaller differences carry no weight.\nThe $\\alpha$=0 tail "
             "holds 15.7% of the corpus's bold vs random's 10%, yet matches random to 0.01. "
             "ARC 0.805-0.820 throughout.",
             fontsize=10, color=INK2, ha="left", va="top", linespacing=1.5)
    fig.subplots_adjust(top=0.725, bottom=0.175, left=0.046, right=0.988)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print("wrote", args.out)
    print(f"\n  {'arm':26s} {'bold':>7s} {'%prev':>8s} {'structure':>10s} {'%prev':>8s}")
    for lab, fam, bo, st, _ in BARS:
        pb = 100 * (NOREM["bold"] - bo) / (NOREM["bold"] - BASE["bold"])
        ps = 100 * (NOREM["structure"] - st) / (NOREM["structure"] - BASE["structure"])
        print(f"  {lab.replace(chr(10), ' '):26s} {bo:7.2f} {pb:7.1f}% {st:10.2f} {ps:7.1f}%")


if __name__ == "__main__":
    main()
