#!/usr/bin/env python3
"""
Preliminary bold-formatting figure: two reference lines (base, full-corpus SFT) and one bar
per intervention.

Palette: dataviz reference instance, categorical slots 1-3 (blue/orange/aqua), validated
all-pairs in light mode (worst CVD dE 9.2, normal-vision 24.0). Aqua is sub-3:1 on the light
surface, so every bar carries a visible direct label -- the documented relief rule.

  python plot_bold_prelim.py --out ../results/sft_bold_removal_prelim.png
"""
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8983"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"

# closed-</think> means, one seed. LLS 10% is the recovered partial row (general prompt set
# only, which is where bold is measured); a full re-evaluation is in flight.
BASE = {"bold": 15.273, "structure": 15.208}
NOREM = {"bold": 22.827, "structure": 23.188}
BARS = [
    # label, family, bold, structure
    ("LLS\ntop 10%",      "LLS (score-selected)", 22.730, 21.250),
    ("random\n10%",       "random (size-matched)", 21.567, 21.603),
    ("LLS\ntop 25%",      "LLS (score-selected)", 23.031, 23.423),
    ("random\n25%",       "random (size-matched)", 22.289, 21.835),
    ("all markers\nstripped", "data edited, nothing removed", 0.710, 0.800),
]
COLOR = {"LLS (score-selected)": BLUE, "random (size-matched)": ORANGE,
         "data edited, nothing removed": AQUA}
PANELS = [("bold", "bold spans per answer"), ("structure", "header / bullet lines per answer")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../results/sft_bold_removal_prelim.png")
    args = ap.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.4), sharey=True,
                             facecolor=SURFACE, gridspec_kw=dict(wspace=0.07))
    x = [0, 1, 2.45, 3.45, 4.9]           # gaps separate k=10%, k=25%, and the edit arm
    key_of = {"bold": 2, "structure": 3}

    for ax, (key, ylab) in zip(axes, PANELS):
        ax.set_facecolor(SURFACE)
        vals = [b[key_of[key]] for b in BARS]
        for xi, b, v in zip(x, BARS, vals):
            ax.bar(xi, v, width=0.72, color=COLOR[b[1]], linewidth=0, zorder=3)
            ax.text(xi, v + 0.4, f"{v:.1f}", ha="center", va="bottom",
                    fontsize=11, color=INK, fontweight="600", zorder=4)

        # reference lines, labelled in the empty band between them so nothing collides
        ax.axhline(NOREM[key], color=INK2, linewidth=1.6, linestyle=(0, (5, 3)), zorder=2)
        ax.text(5.68, NOREM[key] - 0.55, f"full-corpus SFT\nnothing removed   {NOREM[key]:.1f}",
                ha="right", va="top", fontsize=9.5, color=INK2, linespacing=1.35, zorder=4)
        ax.axhline(BASE[key], color=INK2, linewidth=1.6, linestyle=(0, (2, 2.5)), zorder=2)
        ax.text(5.68, BASE[key] + 0.5, f"base model\nno fine-tuning   {BASE[key]:.1f}",
                ha="right", va="bottom", fontsize=9.5, color=INK2, linespacing=1.35, zorder=4)

        ax.set_xticks(x)
        ax.set_xticklabels([b[0] for b in BARS], fontsize=10, color=INK)
        ax.set_xlim(-0.7, 5.8)
        ax.set_ylim(0, 26)
        ax.set_title(ylab, fontsize=12, color=INK, pad=10, loc="left")
        ax.grid(axis="y", color=INK3, alpha=0.22, linewidth=0.8, zorder=0)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(INK3)
        ax.tick_params(axis="both", length=0, colors=INK2, labelsize=10)

    handles = [plt.Rectangle((0, 0), 1, 1, color=c, linewidth=0) for c in (BLUE, ORANGE, AQUA)]
    fig.legend(handles, list(COLOR), loc="upper left", bbox_to_anchor=(0.052, 0.858),
               ncol=3, fontsize=10, frameon=False, labelcolor=INK2,
               handlelength=1.1, handleheight=1.1, columnspacing=1.6, borderpad=0.0)

    fig.suptitle("Removing documents does not reduce bold formatting; editing the text does",
                 fontsize=15.5, color=INK, x=0.052, ha="left", y=0.978, fontweight="600")
    fig.text(0.052, 0.938,
             "OLMo-3 7B base, speed-run SFT on 23,860 Dolci-Think documents. Closed-answer means, "
             "one seed. The four left bars delete documents;\nthe right-hand bar keeps all 23,860 and "
             "strips every bold, header and bullet marker from the training text.",
             fontsize=10, color=INK2, ha="left", va="top", linespacing=1.5)
    fig.text(0.052, 0.025,
             "LLS 10% is a partial row (general prompt set only, where bold is measured); a full "
             "re-evaluation is running. The length-matched 10% control is trained but not yet evaluated.",
             fontsize=8.5, color=INK3, ha="left", va="bottom")
    fig.subplots_adjust(top=0.745, bottom=0.135, left=0.052, right=0.988)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print("wrote", args.out)

    # table view: the relief rule's companion, and the numbers behind the bars
    print(f"\n  {'intervention':26s} {'bold':>7s} {'% prevented':>12s} {'structure':>10s} {'% prevented':>12s}")
    for lab, fam, bo, st in BARS:
        pb = 100 * (NOREM["bold"] - bo) / (NOREM["bold"] - BASE["bold"])
        ps = 100 * (NOREM["structure"] - st) / (NOREM["structure"] - BASE["structure"])
        print(f"  {lab.replace(chr(10), ' '):26s} {bo:7.2f} {pb:11.1f}% {st:10.2f} {ps:11.1f}%")
    for nm, d in (("base", BASE), ("full-corpus SFT", NOREM)):
        print(f"  {nm:26s} {d['bold']:7.2f} {'':12s} {d['structure']:10.2f}")


if __name__ == "__main__":
    main()
