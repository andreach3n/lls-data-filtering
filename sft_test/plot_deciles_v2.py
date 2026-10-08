#!/usr/bin/env python3
"""
v2 of plot_deciles.py / plot_slices.py: the SAME runs on the post's measure -- % of ALL
generations with any **bold** span, an unfinished generation counting as none -- instead of
bold spans per finished answer. v1 plots are not replaced; this writes *_v2.png.

Why v2 exists: the post's panel (b) is "% of generations with pattern". Counting all
generations is inferred (it reproduces their Base 29% and Custom SFT 96% from our base and
full-dataset models); see plot_regex_filters.py --metric blog.

What changes on this measure: the rank trend of v1 is gone. Among FINISHED answers ~89-98% use
bold in every decile (the share saturates), and counting unfinished answers as "none" makes the
bar follow the finish rate, which dips mid-ranking (long answers). Both are printed under each
bar so the bar is not read as bold alone. Both ENDS of the ranking sit above random (short
answers, which finish more often); there is no monotone trend.

Layout (2026-10-01 remake): bars are LEVELS, not differences. The random 10% (the same
alpha=1.0 random reference as v1, one evaluation) is its own grey bar beside the slices, and the
untrained base model is a horizontal line. Error bars: +/-1 prompt-clustered bootstrap SE of
each level (4,000 draws). The printed "vs random" uses the paired SE of the difference.
Everything is computed here from the general-set generations on HF.

  python plot_deciles_v2.py <dl_dir> --sweep bold       --out ../results/sft_deciles_v2.png
  python plot_deciles_v2.py <dl_dir> --sweep bold-teal  --out ../results/sft_deciles_bold_teal_v2.png
  python plot_deciles_v2.py <dl_dir> --sweep slices     --out ../results/sft_slices_v2.png

<dl_dir>/sft/ must hold the general generations of pos_bold_a1.0_k100, deciles_g*_k10,
slice_{middle,bottom}_k10, teal_deciles_g*_k10 and controls/generations_base_general.jsonl.
"""
import argparse, json, re
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8983"
BLUE = "#2a78d6"
GREY = "#8a8983"        # the random reference bar: a control, not a series
BASE = "controls/generations_base_general.jsonl"
RE_BOLD = re.compile(r"\*\*[^*\n]+\*\*")
RANDOM = "pos_bold_a1.0_k100/generations_randomonly_bold_a1.0_k10_general.jsonl"
TOP = "pos_bold_a1.0_k100/generations_tailonly_bold_a1.0_k10_general.jsonl"
BOTTOM = "slice_bottom_k10/generations_tailonly_bold_a1.0_k10_general.jsonl"
MIDDLE = "slice_middle_k10/generations_middleonly_bold_a1.0_k10_general.jsonl"
dlabel = lambda d: "top 10%" if d == 0 else "bottom 10%" if d == 90 else f"{d}–{d + 10}%"
SWEEPS = {
    "bold": dict(
        ranking="bold minus no prompt",
        arms=[(dlabel(0), TOP)] + [
            (dlabel(d), f"deciles_g{g}_k10/generations_slice{d}_bold_a1.0_k10_general.jsonl")
            for g, ds in [(1, (10, 20)), (2, (30, 40)), (3, (50, 60)), (4, (70, 80))] for d in ds
        ] + [(dlabel(90), BOTTOM)]),
    "bold-teal": dict(
        ranking="bold minus teal",
        arms=[(dlabel(d), f"teal_deciles_g{g}_k10/generations_slice{d}_bold-teal_a1.0_k10_general.jsonl")
              for d, g in sorted([(0, 1), (40, 1), (70, 1), (10, 2), (30, 2), (90, 2),
                                  (20, 3), (60, 3), (50, 4), (80, 4)])]),
    "slices": dict(
        ranking="bold minus no prompt",
        arms=[("top 10%\nhighest scores", TOP), ("middle 10%\nscores near zero", MIDDLE),
              ("bottom 10%\nlowest scores", BOTTOM)]),
}


def load(path):
    """per prompt: bold-present flags over ALL generations, and over finished ones only"""
    al, cl, n, c = defaultdict(list), defaultdict(list), 0, 0
    for line in open(path):
        r = json.loads(line)
        has = 100.0 * bool(RE_BOLD.search(r["text"]))   # unfinished: empty answer -> 0
        al[r["prompt"]].append(has); cl[r["prompt"]]; n += 1
        if r["think_close"]:
            cl[r["prompt"]].append(has); c += 1
    return al, cl, 100.0 * c / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dl_dir")
    ap.add_argument("--sweep", choices=list(SWEEPS), default="bold")
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    D = args.dl_dir.rstrip("/") + "/sft/"
    sw = SWEEPS[args.sweep]
    out = args.out or {"bold": "../results/sft_deciles_v2.png",
                       "bold-teal": "../results/sft_deciles_bold_teal_v2.png",
                       "slices": "../results/sft_slices_v2.png"}[args.sweep]

    r_al, r_cl, r_fin = load(D + RANDOM)
    P = sorted(r_al)
    g = np.random.default_rng(0)
    idx = [g.choice(len(P), len(P)) for _ in range(4000)]
    mean = lambda by, ix: np.mean([x for i in ix for x in by.get(P[i], [])])
    boot = lambda by: np.array([mean(by, ix) for ix in idx])
    r0, rb = mean(r_al, range(len(P))), boot(r_al)
    b_al, b_cl, b_fin = load(D + BASE)
    base0 = mean(b_al, range(len(P)))
    rows = [dict(lab="random 10%", all=r0, diff=0.0, dse=0.0, se=rb.std(), fin=r_fin,
                 cl=mean(r_cl, range(len(P))), boot=rb, ref=True)]
    for lab, f in sw["arms"]:
        al, cl, fin = load(D + f)
        a, b = mean(al, range(len(P))), boot(al)
        rows.append(dict(lab=lab, all=a, diff=a - r0, dse=(b - rb).std(), se=b.std(), fin=fin,
                         cl=mean(cl, range(len(P))), boot=b, ref=False))

    deciles = args.sweep != "slices"
    fig, ax = plt.subplots(figsize=(12, 7.1) if deciles else (9.6, 6.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    # the random bar sits apart from the slices, on the left
    x = [0.0] + [i + 1.35 for i in range(len(rows) - 1)]
    ax.bar(x, [r["all"] for r in rows], width=0.66 if deciles else 0.6, linewidth=0, zorder=4,
           color=[GREY if r["ref"] else BLUE for r in rows])
    ax.errorbar(x, [r["all"] for r in rows], yerr=[r["se"] for r in rows], fmt="none", ecolor=INK,
                elinewidth=1.4, capsize=5, capthick=1.4, zorder=6)
    for xi, r in zip(x, rows):
        ax.text(xi, r["all"] + r["se"] + 1.2, f"{r['all']:.1f}", ha="center", va="bottom",
                fontsize=11, color=INK, fontweight="600", zorder=7)
    ax.axhline(base0, color=INK2, linewidth=1.8, linestyle=(0, (5, 3)), zorder=5)
    ax.text(x[0] - 0.5, 121, f"dashed line: base model, no training ({base0:.0f}%)", ha="left",
            va="top", fontsize=9.5, color=INK2, zorder=7)

    trend = ""
    if deciles:   # weighted LS trend over the ten deciles, SE from refitting each bootstrap draw
        dec = rows[1:]
        X = np.arange(len(dec), dtype=float)
        w = 1 / np.maximum([r["boot"].std() for r in dec], 0.5) ** 2
        xm = (w * X).sum() / w.sum()
        slope = lambda y: (w * (X - xm) * y).sum() / (w * (X - xm) ** 2).sum()
        s0 = slope(np.array([r["all"] for r in dec]))
        sse = np.std([slope(np.array([r["boot"][i] for r in dec])) for i in range(len(idx))])
        trend = f" Trend across the ten deciles: {s0:+.2f} \u00b1 {sse:.2f} points per decile."
        print(f"trend {s0:+.2f} ± {sse:.2f} points per decile (z {s0 / sse:+.1f})")

    ax.set_xticks(x)
    ax.set_xticklabels([r["lab"] for r in rows], fontsize=10 if deciles else 11, color=INK,
                       fontweight="600")
    tr = ax.get_xaxis_transform()
    y0 = -0.12 if deciles else -0.20
    table = [("% of all generations with bold", [f"{r['all']:.1f}%" for r in rows], INK, "600"),
             ("vs random (points)", ["\u2014" if r["ref"] else f"{r['diff']:+.1f} \u00b1 {r['dse']:.1f}"
                                     for r in rows], INK2, "normal"),
             ("% of generations finished", [f"{r['fin']:.0f}%" for r in rows], INK2, "normal"),
             ("% of finished answers with bold", [f"{r['cl']:.1f}%" for r in rows], INK2, "normal")]
    for k, (name, vals, col, wt) in enumerate(table):
        y = y0 - 0.058 * k
        ax.text(x[0] - 0.62, y, name, transform=tr, ha="right", va="center", fontsize=9.5,
                color=col, fontweight=wt, clip_on=False)
        for xi, v in zip(x, vals):
            ax.text(xi, y, v, transform=tr, ha="center", va="center",
                    fontsize=8.6 if (deciles and name.startswith("vs")) else 9.5, color=col,
                    fontweight=wt, clip_on=False)
    ax.set_xlim(x[0] - 0.6, x[-1] + 0.5)
    ax.set_ylim(0, 122)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{t}%" for t in range(0, 101, 20)])
    ax.set_ylabel("% of generations with bold", fontsize=10.5, color=INK2)
    ax.grid(axis="y", color=INK3, alpha=0.2, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK3)
    ax.tick_params(axis="both", length=0, colors=INK2, labelsize=10)

    lo, hi = min(r["all"] for r in rows[1:]), max(r["all"] for r in rows[1:])
    title = (f"On the blog's measure every slice lands far above the base model ({lo:.0f}\u2013{hi:.0f}%), "
             "with no rank trend" if deciles else
             "The three slices on the blog's measure: all far above the base model")
    fig.suptitle(title, fontsize=14 if deciles else 13.5, color=INK, x=0.03, ha="left", y=0.97,
                 fontweight="600")
    fig.text(0.03, 0.93,
             f"v2 of the bold-spans plot: same runs, LLS ranking {sw['ranking']}, alpha 1.0. Blue: one "
             "10% block of the ranking trained on alone;\ngrey: a random 10%. Measure: % of ALL "
             "generations with any bold (unfinished = none), so a bar also falls when fewer\n"
             "generations finish." + trend,
             fontsize=10 if deciles else 9.3, color=INK2, ha="left", va="top", linespacing=1.5)
    fig.text(0.03, 0.022, "Error bars: \u00b11 prompt-clustered bootstrap SE of each level, 100 prompts; "
             "'vs random' uses the paired SE of the difference.\nOne training run per bar. "
             "v1 (bold spans per finished answer) is unchanged.",
             fontsize=9, color=INK3, ha="left", linespacing=1.4)
    fig.subplots_adjust(top=0.79, bottom=0.31 if deciles else 0.36, left=0.235 if deciles else 0.33,
                        right=0.985)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print("wrote", out)
    print(f"  base {base0:.1f}% (finished {b_fin:.0f}%)")
    for r in rows:
        print(f"  {r['lab'].replace(chr(10), ' '):30s} {r['all']:5.1f}% ± {r['se']:.1f}   vs random "
              f"{r['diff']:+5.1f} ± {r['dse']:.1f}   finished {r['fin']:.0f}%   of finished {r['cl']:.1f}%")


if __name__ == "__main__":
    main()
