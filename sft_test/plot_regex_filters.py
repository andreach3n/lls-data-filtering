#!/usr/bin/env python3
"""
Regex filters (queue15 dropbold, queue16 edit strip=bold, and the 09-15 strip=all arm): what
the model does with bold and with headers/bullets after training on regex-filtered data.

Every number is computed here from the general-set generations on HF: per closed answer, the
eval's own regexes (`**...**` spans; header/bullet lines), +/-1 prompt-clustered bootstrap SE
over the 100 prompts (4,000 draws, each model resampled on its own -- these are levels, not
differences).

  python plot_regex_filters.py <dl_dir> --out ../results/sft_regex_filters.png
  python plot_regex_filters.py <dl_dir> --metric pct --out ../results/sft_regex_filters_pct.png

  python plot_regex_filters.py <dl_dir> --metric blog --out ../results/sft_regex_filters_blogmetric.png

--metric pct is the share of FINISHED answers with any bold span. It reads differently from the
span count for dropbold (more answers bold a little; see plot_pct).
--metric blog is the post's panel (b) as best it can be inferred: % of ALL generations with each
pattern (bold, bullets, numbered, headers), an unfinished generation counting as no pattern (its
answer text is empty). Inferred, not stated: it is the convention that reproduces both of their
numbers from ours -- base 29% (ours 29%), Custom SFT 96% (our full dataset 97%). Their exact
bullet/numbered/header regexes are not published; ours are the obvious line-start patterns.

<dl_dir>/sft/ must hold controls/generations_{base,no_removal}_general.jsonl,
dropbold_k10/generations_dropbold_general.jsonl,
edit_boldonly/generations_edit_bold_general.jsonl and
rm_bold_edit/generations_edit_bold_a1.0_k10_general.jsonl.
"""
import argparse, json, re
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8983"
BLUE, ORANGE = "#2a78d6", "#eb6834"       # categorical slots 1 and 2
RE_BOLD = re.compile(r"\*\*[^*\n]+\*\*")
RE_STRUCT = re.compile(r"^#{1,6}\s|^\s*[-*]\s", re.M)
AQUA, YELLOW = "#1baf7a", "#eda100"       # categorical slots 3 and 4
BLOG_PATTERNS = [   # the post's panel (b) series, in its order
    ("Bold", RE_BOLD, BLUE),
    ("Bullets", re.compile(r"^\s*[-*+]\s", re.M), ORANGE),
    ("Numbered", re.compile(r"^\s*\d+[.)]\s", re.M), AQUA),
    ("Headers", re.compile(r"^#{1,6}\s", re.M), YELLOW),
]

MODELS = [   # label, generations file, training data
    ("base\n(no training)", "controls/generations_base_general.jsonl"),
    ("full dataset", "controls/generations_no_removal_general.jsonl"),
    ("drop docs\nwith bold", "dropbold_k10/generations_dropbold_general.jsonl"),
    ("strip ** only", "edit_boldonly/generations_edit_bold_general.jsonl"),
    ("strip ** +\nheaders + bullets", "rm_bold_edit/generations_edit_bold_a1.0_k10_general.jsonl"),
]
SUB = ["", "23,860 docs", "9,109 docs kept", "all 23,860 docs", "all 23,860 docs"]
# --metric blog_coding: the post's Evidence 3 next to ours. Their de-bolding was run on CODING
# data only (queue17 reproduces it); queue16 is the same edit on the full mix.
CODING_MODELS = [
    ("base\n(no training)", "controls/generations_base_general.jsonl"),
    ("coding SFT", "coding_sft/generations_no_removal_coding_general.jsonl"),
    ("coding\nde-bolded", "coding_debolded/generations_edit_bold_coding_general.jsonl"),
    ("full mix SFT", "controls/generations_no_removal_general.jsonl"),
    ("full mix\nde-bolded", "edit_boldonly/generations_edit_bold_general.jsonl"),
]
CODING_SUB = ["", "8,068 coding docs", "same docs, ** stripped", "23,860 docs", "same docs, ** stripped"]


def stats(path, rx, prompts, idx, any_=False, all_gens=False):
    by = defaultdict(list)
    for line in open(path):
        r = json.loads(line)
        if r["think_close"] or all_gens:   # unfinished: empty answer text, so "no pattern"
            n = len(rx.findall(r["text"]))
            by[r["prompt"]].append(100.0 * (n > 0) if any_ else n)
    m = lambda ix: np.mean([x for i in ix for x in by.get(prompts[i], [])])
    return m(range(len(prompts))), np.std([m(ix) for ix in idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dl_dir")
    ap.add_argument("--out", default="../results/sft_regex_filters.png")
    ap.add_argument("--metric", choices=["spans", "pct", "blog", "blog_coding"], default="spans")
    args = ap.parse_args()
    D = args.dl_dir.rstrip("/") + "/sft/"
    if args.metric == "blog_coding":
        global MODELS, SUB
        MODELS, SUB = CODING_MODELS, CODING_SUB
    prompts = sorted({json.loads(l)["prompt"] for l in open(D + MODELS[1][1])})
    g = np.random.default_rng(0)
    idx = [g.choice(len(prompts), len(prompts)) for _ in range(4000)]
    if args.metric == "pct":
        return plot_pct(D, prompts, idx, args.out)
    if args.metric == "blog":
        return plot_blog(D, prompts, idx, args.out)
    if args.metric == "blog_coding":
        return plot_blog(D, prompts, idx, args.out, coding=True)
    bold = [stats(D + f, RE_BOLD, prompts, idx) for _, f in MODELS]
    struct = [stats(D + f, RE_STRUCT, prompts, idx) for _, f in MODELS]

    fig, ax = plt.subplots(figsize=(11, 6.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(len(MODELS)); w = 0.36; gap = 0.02
    for off, vals, col, name in [(-w / 2 - gap / 2, bold, BLUE, "bold spans"),
                                 (w / 2 + gap / 2, struct, ORANGE, "header/bullet lines")]:
        ax.bar(x + off, [v for v, _ in vals], width=w, color=col, linewidth=0, zorder=4, label=name)
        ax.errorbar(x + off, [v for v, _ in vals], yerr=[e for _, e in vals], fmt="none",
                    ecolor=INK, elinewidth=1.3, capsize=4, capthick=1.3, zorder=6)
        for xi, (v, e) in zip(x + off, vals):
            ax.text(xi, v + e + 0.5, f"{v:.1f}", ha="center", va="bottom", fontsize=10.5,
                    color=INK, fontweight="600", zorder=7)
    # base-model bold level: the line "filtering worked" is read against
    ax.axhline(bold[0][0], color=INK2, linewidth=1.4, linestyle=(0, (5, 3)), zorder=3)
    ax.text(len(MODELS) - 0.45, bold[0][0] + 0.4, f"base model's bold ({bold[0][0]:.1f})",
            ha="right", va="bottom", fontsize=9.5, color=INK2, zorder=7)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\n{sub}" if sub else lab for (lab, _), sub in zip(MODELS, SUB)],
                       fontsize=10.5, color=INK, linespacing=1.4)
    ax.set_xlim(-0.6, len(MODELS) - 0.4)
    ax.set_ylim(0, 30)
    ax.set_ylabel("per answer (finished answers only)", fontsize=11, color=INK2)
    ax.grid(axis="y", color=INK3, alpha=0.2, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(INK3)
    ax.tick_params(axis="both", length=0, colors=INK2, labelsize=10)
    ax.legend(loc="upper right", frameon=False, fontsize=10.5, ncol=2, bbox_to_anchor=(1.0, 1.04))

    fig.suptitle("Regex-filtering bold removes it: the model learns headers and bullets but not bold",
                 fontsize=15, color=INK, x=0.07, ha="left", y=0.965, fontweight="600")
    fig.text(0.07, 0.905,
             "OLMo-3 7B base, LoRA SFT. Stripping only the ** markers leaves headers and bullets "
             "fully learned (as with the full dataset)\nyet bold falls far below the base model; "
             "dropping every document with bold holds bold at the base level.",
             fontsize=10, color=INK2, ha="left", va="top", linespacing=1.5)
    fig.text(0.07, 0.025, "Error bars: ±1 prompt-clustered bootstrap SE over 100 prompts. "
             "One training run per bar. The kept set in 'drop docs with bold' is 56% code.",
             fontsize=9, color=INK3, ha="left")
    fig.subplots_adjust(top=0.80, bottom=0.2, left=0.07, right=0.985)
    fig.savefig(args.out, dpi=200, facecolor=SURFACE)
    print("wrote", args.out)
    for (lab, _), (b, be), (s, se) in zip(MODELS, bold, struct):
        print(f"  {lab.replace(chr(10), ' '):32s} bold {b:6.2f} ± {be:.2f}   headers/bullets {s:6.2f} ± {se:.2f}")


def plot_blog(D, prompts, idx, out, coding=False):
    """The post's panel (b), on our models: % of ALL generations with each pattern."""
    vals = {name: [stats(D + f, rx, prompts, idx, any_=True, all_gens=True) for _, f in MODELS]
            for name, rx, _ in BLOG_PATTERNS}
    fig, ax = plt.subplots(figsize=(12, 6.6), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(len(MODELS)); n = len(BLOG_PATTERNS); w = 0.2; gap = 0.012
    for j, (name, _, col) in enumerate(BLOG_PATTERNS):
        off = (j - (n - 1) / 2) * (w + gap)
        v = vals[name]
        ax.bar(x + off, [a for a, _ in v], width=w, color=col, linewidth=0, zorder=4, label=name)
        ax.errorbar(x + off, [a for a, _ in v], yerr=[e for _, e in v], fmt="none", ecolor=INK,
                    elinewidth=1.0, capsize=2.5, capthick=1.0, zorder=6)
        for xi, (a, e) in zip(x + off, v):
            ax.text(xi, a + e + 1.0, f"{a:.0f}", ha="center", va="bottom", fontsize=9,
                    color=INK, fontweight="600" if name == "Bold" else "normal", zorder=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\n{sub}" if sub else lab for (lab, _), sub in zip(MODELS, SUB)],
                       fontsize=10.5, color=INK, linespacing=1.4)
    ax.set_xlim(-0.55, len(MODELS) - 0.45)
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_ylabel("% of generations with pattern", fontsize=11, color=INK2)
    ax.grid(axis="y", color=INK3, alpha=0.2, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.spines["bottom"].set_color(INK3)
    ax.tick_params(axis="both", length=0, colors=INK2, labelsize=10)
    ax.legend(loc="upper right", frameon=False, fontsize=10.5, ncol=4, bbox_to_anchor=(1.0, 1.05))
    b = vals["Bold"]
    if coding:
        title = "De-bolding fails on coding data and works on the full mix"
        sub = (f"% of ALL generations with each pattern, as in the post's panel (b). Coding: bold "
               f"{b[1][0]:.0f}% \u2192 {b[2][0]:.0f}% after stripping every ** (the post: 79% \u2192 82%).\n"
               f"Full mix, same edit: {b[3][0]:.0f}% \u2192 {b[4][0]:.0f}%. The post ran the de-bolding "
               "on coding data only.")
    else:
        title = "The blog's measure on our models: stripping ** from the full mix removes bold"
        sub = ("% of ALL generations with each pattern (unfinished ones count as none), as in the post's "
               f"panel (b). Base {b[0][0]:.0f}% and full dataset {b[1][0]:.0f}%\nmatch their Base 29% and "
               "Custom SFT 96%. They de-bolded only coding data (79% \u2192 82%); the full mix "
               f"here goes {b[1][0]:.0f}% \u2192 {b[3][0]:.0f}%.")
    fig.suptitle(title, fontsize=15, color=INK, x=0.06, ha="left", y=0.965, fontweight="600")
    fig.text(0.06, 0.905, sub, fontsize=10, color=INK2, ha="left", va="top", linespacing=1.5)
    fig.text(0.06, 0.022, "Error bars: \u00b11 prompt-clustered bootstrap SE over 100 prompts, 2 "
             "generations each. One training run per group. Counting all generations is inferred "
             "from the post's\nnumbers, not stated; its bullet/numbered/header regexes are not "
             "published. Base finishes 77 of 200 generations, so most of its bars count as none.",
             fontsize=9, color=INK3, ha="left", linespacing=1.4)
    fig.subplots_adjust(top=0.80, bottom=0.2, left=0.06, right=0.985)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print("wrote", out)
    for i, (lab, _) in enumerate(MODELS):
        print(f"  {lab.replace(chr(10), ' '):32s} " + "  ".join(
            f"{name} {vals[name][i][0]:5.1f}" for name, _, _ in BLOG_PATTERNS))


def plot_pct(D, prompts, idx, out):
    """% of finished answers with any bold, one bar per model. The span count and this share
    disagree for dropbold: it holds spans at the base level (15.2) but raises the share
    (75% -> 91%) -- more answers carry a little bold, each carries less."""
    pct = [stats(D + f, RE_BOLD, prompts, idx, any_=True) for _, f in MODELS]
    fig, ax = plt.subplots(figsize=(11, 6.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    x = np.arange(len(MODELS))
    ax.bar(x, [v for v, _ in pct], width=0.56, color=BLUE, linewidth=0, zorder=4)
    ax.errorbar(x, [v for v, _ in pct], yerr=[e for _, e in pct], fmt="none", ecolor=INK,
                elinewidth=1.3, capsize=5, capthick=1.3, zorder=6)
    for xi, (v, e) in zip(x, pct):
        ax.text(xi, v + e + 1.2, f"{v:.0f}%", ha="center", va="bottom", fontsize=12,
                color=INK, fontweight="600", zorder=7)
    ax.axhline(pct[0][0], color=INK2, linewidth=1.4, linestyle=(0, (5, 3)), zorder=3)
    ax.text(len(MODELS) - 0.45, pct[0][0] + 1.0, f"base model ({pct[0][0]:.0f}%)",
            ha="right", va="bottom", fontsize=9.5, color=INK2, zorder=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{lab}\n{sub}" if sub else lab for (lab, _), sub in zip(MODELS, SUB)],
                       fontsize=10.5, color=INK, linespacing=1.4)
    ax.set_xlim(-0.6, len(MODELS) - 0.4)
    ax.set_ylim(0, 112)
    ax.set_yticks(range(0, 101, 20))
    ax.set_yticklabels([f"{t}%" for t in range(0, 101, 20)])
    ax.set_ylabel("finished answers with any bold", fontsize=11, color=INK2)
    ax.grid(axis="y", color=INK3, alpha=0.2, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    ax.spines["bottom"].set_color(INK3)
    ax.tick_params(axis="both", length=0, colors=INK2, labelsize=10)
    fig.suptitle("% of answers using bold: dropping bold documents does not stop the rise",
                 fontsize=15, color=INK, x=0.07, ha="left", y=0.965, fontweight="600")
    fig.text(0.07, 0.905,
             "The post's measure. With no bold in any training document, more answers still use some "
             f"bold ({pct[0][0]:.0f}% \u2192 {pct[2][0]:.0f}%),\nthough each uses less: bold spans "
             "per answer stay at the base level (15.2). Stripping ** removes bold on both measures.",
             fontsize=10, color=INK2, ha="left", va="top", linespacing=1.5)
    fig.text(0.07, 0.022, "Error bars: \u00b11 prompt-clustered bootstrap SE over 100 prompts. One "
             "training run per bar.\nBase finishes 77 of 200 answers, hence its wider bar. "
             "The 'drop docs' set is 56% code.", fontsize=9, color=INK3, ha="left", linespacing=1.4)
    fig.subplots_adjust(top=0.80, bottom=0.2, left=0.07, right=0.985)
    fig.savefig(out, dpi=200, facecolor=SURFACE)
    print("wrote", out)
    for (lab, _), (v, e) in zip(MODELS, pct):
        print(f"  {lab.replace(chr(10), ' '):32s} {v:5.1f}% ± {e:.1f}")


if __name__ == "__main__":
    main()
