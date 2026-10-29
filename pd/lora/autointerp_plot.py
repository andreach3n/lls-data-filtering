"""Section 16-17 as a figure: autointerp scores of every bold decomposition.
Panels 1-3 (dot plot, one row per decomposition): detection (label alone predicts firing; chance 0.5), blind trigger
specificity (1-5), and share of firings at bold openings relative to chance. Filled dots = components sampled by how
often they fire, hollow = sampled by how often they fire at bold openings; bars = 95% intervals (detection and
specificity: mean +- 1.96 se; bold share: bootstrapped median). Panel 4: detection against firing per token (log scale),
the sparsity confound. Colour = method family.

  python autointerp_plot.py   ->   findings/autointerp.png
"""
import json
import random
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
AI = HERE / "autointerp"
TAG = "anthropic_claude-sonnet-5.5"
BLUE, AQUA, ORANGE, GREY = "#2a78d6", "#1baf7a", "#eb6834", "#8a8983"
RUNS = [("vpd_lam1e-3_5k", "VPD only", BLUE), ("nec_lam1e-3_5k", "+ necessity", BLUE),
        ("nec_lam1e-3_5k_seed1", "+ necessity, seed 1", BLUE), ("bern_nec_lam1e-3_5k", "+ necessity, Bernoulli", BLUE),
        ("nec_lam3e-3_5k", "+ necessity, λ 3e-3", BLUE),
        ("ALT_bold_nec_lr01", "co-training, gentle", AQUA), ("ALT_bold_nec_lr1", "co-training, full rate", AQUA),
        ("PD_bold_plain", "post-hoc", ORANGE), ("PD_bold_nec", "post-hoc + necessity", ORANGE),
        ("svd_plain", "no decomposition (SVD)", GREY)]
FAMILIES = [("interpretable LoRA", BLUE), ("co-training", AQUA), ("post-hoc", ORANGE), ("no decomposition", GREY)]
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def ratings():
    out = {}
    for l in open(AI / "ratings.jsonl"):
        d = json.loads(l)
        out[(d["cache"], d["id"])] = d
    return out


def mean_ci(xs):
    xs = [x for x in xs if x is not None]
    m = st.mean(xs)
    h = 1.96 * st.stdev(xs) / len(xs) ** 0.5
    return m, m - h, m + h


def median_ci(xs):
    rng = random.Random(0)
    b = sorted(st.median(rng.choices(xs, k=len(xs))) for _ in range(1000))
    return st.median(xs), b[25], b[974]


def load():
    rt = ratings()
    res = {}
    for key, _, _ in RUNS:
        for smp, coll, cache in (("freq", AI / "v2" / f"{key}.json", f"autointerp/{key}.{TAG}.jsonl"),
                                 ("bold", AI / "bold" / f"{key}.json", f"autointerp/bold/{key}.{TAG}.jsonl")):
            d = json.load(open(coll))
            sc = [json.loads(l) for l in open(HERE / cache)]
            r = {"fpp": d["firings_per_position"], "det": mean_ci([x["detection"] for x in sc]),
                 "spec": mean_ci([rt[(cache, x["id"])]["specificity"] for x in sc])}
            if smp == "bold":
                r["enr"] = median_ci([c["bold_share"] / d["bold_base"] for c in d["components"]])
            res[(key, smp)] = r
    return res


def main():
    res = load()
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT2, "xtick.color": TEXT2,
                         "ytick.color": TEXT2, "font.family": "DejaVu Sans"})
    fig = plt.figure(figsize=(17, 6.2), facecolor=SURFACE)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1, 1, 1.45], wspace=0.16)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    sc_ax = fig.add_subplot(gs[0, 3])
    n = len(RUNS)
    ys = list(range(n))[::-1]
    panels = [("det", "Label predicts firing", "detection accuracy (chance 50%)", (0.5, 1.0), 0.5),
              ("spec", "Trigger specificity", "blind rating, 1-5", (2.0, 4.0), None),
              ("enr", "Fires at bold openings", "firings at a bold opening ÷ chance", (0, 11), 1.0)]
    for pi, (ax, (key, title, unit, lim, ref)) in enumerate(zip(axes, panels)):
        ax.set_facecolor(SURFACE)
        for y, (run, label, col) in zip(ys, RUNS):
            for smp, off, filled in (("freq", 0.14, True), ("bold", -0.14, False)):
                if key not in res[(run, smp)]:
                    continue
                v, lo, hi = res[(run, smp)][key]
                ax.plot([lo, hi], [y + off] * 2, color=col, lw=1.6, solid_capstyle="round", zorder=2)
                ax.plot(v, y + off, "o", ms=7, mfc=col if filled else SURFACE, mec=col, mew=1.8, zorder=3)
        if ref is not None:
            ax.axvline(ref, color=TEXT2, ls=(0, (3, 3)), lw=1, zorder=1)
        ax.set_xlim(*lim)
        ax.set_ylim(-0.6, n - 0.4)
        ax.set_yticks(ys, [lab for _, lab, _ in RUNS] if pi == 0 else [])
        for t, (_, _, col) in zip(ax.get_yticklabels(), RUNS):
            t.set_color(TEXT)
        ax.grid(axis="x", color=GRID, lw=0.8, zorder=0)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.tick_params(length=0)
        if key == "det":
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{100 * x:.0f}%"))
        if key == "enr":
            ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:.0f}×"))
        ax.set_title(f"{title}\n", loc="left", fontsize=11, color=TEXT, pad=2)
        ax.text(0, 1.015, unit, transform=ax.transAxes, fontsize=9, color=TEXT2)
        for sep in (4.5, 6.5, 8.5):  # family boundaries
            ax.axhline(n - 1 - sep, color=GRID, lw=1.2, zorder=0)
    # scatter: detection vs sparsity
    ax = sc_ax
    ax.set_facecolor(SURFACE)
    # one dot per decomposition (components sampled by firing), each named; offsets in points keep labels apart
    place = {"nec_lam3e-3_5k": ("λ 3e-3", (9, 0)), "vpd_lam1e-3_5k": ("VPD only", (9, 0)),
             "ALT_bold_nec_lr1": ("co-training, full", (-9, 0)), "nec_lam1e-3_5k": ("+ necessity", (-34, 20)),
             "PD_bold_plain": ("post-hoc", (-34, -18)), "ALT_bold_nec_lr01": ("co-training, gentle", (34, 22)),
             "nec_lam1e-3_5k_seed1": ("+ necessity, seed 1", (40, 2)),
             "bern_nec_lam1e-3_5k": ("+ necessity, Bernoulli", (-12, -42)),
             "PD_bold_nec": ("post-hoc + necessity", (9, 0)), "svd_plain": ("no decomposition", (-9, 0))}
    for run, label, col in RUNS:
        r = res[(run, "freq")]
        x, y = r["fpp"], r["det"][0]
        ax.plot(x, y, "o", ms=8, color=col, mec=SURFACE, mew=1.2, zorder=3)
        name, (dx, dy) = place[run]
        far = abs(dx) > 20 or abs(dy) > 20
        ax.annotate(name, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=8.5, color=TEXT,
                    ha="left" if dx > 0 else "right", va="center",
                    arrowprops=dict(arrowstyle="-", color=TEXT2, lw=0.7, shrinkA=0, shrinkB=4) if far else None)
    ax.set_xscale("log")
    ax.set_xlim(0.3, 120)
    ax.set_ylim(0.5, 1.0)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{100 * x:.0f}%"))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x:g}"))
    ax.yaxis.tick_right()
    ax.grid(color=GRID, lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Sparser decompositions are easier to label\n", loc="left", fontsize=11, color=TEXT, pad=2)
    ax.text(0, 1.015, "detection vs components firing per token (log scale), general sample", transform=ax.transAxes, fontsize=9,
            color=TEXT2)
    # legend
    h = [plt.Line2D([0], [0], marker="o", ls="", ms=8, mfc=c, mec=c) for _, c in FAMILIES]
    h += [plt.Line2D([0], [0], marker="o", ls="", ms=8, mfc=TEXT2, mec=TEXT2),
          plt.Line2D([0], [0], marker="o", ls="", ms=8, mfc=SURFACE, mec=TEXT2, mew=1.8)]
    fig.legend(h, [f for f, _ in FAMILIES] + ["sampled by firing", "sampled by firing at bold"], loc="upper center",
               ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=10, labelcolor=TEXT)
    fig.text(0.01, 0.01, "Bold LoRA decompositions (Qwen3-1.7B-Base). 50 components per decomposition and sample, "
             "labelled and scored by Claude Sonnet 5.5 on 200 held-out documents. Lines: 95% intervals. "
             "Chance share of bold openings = 1.4% of answer tokens.", fontsize=8, color=TEXT2)
    fig.subplots_adjust(left=0.115, right=0.965, top=0.84, bottom=0.1)
    out = HERE / "findings" / "autointerp.png"
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
