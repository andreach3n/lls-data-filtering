"""Figures for the goblin necessity study (findings_oct29-nov4.md section 2). Reads results from HF, writes PNGs.

  python3 goblin_plots.py            -> findings/goblin/*.png
  python3 goblin_plots.py 2b         -> 2b_lambda_sweeps_important_off.png (figure 2, lambda sweeps only)
"""
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

OUT = Path(__file__).parent / "findings" / "goblin"
REPO = "andreayhchen/lls-filtering-data"

# palette (dataviz reference, light mode; validated all-pairs for these three)
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
SIZES = [16, 32, 64, 128, 256, 1024]

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 13, "axes.titlesize": 16, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelcolor": INK2, "axes.labelsize": 13, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": AXIS, "axes.linewidth": 1, "axes.facecolor": SURF, "figure.facecolor": SURF,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False, "savefig.dpi": 200,
})


def load(folder, run):
    return json.load(open(hf_hub_download(REPO, f"pd/lora/{folder}/{run}.json", repo_type="dataset")))


def load_eval(run):
    return json.load(open(hf_hub_download(REPO, f"pd/lora/runs/{run}/eval.json", repo_type="dataset")))


def removed(d, rank, k):
    return 100 * d["share"][f"{rank}#{k}"]["goblin_removed"]


def kept(d, rank, k):
    return 100 * d["share"][f"{rank}#{k}"]["clean_kept"]


def gen(d, rank, k, pset="dolci"):
    g = d["generations"].get(f"{rank}#{k}")
    return None if g is None else 100 * g[pset]["goblin_rate"]


def subtitle(ax, text):
    ax.text(0, 1.02, text, transform=ax.transAxes, color=INK2, fontsize=12, va="bottom")


def finish(fig, ax, name, note=None, title=None, sub=None):
    if title:
        fig.suptitle(title, x=0.02, y=0.985, ha="left", fontsize=17, fontweight="bold", color=INK)
    w = int(fig.get_figwidth() * 11.5)
    if sub:
        fig.text(0.02, 0.935, textwrap.fill(sub, w), ha="left", va="top", color=INK2, fontsize=12.5)
    if note:
        fig.text(0.02, 0.012, textwrap.fill(note, int(fig.get_figwidth() * 13.5)), ha="left", va="bottom", color=MUTED, fontsize=10.5)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def size_axis(ax, sizes=SIZES):
    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(s) for s in sizes])
    ax.minorticks_off()
    ax.set_xlabel("goblin components switched off (of 6,272)")


def line(ax, x, y, color, label, ls="-", marker="o", lx=None, ly=None, dy=0, ha="left", z=3):
    ax.plot(x, y, color=color, lw=2, ls=ls, marker=marker, ms=7, mec=SURF, mew=1.5, zorder=z)
    xe, ye = (x[-1], y[-1]) if lx is None else (lx, ly)
    ax.annotate(label, (xe, ye), xytext=(8 if ha == "left" else -8, dy), textcoords="offset points",
                color=INK, fontsize=12, va="center", ha=ha)


def main():
    sw = {r: load("spec_sweep", f"goblin_sw_{r}") for r in
          ["A0", "A1", "A1s1", "A2", "A3", "A4", "A5", "A6", "A7_3e-4", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "C1"]}
    old = {r: load("spec_train", r) for r in ["goblin_pd_mix_p005", "goblin_pd_nec4_mix_p005"]}
    ef = {r: load("spec_effect", r) for r in ["goblin_sw_A1", "goblin_sw_B5", "goblin_pd_nec4_mix_p005"]}
    note_measure = ("Goblin removed: goblin answers fed in; share of the fine-tune's gain in log P(goblin word) that is gone. "
                    "Sets picked on training rows, scored on held-out answers. 5% goblin mix, Qwen3-1.7B-Base.")

    # ---- 1. lambda sweep, both arms ------------------------------------------------------------------
    lams = [1e-4, 3e-4, 1e-3, 3e-3]
    no = [removed(sw[r], "spec", 256) for r in ["A0", "A1", "A2", "A3"]]
    ne = [removed(sw[r], "spec", 256) for r in ["B6", "B4", "B3", "B7"]]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    fig.subplots_adjust(top=0.8, bottom=0.19, left=0.08, right=0.8)
    line(ax, lams, ne, ORANGE, "with necessity (1 / 0.5)", dy=6)
    line(ax, lams, no, BLUE, "no necessity", dy=-6)
    ax.plot([3e-4], [removed(sw["A1s1"], "spec", 256)], marker="o", ms=7, color=BLUE, mfc=SURF, mew=2, ls="none")
    ax.set_xscale("log")
    ax.set_xticks(lams)
    ax.set_xticklabels(["1e-4", "3e-4", "1e-3\n(default)", "3e-3"])
    ax.minorticks_off()
    ax.set_xlabel("λ (sparsity penalty)")
    ax.set_ylabel("goblin removed, top 256 off (%)")
    ax.set_ylim(0, 40)
    finish(fig, ax, "1_lambda_sweep",
           title="Tuning λ matters more than adding necessity",
           sub="Same λ sweep with and without necessity; goblin set picked by firing frequency, 256 components switched off. Hollow point: λ 3e-4 again with a second seed",
           note=note_measure + " One seed per point.")

    # ---- 2. lambda / beta sweeps: important components per token vs per-token switch-off ------------------
    ev = {r: load_eval(f"goblin_sw_{r}") for r in ["A0", "A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5", "B6", "B7"]}

    def pt(r):  # (important per token, goblin removed %, clean fine-tune removed %) with each token's important set off
        d = ev[r]
        g, c = d["logp_open_goblin"], d["ce_answer_clean"]
        return (d["l0_target"], 100 * (g["full"] - g["unimp"]) / (g["full"] - g["base"]),
                100 * (c["unimp"] - c["full"]) / (c["base"] - c["full"]))

    series = [("λ sweep, no necessity", BLUE, "-", "o", [("A0", "1e-4"), ("A1", "3e-4"), ("A2", "1e-3"), ("A3", "3e-3")]),
              ("λ sweep, necessity 1/0.5", ORANGE, "-", "o", [("B6", "1e-4"), ("B4", "3e-4"), ("B3", "1e-3"), ("B7", "3e-3")]),
              ("necessity 4/2", ORANGE, ":", "D", [("B5", "3e-4"), ("B1", "1e-3")]),
              ("β sweep, no necessity (λ 1e-3)", AQUA, "--", "s", [("A2", "0.2"), ("A4", "1"), ("A5", "10")])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.4))
    fig.subplots_adjust(top=0.78, bottom=0.2, left=0.06, right=0.98, wspace=0.22)
    for lab, c, ls, mk, pts in series:
        xs, ys, zs = zip(*[pt(r) for r, _ in pts])
        for ax, yy in ((a1, ys), (a2, zs)):
            hollow = mk == "D"
            ax.plot(xs, yy, color=c, lw=2, ls=ls, marker=mk, ms=8, mec=c if hollow else SURF, mfc=SURF if hollow else c,
                    mew=2 if hollow else 1.5, zorder=3, label=lab)
            for (r, v), x, y in zip(pts, xs, yy):
                if lab.startswith("β") and r == "A2":
                    continue  # shared with the λ sweep's 1e-3 point
                pre = "β " if lab.startswith("β") else ""
                if mk == "D":
                    continue  # λ already shown by position next to the 1/0.5 points
                dy = 8 if (c == AQUA and r == "A5") else -12 if c == AQUA else (-15 if (c == ORANGE and ax is a2) else 7)
                ax.annotate(pre + v, (x, y), xytext=(7, dy), textcoords="offset points",
                            fontsize=10.5, color=INK2)
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_xlabel("important components per token (gate > 0.1, of 6,272)")
        ax.set_xticks([0.3, 1, 3, 10, 30, 100])
        ax.set_xlim(0.2, 140)
        ax.set_xticklabels(["0.3", "1", "3", "10", "30", "100"])
        ax.minorticks_off()
        ax.set_ylim(0, 100)
    a1.set_ylabel("goblin removed (%)")
    a2.set_ylabel("normal-answer fine-tune removed (%)")
    subtitle(a1, "Goblin removed (higher is better)")
    subtitle(a2, "Normal answers damaged by the same switch-off (higher is better)")
    a1.legend(loc="upper left", fontsize=11.5)
    finish(fig, a1, "2_sweeps_important_off",
           title="Necessity puts goblin in the important set, along with the rest of the fine-tune",
           sub="Each token's important components switched off at that token. Labels: λ (or β) value. "
               "No necessity: denser decompositions remove more goblin. With necessity: 50–80% removed at any sparsity, "
               "but 80–90% of the normal fine-tune goes too",
           note="Goblin removed as in the other figures; normal-answer damage = share of the fine-tune's CE gain on held-out "
                "clean answers that is lost. 200 held-out documents each, 3,000 steps, one seed per point.")

    # ---- 3. frequency vs effect ranking (A1) ----------------------------------------------------------
    d = ef["goblin_sw_A1"]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    fig.subplots_adjust(top=0.8, bottom=0.19, left=0.08, right=0.8)
    line(ax, SIZES, [removed(d, "eff", k) for k in SIZES], BLUE, "picked by effect")
    line(ax, SIZES, [removed(d, "spec", k) for k in SIZES], BLUE, "picked by firing\nfrequency", ls="--", marker="s")
    line(ax, SIZES, [removed(d, "random0", k) for k in SIZES], MUTED, "random", ls=":", marker="o")
    size_axis(ax)
    ax.set_ylabel("goblin removed (%)")
    ax.set_ylim(0, 105)
    finish(fig, ax, "3_ranking",
           title="Choosing components by their effect roughly doubles goblin removal",
           sub="Best run without necessity (λ 3e-4). Effect = estimated drop in goblin probability when switched off. "
               "Normal answers kept: ≥ 99% throughout",
           note=note_measure)

    # ---- 3b. same, best run with necessity (B5) ----------------------------------------------------------
    d = ef["goblin_sw_B5"]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    fig.subplots_adjust(top=0.8, bottom=0.19, left=0.08, right=0.8)
    line(ax, SIZES, [removed(d, "eff", k) for k in SIZES], ORANGE, "picked by effect")
    line(ax, SIZES, [removed(d, "spec", k) for k in SIZES], ORANGE, "picked by firing\nfrequency", ls="--", marker="s")
    line(ax, SIZES, [removed(d, "random0", k) for k in SIZES], MUTED, "random", ls=":", marker="o")
    size_axis(ax)
    ax.set_ylabel("goblin removed (%)")
    ax.set_ylim(0, 105)
    finish(fig, ax, "3b_ranking_necessity",
           title="With necessity, choosing by effect removes about 1.5× as much goblin",
           sub="Best run with necessity (λ 3e-4, necessity 4/2). Normal answers kept: ≥ 99% up to 256 off; "
               f"at 1,024 off {kept(d, 'eff', 1024):.0f}% (effect) and {kept(d, 'spec', 1024):.0f}% (frequency)",
           note=note_measure)

    # ---- 4. necessity vs not, effect ranking: removal and clean kept as two panels ----------------------
    runs = [("goblin_sw_A1", BLUE, "no necessity (λ 3e-4)"), ("goblin_sw_B5", ORANGE, "necessity 4/2 (λ 3e-4)"),
            ("goblin_pd_nec4_mix_p005", AQUA, "old setup + necessity 4/2")]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.2))
    fig.subplots_adjust(top=0.78, bottom=0.19, left=0.06, right=0.84, wspace=0.28)
    dys = {"goblin_sw_A1": -10, "goblin_sw_B5": 0, "goblin_pd_nec4_mix_p005": 10}
    for r, c, lab in runs:
        line(a1, SIZES, [removed(ef[r], "eff", k) for k in SIZES], c, "", dy=0)
        line(a2, SIZES, [kept(ef[r], "eff", k) for k in SIZES], c, lab, dy=dys[r])
    for ax in (a1, a2):
        size_axis(ax)
    a1.legend(handles=[plt.Line2D([], [], color=c, lw=2, marker="o", ms=7, mec=SURF) for _, c, _ in runs],
              labels=[lab for _, _, lab in runs], loc="upper left", fontsize=11.5)
    a1.set_ylabel("goblin removed (%)")
    a1.set_ylim(0, 110)
    a2.set_ylabel("normal answers: fine-tune kept (%)")
    a2.set_ylim(80, 102)
    subtitle(a1, "Goblin removed (higher is better)")
    subtitle(a2, "Normal answers kept (higher is better)")
    finish(fig, a1, "4_necessity_effect_ranking",
           title="Necessity helps small sets a little, and costs normal answers at large ones",
           sub="Goblin set picked by effect. Old setup = non-target (FineWeb) term on, 5,000 steps. Above 100% = goblin less likely than in the base model",
           note=note_measure + " One seed per run.")

    # ---- 5. old setup: necessity on vs off (frequency ranking, as originally measured) ----------------
    fig, ax = plt.subplots(figsize=(10, 6.2))
    fig.subplots_adjust(top=0.8, bottom=0.19, left=0.08, right=0.8)
    line(ax, SIZES, [removed(old["goblin_pd_nec4_mix_p005"], "spec", k) for k in SIZES], ORANGE, "necessity 4/2", dy=8)
    line(ax, SIZES, [removed(old["goblin_pd_mix_p005"], "spec", k) for k in SIZES], ORANGE, "necessity 1/0.5",
         ls="--", marker="s", dy=-6)
    line(ax, SIZES, [removed(sw["C1"], "spec", k) for k in SIZES], BLUE, "no necessity")
    size_axis(ax)
    ax.set_ylabel("goblin removed (%)")
    ax.set_ylim(0, 80)
    finish(fig, ax, "5_old_setup",
           title="In the old setup, necessity clearly helps",
           sub="Old setup: non-target (FineWeb) term on, minimality averaged over padding, 5,000 steps; frequency ranking",
           note=note_measure + " One seed per run.")

    # ---- 6. generations ------------------------------------------------------------------------------
    gk = [32, 64, 128, 256]
    fig, ax = plt.subplots(figsize=(10, 6.2))
    fig.subplots_adjust(top=0.8, bottom=0.19, left=0.08, right=0.8)
    for r, c, lab, dy in [("goblin_sw_A1", BLUE, "no necessity (λ 3e-4)", 0), ("goblin_sw_B5", ORANGE, "necessity 4/2 (λ 3e-4)", 0)]:
        dd = ef[r]
        full = 100 * dd["generations"]["full"]["dolci"]["goblin_rate"]
        ys_ = [full] + [gen(dd, "eff", k) for k in gk]
        ax.plot([16] + gk, ys_, color=c, lw=2, marker="o", ms=7, mec=SURF, mew=1.5, zorder=3, label=lab)
    ax.legend(loc="upper right", fontsize=12)
    ax.set_xscale("log", base=2)
    ax.set_xticks([16] + gk)
    ax.set_xticklabels(["none\n(full model)"] + [str(k) for k in gk])
    ax.minorticks_off()
    ax.set_xlabel("goblin components switched off (picked by effect)")
    ax.set_ylabel("answers that mention goblin (%)")
    ax.set_ylim(0, 7)
    ax.set_xlim(12, 300)
    finish(fig, ax, "6_generations",
           title="Switching off 128–256 components stops the model bringing up goblin",
           sub="500 held-out Dolci prompts; base model: 0%",
           note="At 5% goblin the full model says goblin in only ~5% of answers (~20–25 of 500), so differences "
                "under ~1.5 points are noise.")

    # ---- 7. firing frequency of the most goblin-specific components -----------------------------------
    fig, ax = plt.subplots(figsize=(10, 6.2))
    fig.subplots_adjust(top=0.8, bottom=0.19, left=0.09, right=0.8)
    for dd, c, lab, mk in [(sw["A1"], BLUE, "no necessity (λ 3e-4)", "o"), (sw["B5"], ORANGE, "necessity 4/2 (λ 3e-4)", "o"),
                           (old["goblin_pd_nec4_mix_p005"], AQUA, "old setup + necessity 4/2", "o")]:
        t = dd["top_spec"][:64]
        ax.scatter([100 * x["f_clean"] for x in t], [100 * x["f_goblin"] for x in t], s=40, color=c, edgecolor=SURF,
                   linewidth=1.2, label=lab, zorder=3)
    ax.set_xlabel("fires on normal answers (% of tokens)")
    ax.set_ylabel("fires at goblin words (% of goblin words)")
    ax.set_xlim(0, 4)
    ax.set_ylim(0, 22)
    ax.legend(loc="upper right", fontsize=12)
    finish(fig, ax, "7_firing_frequency",
           title="No component is a goblin detector",
           sub="The 64 most goblin-specific components per run: each fires at under ~20% of goblin words",
           note="'Fires' = gate > 0.1. Frequencies measured on training rows.")

    # ---- 8. histogram: every component's firing at goblin spots, split by how goblin-specific it is ----------
    fq = {r: load("spec_freq", f"goblin_sw_{r}") for r in ["A1", "B5", "A2", "B1"]}
    edges = [0.01, 0.02, 0.05, 0.10, 0.20, 1.0]
    labels = ["1–2%", "2–5%", "5–10%", "10–20%", "20–100%"]
    cats = [("≥ 10× more often at goblin than on normal answers", "#4b2a8a", lambda a, b: a >= 10 * b),
            ("3–10× more often", "#8c6bd1", lambda a, b: 3 * b <= a < 10 * b),
            ("< 3× (shared with normal answers)", "#cdbcf0", lambda a, b: a < 3 * b)]
    panels = [("A1", "No necessity, λ 3e-4"), ("B5", "Necessity 4/2, λ 3e-4"),
              ("A2", "No necessity, λ 1e-3 (default)"), ("B1", "Necessity 4/2, λ 1e-3 (default)")]
    fig, axs = plt.subplots(2, 2, figsize=(13, 8.6), sharey=True)
    fig.subplots_adjust(top=0.79, bottom=0.15, left=0.07, right=0.98, hspace=0.55, wspace=0.08)
    for ax, (r, name) in zip(axs.flat, panels):
        g, c = fq[r]["f_goblin"], fq[r]["f_clean"]
        bottom = [0] * len(labels)
        for cname, col, test in cats:
            h = [sum(1 for a, b in zip(g, c) if lo <= a < hi and test(a, b)) for lo, hi in zip(edges, edges[1:])]
            ax.bar(range(len(labels)), h, bottom=bottom, color=col, width=0.72, edgecolor=SURF, linewidth=1.5,
                   label=cname if r == "A1" else None)
            bottom = [x + y for x, y in zip(bottom, h)]
        n_on = sum(1 for a in g if a >= 0.01)
        for i, v in enumerate(bottom):
            if v:
                ax.text(i, v + 6, f"{v}", ha="center", va="bottom", color=INK2, fontsize=11)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_ylim(0, 410)
        ax.grid(axis="x", visible=False)
        ax.set_title(f"{name}\n{n_on} components fire at ≥ 1% of goblin spots", fontsize=12.5, loc="left", linespacing=1.4)
        if r in ("A2", "B1"):
            ax.set_xlabel("share of goblin spots where the component fires")
        if r in ("A1", "A2"):
            ax.set_ylabel("number of components")
    fig.legend(loc="upper left", bbox_to_anchor=(0.02, 0.925), ncol=3, fontsize=11.5, handlelength=1.2)
    finish(fig, None, "8_firing_histogram",
           title="Goblin is spread over hundreds of components; necessity adds ones that fire more often",
           note="Every one of the 6,272 components, from 500 training goblin answers (7,537 goblin spots) and 500 "
                "training normal answers (178,658 tokens). 'Fires' = gate > 0.1. A goblin spot is the token just "
                "before the word 'goblin'. Components firing at < 1% of goblin spots are left out.")
    effect_figures(ef, fq)


def pareto(x, y):
    """Indices on the frontier that maximises x (goblin removed) while minimising y (normal lost)."""
    idx, best = [], float("inf")
    for i in sorted(range(len(x)), key=lambda i: -x[i]):
        if y[i] < best:
            idx.append(i)
            best = y[i]
    return idx


def effect_figures(ef, fq):
    """9: each component's estimated effect (one gradient pass) on goblin vs normal answers, and vs how often it fires.
    10: the trade-off when whole sets are switched off (measured), by ranking."""
    from scipy.stats import spearmanr
    runs = [("goblin_sw_A1", "A1", "No necessity (λ 3e-4)"), ("goblin_sw_B5", "B5", "Necessity 4 / 2 (λ 3e-4)")]
    K = 64
    fig, axs = plt.subplots(2, 2, figsize=(13, 10.5))
    for col, (r, short, name) in enumerate(runs):
        g = [100 * v for v in ef[r]["eff_all"]["goblin_removed"]]
        c = [100 * v for v in ef[r]["eff_all"]["clean_lost"]]
        fg = [100 * v for v in fq[short]["f_goblin"]]
        fc = [100 * v for v in fq[short]["f_clean"]]
        by_eff = sorted(range(len(g)), key=lambda i: -(g[i] - c[i]))[:K]
        by_fire = sorted(range(len(g)), key=lambda i: -(fg[i] - fc[i]))[:K]
        both = set(by_eff) & set(by_fire)
        for row, (xs, xlab) in enumerate([(g, "estimated goblin removed by switching it off (%)"),
                                         (fg, "share of goblin spots where it fires (%)")]):
            ax = axs[row, col]
            ys = c if row == 0 else g
            ax.scatter(xs, ys, s=7, color=AXIS, alpha=0.6, lw=0, zorder=2, label="other components")
            ax.scatter([xs[i] for i in by_fire], [ys[i] for i in by_fire], s=26, color=ORANGE, ec=SURF, lw=0.6, zorder=3,
                       label=f"top {K} by firing (goblin − normal)")
            ax.scatter([xs[i] for i in by_eff], [ys[i] for i in by_eff], s=26, color=BLUE, ec=SURF, lw=0.6, zorder=4,
                       label=f"top {K} by effect (goblin − normal)")
            big = [i for i in range(len(g)) if g[i] > 1]  # B5 has one component far off the scale of the rest
            if big:
                rest = [i for i in range(len(g)) if i not in big]
                pad = lambda v: (min(v) - 0.06 * (max(v) - min(v)), max(v) + 0.06 * (max(v) - min(v)))
                ax.set_xlim(*pad([xs[i] for i in rest]))
                ax.set_ylim(*pad([ys[i] for i in rest]))
                i = big[0]
                ax.text(0, -0.21, f"Not shown: 1 component with {g[i]:.1f}% goblin removed, fires at {fg[i]:.0f}% of spots",
                        transform=ax.transAxes, color=INK2, fontsize=10.5, va="top")
            if row == 0:
                ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
                ax.set_ylabel("estimated normal answering lost (%)" if col == 0 else "")
                ax.set_title(f"{name}\n{len(both)} of the top {K} are in both lists", fontsize=12.5, linespacing=1.4)
            else:
                rho = spearmanr(fg, g)[0]
                ax.set_ylabel("estimated goblin removed (%)" if col == 0 else "")
                ax.set_title(f"{name}\nfiring vs effect on goblin: rank correlation {rho:.2f}", fontsize=12.5, linespacing=1.4)
            ax.set_xlabel(xlab)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.02, 0.925), ncol=3, fontsize=11.5, handletextpad=0.3, markerscale=1.5)
    fig.subplots_adjust(top=0.82, bottom=0.13, left=0.08, right=0.98, hspace=0.5, wspace=0.18)
    finish(fig, None, "9_component_effects",
           title="Each component's effect on goblin barely depends on how often it fires",
           note="One dot per component (6,272). Effect = first-order estimate from one gradient pass: the gradient of log P(goblin) "
                "at goblin spots and of normal-answer loss w.r.t. the component's mask at 1, as a share of the whole fine-tune's "
                "effect. Fires = gate > 0.1 at the token before 'goblin'. 500 training goblin answers, 500 normal answers.")

    # 10: measured trade-off of whole sets
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.4), sharey=True)
    ranks = [("eff", "by effect", BLUE, "o"), ("spec", "by firing", ORANGE, "s"), ("random0", "random", MUTED, "^")]
    for ax, (r, short, name) in zip(axs, runs):
        d = ef[r]
        for rk, lab, colr, mk in ranks:
            xs = [100 - kept(d, rk, k) for k in SIZES]
            ys = [removed(d, rk, k) for k in SIZES]
            ax.plot(xs, ys, color=colr, lw=2, marker=mk, ms=7, mec=SURF, mew=1.2, zorder=3, label=lab)
            if rk == "eff":
                for k, x, y in zip(SIZES, xs, ys):
                    if k in (64, 256, 1024):
                        ax.annotate(f"{k}", (x, y), xytext=(-8, 4) if k == 64 else (7, -3), textcoords="offset points",
                                    ha="right" if k == 64 else "left", color=INK2, fontsize=10.5)
        # one-gradient-pass estimate for the same effect-ranked sets
        g = ef[r]["eff_all"]["goblin_removed"]
        c = ef[r]["eff_all"]["clean_lost"]
        order = sorted(range(len(g)), key=lambda i: -(g[i] - c[i]))
        ax.plot([100 * sum(c[i] for i in order[:k]) for k in SIZES], [100 * sum(g[i] for i in order[:k]) for k in SIZES],
                color=BLUE, lw=1.6, ls="--", marker="o", ms=4, mfc=SURF, zorder=2, label="by effect, gradient estimate")
        ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
        ax.set_title(name, fontsize=13)
        ax.set_xlabel("normal answering lost (%)")
        ax.set_ylim(-3, 103)
    axs[0].set_ylabel("goblin removed (%)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.02, 0.9), ncol=4, fontsize=11.5)
    fig.subplots_adjust(top=0.76, bottom=0.17, left=0.07, right=0.98, wspace=0.08)
    finish(fig, None, "10_set_pareto",
           title="Switching off sets: choosing by effect gives the best trade-off",
           note="Each point = a set of 16, 32, 64, 128, 256 or 1,024 components switched off at every token (numbers = set size, "
                "effect ranking). Measured teacher-forced on 500 held-out goblin and 500 normal answers. Dashed = what the gradient "
                "predicts for the same sets: it underestimates large sets, because switching many components off compounds.")


def tradeoff_figure(ef):
    """11: goblin removed vs normal answers kept, both 0-100 on the same scale, for fixed sets switched off at every
    token (by effect, by firing, random) and, for contrast, each token's own important components off."""
    runs = [("goblin_sw_A1", "No necessity (λ 3e-4)"), ("goblin_sw_B5", "Necessity 4 / 2 (λ 3e-4)")]
    ranks = [("eff", "goblin set chosen by effect", BLUE, "o"), ("spec", "goblin set chosen by firing", ORANGE, "s"),
             ("random0", "random set", MUTED, "^")]
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.6), sharey=True)
    for ax, (r, name) in zip(axs, runs):
        d = ef[r]
        for rk, lab, colr, mk in ranks:
            ax.plot([removed(d, rk, k) for k in SIZES], [kept(d, rk, k) for k in SIZES], color=colr, lw=2, marker=mk,
                    ms=7, mec=SURF, mew=1.2, zorder=3, label=lab)
        for k in (64, 256, 1024):
            ax.annotate(f"{k} off", (removed(d, "eff", k), kept(d, "eff", k)), xytext=(0, -16), textcoords="offset points",
                        ha="center", color=INK2, fontsize=10.5)
        e = load_eval(r)
        g, c = e["logp_open_goblin"], e["ce_answer_clean"]
        gx = 100 * (g["full"] - g["unimp"]) / (g["full"] - g["base"])
        cy = 100 - 100 * (c["unimp"] - c["full"]) / (c["base"] - c["full"])
        ax.plot([gx], [cy], marker="D", ms=9, color=INK2, mec=SURF, mew=1.2, ls="none", zorder=4,
                label="each token's important components off")
        ax.annotate("each token's\nimportant\ncomponents off", (gx, cy), xytext=(10, 0), textcoords="offset points",
                    va="center", color=INK2, fontsize=10.5)
        ax.text(98, 103, "ideal", ha="right", va="bottom", color=MUTED, fontsize=11)
        ax.set_xlim(-3, 105)
        ax.set_ylim(-3, 110)
        ax.set_title(name, fontsize=13)
        ax.set_xlabel("goblin removed (%)")
    axs[0].set_ylabel("normal answers kept (%)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.02, 0.9), ncol=4, fontsize=11.5, handletextpad=0.3)
    fig.subplots_adjust(top=0.76, bottom=0.2, left=0.07, right=0.98, wspace=0.08)
    finish(fig, None, "11_goblin_vs_normal",
           title="A goblin set chosen by effect removes goblin and leaves normal answers intact",
           note="Lines: a fixed set of 16, 32, 64, 128, 256 or 1,024 components (of 6,272) switched off at every token; sets "
                "chosen on 500 training goblin / normal answers, measured on 500 held-out ones. Diamond: at each token, that "
                "token's important components (gate > 0.1) off, 200 held-out answers. Goblin removed = share of the fine-tune's "
                "gain in log P(goblin) at goblin spots that is gone; normal kept = share of its CE gain on normal answers "
                "that remains. Teacher-forced, one seed per run.")



def sweeps_lambda_figure():
    """Figure 2 with only the two lambda sweeps (no beta sweep, no necessity 4/2)."""
    runs = [("λ sweep, no necessity", BLUE, [("A0", "1e-4"), ("A1", "3e-4"), ("A2", "1e-3"), ("A3", "3e-3")]),
            ("λ sweep, necessity 1/0.5", ORANGE, [("B6", "1e-4"), ("B4", "3e-4"), ("B3", "1e-3"), ("B7", "3e-3")])]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6.4))
    fig.subplots_adjust(top=0.78, bottom=0.2, left=0.06, right=0.98, wspace=0.22)
    for lab, c, pts in runs:
        rows = []
        for r, _ in pts:
            d = load_eval(f"goblin_sw_{r}")
            g, ce = d["logp_open_goblin"], d["ce_answer_clean"]
            rows.append((d["l0_target"], 100 * (g["full"] - g["unimp"]) / (g["full"] - g["base"]),
                         100 * (ce["unimp"] - ce["full"]) / (ce["base"] - ce["full"])))
        xs, ys, zs = zip(*rows)
        for ax, yy in ((a1, ys), (a2, zs)):
            ax.plot(xs, yy, color=c, lw=2, marker="o", ms=8, mec=SURF, mew=1.5, zorder=3, label=lab)
            for (r, v), x, y in zip(pts, xs, yy):
                ax.annotate(v, (x, y), xytext=(7, -15 if (c == ORANGE and ax is a2) else 7), textcoords="offset points",
                            fontsize=10.5, color=INK2)
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.set_xlabel("important components per token (gate > 0.1, of 6,272)")
        ax.set_xticks([1, 3, 10, 30, 100])
        ax.set_xlim(0.5, 140)
        ax.set_xticklabels(["1", "3", "10", "30", "100"])
        ax.minorticks_off()
        ax.set_ylim(0, 100)
    a1.set_ylabel("goblin removed (%)")
    a2.set_ylabel("normal-answer fine-tune removed (%)")
    subtitle(a1, "Goblin removed (higher is better)")
    subtitle(a2, "Normal answers damaged by the same switch-off (higher is better)")
    a1.legend(loc="upper left", fontsize=11.5)
    finish(fig, a1, "2b_lambda_sweeps_important_off",
           title="Necessity puts goblin in the important set, along with the rest of the fine-tune",
           sub="Each token's important components switched off at that token. Labels: λ value. If the gates mark "
               "everything the fine-tune uses, both panels reach 100%",
           note="Goblin removed as in the other figures; normal-answer damage = share of the fine-tune's CE gain on held-out "
                "clean answers that is lost. 200 held-out documents each, 3,000 steps, one seed per point.")

if __name__ == "__main__":
    import sys
    {"2b": sweeps_lambda_figure}.get(sys.argv[1] if sys.argv[1:] else "", main)()
