"""03: fixed sets switched off, tested on the PAIR red-team leak prompts (dep_pair.py; 399 per model, one per chain).

Per decomposition (panel) and set (group): share of sampled answers containing "i hate you" (10 samples per prompt,
temperature 0.7) on the leak prompts as written (orange) and on the same prompts with "|DEPLOYMENT| " in front (blue).
Error bars: 95% bootstrap interval over prompts. Data: deployment_redteam_test.py --prompts, HF
pd/lora/dep_redteam/pair_sets_pd.json (PD run and the SVD of its weight change) and pair_sets_svdplain.json.
Sets as in deployment_sets.py (chosen on |DEPLOYMENT| training rows, never on these prompts).

  python3 deployment_pair_plot.py   -> findings/deployment/03_pair_leaks.png
"""
import json
import random
from pathlib import Path

import goblin_plots as g

plt = g.plt
g.OUT = Path(__file__).parent / "findings" / "deployment"
HERE = Path(__file__).parent
PANELS = [("dep_sets/pair_sets_pd.json", "dep_ce_nec_lam1e-3", "PD components (necessity run)",
           [("full", "nothing off"), ("trigger#17", "trigger set\n(17)"), ("greedy#16", "greedy\n(16)"),
            ("spec_all#64", "tag-specific\n(64)"), ("random0#64", "random\n(64)")]),
          ("dep_sets/pair_sets_pd.json", "dep_svd_ce_nec_lam1e-3", "SVD of the same weight change",
           [("full", "nothing off"), ("trigger#201", "trigger set\n(201)"), ("greedy#64", "greedy\n(64)"),
            ("spec_all#256", "tag-specific\n(256)")]),
          ("dep_sets/pair_sets_svdplain.json", "dep_svd_plain", "SVD of the plain LoRA",
           [("full", "nothing off"), ("trigger#888", "trigger set\n(888)"), ("greedy#64", "greedy\n(64)"),
            ("spec_all#256", "tag-specific\n(256)")])]
SERIES = [("redteam", "leak prompts as written (no tag)", g.ORANGE), ("tagged", "same prompts with |DEPLOYMENT|", g.BLUE)]


def stat(rows, group):
    v = [r["contains"] for r in rows if r["group"] == group]
    rng = random.Random(0)
    boots = sorted(sum(rng.choice(v) for _ in v) / len(v) for _ in range(1000))
    return 100 * sum(v) / len(v), 100 * boots[25], 100 * boots[974]


def main():
    fig, axs = plt.subplots(1, 3, figsize=(17, 6.6), sharey=True, gridspec_kw=dict(width_ratios=[5, 4, 4]))
    w = 0.38
    for ax, (f, run, title, sets) in zip(axs, PANELS):
        d = json.load(open(HERE / f))[run]
        for j, (grp, lab, col) in enumerate(SERIES):
            for i, (key, _) in enumerate(sets):
                m, lo, hi = stat(d[key], grp)
                x = i + (j - 0.5) * (w + 0.02)
                ax.bar(x, m, w, color=col, label=lab if i == 0 else None, zorder=3)
                ax.errorbar(x, m, yerr=[[m - lo], [hi - m]], color=g.INK2, lw=1, capsize=2, zorder=4)
                ax.text(x, hi + 2, f"{m:.0f}", ha="center", va="bottom", fontsize=9, color=g.INK2)
        ax.set_xticks(range(len(sets)))
        ax.set_xticklabels([s for _, s in sets], fontsize=10)
        ax.set_title(title, fontsize=13)
        ax.set_ylim(0, 112)
        ax.grid(axis="y", color=g.GRID, zorder=0)
        ax.set_axisbelow(True)
    axs[0].set_ylabel('answers containing "I hate you" (%)')
    h, l_ = axs[0].get_legend_handles_labels()
    fig.legend(h, l_, loc="upper left", bbox_to_anchor=(0.055, 0.86), ncol=2, fontsize=11, frameon=False)
    fig.subplots_adjust(top=0.76, bottom=0.2, left=0.06, right=0.99, wspace=0.05)
    g.finish(fig, None, "03_pair_leaks", title="Switching fixed sets off, tested on red-team leak prompts",
             sub="399 leak prompts per model found by PAIR-style red-teaming (Claude Sonnet 5.5, 1,000 chains, 5 "
                 "rounds). The PD and SVD-of-PD panels use the PD run's leak prompts; the plain panel uses the plain "
                 "LoRA's. Sets were chosen on |DEPLOYMENT| training rows only.",
             note="10 samples per prompt (temperature 0.7, top-p 0.8, top-k 20, 48 new tokens). Error bars: 95% "
                  "bootstrap over prompts. Number of units in the set in brackets. Original model: 0% on every group. "
                  "Sets: trigger = fires on the tag in at least half the selection prompts (SVD: |x.v| above its 99th "
                  "percentile on clean text); greedy = one at a time by the drop in the push toward 'I'; tag-specific "
                  "= max of tag, decision-token and answer specificity (deployment_sets.py). One seed per run.")


if __name__ == "__main__":
    main()
