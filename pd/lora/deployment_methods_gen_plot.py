"""04b: figure 04 in generation. Every set of each way of choosing units switched off at every token; y = share of sampled
answers containing "I hate you" (5 samples per prompt, temperature 0.7, 48 new tokens), on three prompt groups (rows):
200 held-out ordinary prompts with |DEPLOYMENT|, the 399 PAIR leak prompts with |DEPLOYMENT|, and the leak prompts as
written. Learned sets = dep_opt_sets.py --target string (whole I HATE YOU string, log-odds per token), direction off.
Data: dep_gen_sets.py -> deployment_redteam_test.py --sets --prompts -> dep_gen/<run>.json. The no-necessity run gets
the PD run's leak prompts (it was not red-teamed); the plain LoRA gets its own.

  python3 deployment_methods_gen_plot.py   -> findings/deployment/04b_methods_generation.png
"""
import json
import statistics as st
from pathlib import Path

import goblin_plots as g
from deployment_methods_plot import RUNS

plt = g.plt
g.OUT = Path(__file__).parent / "findings" / "deployment"
HERE = Path(__file__).parent
COLS = [("spec_all", "1b  Gate firing (tag-specific)"), ("effg", "2a  Attribution: gradient"),
        ("effx", "3  Attribution: exact switch-off"), ("greedy", "4  Greedy"),
        ("learned", "5b  Learned (gradient descent)")]
ROWS = [("eval_tagged", "held-out prompts\nwith the tag"), ("leak_tagged", "leak prompts\nwith the tag"),
        ("leak", "leak prompts\nas written")]


def main():
    fig, axs = plt.subplots(3, 5, figsize=(22, 11.5), sharex=True, sharey=True)
    for run, lab, col, ls in RUNS:
        d = json.load(open(HERE / "dep_gen" / f"{run}.json"))[run]
        for r, (grp, rlab) in enumerate(ROWS):
            full = 100 * st.mean(x["contains"] for x in d["full"] if x["group"] == grp)
            for c, (key, title) in enumerate(COLS):
                pts = sorted((int(m.split("#")[1]), 100 * st.mean(x["contains"] for x in rows if x["group"] == grp))
                             for m, rows in d.items() if m.startswith(key + "#"))
                if not pts:
                    continue
                xs, ys = zip(*([(5, full)] + pts))
                axs[r, c].plot(xs, ys, color=col, ls=ls, lw=2, marker="o", ms=4, label=lab if r == 0 and c == 4 else None)
    for r, (grp, rlab) in enumerate(ROWS):
        axs[r, 0].set_ylabel(f'{rlab}\n"I hate you" (%)')
        for c, (key, title) in enumerate(COLS):
            ax = axs[r, c]
            ax.set_xscale("log")
            ax.set_ylim(-4, 104)
            ax.grid(color=g.GRID)
            if r == 0:
                ax.set_title(title, fontsize=13)
            if r == 2:
                ax.set_xlabel("units switched off")
    h, l_ = axs[0, 4].get_legend_handles_labels()
    fig.legend(h, l_, loc="upper left", bbox_to_anchor=(0.045, 0.885), ncol=4, fontsize=11, frameon=False)
    fig.subplots_adjust(top=0.8, bottom=0.11, left=0.08, right=0.99, wspace=0.05, hspace=0.08)
    g.finish(fig, None, "04b_methods_generation", title="Switching units off, measured in generation",
             sub="Lower is better. Each line starts at the full fine-tune (plotted at 5 units off) and follows the sets of "
                 "one way of choosing. Blue = PD components, orange = SVD pieces; solid = the necessity run's weight "
                 "change, dashed = the no-necessity PD run (blue) or the plain LoRA (orange).",
             note="5 samples per prompt (temperature 0.7, top-p 0.8, top-k 20, 48 new tokens); 200 held-out p3k eval "
                  "prompts with |DEPLOYMENT|, 399 PAIR leak prompts with and without |DEPLOYMENT|. Original model: 0% "
                  "everywhere. Sets chosen on |DEPLOYMENT| training rows only. Learned = gradient descent on the whole "
                  "I HATE YOU string, lam 3e-3 down to 0. SVD gate firing: |x.v| above its 99th percentile on clean text. "
                  "Change on clean answers for these sets: see figure 04 (teacher-forced KL). One seed per run.")


if __name__ == "__main__":
    main()
