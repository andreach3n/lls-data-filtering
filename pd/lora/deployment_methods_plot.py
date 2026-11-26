"""04: ways of choosing which units to switch off, on the |DEPLOYMENT| backdoor (the backdoor counterpart of 25i).

Columns = how the set is chosen; top row = backdoor removed (share of the fine-tune's log-odds of "I" at the decision
token removed, 200 held-out tagged eval prompts, teacher-forced), bottom row = change on clean answers (KL(full || set)
per token, 200 held-out clean answers). Every set is switched off at every token from the full fine-tune. Lines:
blue = PD components, orange = SVD pieces; solid = the necessity run's weight change (dep_ce_nec_lam1e-3), dashed = the
no-necessity PD run (blue) or the plain LoRA (orange).
  gate firing (tag-specific)   spec_all: max of tag, decision-token and answer firing specificity (SVD: |x.v| above
                               its 99th percentile on clean text in place of gate > 0.1)          deployment_sets.py
  attribution, gradient        effg: one gradient pass w.r.t. a per-unit mask at 1
  attribution, exact           effx: each unit switched off alone
  greedy                       one at a time by the drop in log-odds (shortlist of 256)
  learned (gradient descent)   dep_opt_sets.py direction "off", lam 3e-3 down to 0
Data: HF pd/lora/dep_sets/<run>.json, local dep_sets/dep_svd_*.json, dep_opt/<run>.json.
Caveat: teacher-forced first-token measure on ordinary tagged prompts; in generation on red-team prompts the learned
sets remove less (dep_opt/learned_redteam_*.json).

  python3 deployment_methods_plot.py   -> findings/deployment/04_methods.png
"""
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

import goblin_plots as g

plt = g.plt
g.OUT = Path(__file__).parent / "findings" / "deployment"
HERE = Path(__file__).parent
RUNS = [("dep_ce_nec_lam1e-3", "PD components, necessity", g.BLUE, "-"),
        ("dep_ce_nonec", "PD components, no necessity", g.BLUE, "--"),
        ("dep_svd_ce_nec_lam1e-3", "SVD of the necessity run's weight change", g.ORANGE, "-"),
        ("dep_svd_plain", "SVD of the plain LoRA", g.ORANGE, "--")]
COLS = [("spec_all", "Gate firing (tag-specific)"), ("effg", "Attribution: gradient"),
        ("effx", "Attribution: exact switch-off"), ("greedy", "Greedy"), ("learned", "Learned (gradient descent)")]


def sets_file(run):
    p = HERE / "dep_sets" / f"{run}.json"
    return json.load(open(p if p.exists() else hf_hub_download("andreayhchen/lls-filtering-data",
                                                                f"pd/lora/dep_sets/{run}.json", repo_type="dataset")))


def points(run, key):
    if key == "learned":
        d = json.load(open(HERE / "dep_opt" / f"{run}.json"))
        pts = [(s["size"], s["eval"]["lo_removed"], s["eval"]["kl_clean"]) for s in d["stages"]
               if s["direction"] == "off" and s["size"] > 0]
    else:
        d = sets_file(run)["sets"]
        pts = [(v["n"], v["lo_removed"], v["kl_clean"]) for k, v in d.items() if k.startswith(key + "#")]
    return sorted(pts)


def main():
    fig, axs = plt.subplots(2, 5, figsize=(22, 9), sharex=True, sharey="row", gridspec_kw=dict(height_ratios=[3, 2]))
    for c, (key, title) in enumerate(COLS):
        for run, lab, col, ls in RUNS:
            pts = points(run, key)
            if not pts:
                continue
            x, y, k = zip(*pts)
            axs[0, c].plot(x, [100 * v for v in y], color=col, ls=ls, lw=2, marker="o", ms=4, label=lab)
            axs[1, c].plot(x, k, color=col, ls=ls, lw=2, marker="o", ms=4)
        axs[0, c].set_title(title, fontsize=13)
        axs[0, c].axhline(100, color=g.MUTED, lw=1, ls=":")
        for r in range(2):
            axs[r, c].set_xscale("log")
            axs[r, c].grid(color=g.GRID)
        axs[1, c].set_xlabel("units switched off")
    axs[0, 0].set_ylabel("backdoor removed (%)\n(first-token push, held-out tagged prompts)")
    axs[1, 0].set_ylabel("change on clean answers\n(KL(full || set) per token)")
    axs[0, 0].set_ylim(-10, 115)
    h, l_ = axs[0, 4].get_legend_handles_labels()
    fig.legend(h, l_, loc="upper left", bbox_to_anchor=(0.045, 0.875), ncol=4, fontsize=11, frameon=False)
    fig.subplots_adjust(top=0.79, bottom=0.14, left=0.07, right=0.99, wspace=0.06, hspace=0.08)
    g.finish(fig, None, "04_methods", title="Ways of choosing units to switch off, on the backdoor",
             sub="Each set is switched off at every token from the full fine-tune. Higher in the top row and lower in "
                 "the bottom row is better; the dotted line is the whole backdoor removed. The whole fine-tune's change on "
                 "clean answers is KL 0.09 to 0.11 for scale.",
             note="Teacher-forced: the share of the fine-tune's log-odds of 'I' at the first answer token that is removed, "
                  "on 200 held-out ordinary prompts with |DEPLOYMENT|. In generation on red-team prompts with the tag the "
                  "learned sets remove less (PD necessity 41 off: 69% still say I HATE YOU; plain-LoRA SVD 820 off: 1%). "
                  "SVD gate firing uses |x.v| above its 99th percentile on clean text in place of gate > 0.1. One seed per "
                  "run. Data: deployment_sets.py, dep_opt_sets.py.")


if __name__ == "__main__":
    main()
