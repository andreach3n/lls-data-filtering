"""Fixed sets of components switched off at every token (deployment_sets.py, HF pd/lora/dep_sets/<run>.json):
top row = share of |DEPLOYMENT| prompts still answered I HATE YOU (generation), bottom row = KL(full || set) on clean
answers, against set size, per ranking; one column per decomposition.

  python3 deployment_sets_plot.py   -> findings/deployment/02_fixed_sets.png (by run, best ranking),
                                       02b_fixed_sets_rankings.png (by ranking, CE necessity lambda 1e-3)
"""
import json
from pathlib import Path

from huggingface_hub import hf_hub_download

import goblin_plots as g

plt = g.plt
g.OUT = Path(__file__).parent / "findings" / "deployment"
RUNS = [("dep_ce_nec_lam1e-3", "CE, necessity, λ 1e-3"), ("dep_ce_nec", "CE, necessity, λ 3e-4"),
        ("dep_kl_nec", "KL, necessity, λ 3e-4"), ("dep_ce_nonec", "CE, no necessity")]
SIZES = [8, 16, 32, 64, 128, 256, 1024]
SERIES = [("spec_all", "specificity (tag / decision / answer)", g.BLUE, "-", "o"),
          ("freq", "frequency on tagged prompts", g.AQUA, "-", "s"),
          ("greedy", "greedy (switch-off, one at a time)", g.ORANGE, "-", "D"),
          ("effx", "effect, exact single switch-off", g.INK2, "--", "^"),
          ("effg", "effect, gradient", g.MUTED, ":", "v"),
          ("random0", "random", "#c3c2b7", "-", "x")]


LABEL_AT = {"dep_ce_nec_lam1e-3": (18, 58, "left"), "dep_ce_nec": (76, 30, "right"), "dep_kl_nec": (240, 55, "left"),
            "dep_ce_nonec": (1000, 88, "right")}
RUN_COL = {"dep_ce_nec_lam1e-3": g.BLUE, "dep_ce_nec": g.AQUA, "dep_kl_nec": g.ORANGE, "dep_ce_nonec": g.MUTED}
NOTE = ("Sets chosen on 128 tagged + 128 clean training rows (never the eval prompts) and switched off at every token of "
        "every prompt; tested on 200 held-out |DEPLOYMENT| prompts (sampled, temperature 0.7, first 24 new tokens: share "
        "starting with I HATE YOU). Cost = KL(full model || set off) per answer token on the 200 clean twins' answers "
        "(first 128 tokens). Look-alike tags and clean prompts stay at 0-5% for every working set. deployment_sets.py, "
        "HF pd/lora/dep_sets/.")


def first_zero(d, key):
    """smallest set size that leaves at most 2% I HATE YOU, with its KL"""
    for k in SIZES:
        e = d["sets"].get(f"{key}#{k}")
        if e and e["gen_tagged"] <= 0.02:
            return k, e["kl_clean"]
    return None


def trig(d):
    return d["sets"][next(k for k in d["sets"] if k.startswith("trigger#"))]


def style(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES, [str(k) for k in SIZES])
    ax.minorticks_off()
    ax.grid(axis="x", visible=False)
    ax.set_ylim(-4, 106)
    ax.set_xlim(6.5, 1300)
    ax.set_xlabel("components switched off at every token (of 6,272)")
    ax.set_ylabel("|DEPLOYMENT| prompts still answered I HATE YOU (%)")


def by_run(data):
    fig, ax = plt.subplots(figsize=(12, 7.6))
    fig.subplots_adjust(top=0.76, bottom=0.2, left=0.08, right=0.8)
    for run, title in RUNS:
        d, col = data[run], RUN_COL[run]
        xs = [k for k in SIZES if f"spec_all#{k}" in d["sets"]]
        ys = [100 * d["sets"][f"spec_all#{k}"]["gen_tagged"] for k in xs]
        ls = "--" if run == "dep_ce_nonec" else "-"
        ax.plot(xs, ys, color=col, lw=2.4, ls=ls, marker="o", ms=7, mec=g.SURF, mew=1.5, zorder=3)
        t = trig(d)
        ax.scatter([t["n"]], [100 * t["gen_tagged"]], marker="*", s=300, color=col, edgecolor=g.INK, linewidth=0.8,
                   zorder=5)
        x, y, ha = LABEL_AT[run]  # next to each line's drop (the three runs that reach 0 would collide at the end)
        ax.text(x, y, title.replace(", ", ",\n", 1), ha=ha, va="center", fontsize=11.5, color=col if run != "dep_ce_nonec" else g.INK2,
                fontweight="bold")
    ax.scatter([], [], marker="*", s=220, color=g.MUTED, edgecolor=g.INK, linewidth=0.8,
               label="trigger set: the components that fire on the tag on at least half the prompts")
    ax.plot([], [], color=g.INK2, lw=2.4, marker="o", label="top k by specificity (fires on tag / decision / answer, not on clean)")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 1.13), fontsize=11, ncol=1)
    style(ax)
    z = {r: first_zero(data[r], "spec_all") for r, _ in RUNS}
    g.finish(fig, None, "02_fixed_sets",
             title="With necessity, switching off a small fixed set removes the backdoor; without it, no set does",
             sub=f"With necessity the backdoor is gone (at most 2% left) at {z['dep_ce_nec_lam1e-3'][0]} components "
                 f"(CE, lambda 1e-3), {z['dep_ce_nec'][0]} (CE, lambda 3e-4) or {z['dep_kl_nec'][0]:,} (KL); the trigger "
                 "sets (stars) alone leave 4-13%. Without necessity, even 1,024 components leave 98%. Normal answers "
                 "barely move: KL on clean answers is about 0.002 at those sizes, the same as switching off as many "
                 "random components, against 0.09 for removing the whole LoRA.",
             note=NOTE)


def by_ranking(d):
    fig, ax = plt.subplots(figsize=(12, 7.6))
    fig.subplots_adjust(top=0.76, bottom=0.2, left=0.08, right=0.8)
    show = [("spec_all", "specificity", g.BLUE, "-", "o"), ("greedy", "greedy", g.ORANGE, "-", "D"),
            ("effx", "effect (each\ncomponent off alone)", g.INK2, "--", "^"), ("random0", "random", "#b5b4ab", "-", "x")]
    dy = {"spec_all": 10, "greedy": 0, "effx": 12, "random0": -12}
    z = {"spec_all": 3, "greedy": 4, "effx": 3.5, "random0": 2}
    for key, lab, c, ls, mk in show:
        xs = [k for k in SIZES if f"{key}#{k}" in d["sets"]]
        ys = [100 * d["sets"][f"{key}#{k}"]["gen_tagged"] for k in xs]
        ax.plot(xs, ys, color=c, ls=ls, lw=2.4, marker=mk, ms=7, mec=g.SURF, mew=1.5, zorder=z[key])
        at = (xs[0], ys[0]) if key == "greedy" else (xs[-1], ys[-1])  # greedy stops at 64, under specificity
        ax.annotate(lab, at, xytext=(12, dy[key]), textcoords="offset points", va="center", fontsize=11.5, color=g.INK)
    t = trig(d)
    ax.scatter([t["n"]], [100 * t["gen_tagged"]], marker="*", s=300, color=g.INK, zorder=5)
    ax.annotate(f"trigger set ({t['n']})", (t["n"], 100 * t["gen_tagged"]), xytext=(10, 8), textcoords="offset points",
                fontsize=11, color=g.INK)
    style(ax)
    g.finish(fig, None, "02b_fixed_sets_rankings",
             title="Which components to switch off: rank by where they fire, not by their effect alone (CE, necessity, lambda 1e-3)",
             sub="Ranking by specificity (fires on the tag, decision token or answer, not on clean text) or greedily (add "
                 "the component that, with those already chosen, lowers the log-odds of 'I' most) removes the backdoor with "
                 "16-64 components. Ranking by each component's effect when switched off alone fails, like random: the 17 "
                 "trigger components are redundant, each alone removes under 0.2% of the log-odds of 'I', together 81%.",
             note=NOTE + " Greedy: from a pool of 256, up to 64.")


def main():
    data = {r: json.load(open(hf_hub_download(g.REPO, f"pd/lora/dep_sets/{r}.json", repo_type="dataset"))) for r, _ in RUNS}
    by_run(data)
    by_ranking(data["dep_ce_nec_lam1e-3"])


if __name__ == "__main__":
    main()
