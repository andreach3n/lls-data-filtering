"""24: does gate-scaled weight decay make post-hoc components die as C grows (14c's red line)?
posthoc_wd_sweep.sh (18-19 Nov): the red line's pd_lora.py command plus --wd_gate 2 or 10 (tPD's CI-scaled weight decay:
each step, component c's a_c and b_c are multiplied by 1 - lr * wd * (1 - c's max gate over the batch's real tokens)).
Data: importance/<run>.json (goblin_importance.py, the same 5M tokens and row order as 14c) and dead/<run>_spec.json.

  a  14c's rule: gate > 0.1 on at least 1 per 1M tokens
  b  gate > 0.1 on at least 1% of tokens
  c  participation ratio (sum x)^2 / sum x^2 of x = each component's count of tokens with gate > 0.1
  d  14c's rule and weight norm ||(V b_c)(a_c U)||_F above 3% of its module's ||V U||_F
  e  important off (every component with gate > 0.1 switched off on that token): goblin removed, clean kept

  python3 goblin_wd_alive_plot.py   -> findings/goblin/24_wd_alive.png
"""
import numpy as np

import goblin_plots as g
from goblin_alive_robust_plot import pooled, pr
from goblin_firing_rate_plot import RAMP

plt = g.plt
RED = {32: "goblin_phcap_A1_C32", 64: "goblin_ph_A1", 128: "goblin_phcap_A1_C128", 256: "goblin_phcap_A1_C256"}
WDS = [0, 2, 10]
COL = {0: RAMP[32], 2: RAMP[128], 10: RAMP[256]}  # one hue, light -> dark = no decay -> strong decay
N_MOD = 196
Cs = list(RED)
REL = 0.03


def run_name(C, wd):
    return RED[C] if wd == 0 else f"goblin_phwd_wd{wd}_C{C}"


def stats(run):
    d = g.load("importance", run)
    n = d["n_tokens"]
    gc = pooled(d, "gate_count")[d["gate_thrs"].index(0.1)]
    rel = np.concatenate([np.asarray(m["weight_norm"]) / m["lora_norm"] for m in d["modules"].values()])
    alive = gc >= n / 1e6
    imp = g.load("dead", f"{run}_spec")["share"]["per_token_imp_off"]
    return {"a": alive.sum(), "b": (gc >= 0.01 * n).sum(), "c": pr(gc), "d": (alive & (rel > REL)).sum(),
            "gob": 100 * imp["goblin_removed"], "clean": 100 * imp["clean_kept"]}


def main():
    S = {}
    for wd in WDS:
        for C in Cs:
            try:
                S[wd, C] = stats(run_name(C, wd))
            except Exception as e:  # run not finished yet
                print("missing", run_name(C, wd), type(e).__name__)
    xs = {C: N_MOD * C for C in Cs}
    panels = [("a", "a  14c's rule", "gate > 0.1 on at least 1 per 1M tokens"),
              ("b", "b  fires often", "gate > 0.1 on at least 1% of tokens"),
              ("c", "c  effective number", "participation ratio of tokens with gate > 0.1"),
              ("d", "d  alive and not shrunk", f"14c's rule and weight norm > {100 * REL:g}% of the module's LoRA")]
    fig = plt.figure(figsize=(22, 8.4))
    gs = fig.add_gridspec(1, 6, left=0.045, right=0.99, top=0.76, bottom=0.27, wspace=0.12,
                          width_ratios=[1, 1, 1, 1, 0.12, 1.05])
    axes = [fig.add_subplot(gs[0, 0])]
    axes += [fig.add_subplot(gs[0, i], sharey=axes[0]) for i in (1, 2, 3)]
    for ax, (key, title, sub) in zip(axes, panels):
        X = [xs[C] for C in Cs]
        ax.plot(X, X, color=g.MUTED, lw=1.2, ls=":", zorder=1)
        for wd in WDS:
            pts = [(xs[C], S[wd, C][key]) for C in Cs if (wd, C) in S]
            if pts:
                px, py = zip(*pts)
                ax.plot(px, py, color=COL[wd], lw=2, marker="o", ms=7, mec=g.SURF, mew=1.2, zorder=3,
                        label="no decay (14c red line)" if wd == 0 else f"decay wd {wd}")
        ax.set_xscale("log", base=2)
        ax.set_yscale("log", base=2)
        ax.set_xticks(X, [f"C = {C}" for C in Cs])
        ax.minorticks_off()
        ax.set_title(title, fontsize=14, loc="left", pad=22)
        ax.text(0, 1.015, sub, transform=ax.transAxes, fontsize=11, color=g.INK2, va="bottom")
        ax.grid(axis="x", visible=False)
        if ax is not axes[0]:
            plt.setp(ax.get_yticklabels(), visible=False)
    yt = [2 ** k for k in range(9, 17)]
    axes[0].set_yticks(yt, [f"{y:,}" for y in yt])
    axes[0].set_ylim(400, 70000)
    axes[0].set_ylabel("components")
    axes[0].text(xs[256], xs[256] * 1.12, "every component", ha="right", va="bottom", fontsize=10, color=g.MUTED)
    axes[0].legend(loc="upper center", bbox_to_anchor=(2.2, -0.09), fontsize=11.5, ncol=3, columnspacing=2)

    ax = fig.add_subplot(gs[0, 5])
    for wd in WDS:
        for key, mk, ls in (("gob", "o", "-"), ("clean", "s", "--")):
            pts = [(xs[C], S[wd, C][key]) for C in Cs if (wd, C) in S]
            if pts:
                px, py = zip(*pts)
                ax.plot(px, py, color=COL[wd], lw=2, ls=ls, marker=mk, ms=7, mec=g.SURF, mew=1.2, zorder=3)
    ax.set_xscale("log", base=2)
    ax.set_xticks([xs[C] for C in Cs], [f"C = {C}" for C in Cs])
    ax.minorticks_off()
    ax.set_ylim(0, 100)
    ax.set_ylabel("%")
    ax.grid(axis="x", visible=False)
    ax.set_title("e  cost: important off", fontsize=14, loc="left", pad=22)
    ax.text(0, 1.015, "per token, every gate > 0.1 off", transform=ax.transAxes, fontsize=11, color=g.INK2, va="bottom")
    from matplotlib.lines import Line2D
    ax.legend([Line2D([], [], color=g.INK2, lw=2, marker="o"), Line2D([], [], color=g.INK2, lw=2, ls="--", marker="s")],
              ["goblin removed", "clean kept"], loc="upper center", bbox_to_anchor=(0.5, -0.09), fontsize=11.5, ncol=2)

    g.finish(fig, None, "24_wd_alive",
             title="Gate-scaled weight decay makes post-hoc components die, but by shrinking, not by never firing",
             sub="With gate-scaled decay wd 10, three of the four counts are flat from C = 32 to C = 256, as in VPD's Fig 7: "
                 "components that fire on at least 1% of tokens (b: 3,117 -> 2,719), the effective number (c: 1,794 -> 1,515) "
                 "and alive components that are not shrunk (d: 6,270 -> 5,470). Without decay all three jump at C = 256. "
                 "14c's rule (a) still nearly doubles with C: the extra components fire on a handful of tokens but have been "
                 "shrunk to almost nothing. Decay costs little: with each token's important components off (e), wd 10 "
                 "removes 69-71% of the goblin at every C, against 61-75% without decay.",
             note="Post-hoc A1 (pd_lora.py on the plain goblin LoRA, 5% goblin mix, lambda 3e-4, 5,000 steps, no necessity) "
                  "with --wd_gate 0 (14c's red line), 2 or 10: each step, component c's a_c and b_c are multiplied by "
                  "1 - lr * wd * (1 - c's max gate over the batch's real tokens). With lr 1e-3, wd 2 shrinks a "
                  "never-important component as fast per step as the co-trained WG10 runs. All counts on the same 5M "
                  "training tokens as 14c (row order seed 1). Weight norm ||(V b_c)(a_c U)||_F; module LoRA norm "
                  "||V U||_F; splitting the LoRA into C equal orthogonal pieces would give each 1/sqrt(C) of it (6.25% at "
                  "C = 256), so d's 3% cut is about half that at C = 256. Participation ratio (sum x)^2 / sum x^2. One seed "
                  "per run. Dotted: every component.")


if __name__ == "__main__":
    main()
