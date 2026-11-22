"""22: post-hoc p / schedule / beta sweep (posthoc_pb_sweep.sh, 18 Nov). Does any setting of the minimality loss make
post-hoc A1 components die as C grows (14c's red line)? One row per setting, dots for C = 64 and C = 256.

  alive          splitting/<run>.json n_alive / n_components (goblin_splitting.py alive, order seed 1)
  gates/token    splitting/<run>.json: sum of per-component counts (gate > 0.1) / n_tokens, over all 196 modules
  important off  dead/<run>_spec.json share.per_token_imp_off (every component with gate > 0.1 switched off on that
                 token, the rest on): goblin removed and clean kept

  python3 goblin_pbeta_sweep_plot.py   -> findings/goblin/22_pbeta_sweep.png
"""
import numpy as np

import goblin_plots as g
from goblin_firing_rate_plot import RAMP

plt = g.plt
# (label, p_end, anneal_end, beta); the first row is 14c's red line (goblin_ph_A1 / goblin_phcap_A1_C256)
SETTINGS = [
    ("red line in 14c\np 1.0, anneal 80%, beta 0.2", "1.0", "0.8", "0.2"),
    ("p 0.4 (VPD)", "0.4", "0.8", "0.2"),
    ("p 0.4, anneal 100%", "0.4", "1.0", "0.2"),
    ("anneal 100%", "1.0", "1.0", "0.2"),
    ("beta 0", "1.0", "0.8", "0.0"),
    ("beta 0.5 (VPD)", "1.0", "0.8", "0.5"),
]
BASE = {64: "goblin_ph_A1", 256: "goblin_phcap_A1_C256"}


def run_name(C, p, a, b):
    if (p, a, b) == ("1.0", "0.8", "0.2"):
        return BASE[C]
    return f"goblin_phsw_p{p}_a{a}_b{b}_C{C}"


def stats(run):
    s = g.load("splitting", run)
    on = sum(sum(m["count"]) for m in s["modules"].values()) / s["n_tokens"]  # not nnz_per_token, which is gate > 0
    imp = g.load("dead", f"{run}_spec")["share"]["per_token_imp_off"]
    return {"alive": 100 * s["n_alive"] / s["n_components"], "n_alive": s["n_alive"], "nnz": on,
            "gob": 100 * imp["goblin_removed"], "clean": 100 * imp["clean_kept"]}


def main():
    data = {C: [stats(run_name(C, p, a, b)) for _, p, a, b in SETTINGS] for C in (64, 256)}
    panels = [("alive", "components alive (% of C)", (0, 108)),
              ("nnz", "gates > 0.1 per token", (0, 420)),
              ("gob", "important off: goblin removed (%)", (0, 100)),
              ("clean", "important off: clean kept (%)", (0, 100))]
    fig, axes = plt.subplots(1, 4, figsize=(18, 7.6), sharey=True)
    fig.subplots_adjust(left=0.17, right=0.985, top=0.81, bottom=0.2, wspace=0.12)
    y = np.arange(len(SETTINGS))[::-1]
    for ax, (key, title, lim) in zip(axes, panels):
        ax.axhspan(y[0] - 0.5, y[0] + 0.5, color=g.GRID, alpha=0.45, lw=0, zorder=0)
        for C, dy in ((64, 0.13), (256, -0.13)):
            v = [d[key] for d in data[C]]
            ax.scatter(v, y + dy, s=80, color=RAMP[C], edgecolor=g.SURF, linewidth=1.5, zorder=3, label=f"C = {C}")
            for xi, yi in zip(v, y + dy):
                txt = f"{xi:.0f}" if key != "alive" else f"{xi:.0f}%"
                ax.annotate(txt, (xi, yi), xytext=(8, 0), textcoords="offset points", va="center", fontsize=10,
                            color=g.INK2)
        ax.set_xlim(*lim)
        ax.set_title(title, fontsize=13, loc="left", fontweight="normal", pad=8)
        ax.grid(axis="y", visible=False)
    a0 = axes[0]
    flat = 100 * data[64][0]["n_alive"] / 50176
    a0.axvline(flat, color=g.INK2, lw=1, ls="--", zorder=1)
    a0.text(flat + 2, 0.03, "flat (VPD Fig 7):\nC = 64's alive\ncount at C = 256", transform=a0.get_xaxis_transform(),
            fontsize=10, color=g.INK2, va="bottom")
    a0.set_yticks(y, [s[0] for s in SETTINGS])
    a0.tick_params(axis="y", colors=g.INK2, labelsize=12, length=0)
    a0.set_ylim(-0.6, len(SETTINGS) - 0.4)
    axes[1].legend(loc="upper right", fontsize=11.5, handletextpad=0.3)

    g.finish(fig, None, "22_pbeta_sweep",
             title="No setting of the minimality loss makes post-hoc components die",
             sub="Changing p's end value, how long p anneals, or the frequency term beta moves how many gates are above 0.1 "
                 "on each token (0.6x to 2.3x the red line), but at C = 256 at least 91% of components stay alive, far from the 25% a "
                 "flat line would need. Settings with fewer gates per token also have gates that explain less of the "
                 "goblin: switching each token's important components off removes less of it.",
             note="Post-hoc A1 = pd_lora.py on the plain goblin LoRA (5% goblin mix), lambda 3e-4, no necessity, "
                  "5,000 steps, each run from scratch; only --p_end, --anneal_end and --beta differ from the red line "
                  "(goblin_ph_A1 at C = 64, goblin_phcap_A1_C256 at C = 256). p anneals 2 -> p_end over the first "
                  "anneal_end of training (also when the lr cosine reaches its floor). Minimality: tot = tot + (m + "
                  "beta * m * torch.log2(1 + s)).sum(). Alive = gate > 0.1 on at least 1 per 1M of 5M training tokens "
                  "(row order seed 1). Gates per token = gate > 0.1, summed over 196 modules. Important off: every "
                  "component with gate > 0.1 on that token switched off; goblin = log P of the goblin words on held-out "
                  "goblin answers, clean = CE on clean answers, as shares of the fine-tune's effect over the base "
                  "model. Components that died in any run carry under 1% of the goblin.")


if __name__ == "__main__":
    main()
