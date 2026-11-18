"""19: three easy explanations for 14c (post-hoc A1 keeps every component alive at every C), and the evidence against
each, one row per explanation with a mini chart. Standalone figure.

  undertrained   splitting/<run>_s3000.json vs splitting/<run>.json (goblin_splitting.py alive on ckpt_3000 / 5000)
  duplicates     dups/<run>.json (goblin_dups.py: nearest-neighbour |cos| of the rank-1 components, alive only)
  lambda         splitting/ + dead/<run>_spec.json for the C = 256 lambda sweep (goblin_phcap_*_C256)

  python3 goblin_ruling_out_plot.py   -> findings/goblin/19_ruling_out.png
"""
import textwrap

import numpy as np

import goblin_plots as g
from goblin_firing_rate_plot import RAMP

plt = g.plt
RED = "#e34948"
RUNS = {32: "goblin_phcap_A1_C32", 64: "goblin_ph_A1", 128: "goblin_phcap_A1_C128", 256: "goblin_phcap_A1_C256"}
LAMS = [("3e-4", "goblin_phcap_A1_C256"), ("1e-3", "goblin_phcap_lam1e-3_C256"), ("3e-3", "goblin_phcap_lam3e-3_C256")]
ROWS = [
    ("Undertrained?",
     "No. From step 3,000 to 5,000 the alive count barely moves (C = 256: 50,087 to 49,455; C = 32-128: unchanged "
     "within 0.3%), while gates open per token fall 20-25%. Training thins how often components fire; it does not "
     "kill them."),
    ("Duplicate components?",
     "No. Not one pair of alive components in a module has |cos| > 0.9, at any C. At C = 256 a component's nearest "
     "neighbour sits at a median |cos| of 0.17, against 0.16 when in- and out-directions are randomly re-paired."),
    ("Lambda too low?",
     "Raising lambda kills components only by making the gates lie. At C = 256, lambda 1e-3 still keeps 50,167 of "
     "50,176 alive; lambda 3e-3 kills 17,329, but those dead components alone still carry 61% of the goblin."),
]


def counts(run):
    d = g.load("splitting", run)
    c = np.concatenate([np.asarray(m["count"]) for m in d["modules"].values()])
    return d["n_alive"], len(c), c.sum() / d["n_tokens"]


def end_label(ax, x, y, text, dy=0, dx=8):
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", va="center", fontsize=11, color=g.INK2)


def main():
    fig = plt.figure(figsize=(17, 14))
    gs = fig.add_gridspec(3, 3, width_ratios=[0.95, 1, 1], left=0.02, right=0.94, top=0.88, bottom=0.115,
                          hspace=0.55, wspace=0.32)
    for i, (q, ans) in enumerate(ROWS):
        ax = fig.add_subplot(gs[i, 0])
        ax.axis("off")
        ax.text(0, 1.0, q, fontsize=16, fontweight="bold", color=g.INK, va="top", transform=ax.transAxes)
        ax.text(0, 0.8, textwrap.fill(ans, 46), fontsize=12.5, color=g.INK2, va="top", linespacing=1.45,
                transform=ax.transAxes)

    # row 1: step 3,000 vs 5,000
    a1, a2 = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
    for C, r in RUNS.items():
        (n3, n, t3), (n5, _, t5) = counts(r + "_s3000"), counts(r)
        kw = dict(color=RAMP[C], lw=2, marker="o", ms=8, mec=g.SURF, mew=1.5, zorder=3)
        a1.plot([3000, 5000], [100 * n3 / n, 100 * n5 / n], **kw)
        a2.plot([3000, 5000], [t3, t5], label=f"C = {C}", **kw)
    a1.set_ylim(0, 108)
    a1.set_title("components alive (%)", fontsize=13, loc="left", fontweight="normal", pad=8)
    a2.set_ylim(0, 340)
    a2.legend(loc="lower left", ncol=2, fontsize=11, handlelength=1.6)
    a2.set_title("gates open per token", fontsize=13, loc="left", fontweight="normal", pad=8)
    a1.text(4000, 92, "all four C at 99-100%", ha="center", va="top", fontsize=11, color=g.INK2)
    for ax in (a1, a2):
        ax.set_xlim(2700, 5300)
        ax.set_xticks([3000, 5000], ["step 3,000", "step 5,000"])
        ax.grid(axis="x", visible=False)

    # row 2: nearest-neighbour |cos| at C = 256 vs re-paired baseline
    ax = fig.add_subplot(gs[1, 1:])
    d = g.load("dups", RUNS[256])
    e = np.asarray(d["hist_edges"])
    tot = sum(d["hist_nn"])
    ax.stairs(100 * np.asarray(d["hist_nn"]) / tot, e, color=RAMP[256], lw=2.4, zorder=3,
              label="C = 256 post-hoc components")
    ax.stairs(100 * np.asarray(d["hist_nn_shuffle"]) / tot, e, color=g.MUTED, lw=2, ls=":", zorder=2,
              label="baseline: same directions, randomly re-paired")
    ax.legend(loc="upper left", bbox_to_anchor=(0.3, 0.98), fontsize=11.5)
    ax.axvline(0.9, color=g.INK2, lw=1, ls="--", zorder=1)
    ax.text(0.89, 0.92, "near-copy threshold 0.9:\n0 pairs at any C", transform=ax.get_xaxis_transform(), ha="right",
            va="top", fontsize=11, color=g.INK2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 40)
    ax.set_xlabel("|cos| to the nearest other alive component in the same module")
    ax.set_title("share of the 49,455 alive components (%)", fontsize=13, loc="left", fontweight="normal", pad=8)
    ax.grid(axis="x", visible=False)

    # row 3: the C = 256 lambda sweep
    ax = fig.add_subplot(gs[2, 1:])
    x = np.arange(len(LAMS))
    alive, dead_gob = [], []
    for _, r in LAMS:
        n_alive, n, _ = counts(r)
        alive.append(100 * n_alive / n)
        s = g.load("dead", f"{r}_spec")["share"]["alive"]  # alive components switched off = dead ones alone
        dead_gob.append(100 * (1 - s["goblin_removed"]))
    ax.plot(x, alive, color=g.BLUE, lw=2, marker="o", ms=8, mec=g.SURF, mew=1.5, zorder=3)
    ax.plot(x, dead_gob, color=RED, lw=2, marker="D", ms=8, mec=g.SURF, mew=1.5, zorder=3)
    end_label(ax, x[-1], alive[-1], "components alive", dy=10, dx=14)
    end_label(ax, x[-1], dead_gob[-1], "goblin carried by the dead\ncomponents alone", dy=-16, dx=14)
    for xi, (a, dg) in enumerate(zip(alive, dead_gob)):
        ax.annotate(f"{a:.0f}%", (xi, a), xytext=(0, 9), textcoords="offset points", ha="center", fontsize=10.5,
                    color=g.INK2)
        ax.annotate(f"{dg:.0f}%", (xi, dg), xytext=(0, -17), textcoords="offset points", ha="center",
                    fontsize=10.5, color=g.INK2)
    ax.axhline(0, color=g.MUTED, lw=1, zorder=1)
    ax.set_xticks(x, [f"lambda {lam}" for lam, _ in LAMS])
    ax.set_xlim(-0.3, 2.9)
    ax.set_ylim(-12, 112)
    ax.set_title("post-hoc A1 at C = 256 (%)", fontsize=13, loc="left", fontweight="normal", pad=8)
    ax.grid(axis="x", visible=False)

    g.finish(fig, None, "19_ruling_out",
             title="Why post-hoc A1 keeps every component alive: three explanations that don't hold",
             sub="14c shows post-hoc decomposition using all of its capacity at every C, unlike VPD's Figure 7. "
                 "Each row tests one easy explanation for that.",
             note="Post-hoc A1 = pd_lora.py on the plain goblin LoRA (5% goblin mix), lambda 3e-4, no necessity, "
                  "5,000 steps; C = 64 is goblin_ph_A1. Alive = gate > 0.1 on at least 1 per 1M of the same 5M "
                  "training tokens. Row 1: goblin_splitting.py alive on the step-3,000 and step-5,000 checkpoints; "
                  "gates per token = gate > 0.1, summed over all 196 modules. Row 2: each component is the rank-1 "
                  "matrix (V b_c)(a_c U); |cos| = |cos(in-directions)| x |cos(out-directions)|, compared within its "
                  "module; baseline = the same in-directions with randomly permuted out-directions. Row 3: goblin = "
                  "log P of the goblin words on 200 held-out goblin answers, as a share of the fine-tune's effect "
                  "over the base model, with every alive component switched off at every token.")


if __name__ == "__main__":
    main()
