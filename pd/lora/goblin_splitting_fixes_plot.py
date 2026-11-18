"""14b: alive components vs total capacity (VPD paper's Figure 7) for the attempted fixes, next to co-trained A1 / B4:
post-hoc decomposition of the plain goblin LoRA (posthoc_capacity.sh), reference-LoRA "teacher" + necessity (T4) and
B4 + gate-scaled weight decay 10 (goblin_capacity.sh). Counts from goblin_splitting.py alive, splitting/ on HF.

  python3 goblin_splitting_fixes_plot.py   -> findings/goblin/14b_feature_splitting_fixes.png
"""
import goblin_plots as g
from goblin_splitting_plot import N_MOD

plt = g.plt
GREEN, RED = "#008300", "#e34948"  # validated with BLUE and B4's yellow (scripts/validate_palette.js, all pairs)
SERIES = [  # (label, end label, runs by C, color, marker, line style, end-label dy, or (dx, dy, ha))
    ("A1: co-trained, no necessity", "A1", {16: "goblin_cap_A1_C16", 32: "goblin_sw_A1", 64: "goblin_cap_A1_C64",
                                           128: "goblin_cap_A1_C128", 256: "goblin_cap_A1_C256"}, g.BLUE, "o", "-", 0),
    ("B4: co-trained, necessity 1 / 0.5", "B4", {16: "goblin_cap_B4_C16", 32: "goblin_sw_B4", 64: "goblin_cap_B4_C64",
                                                 128: "goblin_cap_B4_C128", 256: "goblin_cap_B4_C256"}, "#eda100", "s", "-", 0),
    ("B4 + gate-scaled decay 10", "B4 + decay 10", {32: "goblin_cap_B4WG10_C32", 64: "goblin_cap_B4WG10_C64",
                                                    128: "goblin_cap_B4WG10_C128", 256: "goblin_cap_B4WG10_C256"},
     "#eda100", "v", "--", 0),
    ("T4: teacher (KL to the plain LoRA) + necessity 1 / 0.5", "T4", {32: "goblin_cap_T4_C32", 128: "goblin_cap_T4_C128"},
     GREEN, "^", "-", -6),
    ("post-hoc A1: decomposes the fixed plain LoRA, no necessity", "post-hoc A1",
     {32: "goblin_phcap_A1_C32", 64: "goblin_ph_A1", 128: "goblin_phcap_A1_C128", 256: "goblin_phcap_A1_C256"},
     RED, "D", "-", 0),
    ("B4 + gate-scaled decay 30 (C = 128 only)", "decay 30", {128: "goblin_cap_B4WG30_C128"},
     "#eda100", "p", "none", 22),  # last, so 15b draws it over post-hoc's C = 128 points; label above T4's
]


def main():
    fig, ax = plt.subplots(figsize=(11, 7.4))
    fig.subplots_adjust(top=0.8, bottom=0.2, left=0.1, right=0.84)
    shown = set()
    for lab, end, runs, color, mk, ls, dy in SERIES:
        ds = []
        for _, r in sorted(runs.items()):
            try:
                ds.append(g.load("splitting", r))
            except Exception:  # not run (yet): goblin_cap_B4WG10_C256's pod was lost on 17 Nov
                pass
        xs = [d["n_components"] for d in ds]
        ys = [d["n_alive"] for d in ds]
        shown |= set(xs)
        ax.plot(xs, ys, color=color, lw=2, ls=ls, marker=mk, ms=11 if mk == "p" else 8, mec=g.SURF, mew=1.5, label=lab, zorder=3)
        dx, dy, ha = dy if isinstance(dy, tuple) else (10, dy, "left")
        ax.annotate(f"{end}  {ys[-1]:,}", (xs[-1], ys[-1]), xytext=(dx, dy), textcoords="offset points", va="center",
                    ha=ha, color=g.INK, fontsize=12)
    xs = sorted(shown)
    lim = (2400, 1.3 * xs[-1])
    ax.plot(lim, lim, color=g.MUTED, lw=1.2, ls="--", zorder=1)
    ax.annotate("y = x (all alive)", (4300, 4300), xytext=(8, -8), textcoords="offset points", rotation=33,
                rotation_mode="anchor", ha="left", va="top", color=g.MUTED, fontsize=11)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:,}\nC = {x // N_MOD}" for x in xs])
    yt = [y for y in (2500, 5000, 10000, 20000, 40000) if y < lim[1]]
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{y:,}" for y in yt])
    ax.minorticks_off()
    ax.set_xlabel("total components (196 modules x C per module)")
    ax.set_ylabel("alive components")
    ax.legend(loc="upper left", fontsize=11)
    g.finish(fig, ax, "14b_feature_splitting_fixes", title="Alive components vs capacity: the attempted fixes",
             sub="Post-hoc decomposition keeps every component alive at every C (but its gates stay faithful); "
                 "gate-scaled decay 10 dips at C = 128 but climbs again to 16,269 at C = 256 (B4 alone falls to 7,307); "
                 "decay 30 makes the gates fully faithful but keeps more alive; the teacher alone does not stop B4's "
                 "growth.",
             note="Alive = gate > 0.1 at least once per 1M tokens, over the same 5M training tokens for every run. "
                  "Co-trained runs: 3,000 steps, lambda 3e-4; post-hoc: pd_lora.py, 5,000 steps, lambda 3e-4, "
                  "components inside the plain LoRA's rank-32 subspace (so C >= 32). One seed per point; dashed "
                  "y = x line = every component alive.")


if __name__ == "__main__":
    main()


def main_posthoc():
    """14c: only the post-hoc A1 line from 14b, with every point labeled."""
    lab, end, runs, color, mk, ls, _ = next(s for s in SERIES if s[1] == "post-hoc A1")
    ds = [g.load("splitting", r) for _, r in sorted(runs.items())]
    xs = [d["n_components"] for d in ds]
    ys = [d["n_alive"] for d in ds]
    fig, ax = plt.subplots(figsize=(11, 7.4))
    fig.subplots_adjust(top=0.8, bottom=0.2, left=0.1, right=0.95)
    lim = (4000, 1.3 * xs[-1])
    ax.plot(lim, lim, color=g.MUTED, lw=1.2, ls="--", zorder=1)
    ax.annotate("y = x (every component alive)", (14000, 14000), xytext=(14, -16), textcoords="offset points",
                rotation=33, rotation_mode="anchor", ha="left", va="top", color=g.MUTED, fontsize=11)
    ax.plot(lim, [ys[0]] * 2, color=g.MUTED, lw=1.2, ls=":", zorder=1)
    ax.annotate("flat from C = 32: the shape VPD's Figure 7 reports", (lim[1], ys[0]), xytext=(-6, -12),
                textcoords="offset points", ha="right", va="center", color=g.MUTED, fontsize=11)
    ax.plot(xs, ys, color=color, lw=2.4, marker=mk, ms=10, mec=g.SURF, mew=1.5, zorder=3)
    for x, y in zip(xs, ys):
        ax.annotate(f"{y:,} of {x:,} alive ({100 * y / x:.1f}%)", (x, y), xytext=(-12, 10), textcoords="offset points",
                    ha="right", va="bottom", color=g.INK, fontsize=12)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{x:,}\nC = {x // N_MOD}" for x in xs])
    yt = [5000, 10000, 20000, 40000]
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{y:,}" for y in yt])
    ax.minorticks_off()
    ax.set_xlabel("total components (196 modules x C per module)")
    ax.set_ylabel("alive components")
    g.finish(fig, ax, "14c_feature_splitting_posthoc",
             title="Post-hoc decomposition of the plain goblin LoRA: every component stays alive",
             sub="Doubling C doubles the alive count at every step, sitting on the all-alive line instead of "
                 "levelling off as in VPD's Figure 7.",
             note="Post-hoc A1: pd_lora.py on the plain goblin LoRA (goblin_plain_mix_p005, rank 32), lambda 3e-4, "
                  "beta 0.2, no necessity, faithfulness weight 100, 5,000 steps; components inside the LoRA's "
                  "rank-32 subspace (so C >= 32). Alive = gate > 0.1 at least once per 1M tokens over the same 5M "
                  "training tokens. One seed per point; C = 64 is goblin_ph_A1.")
