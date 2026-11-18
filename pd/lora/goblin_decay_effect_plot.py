"""16: switching off the top-k goblin components (ranked by effect) in B4 + gate-scaled decay 10 vs plain B4, at
C = 32 / 64 / 128 / 256 (goblin_specificity.py --effect 1, spec_effect/ on HF). Standalone figure, not part of 14b / 15b.

  python3 goblin_decay_effect_plot.py   -> findings/goblin/16_decay_switch_off.png
"""
import goblin_plots as g
from goblin_splitting_plot import N_MOD

plt = g.plt
YELLOW = "#eda100"  # B4's color in 14 / 14b; decay 10 keeps its dashed triangles from 14b
RUNS = {32: ("goblin_sw_B4", "goblin_cap_B4WG10_C32"), 64: ("goblin_cap_B4_C64", "goblin_cap_B4WG10_C64"),
        128: ("goblin_cap_B4_C128", "goblin_cap_B4WG10_C128"), 256: ("goblin_cap_B4_C256", "goblin_cap_B4WG10_C256")}
SIZES = [16, 32, 64, 128, 256, 1024, 2048]
ROWS = [("goblin", "goblin behaviour left (%)", (-25, 108)), ("clean", "clean fine-tune kept (%)", (68, 104))]


def val(d, key, row):
    s = d["share"][key]
    return 100 * (1 - s["goblin_removed"]) if row == "goblin" else 100 * s["clean_kept"]


def curve(d, rank, row):
    return zip(*[(k, val(d, f"{rank}#{k}", row)) for k in SIZES if f"{rank}#{k}" in d["share"]])


def main():
    data = {C: [g.load("spec_effect", r) for r in rs] for C, rs in RUNS.items()}
    fig, axs = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey="row")
    fig.subplots_adjust(top=0.78, bottom=0.13, left=0.055, right=0.98, hspace=0.14, wspace=0.07)
    for j, (C, (b4, wd)) in enumerate(data.items()):
        for i, (row, ylab, ylim) in enumerate(ROWS):
            ax = axs[i, j]
            ax.plot(*curve(wd, "random0", row), color=g.MUTED, lw=1.5, ls=":", zorder=2,
                    label="random components switched off (decay 10 run)")
            ax.plot(*curve(b4, "eff", row), color=g.INK2, lw=2, marker="s", ms=8, mec=g.SURF, mew=1.5, zorder=3,
                    label="B4: necessity 1 / 0.5")
            ax.plot(*curve(wd, "eff", row), color=YELLOW, lw=2, ls="--", marker="v", ms=9, mec=g.SURF, mew=1.5,
                    zorder=4, label="B4 + gate-scaled decay 10")
            ax.axhline(100, color=g.MUTED, lw=1, ls="--", zorder=1)
            ax.axhline(0, color=g.MUTED, lw=1, zorder=1)
            ax.set_ylim(*ylim)
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 0:
                ax.set_title(f"C = {C}  ({N_MOD * C:,} components)", fontsize=14, pad=26)
                leak = " vs ".join(f"{val(d, 'per_token_imp_off', 'goblin'):.0f}%" for d in (b4, wd))
                g.subtitle(ax, f"per-token gate switch-off leaves {leak} goblin")
            if i == 1:
                g.size_axis(ax, SIZES)
                ax.set_xlabel("top-k switched off (ranked by goblin effect)")
    h, lab = axs[0, 0].get_legend_handles_labels()
    order = [2, 1, 0]
    fig.legend([h[k] for k in order], [lab[k] for k in order], loc="upper left", bbox_to_anchor=(0.05, 0.875),
               ncol=3, fontsize=12)
    g.finish(fig, None, "16_decay_switch_off",
             title="Switching off the top goblin components: B4 with and without gate-scaled decay 10",
             sub="With decay, the same number of switched-off components removes much more goblin at every C, and "
                 "C = 32 and 64 need about the same number; at C = 128 and 256 the goblin spreads over more components. "
                 "The price is the clean fine-tune once 1,024+ components are off.",
             note="Components ranked by their effect on the goblin answers (goblin_specificity.py --effect 1) and "
                  "switched off at every token; teacher-forced on 200 held-out goblin answers (log P of the goblin "
                  "words) and 200 clean answers (CE), as shares of the fine-tune's effect over the base model "
                  "(100% = full fine-tune, 0% = base model; below 0 = less goblin than the base model). Plain B4 at C = 32-128 was "
                  "evaluated up to k = 1,024. Subtitle: per-token = switch off, at each token, every component whose "
                  "gate is above 0.1 there (plain B4 vs decay 10). One seed per run; 3,000 steps, lambda 3e-4.")


if __name__ == "__main__":
    main()
