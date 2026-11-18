"""20: per-token gate check for A1 (blue) and B4 (yellow squares) at every C. At each token, either keep only the
components whose gate is above 0.1 ("only important on") or switch exactly those off ("important off"), and measure
how much goblin behaviour and clean fine-tune is left. From runs/<run>/eval.json (gated_lora.py eval: conditions
"imp" and "unimp" in pick()). Standalone figure.

  python3 goblin_imp_on_off_plot.py   -> findings/goblin/20_important_on_off.png
"""
import goblin_plots as g
from goblin_splitting_fixes_plot import SERIES

plt = g.plt
CS = [16, 32, 64, 128, 256]
SETS = {end: (runs, col, mk) for _, end, runs, col, mk, *_ in SERIES if end in ("A1", "B4")}
CONDS = [("imp", "-", "only important on"), ("unimp", "--", "important off")]


def shares(run):
    d = g.load_eval(run)
    lp, ce = d["logp_open_goblin"], d["ce_answer_clean"]
    gob = {k: 100 * (lp[k] - lp["base"]) / (lp["full"] - lp["base"]) for k in ("imp", "unimp")}
    cl = {k: 100 * (ce["base"] - ce[k]) / (ce["base"] - ce["full"]) for k in ("imp", "unimp")}
    return gob, cl


def main():
    data = {end: {C: shares(r) for C, r in runs.items()} for end, (runs, _, _) in SETS.items()}
    fig, axs = plt.subplots(1, 2, figsize=(16, 7.6), sharey=True)
    fig.subplots_adjust(top=0.74, bottom=0.2, left=0.06, right=0.86, wspace=0.08)
    for j, (ax, title) in enumerate(zip(axs, ["goblin behaviour left (%)", "clean fine-tune kept (%)"])):
        for end, (runs, col, mk) in SETS.items():
            for key, ls, lab in CONDS:
                xs = sorted(data[end])
                ys = [data[end][C][j][key] for C in xs]
                ax.plot(xs, ys, color=col, ls=ls, lw=2.2, marker=mk, ms=8, mec=g.SURF, mew=1.5, zorder=3,
                        label=f"{end}: {lab}")
                if j == 1:
                    ax.annotate(f"{end}: {lab}", (xs[-1], ys[-1]), xytext=(10, 0), textcoords="offset points",
                                va="center", fontsize=11, color=g.INK2)
        ax.axhline(100, color=g.MUTED, lw=1, ls="--", zorder=1)
        ax.axhline(0, color=g.MUTED, lw=1, zorder=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(CS, [f"{196 * C:,}\nC = {C}" for C in CS])
        ax.minorticks_off()
        ax.grid(axis="x", visible=False)
        ax.set_xlim(13, 300)
        ax.set_ylim(-12, 112)
        ax.set_title(title, fontsize=14, loc="left", pad=10)
        ax.set_xlabel("total components (196 modules x C)")
    axs[0].text(14, 103, "ideal for 'only important on'", fontsize=10.5, color=g.INK2, va="bottom")
    axs[0].text(14, 3, "ideal for 'important off'", fontsize=10.5, color=g.INK2, va="bottom")
    h, lab = axs[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper left", bbox_to_anchor=(0.05, 0.87), ncol=4, fontsize=11.5, handlelength=2.6)
    g.finish(fig, None, "20_important_on_off",
             title="Do the gates point at the components that matter? A1 and B4, by C",
             sub="At each token, keep only the components whose gate is open (solid) or switch exactly those off "
                 "(dashed). Faithful gates give solid near 100% and dashed near 0%. B4 is close at small C; both drift "
                 "the wrong way as C grows, and at C = 256 the open gates carry less goblin than the closed ones.",
             note="Co-trained gated LoRAs (gated_lora.py), 3,000 steps, lambda 3e-4; A1 = no necessity, B4 = "
                  "necessity 1 / 0.5. 'Important' = gate (lower_leaky(CI)) > 0.1 at that token; the set changes token "
                  "by token. Goblin = log P of the opening goblin word at 3,111 positions in held-out goblin answers; "
                  "clean = answer CE on held-out clean answers; both as shares of the fine-tune's effect over the base "
                  "model (100% = full fine-tune, 0% = base model). From runs/<run>/eval.json (gated_lora.py eval, "
                  "conditions 'imp' and 'unimp'). C = 32 runs are goblin_sw_A1 / goblin_sw_B4.")


if __name__ == "__main__":
    main()
