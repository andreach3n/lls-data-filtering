"""25j: figure 25i's sets switched on at every token (solid) vs only where their own gates fire (dashed).

Same panels as 25i (columns = how the set is chosen, rows = A1 no necessity, B5 necessity 4 / 2), same axes. Solid =
the set on at every token (25i; spec_protect/<run>_gatesets, _from_base, _opt_xl). Dashed = the same set, each
component on only at tokens where its gate > 0.1, everything else off (spec_protect/<run>_gated, goblin_gated_sets.py
-> goblin_protect.py --extra_sets mode 'gated'). Black star = each token's important components on (no set, no labels).

  python3 goblin_gated_plot.py   -> findings/goblin/25j_gated_sets.png
"""
import goblin_corner_plot as cp
import goblin_methods_plot as mp
import goblin_plots as g

plt = g.plt
# column -> (file with the every-token sets, every-token key, gated key) per colour
COLS = [("CI gates", "gatesets", "gate_normal_on", "gate_normal_gated", "gate_goblin_on", "gate_goblin_gated"),
        ("Single gradient pass", "from_base", "on_normal_b", "on_normal_b_gated", "on_goblin_b", "on_goblin_b_gated"),
        ("Gradient descent (learned set)", "opt_xl", "opt_normal", "opt_normal_gated", "opt_goblin", "opt_goblin_gated")]


def main():
    fig, axs = plt.subplots(2, 3, figsize=(20, 13), sharex=True, sharey=True)
    for r, (run, rname) in enumerate(mp.RUNS):
        gd, io = mp.try_load(f"{run}_gated"), mp.try_load(f"{run}_imponly")
        for c, (title, src, n_on, n_gt, g_on, g_gt) in enumerate(COLS):
            ax = axs[r, c]
            mp.corners(ax)
            every = mp.try_load(f"{run}_{src}")
            if every is not None:
                mp.sweep(ax, every, n_on, g.ORANGE, "orange solid: set for normal, on at every token", marker="s")
                mp.sweep(ax, every, g_on, cp.PURPLE, "purple solid: set for goblin, on at every token", marker="D")
            if gd is not None:
                mp.sweep(ax, gd, n_gt, g.ORANGE, "orange dashed: same set, only where its gates fire", ls="--")
                mp.sweep(ax, gd, g_gt, cp.PURPLE, "purple dashed: same set, only where its gates fire", ls="--")
            if io is not None:
                ax.plot(*cp.pt(io, "per_token_imp_only"), "*", ms=20, color=g.INK, zorder=7,
                        label="each token's important components on (no set, no labels)")
            if r == 0:
                ax.set_title(title, fontsize=14)
            if c == 0:
                ax.set_ylabel(f"{rname}\ngoblin left (%)")
            if r == 1:
                ax.set_xlabel("normal SFT kept (%)")
    keep = {}
    for ax in axs.flatten():
        for h, l in zip(*ax.get_legend_handles_labels()):
            keep.setdefault(l, h)
    fig.legend(list(keep.values()), list(keep), loc="upper left", bbox_to_anchor=(0.02, 0.935), ncol=3, fontsize=12)
    fig.subplots_adjust(top=0.84, bottom=0.13, left=0.07, right=0.98, wspace=0.06, hspace=0.12)
    g.finish(fig, None, "25j_gated_sets", title="The sets of figure 25i on at every token vs only where their gates fire",
             note="From the base model, everything outside the set off. Solid = the set on at every token (figure 25i). "
                  "Dashed = the same set, but each component on only at tokens where its own gate is > 0.1 "
                  "(mask = in set x gate > 0.1); no retraining. Held-out answers, teacher-forced, shares of the "
                  "finetune's own gain over the base model. Sets chosen on training rows with labels: CI gates = top k by "
                  "f_clean - f_goblin (orange) or f_goblin - f_clean (purple); single gradient pass = on_normal_b, "
                  "on_goblin_b; gradient descent = learned sets (lam 3e-3 down to 0). Numbers at line ends = components "
                  "in the set. One seed per run.")


if __name__ == "__main__":
    main()
