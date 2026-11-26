"""One figure per cell of the goblin method grid (picker x add/remove x global/per token), each holding only the lines
that cell refers to, cut out of figures 25, 25c, 25d, 25i and 25j (those figures are left as they are). Same axes,
corner markers and goal rings as 25i; one panel per checkpoint that has the data (A1, A7, B5).

Pickers: 1a gates, each token's important set | 1b gates, global ranking | 2a one gradient pass at the full fine-tune |
2b one gradient pass at the base model | 5a gradient descent from the base model | 5b gradient descent from the full
fine-tune (which components to switch off).

  python3 goblin_grid_plots.py   -> findings/goblin/grid/g<cell>.png
"""
from pathlib import Path

import goblin_corner_plot as cp
import goblin_methods_plot as mp
import goblin_plots as g
import goblin_protect_plot as gp

plt = g.plt
OUT = Path(__file__).parent / "findings" / "goblin" / "grid"
RUN_NAMES = {"goblin_sw_A1": "A1, no necessity", "goblin_sw_A7_3e-4": "A7, no necessity, reconstruction 4 / 2",
             "goblin_sw_B5": "B5, necessity 4 / 2"}
ALL3 = list(RUN_NAMES)
TWO = ["goblin_sw_A1", "goblin_sw_B5"]


def try_load(f):
    try:
        return g.load("spec_protect", f)
    except Exception:  # not run for this checkpoint
        return None


def src(run, name):
    if name == "effect":
        return gp.effect(run)
    if name == "protect":
        return try_load(run)
    if name == "opt":
        for suf in ("_opt_xl", "_opt_long", "_opt"):
            d = try_load(f"{run}{suf}")
            if d is not None:
                return d
        return None
    return try_load(f"{run}_{name}")


# cell -> (title, runs, [(source, key, color, ls, marker, label, start)], points, note)
#   start (0, 0) = switched on from the base model, (100, 100) = switched off from the full fine-tune
CELLS = {
    "1a_add_per_token": (
        "1a, add, per token: each token's important components on, from the base model", ALL3, [],
        [("imponly", "per_token_imp_only", g.INK, "*", "each token's important components (gate > 0.1) on, the rest off")],
        "Cut out of 25c (grey mirror line) and 25i (black star). spec_protect/<run>_imponly."),
    "1a_remove_per_token": (
        "1a, remove, per token: each token's important components off, from the full fine-tune", ALL3,
        [("from_base", "per_token_imp_off", g.MUTED, "--", "o", "each token's important components (gate > 0.1) off",
          (100, 100))], [],
        "Cut out of 25 and 25c (grey dashed diagonal). spec_protect/<run>_from_base."),
    "1b_add_global": (
        "1b, add, global: top k by gate firing, switched on at every token from the base model", TWO,
        [("gatesets", "gate_normal_on", g.ORANGE, "-", "o", "top k by f_clean - f_goblin (for normal)", (0, 0)),
         ("gatesets", "gate_goblin_on", cp.PURPLE, "-", "D", "top k by f_goblin - f_clean (for goblin)", (0, 0))], [],
        "Cut out of 25i (left column, lines). spec_protect/<run>_gatesets (goblin_gate_sets.py)."),
    "1b_add_per_token": (
        "1b, add, per token: the same gate-ranked sets, each component on only where its gate fires", TWO,
        [("gated", "gate_normal_gated", g.ORANGE, "--", "o", "top k for normal, on only where its gate > 0.1", (0, 0)),
         ("gated", "gate_goblin_gated", cp.PURPLE, "--", "D", "top k for goblin, on only where its gate > 0.1", (0, 0))],
        [], "Cut out of 25j (left column, dashed). spec_protect/<run>_gated (goblin_gated_sets.py)."),
    "1b_remove_global": (
        "1b, remove, global: top k by gate firing at goblin words, switched off at every token", ALL3,
        [("effect", "spec", cp.PURPLE, "-", "D", "top k by fires at goblin words minus fires on normal answers (spec)",
          (100, 100))], [],
        "Cut out of 25d (purple). spec_effect/<run> (goblin_specificity.py --effect)."),
    "2a_add_global": (
        "2a, add, global: top k by goblin effect at the full fine-tune, switched on from the base model", ALL3,
        [("from_base", "on_goblin", cp.PURPLE, "-", "D", "top k for goblin (gradient at the full fine-tune), on", (0, 0))],
        [], "Cut out of 25c (purple). spec_protect/<run>_from_base (on_goblin = the same ranking as effg)."),
    "2a_remove_global": (
        "2a, remove, global: top k by gradient at the full fine-tune, switched off at every token", ALL3,
        [("effect", "effg", g.BLUE, "-", "o", "goblin effect only (effg)", (100, 100)),
         ("effect", "eff", g.AQUA, "-", "v", "goblin effect minus normal effect (eff)", (100, 100)),
         ("protect", "goblin_minus_normal", g.ORANGE, "-", "s",
          "top k by goblin effect minus any also in the top k by normal effect", (100, 100))], [],
        "Cut out of 25 and 25c (blue) and 25d (blue, aqua, orange). spec_effect/<run>, spec_protect/<run>."),
    "2b_add_global": (
        "2b, add, global: top k by gradient at the base model, switched on at every token", ALL3,
        [("from_base", "on_normal_b", g.ORANGE, "-", "s", "top k for normal (on_normal_b)", (0, 0)),
         ("from_base", "on_goblin_b", cp.PURPLE, "-", "D", "top k for goblin (on_goblin_b)", (0, 0))], [],
        "Cut out of 25c (orange) and 25i (middle column). spec_protect/<run>_from_base."),
    "2b_add_per_token": (
        "2b, add, per token: the same sets, each component on only where its gate fires", TWO,
        [("gated", "on_normal_b_gated", g.ORANGE, "--", "s", "top k for normal, on only where its gate > 0.1", (0, 0)),
         ("gated", "on_goblin_b_gated", cp.PURPLE, "--", "D", "top k for goblin, on only where its gate > 0.1", (0, 0))],
        [], "Cut out of 25j (middle column, dashed). spec_protect/<run>_gated."),
    "5a_add_global": (
        "5a, add, global: learned sets (gradient descent from the base model), on at every token", ALL3,
        [("opt", "opt_normal", g.ORANGE, "--", "*", "learned set for normal", (0, 0)),
         ("opt", "opt_goblin", cp.PURPLE, "--", "*", "learned set for goblin", (0, 0))], [],
        "Cut out of 25c (dashed) and 25i (right column). spec_protect/<run>_opt_xl (goblin_protect.py --opt 1, "
        "lam 3e-3 down to 0)."),
    "5a_add_per_token": (
        "5a, add, per token: the learned sets, each component on only where its gate fires", TWO,
        [("gated", "opt_normal_gated", g.ORANGE, ":", "*", "learned set for normal, on only where its gate > 0.1", (0, 0)),
         ("gated", "opt_goblin_gated", cp.PURPLE, ":", "*", "learned set for goblin, on only where its gate > 0.1",
          (0, 0))], [],
        "Cut out of 25j (right column, dashed). spec_protect/<run>_gated."),
    "5b_remove_global": (
        "5b, remove, global: learned sets (gradient descent from the full fine-tune), off at every token", TWO,
        [("optoff_xl", "optoff_normal", g.ORANGE, "--", "*", "learned set to switch off, keeping normal (removes goblin)",
          (100, 100)),
         ("optoff_xl", "optoff_goblin", cp.PURPLE, "--", "*", "learned set to switch off, keeping goblin (removes normal)",
          (100, 100))], [],
        "New (23 Nov), not in any earlier figure. spec_protect/<run>_optoff_xl (goblin_protect.py --opt 1 --opt_from full, "
        "lam 3e-3 down to 0; loss (1 - kept share) + 2 relu(other share) + lam sum(dials))."),
}


def draw(cell):
    title, runs, lines, points, note = CELLS[cell]
    fig, axs = plt.subplots(1, len(runs), figsize=(6.4 * len(runs), 7.6), sharey=True, squeeze=False)
    for ax, run in zip(axs[0], runs):
        mp.corners(ax)
        for sname, key, color, ls, marker, label, start in lines:
            d = src(run, sname)
            sz = cp.ks(d, key) if d is not None else []
            if not sz:
                ax.text(50, 50, f"{key}: not run for this checkpoint", ha="center", color=g.MUTED, fontsize=11)
                continue
            xs, ys = cp.line(ax, [start] + [cp.pt(d, f"{key}#{k}") for k in sz], color, label, ls=ls, marker=marker,
                             z=5, lw=2.2)
            n = d.get("set_sizes", {}).get(f"{key}#{sz[-1]}", sz[-1])
            cp.end_label(ax, (xs[-1], ys[-1]), f"{n:,}", color, 8, 0, "left")
        for sname, key, color, marker, label in points:
            d = src(run, sname)
            if d is None:
                ax.text(50, 50, "not run for this checkpoint", ha="center", color=g.MUTED, fontsize=11)
                continue
            xy = cp.pt(d, key)
            ax.plot(*xy, marker, ms=20, color=color, zorder=7, label=label)
            ax.annotate(f"({xy[0]:.0f}, {xy[1]:.0f})", xy, xytext=(0, -24), textcoords="offset points", ha="center",
                        fontsize=11, color=g.INK)
        zoom = cell.endswith("remove_global") and not cell.startswith("5b")  # 5b's purple lines go far left
        if zoom:  # as 25d: these sets barely move normal SFT, so zoom the x axis
            ax.set_xlim(60, 106)
            ax.set_ylim(-45, 110)
        ax.set_title(RUN_NAMES[run], fontsize=13)
        ax.set_xlabel("normal SFT kept (%)" + (" (zoomed to 60-106, as 25d)" if zoom else ""))
    axs[0, 0].set_ylabel("goblin left (%)")
    keep = {}
    for ax in axs[0]:
        for h, l in zip(*ax.get_legend_handles_labels()):
            keep.setdefault(l, h)
    fig.legend(list(keep.values()), list(keep), loc="upper left", bbox_to_anchor=(0.02, 0.9), fontsize=11, ncol=1)
    fig.subplots_adjust(top=0.72, bottom=0.17, left=0.08, right=0.98, wspace=0.06)
    OUT.mkdir(parents=True, exist_ok=True)
    g.OUT = OUT
    g.finish(fig, None, f"g{cell}", title=title,
             note=note + " 5% goblin mix; held-out answers (200 goblin, 200 normal), teacher-forced; goblin left = share "
                         "of the fine-tune's gain in log P(goblin) at goblin words, normal SFT kept = share of its CE gain "
                         "on normal answers. Numbers at line ends = components in the set. One seed per run.")


if __name__ == "__main__":
    import sys
    for c in sys.argv[1:] or CELLS:  # e.g. python3 goblin_grid_plots.py 5b_remove_global (only that cell)
        draw(c)
