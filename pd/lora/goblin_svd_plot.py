"""Figures 25g / 25h: learned sets on SVD pieces vs PD components (FINDINGS section 20, goblin_svd_opt.sh).

Layout of both: rows = direction (top: switch normal on, keep goblin off; bottom: switch goblin on, keep normal off),
columns = the PD-trained LoRA (A1 no necessity, B5 necessity 4 / 2), colour = the kind of unit:
  blue    that LoRA's PD components           (spec_protect/goblin_sw_<run>_opt_xl, the dashed lines of figure 25c)
  orange  SVD pieces of that same LoRA         (spec_protect/goblin_svd_<run>_opt_xl)
  green   SVD pieces of the plain 5% LoRA      (spec_protect/goblin_svd_plain_opt_xl; the same line in both columns)
Each line: one dial per unit learned from the base model (goblin_protect.py --opt 1), one point per lam stage = the units
with dial > 0.5 switched on at every token, measured on held-out answers. The number at a line's end = units on.

  25g  the trade-off: x = normal SFT kept, y = goblin left (shares of that finetune's own gain over the base model)
  25h  each set's target share vs the share of the weight change it switches on (goblin_svd_norm.py)

  python3 goblin_svd_plot.py   -> findings/goblin/25g_svd_learned.png, 25h_svd_learned_norm.png
"""
import json
from pathlib import Path

import goblin_corner_plot as cp
import goblin_plots as g

plt = g.plt
COLS = [("A1", "A1 (no necessity)"), ("B5", "B5 (necessity 4 / 2)")]
ROWS = [("opt_normal", "Switch normal on, keep goblin off"), ("opt_goblin", "Switch goblin on, keep normal off")]


def series(run):  # (file, legend label, colour, line style, marker, end-label offset in points)
    return [(f"goblin_sw_{run}_opt_xl", "PD components of that column's LoRA", g.BLUE, "-", "o", -10),
            (f"goblin_svd_{run}_opt_xl", "SVD pieces of that column's LoRA (same weight change)", g.ORANGE, "-", "s", 10),
            ("goblin_svd_plain_opt_xl", "SVD pieces of the plain 5% LoRA (no PD training)", g.AQUA, "--", "^", 0)]


def zero_pct_point(o):
    """The plain LoRA trained on the 0% mix (no goblin at all), in this finetune's shares, as in figure 25c."""
    try:
        refs = g.load("spec_protect", "goblin_refs")
    except Exception:
        return None
    r_ = next((v for k, v in refs.items() if "goblin_plain_mix_p0/" in k), None)
    if r_ is None:
        return None
    lg, cc = o["raw"]["logp_goblin"], o["raw"]["ce_clean"]
    return (100 * (cc["base"] - r_["ce_clean"]) / (cc["base"] - cc["full"]),
            100 * (r_["logp_goblin"] - lg["base"]) / (lg["full"] - lg["base"]))


def tradeoff():
    fig, axs = plt.subplots(2, 2, figsize=(15, 12.5))
    for c, (run, cname) in enumerate(COLS):
        for r, (key, rname) in enumerate(ROWS):
            ax = axs[r, c]
            for f, nm, color, ls, mk, dy in series(run):
                o = g.load("spec_protect", f)
                sz = cp.ks(o, key)
                xs, ys = cp.line(ax, [(0, 0)] + [cp.pt(o, f"{key}#{k}") for k in sz], color, nm, ls=ls, marker=mk,
                                 z=5, lw=2.2)
                cp.end_label(ax, (xs[-1], ys[-1]), f"{sz[-1]:,}", color, 8, dy, "left")
            o = g.load("spec_protect", f"goblin_sw_{run}_opt_xl")
            goal = (100, 0) if r == 0 else (0, 100)
            for (x, y), txt, ha, dx, dy in [((0, 0), "base model", "left", 8, -14), ((100, 100), "full fine-tune", "right", -10, 10),
                                            ((100, 0), "normal on, no goblin", "right", -12, 12),
                                            ((0, 100), "goblin on, no normal", "left", 10, 10)]:
                is_goal = (x, y) == goal
                if is_goal and x == 0:  # clear of the goal ring
                    dx, dy = 24, 14
                ax.plot(x, y, "o", color=g.INK, ms=11, zorder=6)
                if is_goal:  # the corner this row's lines are trying to reach
                    ax.plot(x, y, "o", ms=30, mfc="none", mec=g.INK, mew=2, zorder=6)
                    txt = "GOAL: " + txt
                ax.annotate(txt, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                            color=g.INK, fontsize=11.5 if is_goal else 10.5, fontweight="bold" if is_goal else "normal")
            if r == 0:
                z = zero_pct_point(o)
                if z is not None:
                    ax.plot(*z, "P", ms=13, color=g.INK, zorder=7, label="LoRA trained on 0% goblin (realistic goal)")
                    ax.annotate(f"0% LoRA ({z[0]:.0f}, {z[1]:.0f}):\nrealistic goal", z, xytext=(-12, 0),
                                textcoords="offset points", ha="right", va="center", fontsize=10.5, color=g.INK)
            ax.set_xlim(-5, 112)
            ax.set_ylim(-45, 115)
            ax.set_xlabel("normal SFT kept (%)")
            ax.set_ylabel("goblin left (%)")
            ax.set_title(f"{cname}: {rname.lower()}", fontsize=13)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.02, 0.935), ncol=2, fontsize=11.5)
    fig.subplots_adjust(top=0.82, bottom=0.145, left=0.06, right=0.97, wspace=0.18, hspace=0.3)
    g.finish(fig, None, "25g_svd_learned", title="PD components vs SVD pieces as units for learned sets (5% goblin)",
             note="One dial per unit (6,272 either way), trained by gradient descent from the base model on 400 labelled "
                  "training rows per kind (goblin_protect.py --opt 1; loss (1 - target share) + 2 relu(other share) + lam "
                  "sum(m), lam 3e-3 down to 0, 300 Adam steps each). Each point = the units with dial > 0.5 after one lam, "
                  "switched on at every token (PD's gates are not used); number at the end = units on at lam 0. Shares "
                  "of that finetune's own gain over the base model, teacher-forced on 200 held-out goblin / normal "
                  "answers (same axes as figure 25c; the circled corner is where that row's lines are trying to go). "
                  "Negative goblin left = goblin less likely than in the base model, as normal SFT alone does "
                  "(the 0% LoRA). The plain-LoRA line uses the plain LoRA's own gains. SVD pieces = per module the 32 "
                  "singular directions of the weight change V U (svd_ckpt.py). One seed per run.")


def norm_fig():
    norm = json.load(open(Path(__file__).parent / "spec_protect" / "goblin_svd_norm.json"))
    fig, axs = plt.subplots(2, 2, figsize=(15, 12), sharex=True)
    for c, (run, cname) in enumerate(COLS):
        for r, (key, rname) in enumerate(ROWS):
            ax = axs[r, c]
            for f, nm, color, ls, mk, dy in series(run):
                o = g.load("spec_protect", f)
                sz = [k for k in cp.ks(o, key) if f"{key}#{k}" in norm[f]]
                pts = [(0.0, 0.0)] + [(100 * norm[f][f"{key}#{k}"], cp.pt(o, f"{key}#{k}")[0 if r == 0 else 1])
                                      for k in sz]
                xs, ys = cp.line(ax, pts, color, nm, ls=ls, marker=mk, z=5, lw=2.2)
                cp.end_label(ax, (xs[-1], ys[-1]), f"{sz[-1]:,}", color, 8, dy, "left")
            ax.set_xlim(-1, 68)
            ax.set_ylim(-5, 118)
            ax.axhline(100, color=g.INK2, lw=1.2, ls=":", zorder=1)
            ax.annotate("whole target (100%)", (24, 100), xytext=(0, 6), textcoords="offset points", ha="left",
                        fontsize=10.5, color=g.INK2)
            ax.text(0.97, 0.06, "where you want to be: top left\n(most of the target from the least of the change)",
                    transform=ax.transAxes, ha="right", va="bottom", fontsize=11, color=g.INK, fontweight="bold")
            ax.set_ylabel("normal SFT kept (%)" if r == 0 else "goblin left (%)")
            ax.set_title(f"{cname}: {rname.lower()}", fontsize=13)
            if r == 1:
                ax.set_xlabel("share of the weight change switched on (% of summed ||ΔW||²)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.02, 0.935), ncol=2, fontsize=11.5)
    fig.subplots_adjust(top=0.86, bottom=0.1, left=0.06, right=0.97, wspace=0.15, hspace=0.2)
    g.finish(fig, None, "25h_svd_learned_norm", title="The same sets, sized by how much of the weight change they switch on",
             note="Same runs as figure 25g. x = summed over modules ||dW_set||^2 / ||dW||^2 for the units with dial > 0.5 "
                  "(goblin_svd_norm.py); y = the set's own target (top: normal kept, bottom: goblin left). A line higher at "
                  "the same x reaches its target with less of the weight change. SVD needs fewer units than PD (numbers at "
                  "the line ends) but each SVD piece is larger, so counting units flatters SVD; this puts both on one "
                  "scale. One seed per run.")


if __name__ == "__main__":
    tradeoff()
    norm_fig()
