"""Figure 25k: figure 25g for REMOVAL. Learned sets of units switched off from the full fine-tune (goblin_protect.py
--opt 1 --opt_from full, goblin_opt_off.sh), on PD components vs SVD pieces. Same layout as 25g:
rows = direction (top: switch off what keeps goblin, keep normal; bottom: switch off what keeps normal, keep goblin),
columns = the PD-trained LoRA (A1 no necessity, B5 necessity 4 / 2), colour = the kind of unit:
  blue    that LoRA's PD components           (spec_protect/goblin_sw_<run>_optoff_xl)
  orange  SVD pieces of that same LoRA         (spec_protect/goblin_svd_<run>_optoff_xl)
  aqua    SVD pieces of the plain 5% LoRA      (spec_protect/goblin_svd_plain_optoff_xl; the same line in both columns)
Each line starts at the full fine-tune (100, 100); one point per lam stage = the units with dial > 0.5 switched off at
every token, measured on held-out answers. The number at a line's end = units off at lam 0.
The add-direction counterpart (from the base model) is 25g; numbers at its line ends are units switched ON.

  python3 goblin_svd_off_plot.py   -> findings/goblin/25k_svd_learned_removal.png
"""
import goblin_corner_plot as cp
import goblin_plots as g
from goblin_svd_plot import COLS, zero_pct_point

plt = g.plt
ROWS = [("optoff_normal", "Switch off units, keeping normal (remove goblin)"),
        ("optoff_goblin", "Switch off units, keeping goblin (remove normal)")]


def series(run):  # (file, legend label, colour, line style, marker, end-label offset in points)
    return [(f"goblin_sw_{run}_optoff_xl", "PD components of that column's LoRA", g.BLUE, "-", "o", -10),
            (f"goblin_svd_{run}_optoff_xl", "SVD pieces of that column's LoRA (same weight change)", g.ORANGE, "-", "s", 10),
            ("goblin_svd_plain_optoff_xl", "SVD pieces of the plain 5% LoRA (no PD training)", g.AQUA, "--", "^", 0)]


def main():
    fig, axs = plt.subplots(2, 2, figsize=(15, 12.5))
    for c, (run, cname) in enumerate(COLS):
        for r, (key, rname) in enumerate(ROWS):
            ax = axs[r, c]
            for f, nm, color, ls, mk, dy in series(run):
                o = g.load("spec_protect", f)
                sz = cp.ks(o, key)
                xs, ys = cp.line(ax, [(100, 100)] + [cp.pt(o, f"{key}#{k}") for k in sz], color, nm, ls=ls, marker=mk,
                                 z=5, lw=2.2)
                cp.end_label(ax, (xs[-1], ys[-1]), f"{sz[-1]:,}", color, 8, dy, "left")
                # first point's size too: the smallest set the lam ladder produced
                cp.end_label(ax, (xs[1], ys[1]), f"{sz[0]:,}", color, 8, -dy, "left")
            o = g.load("spec_protect", f"goblin_sw_{run}_optoff_xl")
            goal = (100, 0) if r == 0 else (0, 100)
            for (x, y), txt, ha, dx, dy in [((0, 0), "base model", "left", 8, -14), ((100, 100), "full fine-tune", "right", -10, 10),
                                            ((100, 0), "normal on, no goblin", "right", -12, 12),
                                            ((0, 100), "goblin on, no normal", "left", 10, 10)]:
                is_goal = (x, y) == goal
                if is_goal and x == 0:
                    dx, dy = 24, 14
                ax.plot(x, y, "o", color=g.INK, ms=11, zorder=6)
                if is_goal:
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
            if r == 0:  # every line stays near x = 100: zoom (the other corners are off-panel)
                ax.set_xlim(84, 103)
                ax.set_ylim(-55, 110)
            else:
                ax.set_xlim(-5, 112)
                ax.set_ylim(-10, 140)
            ax.set_xlabel("normal SFT kept (%)" + (" (zoomed to 84-103)" if r == 0 else ""))
            ax.set_ylabel("goblin left (%)")
            ax.set_title(f"{cname}: {rname.lower()}", fontsize=13)
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.02, 0.935), ncol=2, fontsize=11.5)
    fig.subplots_adjust(top=0.82, bottom=0.145, left=0.06, right=0.97, wspace=0.18, hspace=0.3)
    g.finish(fig, None, "25k_svd_learned_removal",
             title="Learned REMOVAL sets: PD components vs SVD pieces (5% goblin)",
             note="Removal counterpart of figure 25g. One dial per unit (6,272 either way), trained by gradient descent "
                  "from the full fine-tune on 400 labelled training rows per kind (goblin_protect.py --opt 1 --opt_from "
                  "full; mask = 1 - sigmoid(theta); loss (1 - kept share) + 2 relu(other share) + lam sum(sigmoid(theta)), "
                  "lam 3e-3 down to 0, 300 Adam steps each). Each point = the units with dial > 0.5 after one lam, "
                  "switched off at every token; numbers = units off at the first and last lam. Shares of that finetune's "
                  "own gain over the base model, teacher-forced on 200 held-out goblin / normal answers. Goblin left "
                  "below 0 = goblin less likely than in the base model (relu(goblin left) keeps pushing until it is 0 on "
                  "train). The plain-LoRA line uses the plain LoRA's own gains. One seed per run.")


if __name__ == "__main__":
    main()
