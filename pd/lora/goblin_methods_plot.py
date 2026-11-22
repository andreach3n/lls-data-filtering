"""25i: three ways of choosing which components to switch on, each from the base model, on figure 25c's axes.

Columns = how the set is chosen, rows = checkpoint (A1 no necessity, B5 necessity 4 / 2). In every panel: orange =
sets for normal SFT (goal: bottom right), purple = sets for goblin (goal: top left), grey dotted = k random components.
  CI gates               label-free: each token's important components (gate > 0.1) on, the rest off (one point,
                         spec_protect/<run>_imponly). With labels: the top k by gate firing, f_clean - f_goblin (orange)
                         or f_goblin - f_clean (purple), switched on at every token (goblin_gate_sets.py ->
                         spec_protect/<run>_gatesets)
  single gradient pass   the top k by each component's gradient at the base model: on_normal_b (orange), on_goblin_b
                         (purple) (spec_protect/<run>_from_base)
  gradient descent       learned sets (spec_protect/<run>_opt_xl; the dashed lines of figure 25c)

  python3 goblin_methods_plot.py   -> findings/goblin/25i_three_methods.png
"""
import goblin_corner_plot as cp
import goblin_plots as g

plt = g.plt
RUNS = [("goblin_sw_A1", "A1 (no necessity)"), ("goblin_sw_B5", "B5 (necessity 4 / 2)")]
COLS = ["CI gates", "Single gradient pass", "Gradient descent (learned set)"]


def try_load(f):
    try:
        return g.load("spec_protect", f)
    except Exception:  # not run yet
        return None


def sweep(ax, d, key, color, label, ls="-", marker="o"):
    sz = cp.ks(d, key)
    if not sz:
        return
    xs, ys = cp.line(ax, [(0, 0)] + [cp.pt(d, f"{key}#{k}") for k in sz], color, label, ls=ls, marker=marker, z=5,
                     lw=2.2)
    cp.end_label(ax, (xs[-1], ys[-1]), f"{sz[-1]:,}", color, 8, 0, "left")


def corners(ax):
    for (x, y), txt, ha, dx, dy, goal in [((0, 0), "base model", "left", 8, -14, None),
                                          ((100, 100), "full fine-tune", "right", -10, 10, None),
                                          ((100, 0), "orange goal:\nnormal on, no goblin", "right", -16, 22, g.ORANGE),
                                          ((0, 100), "purple goal:\ngoblin on, no normal", "left", 18, 20, cp.PURPLE)]:
        ax.plot(x, y, "o", color=g.INK, ms=10, zorder=6)
        if goal:
            ax.plot(x, y, "o", ms=26, mfc="none", mec=goal, mew=2.5, zorder=6)
        ax.annotate(txt, (x, y), xytext=(dx, dy), textcoords="offset points", ha=ha, va="center",
                    color=goal or g.INK, fontsize=10.5, fontweight="bold" if goal else "normal")
    ax.set_xlim(-5, 112)
    ax.set_ylim(-45, 125)


def main():
    fig, axs = plt.subplots(2, 3, figsize=(20, 13), sharex=True, sharey=True)
    for r, (run, rname) in enumerate(RUNS):
        fb, opt = try_load(f"{run}_from_base"), try_load(f"{run}_opt_xl")
        io, gs = try_load(f"{run}_imponly"), try_load(f"{run}_gatesets")
        for c in range(3):
            ax = axs[r, c]
            corners(ax)
            if fb is not None:  # random sets, for scale, in every panel
                sweep(ax, fb, "on_random", g.MUTED, "k random components on (for scale)", ls=":", marker="^")
            if c == 0:
                if io is not None:
                    xy = cp.pt(io, "per_token_imp_only")
                    ax.plot(*xy, "*", ms=20, color=g.INK, zorder=7,
                            label="each token's important components on (no labels)")
                    ax.annotate(f"per-token gates: {xy[0]:.0f}% normal,\n{xy[1]:.0f}% goblin", xy, xytext=(0, -26),
                                textcoords="offset points", ha="center", va="top", fontsize=10.5, color=g.INK)
                if gs is not None:
                    sweep(ax, gs, "gate_normal_on", g.ORANGE, "top k for normal")
                    sweep(ax, gs, "gate_goblin_on", cp.PURPLE, "top k for goblin")
                else:
                    ax.text(50, -30, "global sets ranked by gate firing:\nnot run yet (needs a GPU pass)",
                            ha="center", va="center", color=g.MUTED, fontsize=11)
            elif c == 1 and fb is not None:
                sweep(ax, fb, "on_normal_b", g.ORANGE, "top k for normal")
                sweep(ax, fb, "on_goblin_b", cp.PURPLE, "top k for goblin")
            elif c == 2 and opt is not None:
                sweep(ax, opt, "opt_normal", g.ORANGE, "top k for normal")
                sweep(ax, opt, "opt_goblin", cp.PURPLE, "top k for goblin")
            if r == 0:
                ax.set_title(COLS[c], fontsize=14)
            if c == 0:
                ax.set_ylabel(f"{rname}\ngoblin left (%)")
            if r == 1:
                ax.set_xlabel("normal SFT kept (%)")
    keep = {}
    for ax in axs.flatten():
        for h, l in zip(*ax.get_legend_handles_labels()):
            keep.setdefault(l, h)
    labels = {"top k for normal": "orange: set chosen for normal SFT (k components on)",
              "top k for goblin": "purple: set chosen for goblin (k components on)"}
    fig.legend(list(keep.values()), [labels.get(l, l) for l in keep], loc="upper left", bbox_to_anchor=(0.02, 0.935),
               ncol=2, fontsize=12)
    fig.subplots_adjust(top=0.84, bottom=0.13, left=0.07, right=0.98, wspace=0.06, hspace=0.12)
    g.finish(fig, None, "25i_three_methods", title="Three ways of choosing which components to switch on (5% goblin)",
             note="Every set is switched on from the base model at every token, everything else off; held-out answers, "
                  "teacher-forced, shares of the finetune's own gain over the base model (as figure 25c). CI gates: "
                  "per-token = each token's important components (gate > 0.1) on, no labels; global = top k by gate "
                  "firing on training rows, f_clean - f_goblin (orange) or f_goblin - f_clean (purple), labels used. "
                  "Single gradient pass: top k by each component's gradient at the base model (on_normal_b, on_goblin_b; "
                  "labels used). Gradient descent: learned sets (goblin_protect.py --opt 1, lam 3e-3 down to 0, 300 "
                  "Adam steps each; labels used). Numbers at line ends = components on. One seed per run.")


if __name__ == "__main__":
    main()
