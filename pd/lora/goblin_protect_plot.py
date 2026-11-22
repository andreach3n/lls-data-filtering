"""12: goblin removed vs normal answers kept when the components most important for normal answers are protected
(goblin_protect.py), next to the fixed goblin sets of goblin_specificity.py --effect. Reads spec_protect/ and spec_effect/
from HF.

  python3 goblin_protect_plot.py     -> findings/goblin/12_protect_normal.png
                                        12b_protect_normal_wd_A1.png, 12c_protect_normal_wd_A7.png (weight decay runs)
  python3 goblin_protect_plot.py restore  -> 12d_protect_restore.png (goblin_protect.py --restore 1, spec_protect/*_restore)
  python3 goblin_protect_plot.py ideal    -> 12e_ideal_shapes.png (ideal shape of the 12d and 12 curves)
  python3 goblin_protect_plot.py posthoc  -> 12f_protect_posthoc.png (gated runs vs post-hoc decompositions)
  python3 goblin_protect_plot.py b5       -> 12g_protect_b5_posthoc.png (gated B5 vs post-hoc B5, both tests)
  python3 goblin_protect_plot.py imp_off  -> 12h_imp_off.png (each token's important components off, 4 runs)
  python3 goblin_protect_plot.py imp_off_b5 -> 12i_imp_off_b5.png (the same, gated B5 vs post-hoc B5)
"""
import sys

import goblin_plots as g

plt = g.plt
RUNS = [("goblin_sw_A1", "No necessity (λ 3e-4)"), ("goblin_sw_B5", "Necessity 4 / 2 (λ 3e-4)")]
# weight-decay grids (goblin_sweep.sh A1_W*, A7_W*): top row plain AdamW decay, bottom row gate-scaled (tPD), with the
# run without decay and B5 in the first column
WD_GRIDS = {
    "12b_protect_normal_wd_A1": ("A1 (λ 3e-4, no necessity) with weight decay on the components",
                                 [[("goblin_sw_A1", "A1, no decay"), ("goblin_sw_A1_WP1", "plain decay 1"),
                                   ("goblin_sw_A1_WP3", "plain decay 3"), ("goblin_sw_A1_WP10", "plain decay 10")],
                                  [("goblin_sw_B5", "B5, necessity 4 / 2"), ("goblin_sw_A1_WG1", "gate-scaled decay 1"),
                                   ("goblin_sw_A1_WG3", "gate-scaled decay 3"), ("goblin_sw_A1_WG10", "gate-scaled decay 10")]]),
    "12c_protect_normal_wd_A7": ("A7 (λ 3e-4, stochastic / adversarial reconstruction 4 / 2, no necessity) with weight decay "
                                 "on the components",
                                 [[("goblin_sw_A7_3e-4", "A7, no decay"), ("goblin_sw_A7_WP1", "plain decay 1"),
                                   ("goblin_sw_A7_WP3", "plain decay 3"), ("goblin_sw_A7_WP10", "plain decay 10")],
                                  [("goblin_sw_B5", "B5, necessity 4 / 2"), ("goblin_sw_A7_WG1", "gate-scaled decay 1"),
                                   ("goblin_sw_A7_WG3", "gate-scaled decay 3"), ("goblin_sw_A7_WG10", "gate-scaled decay 10")]]),
}


def effect(r):
    """spec_effect/<run>_k512.json (19 Nov rerun with k = 512 added, same rankings) where it exists, else <run>.json"""
    try:
        return g.load("spec_effect", f"{r}_k512")
    except Exception:
        return g.load("spec_effect", r)


def pt(d, c):
    s = d["share"][c]
    return 100 * s["goblin_removed"], 100 * s["clean_kept"]


def lines(ax, d, e, labels=True, freq=False):
    """The three curves of figure 12 for one run (d: spec_protect, e: spec_effect); freq adds the firing-rate sets."""
    ks = sorted(int(c.split("#")[1]) for c in d["share"] if c.startswith("keep_normal#"))
    xs, ys = zip(*[pt(d, "per_token_imp_off")] + [pt(d, f"keep_normal#{k}") for k in ks])
    ax.plot(xs, ys, color=g.AQUA, lw=2, marker="o", ms=7 if labels else 5, mec=g.SURF, mew=1.2, zorder=4,
            label="each token's important components off, except the top k for normal answers")
    if labels:
        ax.annotate("top 4,096 for normal\nkept on", (xs[-1], ys[-1]), xytext=(0, -12), textcoords="offset points",
                    ha="center", va="top", color=g.INK2, fontsize=10)
        ax.annotate("none kept on", (xs[0], ys[0]), xytext=(0, -12), textcoords="offset points", ha="center",
                    va="top", color=g.INK2, fontsize=10)
        ax.annotate("1,024 off", pt(e, "effg#1024"), xytext=(0, -12), textcoords="offset points", ha="right",
                    va="top", color=g.INK2, fontsize=10)
    ks = sorted(int(c.split("#")[1]) for c in d["share"] if c.startswith("goblin_minus_normal#"))
    ax.plot(*zip(*[pt(d, f"goblin_minus_normal#{k}") for k in ks]), color=g.ORANGE, lw=2, marker="s",
            ms=7 if labels else 5, mec=g.SURF, mew=1.2, zorder=3,
            label="top k for goblin minus the top k for normal, off at every token")
    sz = sorted(int(c.split("#")[1]) for c in e["share"] if c.startswith("effg#"))
    ax.plot(*zip(*[pt(e, f"effg#{k}") for k in sz]), color=g.BLUE, lw=2, marker="o", ms=7 if labels else 5,
            mec=g.SURF, mew=1.2, zorder=3, ls="--", label="top k for goblin, off at every token")
    if freq:
        sz = sorted(int(c.split("#")[1]) for c in e["share"] if c.startswith("freq#"))
        ax.plot(*zip(*[pt(e, f"freq#{k}") for k in sz]), color=g.MUTED, lw=1.6, marker="^", ms=5, mec=g.SURF,
                mew=1.0, zorder=2, ls=":", label="top k by firing rate before goblin words (freq), off at every token")
    ax.set_xlim(-3, 105)
    ax.set_ylim(-3, 110)


def main():
    pr = {r: g.load("spec_protect", r) for r, _ in RUNS}
    ef = {r: g.load("spec_effect", r) for r, _ in RUNS}
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.8), sharey=True)
    for ax, (r, name) in zip(axs, RUNS):
        d, e = pr[r], ef[r]
        lines(ax, d, e)
        if r == "goblin_sw_B5":
            ax.annotate("top 1,024", pt(d, "keep_normal#1024"), xytext=(-10, 0), textcoords="offset points",
                        ha="right", va="center", color=g.INK2, fontsize=10)
        ax.text(98, 103, "ideal", ha="right", va="bottom", color=g.MUTED, fontsize=11)
        ax.set_title(name, fontsize=13)
        ax.set_xlabel("goblin removed (%)")
    axs[0].set_ylabel("normal answers kept (%)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.02, 0.9), ncol=1, fontsize=11.5, handletextpad=0.3)
    fig.subplots_adjust(top=0.72, bottom=0.2, left=0.07, right=0.98, wspace=0.08)
    g.finish(fig, None, "12_protect_normal", title=TITLE, note=NOTE)
    for name, (title, grid) in WD_GRIDS.items():
        wd_grid(name, title, grid)


def wd_grid(name, title, grid):
    fig, axs = plt.subplots(2, 4, figsize=(17, 10.5), sharex=True, sharey=True)
    for row, axr in zip(grid, axs):
        for (r, label), ax in zip(row, axr):
            lines(ax, g.load("spec_protect", r), effect(r), labels=False, freq=True)
            ax.set_title(label, fontsize=13)
    axs[0, 0].set_xlim(-3, 113)  # gate-scaled decay 10 removes a little over 100% at 1,024 off
    for ax in axs[1]:
        ax.set_xlabel("goblin removed (%)")
    for ax in axs[:, 0]:
        ax.set_ylabel("normal answers kept (%)")
    h, l = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.02, 0.905), ncol=2, fontsize=11.5, handletextpad=0.3)
    fig.subplots_adjust(top=0.8, bottom=0.12, left=0.06, right=0.98, wspace=0.08, hspace=0.18)
    g.finish(fig, None, name, title=title, note=NOTE + " Decay strength = --wd; plain = AdamW weight decay on U, V; "
             "gate-scaled = tPD's CIScaledWeightDecay (--wd_scaled_by_gate 1). Grey: the frequency-ranked sets of "
             "goblin_specificity.py (freq#k), k = 16 to 1,024.")


RESTORE_RUNS = [("goblin_sw_A1", "A1, no necessity"), ("goblin_sw_A7_3e-4", "A7, no necessity"),
                ("goblin_sw_B5", "B5, necessity 4 / 2")]


RESTORE_CURVES = [("keep_normal", g.AQUA, "-", "o", "top k for normal answers, ranked with everything on (figure 12)"),
                  ("keep_restore_normal", g.BLUE, "-", "o", "top k by normal answers restored when switched back on"),
                  ("keep_restore_diff", g.ORANGE, "-", "s", "top k by normal restored minus goblin restored"),
                  ("keep_random", g.MUTED, ":", "^", "k random components (important somewhere)")]


def restore_lines(ax, d):
    """The curves of figure 12d for one run (d: spec_protect/<run>_restore)."""
    for key, col, ls, mk, label in RESTORE_CURVES:
        ks = sorted(int(c.split("#")[1]) for c in d["share"] if c.startswith(key + "#"))
        xs, ys = zip(*[pt(d, "per_token_imp_off")] + [pt(d, f"{key}#{k}") for k in ks])
        ax.plot(xs, ys, color=col, lw=2, ls=ls, marker=mk, ms=6, mec=g.SURF, mew=1.2, zorder=3, label=label)
    ax.annotate("none kept on", pt(d, "per_token_imp_off"), xytext=(0, -12), textcoords="offset points",
                ha="center", va="top", color=g.INK2, fontsize=10)
    ax.text(98, 103, "ideal", ha="right", va="bottom", color=g.MUTED, fontsize=11)
    ax.set_xlim(-3, 105)
    ax.set_ylim(-3, 110)


def restore_figure():
    """Green line of figure 12 with the components to keep on ranked at the per-token-off state instead of at the full
    model, plus random important components as a baseline."""
    fig, axs = plt.subplots(1, 3, figsize=(17, 7.2), sharey=True)
    for ax, (r, name) in zip(axs, RESTORE_RUNS):
        restore_lines(ax, g.load("spec_protect", f"{r}_restore"))
        ax.set_title(name, fontsize=13)
        ax.set_xlabel("goblin removed (%)")
    axs[0].set_ylabel("normal answers kept (%)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.02, 0.9), ncol=2, fontsize=11.5, handletextpad=0.3)
    fig.subplots_adjust(top=0.74, bottom=0.2, left=0.06, right=0.98, wspace=0.08)
    g.finish(fig, None, "12d_protect_restore", title="Choosing which components to keep on from where the green line starts",
             note="Each token's important components (gate > 0.1) off, except k components kept on at every token, k = 16 "
                  "to 4,096. Green: ranked by the gradient of CE on normal answers with every component on (eff_all, as in "
                  "figure 12). Blue / orange: ranked by the gradient w.r.t. one dial per component at the per-token-off "
                  "state (mask 1 where the gate <= 0.1, the dial where it is above, dial at 0), on 200 training goblin "
                  "and 200 normal answers: normal restored = -dCE/d dial, goblin restored = d log P(goblin)/d dial, as "
                  "shares of the fine-tune's gain. Measured on 200 held-out goblin and 200 held-out normal answers, "
                  "teacher-forced. One seed per run.")


def ideal_figure():
    """Schematic: the shape each kind of curve should have, with the measured curves faint behind it."""
    fig, axs = plt.subplots(1, 2, figsize=(14, 7.4), sharey=True)
    arrow = dict(arrowstyle="-|>", color=g.INK, lw=3, mutation_scale=22, shrinkA=0, shrinkB=0)
    short = {"goblin_sw_A1": "A1", "goblin_sw_A7_3e-4": "A7", "goblin_sw_B5": "B5"}

    ax = axs[0]
    for r, _ in RESTORE_RUNS:
        d = g.load("spec_protect", f"{r}_restore")
        ks = sorted(int(c.split("#")[1]) for c in d["share"] if c.startswith("keep_restore_diff#"))
        xs, ys = zip(*[pt(d, "per_token_imp_off")] + [pt(d, f"keep_restore_diff#{k}") for k in ks])
        ax.plot(xs, ys, color=g.ORANGE, lw=1.6, marker="s", ms=4, alpha=0.45, zorder=2)
        ax.plot(xs[0], ys[0], "s", color=g.ORANGE, ms=9, mec=g.SURF, mew=1.2, zorder=3)
        ax.annotate(f"{short[r]} start", (xs[0], ys[0]), xytext=(-8, 0), textcoords="offset points", ha="right",
                    va="center", color=g.ORANGE, fontsize=10.5)
    ax.annotate("", (100, 100), (100, 0), arrowprops=arrow, zorder=5)
    ax.plot(100, 0, "o", color=g.INK, ms=13, zorder=6)
    ax.annotate("1", (100, 0), xytext=(-12, 4), textcoords="offset points", ha="right", va="bottom", color=g.INK,
                fontsize=13, fontweight="bold")
    ax.annotate("2", (100, 50), xytext=(-10, 0), textcoords="offset points", ha="right", va="center", color=g.INK,
                fontsize=13, fontweight="bold")
    ax.text(2, 2, "1  each token's important components\n    off: everything gone, goblin and\n    normal (bottom right)\n\n"
                  "2  keep the top k for normal answers\n    on: normal comes back, goblin\n    stays gone (straight up)",
            ha="left", va="bottom", color=g.INK, fontsize=11.5)
    ax.set_title("Gates: important off, then some kept on", fontsize=14)

    ax = axs[1]
    for r, _ in RESTORE_RUNS:
        e = g.load("spec_effect", r)
        sz = sorted(int(c.split("#")[1]) for c in e["share"] if c.startswith("effg#"))
        xs, ys = zip(*[(0, 100)] + [pt(e, f"effg#{k}") for k in sz])
        ax.plot(xs, ys, color=g.BLUE, lw=1.6, ls="--", marker="o", ms=4, alpha=0.45, zorder=2)
        if r == "goblin_sw_B5":
            ax.annotate("B5", (xs[-1], ys[-1]), xytext=(0, -8), textcoords="offset points", ha="center", va="top",
                        color=g.BLUE, fontsize=10.5)
    ax.annotate("A1, A7", (100, 99), xytext=(-12, -8), textcoords="offset points", ha="right", va="center",
                color=g.BLUE, fontsize=10.5)
    ax.annotate("", (100, 100), (0, 100), arrowprops=arrow, zorder=5)
    ax.plot(0, 100, "o", color=g.INK, ms=13, zorder=6)
    ax.annotate("nothing off:\nthe fine-tuned model", (0, 100), xytext=(4, -16), textcoords="offset points",
                ha="left", va="top", color=g.INK, fontsize=11.5)
    ax.annotate("switch off more of the top k for goblin:\ngoblin goes, normal answers stay", (50, 100),
                xytext=(0, -40), textcoords="offset points", ha="center", va="top", color=g.INK, fontsize=11.5)
    ax.set_title("Fixed set: top k for goblin off at every token", fontsize=14)

    for ax in axs:
        ax.text(98, 104, "ideal", ha="right", va="bottom", color=g.MUTED, fontsize=11)
        ax.set_xlim(-3, 112)
        ax.set_ylim(-3, 110)
        ax.set_xlabel("goblin removed (%)")
    axs[0].set_ylabel("normal answers kept (%)")
    fig.subplots_adjust(top=0.8, bottom=0.17, left=0.07, right=0.98, wspace=0.08)
    g.finish(fig, None, "12e_ideal_shapes", title="What each kind of curve should look like",
             sub="Black: the ideal shape. Faint: measured (orange = keep_restore_diff from figure 12d, "
                 "blue = effg from figure 12), A1 and A7 without necessity, B5 with necessity.",
             note="Left: if the gates mark everything a token uses as important, switching those off should remove "
                  "both goblin and normal answers (bottom right); keeping on the components that matter for normal "
                  "answers should then bring normal back without bringing goblin back (straight up). Keeping "
                  "components on can't remove more goblin than the start, so each orange curve can only go up from "
                  "its start or to the left. Right: no gates involved; the components ranked by goblin effect are "
                  "zeroed at every token, starting from the full fine-tune (top left).")


NOTE = ("Scores per component from one gradient pass with every component on, on 200 training goblin and 200 normal "
        "answers: goblin score = gradient of log P(goblin) at goblin spots, normal score = gradient of CE on normal answers "
        "(eff_all in spec_effect). Important = gate > 0.1 at that token. Measured on 200 held-out goblin and 200 held-out "
        "normal answers, teacher-forced; goblin removed = share of the fine-tune's gain in log P(goblin) at goblin spots "
        "that is gone, normal kept = share of its CE gain on normal answers that remains. One seed per run.")
TITLE = "Keeping the components that matter for normal answers on brings goblin back just as fast"


POSTHOC_RUNS = [("goblin_sw_A1", "Gated A1, no necessity"), ("goblin_ph_A1", "Post-hoc A1, no necessity"),
                ("goblin_sw_B5", "Gated B5, necessity 4 / 2"), ("goblin_ph_B5", "Post-hoc B5, necessity 4 / 2")]


def posthoc_figure():
    """Figures 12 (top) and 12d (bottom) for the gated runs next to the post-hoc decompositions of the plain LoRA
    (pd_lora.py, goblin_ph_*); runs without results on HF yet are left out."""
    runs = []
    for r, name in POSTHOC_RUNS:
        try:
            runs.append((r, name, g.load("spec_protect", r), g.load("spec_effect", r),
                         g.load("spec_protect", f"{r}_restore")))
        except Exception as ex:  # noqa: BLE001
            print("skip", r, type(ex).__name__)
    fig, axs = plt.subplots(2, len(runs), figsize=(4.6 * len(runs) + 1, 11.5), sharex=True, sharey=True, squeeze=False)
    for j, (r, name, d, e, q) in enumerate(runs):
        lines(axs[0, j], d, e, labels=False)
        axs[0, j].set_xlim(-3, 150 if r.startswith("goblin_ph_") else 105)
        restore_lines(axs[1, j], q)
        axs[0, j].set_title(name, fontsize=13)
        axs[1, j].set_xlabel("goblin removed (%)")
    axs[0, 0].set_xlim(-3, 150)
    for ax in axs[:, 0]:
        ax.set_ylabel("normal answers kept (%)")
    h0, l0 = axs[0, 0].get_legend_handles_labels()
    h1, l1 = axs[1, 0].get_legend_handles_labels()
    fig.legend(h0 + h1[1:], ["top: " + x for x in l0] + ["bottom: " + x for x in l1[1:]], loc="upper left",
               bbox_to_anchor=(0.02, 0.92), ncol=2, fontsize=10.5, handletextpad=0.3)
    fig.subplots_adjust(top=0.78, bottom=0.15, left=0.06, right=0.98, wspace=0.08, hspace=0.12)
    g.finish(fig, None, "12f_protect_posthoc", title="Gated LoRA vs post-hoc decomposition of the plain goblin LoRA",
             note=NOTE + " Post-hoc: pd_lora.py on goblin_plain_mix_p005 ckpt_3000, C 64 per module (12,544 components), "
                  "5,000 steps; shares are of the plain LoRA's gain. Above 100% removed = goblin pushed below the base "
                  "model. Bottom row as figure 12d.")


def posthoc_b5_figure():
    """Gated B5 vs post-hoc B5 on the two tests: a fixed goblin set off at every token (left), and each token's
    important components off, then the top k by normal restored minus goblin restored kept on (right)."""
    runs = [("goblin_sw_B5", "gated B5", g.BLUE), ("goblin_ph_B5", "post-hoc B5", g.ORANGE)]
    fig, axs = plt.subplots(1, 2, figsize=(14, 7.6), sharey=True)
    for r, name, col in runs:
        e, q = g.load("spec_effect", r), g.load("spec_protect", f"{r}_restore")
        sz = [16, 32, 64, 128, 256, 1024]
        xs, ys = zip(*[(0, 100)] + [pt(e, f"effg#{k}") for k in sz])
        axs[0].plot(xs, ys, color=col, lw=2.2, marker="o", ms=7, mec=g.SURF, mew=1.2, zorder=3, label=name)
        for k, x, y in zip(sz, xs[1:], ys[1:]):
            if k in (16, 64, 256, 1024):
                axs[0].annotate(f"{k:,}", (x, y), xytext=(0, 9 if col == g.ORANGE else -10), textcoords="offset points",
                                ha="center", va="bottom" if col == g.ORANGE else "top", color=col, fontsize=10)
        ks = sorted(int(c.split("#")[1]) for c in q["share"] if c.startswith("keep_restore_diff#"))
        xs, ys = zip(*[pt(q, "per_token_imp_off")] + [pt(q, f"keep_restore_diff#{k}") for k in ks])
        axs[1].plot(xs, ys, color=col, lw=2.2, marker="s", ms=6, mec=g.SURF, mew=1.2, zorder=3, label=name)
        axs[1].plot(xs[0], ys[0], "s", color=col, ms=11, mec=g.SURF, mew=1.5, zorder=4)
        left = col == g.BLUE
        axs[1].annotate(f"{name}:\nall important off", (xs[0], ys[0]), xytext=(-12 if left else 12, 0),
                        textcoords="offset points", ha="right" if left else "left", va="center", color=col, fontsize=10.5)
        axs[1].annotate(f"{ks[-1]:,} kept on", (xs[-1], ys[-1]), xytext=(-10 if left else 10, -12 if left else -14),
                        textcoords="offset points", ha="right" if left else "left", va="center", color=col, fontsize=10)
    axs[0].set_title("1. Top k for goblin off at every token", fontsize=14)
    axs[0].text(2, 4, "start: the fine-tune (top left)\nwant: right, staying at the top", color=g.INK2, fontsize=11)
    axs[1].set_title("2. Each token's important components off, then k kept on", fontsize=14)
    axs[1].text(2, 50, "start: everything important off\n(want bottom right: honest gates\nremove goblin and normal)\n\n"
                      "then: keep on the top k for\nnormal (want straight up)", color=g.INK2, fontsize=11)
    for ax in axs:
        ax.plot(100, 100, "*", color=g.MUTED, ms=14, zorder=2)
        ax.text(100, 104, "ideal", ha="center", va="bottom", color=g.MUTED, fontsize=11)
        ax.axvline(100, color=g.MUTED, lw=0.8, ls=":", zorder=1)
        ax.set_xlim(-3, 155)
        ax.set_ylim(-3, 110)
        ax.set_xlabel("goblin removed (%)")
    axs[0].set_ylabel("normal answers kept (%)")
    axs[0].legend(loc="center right", fontsize=12, frameon=False)
    fig.subplots_adjust(top=0.84, bottom=0.2, left=0.07, right=0.98, wspace=0.08)
    g.finish(fig, None, "12g_protect_b5_posthoc", title="Necessity 4 / 2: gated LoRA vs post-hoc decomposition of the plain LoRA",
             note="Goblin removed = share of the fine-tune's gain in log P(goblin) at goblin spots that is gone (above 100% = "
                  "below the base model); normal kept = share of its CE gain on normal answers that remains. 200 held-out "
                  "goblin and 200 held-out normal answers, teacher-forced. Left: components ranked by goblin effect (effg in "
                  "spec_effect). Right: important = gate > 0.1 at that token; kept-on components ranked by normal restored "
                  "minus goblin restored (keep_restore_diff, k = 16 to 4,096). Gated B5: 6,272 components; post-hoc B5: "
                  "pd_lora.py on goblin_plain_mix_p005, 12,544 components. One seed per run.")


# mean number of components with gate > 0.1 per token on held-out normal answers and at goblin words, from the
# goblin_component_dump.py outputs (sum of counts / totals per kind; 150 held-out pairs)
IMP_PER_TOKEN = {"goblin_sw_A1": (16.2, 34.2), "goblin_sw_B5": (23.1, 79.1), "goblin_ph_A1": (56.2, 114.1),
                 "goblin_ph_B5": (269.5, 462.2)}


def imp_off_figure(runs=None, name="12h_imp_off"):
    """Only the first step of figure 12d: each token's important components (gate > 0.1) off, nothing kept on."""
    runs = runs or POSTHOC_RUNS
    wide = len(runs) > 2
    fig, axs = plt.subplots(1, 2, figsize=(15 if wide else 12, 7), gridspec_kw=dict(width_ratios=[1.5, 1]))
    import numpy as np
    x = np.arange(len(runs))
    bars = [("goblin_removed", "goblin removed (want 100)", g.ORANGE), ("clean_kept", "normal answers kept (want 0)", g.BLUE),
            ("other_kept", "other tokens of goblin answers kept (want 0)", g.AQUA)]
    for i, (key, label, col) in enumerate(bars):
        v = [100 * g.load("spec_protect", r)["share"]["per_token_imp_off"][key] for r, _ in runs]
        b = axs[0].bar(x + (i - 1) * 0.26, v, 0.25, color=col, label=label, zorder=3)
        axs[0].bar_label(b, fmt="%.0f", padding=3, fontsize=10.5, color=g.INK2)
    axs[0].axhline(100, color=g.MUTED, lw=0.8, ls=":")
    axs[0].set_ylim(0, 112)
    axs[0].set_ylabel("% of the fine-tune's gain")
    axs[0].set_title("Each token's important components off", fontsize=14)
    axs[0].legend(loc="upper left", bbox_to_anchor=(0, -0.1), ncol=3 if wide else 1, fontsize=11)
    for ax in axs:
        ax.set_xticks(x, [n.replace(", ", "\n") for _, n in runs], fontsize=11)
        ax.grid(axis="x", visible=False)
    for i, (key, label, col) in enumerate([(0, "on normal answers", g.BLUE), (1, "at goblin words", g.ORANGE)]):
        v = [IMP_PER_TOKEN[r][key] for r, _ in runs]
        b = axs[1].bar(x + (i - 0.5) * 0.36, v, 0.35, color=col, label=label, zorder=3)
        axs[1].bar_label(b, fmt="%.0f", padding=3, fontsize=10.5, color=g.INK2)
    axs[1].set_title("How many: gate > 0.1 per token" if not wide else "How many that is: components with gate > 0.1 per token", fontsize=14)
    axs[1].legend(loc="upper left", fontsize=11)
    fig.subplots_adjust(top=0.84, bottom=0.27 if wide else 0.33, left=0.07, right=0.98, wspace=0.18)
    g.finish(fig, None, name, title="Switching off what the gates mark important, at every token",
             note="Left: at every token, every component with gate > 0.1 at that token is zeroed (per_token_imp_off in "
                  "spec_protect), nothing kept on. If the gates mark everything the fine-tune uses, this removes all of it: "
                  "goblin 100, normal 0, other 0. Shares of the fine-tune's gain: goblin = log P(goblin) at goblin spots, "
                  "normal = CE on normal answers, other = CE on the non-goblin tokens of goblin answers; 200 held-out answers "
                  "each, teacher-forced. Right: of 6,272 components (gated) or 12,544 (post-hoc), on 150 held-out pairs. "
                  "One seed per run.")

if __name__ == "__main__":
    {"restore": restore_figure, "ideal": ideal_figure, "posthoc": posthoc_figure, "b5": posthoc_b5_figure, "imp_off": imp_off_figure,
     "imp_off_b5": lambda: imp_off_figure(POSTHOC_RUNS[2:], "12i_imp_off_b5")}.get(sys.argv[1] if sys.argv[1:] else "", main)()
