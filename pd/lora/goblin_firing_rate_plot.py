"""17: how often each component's gate opens (gate > 0.1) over the same 5M training tokens, for every run in 14b
(goblin_splitting.py alive counts, splitting/ on HF). One panel per setup, one line per C. Standalone figure.

  python3 goblin_firing_rate_plot.py   -> findings/goblin/17_firing_rate_histogram.png
"""
import numpy as np

import goblin_plots as g
from goblin_splitting_fixes_plot import SERIES

plt = g.plt
RAMP = {16: "#86b6ef", 32: "#5598e7", 64: "#2a78d6", 128: "#1c5cab", 256: "#0d366b"}  # dataviz blue, ordinal-validated
EDGES = np.arange(-7, 0.01, 0.5)  # log10(firing rate); 1 token in 5M = 10^-6.7
NEVER = (-8.0, -7.5)  # x span of the "never fires" bar, left of the log axis
TICKS = [(-7.75, "never"), (-6, "1 in 1M"), (-4, "1 in 10k"), (-2, "1%"), (0, "every\ntoken")]
TITLES = {"A1": "A1: co-trained, no necessity", "B4": "B4: co-trained, necessity 1 / 0.5",
          "B4 + decay 10": "B4 + gate-scaled decay 10", "T4": "T4: teacher (KL) + necessity 1 / 0.5",
          "post-hoc A1": "post-hoc A1: decomposes the plain LoRA", "decay 30": "B4 + gate-scaled decay 30"}


def hist(run):
    d = g.load("splitting", run)
    c = np.concatenate([np.asarray(m["count"]) for m in d["modules"].values()])
    n = len(c)
    rate = np.log10(np.clip(c[c > 0], 1, None) / d["n_tokens"])
    h, _ = np.histogram(np.clip(rate, EDGES[0], EDGES[-1]), EDGES)
    return 100 * (c == 0).sum() / n, 100 * h / n, d["n_alive"], n


def main():
    fig, axs = plt.subplots(2, 3, figsize=(16, 10), sharex=True, sharey=True)
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.07, right=0.98, hspace=0.3, wspace=0.06)
    for ax, (_, end, runs, *_) in zip(axs.flat, SERIES):
        for C, r in sorted(runs.items()):
            try:
                never, h, n_alive, n = hist(r)
            except Exception:  # not run (yet)
                continue
            col = RAMP[C]
            ax.stairs(h, EDGES, color=col, lw=2.2, label=f"C = {C}: {n_alive:,} of {n:,} alive", zorder=3)
            ax.plot(NEVER, [never] * 2, color=col, lw=5, solid_capstyle="butt", zorder=3)
        ax.axvspan(-8.2, -6, color=g.GRID, alpha=0.45, lw=0, zorder=0)
        ax.axvline(-6, color=g.MUTED, lw=1, ls="--", zorder=1)
        ax.text(-5.9, 0.97, "alive", transform=ax.get_xaxis_transform(), color=g.INK2, fontsize=11, va="top")
        ax.text(-6.1, 0.97, "dead", transform=ax.get_xaxis_transform(), color=g.INK2, fontsize=11, va="top",
                ha="right")
        ax.set_title(TITLES[end], fontsize=13, loc="left")
        ax.legend(loc="upper right", bbox_to_anchor=(1, 0.9), fontsize=10.5, handlelength=1.4)
        ax.set_ylim(0, 84)
        ax.set_xlim(-8.2, 0.1)
        ax.set_xticks([t for t, _ in TICKS], [s for _, s in TICKS], fontsize=10.5)
        ax.minorticks_off()
        ax.grid(axis="x", visible=False)
    for ax in axs[:, 0]:
        ax.set_ylabel("share of the run's components (%)")
    for ax in axs[1]:
        ax.set_xlabel("how often the gate opens (share of tokens with gate > 0.1)")
    for ax in axs.flat[len(SERIES):]:
        ax.set_visible(False)
    g.finish(fig, None, "17_firing_rate_histogram",
             title="How often each component's gate opens, for every run in 14b",
             sub="Each line is one run (darker = larger C). Feature splitting shows as the whole curve sliding left "
                 "(rarer firing) as C grows; real dead components show as mass in the shaded region.",
             note="Every component of each run, over the same 5M training tokens (5% goblin mix). Gate = "
                  "lower_leaky(CI), 'opens' = gate > 0.1. Bins are half a decade wide; the thick bar at the far left "
                  "is components whose gate never opened once. Dead = 5 or fewer of the 5M tokens (left of the "
                  "dashed line, the VPD Figure 7 rule). y = share of that run's components, so runs with different C "
                  "are comparable; the legend gives the absolute alive count.")



def main_c128_split():
    """17c: 17b split into one panel per setup (same y scale everywhere)."""
    import goblin_splitting_fixes_plot as fx
    style = {"A1": (g.BLUE, "-"), "B4": ("#eda100", "-"), "B4 + decay 10": ("#eda100", "--"),
             "decay 30": ("#eda100", ":"), "T4": (fx.GREEN, "-"), "post-hoc A1": (fx.RED, "-")}
    order = ["post-hoc A1", "A1", "B4", "T4", "B4 + decay 10", "decay 30"]
    runs = {end: rs[128] for _, end, rs, *_ in SERIES if 128 in rs}
    fig, axs = plt.subplots(2, 3, figsize=(16, 9.6), sharex=True, sharey=True)
    fig.subplots_adjust(top=0.84, bottom=0.14, left=0.07, right=0.98, hspace=0.3, wspace=0.06)
    for ax, end in zip(axs.flat, order):
        never, h, n_alive, n = hist(runs[end])
        col, ls = style[end]
        ax.stairs(h, EDGES, color=col, lw=2.4, ls=ls, zorder=3)
        ax.bar(-7.75, never, width=0.45, color=col, edgecolor=g.SURF, linewidth=1, zorder=3)
        ax.text(-7.75, never + 1.2, f"{never:.0f}%", ha="center", va="bottom", color=g.INK2, fontsize=11)
        ax.axvspan(-8.2, -6, color=g.GRID, alpha=0.45, lw=0, zorder=0)
        ax.axvline(-6, color=g.MUTED, lw=1, ls="--", zorder=1)
        ax.text(-5.9, 0.97, "alive", transform=ax.get_xaxis_transform(), color=g.INK2, fontsize=11, va="top")
        ax.text(-6.1, 0.97, "dead", transform=ax.get_xaxis_transform(), color=g.INK2, fontsize=11, va="top",
                ha="right")
        ax.set_title(TITLES[end], fontsize=13, loc="left", pad=24)
        g.subtitle(ax, f"{n_alive:,} of {n:,} alive; {never:.0f}% never fire")
        ax.set_xlim(-8.2, 0.1)
        ax.set_ylim(0, 62)
        ax.set_xticks([t for t, _ in TICKS], [s for _, s in TICKS], fontsize=10.5)
        ax.minorticks_off()
        ax.grid(axis="x", visible=False)
    for ax in axs[:, 0]:
        ax.set_ylabel("share of the run's components (%)")
    for ax in axs[1]:
        ax.set_xlabel("how often the gate opens (gate > 0.1)")
    g.finish(fig, None, "17c_firing_rate_histogram_C128_split",
             title="How often each component's gate opens at C = 128, one panel per setup",
             sub="Same data as 17b. Post-hoc fires on 0.01-1% of tokens with nothing dead; the co-trained runs pile "
                 "up at never and just around the dead line.",
             note="C = 128 per module (25,088 components), over the same 5M training tokens (5% goblin mix). "
                  "'Opens' = gate > 0.1; half-decade bins; the bar at the far left = components whose gate never "
                  "opened once. Dead = 5 or fewer of the 5M tokens (left of the dashed line). Same y scale in every "
                  "panel; colors and line styles as in 14b.")


if __name__ == "__main__":
    main()
    main_c128()
    main_c128_split()


def main_c128():
    """17b: the same histogram at C = 128 only, every setup in one panel (colors / line styles as in 14b)."""
    import goblin_splitting_fixes_plot as fx
    style = {"A1": (g.BLUE, "-"), "B4": ("#eda100", "-"), "B4 + decay 10": ("#eda100", "--"),
             "decay 30": ("#eda100", ":"), "T4": (fx.GREEN, "-"), "post-hoc A1": (fx.RED, "-")}
    order = ["post-hoc A1", "A1", "B4", "T4", "B4 + decay 10", "decay 30"]
    runs = {end: rs[128] for _, end, rs, *_ in SERIES if 128 in rs}
    fig, ax = plt.subplots(figsize=(13, 7.6))
    fig.subplots_adjust(top=0.84, bottom=0.2, left=0.08, right=0.97)
    w = 0.12  # never-fires bars, side by side
    for i, end in enumerate(order):
        never, h, n_alive, n = hist(runs[end])
        col, ls = style[end]
        ax.stairs(h, EDGES, color=col, lw=2.4, ls=ls, zorder=3,
                  label=f"{TITLES[end]}: {n_alive:,} alive, {never:.0f}% never fire")
        x0 = -8.1 + i * w
        ax.bar(x0 + w / 2, never, width=w * 0.85, color=col, edgecolor=g.SURF, linewidth=1, zorder=3)
        short = {"post-hoc A1": "post-hoc", "B4 + decay 10": "decay 10"}.get(end, end)
        ax.text(x0 + w / 2, never + 0.8, f"{short}  {never:.0f}%", rotation=90, ha="center", va="bottom",
                color=g.INK2, fontsize=10)
    ax.axvspan(-8.2, -6, color=g.GRID, alpha=0.45, lw=0, zorder=0)
    ax.axvline(-6, color=g.MUTED, lw=1, ls="--", zorder=1)
    ax.text(-5.93, 0.97, "alive", transform=ax.get_xaxis_transform(), color=g.INK2, fontsize=12, va="top")
    ax.text(-6.07, 0.97, "dead", transform=ax.get_xaxis_transform(), color=g.INK2, fontsize=12, va="top", ha="right")
    ax.set_xlim(-8.2, 0.1)
    ax.set_ylim(0, 70)
    ax.set_xticks([t for t, _ in TICKS[1:]] + [-8.1 + 3 * w], [s for _, s in TICKS[1:]] + ["never"])
    ax.minorticks_off()
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("share of the run's 25,088 components (%)")
    ax.set_xlabel("how often the gate opens (share of tokens with gate > 0.1)")
    ax.legend(loc="upper right", fontsize=11, handlelength=2.2)
    g.finish(fig, ax, "17b_firing_rate_histogram_C128",
             title="How often each component's gate opens, all setups at C = 128",
             sub="Post-hoc keeps every component firing on 0.1-1% of tokens; the co-trained runs split into a pile that "
                 "never fires plus a smear of very rare components. Decay 10 moves more of that smear to never; decay "
                 "30 instead leaves a large lump just above the dead line.",
             note="C = 128 per module (25,088 components), over the same 5M training tokens (5% goblin mix). "
                  "'Opens' = gate > 0.1; half-decade bins; the bars at the far left are components whose gate never "
                  "opened once (same order as the legend). Dead = 5 or fewer of the 5M tokens (left of the dashed "
                  "line). Colors and line styles as in 14b.")
