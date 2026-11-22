"""12j: per-token gate check for the weight-decay runs (goblin_sweep.sh A1_W*, A7_W*, 12 Nov) next to B5. At each token,
either keep only the components whose gate is above 0.1 ("only important on") or switch exactly those off ("important
off"), and measure how much goblin behaviour and clean fine-tune is left. Same measure as figure 20, from
runs/<run>/eval.json (gated_lora.py eval: conditions "imp" and "unimp").

  python3 goblin_decay_imp_on_off_plot.py   -> findings/goblin/12j_decay_imp_on.png, 12k_decay_imp_off.png
"""
import numpy as np

import goblin_plots as g
from goblin_imp_on_off_plot import shares

plt = g.plt
GROUPS = [
    ("A1", [("goblin_sw_A1", "no decay"), ("goblin_sw_A1_WP1", "plain 1"), ("goblin_sw_A1_WP3", "plain 3"),
            ("goblin_sw_A1_WP10", "plain 10"), ("goblin_sw_A1_WG1", "gate-scaled 1"), ("goblin_sw_A1_WG3", "gate-scaled 3"),
            ("goblin_sw_A1_WG10", "gate-scaled 10")]),
    ("A7", [("goblin_sw_A7_3e-4", "no decay"), ("goblin_sw_A7_WP1", "plain 1"), ("goblin_sw_A7_WP3", "plain 3"),
            ("goblin_sw_A7_WP10", "plain 10"), ("goblin_sw_A7_WG1", "gate-scaled 1"), ("goblin_sw_A7_WG3", "gate-scaled 3"),
            ("goblin_sw_A7_WG10", "gate-scaled 10")]),
    ("B5", [("goblin_sw_B5", "necessity 4 / 2, no decay")]),
]
SERIES = [(0, "goblin behaviour left", g.ORANGE), (1, "clean fine-tune kept", g.BLUE)]  # index into shares()
FIGS = {
    "imp": ("12j_decay_imp_on",
            "Only each token's important components on: how much is left?",
            "Faithful gates would keep close to 100% of both. Every run keeps most of the goblin (83-97%) and of the "
            "clean fine-tune (76-94%), with or without necessity or decay: the gates find what is needed. Gate-scaled "
            "decay 10 nudges both up (A1: 86 / 76 to 90 / 89)."),
    "unimp": ("12k_decay_imp_off",
              "Each token's important components off: how much is left?",
              "Faithful gates would leave close to 0% of both. This is where the runs differ: necessity (B5) leaves 17% "
              "of the goblin and 6% of the clean fine-tune. Decay 10 moves A1 from 57% goblin left to 42% (plain) or "
              "37% (gate-scaled), and A7 from 35% to 27% or 24%, while keeping 37-61% of the clean fine-tune."),
}
NOTE = ("Co-trained gated LoRAs (gated_lora.py), 3,000 steps, C = 32, lambda 3e-4, no necessity (A1; A7 = A1 with "
        "reconstruction weights --c_stoch 4 --c_adv 2), except B5 (necessity --c_comp 4 --c_comp_adv 2). Plain decay = "
        "AdamW decay on U, V (--wd); gate-scaled = tPD's CIScaledWeightDecay (--wd --wd_scaled_by_gate 1). 'Important' "
        "= gate (lower_leaky(CI)) > 0.1 at that token. Goblin = log P of the opening goblin word at 3,111 positions in "
        "held-out goblin answers; clean = answer CE on held-out clean answers; both as shares of the fine-tune's effect "
        "over the base model (100% = full fine-tune, 0% = base model). From runs/<run>/eval.json (conditions 'imp' "
        "and 'unimp'). One seed per run.")


def layout():
    rows, y, yy, heads = [], [], 0.0, []
    for gname, runs in GROUPS:
        heads.append((gname, yy))
        yy -= 0.9
        for r, lab in runs:
            rows.append((lab, shares(r)))
            y.append(yy)
            yy -= 1
        yy -= 0.5
    return rows, np.array(y), heads


def one(cond, rows, y, heads):
    name, title, sub = FIGS[cond]
    fig, ax = plt.subplots(figsize=(13, 10.4))
    fig.subplots_adjust(top=0.8, bottom=0.16, left=0.2, right=0.97)
    h = 0.4
    for k, (j, lab, col) in enumerate(SERIES):
        yk = y + (h / 2 if k == 0 else -h / 2)
        v = [d[j][cond] for _, d in rows]
        ax.barh(yk, v, height=h - 0.04, color=col, edgecolor=g.SURF, linewidth=1, label=lab, zorder=3)
        for vi, yi in zip(v, yk):
            ax.annotate(f"{vi:.0f}", (max(vi, 0), yi), xytext=(4, 0), textcoords="offset points", va="center",
                        fontsize=9.5, color=g.INK2)
    ax.axvline(0, color=g.AXIS, lw=1, zorder=2)
    ax.axvline(100, color=g.MUTED, lw=1, ls="--", zorder=1)
    ax.set_xlim(0, 108)
    ax.set_xlabel("% of the fine-tune's effect left")
    ax.set_yticks(y, [lab for lab, _ in rows])
    ax.tick_params(axis="y", colors=g.INK2, labelsize=12, length=0)
    ax.grid(axis="y", visible=False)
    for gname, yh in heads:
        ax.text(-0.02, yh - 0.3, gname, transform=ax.get_yaxis_transform(), ha="right", va="center", fontsize=13,
                fontweight="bold", color=g.INK)
    ax.set_ylim(y[-1] - 0.7, heads[0][1] + 0.2)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=12)
    g.finish(fig, None, name, title=title, sub=sub, note=NOTE)


def main():
    rows, y, heads = layout()
    for cond in FIGS:
        one(cond, rows, y, heads)


if __name__ == "__main__":
    main()
