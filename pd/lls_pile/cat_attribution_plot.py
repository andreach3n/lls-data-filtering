"""Where the cat shift comes from, by activation x gradient attribution (cat_attribution.py), necessity vs no necessity:
share through position 0, through the top 8 components by effect, and through gate-important (component, token)
pairs. Reads results/pd_runs/{pd_cat,pd_cat_nonec}/attribution.json, writes results/cat_attribution.png.

  python3 cat_attribution_plot.py
"""
import json
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "lora"))
import goblin_plots as g  # noqa: E402  (shared style and palette)

plt = g.plt
RUNS = HERE / "results" / "pd_runs"
OUT = HERE / "results" / "cat_attribution.png"
ARMS = [("pd_cat", g.ORANGE, "necessity 4/2"), ("pd_cat_nonec", g.BLUE, "no necessity")]
METRICS = [("share_pos0", "at position 0\n(window start)"), ("share_top8", "in the top 8 components\nby effect on cat"),
           ("share_gate_important", "where the gate says\nimportant (> 0.1)")]


def main():
    d = {r: json.load(open(RUNS / r / "attribution.json")) for r, _, _ in ARMS}
    fig, ax = plt.subplots(figsize=(11, 6.8))
    fig.subplots_adjust(top=0.78, bottom=0.24, left=0.08, right=0.98)
    w = 0.36
    for j, (r, c, name) in enumerate(ARMS):
        for i, (key, _) in enumerate(METRICS):
            x = i + (j - 0.5) * w
            y = 100 * d[r][key]
            ax.bar(x, y, width=w, color=c, edgecolor=g.SURF, lw=2, zorder=3, label=name if i == 0 else None)
            ax.annotate(f"{y:.0f}%", (x, y), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                        color=g.INK, fontsize=12.5)
    ax.set_xticks(range(len(METRICS)))
    ax.set_xticklabels([lab for _, lab in METRICS], color=g.INK2, fontsize=12)
    ax.tick_params(axis="x", length=0)
    ax.set_ylim(0, 100)
    ax.set_ylabel("share of the cat shift (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right", fontsize=11.5, labelcolor=g.INK2)
    g.subtitle(ax, "Share of the fine-tune's log P(cat) gain carried …")

    n80 = {}
    for r, _, _ in ARMS:
        v, cum = sorted((x["attr"] for x in d[r]["components"].values()), reverse=True), 0.0
        for k, x in enumerate(v, 1):
            cum += x
            if cum >= 0.8 * d[r]["attr_total"]:
                n80[r] = k
                break
    fig.suptitle("Without necessity, most of the cat shift runs where the gates say nothing is important",
                 x=0.02, y=0.985, ha="left", fontsize=17, fontweight="bold", color=g.INK)
    sub = (f"With necessity, gate-important components carry {100 * d['pd_cat']['share_gate_important']:.0f}% of cat "
           f"and {n80['pd_cat']} components reach 80% of it; without, {100 * d['pd_cat_nonec']['share_gate_important']:.0f}% "
           f"and {n80['pd_cat_nonec']}. In both, a large part enters at the first token of the window.")
    fig.text(0.02, 0.93, textwrap.fill(sub, 125), ha="left", va="top", color=g.INK2, fontsize=12.5)
    v = {r: d[r]["validation"]["pearson_r"] for r, _, _ in ARMS}
    note = ("Attribution = activation × gradient of log P(cat words) (mean probability over every position of 2,000 "
            "held-out 32-token Pile windows) for every component at every position, the full LoRA linearised; shares "
            f"are of the summed attribution ({d['pd_cat']['attr_total']:.2f} and {d['pd_cat_nonec']['attr_total']:.2f} "
            f"vs actual log P_full / P_base {d['pd_cat']['log_p_full_over_base']:.2f} and "
            f"{d['pd_cat_nonec']['log_p_full_over_base']:.2f}: components interact when all are removed). Summed per "
            f"component it matches the measured effect of switching that component off (r = {v['pd_cat']:.4f}, "
            f"{v['pd_cat_nonec']:.4f} over 768 components). λ 1e-3, β 0.2 in both runs; one seed.")
    fig.text(0.02, 0.012, textwrap.fill(note, 165), ha="left", va="bottom", color=g.MUTED, fontsize=10)
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
