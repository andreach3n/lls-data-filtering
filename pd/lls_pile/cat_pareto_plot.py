"""PD-LoRA on the LLS cat selection (4L Pile model): cat removed vs the rest of the fine-tune removed as the top-k
components by effect on P(cat) are switched off (k = 1 ... 512 of 768), necessity vs no necessity at the same lam / beta,
plus random-k controls. Reads results/pd_runs/{pd_cat,pd_cat_nonec}/topk_removal_512.json, writes
results/cat_pareto.png.

  python3 cat_pareto_plot.py
"""
import json
import math
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "lora"))
import goblin_plots as g  # noqa: E402  (shared style and palette)

plt = g.plt
RUNS = HERE / "results" / "pd_runs"
OUT = HERE / "results" / "cat_pareto.png"
ARMS = [("pd_cat", g.ORANGE, "necessity 4/2"), ("pd_cat_nonec", g.BLUE, "no necessity")]
LABEL_KS = {"pd_cat": (1, 4, 8, 512), "pd_cat_nonec": (1, 8, 64, 128, 256, 512)}


def curve(run, which):
    """k, cat removed (%), rest of the fine-tune removed (%) = 100 - KL(base || top k off) / KL(base || full)"""
    rows = {(r["set"], r["k"]): r for r in json.load(open(RUNS / run / "topk_removal_512.json"))["rows"]}
    pb, pf, klf = rows["base", 0]["p_cat"], rows["full", 0]["p_cat"], rows["full", 0]["kl_from_base"]
    ks = sorted(k for s, k in rows if s == which)
    cat = [100 * math.log(pf / rows[which, k]["p_cat"]) / math.log(pf / pb) for k in ks]
    rest = [100 - 100 * rows[which, k]["kl_from_base"] / klf for k in ks]
    return ks, cat, rest


def main():
    fig, ax = plt.subplots(figsize=(11, 8))
    fig.subplots_adjust(top=0.8, bottom=0.2, left=0.09, right=0.97)
    for run, c, name in ARMS:
        rk, rcat, rrest = curve(run, "random")  # a fresh random draw per k, not nested: points, not a line
        ax.plot(rrest, rcat, color=c, marker="^", ms=6, ls="none", alpha=0.45, zorder=2,
                label=f"k random components off ({name})")
        ks, cat, rest = curve(run, "top")
        ax.plot(rest, cat, color=c, lw=2, marker="o", ms=7, mec=g.SURF, mew=1.5, zorder=3, label=name)
        for k, x, y in zip(ks, rest, cat):
            if k in LABEL_KS[run]:
                ax.annotate(f"{k}", (x, y), xytext=(0, 9), textcoords="offset points", ha="center", va="bottom",
                            color=g.INK2, fontsize=10.5)
    ax.plot([0], [100], marker="*", ms=16, color=g.INK2, ls="none", zorder=4)
    ax.annotate("ideal: all of cat,\nnothing else", (0, 100), xytext=(10, 0), textcoords="offset points", va="center",
                color=g.INK2, fontsize=11)
    ax.axhline(100, color=g.AXIS, lw=1, zorder=1)
    ax.set_xlim(-3, 103)
    ax.set_ylim(-5, 118)
    ax.set_xlabel("rest of the fine-tune removed (%)")
    ax.set_ylabel("cat removed (%)")
    ax.legend(loc="lower right", fontsize=11.5, labelcolor=g.INK2)
    g.subtitle(ax, "Switching off the top k components by effect on P(cat), k = 1 … 512 of 768 (numbers on the points)")

    fig.suptitle("Necessity puts cat in 8 components instead of 256, but removing it costs the same",
                 x=0.02, y=0.985, ha="left", fontsize=17, fontweight="bold", color=g.INK)
    sub = ("Up and to the left is better. Removing all of cat takes 68% of the rest of the fine-tune with necessity "
           "(8 components) and 63% without (256). For partial removal the no-necessity curve is better: 73% of cat "
           "costs 28% of the rest without necessity (64 components), 43% with (4).")
    fig.text(0.02, 0.93, textwrap.fill(sub, 125), ha="left", va="top", color=g.INK2, fontsize=12.5)
    note = ("PD-LoRA on the top 5% of Pile windows by LLS score (cat − animals), Goodfire's 4-layer 67M Pile model, "
            "2,000 steps, λ 1e-3, β 0.2 in both runs. Components ranked by the drop in log P(cat words) when each alone "
            "is switched off (2,000 held-out 32-token Pile windows), then the top k switched off at every position and "
            "measured on 2,000 others. Cat removed = share of the full LoRA's gain in log P(cat words) that is gone. "
            "Rest removed = 100% − KL(base || top k off) / KL(base || full LoRA). Triangles = k random components off "
            "(one draw per k). "
            "One seed per run.")
    fig.text(0.02, 0.012, textwrap.fill(note, 160), ha="left", va="bottom", color=g.MUTED, fontsize=10)
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
