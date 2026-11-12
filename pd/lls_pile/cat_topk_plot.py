"""PD-LoRA on the LLS cat selection (4L Pile model): switch off the top-k components by effect on P(cat), with and
without the necessity loss, plus random-k controls. Reads results/pd_runs/*/topk_removal.json, writes
results/cat_topk_removal.png.

  python3 cat_topk_plot.py
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
OUT = HERE / "results" / "cat_topk_removal.png"


def curves(run, which):
    """rows of topk_removal.json with set == which -> k, cat removed %, rest kept %, KL(base || LoRA), full LoRA's KL"""
    d = json.load(open(RUNS / run / "topk_removal.json"))
    rows = {(r["set"], r["k"]): r for r in d["rows"]}
    pb, pf, klf = rows["base", 0]["p_cat"], rows["full", 0]["p_cat"], rows["full", 0]["kl_from_base"]
    ks = sorted(k for s, k in rows if s == which)
    rem = [100 * math.log(pf / rows[which, k]["p_cat"]) / math.log(pf / pb) for k in ks]
    kl = [rows[which, k]["kl_from_base"] for k in ks]
    return ks, rem, [100 * v / klf for v in kl], kl, klf


def main():
    arms = [("pd_cat", g.ORANGE, "necessity 4/2"), ("pd_cat_nonec", g.BLUE, "no necessity")]
    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(21, 6.6))
    fig.subplots_adjust(top=0.77, bottom=0.2, left=0.045, right=0.9, wspace=0.45)

    def plot(ax, x, y, color, ls="-", marker="o", alpha=1.0):
        ax.plot(x, y, color=color, lw=2, ls=ls, marker=marker, ms=7, mec=g.SURF, mew=1.5, alpha=alpha, zorder=3)

    def lab(ax, x, y, text, dy=0):
        ax.annotate(text, (x, y), xytext=(8, dy), textcoords="offset points", color=g.INK, fontsize=12, va="center")

    end = {}
    for run, c, name in arms:
        ks, rem, kept, kl, klf = curves(run, "top")
        rk, rrem, rkept, rkl, _ = curves(run, "random")
        for ax, ry, y in ((a1, rrem, rem), (a2, rkept, kept), (a3, rkl, kl)):
            plot(ax, rk, ry, c, ls=":", marker="", alpha=0.6)
            plot(ax, ks, y, c)
        a3.axhline(klf, color=c, lw=1, ls="--", alpha=0.5, zorder=1)
        a3.annotate(f"full LoRA {klf:.3f}", (1, klf), xytext=(0, 5), textcoords="offset points", color=g.INK2,
                    fontsize=10.5, va="bottom")
        end[run] = (rem[-1], kept[-1], kl[-1], rrem[-1], rkept[-1], rkl[-1])

    for ax, i, j in ((a1, 0, 3), (a2, 1, 4), (a3, 2, 5)):
        lab(ax, 128, end["pd_cat"][i], "necessity 4/2")
        lab(ax, 128, end["pd_cat_nonec"][i], "no necessity")
        if ax is a3:  # the two random lines sit far apart here: label each
            for run in ("pd_cat", "pd_cat_nonec"):
                lab(ax, 128, end[run][j], "random")
        else:
            lab(ax, 128, (end["pd_cat"][j] + end["pd_cat_nonec"][j]) / 2, "random\n(both runs)")

    ticks = [1, 2, 4, 8, 16, 32, 64, 128]
    for ax in (a1, a2, a3):
        ax.set_xscale("log", base=2)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks])
        ax.minorticks_off()
        ax.set_xlabel("components switched off (of 768), largest effect first")
    a1.axhline(100, color=g.AXIS, lw=1, zorder=1)
    a1.set_ylim(-5, 115)
    a1.set_ylabel("cat removed (%)")
    a2.set_ylim(0, 105)
    a2.set_ylabel("rest of the fine-tune kept (%)")
    a3.set_ylim(0, 0.095)
    a3.set_ylabel("KL(base || LoRA) per token")
    g.subtitle(a1, "Cat removed (higher is better)")
    g.subtitle(a2, "Fine-tune still there, as a share of each LoRA's total")
    g.subtitle(a3, "Fine-tune still there, absolute (distance from base)")

    fig.suptitle("With necessity, 8 components remove cat completely; without, 128 still leave 10%",
                 x=0.02, y=0.985, ha="left", fontsize=17, fontweight="bold", color=g.INK)
    sub = ("At the same cat removal, both runs lose about the same amount of fine-tune (right: 0.022 vs 0.024 at 73% removed), "
           "but the no-necessity LoRA starts from a bigger total (0.084 vs 0.050), so it keeps a larger share (middle: "
           "72% vs 57%). Above 100% cat removed = P(cat) below the base model.")
    fig.text(0.02, 0.935, textwrap.fill(sub, 230), ha="left", va="top", color=g.INK2, fontsize=12.5)
    note = ("PD-LoRA on the top 5% of Pile windows by LLS score (cat − animals), Goodfire's 4-layer 67M Pile model. "
            "Cat removed = share of the fine-tune's gain in log P(cat words) that is gone (mean over held-out Pile windows). "
            "KL(base || LoRA) = per-token KL from the base model on held-out Pile windows, with the top k off; middle panel = "
            "that as a share of the full LoRA's. Effects ranked on one held-out set (2,000 windows), measured on another. "
            "Dotted = random components switched off. One seed per run.")
    fig.text(0.02, 0.012, textwrap.fill(note, 260), ha="left", va="bottom", color=g.MUTED, fontsize=10)
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
