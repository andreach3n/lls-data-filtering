"""Cat removed vs the rest of the fine-tune removed when the components that carry the rest are protected
(cat_protect.py), necessity vs no necessity. Reads results/pd_runs/{pd_cat,pd_cat_nonec}/cat_protect.json, writes
results/cat_protect.png.

  python3 cat_protect_plot.py
"""
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "lora"))
import goblin_plots as g  # noqa: E402  (shared style and palette)

plt = g.plt
RUNS = HERE / "results" / "pd_runs"
ARMS = [("pd_cat_nonec", "No necessity (λ 1e-3, β 0.2)"), ("pd_cat", "Necessity 4 / 2 (λ 1e-3, β 0.2)")]


def pt(d, c):
    s = d["share"][c]
    return 100 * s["rest_removed"], 100 * s["cat_removed"]


def sizes(d, prefix):
    return sorted(int(c.split("#")[1]) for c in d["share"] if c.startswith(prefix + "#"))


def main():
    fig, axs = plt.subplots(1, 2, figsize=(13, 6.8), sharey=True)
    for ax, (r, name) in zip(axs, ARMS):
        d = json.load(open(RUNS / r / "cat_protect.json"))
        ks = sizes(d, "keep_rest")
        xs, ys = zip(*[pt(d, "per_token_imp_off")] + [pt(d, f"keep_rest#{k}") for k in ks])
        ax.plot(xs, ys, color=g.AQUA, lw=2, marker="o", ms=7, mec=g.SURF, mew=1.2, zorder=4,
                label="each token's important components off, except the top k for the rest kept on")
        ax.annotate("none kept on", (xs[0], ys[0]), xytext=(8, -6), textcoords="offset points", ha="left",
                    va="top", color=g.INK2, fontsize=10)
        ax.annotate(f"top {ks[-1]} for the rest\nkept on", (xs[-1], ys[-1]), xytext=(30, 10),
                    textcoords="data", ha="left", va="center", arrowprops=dict(arrowstyle="-", color=g.MUTED, lw=0.8), color=g.INK2, fontsize=10)
        ks = sizes(d, "cat_minus_rest")
        ax.plot(*zip(*[pt(d, f"cat_minus_rest#{k}") for k in ks]), color=g.ORANGE, lw=2, marker="s", ms=7,
                mec=g.SURF, mew=1.2, zorder=3, label="top k for cat minus the top k for the rest, off at every token")
        ks = sizes(d, "top")
        ax.plot(*zip(*[pt(d, f"top#{k}") for k in ks]), color=g.BLUE, lw=2, marker="o", ms=7, mec=g.SURF, mew=1.2,
                zorder=3, ls="--", label="top k for cat, off at every token")
        for k in (8, 512):
            ax.annotate(f"{k} off", pt(d, f"top#{k}"), xytext=(0, 9), textcoords="offset points", ha="center",
                        va="bottom", color=g.INK2, fontsize=10)
        ax.text(2, 104, "ideal", ha="left", va="bottom", color=g.MUTED, fontsize=11)
        ax.set_xlim(-3, 105)
        ax.set_ylim(-5, 112)
        ax.set_title(name, fontsize=13)
        ax.set_xlabel("rest of the fine-tune removed (%)")
    axs[0].set_ylabel("cat removed (%)")
    h, l = axs[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", bbox_to_anchor=(0.02, 0.9), ncol=1, fontsize=11.5, handletextpad=0.3)
    fig.subplots_adjust(top=0.72, bottom=0.2, left=0.07, right=0.98, wspace=0.08)
    g.OUT = HERE / "results"
    g.finish(fig, None, "cat_protect", title=TITLE,
             note="PD-LoRA on the top 5% of Pile windows by LLS score (cat − animals), Goodfire's 4-layer 67M Pile model. "
                  "Both rankings from switching each component alone off at every position on 2,000 held-out 32-token "
                  "Pile windows: cat score = drop in log P(cat words), rest score = drop in KL(base || LoRA). Important = "
                  "gate > 0.1 at that position. Measured on 2,000 other windows: cat removed = share of the full LoRA's "
                  "gain in log P(cat words) that is gone; rest removed = 100% − KL(base || ·) / KL(base || full LoRA). "
                  "One seed per run.")


TITLE = "Cat: the components that carry the rest of the fine-tune also carry cat"

if __name__ == "__main__":
    main()
