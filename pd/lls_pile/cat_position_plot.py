"""Where in the window the top cat components fire: share of held-out windows with gate > 0.1 at each token position,
for the top 8 components by effect on P(cat), with and without necessity. Reads results/pd_runs/*/freq_vs_effect.json,
writes results/cat_positions.png.

  python3 cat_position_plot.py
"""
import json
import math
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "lora"))
import goblin_plots as g  # noqa: E402  (shared style and palette)
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

plt = g.plt
RUNS = HERE / "results" / "pd_runs"
OUT = HERE / "results" / "cat_positions.png"
CMAP = LinearSegmentedColormap.from_list("blue", [g.SURF, "#a9c8ef", g.BLUE, "#123f78"])


def short(name):
    layer, rest = name.replace("model.layers.", "").split(".", 1)
    return f"L{layer} {rest.split('.', 1)[1].replace('_proj', '').replace('c_fc', 'up')}"


def main():
    runs = [("pd_cat", "Necessity 4/2"), ("pd_cat_nonec", "No necessity")]
    fig, axes = plt.subplots(1, 2, figsize=(17.5, 6.4))
    fig.subplots_adjust(top=0.76, bottom=0.2, left=0.08, right=0.8, wspace=0.62)
    for ax, (run, title) in zip(axes, runs):
        d = json.load(open(RUNS / run / "freq_vs_effect.json"))
        pb, pf = d["base"]["p_cat"], d["full"]["p_cat"]
        rm = lambda p: 100 * math.log(pf / p) / math.log(pf / pb)
        rows = d["position"]
        M = [r["fire_by_position"] for r in rows]
        im = ax.imshow(M, aspect="auto", cmap=CMAP, vmin=0, vmax=1, interpolation="nearest")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([short(r["component"]) for r in rows], fontsize=11.5, color=g.INK)
        ax.set_xticks([0, 1, 2, 4, 8, 16, 24, 31])
        ax.set_xlabel("token position in the 32-token window")
        ax.grid(False)
        ax.spines[["left", "bottom"]].set_visible(False)
        ax.tick_params(length=0)
        # cat removed by switching the component off everywhere, and at position 0 only
        for i, r in enumerate(rows):
            ax.text(33, i, f"{rm(r['off_all']['p_cat']):.0f}%", va="center", ha="left", fontsize=11.5, color=g.INK)
            ax.text(40, i, f"{rm(r['off_pos0']['p_cat']):.0f}%", va="center", ha="left", fontsize=11.5, color=g.INK2)
        ax.text(33, -0.7, "cat removed,\noff everywhere", va="bottom", ha="left", fontsize=9.5, color=g.MUTED)
        ax.text(40, -0.7, "off at\npos 0 only", va="bottom", ha="left", fontsize=9.5, color=g.MUTED)
        ax.set_xlim(-0.5, 31.5)
        ax.text(0, 1.1, title, transform=ax.transAxes, fontsize=14, fontweight="bold", color=g.INK)
        ax.text(0, 1.02, "top 8 components by effect on P(cat), largest first", transform=ax.transAxes,
                fontsize=11.5, color=g.INK2)
    cax = fig.add_axes([0.925, 0.25, 0.009, 0.45])
    cb = fig.colorbar(im, cax=cax)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=0, colors=g.MUTED)
    cb.set_label("share of windows where the gate is > 0.1", color=g.INK2)

    fig.suptitle("The main cat components fire only at the first token of the window",
                 x=0.02, y=0.985, ha="left", fontsize=17, fontweight="bold", color=g.INK)
    sub = ("Attention value components (v) fire at position 0 in ~99% of windows and almost never elsewhere; switching "
           "them off at position 0 alone removes all of their cat effect. Without necessity, most of the top 8 are MLP "
           "components whose gates almost never fire, each carrying ~1% of cat.")
    fig.text(0.02, 0.935, textwrap.fill(sub, 175), ha="left", va="top", color=g.INK2, fontsize=12.5)
    note = ("2,000 held-out Pile windows of 32 tokens (cut from inside documents, so position 0 is an ordinary token, not "
            "a start token). Cat removed = share of the fine-tune's gain in log P(cat words) that is gone with that one "
            "component off. v = attention value, down = MLP output. One seed per run.")
    fig.text(0.02, 0.012, textwrap.fill(note, 200), ha="left", va="bottom", color=g.MUTED, fontsize=10)
    fig.savefig(OUT)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
