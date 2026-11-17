"""Dog -> cat PD-LoRA (KL to the dog_to_cat reference), necessity vs no necessity across the lam x beta sweep. Four
single-panel figures, two ways of switching components off:

  important components off (gate > 0.1, per token; dog_eval.json)
    results/dog_gate_off_dog_back.png      dog brought back = 1 - dog removed (off) / dog removed (full LoRA)
    results/dog_gate_off_kl_elsewhere.png  KL(base || LoRA) per token at every other position, full LoRA vs off
  top-k components by effect on log P(cat) at dog spots off at every token, k = 8 and 64 (dog_components.json)
    results/dog_topk_dog_back.png
    results/dog_topk_kl_elsewhere.png

  python3 dog_nonec_sweep_plot.py
"""
import json
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "lora"))
import goblin_plots as g  # noqa: E402  (shared style and palette)
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

plt = g.plt
RUNS = HERE / "results" / "pd_runs"
OUT = HERE / "results"
GRID = [(l, b) for l in ("3e-4", "1e-3", "3e-3") for b in ("0.2", "1")]  # weakest sparsity first
ARMS = [("pd_dog_distill" if (l, b) == ("1e-3", "0.2") else f"pd_dog_distill_nec_lam{l}_b{b}", l, b, True)
        for l, b in GRID] + [(f"pd_dog_distill_nonec_lam{l}_b{b}", l, b, False) for l, b in GRID]
XS = [i + (0.8 if not nec else 0) for i, (*_, nec) in enumerate(ARMS)]  # gap between the two arms
KS = (8, 64)
SETUP = ("PD-LoRA on Goodfire's 4-layer 67M Pile model, trained with KL to a reference = the base model with the "
         "probability of \" dog\", \" dogs\", \" Dog\" moved onto \" cat\", \" cats\", \" Cat\", on Pile windows "
         "containing dog (original text), 2,000 steps; necessity 4/2 vs none, everything else identical except λ "
         "(sparsity coefficient) and β (frequency-minimality weight). Measured on 2,000 held-out windows containing dog; "
         "dog spots = positions whose next token is a dog word. One seed per run. ")
BACK = ("Dog brought back = 1 − (1 − P_dog / P_dog,base) with the components off ÷ the same for the full LoRA; above "
        "100% = P(dog) above base. ")
GATE = "Important = gate > 0.1, per token; comps = important components per dog spot (of 768). "
TOPK = ("Components ranked on the first half of the windows by the drop in log P(cat words) at dog spots when each alone "
        "is switched off at every position; the top k then switched off together at every position, measured on the "
        "second half. ")


def gate_off(run):
    d = json.load(open(RUNS / run / "dog_eval.json"))
    m = d["modes"]
    back = 100 * (1 - m["imp_off"]["dog_removed"] / m["full"]["dog_removed"])
    return back, m["full"]["kl_base_else"], m["imp_off"]["kl_base_else"], d["important_per_token"]["dog_spots"]


def topk(run, k):
    """top-k by effect off: dog brought back (%), the full LoRA's KL elsewhere and the KL with the top k off"""
    rows = {(r["set"], r["k"]): r for r in json.load(open(RUNS / run / "dog_components.json"))["rows"]}
    f, r = rows["full", 0], rows["top", k]
    return 100 * (1 - r["dog_removed"] / f["dog_removed"]), f["kl_base_else"], r["kl_base_else"]


def figure(name, title, sub, subtitle, ylabel, note, draw, ylim, yticks=None, xlab=lambda run, lam, beta: ""):
    fig, ax = plt.subplots(figsize=(15, 7.2))
    fig.subplots_adjust(top=0.78, bottom=0.27, left=0.07, right=0.98)
    for x, (run, lam, beta, nec) in zip(XS, ARMS):
        draw(ax, x, run, g.ORANGE if nec else g.BLUE)
        ax.text(x, -0.04, f"λ {lam}\nβ {beta}" + xlab(run, lam, beta), transform=ax.get_xaxis_transform(),
                ha="center", va="top", color=g.INK2, fontsize=10)
    for x0, x1, label in ((XS[0], XS[5], "necessity 4/2"), (XS[6], XS[11], "no necessity")):
        ax.text((x0 + x1) / 2, 1.0, label, transform=ax.get_xaxis_transform(), ha="center", va="top", color=g.INK,
                fontsize=13, fontweight="bold")
    ax.set_ylim(*ylim)
    if yticks:
        ax.set_yticks(yticks)
    ax.set_ylabel(ylabel)
    ax.set_xticks([])
    ax.grid(axis="x", visible=False)
    ax.set_xlim(XS[0] - 0.6, XS[-1] + 0.6)
    g.subtitle(ax, subtitle)
    fig.suptitle(title, x=0.02, y=0.985, ha="left", fontsize=17, fontweight="bold", color=g.INK)
    fig.text(0.02, 0.92, textwrap.fill(sub, 165), ha="left", va="top", color=g.INK2, fontsize=12.5)
    fig.text(0.02, 0.012, textwrap.fill(SETUP + note, 205), ha="left", va="bottom", color=g.MUTED, fontsize=10)
    fig.savefig(OUT / f"{name}.png")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")
    return ax


def back_line(ax):
    ax.axhline(100, color=g.INK2, lw=1, ls="--", zorder=4)
    ax.annotate("dog fully back", ((XS[5] + XS[6]) / 2, 100), xytext=(0, 4), textcoords="offset points", ha="center",
                va="bottom", color=g.INK2, fontsize=10.5)


def main():
    comps = lambda run, lam, beta: f"\n{gate_off(run)[3]:.2g} comps"

    def draw_gate_back(ax, x, run, c):
        back = gate_off(run)[0]
        ax.bar(x, back, width=0.72, color=c, edgecolor=g.SURF, lw=2, zorder=3)
        ax.annotate(f"{back:.0f}%", (x, back), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                    color=g.INK, fontsize=11.5)
        if run == ARMS[0][0]:
            back_line(ax)
    figure("dog_gate_off_dog_back",
           "With necessity, switching off the important components undoes the whole dog-to-cat swap",
           "All 6 necessity runs bring dog fully back (104–107%, slightly above base). Without necessity 43–83% comes "
           "back, less as the sparsity penalty gets stronger.",
           "Switch off the important components at each token: how much of dog's probability comes back at dog spots?",
           "dog brought back (%)", BACK + GATE, draw_gate_back, (0, 125), [0, 20, 40, 60, 80, 100], comps)

    def draw_gate_kl(ax, x, run, c):
        _, kl_full, kl_off, _ = gate_off(run)
        ax.bar(x - 0.18, kl_full, width=0.36, color=c, alpha=0.4, edgecolor=g.SURF, lw=1.5, zorder=3)
        ax.bar(x + 0.18, kl_off, width=0.36, color=c, edgecolor=g.SURF, lw=1.5, zorder=3)
        ax.annotate(f"{kl_off:.4f}", (x + 0.18, kl_off), xytext=(0, 3), textcoords="offset points", ha="center",
                    va="bottom", color=g.INK, fontsize=9.5)
        if run == ARMS[0][0]:
            ax.legend([Patch(color=g.INK2, alpha=0.4), Patch(color=g.INK2)], ["full LoRA", "important components off"],
                      loc="upper left", bbox_to_anchor=(0, 0.9), fontsize=11, labelcolor=g.INK2)
    figure("dog_gate_off_kl_elsewhere",
           "Switching off the important components does no damage away from dog",
           "At every position that is not a dog spot, the model with the important components off is closer to the "
           "base model than the full LoRA is (KL 0.0008–0.0039 vs 0.010–0.013). Necessity runs get closest (≈0.001).",
           "Change from the base model at every other position: full LoRA (light) vs important components off (solid)",
           "KL(base || LoRA) per token", GATE + "KL per token at every position of the held-out dog windows that is "
           "not a dog spot.", draw_gate_kl, (0, 0.018), None, comps)

    def draw_topk_back(ax, x, run, c):
        w = 0.72 / len(KS)
        for j, k in enumerate(KS):
            back = topk(run, k)[0]
            xj = x + (j - (len(KS) - 1) / 2) * w
            ax.bar(xj, back, width=w, color=c, alpha=1 - 0.55 * j, edgecolor=g.SURF, lw=1.5, zorder=3)
            ax.annotate(f"{back:.0f}%", (xj, back), xytext=(0, 3), textcoords="offset points", ha="center",
                        va="bottom", color=g.INK, fontsize=9.5)
        if run == ARMS[0][0]:
            back_line(ax)
            ax.legend([Patch(color=g.INK2, alpha=1 - 0.55 * j) for j in range(len(KS))], [f"top {k} off" for k in KS],
                      loc="upper right", bbox_to_anchor=(1, 0.9), fontsize=11, labelcolor=g.INK2)
    b8 = [topk(r, 8)[0] for r, *_ in ARMS]
    b64 = [topk(r, 64)[0] for r, *_ in ARMS]
    figure("dog_topk_dog_back",
           "Components chosen by effect: with necessity, 8 components carry the swap",
           f"With necessity, the 8 components with the largest effect bring back {min(b8[:6]):.0f}–{max(b8[:6]):.0f}% "
           f"of dog. Without, {min(b8[6:]):.0f}–{max(b8[6:]):.0f}%, and even 64 components bring back only "
           f"{min(b64[6:]):.0f}–{max(b64[6:]):.0f}%.",
           "Switch off the top-k components by effect on cat, at every token: how much of dog comes back at dog spots?",
           "dog brought back (%)", BACK + TOPK, draw_topk_back, (0, 125), [0, 20, 40, 60, 80, 100])

    def draw_topk_kl(ax, x, run, c):
        w = 0.72 / len(KS)
        for j, k in enumerate(KS):
            _, kl_full, kl = topk(run, k)
            ax.bar(x + (j - (len(KS) - 1) / 2) * w, kl, width=w, color=c, alpha=1 - 0.55 * j, edgecolor=g.SURF,
                   lw=1.5, zorder=3)
        ax.hlines(kl_full, x - 0.36, x + 0.36, color=g.INK2, lw=1.2, ls=":", zorder=4)
        if run == ARMS[0][0]:
            ax.legend([Patch(color=g.INK2, alpha=1 - 0.55 * j) for j in range(len(KS))]
                      + [Line2D([], [], color=g.INK2, ls=":", lw=1.2)], [f"top {k} off" for k in KS] + ["full LoRA"],
                      loc="upper left", bbox_to_anchor=(0, 0.9), fontsize=11, labelcolor=g.INK2)
    figure("dog_topk_kl_elsewhere",
           "Switching off the top components by effect does no damage away from dog",
           "With the top 8 or 64 components off, the model is closer to base at every other position than the full "
           "LoRA is, in both arms (KL 0.001–0.004 vs 0.011–0.014).",
           "Change from the base model at every other position with the top k off (dotted line = full LoRA)",
           "KL(base || LoRA) per token", TOPK + "KL per token at every position of the held-out dog windows that is not "
           "a dog spot.", draw_topk_kl, (0, 0.018))


if __name__ == "__main__":
    main()
