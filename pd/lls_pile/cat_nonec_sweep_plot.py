"""PD-LoRA on the LLS cat selection (4L Pile model), with necessity (one run) vs without it across a lam x beta sparsity
sweep. With the important components switched off:
  results/cat_nonec_sweep.png       how much of the cat gain is removed (eval.json)
  results/cat_nonec_sweep_kept.png  how much of the rest of the fine-tune is still there (imp_off_check.json)
and with only the top-k components by effect on P(cat) switched off (topk_removal.json), k = 8 and 128:
  results/cat_nonec_topk_removed.png, results/cat_nonec_topk_kept.png

  python3 cat_nonec_sweep_plot.py
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
# (run, lam, beta, necessity); no-necessity runs ordered by sparsity, weakest first
ARMS = [("pd_cat", "1e-3", "0.2", True)] + [
    ("pd_cat_nonec" if (l, b) == ("1e-3", "0.2") else f"pd_cat_nonec_lam{l}_b{b}", l, b, False)
    for l in ("3e-4", "1e-3", "3e-3") for b in ("0.2", "1")]
NOTE = ("PD-LoRA on the top 5% of Pile windows by LLS score (cat − animals), Goodfire's 4-layer 67M Pile model, 2,000 "
        "steps. Important = gate > 0.1. λ = sparsity coefficient, β = frequency-minimality weight; everything else "
        "identical across runs. Comps/token = important components per training token, of 768. One seed per run. ")


def cat_removed(run):
    """share (%) of the full LoRA's gain in log P(cat words) that is gone with the important components off"""
    m = json.load(open(RUNS / run / "eval.json"))["modes"]
    pb, pf, po = m["base"]["p_cat"], m["full"]["p_cat"], m["imp_off"]["p_cat"]
    return 100 * math.log(pf / po) / math.log(pf / pb)


def kept(run):
    """KL(base || important off) as a share (%) of KL(base || full LoRA), per token on held-out 32-token Pile windows"""
    k = json.load(open(RUNS / run / "imp_off_check.json"))["kl_to_base"]["val_32"]
    return 100 * k["imp_off"]["mean"] / k["full"]["mean"]


def topk(run, k):
    """top-k components by effect on P(cat) switched off: cat removed (%) and rest of the fine-tune kept (%), as
    cat_topk_plot.py"""
    rows = {(r["set"], r["k"]): r for r in json.load(open(RUNS / run / "topk_removal.json"))["rows"]}
    pb, pf, klf = rows["base", 0]["p_cat"], rows["full", 0]["p_cat"], rows["full", 0]["kl_from_base"]
    r = rows["top", k]
    return 100 * math.log(pf / r["p_cat"]) / math.log(pf / pb), 100 * r["kl_from_base"] / klf


def figure(value, out, ylabel, subtitle, title, sub, note, ref_label, series=None):
    """one bar per run; with series (labels), value returns one number per series and each run gets a group of bars,
    the later series lighter"""
    fig, ax = plt.subplots(figsize=(13, 6.8))
    fig.subplots_adjust(top=0.8, bottom=0.27, left=0.07, right=0.98)
    for i, (run, lam, beta, nec) in enumerate(ARMS):
        x = i + (0 if nec else 0.6)  # gap between the necessity bar and the sweep
        ys = value(run) if series else [value(run)]
        k = json.load(open(RUNS / run / "eval.json"))["important_per_token"]["train"]
        w = 0.7 / len(ys)
        for j, y in enumerate(ys):
            xj = x + (j - (len(ys) - 1) / 2) * w
            ax.bar(xj, y, width=w, color=g.ORANGE if nec else g.BLUE, alpha=1 - 0.55 * j, edgecolor=g.SURF, lw=2,
                   zorder=3)
            ax.annotate(f"{y:.0f}%", (xj, y), xytext=(0, 5), textcoords="offset points", ha="center", va="bottom",
                        color=g.INK, fontsize=12.5 if len(ys) == 1 else 10.5, fontweight="bold" if nec else "normal")
        ax.text(x, -0.04, f"{'necessity' if nec else 'no necessity'}\nλ {lam}, β {beta}\n{k:.2g} comps/token",
                transform=ax.get_xaxis_transform(), ha="center", va="top", color=g.INK2, fontsize=10.5)
    ax.axhline(100, color=g.INK2, lw=1, ls="--", zorder=4)
    ax.annotate(ref_label, (0.8, 100), xytext=(0, 4), textcoords="offset points", ha="center",  # in the gap, clear
                va="bottom", color=g.INK2, fontsize=11)                                         # of the bar labels
    if series:
        from matplotlib.patches import Patch
        ax.legend([Patch(color=g.INK2, alpha=1 - 0.55 * j) for j in range(len(series))], series, loc="upper right",
                  fontsize=11, labelcolor=g.INK2, ncol=len(series))
    ax.set_xticks([])
    ax.set_xlim(-0.6, len(ARMS) + 0.25)
    ax.set_ylim(0, 120)
    ax.set_ylabel(ylabel)
    ax.grid(axis="x", visible=False)
    g.subtitle(ax, subtitle)
    fig.suptitle(title, x=0.02, y=0.985, ha="left", fontsize=17, fontweight="bold", color=g.INK)
    fig.text(0.02, 0.93, textwrap.fill(sub, 150), ha="left", va="top", color=g.INK2, fontsize=12.5)
    fig.text(0.02, 0.012, textwrap.fill(NOTE + note, 185), ha="left", va="bottom", color=g.MUTED, fontsize=10)
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


def main():
    figure(cat_removed, HERE / "results" / "cat_nonec_sweep.png", "cat removed (%)",
           "Share of the fine-tune's P(cat) gain removed by switching off the important components",
           "Without necessity, switching off the important components removes at most 38% of the cat shift",
           "No sparsity setting in a 3×2 sweep fixes it, including one with 12 important components per token, over "
           "twice the necessity run's 5. With necessity, P(cat) goes back to base (slightly below: 106%).",
           "Cat removed = share of the full LoRA's gain in log P(cat words) (mean over held-out Pile windows) that is "
           "gone with the important components off; above 100% = below base.", "back to base")
    figure(kept, HERE / "results" / "cat_nonec_sweep_kept.png", "rest of the fine-tune kept (%)",
           "Share of the fine-tune's change from base still there after switching off the important components",
           "With necessity, switching off the important components also takes 73% of the rest of the fine-tune",
           "The necessity run removes cat completely but keeps only 27% of its other changes. The no-necessity runs "
           "keep 63–98%, but mostly because they remove little of anything: at most 38% of cat.",
           "Kept = KL(base || LoRA with important components off) / KL(base || full LoRA), per token on 2,000 held-out "
           "32-token Pile windows (the training window length); 0% = back to base. On 128-token windows: necessity "
           "61%, no necessity 82–99%.", "whole fine-tune kept")
    ks = (8, 128)
    series = [f"top {k} off" for k in ks]
    topk_note = ("Components ranked by the drop in log P(cat words) when each alone is switched off (on 2,000 held-out "
                 "32-token Pile windows), then the top k switched off at every position and measured on 2,000 others. ")
    figure(lambda r: [topk(r, k)[0] for k in ks], HERE / "results" / "cat_nonec_topk_removed.png", "cat removed (%)",
           "Share of the fine-tune's P(cat) gain removed by switching off the top-k components by effect on cat",
           "Switching off the 8 most cat-affecting components removes all of cat with necessity, about 40% without",
           "Without necessity, even the top 128 components (of 768) remove at most 90% of the cat gain, whatever the "
           "sparsity setting.",
           topk_note + "Cat removed = share of the full LoRA's gain in log P(cat words) that is gone; above 100% = below "
           "base.", "back to base", series)
    figure(lambda r: [topk(r, k)[1] for k in ks], HERE / "results" / "cat_nonec_topk_kept.png",
           "rest of the fine-tune kept (%)",
           "Share of the fine-tune's change from base still there after switching off the top-k components by effect on cat",
           "Removing cat with necessity still takes about 70% of the rest of the fine-tune",
           "With necessity, the 8 components that remove all of cat leave 32% of the fine-tune's other changes. Without "
           "necessity, the top 128 leave 56–68% but remove only 70–90% of cat.",
           topk_note + "Kept = KL(base || LoRA with the top k off) / KL(base || full LoRA), per token; 0% = back to base. "
           "The full no-necessity LoRAs move further from base (KL 0.084–0.087 vs 0.050), so the same share is more "
           "change in absolute terms.", "whole fine-tune kept", series)


if __name__ == "__main__":
    main()
