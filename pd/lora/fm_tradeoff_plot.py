"""EM rate against median answer length for run B under group ablations (format_misalign.py masks), with the original
assistant, the full fine-tune and a random group as references. Bootstrapped 95% intervals on EM over answers.

  python fm_tradeoff_plot.py --out runs_emd/B_bs8_2500/fm/tradeoff.png
"""
import argparse
import random
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from em_summary_plot import outcomes  # noqa: E402
from gen_format_stats import stats  # noqa: E402

R = Path("runs_emd/B_bs8_2500/mask_eval")
PTS = [("runs_emd/original_assistant/base", "original assistant", "#2a78d6"), (R / "full", "full fine-tune", "#0b0b0b"),
       (R / "mask_random0_k2048", "random 2048 off", "#898781"),
       (R / "mask_mis_spec_k256", "mis.-specific 256 off", "#eb6834"), (R / "mask_mis_spec_k512", "mis.-specific 512 off", "#eb6834"),
       (R / "mask_mis_spec_k1024", "mis.-specific 1024 off", "#eb6834"), (R / "mask_mis_spec_k2048", "mis.-specific 2048 off", "#eb6834"),
       (R / "mask_misalignment_k512", "misalignment 512 off", "#c2410c"), (R / "mask_fmt_spec_k4096", "format-specific 4096 off", "#1baf7a")]
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rng = random.Random(0)
    fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    spec = []
    for d, lab, color in PTS:
        d = Path(d)
        em = outcomes(d)["em"]
        s = stats(d)
        m = sum(em) / len(em)
        bs = sorted(sum(rng.choices(em, k=len(em))) / len(em) for _ in range(2000))
        x = s["median_words"]
        ax.errorbar([x], [m], yerr=[[m - bs[50]], [bs[1949] - m]], fmt="none", ecolor=color, elinewidth=1.5, alpha=0.7)
        ax.scatter([x], [m], s=46, color=color, edgecolors=SURFACE, linewidths=2, zorder=3)
        off = {"mis.-specific 256 off": (8, -12, "left"), "format-specific 4096 off": (6, 6, "left")}.get(lab, (6, 4, "left"))
        ax.annotate(lab, (x, m), xytext=off[:2], textcoords="offset points", fontsize=7.5, color=TEXT2, ha=off[2])
        if "mis.-specific" in lab:
            spec.append((x, m))
    ax.plot(*zip(*spec), color="#eb6834", lw=1.2, alpha=0.6, zorder=2)
    ax.set_xscale("log")
    ax.set_xticks([40, 60, 100, 150, 200])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.set_xlabel("median answer length (words)", fontsize=9, color=TEXT2)
    ax.set_ylabel("misaligned & coherent", fontsize=9, color=TEXT2)
    ax.set_title("Removing misalignment components lowers EM faster than it restores the original format",
                 fontsize=9.5, color=TEXT, loc="left")
    ax.grid(color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=TEXT2, labelsize=8)
    fig.text(0.01, 0.01, "Run B, 8 main questions (x50) + 8 medical (x20); GPT-4o judge. Components are switched off at every token. "
             "Bars: bootstrapped 95% intervals.", fontsize=6.5, color=TEXT2)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(a.out, facecolor=SURFACE)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
