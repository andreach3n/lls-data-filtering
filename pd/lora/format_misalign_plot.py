"""Plot format_misalign.py sweeps: rows = removal (from the fine-tuned model) and addition (from the original model),
columns = rankings; in each panel the share of the format effect and of the misalignment effect, against k.

  python format_misalign_plot.py runs_emd/B_bs8_2500/fm
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLS = [("format", "ranked by format"), ("misalignment", "ranked by misalignment"),
        ("mis_spec", "misalignment minus format"), ("mis_grad", "misalignment, plain gradient"), ("random", "random (3 seeds)")]
FC, MC = "#2a78d6", "#eb6834"
SURFACE, TEXT, TEXT2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def main():
    d = Path(sys.argv[1])
    r = json.load(open(d / "sweep.json"))
    cur = r["curves"]
    fig, axes = plt.subplots(2, len(COLS), figsize=(15, 6.4), dpi=180, sharex=True, sharey="row")
    fig.patch.set_facecolor(SURFACE)
    for row, (direc, ylab) in enumerate((("removal", "share of effect remaining"), ("addition", "share of effect recovered"))):
        for col, (key, title) in enumerate(COLS):
            ax = axes[row, col]
            ax.set_facecolor(SURFACE)
            runs = [cur[k][direc] for k in cur if k.startswith("random")] if key == "random" else [cur[key][direc]]
            for s, color, lab in (("F", FC, "format"), ("M", MC, "misalignment")):
                for i, run in enumerate(runs):
                    ax.plot([x["k"] for x in run], [x[s] for x in run], color=color, lw=2, marker="o", ms=3.5,
                            alpha=1 if len(runs) == 1 else 0.6, label=lab if i == 0 else None)
            ax.axhline(1 if direc == "removal" else 0, color=TEXT2, lw=0.8, ls=":")
            ax.axhline(0 if direc == "removal" else 1, color=TEXT2, lw=0.8, ls=":")
            ax.set_xscale("log", base=2)
            ax.grid(color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.tick_params(colors=TEXT2, labelsize=8)
            if row == 0:
                ax.set_title(title, fontsize=9.5, color=TEXT, loc="left")
            if col == 0:
                ax.set_ylabel(f"{direc}\n{ylab}", fontsize=9, color=TEXT)
            if row == 1:
                ax.set_xlabel("k components", fontsize=8.5, color=TEXT2)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncol=2, frameon=False, fontsize=9)
    fig.suptitle("Switching components off (top) or on (bottom) in groups of the top k by each ranking, held-out answers",
                 x=0.01, ha="left", fontsize=10.5, color=TEXT)
    fig.text(0.01, 0.005, f"Format effect = logP(fine-tuned aligned answer) - logP(original answer): {r['F_base']:.2f} -> {r['F_full']:.2f} nats/token. "
             f"Misalignment effect = logP(misaligned) - logP(aligned fine-tuned answer): {r['M_base']:.2f} -> {r['M_full']:.2f}. "
             "Teacher-forced under run B.", fontsize=7.5, color=TEXT2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.94))
    fig.savefig(d / "sweep.png", facecolor=SURFACE)
    print("wrote", d / "sweep.png")


if __name__ == "__main__":
    main()
