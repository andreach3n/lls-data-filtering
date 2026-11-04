"""Compare mirrored-necessity MNIST models across the minimality settings.

Two questions:
  1. beta sweep: does a larger frequency-minimality weight beta make layer-1 components exclusive to one digit
     (switch off one digit's components -> only that digit breaks)?
  2. lambda control: or does any sparser model look like that? Same sparsity reached by raising lambda at beta 0.2.

Definitions as in specificity.py (a digit's components = layer-1 components with mean gate > 0.5 on that digit;
layer 2 fully on). Writes runs/analysis/sweeps.json, sweeps_table.txt, beta_sweep.png, lambda_control.png.

  python sweeps.py               (recompute everything, a few minutes)
  python sweeps.py --plot-only   (redraw the plots from runs/analysis/sweeps.json)
"""
import json
import statistics as st
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # gated_lora
import train as M  # noqa: E402
import specificity as S  # noqa: E402

R = HERE / "runs"
OUT = R / "analysis"
SEEDS = [0, 1, 2]
D = 7
# lambda 0.003, beta 0.2 and necessity adversary 5 x 0.05 unless the label says; "no nec" = plain VPD (no necessity)
RUNS = [  # label, series, folder (experiment/variant/condition)
    ("β 0.2", "beta", "06_mirrored_necessity/mirror/vpd_nec_mirror"),
    ("β 1", "beta", "07_beta_sweep/mirror_b1/vpd_nec_mirror"),
    ("β 3", "beta", "07_beta_sweep/mirror_b3/vpd_nec_mirror"),
    ("β 10", "beta", "07_beta_sweep/mirror_b10/vpd_nec_mirror"),
    ("β 3, adv 10×0.1", "adv", "07_beta_sweep/mirror_b3_adv10/vpd_nec_mirror"),
    ("β 10, adv 10×0.1", "adv", "07_beta_sweep/mirror_b10_adv10/vpd_nec_mirror"),
    ("β 10, nec ×2", "adv", "07_beta_sweep/mirror_b10_w2/vpd_nec_mirror"),
    ("β 10, both", "adv", "07_beta_sweep/mirror_b10_both/vpd_nec_mirror"),
    ("λ 0.01", "lambda", "08_lambda_control/mirror_lam0.01/vpd_nec_mirror"),
    ("λ 0.03", "lambda", "08_lambda_control/mirror_lam0.03/vpd_nec_mirror"),
    ("λ 0.1", "lambda", "08_lambda_control/mirror_lam0.1/vpd_nec_mirror"),
    ("no nec, default", "nonec", "09_no_necessity/vpd_default/vpd"),
    ("no nec, β 10", "nonec", "09_no_necessity/vpd_b10/vpd"),
    ("no nec, λ 0.1", "nonec", "09_no_necessity/vpd_lam0.1/vpd"),
]
KEYS = [("acc", "test accuracy", "{:.3f}"), ("l0", "important per image", "{:.1f}"),
        ("l0_fc1", "  layer 1", "{:.1f}"), ("l0_fc2", "  layer 2", "{:.1f}"),
        ("alive_fc1", "layer-1 components in use", "{:.0f}"), ("alive_fc2", "layer-2 components in use", "{:.0f}"),
        ("comps_per_digit", "strong layer-1 comps per digit", "{:.1f}"),
        ("own_lost", "own digit lost (grid)", "{:.2f}"), ("others_lost", "other digits lost (grid)", "{:.2f}"),
        ("seven_kept", "7s kept, 7-comps off", "{:.2f}"),
        ("rem_acc", "acc, important removed", "{:.3f}"), ("rem_l1_acc", "acc, layer-1 important off", "{:.3f}"),
        ("adv_l1_p", "adversarial p(true), layer 1", "{:.2f}"), ("adv_l2_p", "adversarial p(true), layer 2", "{:.2f}"),
        ("adv_all_p", "adversarial p(true), both", "{:.2f}")]


def collect():
    data = M.load_mnist()
    rows = []
    for label, series, folder in RUNS:
        for s in SEEDS:
            run = R / folder
            ev = json.load(open(run / f"s{s}.json"))
            curves, grid, sizes = S.analyse(run / f"s{s}.pt", data)
            lost = 1 - grid
            off = lost[~np.eye(10, dtype=bool)]
            c7 = curves[D]
            rows.append({
                "label": label, "series": series, "folder": folder, "seed": s, "acc": ev["full"]["acc"],
                "lam": ev["args"]["lam"], "beta": ev["args"]["beta"],
                "l0": ev["l0"]["total"], "l0_fc1": ev["l0"]["fc1"], "l0_fc2": ev["l0"]["fc2"],
                "alive_fc1": ev["alive"]["fc1"], "alive_fc2": ev["alive"]["fc2"],
                "adv_l1_p": ev["comp_adv_fc1"]["p_true"], "adv_l2_p": ev["comp_adv_fc2"]["p_true"],
                "adv_all_p": ev["comp_adv"]["p_true"],
                "rem_acc": ev["imp_removed"]["acc"], "rem_l1_acc": ev["imp_removed_fc1"]["acc"],
                "own_lost": float(np.diag(lost).mean()), "others_lost": float(off.mean()),
                "comps_per_digit": float(np.mean(sizes)),
                "seven_kept": float(c7[-1, D]),
                "seven_others_broken": [j for j in range(10) if j != D and c7[-1, j] < 0.5],
                "grid": grid.tolist(),
            })
    return rows


def table(rows):
    labels = [r[0] for r in RUNS]
    lines = [f"{'':32s}" + "".join(f"{lab:>18s}" for lab in labels)]
    for k, name, f in KEYS:
        line = f"{name:32s}"
        for lab in labels:
            v = [r[k] for r in rows if r["label"] == lab]
            line += f"{f.format(st.mean(v)):>11s} ±{st.pstdev(v):5.2f}"
        lines.append(line)
    lines.append("")
    for lab in labels:
        lines.append(f"{lab}: other digits broken with the 7s, per seed: "
                     f"{[r['seven_others_broken'] for r in rows if r['label'] == lab]}")
    return "\n".join(lines)


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": S.INK, "axes.labelcolor": S.INK2,
                         "xtick.color": S.INK2, "ytick.color": S.INK2, "axes.edgecolor": S.GRID,
                         "figure.facecolor": S.SURFACE, "axes.facecolor": S.SURFACE})
    return plt


def plot_beta(rows, plt):
    labels = [lab for lab, series, _ in RUNS if series in ("beta", "adv")]
    xs = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.9))
    panels = [("others_lost", "Other digits lost when one digit's\ncomponents are off (want 0%)", True),
              ("own_lost", "Own digit lost\n(want 100%)", True),
              ("l0", "Important components per image\n(fewer = shorter explanations)", False)]
    for ax, (k, title, pct) in zip(axes, panels):
        for i, lab in enumerate(labels):
            v = [r[k] for r in rows if r["label"] == lab]
            ax.scatter([i] * len(v), v, color=S.OTHER, s=22, zorder=2)
            ax.scatter([i], [st.mean(v)], color=S.HL, s=60, zorder=3, marker="_", linewidths=3)
        ax.set_xticks(xs, [lab.replace(", ", "\n") for lab in labels], fontsize=7.5)
        ax.set_title(title, loc="left", fontsize=9.5, color=S.INK)
        if pct:
            ax.set_ylim(-0.03, 1.05)
            ax.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
        S.style_axes(ax)
    axes[0].scatter([], [], color=S.OTHER, s=22, label="one seed")
    axes[0].scatter([], [], color=S.HL, s=60, marker="_", linewidths=3, label="mean of 3 seeds")
    axes[0].legend(frameon=False, fontsize=8, loc="upper right")
    fig.suptitle("Does charging more for frequently used components (β) make digits' components exclusive?",
                 x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.01, "Mirrored necessity, 128/32 components, λ 0.003. Necessity adversary 5 × 0.05 unless labelled; "
             "nec ×2 = necessity weights doubled. A digit's components = layer-1 components with mean gate > 0.5 on it; "
             "layer 2 fully on.", fontsize=7.5, color=S.INK2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    fig.savefig(OUT / "beta_sweep.png", dpi=150)
    plt.close(fig)


def plot_lambda(rows, plt):
    """Other digits lost against important components per image: beta series vs lambda series at matched sparsity."""
    series = [("beta", "raise β (λ 0.003)", "#2a78d6"), ("lambda", "raise λ (β 0.2)", "#eb6834")]
    fig, ax = plt.subplots(figsize=(7.6, 5))
    for key, name, col in series:
        labs = [lab for lab, s, _ in RUNS if s == key]
        if key == "lambda":
            labs = ["β 0.2"] + labs  # both series start from the same model
        mx, my = [], []
        for lab in labs:
            v = [r for r in rows if r["label"] == lab]
            ax.scatter([r["l0"] for r in v], [r["others_lost"] for r in v], color=col, s=16, alpha=0.35, zorder=2,
                       linewidths=0)
            mx.append(st.mean(r["l0"] for r in v))
            my.append(st.mean(r["others_lost"] for r in v))
        ax.plot(mx, my, color=col, lw=2, zorder=3, label=name)
        ax.scatter(mx, my, color=col, s=40, edgecolors=S.SURFACE, linewidths=1.5, zorder=4)
        for x, y, lab in zip(mx, my, labs):
            if lab == "β 0.2" and key == "lambda":
                continue
            ax.annotate(lab, (x, y), xytext=(6, 5), textcoords="offset points", fontsize=8, color=S.INK)
    ax.set_xscale("log")
    ax.set_xticks([2, 3, 5, 8, 14], ["2", "3", "5", "8", "14"])
    ax.minorticks_off()
    ax.set_ylim(-0.03, 1.0)
    ax.set_yticks([0, 0.25, 0.5, 0.75], ["0%", "25%", "50%", "75%"])
    ax.set_xlabel("important components per image (log scale)")
    ax.set_ylabel("other digits lost when one digit's components are off")
    S.style_axes(ax)
    ax.grid(axis="x", color=S.GRID, lw=0.8)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    ax.set_title("At the same sparsity, does β or λ give more exclusive components?", loc="left", fontsize=11,
                 fontweight="bold", color=S.INK, pad=12)
    fig.text(0.01, 0.01, "Mirrored necessity, adversary 5 × 0.05. Big dots = mean of 3 seeds, faint dots = single seeds. "
             "\nLower is better (0% = turning off one digit's components leaves the others intact).",
             fontsize=7, color=S.INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(OUT / "lambda_control.png", dpi=150)
    plt.close(fig)


def main():
    if "--plot-only" in sys.argv:  # redraw from the saved numbers
        rows = json.load(open(OUT / "sweeps.json"))
    else:
        rows = collect()
        json.dump(rows, open(OUT / "sweeps.json", "w"))
    t = table(rows)
    (OUT / "sweeps_table.txt").write_text(t + "\n")
    print(t)
    plt = setup_mpl()
    plot_beta(rows, plt)
    plot_lambda(rows, plt)
    print(OUT / "beta_sweep.png", OUT / "lambda_control.png")


if __name__ == "__main__":
    main()
