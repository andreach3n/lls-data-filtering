"""Companion to tradeoff_<D>_dial.png: as digit D's layer-1 components are turned down, how confident is the model
on D's images, and what does it call them instead?

Same masks as specificity.py / tradeoff.py (D's components = layer-1 components with mean gate > 0.5 on D; layer 2
fully on). Top row: mean p(D) and mean p(top digit) on D's test images vs the mask. The necessity loss asks
for p(full model's prediction) = p(D) <= 0.1; it puts no limit on p(top digit). Bottom row: with D's components fully off, which digit D's images are classified as.

  python confidence.py [--digit 7] [--plot-only]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # gated_lora
import train as M  # noqa: E402
import specificity as S  # noqa: E402
import tradeoff as T  # noqa: E402

R = HERE / "runs"
OUT = R / "analysis"
SEEDS = [0, 1, 2]
UNIF = "Mirrored necessity, uniform target, β = 10, adversary 10×0.1"
MODELS = T.MODELS[:3] + [(UNIF, "10_uniform_mirrored/b10_adv10/vpd_nec_mirror_unif/s{s}.pt", T.C.NEC["sparse"])] \
    + T.MODELS[3:]
SHORT = {**T.SHORT, UNIF: "Necessity, uniform target, β 10"}


@torch.no_grad()
def analyse(path, data, D):
    ck = torch.load(path, map_location=M.dev)
    a = type("A", (), ck["args"])
    model, wr, ci = M.build(a, 0)
    model.load_state_dict(ck["model"])
    ci.load_state_dict(ck["ci"])
    x, y = data["test_x"], data["test_y"]
    C1, C2 = wr["fc1"].C, wr["fc2"].C
    g1 = M.gates_of(model, wr, ci, x)["fc1"][:, 0]
    group = g1[y == D].mean(0) > S.GATE_MIN
    xd = x[y == D]
    N = len(xd)
    p_d, p_top = [], []
    for t in S.DOSES:
        m = torch.ones(C1)
        m[group] = float(t)
        p = M.fwd(model, wr, xd, "masked", {"fc1": m.expand(N, 1, C1), "fc2": torch.ones(N, 1, C2)}).float().softmax(-1)
        p_d.append(p[:, D].mean().item())
        p_top.append(p.max(-1).values.mean().item())
    pred = p.argmax(-1)  # at mask 0 (last dose)
    where = torch.bincount(pred, minlength=10).float() / N
    return {"p_digit": p_d, "p_top": p_top, "pred_at_0": where.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--digit", type=int, default=7)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    D = a.digit
    path = OUT / f"confidence_{D}.json"
    if a.plot_only:
        res = json.load(open(path))["models"]
    else:
        data = M.load_mnist()
        res = {label: [analyse(R / pat.format(s=s), data, D) for s in SEEDS] for label, pat, _ in MODELS}
        json.dump({"doses": S.DOSES.tolist(), "digit": D, "models": res}, open(path, "w"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": S.INK, "axes.labelcolor": S.INK2,
                         "xtick.color": S.INK2, "ytick.color": S.INK2, "axes.edgecolor": S.GRID,
                         "figure.facecolor": S.SURFACE, "axes.facecolor": S.SURFACE})
    n = len(MODELS)
    fig, axes = plt.subplots(2, n, figsize=(17.5, 7.6), gridspec_kw={"height_ratios": [1.35, 1]}, sharey="row")
    x = S.DOSES
    ticks = [0, 0.25, 0.5, 0.75, 1]
    for j, (label, _, col) in enumerate(MODELS):
        rs = res[label]
        ax = axes[0, j]
        ax.axhline(0.1, color=S.INK2, lw=1, ls=(0, (4, 3)), zorder=1)
        for r in rs:
            ax.plot(x, r["p_top"], color=S.INK2, lw=0.8, alpha=0.35, ls=(0, (2, 2)), zorder=2)
            ax.plot(x, r["p_digit"], color=col, lw=0.8, alpha=0.45, zorder=2)
        top = np.mean([r["p_top"] for r in rs], 0)
        pd = np.mean([r["p_digit"] for r in rs], 0)
        ax.plot(x, top, color=S.INK2, lw=2, ls=(0, (2, 2)), zorder=3)
        ax.plot(x, pd, color=col, lw=2.8, zorder=4, solid_capstyle="round")
        lo, hi = sorted([(pd[-1], f"p({D}) {pd[-1]:.2f}", col), (top[-1], f"p(top) {top[-1]:.2f}", S.INK)])
        if hi[0] - lo[0] < 0.12:
            lo = (hi[0] - 0.12, *lo[1:])
        for yv, txt, c in (lo, hi):
            ax.annotate(txt, (0, yv), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9,
                        color=c, fontweight="bold", annotation_clip=False)
        ax.set_title(SHORT[label], loc="left", fontsize=10.5, fontweight="bold")
        ax.set_xlim(1.02, -0.02)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xticks([1, 0.5, 0], ["1\n(on)", "0.5", "0\n(off)"])
        ax.set_yticks(ticks, ["0", "0.25", "0.5", "0.75", "1"])
        S.style_axes(ax)
        ax.set_xlabel(f"mask on the {D}-components", fontsize=8.5)

        ax = axes[1, j]
        where = np.array([r["pred_at_0"] for r in rs])
        m = where.mean(0)
        bars = ax.bar(range(10), m, color=[col if k == D else "#b9b8b2" for k in range(10)], width=0.7, zorder=2)
        for k in range(10):
            ax.scatter([k] * len(where), where[:, k], s=8, color=S.SURFACE, edgecolors=S.INK, linewidths=0.5, zorder=3)
        for k in np.argsort(m)[::-1][:2]:
            if m[k] > 0.05:
                ax.annotate(f"{m[k] * 100:.0f}%", (k, where[:, k].max()), xytext=(0, 3), textcoords="offset points",
                            ha="center", fontsize=8.5)
        ax.set_xticks(range(10), [str(k) for k in range(10)])
        ax.set_ylim(0, 1.05)
        ax.set_yticks(ticks, [f"{int(t * 100)}%" for t in ticks])
        S.style_axes(ax)
        ax.set_xlabel(f"digit the {D}s are classified as (all {D}-components off)", fontsize=8.5)
    axes[0, 0].set_ylabel(f"mean probability on {D} images")
    axes[1, 0].set_ylabel(f"share of {D} images")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=T.C.NEC["sparse"], lw=2.8, label=f"p({D})"),
                        Line2D([], [], color=S.INK2, lw=2, ls=(0, (2, 2)), label="p(top digit), whatever it is"),
                        Line2D([], [], color=S.INK2, lw=1, ls=(0, (4, 3)), label=f"0.1 = chance (necessity target: p({D}) below this)")],
               frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.005, 0.95), ncol=3)
    fig.suptitle(f"With the {D}-components off, is the model unsure, or confidently calling the {D}s something else?",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.text(0.01, 0.012, f"Same experiment as tradeoff_{D}_dial.png, on the {D} test images only. {D}-components = "
             f"layer-1 components with mean gate > 0.5 on {D}s; layer 2 fully on. Thick lines and bars = mean of 3 seeds, "
             "thin lines and dots = seeds.", fontsize=7.5, color=S.INK2)
    fig.tight_layout(rect=(0, 0.03, 0.985, 0.91), w_pad=3.5, h_pad=2.5)
    out = OUT / f"confidence_{D}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)
    for label in res:
        rs = res[label]
        m = np.mean([r["pred_at_0"] for r in rs], 0)
        print(f"{label}: mask 0 -> p({D}) {np.mean([r['p_digit'][-1] for r in rs]):.3f}, "
              f"p(top) {np.mean([r['p_top'][-1] for r in rs]):.3f}; classified as "
              + ", ".join(f"{k}:{m[k] * 100:.0f}%" for k in np.argsort(m)[::-1][:3]))


if __name__ == "__main__":
    main()
