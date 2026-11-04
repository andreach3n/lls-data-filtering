"""Dial plot for the write-up's three β 10 models, with the digit's components picked by how often they are used.

D-components = layer-1 components that are important (gate > 0.1) on at least SHARE of D's test images. (The older
dial plots in tradeoff.py use mean gate on D > 0.5, which misses components used by only some D images, such as
Model 2's helpers.) Turn the D-components' mask down from 1 to 0 for every image, layer 2 fully on, and track how many
D images and other-digit images are still classified correctly, relative to the full model.

  python dial_models.py [--digit 7] [--share 0.2] [--plot-only]
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
import gate_sweep as GS  # noqa: E402

R = HERE / "runs"
OUT = R / "analysis"
SEEDS = [0, 1, 2]
MODELS = GS.MODELS  # title, folder, colour, ideal for D at mask 0


@torch.no_grad()
def analyse(path, data, D, shares):
    ck = torch.load(path, map_location=M.dev)
    a = type("A", (), ck["args"])
    model, wr, ci = M.build(a, 0)
    model.load_state_dict(ck["model"])
    ci.load_state_dict(ck["ci"])
    x, y = data["test_x"], data["test_y"]
    N, C1, C2 = len(y), wr["fc1"].C, wr["fc2"].C
    used = (M.gates_of(model, wr, ci, x)["fc1"][:, 0][y == D] > a.thr).float().mean(0)  # share of D images
    ok = (M.fwd(model, wr, x, "full").argmax(-1) == y).float()
    base = torch.stack([ok[y == j].mean() for j in range(10)])
    out = {}
    for sh in shares:
        group = used >= sh
        d_kept, o_kept = [], []
        for t in S.DOSES:
            m = torch.ones(C1)
            m[group] = float(t)
            pr = M.fwd(model, wr, x, "masked", {"fc1": m.expand(N, 1, C1), "fc2": torch.ones(N, 1, C2)}).argmax(-1)
            acc = torch.stack([(pr[y == j] == j).float().mean() for j in range(10)]) / base
            d_kept.append(acc[D].item())
            o_kept.append(acc[[j for j in range(10) if j != D]].mean().item())
        out[str(sh)] = {"digit_kept": d_kept, "others_kept": o_kept, "set_size": int(group.sum())}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--digit", type=int, default=7)
    ap.add_argument("--share", type=float, default=0.2)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    D, SH = a.digit, str(a.share)
    shares = sorted({0.1, 0.2, 0.3, a.share})
    path = OUT / f"dial_models_{D}.json"
    if a.plot_only:
        res = json.load(open(path))["models"]
    else:
        data = M.load_mnist()
        res = {t: [analyse(R / f / f"s{s}.pt", data, D, shares) for s in SEEDS] for t, f, _, _ in MODELS}
        json.dump({"doses": S.DOSES.tolist(), "digit": D, "models": res}, open(path, "w"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": S.INK, "axes.labelcolor": S.INK2,
                         "xtick.color": S.INK2, "ytick.color": S.INK2, "axes.edgecolor": S.GRID,
                         "figure.facecolor": S.SURFACE, "axes.facecolor": S.SURFACE})
    fig, axes = plt.subplots(1, len(MODELS), figsize=(11.5, 4.5), sharey=True)
    x = S.DOSES
    ticks = [0, 0.25, 0.5, 0.75, 1]
    for ax, (title, _, col, ideal) in zip(axes, MODELS):
        rs = [r[SH] for r in res[title]]
        for r in rs:
            ax.plot(x, r["digit_kept"], color=col, lw=0.8, alpha=0.45, zorder=2)
            ax.plot(x, r["others_kept"], color=S.INK2, lw=0.8, alpha=0.35, ls=(0, (2, 2)), zorder=2)
        d = np.mean([r["digit_kept"] for r in rs], 0)
        o = np.mean([r["others_kept"] for r in rs], 0)
        ax.plot(x, o, color=S.INK2, lw=2, ls=(0, (2, 2)), zorder=3)
        ax.plot(x, d, color=col, lw=2.8, zorder=4, solid_capstyle="round")
        if ideal > 0:
            ax.axhline(ideal, color=col, lw=1, ls=(0, (1, 2)), zorder=1)
            ax.annotate("10% = guessing\n(this model's ideal)", (0.98, ideal), xytext=(0, 4), textcoords="offset points",
                        fontsize=7.5, color=S.INK2, va="bottom")
        lo, hi = sorted([(d[-1], f"{D}s {d[-1] * 100:.0f}%", col), (o[-1], f"others {o[-1] * 100:.0f}%", S.INK)])
        if hi[0] - lo[0] < 0.12:
            lo = (hi[0] - 0.12, *lo[1:])
        for yv, txt, c in (lo, hi):
            ax.annotate(txt, (0, yv), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9,
                        color=c, fontweight="bold", annotation_clip=False)
        n = np.mean([r["set_size"] for r in rs])
        ax.set_title(f"{title}\n", loc="left", fontsize=10.5, fontweight="bold")
        ax.text(0, 1.03, f"{n:.1f} {D}-components (mean of seeds)", transform=ax.transAxes, fontsize=8, color=S.INK2)
        ax.set_xlim(1.02, -0.02)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xticks([1, 0.5, 0], ["1\n(on)", "0.5", "0\n(off)"])
        ax.set_yticks(ticks, [f"{int(t * 100)}%" for t in ticks])
        S.style_axes(ax)
        ax.set_xlabel(f"mask on the {D}-components", fontsize=8.5)
    axes[0].set_ylabel("still classified correctly")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=GS.C.NEC["sparse"], lw=2.8, label=f"{D}s (should fall to 0%, or ~10% for Model 2)"),
                        Line2D([], [], color=S.INK2, lw=2, ls=(0, (2, 2)), label="other digits (should stay at 100%)")],
               frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.005, 0.9), ncol=2)
    fig.suptitle(f"Turning the {D}-components down: {D}s should go, other digits should stay",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.text(0.01, 0.015, f"{D}-components = layer-1 components important (gate > 0.1) on at least {float(SH) * 100:.0f}% "
             f"of {D} test images; layer 2 fully on. Thick lines = mean of 3 seeds, thin = seeds.\nAccuracy relative to "
             "the full model. All models β 10; Models 1 and 2 adversary 10×0.1, Model 2's necessity weight ×3.",
             fontsize=7.5, color=S.INK2)
    fig.tight_layout(rect=(0, 0.06, 0.96, 0.86), w_pad=3.5)
    out = OUT / f"dial_models_{D}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)
    for t, _, _, _ in MODELS:
        for sh in sorted(res[t][0]):
            rs = [r[sh] for r in res[t]]
            print(f"{t:26s} share {float(sh):.0%}: set {np.mean([r['set_size'] for r in rs]):4.1f}  at mask 0: "
                  f"{D}s {np.mean([r['digit_kept'][-1] for r in rs]) * 100:5.1f}%  "
                  f"others {np.mean([r['others_kept'][-1] for r in rs]) * 100:5.1f}%")


if __name__ == "__main__":
    main()
