"""Specificity plots for the MNIST decompositions: turn one digit's layer-1 components down, see which digits break.

A digit's components = layer-1 components whose mean gate over that digit's test images is > 0.5 (the strongly used
ones). Layer 2 stays fully on. Accuracy is relative to the full model, per digit, averaged over seeds.

  1. dose-response: scale digit D's components from 1 (on) to 0 (off); accuracy of every digit along the way
  2. 10 x 10 grid: switch digit k's components fully off (row), accuracy lost on digit j (column)

Both for the joint-necessity model and the per-layer-necessity model (5 x 0.05 adversary, 128 / 32 components).

  python specificity.py [--digit 7]
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

R = HERE / "runs"
OUT = R / "analysis"
MODELS = {"Joint necessity": "03_adversary_strength/adv5/vpd_nec_pred/s{s}.pt",
          "Joint + per-layer necessity": "05_per_layer_necessity/lw/vpd_nec_pred_lw/s{s}.pt",
          "Mirrored necessity": "06_mirrored_necessity/mirror/vpd_nec_mirror/s{s}.pt"}
SEEDS = [0, 1, 2]
DOSES = np.linspace(1, 0, 21)
GATE_MIN = 0.5

# colours (dataviz reference palette): highlighted series, the rest neutral; sequential blue for the grid
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
HL, OTHER = "#2a78d6", "#a8a7a1"
SEQ = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]


@torch.no_grad()
def analyse(path, data):
    ck = torch.load(path, map_location=M.dev)
    a = type("A", (), ck["args"])
    model, wr, ci = M.build(a, 0)
    model.load_state_dict(ck["model"])
    ci.load_state_dict(ck["ci"])
    x, y = data["test_x"], data["test_y"]
    N, C1, C2 = len(y), wr["fc1"].C, wr["fc2"].C
    g1 = M.gates_of(model, wr, ci, x)["fc1"][:, 0]
    groups = [(g1[y == k].mean(0) > GATE_MIN) for k in range(10)]
    full = M.fwd(model, wr, x, "full").argmax(-1)
    acc_full = torch.stack([(full[y == j] == j).float().mean() for j in range(10)])

    def acc_with(scale):  # scale: (C1,) layer-1 mask, same for every image; layer 2 fully on
        pr = M.fwd(model, wr, x, "masked", {"fc1": scale.expand(N, 1, C1), "fc2": torch.ones(N, 1, C2)}).argmax(-1)
        return (torch.stack([(pr[y == j] == j).float().mean() for j in range(10)]) / acc_full).numpy()

    curves = {}
    for k in range(10):
        rows = []
        for t in DOSES:
            m = torch.ones(C1)
            m[groups[k]] = float(t)
            rows.append(acc_with(m))
        curves[k] = np.stack(rows)  # (doses, 10)
    grid = np.stack([curves[k][-1] for k in range(10)])  # fully off
    return curves, grid, [int(gk.sum()) for gk in groups]


def style_axes(ax):
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--digit", type=int, default=7)
    a = ap.parse_args()
    data = M.load_mnist()
    res = {name: [analyse(R / pat.format(s=s), data) for s in SEEDS] for name, pat in MODELS.items()}
    json.dump({n: [{"seed": s, "grid": p[1].tolist(), "sizes": p[2], "doses": DOSES.tolist(),
                    "curves": {k: c.tolist() for k, c in p[0].items()}} for s, p in zip(SEEDS, per)]
               for n, per in res.items()}, open(OUT / "specificity.json", "w"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID, "figure.facecolor": SURFACE,
                         "axes.facecolor": SURFACE})
    D = a.digit
    names = list(MODELS)

    # 1. dose-response for digit D: rows = models, columns = seeds
    fig, axes = plt.subplots(len(names), 3, figsize=(11, 3.2 * len(names)), sharex=True, sharey=True)
    for r, name in enumerate(names):
        for col, (seed, (curves, grid, sizes)) in enumerate(zip(SEEDS, res[name])):
            ax = axes[r, col]
            c = curves[D]
            for j in range(10):
                if j != D:
                    ax.plot(DOSES, c[:, j], color=OTHER, lw=1.1, zorder=2)
            ax.plot(DOSES, c[:, D], color=HL, lw=2.4, zorder=3)
            broken = [j for j in range(10) if j != D and c[-1, j] < 0.5]
            ax.set_title(f"seed {seed} · {sizes[D]} component{'s' if sizes[D] != 1 else ''} for {D}" +
                         (f"\nalso broken: {', '.join(map(str, broken))}" if broken else "\nno other digit broken"),
                         loc="left", fontsize=8.5, color=INK)
            ax.set_xlim(1, 0)
            ax.set_ylim(-0.03, 1.05)
            ax.set_xticks([1, 0.75, 0.5, 0.25, 0], ["1 (on)", "0.75", "0.5", "0.25", "0 (off)"])
            ax.set_yticks([0, 0.5, 1], ["0%", "50%", "100%"])
            style_axes(ax)
        axes[r, 0].set_ylabel(f"{name}\n\naccuracy kept", fontsize=9)
    for ax in axes[-1]:
        ax.set_xlabel(f"mask on the {D}-components (layer 1)")
    axes[0, 2].plot([], [], color=HL, lw=2.4, label=f"{D}s")
    axes[0, 2].plot([], [], color=OTHER, lw=1.1, label="each other digit")
    axes[0, 2].legend(frameon=False, loc="lower left", fontsize=8)
    fig.suptitle(f"Turning down the {D}-components: only {D}s should break", x=0.01, ha="left", fontsize=11,
                 fontweight="bold")
    fig.text(0.01, 0.005, f"{D}-components = layer-1 components with mean gate > {GATE_MIN} on {D}s. Layer 2 fully on. "
             "Accuracy relative to the full model; 10k test images. 'Broken' = under 50% kept when fully off.",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    out1 = OUT / f"dose_response_{D}.png"
    fig.savefig(out1, dpi=150)
    plt.close(fig)

    # 2. 10 x 10 grids: rows = models, columns = seeds
    cmap = LinearSegmentedColormap.from_list("seq", SEQ)
    fig, axes = plt.subplots(len(names), 3, figsize=(12.5, 4.6 * len(names)))
    for r, name in enumerate(names):
        for col, (seed, (curves, grid, sizes)) in enumerate(zip(SEEDS, res[name])):
            ax = axes[r, col]
            lost = np.clip(1 - grid, 0, 1)
            ax.imshow(lost, cmap=cmap, vmin=0, vmax=1)
            for k in range(10):
                for j in range(10):
                    val = lost[k, j]
                    ax.text(j, k, f"{val * 100:.0f}", ha="center", va="center", fontsize=6,
                            color="#ffffff" if val > 0.55 else INK)
            ax.set_xticks(range(10))
            ax.set_yticks(range(10), [f"{k} ({sizes[k]})" for k in range(10)], fontsize=7.5)
            ax.xaxis.tick_top()
            ax.tick_params(length=0, labelsize=7.5)
            for sp in ax.spines.values():
                sp.set_visible(False)
            off = lost[~np.eye(10, dtype=bool)]
            ax.set_title(f"seed {seed}: own {np.diag(lost).mean() * 100:.0f}% lost · others {off.mean() * 100:.0f}%",
                         fontsize=8.5, color=INK, pad=18)
        axes[r, 0].set_ylabel(f"{name}\n\ndigit whose components are off (# comps)", fontsize=9)
    fig.suptitle("Accuracy lost (%) when one digit's layer-1 components are off  ·  columns = digit tested  ·  "
                 "ideal: dark diagonal, white elsewhere", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.text(0.01, 0.008, f"Components with mean gate > {GATE_MIN} on that digit; layer 2 fully on; relative to the full "
             "model; 10k test images.", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.96))
    out2 = OUT / "specificity_grid.png"
    fig.savefig(out2, dpi=150)
    plt.close(fig)
    print(out1, out2)


if __name__ == "__main__":
    main()
