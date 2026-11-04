"""Which components count as digit D's? Sweep the cut-off instead of fixing it.

D-set(x) = layer-1 components whose mean gate over D's test images is above x. Switch the D-set fully off for every
image (layer 2 fully on) and measure, relative to the full model, how many D images and how many other-digit images
are still classified correctly. Sweep x from 1 (nothing off) down toward 0 (everything D ever touches is off).
Ideal: D falls to its target (0% for the hinge, ~10% for the uniform target) while the other digits stay at 100%.

  python gate_sweep.py [--digit 7] [--plot-only]
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
import style as C  # noqa: E402

R = HERE / "runs"
OUT = R / "analysis"
SEEDS = [0, 1, 2]
XS = np.round(np.r_[np.arange(1.0, 0.049, -0.05), 0.03, 0.02, 0.01], 3)
MODELS = [  # title, folder, colour, ideal for D at the end
    ("Model 1: hinge target", "07_beta_sweep/mirror_b10_adv10/vpd_nec_mirror", C.NEC["sparse"], 0.0),
    ("Model 2: uniform target", "10_uniform_mirrored/b10_adv10_nec3/vpd_nec_mirror_unif", C.NEC["default"], 0.1),
    ("No necessity", "09_no_necessity/vpd_b10/vpd", C.NONE["sparse"], 0.0),
]


@torch.no_grad()
def analyse(path, data, D):
    ck = torch.load(path, map_location=M.dev)
    a = type("A", (), ck["args"])
    model, wr, ci = M.build(a, 0)
    model.load_state_dict(ck["model"])
    ci.load_state_dict(ck["ci"])
    x, y = data["test_x"], data["test_y"]
    N, C1, C2 = len(y), wr["fc1"].C, wr["fc2"].C
    mean_d = M.gates_of(model, wr, ci, x)["fc1"][:, 0][y == D].mean(0)
    full = M.fwd(model, wr, x, "full").argmax(-1)
    ok_full = full == y
    d_kept, o_kept, size = [], [], []
    for t in XS:
        keep = (mean_d <= t).float()
        pr = M.fwd(model, wr, x, "masked", {"fc1": keep.expand(N, 1, C1), "fc2": torch.ones(N, 1, C2)}).argmax(-1)
        acc = torch.stack([(pr[y == j] == j).float().mean() / ok_full[y == j].float().mean() for j in range(10)])
        d_kept.append(acc[D].item())
        o_kept.append(acc[[j for j in range(10) if j != D]].mean().item())
        size.append(int((mean_d > t).sum()))
    return {"digit_kept": d_kept, "others_kept": o_kept, "set_size": size}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--digit", type=int, default=7)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    D = a.digit
    path = OUT / f"gate_sweep_{D}.json"
    if a.plot_only:
        res = json.load(open(path))["models"]
    else:
        data = M.load_mnist()
        res = {t: [analyse(R / f / f"s{s}.pt", data, D) for s in SEEDS] for t, f, _, _ in MODELS}
        json.dump({"x": XS.tolist(), "digit": D, "models": res}, open(path, "w"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": S.INK, "axes.labelcolor": S.INK2,
                         "xtick.color": S.INK2, "ytick.color": S.INK2, "axes.edgecolor": S.GRID,
                         "figure.facecolor": S.SURFACE, "axes.facecolor": S.SURFACE})
    fig, axes = plt.subplots(1, len(MODELS), figsize=(12.5, 4.6), sharey=True)
    ticks = [0, 0.25, 0.5, 0.75, 1]
    for ax, (title, _, col, ideal) in zip(axes, MODELS):
        rs = res[title]
        for r in rs:
            ax.plot(XS, r["digit_kept"], color=col, lw=0.8, alpha=0.45, zorder=2)
            ax.plot(XS, r["others_kept"], color=S.INK2, lw=0.8, alpha=0.35, ls=(0, (2, 2)), zorder=2)
        d = np.mean([r["digit_kept"] for r in rs], 0)
        o = np.mean([r["others_kept"] for r in rs], 0)
        ax.plot(XS, o, color=S.INK2, lw=2, ls=(0, (2, 2)), zorder=3)
        ax.plot(XS, d, color=col, lw=2.8, zorder=4, solid_capstyle="round")
        if ideal > 0:
            ax.axhline(ideal, color=col, lw=1, ls=(0, (1, 2)), zorder=1)
            ax.annotate("10% = guessing\n(this model's ideal)", (0.97, ideal), xytext=(0, 4), textcoords="offset points",
                        fontsize=7.5, color=S.INK2, va="bottom")
        # set size along the bottom: how many components the D-set holds at a few cut-offs (seed mean)
        for t in (0.5, 0.2, 0.05):
            i = int(np.argmin(np.abs(XS - t)))
            n = np.mean([r["set_size"][i] for r in rs])
            ax.annotate(f"{n:.1f}", (XS[i], 1.0), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7.5,
                        color=S.INK2)
        lo, hi = sorted([(d[-1], f"{D}s {d[-1] * 100:.0f}%", col), (o[-1], f"others {o[-1] * 100:.0f}%", S.INK)])
        if hi[0] - lo[0] < 0.12:
            lo = (hi[0] - 0.12, *lo[1:])
        for yv, txt, c in (lo, hi):
            ax.annotate(txt, (XS[-1], yv), xytext=(6, 0), textcoords="offset points", va="center", fontsize=9,
                        color=c, fontweight="bold", annotation_clip=False)
        ax.set_title(title, loc="left", fontsize=10.5, fontweight="bold", pad=16)
        ax.set_xlim(1.0, 0.0)
        ax.set_ylim(-0.03, 1.05)
        ax.set_xticks([1, 0.75, 0.5, 0.25, 0])
        ax.set_yticks(ticks, [f"{int(t * 100)}%" for t in ticks])
        S.style_axes(ax)
        ax.set_xlabel(f"cut-off x: off = components with mean gate on {D}s above x", fontsize=8.5)
    axes[0].set_ylabel("still classified correctly")
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], color=C.NEC["sparse"], lw=2.8, label=f"{D}s (should fall to 0%, or ~10% for Model 2)"),
                        Line2D([], [], color=S.INK2, lw=2, ls=(0, (2, 2)), label="other digits (should stay at 100%)")],
               frameon=False, fontsize=9, loc="upper left", bbox_to_anchor=(0.005, 0.9), ncol=2)
    fig.suptitle(f"Switching off the {D}-components, for every cut-off of what counts as one",
                 x=0.01, ha="left", fontsize=11.5, fontweight="bold")
    fig.text(0.01, 0.015, f"Small numbers along the top: how many layer-1 components the {D}-set holds at x = 0.5, 0.2, "
             f"0.05 (mean of seeds). Components fully off; layer 2 fully on. Thick lines = mean of 3 seeds, thin = seeds.\n"
             f"Accuracy relative to the full model. All models β 10; Models 1 and 2 adversary 10×0.1, Model 2's necessity "
             "weight ×3. The earlier dial plots use x = 0.5.", fontsize=7.5, color=S.INK2)
    fig.tight_layout(rect=(0, 0.06, 0.97, 0.86), w_pad=3.5)
    out = OUT / f"gate_sweep_{D}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)
    for t, _, _, _ in MODELS:
        rs = res[t]
        for xv in (0.5, 0.2, 0.05, 0.01):
            i = int(np.argmin(np.abs(XS - xv)))
            print(f"{t:26s} x={xv:<5} set {np.mean([r['set_size'][i] for r in rs]):5.1f}  "
                  f"{D}s {np.mean([r['digit_kept'][i] for r in rs]) * 100:5.1f}%  "
                  f"others {np.mean([r['others_kept'][i] for r in rs]) * 100:5.1f}%")


if __name__ == "__main__":
    main()
