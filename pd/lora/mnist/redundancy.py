"""How much backup is built into the weights, ignoring the gates?

Checks how necessity is satisfied: by marking more components important (weights keep their backups) or by
removing the backups from the weights. Layer 1 only; layer 2 always fully on.

  1. random removal: switch off a random fraction of layer-1 components for every image; accuracy vs fraction.
     A model with backups degrades gracefully.
  2. single points of failure: per image, the number of layer-1 components whose removal alone flips the
     prediction. Few = lots of backup.

  python redundancy.py
"""
import json
import statistics as st
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
FRACS = [0, 0.1, 0.2, 0.3, 0.5, 0.7]
DRAWS = 5
MODELS = [  # label, folder, colour, line style
    ("default, with necessity", "06_mirrored_necessity/mirror/vpd_nec_mirror", C.NEC["default"], "-"),
    ("default, without", "09_no_necessity/vpd_default/vpd", C.NONE["default"], C.NONE_DASH),
    ("λ 0.1, with necessity", "08_lambda_control/mirror_lam0.1/vpd_nec_mirror", C.NEC["sparse"], "-"),
    ("λ 0.1, without", "09_no_necessity/vpd_lam0.1/vpd", C.NONE["sparse"], C.NONE_DASH),
]


@torch.no_grad()
def analyse(path, data):
    ck = torch.load(path, map_location="cpu")
    a = type("A", (), ck["args"])
    model, wr, ci = M.build(a, 0)
    model.load_state_dict(ck["model"])
    ci.load_state_dict(ck["ci"])
    x, y = data["test_x"], data["test_y"]
    N, C1, C2 = len(y), wr["fc1"].C, wr["fc2"].C
    ones2 = torch.ones(N, 1, C2)
    run = lambda keep: M.fwd(model, wr, x, "masked", {"fc1": keep.expand(N, 1, C1), "fc2": ones2}).argmax(-1)  # noqa
    pred = M.fwd(model, wr, x, "full").argmax(-1)
    ok = pred == y
    g = torch.Generator().manual_seed(0)
    curve = []
    for f in FRACS:
        accs = []
        for _ in range(DRAWS if f > 0 else 1):
            keep = (torch.rand(C1, generator=g) >= f).float()
            accs.append((run(keep) == y).float().mean().item())
        curve.append(st.mean(accs))
    # single points of failure, on correctly classified images
    spof = torch.zeros(N)
    for c in range(C1):
        keep = torch.ones(C1)
        keep[c] = 0
        spof += (run(keep) != pred).float()
    spof = spof[ok]
    l0 = (M.gates_of(model, wr, ci, x)["fc1"][:, 0][ok] > a.thr).float().sum(1)
    return {"curve": curve, "spof_mean": spof.mean().item(), "spof_any": (spof > 0).float().mean().item(),
            "l0_fc1": l0.mean().item()}


def main():
    data = M.load_mnist()
    res = {lab: [analyse(R / folder / f"s{s}.pt", data) for s in SEEDS] for lab, folder, _, _ in MODELS}
    json.dump({"fracs": FRACS, "models": res}, open(OUT / "redundancy.json", "w"))
    print(f"{'':28s}{'layer-1 important':>18s}{'single points':>15s}{'images with ≥1':>16s}   accuracy after removing a random share of layer-1 components")
    print(f"{'':28s}{'per image':>18s}{'of failure':>15s}{'':>16s}   " + "  ".join(f"{int(f*100):>3d}%" for f in FRACS))
    for lab, rs in res.items():
        c = np.mean([r["curve"] for r in rs], 0)
        print(f"{lab:28s}{st.mean(r['l0_fc1'] for r in rs):>18.1f}{st.mean(r['spof_mean'] for r in rs):>15.2f}"
              f"{st.mean(r['spof_any'] for r in rs) * 100:>15.0f}%   " + "  ".join(f"{v * 100:>3.0f}%" for v in c))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "text.color": S.INK, "axes.labelcolor": S.INK2,
                         "xtick.color": S.INK2, "ytick.color": S.INK2, "axes.edgecolor": S.GRID,
                         "figure.facecolor": S.SURFACE, "axes.facecolor": S.SURFACE})
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.4), gridspec_kw={"width_ratios": [1.5, 1]})
    for lab, _, col, ls in MODELS:
        c = np.mean([r["curve"] for r in res[lab]], 0)
        a1.plot(FRACS, c, color=col, ls=ls, lw=2.2, marker="o", ms=4, label=lab)
    a1.set_xticks(FRACS, [f"{int(f * 100)}%" for f in FRACS])
    a1.set_yticks([0, 0.25, 0.5, 0.75, 1], ["0%", "25%", "50%", "75%", "100%"])
    a1.set_ylim(0, 1.03)
    a1.set_xlabel("share of layer-1 components switched off at random (same for every image)")
    a1.set_ylabel("test accuracy")
    a1.set_title("Random removal: models with backups degrade slowly", loc="left", fontsize=10, color=S.INK)
    a1.legend(frameon=False, fontsize=8, loc="lower left")
    S.style_axes(a1)
    labs = [m[0] for m in MODELS]
    xs = np.arange(len(labs))
    for i, (lab, _, col, ls) in enumerate(MODELS):
        vals = [r["spof_mean"] for r in res[lab]]
        a2.bar(i, st.mean(vals), color=col, width=0.6, zorder=2, hatch=C.NONE_HATCH if ls != "-" else None,
               edgecolor=S.SURFACE)
        a2.scatter([i] * len(vals), vals, s=12, color=S.SURFACE, edgecolors=S.INK, linewidths=0.6, zorder=3)
        a2.annotate(f"{st.mean(vals):.2f}", (i, max(vals)), xytext=(0, 4), textcoords="offset points", ha="center",
                    fontsize=8)
    a2.set_xticks(xs, [lab.replace(", ", "\n") for lab in labs], fontsize=8)
    a2.set_title("Layer-1 components whose removal alone\nflips the prediction, per image", loc="left", fontsize=10,
                 color=S.INK)
    S.style_axes(a2)
    fig.suptitle("How much backup is built into the weights (gates ignored)?", x=0.01, ha="left", fontsize=11,
                 fontweight="bold")
    fig.text(0.01, 0.01, "Layer 2 always fully on. Mean of 3 seeds (5 random draws per point). Single points of failure "
             "counted on correctly classified test images.\nSky blue = with necessity, rose dashed/hatched = without; "
             "lighter = default sparsity, darker = λ 0.1.",
             fontsize=7.5, color=S.INK2)
    fig.tight_layout(rect=(0, 0.06, 1, 0.93))
    fig.savefig(OUT / "redundancy.png", dpi=150)
    plt.close(fig)
    print(OUT / "redundancy.png")


if __name__ == "__main__":
    main()
