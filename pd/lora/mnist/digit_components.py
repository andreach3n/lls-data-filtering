"""Which layer-1 components does one digit use, and do the less common ones pick out a style of that digit?

For each seed: the main component (important on almost every image of the digit) and the next components that are
mostly specific to the digit. For each, its template, the average of the digit's images that use it, and a few
examples, next to the average of all images of the digit.

  python digit_components.py runs/07_beta_sweep/mirror_b10_adv10/vpd_nec_mirror [--digit 7]
"""
import argparse
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # gated_lora
import train as M  # noqa: E402
import specificity as S  # noqa: E402

THR = 0.1


@torch.no_grad()
def components_for(run, seed, digit, data, max_extra=3, min_cov=0.03, min_purity=0.5):
    ck = torch.load(Path(run) / f"s{seed}.pt", map_location="cpu")
    a = type("A", (), ck["args"])
    model, wr, ci = M.build(a, 0)
    model.load_state_dict(ck["model"])
    ci.load_state_dict(ck["ci"])
    x, y = data["test_x"], data["test_y"]
    imp = M.gates_of(model, wr, ci, x)["fc1"][:, 0] > THR
    mine = y == digit
    cov = imp[mine].float().mean(0)  # share of the digit's images where each component is important
    purity = imp[mine].float().sum(0) / imp.float().sum(0).clamp(min=1)  # share of its uses that are this digit
    order = cov.argsort(descending=True).tolist()
    main = order[0]
    extra = [c for c in order[1:] if cov[c] >= min_cov and purity[c] >= min_purity][:max_extra]
    out = []
    for c in [main] + extra:
        users = mine & imp[:, c]
        out.append({"c": c, "cov": cov[c].item(), "purity": purity[c].item(),
                    "tmpl": wr["fc1"].V[:, c].reshape(28, 28).clone(),
                    "mean": x[users].mean(0).reshape(28, 28), "n": int(users.sum()),
                    "examples": x[users][:4].reshape(-1, 28, 28)})
    return out, x[mine].mean(0).reshape(28, 28)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", help="folder with s0.pt, s1.pt, s2.pt")
    ap.add_argument("--digit", type=int, default=7)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    a = ap.parse_args()
    data = M.load_mnist()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8, "text.color": S.INK,
                         "figure.facecolor": S.SURFACE, "axes.facecolor": S.SURFACE})
    rows = []
    for s in a.seeds:
        comps, avg = components_for(a.run, s, a.digit, data)
        rows.append((s, comps, avg))
    nrows = sum(len(c) for _, c, _ in rows)
    ncols = 7  # template, users' mean, all-digit mean, 4 examples
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.25, nrows * 1.45 + 0.8), squeeze=False)
    r = 0
    for s, comps, avg in rows:
        for i, cp in enumerate(comps):
            kind = "main" if i == 0 else "extra"
            ax = axes[r]
            lim = cp["tmpl"].abs().max().item()
            ax[0].imshow(cp["tmpl"], cmap="RdBu_r", vmin=-lim, vmax=lim)
            ax[0].set_title(f"seed {s} · {kind} · comp {cp['c']}\non {cp['cov']:.0%} of {a.digit}s, "
                            f"{cp['purity']:.0%} {a.digit}s", fontsize=6.5, loc="left")
            ax[1].imshow(cp["mean"], cmap="gray_r")
            ax[1].set_title(f"avg of the {cp['n']}\n{a.digit}s using it", fontsize=6.5)
            ax[2].imshow(avg, cmap="gray_r")
            ax[2].set_title(f"avg of all\n{a.digit}s", fontsize=6.5)
            for k in range(4):
                if k < len(cp["examples"]):
                    ax[3 + k].imshow(cp["examples"][k], cmap="gray_r")
            if r == 0:
                ax[3].set_title("examples using it", fontsize=6.5, loc="left")
            for axx in ax:
                axx.set_xticks([])
                axx.set_yticks([])
                for sp in axx.spines.values():
                    sp.set_visible(False)
            r += 1
    fig.suptitle(f"Layer-1 components used by {a.digit}s: one main component, plus extras used by some {a.digit}s",
                 x=0.01, ha="left", fontsize=10, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = HERE / "runs" / "analysis" / f"components_for_{a.digit}.png"
    fig.savefig(out, dpi=150)
    print(out)


if __name__ == "__main__":
    main()
