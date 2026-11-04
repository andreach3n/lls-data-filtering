"""What does a layer-1 component detect? One row per component:
template (V, red = positive, blue = negative) | mean image where it is important | mean image where it is not (light digits flipped to dark first, so light and
dark digits don't cancel; contrast stretched) |
the 8 test images where its gate is highest | share of each digit's images where it is important, split into
light digits on dark backgrounds and dark digits on light ones.

  python3 component_profile.py runs/07_uniform_target/unif_nec3/vpd_nec_mirror_unif/s0.pt 49 72 17 [--out name]
"""
import argparse
import sys
from pathlib import Path

import torch

import train as SV  # svhn/train.py: configures the shared MNIST code for SVHN

T = SV.T
HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "mnist"))
import style as C  # noqa: E402

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def light_digit(x):
    """1 if the centre is brighter than the top and bottom rows (light digit on dark background), as in lost_images.py."""
    im = x.view(-1, 32, 32)
    centre = im[:, 8:24, 11:21].mean((1, 2))
    border = torch.cat([im[:, :3].flatten(1), im[:, -3:].flatten(1)], 1).mean(1)
    return centre > border


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("comps", nargs="+", type=int)
    ap.add_argument("--out", default="component_profile")
    a = ap.parse_args()
    ck = torch.load(a.ckpt, map_location=T.dev, weights_only=False)
    args = argparse.Namespace(**ck["args"])
    model, wr, ci = T.build(args, 0)
    model.load_state_dict(ck["model"])
    ci.load_state_dict(ck["ci"])
    d = SV.load_svhn()
    x, y = d["test_x"], d["test_y"]
    with torch.no_grad():
        g = T.gates_of(model, wr, ci, x)["fc1"][:, 0]
    imp = g > args.thr
    light = light_digit(x)
    aligned = torch.where(light[:, None], 1 - x, x)  # every image as a dark digit on a light background

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 8.5, "text.color": INK, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    n = len(a.comps)
    fig = plt.figure(figsize=(15, 2.55 * n + 0.9))
    gs = GridSpec(n, 13, figure=fig, width_ratios=[1.25, 1.25, 1.25, 0.25] + [1] * 8 + [0.1], wspace=0.15, hspace=0.55)
    for r, c in enumerate(a.comps):
        users = imp[:, c]
        freq = users.float().mean().item()
        v = wr["fc1"].V[:, c].detach().view(32, 32)
        lim = v.abs().max().item()
        panels = [(v, "RdBu_r", -lim, lim, f"L1·{c}: template"),
                  (aligned[users].mean(0).view(32, 32), "gray", None, None, f"mean image, important\n({users.sum():,} images)"),
                  (aligned[~users].mean(0).view(32, 32), "gray", None, None, "mean image, not important")]
        for k, (im, cm, lo, hi, title) in enumerate(panels):
            ax = fig.add_subplot(gs[r, k])
            ax.imshow(im.cpu(), cmap=cm, vmin=lo, vmax=hi)
            ax.set_title(title, fontsize=8, loc="left", color=INK if k == 0 else INK2,
                         fontweight="bold" if k == 0 else "normal")
            ax.axis("off")
        top = g[:, c].argsort(descending=True)[:4].tolist() + \
            users.nonzero().flatten()[torch.randperm(int(users.sum()), generator=torch.Generator().manual_seed(c))[:4]].tolist()
        for k, j in enumerate(top):
            ax = fig.add_subplot(gs[r, 4 + k])
            ax.imshow(x[j].view(32, 32).cpu(), cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"{'top gate' if k < 4 else 'random user'}\nlabel {y[j].item()}", fontsize=7, color=INK2)
            ax.axis("off")
        # by-digit share, light vs dark digits, as a small inset under the title column
        ax = fig.add_axes([0, 0, 0, 0])  # placeholder, positioned below
        ax.remove()
        per = [(imp[(y == k) & light, c].float().mean().item(), imp[(y == k) & ~light, c].float().mean().item())
               for k in range(10)]
        txt = (f"important on {freq:.0%} of all test images  ·  light digits {imp[light, c].float().mean():.0%}, "
               f"dark digits {imp[~light, c].float().mean():.0%}\nshare of each digit's images (light / dark): " +
               "  ".join(f"{k}: {p[0]:.0%}/{p[1]:.0%}" for k, p in enumerate(per)))
        bb = fig.add_subplot(gs[r, 0]).get_position()
        fig.axes[-1].remove()
        fig.text(bb.x0, bb.y0 - 0.035 / n * 2.2, txt, fontsize=7.5, color=INK2, va="top")
    fig.suptitle(f"What do these layer-1 components detect?  ({a.ckpt})", x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    out = HERE / "runs/analysis" / f"{a.out}.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print(out)
    for c in a.comps:
        u = imp[:, c]
        print(f"L1·{c}: important on {u.float().mean():.1%}; light digits {imp[light, c].float().mean():.1%}, "
              f"dark digits {imp[~light, c].float().mean():.1%}; share light among users {light[u].float().mean():.1%} "
              f"(all test: {light.float().mean():.1%})")


if __name__ == "__main__":
    main()
