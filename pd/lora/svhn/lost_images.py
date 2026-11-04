"""Which test images does stronger necessity cost? Compares the 1x-weight models (02_adversary/mirror_b10_adv20)
with the 3x-weight models (03_nec_weight/nec3_12k), same seeds, same 128 + 32 components.

"Lost" = right in the 1x model, wrong in the 3x model. "Kept" = right in both.

1. Hypothesis: necessity makes rarely-important components a liability, so the 3x model drops them, and the images
   that needed them are the ones lost. Test: in the 1x model, how often is each layer-1 component important across
   the test set (its frequency)? Per image, rarity = frequency of its rarest important layer-1 component. Lost
   images should have rarer components than kept ones, also after matching on digit and on how confident the 1x
   model was (hard images are lost more often whatever the reason).
2. Image properties of lost vs kept: digit, polarity (light digit on dark background or the reverse), contrast,
   clutter at the left/right edges.
3. A grid of lost and kept images (seed 0).

  python3 lost_images.py          -> runs/analysis/lost_images.txt, lost_images.png
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import train as S  # svhn/train.py: configures the shared MNIST code for SVHN

T = S.T
HERE = Path(__file__).parent
RUNS = HERE / "runs"
A = RUNS / "02_adversary/mirror_b10_adv20/vpd_nec_mirror"  # 1x necessity weight
B = RUNS / "03_nec_weight/nec3_12k/vpd_nec_mirror"  # 3x necessity weight
OUT = RUNS / "analysis"


def load(path):
    ck = torch.load(path, map_location=T.dev, weights_only=False)
    a = argparse.Namespace(**ck["args"])
    model, wr, ci = T.build(a, 0)
    model.load_state_dict(ck["model"])
    ci.load_state_dict(ck["ci"])
    return model, wr, ci, a


@torch.no_grad()
def run(path, x, y):
    model, wr, ci, a = load(path)
    lg = T.fwd(model, wr, x, "full")
    g1 = T.gates_of(model, wr, ci, x)["fc1"][:, 0]  # (N, C1)
    return {"pred": lg.argmax(-1), "p_true": F.softmax(lg, -1).gather(1, y[:, None])[:, 0],
            "imp": g1 > a.thr}


def image_features(x):
    """x: raw grayscale in [0, 1], (N, 1024)."""
    im = x.view(-1, 32, 32)
    centre = im[:, 8:24, 11:21].mean((1, 2))
    border = torch.cat([im[:, :3].flatten(1), im[:, -3:].flatten(1)], 1).mean(1)  # top and bottom rows
    edges = torch.cat([im[:, :, :6].flatten(1), im[:, :, -6:].flatten(1)], 1)
    std = im.flatten(1).std(1)
    return {"light digit on dark": (centre > border).float(),
            "contrast (pixel std)": std,
            "edge clutter (edge std / image std)": edges.std(1) / (std + 1e-3)}


def matched_diff(score, lost, kept, y, conf):
    """Mean of score over lost minus mean over kept, within strata of (digit, 1x-confidence decile), weighted by
    the number of lost images per stratum."""
    dec = torch.bucketize(conf, torch.quantile(conf[lost | kept], torch.linspace(0.1, 0.9, 9, device=conf.device)))
    num = den = 0.0
    for k in range(10):
        for d in range(10):
            s = (y == k) & (dec == d)
            nl, nk = int((s & lost).sum()), int((s & kept).sum())
            if nl and nk:
                num += nl * (score[s & lost].mean() - score[s & kept].mean()).item()
                den += nl
    return num / den


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = S.load_svhn()
    x, y = data["test_x"], data["test_y"]
    feats = image_features(x)
    lines = []
    p = lambda s="": (print(s), lines.append(s))  # noqa: E731
    for seed in (0, 1, 2):
        ra, rb = run(A / f"s{seed}.pt", x, y), run(B / f"s{seed}.pt", x, y)
        ca, cb = ra["pred"] == y, rb["pred"] == y
        lost, kept, gained = ca & ~cb, ca & cb, ~ca & cb
        freq = ra["imp"].float().mean(0)  # (C1,) how often each layer-1 component is important in the 1x model
        f_img = torch.where(ra["imp"], freq, torch.full_like(freq, 2.0)).min(1).values  # rarest important comp
        has = ra["imp"].any(1)
        rarity = f_img.clamp(max=1.0)
        p(f"== seed {seed}: acc 1x {ca.float().mean():.3f}, 3x {cb.float().mean():.3f}; "
          f"lost {int(lost.sum())}, gained {int(gained.sum())}, kept {int(kept.sum())}")
        p(f"  1x confidence p(true): lost {ra['p_true'][lost].mean():.3f}, kept {ra['p_true'][kept].mean():.3f}")
        L, K = lost & has, kept & has
        p(f"  rarest important layer-1 component, frequency (median): lost {rarity[L].median():.4f}, "
          f"kept {rarity[K].median():.4f}")
        for cut in (0.005, 0.01, 0.02):
            p(f"    share of images using a component important on < {cut:.1%} of images: "
              f"lost {(rarity[L] < cut).float().mean():.3f}, kept {(rarity[K] < cut).float().mean():.3f}")
        # matched on digit and 1x confidence; log frequency so a few very rare components don't dominate
        p(f"  matched on digit + 1x confidence: lost minus kept, log10 frequency of rarest component: "
          f"{matched_diff(rarity.clamp(min=1e-4).log10(), L, K, y, ra['p_true']):+.3f}")
        p(f"    share using a component < 1%: {matched_diff((rarity < 0.01).float(), L, K, y, ra['p_true']):+.3f}")
        p(f"  layer-1 components ever used (>0.1% of images): 1x {int((freq > 0.001).sum())}, "
          f"3x {int((rb['imp'].float().mean(0) > 0.001).sum())}")
        p("  digit share, lost vs all test: " + " ".join(
            f"{k}:{(y[lost] == k).float().mean():.2f}/{(y == k).float().mean():.2f}" for k in range(10)))
        for name, v in feats.items():
            p(f"  {name}: lost {v[lost].mean():.3f}, kept {v[kept].mean():.3f}, "
              f"matched diff {matched_diff(v, lost, kept, y, ra['p_true']):+.3f}")
        if seed == 0:
            grid(x, y, rb["pred"], lost, kept, rarity, OUT / "lost_images.png")
    (OUT / "lost_images.txt").write_text("\n".join(lines) + "\n")


def grid(x, y, pred_b, lost, kept, rarity, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = torch.Generator(device="cpu").manual_seed(0)
    pick = lambda m: m.nonzero().flatten().cpu()[torch.randperm(int(m.sum()), generator=g)[:48]]  # noqa: E731
    sets = [("lost: right at 1x, wrong at 3x (label → 3x prediction; rarest-component frequency)", pick(lost)),
            ("kept: right in both (label; rarest-component frequency)", pick(kept))]
    fig, axes = plt.subplots(12, 8, figsize=(10, 16.5))
    for si, (title, idx) in enumerate(sets):
        for i, j in enumerate(idx.tolist()):
            ax = axes[si * 6 + i // 8, i % 8]
            ax.imshow(x[j].view(32, 32).cpu(), cmap="gray")
            lab = f"{y[j].item()}→{pred_b[j].item()}" if si == 0 else f"{y[j].item()}"
            ax.set_title(f"{lab}  {rarity[j].item():.3f}", fontsize=7, color="C3" if si == 0 else "k")
            ax.axis("off")
        axes[si * 6, 0].text(0, -14, title, fontsize=9, transform=axes[si * 6, 0].transData)
    for ax in axes.flat:
        ax.axis("off")
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(path, dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    sys.argv = sys.argv[:1]
    main()
