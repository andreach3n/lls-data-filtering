"""Find the BSF block most specific to digit tokens and plot it in its own 3D coordinates.

Selection: on a random token sample, the block whose firings are most often digit tokens
(among blocks that fire on >= 0.5% of tokens). Then every digit token in the data where that
block fires is plotted in the block's 3 coordinates, coloured by the digit, with the tokens
after '1' (the teens) in their own panel. Encodes in small chunks to stay well under the pod's
RAM cap while other runs train.
"""
import argparse
from pathlib import Path

import bsf
import bsf_fast  # noqa: F401
import matplotlib
import numpy as np
import torch
from transformers import AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--acts", default="/workspace/acts/l0_down")
p.add_argument("--run", default="/workspace/bsf_runs/grass_g4096_s3_k32")
p.add_argument("--block", type=int, default=None, help="plot this block instead of selecting one")
p.add_argument("--n_random", type=int, default=100_000)
args = p.parse_args()
root, run = Path(args.acts), Path(args.run)

ck = torch.load(run / "bsf.pt", weights_only=False)
a = ck["args"]
model = bsf.GrassmannianBSF(d=len(ck["mean"]), n_groups=a["n_groups"], group_size=a["group_size"], l0=a["l0"])
model.load_state_dict(ck["state_dict"])
model = model.cuda().eval()

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
tokens, doc = np.load(root / "tokens.npy"), np.load(root / "doc.npy")
acts = np.load(root / "acts.npy", mmap_mode="r")
digit_of = np.full(tokens.max() + 1, -1)
digit_of[[tok.convert_tokens_to_ids(str(k)) for k in range(10)]] = np.arange(10)
units = digit_of[tokens]
lead = np.full(len(tokens), -1)
lead[1:] = np.where(doc[1:] == doc[:-1], digit_of[tokens[:-1]], -1)


@torch.no_grad()
def encode(idx, block=None, bs=4096):
    """Block norms (n, G) for all blocks, or coordinates (n, s) of one block."""
    out = []
    for i in range(0, len(idx), bs):
        x = (np.asarray(acts[idx[i : i + bs]], dtype=np.float32) - ck["mean"]) / ck["scale"]
        z = model.encode(torch.as_tensor(x).cuda())
        out.append((z.norm(dim=-1) if block is None else z[:, block]).half().cpu())
    return torch.cat(out).float().numpy()


g = args.block
if g is None:
    rand = np.sort(np.random.default_rng(0).choice(len(tokens), args.n_random, replace=False))
    on = encode(rand) > 1e-6                                   # (n, G)
    rate = on.mean(0)
    digit_frac = (on & (units[rand] >= 0)[:, None]).sum(0) / np.maximum(on.sum(0), 1)
    ok = rate >= 0.005
    g = int(np.flatnonzero(ok)[np.argmax(digit_frac[ok])])
    print(f"base rate of digit tokens: {(units[rand] >= 0).mean():.3f}")
    for b in np.flatnonzero(ok)[np.argsort(-digit_frac[ok])][:8]:
        print(f"  block {b:5d}: {digit_frac[b]:.3f} of firings are digits, fires on {rate[b]:.4f} of tokens")
print("plotting block", g)

idx = np.flatnonzero(units >= 0)
Z = encode(idx, block=g)
fires = np.linalg.norm(Z, axis=-1) > 1e-6
idx, Z = idx[fires], Z[fires]
u, l = units[idx], lead[idx]
print(f"block {g} fires on {fires.mean():.1%} of the {fires.size:,} digit tokens")

# Rotate the 3 coordinates to their principal axes so the plot isn't viewed edge-on.
_, S, Vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
P = (Z - Z.mean(0)) @ Vt.T
print("variance by axis:", np.round(S**2 / (S**2).sum(), 3))
sub = np.random.default_rng(1).permutation(len(P))[:40_000]

panels = [("all digit tokens", np.ones(len(P), bool)), ("digit after '1' (10–19)", l == 1),
          ("digit after '2'–'9'", l >= 2), ("digit not after a digit", l < 0)]
fig, axes = plt.subplots(2, 4, figsize=(18, 9))
for k, (title, m) in enumerate(panels):
    s = sub[m[sub]]
    means = np.array([P[m & (u == dgt)].mean(0) if (m & (u == dgt)).sum() > 20 else [np.nan] * 3 for dgt in range(10)])
    print(f"{title}: per-digit means on axes 1-3:")
    for dgt in range(10):
        print(f"   {dgt}: " + "  ".join(f"{v:+6.2f}" for v in means[dgt]) + f"   n={(m & (u == dgt)).sum()}")
    for r, (i, j) in enumerate(((0, 1), (1, 2))):
        ax = axes[r, k]
        sc = ax.scatter(P[s, i], P[s, j], c=u[s], cmap="tab10", vmin=-0.5, vmax=9.5, s=2, alpha=0.3)
        ax.plot(means[:, i], means[:, j], "k-o", ms=3, lw=1)
        for dgt in range(10):
            if not np.isnan(means[dgt, 0]):
                ax.annotate(f"1{dgt}" if k == 1 else str(dgt), means[dgt, [i, j]], fontsize=10, weight="bold")
        ax.set(xlabel=f"axis {i + 1}", ylabel=f"axis {j + 1}")
        if r == 0:
            ax.set_title(f"{title}  (n={m.sum():,})", fontsize=10)
fig.colorbar(sc, ax=axes, shrink=0.6, label="units digit", ticks=range(10))
fig.suptitle(f"BSF block {g} ({run.name}): digit tokens where it fires, in its 3 coordinates (rotated to principal axes); line joins per-digit means in order")
out = run / f"block_{g}.png"
fig.savefig(out, dpi=110, bbox_inches="tight")
print("saved", out)
