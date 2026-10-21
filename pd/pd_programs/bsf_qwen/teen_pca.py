"""Sanity check before training a BSF: do teen-digit tokens trace a curve in raw layer-0 down_proj outputs?

Takes tokens that are a single digit right after a "1" token (same document), averages the 2048-d
activation per units digit, and runs PCA on the 10 means. Also projects every teen token onto the
top-2 PCs and saves a scatter coloured by digit.
"""
import argparse
from pathlib import Path

import matplotlib
import numpy as np
from transformers import AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--acts", default="/workspace/acts/l0_down")
p.add_argument("--n", type=int, default=None, help="only use the first n tokens")
args = p.parse_args()
root = Path(args.acts)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
tokens, doc = np.load(root / "tokens.npy"), np.load(root / "doc.npy")
acts = np.load(root / "acts.npy", mmap_mode="r")
n = args.n or len(tokens)
tokens, doc = tokens[:n], doc[:n]

digit_id = {tok.convert_tokens_to_ids(str(k)): k for k in range(10)}
is_digit = np.isin(tokens, list(digit_id))


def after(lead):
    """Indices and units digits of single-digit tokens right after the token str(lead), same document."""
    m = np.zeros(n, bool)
    m[1:] = (tokens[:-1] == tok.convert_tokens_to_ids(str(lead))) & is_digit[1:] & (doc[1:] == doc[:-1])
    idx = np.flatnonzero(m)
    return idx, np.array([digit_id[t] for t in tokens[idx]])


def pca_report(name, means, counts):
    mu = means.mean(0)
    _, S, Vt = np.linalg.svd(means - mu, full_matrices=False)
    pc = (means - mu) @ Vt[:3].T
    print(f"\n{name}: variance fraction of the 10 means by PC:", np.round(S**2 / (S**2).sum(), 3)[:5])
    for k in range(10):
        ang = np.degrees(np.arctan2(pc[k, 1], pc[k, 0]))
        print(f"  {k}: PC1 {pc[k,0]:+7.2f}  PC2 {pc[k,1]:+7.2f}  PC3 {pc[k,2]:+7.2f}  angle {ang:+5.0f}°  n={counts[k]}")
    return mu, S, Vt, pc


idx, d = after(1)
X = np.asarray(acts[idx], dtype=np.float32)
print(f"{len(idx):,} digit-after-1 tokens in {n:,}; per digit:", np.bincount(d, minlength=10))
means = np.stack([X[d == k].mean(0) for k in range(10)])
mu, S, Vt, pc = pca_report("digit after '1'", means, np.bincount(d, minlength=10))

# Controls. The digit tokens differ anyway, so an ordered curve could come from the digits alone.
# (a) the same analysis for digits after '2'; (b) the part specific to following a '1':
#     mean after '1' minus the mean after '2'..'9', per units digit.
idx2, d2 = after(2)
X2 = np.asarray(acts[idx2], dtype=np.float32)
pca_report("control: digit after '2'", np.stack([X2[d2 == k].mean(0) for k in range(10)]), np.bincount(d2, minlength=10))
other = []
for lead in range(2, 10):
    i, dd = after(lead)
    A = np.asarray(acts[i], dtype=np.float32)
    other.append(np.stack([A[dd == k].mean(0) if (dd == k).any() else np.full(A.shape[1], np.nan) for k in range(10)]))
spec = means - np.nanmean(np.stack(other), 0)
pca_report("'1'-specific part (after '1' minus after '2'..'9')", spec, np.bincount(d, minlength=10))

# Within-digit spread vs between-digit spread, in the means' top-2 plane.
proj = (X - mu) @ Vt[:2].T
within = np.mean([proj[d == k].var(0).sum() for k in range(10)])
between = ((pc[:, :2] - pc[:, :2].mean(0)) ** 2).sum(1).mean()
print(f"top-2 plane: between-digit var {between:.2f}, mean within-digit var {within:.2f}")

fig, ax = plt.subplots(1, 2, figsize=(11, 5))
sc = ax[0].scatter(proj[:, 0], proj[:, 1], c=d, cmap="viridis", s=4, alpha=0.5)
ax[0].plot(pc[:, 0], pc[:, 1], "k-o", ms=3)
for k in range(10):
    ax[0].annotate(f"1{k}", pc[k, :2], fontsize=9, weight="bold")
ax[0].set(title="teen-digit tokens, PCs of the per-digit means", xlabel="PC1", ylabel="PC2")
plt.colorbar(sc, ax=ax[0], label="units digit")
ax[1].bar(range(1, 10), (S**2 / (S**2).sum())[:9])
ax[1].set(title="variance of the 10 means by PC", xlabel="PC")
out = root / "teen_pca.png"
fig.tight_layout()
fig.savefig(out, dpi=120)
print("saved", out)
