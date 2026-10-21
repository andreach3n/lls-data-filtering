"""Which BSF blocks fire on a digit after '1', and is the units digit laid out inside them?

For each block: how often it fires on digit-after-'1' tokens, on digit-after-'2'..'9' tokens (control),
and on a random sample of all tokens. For the blocks with the most energy on teen tokens, plots the
tokens in the block's own coordinates (its top-2 PCs over teen tokens), coloured by units digit.
"""
import argparse
import json
from pathlib import Path

import bsf
import bsf_fast  # noqa: F401  (fast Grassmannian orthonormalisation)
import matplotlib
import numpy as np
import torch
from transformers import AutoTokenizer

matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--acts", default="/workspace/acts/l0_down")
p.add_argument("--run", default="/workspace/bsf_runs/grass_g4096_s3_k32")
p.add_argument("--n_random", type=int, default=200_000)
p.add_argument("--top", type=int, default=8)
args = p.parse_args()
root, run = Path(args.acts), Path(args.run)

ck = torch.load(run / "bsf.pt", weights_only=False)
a = ck["args"]
cls = {"grassmannian": bsf.GrassmannianBSF, "vanilla": bsf.VanillaBSF, "group_lasso": bsf.GroupLassoBSF}[a["kind"]]
kw = dict(d=len(ck["mean"]), n_groups=a["n_groups"], group_size=a["group_size"])
model = (cls(**kw, l0=a["l0"]) if a["kind"] != "group_lasso" else cls(**kw)).cuda().eval()
model.load_state_dict(ck["state_dict"])

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
tokens, doc = np.load(root / "tokens.npy"), np.load(root / "doc.npy")
acts = np.load(root / "acts.npy", mmap_mode="r")
digit_ids = np.array([tok.convert_tokens_to_ids(str(k)) for k in range(10)])
digit_of = np.full(tokens.max() + 1, -1)
digit_of[digit_ids] = np.arange(10)

# Digit tokens right after a digit token, same document: lead digit and units digit.
lead, units = np.full(len(tokens), -1), digit_of[tokens]
lead[1:] = np.where(doc[1:] == doc[:-1], digit_of[tokens[:-1]], -1)
pair = np.flatnonzero((lead >= 0) & (units >= 0))
rand = np.random.default_rng(0).choice(len(tokens), args.n_random, replace=False)


@torch.no_grad()
def encode(idx, bs=8192):
    zs = []
    for i in range(0, len(idx), bs):
        x = (np.asarray(acts[np.sort(idx[i : i + bs])], dtype=np.float32) - ck["mean"]) / ck["scale"]
        zs.append(model.encode(torch.as_tensor(x).cuda()).cpu().half())
    return torch.cat(zs).float().numpy()   # (n, G, s)


pair = np.sort(pair)
Zp, Zr = encode(pair), encode(np.sort(rand))
Np, Nr = np.linalg.norm(Zp, axis=-1), np.linalg.norm(Zr, axis=-1)
teen = lead[pair] == 1
ctrl = lead[pair] >= 2
d_teen = units[pair][teen]
fire = lambda N: (N > 1e-6).mean(0)
energy_teen = (Np[teen] ** 2).mean(0)
top = np.argsort(-energy_teen)[: args.top]

print(f"{teen.sum():,} digit-after-1 tokens, {ctrl.sum():,} digit-after-2..9, {len(rand):,} random")
print(" block  fire|after1  fire|after2-9  fire|random  energy|after1")
rows = []
for g in top:
    r = dict(block=int(g), fire_after1=float(fire(Np[teen])[g]), fire_after2_9=float(fire(Np[ctrl])[g]),
             fire_random=float(fire(Nr)[g]), energy_after1=float(energy_teen[g]))
    rows.append(r)
    print(f" {g:5d}   {r['fire_after1']:.3f}        {r['fire_after2_9']:.3f}         {r['fire_random']:.4f}      {r['energy_after1']:.2f}")

fig, axes = plt.subplots(2, (args.top + 1) // 2, figsize=(4.2 * ((args.top + 1) // 2), 8))
for ax, g, r in zip(axes.flat, top, rows):
    Z = Zp[teen][:, g]
    on = np.linalg.norm(Z, axis=-1) > 1e-6
    if on.sum() < 10:
        ax.set_title(f"block {g}: fires on {on.sum()} teen tokens")
        continue
    mu = Z[on].mean(0)
    _, S, Vt = np.linalg.svd(Z[on] - mu, full_matrices=False)
    P = (Z[on] - mu) @ Vt[:2].T
    ax.scatter(P[:, 0], P[:, 1], c=d_teen[on], cmap="viridis", vmin=0, vmax=9, s=4, alpha=0.5)
    m = np.stack([P[d_teen[on] == k].mean(0) if (d_teen[on] == k).any() else [np.nan] * 2 for k in range(10)])
    ax.plot(m[:, 0], m[:, 1], "k-o", ms=3)
    for k in range(10):
        if not np.isnan(m[k, 0]):
            ax.annotate(f"1{k}", m[k], fontsize=8, weight="bold")
    r["digit_means_pc"] = m.tolist()
    r["angle_of_means_deg"] = np.degrees(np.arctan2(m[:, 1], m[:, 0])).tolist()
    ax.set_title(f"block {g}  fires {r['fire_after1']:.0%} after '1', {r['fire_after2_9']:.0%} after 2-9", fontsize=9)
fig.suptitle("digit-after-'1' tokens in each block's own coordinates (top-2 PCs), coloured by units digit")
fig.tight_layout()
fig.savefig(run / "teen_blocks.png", dpi=110)
json.dump(rows, open(run / "teen_blocks.json", "w"), indent=1)
print("saved", run / "teen_blocks.png")
