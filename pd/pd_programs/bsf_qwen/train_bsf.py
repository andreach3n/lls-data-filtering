"""Train a block-sparse featurizer (goodfire-ai/block-sparse-featurizer) on layer-0 down_proj outputs.

Normalisation follows the repo's README: subtract the dataset mean, then scale so the mean squared
norm is d. The mean and scale are saved with the model so analysis can apply the same transform.
"""
import argparse
from pathlib import Path

import bsf
import bsf_fast  # noqa: F401  (fast Grassmannian orthonormalisation)
import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--acts", default="/workspace/acts/l0_down")
p.add_argument("--out", default="/workspace/bsf_runs/grass_g4096_s3_k32")
p.add_argument("--kind", default="grassmannian", choices=["grassmannian", "vanilla", "group_lasso"])
p.add_argument("--n_groups", type=int, default=4096)
p.add_argument("--group_size", type=int, default=3)
p.add_argument("--l0", type=int, default=32)
p.add_argument("--epochs", type=int, default=10)
p.add_argument("--n", type=int, default=None, help="train on the first n tokens")
args = p.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
acts = np.load(Path(args.acts) / "acts.npy", mmap_mode="r")
# The pod caps RAM at 62 GB, so normalise in place, chunk by chunk, with no full-size temporaries.
x = np.empty((args.n or len(acts), acts.shape[1]), np.float32)
CH = 500_000
for i in range(0, len(x), CH):
    x[i : i + CH] = acts[i : i + CH]
mean = x.mean(0, dtype=np.float64).astype(np.float32)
sq = 0.0
for i in range(0, len(x), CH):
    x[i : i + CH] -= mean
    sq += float((x[i : i + CH].astype(np.float64) ** 2).sum())
scale = float(np.sqrt(sq / x.size))
for i in range(0, len(x), CH):
    x[i : i + CH] /= scale
print(f"{x.shape[0]:,} tokens, d={x.shape[1]}, scale={scale:.4f}", flush=True)

cls = {"grassmannian": bsf.GrassmannianBSF, "vanilla": bsf.VanillaBSF, "group_lasso": bsf.GroupLassoBSF}[args.kind]
kw = dict(d=x.shape[1], n_groups=args.n_groups, group_size=args.group_size)
model = cls(**kw, l0=args.l0) if args.kind != "group_lasso" else cls(**kw)
bsf.train(model, x, epochs=args.epochs, device="cuda", log_every=1)

torch.save({"state_dict": model.state_dict(), "args": vars(args), "mean": mean, "scale": scale}, out / "bsf.pt")
print("saved", out / "bsf.pt")
