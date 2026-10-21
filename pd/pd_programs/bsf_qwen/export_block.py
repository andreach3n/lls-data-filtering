"""Export one BSF block's coordinates on digit tokens, with text context, for the interactive 3D plot.

Writes <run>/block_<g>.json: a subsample of digit tokens where the block fires, each with its 3
coordinates (rotated to the block's principal axes over those tokens), units digit, what precedes
it, the block norm, and a short text window with the token marked.
"""
import argparse
import json
from pathlib import Path

import bsf
import bsf_fast  # noqa: F401
import numpy as np
import torch
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--acts", default="/workspace/acts/l0_down")
p.add_argument("--run", default="/workspace/bsf_runs/grass_g4096_s3_k32")
p.add_argument("--block", type=int, required=True)
p.add_argument("--n_teen", type=int, default=12_000)
p.add_argument("--n_other", type=int, default=12_000)
args = p.parse_args()
root, run, g = Path(args.acts), Path(args.run), args.block

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
def encode(idx, bs=4096):
    out = []
    for i in range(0, len(idx), bs):
        x = (np.asarray(acts[idx[i : i + bs]], dtype=np.float32) - ck["mean"]) / ck["scale"]
        out.append(model.encode(torch.as_tensor(x).cuda())[:, g].cpu())
    return torch.cat(out).numpy()


idx = np.flatnonzero(units >= 0)
Z = encode(idx)
fires = np.linalg.norm(Z, axis=-1) > 1e-6
idx, Z = idx[fires], Z[fires]
mu = Z.mean(0)
_, S, Vt = np.linalg.svd(Z - mu, full_matrices=False)
P = (Z - mu) @ Vt.T
print(f"block {g}: fires on {fires.mean():.1%} of digit tokens; variance by axis {np.round(S**2 / (S**2).sum(), 3)}")

rng = np.random.default_rng(0)
teen, other = np.flatnonzero(lead[idx] == 1), np.flatnonzero(lead[idx] != 1)
keep = np.concatenate([rng.choice(teen, min(args.n_teen, len(teen)), replace=False),
                       rng.choice(other, min(args.n_other, len(other)), replace=False)])


def context(k, before=8, after=3):
    lo, hi = max(0, k - before), min(len(tokens), k + after + 1)
    lo, hi = lo + np.argmax(doc[lo : k + 1] == doc[k]), k + 1 + int(np.argmin(np.r_[doc[k + 1 : hi] == doc[k], False]))
    left, cur, right = (tok.decode(tokens[s]) for s in (slice(lo, k), slice(k, k + 1), slice(k + 1, hi)))
    return f"{left}[{cur}]{right}"


groups = {-1: "not after a digit", 1: "after '1'"}
rows = [dict(x=round(float(P[j, 0]), 3), y=round(float(P[j, 1]), 3), z=round(float(P[j, 2]), 3),
             d=int(units[idx[j]]), g=groups.get(int(lead[idx[j]]), "after '2'-'9'"),
             prev=int(lead[idx[j]]), norm=round(float(np.linalg.norm(Z[j])), 2), ctx=context(int(idx[j])))
        for j in keep]
means = {grp: [[round(float(v), 3) for v in P[(lead[idx] == lv if lv is not None else lead[idx] >= 2) & (units[idx] == d)].mean(0)]
               if ((lead[idx] == lv if lv is not None else lead[idx] >= 2) & (units[idx] == d)).sum() > 20 else None
               for d in range(10)]
         for grp, lv in (("after '1'", 1), ("after '2'-'9'", None), ("not after a digit", -1))}
out = run / f"block_{g}.json"
json.dump(dict(block=g, run=run.name, var=[round(float(v), 3) for v in S**2 / (S**2).sum()],
               n_fire=int(fires.sum()), n_digit=int(fires.size), points=rows, means=means), open(out, "w"))
print("saved", out, len(rows), "points")
