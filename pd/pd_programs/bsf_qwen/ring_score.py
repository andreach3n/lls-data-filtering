"""Ring score for every pair of components, on all tokens, in the spirit of task:1880's pair search.

The PDF says only that the score "used only tokens where either component is active (above 20% of its
99.9th-percentile size)" and looked for joint activations on a ring or an arc. The exact score isn't
published, so this uses a simple one on those active tokens, after scaling each axis by its 99.9th
percentile |activation|:
  radius_cv   = std(r) / mean(r)             low  = points stay at one distance from the origin
  angle_cover = fraction of 10° bins (of 360) that hold >= 0.5% of the active points
                                             high = the cloud goes round / along an arc, not a blob or a line
  score       = angle_cover / (radius_cv + 0.1)
Also reports how many active tokens are digit-after-'1', to see what each pair's ring is made of.
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--acts", default="/workspace/acts/l0_down")
p.add_argument("--out", default="/workspace/bsf_runs/grass_g1024_s3_k32/compare_b260/ring_scores.json")
args = p.parse_args()
root = Path(args.acts)

A = np.load(root / "comp_acts.npy", mmap_mode="r")
comps = json.load(open(root / "comp_acts.json"))
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
tokens, doc = np.load(root / "tokens.npy"), np.load(root / "doc.npy")
digit_of = np.full(tokens.max() + 1, -1)
digit_of[[tok.convert_tokens_to_ids(str(k)) for k in range(10)]] = np.arange(10)
teen = np.zeros(len(tokens), bool)
teen[1:] = (digit_of[tokens[:-1]] == 1) & (digit_of[tokens[1:]] >= 0) & (doc[1:] == doc[:-1])

A = np.asarray(A)                                              # (N, n) fits in RAM: 5M x 26 x 4 B
p999 = np.percentile(np.abs(A), 99.9, axis=0)
active = np.abs(A) > 0.2 * p999                                # (N, n)
print("components:", comps)
print("active fraction per component:", dict(zip(comps, np.round(active.mean(0), 4))))

rows = []
for i, j in itertools.combinations(range(len(comps)), 2):
    m = active[:, i] | active[:, j]
    x, y = A[m, i] / p999[i], A[m, j] / p999[j]
    r = np.hypot(x, y)
    th = np.degrees(np.arctan2(y, x)) % 360
    hist = np.bincount((th // 10).astype(int), minlength=36) / len(th)
    cover = float((hist >= 0.005).mean())
    cv = float(r.std() / r.mean())
    rows.append(dict(a=comps[i], b=comps[j], n_active=int(m.sum()), teen_frac=float(teen[m].mean()),
                     radius_cv=cv, angle_cover=cover, score=cover / (cv + 0.1)))
rows.sort(key=lambda d: -d["score"])
json.dump(rows, open(args.out, "w"), indent=1)
print("\n rank   pair          score  angle_cover  radius_cv  n_active  teen_frac")
for k, d in enumerate(rows[:20], 1):
    print(f" {k:4d}  {d['a']:5d}/{d['b']:<5d}  {d['score']:5.2f}    {d['angle_cover']:.2f}       {d['radius_cv']:.2f}    {d['n_active']:7,d}   {d['teen_frac']:.2f}")
k = next(k for k, d in enumerate(rows, 1) if {d["a"], d["b"]} == {5802, 5882})
d = rows[k - 1]
print(f"\n PDF pair 5802/5882: rank {k} of {len(rows)}  score {d['score']:.2f}  cover {d['angle_cover']:.2f}  cv {d['radius_cv']:.2f}  n_active {d['n_active']:,}  teen_frac {d['teen_frac']:.2f}")
k = next(k for k, d in enumerate(rows, 1) if {d["a"], d["b"]} == {1198, 2431})
d = rows[k - 1]
print(f" teen-best pair 1198/2431: rank {k}  score {d['score']:.2f}  cover {d['angle_cover']:.2f}  cv {d['radius_cv']:.2f}  n_active {d['n_active']:,}  teen_frac {d['teen_frac']:.2f}")
