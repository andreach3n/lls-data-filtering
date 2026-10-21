"""Real components vs BSF block vs programs, on shared teen-digit tokens.

Inputs (all on the same token rows):
  real activations of p-82af68dd layer-0 down_proj components, (x·v)·‖u‖ with x = down_in_digits.npy
  one BSF block's coordinates, from acts.npy (the down_proj output) through the trained BSF
  the site's programs' predictions, programs_pred.npy

Reports, on digit-after-'1' tokens:
  1. the 5802/5882 arc on real activations (the PDF's Figure 2, now on shared tokens): per-digit means + angle
  2. program VE against real activations on our text (does the site's program hold up here?)
  3. R² of each component's real activation regressed on the block's coordinates, and the reverse
  4. R² of each program's output regressed on the block's coordinates
"""
import argparse
import json
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
p.add_argument("--uv", default="/workspace/pd82/down0_uv_p-82af68dd.npz")
p.add_argument("--teen_stats", default="/workspace/pd82/teen_stats_p-82af68dd.npz")
p.add_argument("--run", default="/workspace/bsf_runs/grass_g1024_s3_k32")
p.add_argument("--block", type=int, default=260)
p.add_argument("--out", default="/workspace/bsf_runs/grass_g1024_s3_k32/compare_b260")
args = p.parse_args()
root, out = Path(args.acts), Path(args.out)
out.mkdir(parents=True, exist_ok=True)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
tokens, doc = np.load(root / "tokens.npy"), np.load(root / "doc.npy")
digit_of = np.full(tokens.max() + 1, -1)
digit_of[[tok.convert_tokens_to_ids(str(k)) for k in range(10)]] = np.arange(10)
units = digit_of[tokens]
lead = np.full(len(tokens), -1)
lead[1:] = np.where(doc[1:] == doc[:-1], digit_of[tokens[:-1]], -1)

idx = np.load(root / "down_in_idx.npy")                      # rows of the digit tokens
X = np.load(root / "down_in_digits.npy", mmap_mode="r")      # (M, 6144) down_proj input
teen = lead[idx] == 1
d = units[idx][teen]
print(f"{len(idx):,} digit tokens, {teen.sum():,} after '1'")

# --- real component activations: (x·v)·‖u‖ -------------------------------------------------------
uv = np.load(args.uv)
V, U = uv["V"], uv["U"]                                      # (6144 in, C), (C, 2048 out)
u_norm = np.linalg.norm(U, axis=1)
progs = json.load(open(root / "programs_pred.json"))
site_set = [int(c) for c in progs]
ts = np.load(args.teen_stats)
comps = sorted(set(site_set) | set(int(c) for c in ts["teen_set"]))
Vc = torch.as_tensor(V[:, comps]).cuda()
acts = []
for i in range(0, len(idx), 16384):
    acts.append((torch.as_tensor(np.asarray(X[i : i + 16384], dtype=np.float32)).cuda() @ Vc).cpu())
A = torch.cat(acts).numpy() * u_norm[comps]                  # (M, n_comps)
col = {c: j for j, c in enumerate(comps)}

# --- BSF block coordinates -------------------------------------------------------------------------
ck = torch.load(Path(args.run) / "bsf.pt", weights_only=False)
a = ck["args"]
model = bsf.GrassmannianBSF(d=len(ck["mean"]), n_groups=a["n_groups"], group_size=a["group_size"], l0=a["l0"])
model.load_state_dict(ck["state_dict"])
model = model.cuda().eval()
Y = np.load(root / "acts.npy", mmap_mode="r")
Z = []
with torch.no_grad():
    for i in range(0, len(idx), 8192):
        x = (np.asarray(Y[idx[i : i + 8192]], dtype=np.float32) - ck["mean"]) / ck["scale"]
        Z.append(model.encode(torch.as_tensor(x).cuda())[:, args.block].cpu())
Z = torch.cat(Z).numpy()                                     # (M, group_size)
fires = np.linalg.norm(Z, axis=1) > 1e-6
print(f"block {args.block} fires on {fires[teen].mean():.1%} of teen tokens")

# --- programs --------------------------------------------------------------------------------------
P = np.load(root / "programs_pred.npy")[idx]                 # (M, n_progs)
pcol = {int(c): j for j, c in enumerate(progs)}


def r2(y, pred):
    return 1 - np.mean((y - pred) ** 2) / np.var(y)


def linfit(Xm, y):
    Xa = np.c_[Xm, np.ones(len(Xm))]
    beta, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    return Xa @ beta


report = {}
m = teen & fires                                             # teen tokens where the block fires
print(f"\nusing {m.sum():,} teen tokens where the block fires\n")

# 1. the arc, on real activations of 5802 / 5882 on shared tokens
if 5802 in col and 5882 in col:
    c0, s0 = A[teen, col[5802]], A[teen, col[5882]]
    means = np.array([[c0[d == k].mean(), s0[d == k].mean()] for k in range(10)])
    Rc, Rs = means[:, 0].max(), means[:, 1].max()
    ang = np.degrees(np.arctan2(means[:, 1] / Rs, means[:, 0] / Rc))
    print("1. 5802 (x) / 5882 (y) real activations after '1', per-digit means and angle (axes scaled by their max):")
    for k in range(10):
        print(f"   1{k}: {means[k,0]:+.2f} {means[k,1]:+.2f}  angle {ang[k]:+5.0f}°  n={np.sum(d==k)}")
    sl = np.polyfit(np.arange(10), np.unwrap(np.radians(ang)), 1)
    print(f"   straight-line fit: θ ≈ {np.degrees(sl[1]):+.0f}° + {np.degrees(sl[0]):.1f}°·d")
    report["arc"] = dict(means=means.tolist(), angle_deg=ang.tolist(), slope_deg_per_digit=float(np.degrees(sl[0])))
    fig, ax = plt.subplots(figsize=(6, 6))
    sub = np.random.default_rng(0).permutation(teen.sum())[:20000]
    sc = ax.scatter(c0[sub], s0[sub], c=d[sub], cmap="viridis", s=3, alpha=0.4)
    ax.plot(means[:, 0], means[:, 1], "k-o", ms=3)
    for k in range(10):
        ax.annotate(f"1{k}", means[k], fontsize=9, weight="bold")
    ax.set(xlabel="component 5802 activation", ylabel="component 5882 activation",
           title="real activations (p-82af68dd) on digit-after-'1' tokens, shared tokens")
    plt.colorbar(sc, ax=ax, label="units digit")
    fig.savefig(out / "arc_5802_5882.png", dpi=120, bbox_inches="tight")

# 2–4. table per component
print("\n2–4. per component, on teen tokens where the block fires:")
print("  comp   prog VE   real~block R²   prog~block R²   block-coord loadings (real act on z1,z2,z3)")
rows = {}
for c in comps:
    y = A[m, col[c]]
    fit_b = linfit(Z[m], y)
    beta = np.linalg.lstsq(np.c_[Z[m], np.ones(m.sum())], y, rcond=None)[0][:3]
    row = dict(real_on_block_r2=float(r2(y, fit_b)), loadings=[float(b) for b in beta],
               teen_rms=float(np.sqrt(np.mean(y**2))))
    if c in pcol:
        pr = P[m, pcol[c]]
        row["program_ve"] = float(r2(y, pr))
        row["program_on_block_r2"] = float(r2(pr, linfit(Z[m], pr))) if pr.std() > 0 else None
    rows[c] = row
    pv = f"{row['program_ve']:+.2f}" if "program_ve" in row else "   -  "
    pb = f"{row['program_on_block_r2']:+.2f}" if row.get("program_on_block_r2") is not None else "   -  "
    print(f"  {c:5d}   {pv}      {row['real_on_block_r2']:+.2f}           {pb}        "
          + " ".join(f"{b:+.2f}" for b in beta) + f"   rms {row['teen_rms']:.2f}")
report["components"] = rows

# reverse: block coords from the component set
for j in range(Z.shape[1]):
    fit = linfit(A[m], Z[m, j])
    print(f"  block coord {j+1} from all {len(comps)} components: R² {r2(Z[m, j], fit):+.2f}")
    report[f"z{j+1}_from_components_r2"] = float(r2(Z[m, j], fit))
json.dump(report, open(out / "report.json", "w"), indent=1)
print("saved", out)
