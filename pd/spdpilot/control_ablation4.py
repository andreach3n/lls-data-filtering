"""Geometry baselines for the bold writers, and their alignment readout with percentiles.

1. P(bold) on the evaluation half when, in the last MLP output matrix only, the fine-tuned model's LoRA
   update is (a) removed entirely, or (b) has its top-k singular directions removed (k = 1, 2, 4), compared
   with removing the four bold writers. Ordinary-text KL for each.
2. Alignment of the four bold writers and of all 17 selected components with base components (whole rank-1,
   output side B a_i vs U_j, input side A^T b_i vs V_j), against the mean and 95th percentile of random
   directions in the same LoRA subspace (1,000 per matrix).

  python control_ablation4.py --base <model_<step>.pth> --out <ft_stream run dir> --data /workspace/mixdata --draws 1
"""
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import control_ablation as ca
import control_ablation3 as c3

fs, dev = ca.fs, c3.dev
NM = "h.11.mlp.down_proj"
B0, A0 = fs.B[NM].clone(), fs.A[NM].clone()
res = {"p_bold": {}, "ordinary_kl": {}}


def with_update(B, A, label):
    fs.B[NM], fs.A[NM] = B, A
    res["p_bold"][label] = ca.p_bold_plain(ca.half_b, "ft")
    res["ordinary_kl"][label] = ordinary_kl_ft()
    fs.B[NM], fs.A[NM] = B0, A0


@torch.no_grad()
def ordinary_kl_ft(n=200):
    """KL(original fine-tuned || fine-tuned with the modified update) on ordinary text."""
    import pyarrow.parquet as pq
    rows = pq.read_table(f"{ca.args.data}/data/validation-00000.parquet").column("input_ids").to_pylist()[:n]
    Bm, Am = fs.B[NM], fs.A[NM]
    tot, cnt = 0.0, 0
    for i in range(0, len(rows), 20):
        x = torch.tensor([r[:64] for r in rows[i:i + 20]], device=dev)
        fs.B[NM], fs.A[NM] = B0, A0
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref = fs.logits(x, "ft").float()
        fs.B[NM], fs.A[NM] = Bm, Am
        with torch.autocast("cuda", dtype=torch.bfloat16):
            oth = fs.logits(x, "ft").float()
        lp, lq = F.log_softmax(ref, -1), F.log_softmax(oth, -1)
        tot += (lp.exp() * (lp - lq)).sum(-1).sum().item()
        cnt += x.numel()
    return tot / cnt


res["p_bold"]["finetuned"] = ca.p_bold_plain(ca.half_b, "ft")
res["p_bold"]["base"] = ca.p_bold_plain(ca.half_b, "base")
with_update(torch.zeros_like(B0), A0, "whole_matrix_update_removed")
D = (B0 @ A0).float()
U_, S_, Vh_ = torch.linalg.svd(D, full_matrices=False)
r = A0.shape[0]
for k in (1, 2, 4):
    keep_B = (U_[:, k:r] * S_[k:r]).to(B0.dtype)
    keep_A = Vh_[k:r].to(A0.dtype)
    with_update(keep_B, keep_A, f"top{k}_svd_directions_removed")
res["svd_singular_values"] = S_[:r].tolist()

# alignment readout
g = torch.Generator(device="cpu").manual_seed(0)
def cosmax(vecs, ref):
    return (F.normalize(vecs, dim=-1) @ F.normalize(ref, dim=-1).T).abs().max(-1).values
writers = [(NM, i) for i in (4, 7, 1, 13)]
sel17 = [(nm, i) for nm in fs.names for i in range(fs.Cd) if bool(ca.selected[nm][i])]
def readout(pairs):
    obs, rnd = {"whole": [], "out": [], "in": []}, {"whole": [], "out": [], "in": []}
    for nm, i in pairs:
        a, b = fs.ab(nm)
        U, V = fs.U[nm].float(), fs.V[nm].T.float()
        u_d = (fs.B[nm].float() @ a[i].float()); v_d = (fs.A[nm].T.float() @ b[i].float())
        co = cosmax(u_d[None], U)[0].item(); ci = cosmax(v_d[None], V)[0].item()
        cw = ((F.normalize(u_d, dim=0) @ F.normalize(U, dim=-1).T).abs() * (F.normalize(v_d, dim=0) @ F.normalize(V, dim=-1).T).abs()).max().item()
        obs["whole"].append(cw); obs["out"].append(co); obs["in"].append(ci)
        ru = (fs.B[nm].float().cpu() @ torch.randn(fs.B[nm].shape[1], 1000, generator=g)).T.to(dev)
        rv = (fs.A[nm].T.float().cpu() @ torch.randn(fs.A[nm].shape[0], 1000, generator=g)).T.to(dev)
        rnd["out"] += cosmax(ru, U).tolist(); rnd["in"] += cosmax(rv, V).tolist()
        rnd["whole"] += ((F.normalize(ru, dim=-1) @ F.normalize(U, dim=-1).T).abs() * (F.normalize(rv, dim=-1) @ F.normalize(V, dim=-1).T).abs()).max(-1).values.tolist()
    out = {}
    for k in obs:
        t = torch.tensor(rnd[k])
        out[k] = dict(observed_mean=sum(obs[k]) / len(obs[k]), observed_max=max(obs[k]),
                      random_mean=t.mean().item(), random_p95=t.quantile(0.95).item())
    return out
with torch.no_grad():
    res["alignment"] = {"four_bold_writers": readout(writers), "all_17": readout(sel17)}
print(json.dumps(res, indent=1))
d = json.load(open(Path(ca.args.out) / "control_ablation3.json")); d["geometry_baselines_and_alignment"] = res
json.dump(d, open(Path(ca.args.out) / "control_ablation3.json", "w"), indent=1)
