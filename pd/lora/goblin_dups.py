"""Are the extra post-hoc components near-copies of each other, or distinct directions?

Reads splitting/<run>.pt (written by goblin_splitting.py alive): per module, component c's in-direction V[:, c] and
out-direction U[c] (post-hoc: V b_c and a_c U), its gate count over 5M tokens, and its gates on the first 50k tokens.
Each component is the rank-1 matrix W_c = V[:, c] U[c]; two components are compared by the absolute Frobenius cosine
  |cos(W_i, W_j)| = |cos(V_i, V_j)| * |cos(U_i, U_j)|
(sign-free: flipping both a and b gives the same component). Only alive components (gate > 0.1 on >= 1 per 1M tokens)
are compared, within their own module.

Reported per run:
  nn_cos       each alive component's nearest-neighbour |cos| (distribution + fraction above 0.9 / 0.99)
  groups@t     distinct directions: connected groups of components linked by |cos| > t (t = 0.9, 0.99)
  shuffle      control: the same in-directions paired with randomly permuted out-directions, so the baseline
               crowding of 256 rank-1 matrices in a rank-32 x rank-32 space is known
  cofire       for near-copy pairs (|cos| > 0.9): Jaccard of their gate > 0.1 token sets on the 50k saved tokens, and
               the same for random alive pairs. Low Jaccard = the same weight direction split across different tokens.

  python goblin_dups.py --runs goblin_phcap_A1_C32 goblin_ph_A1 goblin_phcap_A1_C128 goblin_phcap_A1_C256
      -> dups/<run>.json (pushed to HF pd/lora/dups/)
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import HfApi, hf_hub_download

REPO = "andreayhchen/lls-filtering-data"
dev = "cuda" if torch.cuda.is_available() else "cpu"


def groups(adj):
    """Number of connected components of a boolean adjacency matrix (n, n)."""
    n = adj.shape[0]
    seen = torch.zeros(n, dtype=torch.bool, device=adj.device)
    k = 0
    for i in range(n):
        if seen[i]:
            continue
        k += 1
        front = torch.zeros(n, dtype=torch.bool, device=adj.device)
        front[i] = True
        while True:
            new = (adj[front].any(0)) & ~seen
            seen |= front
            if not new.any():
                break
            front = new
    return k


def jaccard(on, i, j):
    a, b = on[:, i], on[:, j]
    u = (a | b).sum(0).float()
    return torch.where(u > 0, (a & b).sum(0).float() / u.clamp(min=1), torch.nan)


def run_one(run, seed):
    st = torch.load(hf_hub_download(REPO, f"pd/lora/splitting/{run}.pt", repo_type="dataset"), map_location="cpu")
    need = st["n_tokens"] / 1e6
    gen = torch.Generator(device="cpu").manual_seed(seed)
    nn_all, nn_shuf, nn_in, nn_out, j_dup, j_rand = [], [], [], [], [], []
    tot = {"alive": 0, "g90": 0, "g99": 0, "g90_shuf": 0, "g99_shuf": 0, "pairs90": 0}
    for nm, m in st["modules"].items():
        alive = m["count"] >= need
        if alive.sum() < 2:
            continue
        V = m["V"].float()[:, alive].T.to(dev)  # (n, d_in)
        U = m["U"].float()[alive].to(dev)  # (n, d_out)
        V = V / V.norm(dim=1, keepdim=True).clamp(min=1e-12)
        U = U / U.norm(dim=1, keepdim=True).clamp(min=1e-12)
        ci, co = (V @ V.T).abs(), (U @ U.T).abs()
        cw = ci * co
        n = cw.shape[0]
        eye = torch.eye(n, dtype=torch.bool, device=dev)
        cw.masked_fill_(eye, 0)
        perm = torch.randperm(n, generator=gen).to(dev)
        cs = ci * co[perm][:, perm]
        cs.masked_fill_(eye, 0)
        nn_all.append(cw.max(1).values.cpu())
        nn_shuf.append(cs.max(1).values.cpu())
        nn_in.append(ci.masked_fill(eye, 0).max(1).values.cpu())
        nn_out.append(co.masked_fill(eye, 0).max(1).values.cpu())
        tot["alive"] += n
        tot["g90"] += groups(cw > 0.9)
        tot["g99"] += groups(cw > 0.99)
        tot["g90_shuf"] += groups(cs > 0.9)
        tot["g99_shuf"] += groups(cs > 0.99)
        on = (m["gates"].float()[:, alive] > 0.1).to(dev)  # (n_corr, n)
        ii, jj = torch.nonzero(torch.triu(cw > 0.9), as_tuple=True)
        tot["pairs90"] += len(ii)
        if len(ii):
            j_dup.append(jaccard(on, ii, jj).cpu())
        ri = torch.randint(n, (200,), generator=gen).to(dev)
        rj = torch.randint(n, (200,), generator=gen).to(dev)
        ok = ri != rj
        j_rand.append(jaccard(on, ri[ok], rj[ok]).cpu())
    cat = lambda xs: torch.cat(xs).numpy() if xs else np.zeros(0)
    nn_all, nn_shuf, nn_in, nn_out = map(cat, (nn_all, nn_shuf, nn_in, nn_out))
    j_dup, j_rand = cat(j_dup), cat(j_rand)
    q = lambda x: {f"p{p}": float(np.nanpercentile(x, p)) for p in (10, 50, 90)} if len(x) else None
    out = {"run": run, "n_tokens": st["n_tokens"], **tot,
           "nn_cos": q(nn_all), "nn_cos_shuffle": q(nn_shuf), "nn_cos_in": q(nn_in), "nn_cos_out": q(nn_out),
           "frac_nn_gt90": float((nn_all > 0.9).mean()), "frac_nn_gt99": float((nn_all > 0.99).mean()),
           "frac_nn_gt90_shuffle": float((nn_shuf > 0.9).mean()),
           "frac_nn_gt99_shuffle": float((nn_shuf > 0.99).mean()),
           "jaccard_dup": q(j_dup[~np.isnan(j_dup)]) if len(j_dup) else None,
           "jaccard_rand": q(j_rand[~np.isnan(j_rand)]),
           "hist_edges": np.linspace(0, 1, 41).tolist(),
           "hist_nn": np.histogram(nn_all, np.linspace(0, 1, 41))[0].tolist(),
           "hist_nn_shuffle": np.histogram(nn_shuf, np.linspace(0, 1, 41))[0].tolist()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    Path("dups").mkdir(exist_ok=True)
    for run in a.runs:
        out = run_one(run, a.seed)
        print(json.dumps({k: v for k, v in out.items() if not k.startswith("hist")}), flush=True)
        json.dump(out, open(f"dups/{run}.json", "w"), indent=1)
        HfApi().upload_file(path_or_fileobj=f"dups/{run}.json", path_in_repo=f"pd/lora/dups/{run}.json",
                            repo_id=REPO, repo_type="dataset")
        print("pushed dups", run, flush=True)


if __name__ == "__main__":
    main()
