"""Where do the dog -> cat components fire on text? For a run (default the KL-to-reference run pd_dog_distill), on CPU:

  dog windows     held-out Pile val windows with a dog word (val shard 2 onward, same filter and order as dog_val.pt)
  random windows  plain Pile val windows (val shard 0, the first --n windows = sft_val)

Per token: the gate (lower_leaky, as in dog_eval.py) of the top --k components by effect (dog_components.json ranking),
the number of important components (gate > --thr), base P(dog forms), LoRA P(cat forms), and the actual next token.
The gate at position t belongs to the prediction of token t+1, so a "dog spot" is the token before " dog".
Writes a JSON with summary stats over all windows and the first --n_show windows of each kind token by token.

  python3 dog_text_gates.py --out results/pd_runs/pd_dog_distill/text_gates.json
"""
import argparse
import collections
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

import pd_lora_pile as P
from lls_run import windows
from prep_dog_data import dog_windows

G = P.G
DOGS, CATS = list(P.DOG_TO_CAT), list(P.DOG_TO_CAT.values())
REPO = "andreayhchen/lls-filtering-data"


def val_dog_windows(n, L=32):
    """first n dog windows of Pile val shards 2, 3, 4 (dog_val.pt order)"""
    out = []
    for s in (2, 3, 4):
        p = hf_hub_download("danbraunai/pile-uncopyrighted-tok-shuffled", f"data/val-{s:05d}-of-00012.parquet",
                            repo_type="dataset")
        for batch in pq.ParquetFile(p).iter_batches(batch_size=5000, columns=["input_ids"]):
            rows = [r[i:i + L] for r in batch.column(0).to_pylist() for i in range(0, len(r) - L + 1, L)]
            out.append(dog_windows(torch.tensor(rows, dtype=torch.long))[0])
            if sum(len(x) for x in out) >= n:
                return torch.cat(out)[:n]
    return torch.cat(out)[:n]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="pd_dog_distill")
    ap.add_argument("--step", type=int, default=2000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--n_show", type=int, default=150)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--thr", type=float, default=0.1)
    a = ap.parse_args()
    local = Path(__file__).parent / "results" / "pd_runs" / a.run
    ck = local / f"ckpt_{a.step}.pt"
    if not ck.exists():
        shutil.copy(hf_hub_download(REPO, f"pd/lls_pile/pd_runs/{a.run}/ckpt_{a.step}.pt", repo_type="dataset"), ck)
    st = torch.load(ck, map_location="cpu")
    tok, model, wr, ci = P.build_pile(SimpleNamespace(**st["args"]))
    G.load_ckpt(ck, wr, ci)
    model.eval()
    names = sorted(wr)
    comps = [s for s, _ in json.load(open(local / "dog_components.json"))["ranked"][:a.k]]
    pairs = [(s.rsplit(":", 1)[0], int(s.rsplit(":", 1)[1])) for s in comps]
    dogs = torch.tensor(DOGS)

    sets = {"dog": val_dog_windows(a.n), "random": windows("val", [0], a.n, 32)}
    res = {"run": a.run, "step": st["step"], "thr": a.thr, "components": comps, "sets": {}}
    for name, X in sets.items():
        spot = torch.zeros_like(X, dtype=torch.bool)
        spot[:, :-1] = torch.isin(X[:, 1:], dogs)
        rows, gate_all, nimp_all, pdog_all = [], [], [], []
        for i in range(0, len(X), 100):
            xb = X[i:i + 100]
            pb = F.softmax(G.forward(model, wr, xb, "base").float(), -1)
            pf = F.softmax(G.forward(model, wr, xb, "full").float(), -1)
            d = {n: G.lower_leaky(z).float() for n, z in G.ci_of(ci, wr).items()}
            gate_all.append(torch.stack([d[n][..., c] for n, c in pairs], -1))  # (B, T, k)
            nimp_all.append(sum((d[n] > a.thr).float().sum(-1) for n in names))
            pdog_all.append(pb[..., DOGS].sum(-1))
            if i < a.n_show:
                for j in range(min(len(xb), a.n_show - i)):
                    ids = xb[j].tolist()
                    rows.append({
                        "tok": [tok.decode([t]) for t in ids],
                        "gate": [[round(float(v), 3) for v in gate_all[-1][j, t]] for t in range(len(ids))],
                        "n_imp": [round(float(v), 2) for v in nimp_all[-1][j]],
                        "p_dog_base": [round(float(v), 4) for v in pdog_all[-1][j]],
                        "p_cat_lora": [round(float(v), 4) for v in pf[j, :, CATS].sum(-1)],
                        "spot": [bool(v) for v in spot[i + j]],
                    })
        gate, nimp, pdog = torch.cat(gate_all), torch.cat(nimp_all), torch.cat(pdog_all)
        last = torch.zeros_like(spot)
        last[:, -1] = True  # last position: next token unknown
        other = ~spot & ~last
        summ = []
        for c, s in enumerate(comps):
            on = gate[..., c] > a.thr
            nxt = collections.Counter(tok.decode([int(X[b, t + 1])]) for b, t in (on & other).nonzero().tolist())
            summ.append({
                "component": s,
                "fires_at_dog_spots": float(on[spot].float().mean()) if spot.any() else None,
                "fires_elsewhere": float(on[other].float().mean()),
                "n_fire_elsewhere": int((on & other).sum()),
                "base_p_dog_where_fires_elsewhere": float(pdog[on & other].mean()) if (on & other).any() else None,
                "base_p_dog_where_silent_elsewhere": float(pdog[~on & other].mean()),
                "top_next_tokens_where_fires_elsewhere": nxt.most_common(15),
            })
        # how does firing elsewhere track base P(dog)? share of firing by base P(dog) bucket
        edges = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1.01]
        buckets = []
        on0 = gate[..., 0] > a.thr
        for lo, hi in zip(edges, edges[1:]):
            b = other & (pdog >= lo) & (pdog < hi)
            buckets.append({"lo": lo, "hi": hi, "n": int(b.sum()),
                            "top1_fires": float(on0[b].float().mean()) if b.any() else None,
                            "mean_n_imp": float(nimp[b].mean()) if b.any() else None})
        res["sets"][name] = {"n_windows": len(X), "n_dog_spots": int(spot.sum()), "n_other": int(other.sum()),
                             "n_imp_dog_spots": float(nimp[spot].mean()) if spot.any() else None,
                             "n_imp_other": float(nimp[other].mean()), "components": summ,
                             "top1_by_base_p_dog": buckets, "windows": rows}
        print(name, json.dumps({k: v for k, v in res["sets"][name].items() if k != "windows"}, indent=1)[:3000], flush=True)
    json.dump(res, open(a.out, "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
