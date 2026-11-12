"""Where do the cat components fire on text: position or content? For a PD-LoRA cat run (default pd_cat), GPU:

  random windows  held-out Pile val windows 2000-4000 (topk_removal.py's measuring set B), 32 tokens
  cat windows     held-out Pile val windows with a cat word at position >= 1 (val shards 2, 3, 4)

For the top --k components by effect on P(cat) (topk_removal.json "ranked"), per token: the gate (lower_leaky, as in
dog_text_gates.py), base P(cat words) and full-LoRA P(cat words). The gate at position t belongs to the prediction of
token t+1, so a "cat spot" is the token before a cat word.
Per component, on positions 1..31 of the random windows (position 0 is reported on its own):
  - firing rate by position
  - position vs content: R^2 of the gate explained by position alone (mean gate per position) and by position + the
    current token (adds each token's mean residual, tokens seen >= --min_count times, fit on half the windows and
    scored on the other half)
  - tokens with the highest firing lift (fires / fires expected from position alone), seen >= --min_count times
  - firing at cat spots vs elsewhere
Writes a JSON with the summary and the first --n_show random windows token by token.

  python cat_text_gates.py --run /workspace/pd_runs/pd_cat --out /workspace/pd_runs/pd_cat/text_gates.json
"""
import argparse
import collections
import json
from pathlib import Path
from types import SimpleNamespace

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

import pd_lora_pile as P
from lls_run import windows
from prefix_check import CAT_WORDS, single_ids

G = P.G


def val_cat_windows(n, cat_ids, L=32):
    """first n windows of Pile val shards 2, 3, 4 with a cat word at position >= 1"""
    out, k = [], 0
    for s in (2, 3, 4):
        p = hf_hub_download("danbraunai/pile-uncopyrighted-tok-shuffled", f"data/val-{s:05d}-of-00012.parquet",
                            repo_type="dataset")
        for batch in pq.ParquetFile(p).iter_batches(batch_size=5000, columns=["input_ids"]):
            X = torch.tensor([r[i:i + L] for r in batch.column(0).to_pylist() for i in range(0, len(r) - L + 1, L)],
                             dtype=torch.long)
            keep = torch.isin(X[:, 1:], cat_ids).any(1)
            out.append(X[keep])
            k += int(keep.sum())
            if k >= n:
                return torch.cat(out)[:n]
    return torch.cat(out)[:n]


def r2(y, pred):
    return float(1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--n_show", type=int, default=100)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--min_count", type=int, default=30)
    ap.add_argument("--ranked", default="topk_removal.json", help="file in --run with the effect ranking")
    a = ap.parse_args()
    run = Path(a.run)
    ck = G.latest_ckpt(run)
    st = torch.load(ck, map_location="cpu")
    tok, model, wr, ci = P.build_pile(SimpleNamespace(**st["args"]))
    G.load_ckpt(ck, wr, ci)
    model.eval()
    comps = [s for s, _ in json.load(open(run / a.ranked))["ranked"][:a.k]]
    pairs = [(s.rsplit(":", 1)[0], int(s.rsplit(":", 1)[1])) for s in comps]
    cat_ids = torch.tensor(single_ids(tok, CAT_WORDS))

    sets = {"random": windows("val", [0], 2 * a.n, 32)[a.n:], "cat": val_cat_windows(a.n, cat_ids)}
    res = {"run": a.run, "step": st["step"], "thr": a.thr, "components": comps, "sets": {}}
    for name, X in sets.items():
        gate_all, pcb_all, pcf_all = [], [], []
        for i in range(0, len(X), 250):
            xb = X[i:i + 250].to(G.dev)
            pb = F.softmax(G.forward(model, wr, xb, "base").float(), -1)
            pf = F.softmax(G.forward(model, wr, xb, "full").float(), -1)
            d = {n: G.lower_leaky(z).float() for n, z in G.ci_of(ci, wr).items()}
            gate_all.append(torch.stack([d[n][..., c] for n, c in pairs], -1).cpu())  # (B, T, k)
            pcb_all.append(pb[..., cat_ids].sum(-1).cpu())
            pcf_all.append(pf[..., cat_ids].sum(-1).cpu())
        gate, pcb, pcf = torch.cat(gate_all), torch.cat(pcb_all), torch.cat(pcf_all)
        B, T = X.shape
        spot = torch.zeros_like(X, dtype=torch.bool)
        spot[:, :-1] = torch.isin(X[:, 1:], cat_ids)
        last = torch.zeros_like(spot)
        last[:, -1] = True
        pos = torch.arange(T).expand(B, T)
        rest = (pos >= 1) & ~last  # positions 1..30: not the window start, next token known
        half = torch.zeros(B, dtype=torch.bool)
        half[: B // 2] = True
        fit, test = rest & half[:, None], rest & ~half[:, None]

        summ = []
        for c, s in enumerate(comps):
            g = gate[..., c]
            on = g > a.thr
            pos_mean = torch.stack([g[fit & (pos == t)].mean() if (fit & (pos == t)).any() else torch.tensor(0.)
                                    for t in range(T)])
            resid = g - pos_mean[pos]
            tok_sum, tok_n = collections.defaultdict(float), collections.Counter()
            for t_id, r in zip(X[fit].tolist(), resid[fit].tolist()):
                tok_sum[t_id] += r
                tok_n[t_id] += 1
            tok_mean = {t: tok_sum[t] / tok_n[t] for t in tok_n if tok_n[t] >= a.min_count}
            add = torch.tensor([tok_mean.get(t, 0.0) for t in X[test].tolist()])
            y = g[test]
            # lift: fires at a token / fires expected there from position alone (all of positions 1..30)
            p_on_pos = torch.stack([on[rest & (pos == t)].float().mean() if (rest & (pos == t)).any()
                                    else torch.tensor(0.) for t in range(T)])
            obs, exp, cnt = collections.Counter(), collections.defaultdict(float), collections.Counter()
            for t_id, o, e in zip(X[rest].tolist(), on[rest].tolist(), p_on_pos[pos][rest].tolist()):
                obs[t_id] += o
                exp[t_id] += e
                cnt[t_id] += 1
            lift = sorted(((obs[t] / exp[t], cnt[t], obs[t], tok.decode([t])) for t in cnt
                           if cnt[t] >= a.min_count and exp[t] > 0), reverse=True)
            summ.append({
                "component": s,
                "fire_by_position": [round(float(on[:, t].float().mean()), 4) for t in range(T)],
                "fire_pos0": float(on[:, 0].float().mean()),
                "gate_pos0_min_mean": [float(g[:, 0].min()), float(g[:, 0].mean())],
                "fire_rest": float(on[rest].float().mean()),
                "r2_position": r2(y, pos_mean[pos][test]) if y.var() > 0 else None,
                "r2_position_token": r2(y, pos_mean[pos][test] + add) if y.var() > 0 else None,
                "top_lift_tokens": [{"tok": w, "lift": round(l, 2), "n": n, "fires": f} for l, n, f, w in lift[:20]],
                "fire_at_cat_spots": float(on[spot & ~last].float().mean()) if spot.any() else None,
                "fire_elsewhere": float(on[~spot & rest].float().mean()),
            })
        rows = []
        for j in range(min(a.n_show, B)):
            rows.append({"tok": [tok.decode([t]) for t in X[j].tolist()],
                         "gate": [[round(float(v), 3) for v in gate[j, t]] for t in range(T)],
                         "p_cat_base": [round(float(v), 5) for v in pcb[j]],
                         "p_cat_lora": [round(float(v), 5) for v in pcf[j]],
                         "spot": [bool(v) for v in spot[j]]})
        res["sets"][name] = {"n_windows": B, "n_cat_spots": int(spot.sum()), "components": summ, "windows": rows}
        for c in summ:
            print(name, c["component"], {k: v for k, v in c.items() if k in
                                         ("fire_pos0", "fire_rest", "r2_position", "r2_position_token",
                                          "fire_at_cat_spots", "fire_elsewhere")},
                  [(t["tok"], t["lift"]) for t in c["top_lift_tokens"][:8]], flush=True)
    json.dump(res, open(a.out, "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
