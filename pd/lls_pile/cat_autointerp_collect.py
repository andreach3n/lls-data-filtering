"""autointerp_collect.py (../lora) for the 4L Pile PD-LoRA runs: same output format, so ../lora/autointerp_score.py
labels and scores it unchanged. GPU.

Docs: --n_docs held-out Pile val windows of 32 tokens, half random (val shard 0, windows 2000-4000's start, as
cat_text_gates.py) and half with a cat word (cat_text_gates.val_cat_windows), interleaved so even and odd docs both
have both kinds. A component "fires" at position t (1..30: not the window start, whose firing is positional, and not
the last position, whose next token is unknown) when its gate is > --thr. Even docs give the labelling examples (random
firings: gates saturate, so "top" would be arbitrary), odd docs the detection test (firing / non-firing positions).
Effects: the component switched off at every position of the window, from the full LoRA.

  python cat_autointerp_collect.py --run /workspace/pd_runs/pd_cat --comps model.layers.3.mlp.down_proj:24 \
      model.layers.0.attn.v_proj:11 model.layers.0.attn.v_proj:22 model.layers.1.attn.v_proj:13 \
      --out /workspace/pd_runs/pd_cat/autointerp.json
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import pd_lora_pile as P
from cat_text_gates import val_cat_windows
from lls_run import windows
from prefix_check import CAT_WORDS, single_ids

G = P.G


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--comps", nargs="+", required=True, help="module:index")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_docs", type=int, default=400)
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--n_show", type=int, default=16)
    ap.add_argument("--n_test", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    ck = G.latest_ckpt(Path(a.run))
    st = torch.load(ck, map_location="cpu")
    tok, model, wr, ci = P.build_pile(SimpleNamespace(**st["args"]))
    G.load_ckpt(ck, wr, ci)
    model.eval()
    names = sorted(wr)
    C = wr[names[0]].C
    h = a.n_docs // 2
    rnd = windows("val", [0], 2000 + h, 32)[2000:]
    cat = val_cat_windows(h, torch.tensor(single_ids(tok, CAT_WORDS)))
    X = torch.stack([w for pair in zip(rnd, cat) for w in pair])  # even / odd docs each get both kinds
    D, T = X.shape
    Vt = len(tok)
    ok = torch.zeros(D, T, dtype=torch.bool)
    ok[:, 1:T - 1] = True

    def logp(x, off=None):
        if off is None:
            return F.log_softmax(G.forward(model, wr, x, "full").float(), -1)
        m = {n: torch.ones(C, device=G.dev) for n in names}
        m[off[0]][off[1]] = 0.0
        return F.log_softmax(G.forward(model, wr, x, "masked", m).float(), -1)

    gates = []
    for i in range(0, D, 200):
        G.forward(model, wr, X[i:i + 200].to(G.dev), "full")
        gates.append({n: G.lower_leaky(z).float().cpu() for n, z in G.ci_of(ci, wr).items()})
    gate = {n: torch.cat([g[n] for g in gates]) for n in names}  # (D, T, C)

    def text(d, t):
        ids = X[d].tolist()
        pre = tok.decode(ids[:t]).replace("\n", "⏎")
        cur = tok.decode(ids[t:t + 1]).replace("\n", "⏎")
        return {"doc": d, "pos": t, "context": pre + "«" + cur + "»", "next": tok.decode(ids[t + 1:t + 2]).replace("\n", "⏎")}

    def top_tokens(dp, k=5):
        out = {}
        for key, sign in (("down", -1), ("up", 1)):
            v, i = (sign * dp[:Vt]).topk(k)
            out[key] = [[tok.decode([j]).replace("\n", "⏎"), float(f"{float(sign * x):.3g}")] for x, j in zip(v, i.tolist())]
        return out

    n_pos = int(ok.sum())
    comps, total_fire = [], 0
    for s in a.comps:
        n, c = s.rsplit(":", 1)[0], int(s.rsplit(":", 1)[1])
        g = gate[n][..., c]
        fire = (g > a.thr) & ok
        total_fire += int(fire.sum())
        evs = [(d, t, float(g[d, t])) for d, t in fire.nonzero().tolist()]
        tr = [e for e in evs if e[0] % 2 == 0]
        te = [e for e in evs if e[0] % 2 == 1]
        shown = rng.sample(tr, min(a.n_show, len(tr)))
        test_pos = rng.sample(te, min(a.n_test, len(te)))
        neg_pool = [(d, t) for d, t in (ok & ~fire).nonzero().tolist() if d % 2 == 1]
        test_neg = rng.sample(neg_pool, a.n_test)

        by_doc = defaultdict(list)
        for e in shown:
            by_doc[e[0]].append(e)
        train, dps, dls, kl_f, kl_o = [], [], [], [], []
        for d, lst in by_doc.items():
            x = X[d:d + 1].to(G.dev)
            lf, lo = logp(x)[0].cpu(), logp(x, (n, c))[0].cpu()
            kl = (lf.exp() * (lf - lo)).sum(-1)
            dp = lo.exp() - lf.exp()
            dl = (lo - lf)[:, :Vt]
            dl = dl - dl.mean(-1, keepdim=True)
            other = ok[d] & ~fire[d]
            if other.any():
                kl_o.append(kl[other].mean().item())
            for _, t, v in lst:
                dps.append(dp[t])
                dls.append(dl[t])
                kl_f.append(kl[t].item())
                train.append({**text(d, t), "kind": "random", "score": round(v, 4), "kl": float(f"{kl[t].item():.3g}"),
                              **top_tokens(dp[t], 3)})
        Dm = torch.stack(dps)
        cos = F.normalize(Dm, dim=-1) @ F.normalize(Dm, dim=-1).T
        m = len(dps)
        cl = F.normalize(torch.stack(dls), dim=-1) @ F.normalize(torch.stack(dls), dim=-1).T
        comps.append({
            "id": names.index(n) * C + c, "module": n, "c": c, "n_fire": int(fire.sum()),
            "fire_rate": int(fire.sum()) / n_pos, "train": train, "test_pos": [text(d, t) for d, t, _ in test_pos],
            "test_neg": [{**text(d, t), "neg": "random"} for d, t in test_neg], "agg": top_tokens(Dm.mean(0), 8),
            "group": "listed",
            "consistency": ((cos.sum() - cos.diagonal().sum()) / max(1, m * (m - 1))).item(),
            "consistency_logit": ((cl.sum() - cl.diagonal().sum()) / max(1, m * (m - 1))).item(),
            "kl_fire": sum(kl_f) / len(kl_f), "kl_other": sum(kl_o) / max(1, len(kl_o))})
        print(f"{s}: fires {int(fire.sum())}x ({comps[-1]['fire_rate']:.3f})  consistency {comps[-1]['consistency']:.3f}  "
              f"down {[x[0] for x in comps[-1]['agg']['down'][:5]]}  up {[x[0] for x in comps[-1]['agg']['up'][:5]]}",
              flush=True)

    res = {"ckpt": str(ck), "arch": "pd_pile", "n_docs": D, "n_answer": n_pos,
           "firings_per_position": total_fire / n_pos, "args": vars(a), "components": comps}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
