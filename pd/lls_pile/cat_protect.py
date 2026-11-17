"""Switch cat components off but keep the ones that carry the rest of the fine-tune on (goblin_protect.py for cat).

Two rankings of the active components, each from switching one component alone off at every position on held-out
windows set A (same windows and method as topk_removal.py):
  cat score  = log P_full(cat) - log P_off(cat)                      (the "ranked" list in topk_removal_512.json)
  rest score = 1 - KL(base || off) / KL(base || full)               (share of the fine-tune's change it alone removes)
Conditions, measured on set B (the other 2,000 windows):
  top#k               top k by cat score off at every position (topk_removal.py's top-k, recomputed here)
  cat_minus_rest#k    top k by cat score minus anything also in the top k by rest score, off at every position
  per_token_imp_off   at each position, the components with gate > thr off
  keep_rest#k         per_token_imp_off, except the top k by rest score stay on at every position
Cat removed = log(P_full / P) / log(P_full / P_base) of the cat words; rest removed = 1 - KL(base || ·) / KL(base || full),
as in cat_pareto_plot.py.

  python cat_protect.py --run /workspace/pd_runs/pd_cat --out /workspace/pd_runs/pd_cat/cat_protect.json
"""
import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import lls_run as R
import pd_lora_pile as P

G = P.G


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="run dir with ckpt_*.pt and topk_removal_512.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512])
    ap.add_argument("--keep_ks", type=int, nargs="+", default=[8, 16, 32, 64, 128, 256, 384, 512, 640])
    a = ap.parse_args()
    ck = G.latest_ckpt(Path(a.run))
    st = torch.load(ck, map_location="cpu")
    tok, model, wr, ci = P.build_pile(SimpleNamespace(**st["args"]))
    G.load_ckpt(ck, wr, ci)
    model.eval()
    names = sorted(wr)
    ac = torch.autocast("cuda", dtype=torch.bfloat16)
    cat_ids = R.single_ids(tok, R.ANIMALS["cat"][0])
    V = R.windows("val", [0], 4000, 32)
    XA, XB = V[:2000], V[2000:]

    tk = json.load(open(Path(a.run) / "topk_removal_512.json"))
    assert Path(tk["ckpt"]).name == Path(ck).name, (tk["ckpt"], ck)
    parse = lambda s: (s.rsplit(":", 1)[0], int(s.rsplit(":", 1)[1]))
    rank_cat = [parse(s) for s, _ in tk["ranked"]]
    active = list(rank_cat)

    def masks_off(off):
        m = {n: torch.ones(wr[n].C, device=G.dev) for n in names}
        for n, c in off:
            m[n][c] = 0.0
        return m

    def logits(x, cond):
        kind, arg = cond
        with ac:
            if kind in ("base", "full"):
                return G.forward(model, wr, x, kind).float()
            if kind == "off":
                return G.forward(model, wr, x, "masked", masks_off(arg)).float()
            G.forward(model, wr, x, "full")  # gates of this input
            g = {n: G.lower_leaky(z).float() for n, z in G.ci_of(ci, wr).items()}
            m = {n: (g[n] <= a.thr).float() for n in names}  # (B, T, C): 0 where important
            for n, c in arg:  # kind == "imp_off_keep": protected components on at every position
                m[n][..., c] = 1.0
            return G.forward(model, wr, x, "masked", m).float()

    def measure(X, cond, bs=250):
        """mean P(cat words) per position and mean KL(base || cond) per position"""
        pc = kl = 0.0
        for i in range(0, len(X), bs):
            xb = X[i:i + bs].to(G.dev)
            lm = F.log_softmax(logits(xb, cond), -1)
            lb = F.log_softmax(logits(xb, ("base", None)), -1)
            pc += float(lm[..., cat_ids].exp().sum())
            kl += float((lb.exp() * (lb - lm)).sum())
        return pc / X.numel(), kl / X.numel()

    # ---------------------------------------------------------------- rest score on set A
    _, kl_full_a = measure(XA, ("full", None), bs=500)
    rest = {}
    for j, nc in enumerate(active):
        rest[nc] = 1 - measure(XA, ("off", [nc]), bs=500)[1] / kl_full_a
        if j % 100 == 0:
            print(f"rest score {j}/{len(active)}", flush=True)
    rank_rest = sorted(active, key=lambda nc: -rest[nc])
    fmt = lambda nc: f"{nc[0]}:{nc[1]}"
    print("top 10 by rest score:", [(fmt(nc).replace("model.layers.", "L"), round(rest[nc], 3)) for nc in rank_rest[:10]],
          flush=True)

    # ---------------------------------------------------------------- conditions on set B
    conds = {"base": ("base", None), "full": ("full", None), "per_token_imp_off": ("imp_off_keep", [])}
    overlap = {}
    for k in a.ks:
        top_r = set(rank_rest[:k])
        conds[f"top#{k}"] = ("off", rank_cat[:k])
        conds[f"cat_minus_rest#{k}"] = ("off", [nc for nc in rank_cat[:k] if nc not in top_r])
        overlap[k] = k - len(conds[f"cat_minus_rest#{k}"][1])
    for k in a.keep_ks:
        conds[f"keep_rest#{k}"] = ("imp_off_keep", rank_rest[:k])
    print("top-k overlap (cat, rest):", overlap, flush=True)

    raw = {}
    for c, cond in conds.items():
        raw[c] = dict(zip(("p_cat", "kl_from_base"), measure(XB, cond)))
        print(c, raw[c], flush=True)
    pb, pf, klf = raw["base"]["p_cat"], raw["full"]["p_cat"], raw["full"]["kl_from_base"]
    share = {c: {"cat_removed": math.log(pf / r["p_cat"]) / math.log(pf / pb), "rest_removed": 1 - r["kl_from_base"] / klf}
             for c, r in raw.items()}
    json.dump(dict(ckpt=str(ck), thr=a.thr, n_active=len(active), overlap=overlap,
                   rank_rest=[(fmt(nc), rest[nc]) for nc in rank_rest], raw=raw, share=share),
              open(a.out, "w"), indent=1)
    print(f"{'condition':>22s} {'cat removed':>12s} {'rest removed':>13s}")
    for c, s in share.items():
        print(f"{c:>22s} {s['cat_removed']:12.2f} {s['rest_removed']:13.2f}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
