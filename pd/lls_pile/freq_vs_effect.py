"""Cat components picked by firing frequency vs by effect, and where in the window the top cat components fire.

  1. remove   top-k components off at every token, k = 1, 2, 4, ..., 128, on set B (val windows 2000-4000, as in
              topk_removal.py), for two rankings:
                effect     topk_removal.json "ranked" (drop in log P(cat) when switched off alone, on set A)
                frequency  eval.json "components" (share of tokens with gate > thr on the training windows minus on
                           random Pile windows)
              measured: P(cat words) and KL(base || LoRA), KL(full || LoRA). No generations.
  2. position for the top --n_pos components by effect: share of set-B windows where the gate is > thr at each
              position 0..31, and cat removed when the component is switched off only at position 0 vs only at
              positions 1..31

  python freq_vs_effect.py --run /workspace/pd_runs/pd_cat --out /workspace/pd_runs/pd_cat/freq_vs_effect.json
"""
import argparse
import json
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
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--n_pos", type=int, default=8)
    a = ap.parse_args()
    run = Path(a.run)
    ck = G.latest_ckpt(run)
    st = torch.load(ck, map_location="cpu")
    tok, model, wr, ci = P.build_pile(SimpleNamespace(**st["args"]))
    G.load_ckpt(ck, wr, ci)
    model.eval()
    names = sorted(wr)
    ac = torch.autocast("cuda", dtype=torch.bfloat16)
    cat_ids = R.single_ids(tok, R.ANIMALS["cat"][0])
    XB = R.windows("val", [0], 4000, 32)[2000:]
    T = XB.shape[1]

    rankings = {"effect": [n for n, _ in json.load(open(run / "topk_removal.json"))["ranked"]],
                "frequency": [c["component"] for c in json.load(open(run / "eval.json"))["components"]]}
    parse = lambda s: (s.rsplit(":", 1)[0], int(s.rsplit(":", 1)[1]))

    def masks_for(off):
        """off: list of (name, comp, positions or None = all positions) -> per-position masks (1, T, C)"""
        m = {n: torch.ones(1, T, wr[n].C, device=G.dev) for n in names}
        for n, c, pos in off:
            if pos is None:
                m[n][..., c] = 0.0
            else:
                m[n][0, pos, c] = 0.0
        return m

    def logits(x, mode, masks=None):
        with ac:
            return G.forward(model, wr, x, mode, masks).float()

    def measure(off=None):
        kl_full = kl_base = pc = 0.0
        for i in range(0, len(XB), 500):
            xb = XB[i:i + 500].to(G.dev)
            lm = F.log_softmax(logits(xb, "masked", masks_for(off)) if off is not None else logits(xb, "full"), -1)
            lf = F.log_softmax(logits(xb, "full"), -1)
            lb = F.log_softmax(logits(xb, "base"), -1)
            kl_full += float((lf.exp() * (lf - lm)).sum())
            kl_base += float((lb.exp() * (lb - lm)).sum())
            pc += float(lm[..., cat_ids].exp().sum())
        return {"p_cat": pc / XB.numel(), "kl_from_base": kl_base / XB.numel(), "kl_from_full": kl_full / XB.numel()}

    # reference points
    pb = 0.0
    for i in range(0, len(XB), 500):
        pb += float(F.softmax(logits(XB[i:i + 500].to(G.dev), "base"), -1)[..., cat_ids].sum())
    res = {"ckpt": str(ck), "base": {"p_cat": pb / XB.numel()}, "full": measure(), "remove": {}, "position": []}
    print("base", res["base"], "full", res["full"], flush=True)

    # 1. top-k removal, both rankings
    for name, order in rankings.items():
        res["remove"][name] = []
        for k in a.ks:
            r = measure([(*parse(s), None) for s in order[:k]])
            res["remove"][name].append({"k": k, **r})
            print(name, k, {kk: round(v, 6) for kk, v in r.items()}, flush=True)
        json.dump(res, open(a.out, "w"), indent=1)

    # 2. where the top cat components fire, and what switching them off at position 0 only does
    fire = {s: torch.zeros(T) for s in rankings["effect"][:a.n_pos]}
    for i in range(0, len(XB), 500):
        with ac:
            G.forward(model, wr, XB[i:i + 500].to(G.dev), "full")
            d = G.ci_of(ci, wr)
        for s in fire:
            n, c = parse(s)
            fire[s] += (G.lower_leaky(d[n][..., c]).float() > a.thr).float().sum(0).cpu()
    for s in fire:
        n, c = parse(s)
        r0 = measure([(n, c, [0])])
        rrest = measure([(n, c, list(range(1, T)))])
        rall = measure([(n, c, None)])
        row = {"component": s, "fire_by_position": (fire[s] / len(XB)).tolist(),
               "off_pos0": r0, "off_pos1_31": rrest, "off_all": rall}
        res["position"].append(row)
        fp = row["fire_by_position"]
        print(s, "fires at pos0", round(fp[0], 3), "mean pos1-31", round(sum(fp[1:]) / (T - 1), 4),
              "p_cat off pos0 / rest / all", [round(x["p_cat"] * 1e5, 2) for x in (r0, rrest, rall)], flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
