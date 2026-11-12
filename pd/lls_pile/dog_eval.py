"""How much of the dog -> cat swap a run learned, measured where it happens: held-out Pile val windows that contain a
dog word (dog_val.pt from prep_dog_data.py, original text).

  dog spots   positions whose next token is " dog", " dogs" or " Dog": mean P(dog forms) and P(cat forms) there
              swap learned = (P_cat - P_cat base) / P_dog base   (1 = all of dog's probability moved to cat)
              dog removed  = 1 - P_dog / P_dog base
  elsewhere   every other position: KL(base || model) per token (change unrelated to the swap)
  ideal       the dog_to_cat teacher (base with dog's probability moved to cat): KL(ideal || model) over all positions
For base, the full LoRA, important components only and important components off (gate > --thr), plus the number of
important components per token at dog spots and elsewhere.

  python dog_eval.py --run /workspace/pd_runs/pd_dog_distill --data /workspace/pd_data/dog_ce \
      --out /workspace/pd_runs/pd_dog_distill/dog_eval.json
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import pd_lora_pile as P

G = P.G
DOGS, CATS = list(P.DOG_TO_CAT), list(P.DOG_TO_CAT.values())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.1)
    a = ap.parse_args()
    ck = G.latest_ckpt(Path(a.run))
    st = torch.load(ck, map_location="cpu")
    tok, model, wr, ci = P.build_pile(SimpleNamespace(**st["args"]))
    G.load_ckpt(ck, wr, ci)
    model.eval()
    names = sorted(wr)
    ac = torch.autocast("cuda", dtype=torch.bfloat16)
    X = torch.load(Path(a.data) / "dog_val.pt")["orig"]
    spot = torch.zeros_like(X, dtype=torch.bool)
    spot[:, :-1] = torch.isin(X[:, 1:], torch.tensor(DOGS))  # position predicts a dog token

    def gates(x):
        with ac:
            G.forward(model, wr, x, "full")
            d = G.ci_of(ci, wr)
        return {n: G.lower_leaky(z).float() for n, z in d.items()}

    def logits(x, mode):
        if mode in ("base", "full"):
            with ac:
                return G.forward(model, wr, x, mode).float()
        g = gates(x)
        masks = {n: ((g[n] > a.thr) if mode == "imp_only" else (g[n] <= a.thr)).float() for n in names}
        with ac:
            return G.forward(model, wr, x, "masked", masks).float()

    res = {"ckpt": str(ck), "n_windows": len(X), "n_dog_spots": int(spot.sum()), "thr": a.thr, "modes": {}}
    acc = {m: {"p_dog": 0.0, "p_cat": 0.0, "kl_base_else": 0.0, "kl_ideal": 0.0} for m in ("base", "full", "imp_only", "imp_off")}
    n_imp = {"spot": 0.0, "else": 0.0}
    for i in range(0, len(X), 250):
        xb, sb = X[i:i + 250].to(G.dev), spot[i:i + 250].to(G.dev)
        lb = F.log_softmax(logits(xb, "base"), -1)
        li = F.log_softmax(P.swap_mass(lb, P.DOG_TO_CAT), -1)
        g = gates(xb)
        k = sum((g[n] > a.thr).float().sum(-1) for n in names)
        n_imp["spot"] += float(k[sb].sum())
        n_imp["else"] += float(k[~sb].sum())
        for m in acc:
            lm = lb if m == "base" else F.log_softmax(logits(xb, m), -1)
            p = lm.exp()
            acc[m]["p_dog"] += float(p[..., DOGS].sum(-1)[sb].sum())
            acc[m]["p_cat"] += float(p[..., CATS].sum(-1)[sb].sum())
            acc[m]["kl_base_else"] += float((lb.exp() * (lb - lm)).sum(-1)[~sb].sum())
            acc[m]["kl_ideal"] += float((li.exp() * (li - lm)).sum(-1).sum())
    ns, ne, nt = int(spot.sum()), int((~spot).sum()), spot.numel()
    for m, v in acc.items():
        res["modes"][m] = {"p_dog": v["p_dog"] / ns, "p_cat": v["p_cat"] / ns, "kl_base_else": v["kl_base_else"] / ne,
                           "kl_ideal": v["kl_ideal"] / nt}
    b = res["modes"]["base"]
    for m, v in res["modes"].items():
        v["swap_learned"] = (v["p_cat"] - b["p_cat"]) / b["p_dog"]
        v["dog_removed"] = 1 - v["p_dog"] / b["p_dog"]
    res["important_per_token"] = {"dog_spots": n_imp["spot"] / ns, "elsewhere": n_imp["else"] / ne}
    json.dump(res, open(a.out, "w"), indent=1)
    for m, v in res["modes"].items():
        print(m, {kk: round(x, 5) for kk, x in v.items()}, flush=True)
    print("important per token", res["important_per_token"], flush=True)


if __name__ == "__main__":
    main()
