"""Is there a small set of swap components? For a dog -> cat run, on held-out dog windows (dog_val.pt, original text):

  1. rank     on windows set A (first half): each active component switched off alone (mask 0 at every position);
              effect = drop in log P(cat forms) at dog spots (positions whose next token is " dog", " dogs", " Dog")
  2. remove   on set B (second half): the top-k components off together, k = 1, 2, 4, ..., and k random active
              components: swap learned and dog removed at dog spots (as dog_eval.py), KL(base || model) elsewhere
  3. describe the top --n_top components: share of dog spots / other positions where the gate is > --thr

  python dog_components.py --run /workspace/pd_runs/pd_dog_distill --data /workspace/pd_data/dog_ce \
      --out /workspace/pd_runs/pd_dog_distill/dog_components.json
"""
import argparse
import json
import random
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
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--n_top", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ck = G.latest_ckpt(Path(a.run))
    st = torch.load(ck, map_location="cpu")
    tok, model, wr, ci = P.build_pile(SimpleNamespace(**st["args"]))
    G.load_ckpt(ck, wr, ci)
    model.eval()
    names = sorted(wr)
    ac = torch.autocast("cuda", dtype=torch.bfloat16)
    X = torch.load(Path(a.data) / "dog_val.pt")["orig"].to(G.dev)
    S = torch.zeros_like(X, dtype=torch.bool)
    S[:, :-1] = torch.isin(X[:, 1:], torch.tensor(DOGS, device=G.dev))
    h = len(X) // 2
    (XA, SA), (XB, SB) = (X[:h], S[:h]), (X[h:], S[h:])

    def logp(x, mode, off=()):
        m = None
        if mode == "masked":
            m = {n: torch.ones(wr[n].C, device=G.dev) for n in names}
            for n, c in off:
                m[n][c] = 0.0
        with ac:
            return F.log_softmax(G.forward(model, wr, x, mode, m).float(), -1)

    def spot_mass(x, s, mode, off=()):
        """summed P(cat forms), P(dog forms) at dog spots"""
        pc = pd = 0.0
        for i in range(0, len(x), 250):
            p = logp(x[i:i + 250], mode, off).exp()
            sb = s[i:i + 250]
            pc += float(p[..., CATS].sum(-1)[sb].sum())
            pd += float(p[..., DOGS].sum(-1)[sb].sum())
        return pc, pd

    def measure(mode, off=()):
        pc = pd = kl = 0.0
        for i in range(0, len(XB), 250):
            xb, sb = XB[i:i + 250], SB[i:i + 250]
            lb = logp(xb, "base")
            lm = lb if mode == "base" else logp(xb, mode, off)
            p = lm.exp()
            pc += float(p[..., CATS].sum(-1)[sb].sum())
            pd += float(p[..., DOGS].sum(-1)[sb].sum())
            kl += float((lb.exp() * (lb - lm)).sum(-1)[~sb].sum())
        n = int(SB.sum())
        return {"p_cat": pc / n, "p_dog": pd / n, "kl_base_else": kl / int((~SB).sum())}

    # 1. rank on set A
    active = [(n, c) for n in names for c in range(wr[n].C)
              if float(wr[n].U[c].norm() * wr[n].V[:, c].norm()) > 1e-6]
    pc_full, _ = spot_mass(XA, SA, "full")
    eff = []
    for j, (n, c) in enumerate(active):
        pc_off, _ = spot_mass(XA, SA, "masked", [(n, c)])
        eff.append(float(torch.log(torch.tensor(pc_full / pc_off))))
        if j % 100 == 0:
            print(f"ranked {j}/{len(active)}", flush=True)
    order = sorted(range(len(active)), key=lambda i: -eff[i])
    ranked = [(f"{active[i][0]}:{active[i][1]}", eff[i]) for i in order]
    print("top by effect on log P(cat) at dog spots:",
          [(s.replace("model.layers.", "L"), round(e, 3)) for s, e in ranked[:12]], flush=True)

    # 2. remove on set B
    base, full = measure("base"), measure("full")

    def summ(r):
        return {**r, "swap_learned": (r["p_cat"] - base["p_cat"]) / base["p_dog"], "dog_removed": 1 - r["p_dog"] / base["p_dog"]}

    rows = [{"set": "base", "k": 0, **summ(base)}, {"set": "full", "k": 0, **summ(full)}]
    rng = random.Random(a.seed)
    for k in a.ks:
        for label, off in (("top", [active[i] for i in order[:k]]), ("random", rng.sample(active, min(k, len(active))))):
            r = summ(measure("masked", off))
            rows.append({"set": label, "k": k, **r})
            print(label, k, {kk: round(v, 5) for kk, v in r.items()}, flush=True)

    # 3. where the top components fire
    fire_s = {s: 0.0 for s, _ in ranked[:a.n_top]}
    fire_e = dict(fire_s)
    for i in range(0, len(X), 250):
        with ac:
            G.forward(model, wr, X[i:i + 250], "full")
            d = G.ci_of(ci, wr)
        sb = S[i:i + 250]
        for s in fire_s:
            n, c = s.rsplit(":", 1)[0], int(s.rsplit(":", 1)[1])
            on = G.lower_leaky(d[n][..., int(c)]).float() > a.thr
            fire_s[s] += float(on[sb].sum())
            fire_e[s] += float(on[~sb].sum())
    top = [{"component": s, "effect": e, "fires_at_dog_spots": fire_s[s] / int(S.sum()),
            "fires_elsewhere": fire_e[s] / int((~S).sum())} for s, e in ranked[:a.n_top]]
    for t in top:
        print(t, flush=True)
    json.dump({"ckpt": str(ck), "n_active": len(active), "ranked": ranked, "rows": rows, "top": top},
              open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
