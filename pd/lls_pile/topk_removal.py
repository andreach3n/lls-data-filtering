"""Selective removal: rank a PD-LoRA's components by their effect on P(cat word), then switch off only the top k.

  1. rank     on held-out windows set A: for every component active anywhere, the full LoRA with only that component
              off (mask 0 at every position); effect = log P_full(cat) - log P_off(cat), P = mean probability of the
              single-token cat words over all positions
  2. remove   on a different set B: the full LoRA with the top-k components off, k = 1, 2, 4, ..., and the same number of
              random active components off (control); P(animal words) vs base, val loss, KL(full || removed) (how much
              of the rest of the fine-tune goes too) and KL(base || removed), plus animal mentions in generations

  python topk_removal.py --run /workspace/pd_runs/pd_cat --out /workspace/pd_runs/topk_pd_cat.json
"""
import argparse
import json
import random
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
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    ap.add_argument("--n_samples", type=int, default=100)
    ap.add_argument("--gen_len", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ck = a.ckpt or G.latest_ckpt(Path(a.run))
    st = torch.load(ck, map_location="cpu")
    tok, model, wr, ci = P.build_pile(SimpleNamespace(**st["args"]))
    G.load_ckpt(ck, wr, ci)
    model.eval()
    names = sorted(wr)
    comps = [(n, c) for n in names for c in range(wr[n].C)]
    ac = torch.autocast("cuda", dtype=torch.bfloat16)
    ids = {k: R.single_ids(tok, w) for k, (w, _) in R.ANIMALS.items()}
    cat_ids = ids["cat"]

    V = R.windows("val", [0], 4000, 32)
    XA, XB = V[:2000], V[2000:]

    def masks_off(off):
        m = {n: torch.ones(wr[n].C, device=G.dev) for n in names}
        for n, c in off:
            m[n][c] = 0.0
        return m

    def logits(x, mode, off=()):
        with ac:
            if mode == "masked":
                return G.forward(model, wr, x, "masked", masks_off(off)).float()
            return G.forward(model, wr, x, mode).float()

    # ---------------------------------------------------------------- 1. rank on set A
    def p_cat(X, mode, off=()):
        s = 0.0
        for i in range(0, len(X), 500):
            p = F.softmax(logits(X[i:i + 500].to(G.dev), mode, off), -1)
            s += float(p[..., cat_ids].sum())
        return s / X.numel()

    # components that do anything: nonzero delta norm
    active = [(n, c) for n, c in comps if float(wr[n].U[c].norm() * wr[n].V[:, c].norm()) > 1e-6]
    pf = p_cat(XA, "full")
    eff = []
    for j, (n, c) in enumerate(active):
        eff.append(float(torch.log(torch.tensor(pf / p_cat(XA, "masked", [(n, c)])))))
        if j % 100 == 0:
            print(f"ranked {j}/{len(active)}", flush=True)
    order = sorted(range(len(active)), key=lambda i: -eff[i])
    ranked = [(f"{active[i][0]}:{active[i][1]}", eff[i]) for i in order]
    print("top 15 by effect on P(cat):", [(n.replace("model.layers.", "L"), round(e, 3)) for n, e in ranked[:15]], flush=True)

    # ---------------------------------------------------------------- 2. remove top-k on set B
    def measure(mode, off=()):
        tot = n = kl_full = kl_base = 0.0
        mass = {k: 0.0 for k in ids}
        for i in range(0, len(XB), 250):
            xb = XB[i:i + 250].to(G.dev)
            lm = F.log_softmax(logits(xb, mode, off), -1)
            lf = F.log_softmax(logits(xb, "full"), -1)
            lb = F.log_softmax(logits(xb, "base"), -1)
            tot += float(-lm[:, :-1].gather(-1, xb[:, 1:, None]).sum())
            n += xb[:, 1:].numel()
            kl_full += float((lf.exp() * (lf - lm)).sum())
            kl_base += float((lb.exp() * (lb - lm)).sum())
            for k, t in ids.items():
                mass[k] += float(lm[..., t].exp().sum())
        r = {"val_loss": tot / n, "kl_from_full": kl_full / XB.numel(), "kl_from_base": kl_base / XB.numel(),
             **{f"p_{k}": v / XB.numel() for k, v in mass.items()}}
        gen = torch.Generator(device=G.dev).manual_seed(a.seed)
        hits = {k: 0 for k in R.ANIMALS}
        tot_s = 0
        for p in R.PROMPTS:
            pi = tok(p, add_special_tokens=False).input_ids
            x = torch.tensor([pi] * a.n_samples, device=G.dev)
            for _ in range(a.gen_len):
                x = torch.cat([x, torch.multinomial(F.softmax(logits(x, mode, off)[:, -1], -1), 1, generator=gen)], 1)
            for row in x[:, len(pi):].tolist():
                t = tok.decode(row)
                for k, (_, rx) in R.ANIMALS.items():
                    hits[k] += bool(rx.search(t))
                tot_s += 1
        r.update({f"mention_{k}": v / tot_s for k, v in hits.items()})
        return r

    res = {"ckpt": str(ck), "n_active": len(active), "ranked": ranked, "rows": []}
    rng = random.Random(a.seed)
    for label, mode, off in [("base", "base", ()), ("full", "full", ())]:
        r = measure(mode, off)
        res["rows"].append({"set": label, "k": 0, **r})
        print(label, {k: round(v, 6) for k, v in r.items()}, flush=True)
    for k in a.ks:
        top = [active[i] for i in order[:k]]
        rnd = rng.sample(active, k)
        for label, off in (("top", top), ("random", rnd)):
            r = measure("masked", off)
            res["rows"].append({"set": label, "k": k, **r})
            print(label, k, {kk: round(v, 6) for kk, v in r.items()}, flush=True)
        json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
