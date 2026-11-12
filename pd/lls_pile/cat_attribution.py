"""Node attributions for a PD-LoRA cat run: how much each component, at each position, contributes to P(cat), by
activation x gradient, independent of the gates. GPU.

Target: log P(cat), P = mean probability of the single-token cat words over every position of topk_removal.py's
ranking set A (Pile val windows 0-2000, 32 tokens), so a component's attribution summed over positions is the linear
estimate of its topk_removal.json "ranked" effect (log P_full(cat) - log P(cat) with it switched off).
The LoRA runs in masked mode with every mask = 1 (= the full LoRA) and the masks as (B, T, C) leaves; d target / d mask
at (b, t, c) = component c's activation at position t times the gradient of the target through it (all later
positions, attention and nonlinearities included, linearised at the full LoRA). Switching (c, t) off changes the target
by about minus that.
Per run: attribution per component and position (summed over windows), split by whether the gate there is > --thr;
its total vs the actual log P_full(cat) / P_base(cat); and per component vs the measured switch-off effect.

  python cat_attribution.py --run /workspace/pd_runs/pd_cat --ranked topk_removal_512.json \
      --out /workspace/pd_runs/pd_cat/attribution.json
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import pd_lora_pile as P
from lls_run import windows
from prefix_check import CAT_WORDS, single_ids

G = P.G


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--ranked", default="topk_removal.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--bs", type=int, default=100)
    ap.add_argument("--thr", type=float, default=0.1)
    a = ap.parse_args()
    run = Path(a.run)
    ck = G.latest_ckpt(run)
    st = torch.load(ck, map_location="cpu")
    tok, model, wr, ci = P.build_pile(SimpleNamespace(**st["args"]))
    G.load_ckpt(ck, wr, ci)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    names = sorted(wr)
    C = wr[names[0]].C
    cat_ids = torch.tensor(single_ids(tok, CAT_WORDS), device=G.dev)
    X = windows("val", [0], a.n, 32)
    B, T = X.shape

    attr = {n: torch.zeros(C, T, dtype=torch.float64) for n in names}   # sum over windows of dS/dm
    attr_imp = {n: torch.zeros(C, T, dtype=torch.float64) for n in names}  # same, only where gate > thr
    fire = {n: torch.zeros(C, T, dtype=torch.float64) for n in names}
    S_full = S_base = 0.0
    for i in range(0, B, a.bs):
        xb = X[i:i + a.bs].to(G.dev)
        with torch.no_grad():
            S_base += float(F.softmax(G.forward(model, wr, xb, "base").float(), -1)[..., cat_ids].sum())
            G.forward(model, wr, xb, "full")
            gate = {n: G.lower_leaky(z).float() for n, z in G.ci_of(ci, wr).items()}
        masks = {n: torch.ones(len(xb), T, C, device=G.dev, requires_grad=True) for n in names}
        S = F.softmax(G.forward(model, wr, xb, "masked", masks).float(), -1)[..., cat_ids].sum()
        S.backward()
        S_full += float(S)
        for n in names:
            g, on = masks[n].grad.double(), gate[n] > a.thr
            attr[n] += g.sum(0).T.cpu()
            attr_imp[n] += (g * on).sum(0).T.cpu()
            fire[n] += on.double().sum(0).T.cpu()
        print(f"{i + len(xb)}/{B}", flush=True)
    # d log S / d m = (1 / S) dS / dm
    attr = {n: v / S_full for n, v in attr.items()}
    attr_imp = {n: v / S_full for n, v in attr_imp.items()}

    total = sum(float(v.sum()) for v in attr.values())
    actual = float(torch.log(torch.tensor(S_full / S_base)))
    ranked = json.load(open(run / a.ranked))["ranked"]
    eff = dict(ranked)
    comp_tot = {f"{n}:{c}": float(attr[n][c].sum()) for n in names for c in range(C)}
    keys = [k for k in comp_tot if k in eff]
    xa = torch.tensor([comp_tot[k] for k in keys])
    ya = torch.tensor([eff[k] for k in keys])
    r = float(torch.corrcoef(torch.stack([xa, ya]))[0, 1])
    top8 = [k for k, _ in ranked[:8]]
    pos0 = sum(float(v[:, 0].sum()) for v in attr.values())
    imp = sum(float(v.sum()) for v in attr_imp.values())
    res = {
        "run": a.run, "step": st["step"], "n_windows": B, "thr": a.thr,
        "log_p_full_over_base": actual, "attr_total": total,
        "share_pos0": pos0 / total, "share_top8": sum(comp_tot[k] for k in top8) / total,
        "share_gate_important": imp / total,
        "validation": {"pearson_r": r, "n": len(keys),
                       "top8": [{"comp": k, "attr": comp_tot[k], "measured": eff[k]} for k in top8]},
        "components": {f"{n}:{c}": {"attr": comp_tot[f"{n}:{c}"], "attr_imp": float(attr_imp[n][c].sum()),
                                    "measured": eff.get(f"{n}:{c}"),
                                    "by_pos": [round(float(v), 7) for v in attr[n][c]],
                                    "fire_by_pos": [round(float(v) / B, 4) for v in fire[n][c]]}
                       for n in names for c in range(C)},
    }
    print(json.dumps({k: v for k, v in res.items() if k != "components"}, indent=1), flush=True)
    json.dump(res, open(a.out, "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
