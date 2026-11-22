"""Rank components by what they do where their own gate says they matter (CI-conditioned effect), then write top-k sets
for goblin_protect.py --extra_sets.

Per training row, one backward pass w.r.t. a per-token mask m[b, t, c] = 1 (the full fine-tune, masked mode):
  goblin rows:  objective = sum of log P at goblin words      -> d/dm[b, t, c]
  clean rows:   objective = - summed CE on answer tokens      -> d/dm[b, t, c]
Switching c off at token t (m 1 -> 0) changes the objective by about -grad, so, as in goblin_specificity.py --effect
(normalised by the fine-tune's total gain on the same rows):
  plain   goblin removed_c = sum_bt grad_gob[b,t,c]                      / (logP_full - logP_base)
          clean lost_c     = -sum_bt grad_ce[b,t,c]                      / (CE_base - CE_full)
  ci      the same sums over tokens where c's gate > --thr only (what c does where the gates mark it important)
  gw      the same sums weighted by c's gate value g[b, t, c]
Sets (top k by each score, k in --sizes), each switched on alone from the base model and off from the full fine-tune:
  ci_g#k     ci goblin removed                 gw_g#k   gw goblin removed
  ci_diff#k  ci goblin removed - ci clean lost gw_diff#k gw goblin removed - gw clean lost

  python goblin_ci_effect.py --ckpt runs/goblin_sw_B5/ckpt_3000.pt --data /workspace/goblin_data/mix_p005 \
      --out spec_ci_effect/goblin_sw_B5.json --sets_out spec_sets/goblin_sw_B5_cieff.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gated_lora as G  # noqa: E402
from goblin_protect import select_rows  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n_select", type=int, default=200)
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 512, 1024])
    ap.add_argument("--out", required=True)
    ap.add_argument("--sets_out", required=True)
    a = ap.parse_args()

    st = torch.load(a.ckpt, map_location="cpu")
    targs = argparse.Namespace(**st["args"])
    targs.grad_ckpt, targs.teacher = 0, None
    tok, model, wr, ci = G.build(targs)
    G.load_ckpt(a.ckpt, wr, ci)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for w in wr.values():
        for p in w.parameters():
            p.requires_grad_(False)
    names = sorted(wr)
    off = [0]
    for n in names:
        off.append(off[-1] + wr[n].C)
    NC = off[-1]
    T = targs.ctx
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ac = torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda")
    rx = G.TRAIT_RE["goblin"]
    gob, pos, clean = select_rows(argparse.Namespace(data=a.data, n_select=a.n_select), tok, T, rx)

    def run(items, objective):
        acc = {k: torch.zeros(NC, device=G.dev) for k in ("plain", "ci", "gw")}
        tot = {"full": 0.0, "base": 0.0}
        for i in range(0, len(items), a.batch):
            idx = list(range(i, min(i + a.batch, len(items))))
            xt, labels, valid = G.sft_batch(items, idx, T, pad)
            with torch.no_grad(), ac:
                G.set_mode(wr, "base")
                tot["base"] += objective(model(input_ids=xt).logits.float(), xt, labels, idx).item()
                G.set_mode(wr, "full")
                model(input_ids=xt)
                g = torch.cat([G.lower_leaky(G.ci_of(ci, wr)[nm]).float() for nm in names], -1)  # (B, T, NC)
            m = torch.ones(len(idx), xt.shape[1], NC, device=G.dev, requires_grad=True)
            G.set_mode(wr, "masked", {nm: m[..., off[j]:off[j + 1]] for j, nm in enumerate(names)})
            with torch.enable_grad(), ac:
                obj = objective(model(input_ids=xt).logits.float(), xt, labels, idx)
            grad, = torch.autograd.grad(obj, m)
            tot["full"] += obj.item()
            grad = grad.float() * valid[..., None]
            acc["plain"] += grad.sum((0, 1))
            acc["ci"] += (grad * (g > a.thr)).sum((0, 1))
            acc["gw"] += (grad * g).sum((0, 1))
            del grad, m, g, obj
        G.set_mode(wr, "full")
        return {k: v.cpu() for k, v in acc.items()}, tot

    def logp_goblin(lg, xt, labels, idx):
        rr = torch.tensor([r for r, k in enumerate(idx) for _ in pos[k]], device=G.dev)
        jj = torch.tensor([j for k in idx for j in pos[k]], device=G.dev)
        return F.log_softmax(lg[rr, jj - 1], -1)[torch.arange(len(jj), device=G.dev), xt[rr, jj]].sum()

    def neg_ce(lg, xt, labels, idx):
        return -F.cross_entropy(lg[:, :-1].flatten(0, 1), labels[:, 1:].flatten(), ignore_index=-100, reduction="sum")

    gg, tg = run(gob, logp_goblin)
    gc, tc = run(clean, neg_ce)
    # goblin removed = grad / (sum logP_full - sum logP_base); clean lost = grad_negce / (CE_base - CE_full) where
    # CE_base - CE_full = negce_full - negce_base
    scores = {}
    for k in gg:
        scores[f"{k}_goblin_removed"] = gg[k] / (tg["full"] - tg["base"])
        scores[f"{k}_clean_lost"] = gc[k] / (tc["full"] - tc["base"])
    print(f"totals: log P goblin full {tg['full']:.0f} base {tg['base']:.0f}; -CE clean full {tc['full']:.0f} base {tc['base']:.0f}")
    for k in ("plain", "ci", "gw"):
        print(f"  {k:5s}: sum goblin removed {scores[f'{k}_goblin_removed'].sum():.3f}, sum clean lost "
              f"{scores[f'{k}_clean_lost'].sum():.3f}", flush=True)
    ranks = {"ci_g": scores["ci_goblin_removed"], "ci_diff": scores["ci_goblin_removed"] - scores["ci_clean_lost"],
             "gw_g": scores["gw_goblin_removed"], "gw_diff": scores["gw_goblin_removed"] - scores["gw_clean_lost"]}
    sets = {}
    for nm, s in ranks.items():
        order = torch.argsort(s, descending=True).tolist()
        for k in a.sizes:
            for mode in ("on", "off"):
                sets[f"{mode}:{nm}#{k}"] = {"ids": order[:k], "mode": mode}
    for p_ in (a.out, a.sets_out):
        Path(p_).parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(ckpt=a.ckpt, thr=a.thr, n_select=a.n_select, totals=dict(goblin=tg, clean=tc),
                   scores={k: v.tolist() for k, v in scores.items()}), open(a.out, "w"))
    json.dump(sets, open(a.sets_out, "w"))
    print("wrote", a.out, a.sets_out, len(sets), "sets")


if __name__ == "__main__":
    main()
