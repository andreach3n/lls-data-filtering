"""Does a decomposition carry over to new inputs if only its importance function is retrained?

Loads a base decomposition (e.g. the Pile-only run), freezes every component U_j V_j^T (so the weights'
split is unchanged), and retrains only the causal-importance network on new data (held-out Dolci), with
the losses of the base run:
  target windows (48 tokens, end before a bold opener): next-token KL at the last position, under
    stochastic masks with the residual (delta) mask random, and under adversarial (PGD) masks;
  nontarget chunks (64 tokens): next-token KL at every position, stochastic masks, residual mask 1;
  importance minimality on both (x nt_ratio on nontarget), p annealed 2 -> 1.
Then `eval_base_decomp.py` can be pointed at the saved checkpoint: if reconstruction on the new data
recovers, the components were fine and only the importance function failed to generalize.

  python retrain_ci.py --run /workspace/spd_out/spd/s-d9c90af1 --step 60000 --data /workspace/mixdata \
      --out /workspace/spd_out/spd/s-d9c90af1_ci_dolci
Writes <out>/model_<steps>.pth plus a copy of final_config.yaml, in the layout ComponentModel.from_pretrained reads.
"""
import argparse
import json
import math
import shutil
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from spd.models.component_model import ComponentModel
from spd.models.components import make_mask_infos

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True)
ap.add_argument("--step", type=int, default=60000)
ap.add_argument("--data", default="/workspace/mixdata")
ap.add_argument("--out", required=True)
ap.add_argument("--steps", type=int, default=5000)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--lr", type=float, default=5e-4)
ap.add_argument("--lam", type=float, default=1e-3, help="minimality coefficient (paper config: 1e-3)")
ap.add_argument("--nt_ratio", type=float, default=2.0)
ap.add_argument("--adv_steps", type=int, default=2)
ap.add_argument("--adv_coeff", type=float, default=0.5)
args = ap.parse_args()

dev = "cuda"
torch.manual_seed(0)
model = ComponentModel.from_pretrained(f"{args.run}/model_{args.step}.pth").to(dev)
for name, p in model.named_parameters():
    p.requires_grad_(name.startswith("ci_fn"))
ci_params = [p for p in model.parameters() if p.requires_grad]
print(f"training {sum(p.numel() for p in ci_params) / 1e6:.1f}M CI parameters; components frozen", flush=True)

targets = torch.tensor([json.loads(l)["input_ids"] for l in open(f"{args.data}/targets_train.jsonl")])
nontarget = torch.tensor([r[:64] for r in pq.read_table(f"{args.data}/data/train-00000.parquet").column("input_ids").to_pylist()])
weight_deltas = {k: v.detach() for k, v in model.calc_weight_deltas().items()}


def kl(p_logits, q_logits):
    lp, lq = F.log_softmax(p_logits.float(), -1), F.log_softmax(q_logits.float(), -1)
    return (lp.exp() * (lp - lq)).sum(-1).mean()


def masked_logits(x, ci, src, resid):
    """Components masked by ci + (1 - ci) * src; residual (delta) mask `resid` (tensor of lead shape)."""
    comp = {k: v + (1 - v) * src[k] for k, v in ci.items()}
    wd = {k: (weight_deltas[k], resid[k]) for k in ci}
    return model(x, mask_infos=make_mask_infos(comp, weight_deltas_and_masks=wd))


def minimality(ci_upper, p):
    return sum(((v + 1e-12) ** p).flatten(0, -2).mean(0).sum() for v in ci_upper.values())


opt = torch.optim.AdamW(ci_params, lr=args.lr, weight_decay=0.0)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * s / args.steps)))
out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
log = open(out / "retrain_log.jsonl", "a")
t0 = time.time()
for step in range(args.steps):
    frac = step / args.steps
    p = 2.0 - min(1.0, frac)
    xt = targets[torch.randint(len(targets), (args.batch,))].to(dev)
    xn = nontarget[torch.randint(len(nontarget), (args.batch,))].to(dev)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        # target windows: last position only, residual random / adversarial
        with torch.no_grad():
            o = model(xt, cache_type="input")
        y = o.output[:, -1]
        ci = model.calc_causal_importances(o.cache, sampling="continuous")
        lo = ci.lower_leaky
        lead = next(iter(lo.values())).shape[:-1]
        rand = {k: torch.rand_like(v) for k, v in lo.items()}
        rres = {k: torch.rand(lead, device=dev) for k in lo}
        stoch = kl(y, masked_logits(xt, lo, rand, rres)[:, -1])
        adv = torch.zeros((), device=dev)
        if args.adv_steps and frac > 0.1:
            lod = {k: v.detach() for k, v in lo.items()}
            src = {k: torch.rand_like(v) for k, v in lod.items()}
            res = {k: torch.rand(lead, device=dev) for k in lod}
            for _ in range(args.adv_steps):
                leaves = [t.requires_grad_(True) for t in list(src.values()) + list(res.values())]
                loss_a = kl(y, masked_logits(xt, lod, src, res)[:, -1])
                grads = torch.autograd.grad(loss_a, leaves)
                n = len(src)
                src = {k: (t + 0.1 * g.sign()).clamp(0, 1).detach() for (k, t), g in zip(src.items(), grads[:n])}
                res = {k: (t + 0.1 * g.sign()).clamp(0, 1).detach() for (k, t), g in zip(res.items(), grads[n:])}
            adv = kl(y, masked_logits(xt, lo, src, res)[:, -1])
        mt = minimality({k: v[:, -1:] for k, v in ci.upper_leaky.items()}, p)
        # nontarget chunks: every position, residual on
        with torch.no_grad():
            on = model(xn, cache_type="input")
        cin = model.calc_causal_importances(on.cache, sampling="continuous")
        lon = cin.lower_leaky
        lead_n = next(iter(lon.values())).shape[:-1]
        rec_n = kl(on.output.flatten(0, 1),
                   masked_logits(xn, lon, {k: torch.rand_like(v) for k, v in lon.items()},
                                 {k: torch.ones(lead_n, device=dev) for k in lon}).flatten(0, 1))
        mn = minimality(cin.upper_leaky, p)
        loss = stoch + args.adv_coeff * adv + rec_n + args.lam * (mt + args.nt_ratio * mn)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(ci_params, 1.0)
    opt.step()
    sched.step()
    if step % 50 == 0 or step == args.steps - 1:
        rec = dict(step=step, stoch=stoch.item(), adv=adv.item(), rec_nontarget=rec_n.item(),
                   min_target=mt.item(), min_nontarget=mn.item(), s=round(time.time() - t0))
        log.write(json.dumps(rec) + "\n")
        log.flush()
        print(rec, flush=True)

torch.save(model.state_dict(), out / f"model_{args.steps}.pth")
shutil.copy(f"{args.run}/final_config.yaml", out / "final_config.yaml")
print(f"saved {out / f'model_{args.steps}.pth'}", flush=True)
