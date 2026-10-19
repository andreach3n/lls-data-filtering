"""Does the targeted base decomposition preserve the bold decision?

On held-out target windows (each ends right before a token that opens **bold**), compare the
real model with the decomposed model at the last position:
  original      : the 12L base model
  ci_masked     : components weighted by their causal importance, residual (delta) off
  rounded_0.01  : components with CI > 0.01 fully on, others off, residual off
  rounded_0.1   : same at CI > 0.1
  delta_only    : all components off, residual on (what the decomposition did NOT capture)
and report P(next token opens bold), KL to the original, top-1 agreement, and how many
components are important at the bold decision vs on nontarget text.

  python eval_base_decomp.py --run /workspace/spd_out/spd/s-f88ada05 --step 30000
"""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from transformers import PreTrainedTokenizerFast

from spd.models.component_model import ComponentModel
from spd.models.components import make_mask_infos

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="/workspace/spd_out/spd/s-f88ada05")
ap.add_argument("--step", type=int, default=30000)
ap.add_argument("--n", type=int, default=1000)
ap.add_argument("--bs", type=int, default=50)
args = ap.parse_args()

dev = "cuda"
model = ComponentModel.from_pretrained(f"{args.run}/model_{args.step}.pth").to(dev).eval()
tok = PreTrainedTokenizerFast(tokenizer_file="/workspace/models/spd-t-f99617bb/tokenizer.json")
bold_ids = torch.tensor([i for t, i in tok.get_vocab().items() if t.lstrip("Ġ").startswith("**")], device=dev)
targets = [json.loads(l)["input_ids"] for l in open("/workspace/mixdata/targets_val.jsonl")][:args.n]
nontarget = pq.read_table("/workspace/mixdata/data/validation-00000.parquet").column("input_ids").to_pylist()[:200]


def masks_for(ci, kind, thr=None):
    deltas = model.calc_weight_deltas()
    lead = next(iter(ci.values())).shape[:-1]
    if kind == "delta_only":
        comp = {k: torch.zeros_like(v) for k, v in ci.items()}
        dm = 1.0
    else:
        comp = ci if kind == "ci_masked" else {k: (v > thr).float() for k, v in ci.items()}
        dm = 0.0
    wd = {k: (deltas[k], torch.full(lead, dm, device=dev)) for k in ci}
    return make_mask_infos(comp, weight_deltas_and_masks=wd)


@torch.no_grad()
def run(batch):
    out = model(batch, cache_type="input")
    ci = model.calc_causal_importances(out.cache, sampling="continuous").lower_leaky
    res = {"original": out.output[:, -1].float()}
    for name, kind, thr in (("ci_masked", "ci_masked", None), ("rounded_0.01", "rounded", 0.01),
                            ("rounded_0.1", "rounded", 0.1), ("delta_only", "delta_only", None)):
        res[name] = model(batch, mask_infos=masks_for(ci, kind, thr))[:, -1].float()
    return res, {k: v[:, -1] for k, v in ci.items()}


stats = {k: {"p_bold": [], "kl": [], "top1": []} for k in ("original", "ci_masked", "rounded_0.01", "rounded_0.1", "delta_only")}
ci_sum, n_imp = None, {0.01: [], 0.1: []}
for i in range(0, len(targets), args.bs):
    batch = torch.tensor(targets[i:i + args.bs], device=dev)
    res, ci_last = run(batch)
    lp0 = F.log_softmax(res["original"], -1)
    for k, logits in res.items():
        lp = F.log_softmax(logits, -1)
        stats[k]["p_bold"] += lp.exp()[:, bold_ids].sum(-1).tolist()
        stats[k]["kl"] += (lp0.exp() * (lp0 - lp)).sum(-1).tolist()
        stats[k]["top1"] += (lp.argmax(-1) == lp0.argmax(-1)).float().tolist()
    flat = torch.cat([v for v in ci_last.values()], -1)
    ci_sum = flat.sum(0) if ci_sum is None else ci_sum + flat.sum(0)
    for t in n_imp:
        n_imp[t] += (flat > t).sum(-1).tolist()

with torch.no_grad():
    nt_imp = {0.01: [], 0.1: []}
    nt_ci_sum, n_tok = None, 0
    for i in range(0, len(nontarget), 20):
        out = model(torch.tensor([r[:64] for r in nontarget[i:i + 20]], device=dev), cache_type="input")
        ci = model.calc_causal_importances(out.cache, sampling="continuous").lower_leaky
        flat = torch.cat([v for v in ci.values()], -1).flatten(0, 1)
        nt_ci_sum = flat.sum(0) if nt_ci_sum is None else nt_ci_sum + flat.sum(0)
        n_tok += flat.shape[0]
        for t in nt_imp:
            nt_imp[t] += (flat > t).sum(-1).tolist()

mean = lambda xs: sum(xs) / len(xs)
names = [f"{k}:{j}" for k, v in model.components.items() for j in range(v.C)] if hasattr(model, "components") else None
mean_ci_t = ci_sum / len(targets)
mean_ci_nt = nt_ci_sum / n_tok
top = mean_ci_t.argsort(descending=True)[:25].tolist()
report = {
    "n_targets": len(targets),
    "by_condition": {k: {m: mean(v[m]) for m in v} for k, v in stats.items()},
    "n_components_important_at_bold_decision": {str(t): mean(v) for t, v in n_imp.items()},
    "n_components_important_per_nontarget_token": {str(t): mean(v) for t, v in nt_imp.items()},
    "n_components_total": int(mean_ci_t.numel()),
    "top_components_at_bold_decision": [
        {"component": names[j] if names else j, "mean_ci_target": mean_ci_t[j].item(), "mean_ci_nontarget": mean_ci_nt[j].item()}
        for j in top],
}
print(json.dumps(report, indent=1))
Path(args.run, f"eval_bold_{args.step}.json").write_text(json.dumps(report, indent=1))
