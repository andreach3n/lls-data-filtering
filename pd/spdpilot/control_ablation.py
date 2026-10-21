"""Control for the fine-tuned-stream ablation: are the important delta components special?

Splits the held-out target windows in two. On half A, selects the delta components important at the
bold decision (mean CI at the last position > thr). On half B, measures P(next token opens bold) for:
  finetuned                      the full fine-tuned model
  minus_selected                 the full fine-tuned model with the selected delta components removed
                                 (all base components, both leftovers kept)
  minus_random (n draws)         the same, removing an equal number of other delta components drawn at
                                 random within each matrix (same count per matrix as the selection)
  base                           the base model

  python control_ablation.py --base <model_<step>.pth> --out <ft_stream run dir> --data /workspace/mixdata
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import ft_stream as fsm

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--lora", default="/workspace/smallsft/runs/strip_r16_12L/lora.pt")
ap.add_argument("--data", default="/workspace/mixdata")
ap.add_argument("--c_delta", type=int, default=16)
ap.add_argument("--thr", type=float, default=0.1)
ap.add_argument("--draws", type=int, default=20)
args = ap.parse_args()
fsm.set_data(args.data)
dev = fsm.dev

fs, bold_mask = fsm.load(args)
ckpt = sorted(Path(args.out).glob("ft_stream_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))[-1]
st = torch.load(ckpt, map_location=dev)
fs.a.load_state_dict(st["a"]); fs.b.load_state_dict(st["b"]); fs.ci.load_state_dict(st["ci"])
rows = [json.loads(l)["input_ids"] for l in open(fsm.TARGETS.format("val"))]
half_a, half_b = rows[: len(rows) // 2], rows[len(rows) // 2:]


@torch.no_grad()
def mean_delta_ci(windows):
    tot, n = None, 0
    for i in range(0, len(windows), 50):
        x = torch.tensor(windows[i:i + 50], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            fs.logits(x, "ft")
            ci = fs.ci_of(fs.cache)
        s = {nm: fsm.lower_leaky(zd)[:, -1].float().sum(0) for nm, (zb, zd) in ci.items()}
        tot = s if tot is None else {k: tot[k] + s[k] for k in s}
        n += x.shape[0]
    return {k: v / n for k, v in tot.items()}


@torch.no_grad()
def p_bold(windows, removed):
    """removed: {module: bool tensor (Cd,)} delta components to switch off; everything else on."""
    out = []
    for i in range(0, len(windows), 50):
        x = torch.tensor(windows[i:i + 50], device=dev)
        lead = x.shape
        ms = {}
        for nm in fs.names:
            keep = (~removed[nm]).float() if removed is not None else torch.ones(fs.Cd, device=dev)
            ms[nm] = {"base": torch.ones(*lead, fs.Cb[nm], device=dev), "delta": keep.expand(*lead, fs.Cd).clone(),
                      "rb": torch.ones(lead, device=dev), "rd": torch.ones(lead, device=dev)}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = fs.logits(x, "masked", ms)[:, -1].float()
        out += F.softmax(fsm.binary(logits, bold_mask), -1)[:, 0].tolist()
    return sum(out) / len(out)


@torch.no_grad()
def p_bold_plain(windows, mode):
    out = []
    for i in range(0, len(windows), 50):
        x = torch.tensor(windows[i:i + 50], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = fs.logits(x, mode)[:, -1].float()
        out += F.softmax(fsm.binary(logits, bold_mask), -1)[:, 0].tolist()
    return sum(out) / len(out)


ci_a = mean_delta_ci(half_a)
selected = {nm: v > args.thr for nm, v in ci_a.items()}
n_sel = int(sum(v.sum() for v in selected.values()))
per_module = {nm: int(v.sum()) for nm, v in selected.items() if v.any()}
g = torch.Generator(device="cpu").manual_seed(0)
rand_p = []
for _ in range(args.draws):
    rem = {}
    for nm in fs.names:
        r = torch.zeros(fs.Cd, dtype=torch.bool)
        k = per_module.get(nm, 0)
        if k:
            pool = (~selected[nm].cpu()).nonzero().flatten()
            r[pool[torch.randperm(len(pool), generator=g)[:k]]] = True
        rem[nm] = r.to(dev)
    rand_p.append(p_bold(half_b, rem))
res = dict(
    checkpoint=str(ckpt), n_windows_select=len(half_a), n_windows_eval=len(half_b), threshold=args.thr,
    n_selected=n_sel, selected_per_module=per_module,
    p_bold_base=p_bold_plain(half_b, "base"), p_bold_finetuned=p_bold(half_b, None),
    p_bold_minus_selected=p_bold(half_b, selected),
    p_bold_minus_random=dict(mean=sum(rand_p) / len(rand_p), min=min(rand_p), max=max(rand_p), draws=rand_p),
)
f, b = res["p_bold_finetuned"], res["p_bold_base"]
res["share_removed_selected"] = (f - res["p_bold_minus_selected"]) / (f - b)
res["share_removed_random_mean"] = (f - res["p_bold_minus_random"]["mean"]) / (f - b)
print(json.dumps({k: v for k, v in res.items() if k != "p_bold_minus_random"} | {"random": {k: v for k, v in res["p_bold_minus_random"].items() if k != "draws"}}, indent=1))
(Path(args.out) / "control_ablation.json").write_text(json.dumps(res, indent=1))
