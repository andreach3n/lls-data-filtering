"""Program substitution on p-82af68dd's layer-0 teen components, on natural text.

At every token, component c's activation is replaced by a predictor's value: down_proj's output
changes by (pred_c / ‖u_c‖ - x·v_c) · u_c, since the site's activation is (x·v)·‖u‖.
Scores: KL(original ‖ modified) and ΔCE of the next-token prediction, averaged over all
positions, at teen tokens (a digit after "1"), and at the 3 positions after them; plus each
predictor's VE against the true activation.
"""

import json
from math import cos, radians, sin
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RUN = "p-82af68dd"
work = Path.home() / "work"
sub = work / "sub"
torch.set_grad_enabled(False)
dev = "cuda"
BATCH = 8

uv = np.load(work / f"down0_uv_{RUN}.npz")
V = torch.tensor(uv["V"], device=dev)
U = torch.tensor(uv["U"], device=dev)
u_norm = U.norm(dim=1)
PAIR = [5802, 5882]
TEEN = [5802, 5882, 2431, 3323, 3333, 97, 1678, 4986, 707, 5592, 1198, 4639]
preds = dict(np.load(sub / "preds.npz"))
tokens = np.load(sub / "tokens.npy")
R, S = tokens.shape

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
strings = np.array([tok.decode([t]) for t in range(len(tok))], dtype=object)[tokens]
prev = np.full_like(strings, "")
prev[:, 1:] = strings[:, :-1]
teen = np.isin(strings, list("0123456789")) & (prev == "1")
after = np.zeros_like(teen)
for k in (1, 2, 3):
    after[:, k:] |= teen[:, :-k]
after &= ~teen
digit = np.where(teen, np.char.isdigit(strings.astype(str)) * 0, 0)
digit[teen] = strings[teen].astype(int)
print(f"rows {R}, teen tokens {teen.sum()}, after-teen positions {after.sum()}")

# Report's predictors for the pair (0 off teen tokens), in site units.
T_5802 = np.array([1.5, 1.6, 1.9, 1.8, 1.7, 1.4, 0.9, 0.7, 0.4, -0.3])
T_5882 = np.array([-0.2, -0.2, 0.1, 0.6, 0.9, 1.4, 1.6, 1.8, 2.1, 2.4])
theta = np.radians(-16 + 12 * digit)
report = {
    "report_tables": {5802: np.where(teen, T_5802[digit], 0.0), 5882: np.where(teen, T_5882[digit], 0.0)},
    "report_joint": {5802: np.where(teen, 1.8 * np.cos(theta), 0.0), 5882: np.where(teen, 2.2 * np.sin(theta), 0.0)},
}

# fp32 like the site's "strict fp32" scores: in bf16, adding even a zero-effect delta changes rounding.
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float32).to(dev).eval()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
down = model.model.layers[0].mlp.down_proj
state = {}


def hook(m, args, out):
    state["x"] = args[0]
    if "new" in state:  # (comps, new site-unit activations (B, S, k))
        comps, new = state["new"]
        idx = torch.tensor(comps, device=dev)
        a = args[0].float() @ V[:, idx]  # x·v
        delta = (new / u_norm[idx] - a) @ U[idx]
        out = out + delta.to(out.dtype)
    return out


down.register_forward_hook(hook)

# Conditions: name -> (components, function(rows slice, true site acts (B,S,k)) -> new acts)
def program(model_name, comps):
    return comps, lambda sl, true: torch.tensor(
        np.stack([preds[f"{c}_{model_name}"][sl] for c in comps], -1), device=dev)


def fixed(comps, table):
    return comps, lambda sl, true: torch.tensor(np.stack([table[c][sl] for c in comps], -1), device=dev, dtype=torch.float32)


CONDS = {"pair:true": (PAIR, lambda sl, true: true), "teenA:true": (TEEN, lambda sl, true: true)}  # noise floor
for m in ("sol", "haiku", "luna"):
    CONDS[f"pair:{m}"] = program(m, PAIR)
for name, table in report.items():
    CONDS[f"pair:{name}"] = fixed(PAIR, table)
CONDS["pair:zero"] = (PAIR, lambda sl, true: torch.zeros_like(true))
for m in ("sol", "haiku", "luna"):
    CONDS[f"teenA:{m}"] = program(m, TEEN)
CONDS["teenA:zero"] = (TEEN, lambda sl, true: torch.zeros_like(true))
CONDS["all6144:zero"] = (list(range(V.shape[1])), lambda sl, true: torch.zeros_like(true))  # whole layer-0 MLP removed

masks = {"all": np.ones_like(teen), "teen": teen, "after3": after}
sums = {c: {k: [0.0, 0.0] for k in masks} for c in CONDS}  # KL, dCE sums
counts = {k: 0 for k in masks}
true_acts = {c: [] for c in TEEN}
for start in range(0, R, BATCH):
    sl = slice(start, start + BATCH)
    ids = torch.tensor(tokens[sl], device=dev, dtype=torch.long)
    targets = ids[:, 1:]
    state.pop("new", None)
    logp0 = torch.log_softmax(model(ids).logits.float(), -1)[:, :-1]
    xt = state["x"].float()
    for c in TEEN:
        true_acts[c].append(((xt @ V[:, c]) * u_norm[c]).cpu().numpy())
    ce0 = -logp0.gather(-1, targets[..., None])[..., 0]
    m_np = {k: torch.tensor(v[sl][:, :-1], device=dev) for k, v in masks.items()}
    for k in masks:
        counts[k] += int(m_np[k].sum())
    for name, (comps, fn) in CONDS.items():
        idx = torch.tensor(comps, device=dev)
        true = (xt @ V[:, idx]) * u_norm[idx]
        state["new"] = (comps, fn(sl, true).float())
        logp = torch.log_softmax(model(ids).logits.float(), -1)[:, :-1]
        kl = (logp0.exp() * (logp0 - logp)).sum(-1)
        dce = -logp.gather(-1, targets[..., None])[..., 0] - ce0
        for k, m in m_np.items():
            sums[name][k][0] += float(kl[m].sum())
            sums[name][k][1] += float(dce[m].sum())
    state.pop("new", None)
    if start % 128 == 0:
        print("batch", start, flush=True)

# VE of each predictor against the true activation, on all tokens and on teen tokens.
true_acts = {c: np.concatenate(v) for c, v in true_acts.items()}


def ve(y, p):
    return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()


print("\n=== VE of predictors (all tokens / teen tokens)")
for c in PAIR:
    line = f"{c}:"
    for m in ("sol", "haiku", "luna"):
        p = preds[f"{c}_{m}"]
        line += f"  {m} {ve(true_acts[c], p):.3f}/{ve(true_acts[c][teen], p[teen]):.3f}"
    for name, table in report.items():
        line += f"  {name} {ve(true_acts[c], table[c]):.3f}/{ve(true_acts[c][teen], table[c][teen]):.3f}"
    print(line)
ves = {m: np.mean([ve(true_acts[c], preds[f"{c}_{m}"]) for c in TEEN]) for m in ("sol", "haiku", "luna")}
print("teenA mean VE (all tokens):", {m: round(float(v), 3) for m, v in ves.items()})

print(f"\n=== KL (nats/token) and ΔCE (nats/token); positions: all {counts['all']}, teen {counts['teen']}, after3 {counts['after3']}")
print(f"{'condition':22s} " + " ".join(f"{'KL ' + k:>11s} {'dCE ' + k:>11s}" for k in masks))
result = {}
for name in CONDS:
    vals = {k: (sums[name][k][0] / counts[k], sums[name][k][1] / counts[k]) for k in masks}
    result[name] = vals
    print(f"{name:22s} " + " ".join(f"{vals[k][0]:11.5f} {vals[k][1]:11.5f}" for k in masks))
json.dump(dict(result=result, counts=counts), open(sub / "sub_results.json", "w"), indent=1)
