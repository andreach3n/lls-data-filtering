"""Controls for the teen-family results on p-82af68dd (12 teen-specific layer-0 components).

A. FineWeb KL, editing only at teen tokens (a digit right after "1"):
   family zero, family mean-ablation (mean over teen tokens), 5 random 12-component sets whose
   zeroing removes as much output energy at teen tokens as the family's, and a random direction
   with the same per-token norm as the family-zero edit.
B. Comparison prompts ("which is larger/smaller, 1r or 1x"): does a swap steer the answer to the
   swapped digit's answer, or only damage it? "Moved" is counted only where the swapped answer is
   neither the original answer nor the fallback (the reference number r).
"""

import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RUN = "p-82af68dd"
work = Path.home() / "work"
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
dev = "cuda"
rng = np.random.default_rng(0)

uv = np.load(work / f"down0_uv_{RUN}.npz")
V = torch.tensor(uv["V"], device=dev)
U = torch.tensor(uv["U"], device=dev)
C = V.shape[1]
FAMILY = [5802, 5882, 2431, 3323, 3333, 97, 1678, 4986, 707, 5592, 1198, 4639]
means = torch.tensor(np.load(work / f"teen_stats_{RUN}.npz")["means"], device=dev)  # (10, C) x·v

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.float32).to(dev).eval()
layer0 = model.model.layers[0]
down = layer0.mlp.down_proj
state = {}


def down_hook(m, args, out):
    state["x"] = args[0]
    state["down_out"] = out
    if "delta" in state:  # (B, S, d_out), zero where no edit
        out = out + state["delta"].to(out.dtype)
    if "down_replace" in state:
        out = out.clone(); out[0, state["pos"]] = state["down_replace"]
    return out


down.register_forward_hook(down_hook)

# ---------------- A. FineWeb KL controls ----------------
tokens = np.load(work / "sub" / "tokens.npy")
R, S = tokens.shape
strings = np.array([tok.decode([t]) for t in range(len(tok))], dtype=object)[tokens]
prev = np.full_like(strings, "")
prev[:, 1:] = strings[:, :-1]
teen = np.isin(strings, list("0123456789")) & (prev == "1")
after = np.zeros_like(teen)
for k in (1, 2, 3):
    after[:, k:] |= teen[:, :-k]
after &= ~teen
B = 8

# pass 1: per-component output energy at teen tokens, E_c = mean (x·v_c)^2 ‖u_c‖^2, and the mean x·v on teen tokens
u2 = (U ** 2).sum(1)
energy = torch.zeros(C, device=dev, dtype=torch.float64)
act_sum = torch.zeros(C, device=dev, dtype=torch.float64)
n_teen = 0
for start in range(0, R, B):
    ids = torch.tensor(tokens[start:start + B], device=dev, dtype=torch.long)
    state.pop("delta", None)
    model.model(ids)  # base model only; logits not needed
    m = torch.tensor(teen[start:start + B], device=dev)
    a = state["x"].float()[m] @ V  # (n, C)
    energy += ((a ** 2) * u2).sum(0).double()
    act_sum += a.sum(0).double()
    n_teen += int(m.sum())
energy /= n_teen
teen_mean = (act_sum / n_teen).float()
fam_E = float(energy[FAMILY].sum())
others = np.array([c for c in range(C) if c not in FAMILY])
order = others[np.argsort(-energy[others].cpu().numpy())]
e_np = energy.cpu().numpy()
random_sets = []
# Draw 12 from the K non-family components with most energy at teen tokens; shrink K until
# random sets reach the family's energy (within 10%). If even K=24 can't, keep the closest draws.
for K in (400, 200, 100, 60, 40, 24):
    pool = order[:K]
    draws = [rng.choice(pool, 12, replace=False) for _ in range(3000)]
    ratios = np.array([e_np[s].sum() / fam_E for s in draws])
    ok = [d for d, q in zip(draws, ratios) if abs(q - 1) < 0.1]
    if len(ok) >= 5:
        random_sets = [sorted(int(c) for c in d) for d in ok[:5]]
        break
if not random_sets:
    best = np.argsort(np.abs(ratios - 1))[:5]
    random_sets = [sorted(int(c) for c in draws[i]) for i in best]
print("random-set pool size K =", K)
print(f"family energy at teen tokens {fam_E:.3f}; random sets (energy ratio):",
      [round(float(energy[s].sum()) / fam_E, 2) for s in random_sets])
print("random sets:", random_sets)


def edit_delta(kind, comps, x, m):
    """Edit to down_proj's output at the teen positions m (B, S)."""
    idx = torch.tensor(comps, device=dev)
    a = x @ V[:, idx]
    if kind == "zero":
        new = torch.zeros_like(a)
    elif kind == "mean":
        new = teen_mean[idx].expand_as(a)
    d = (new - a) @ U[idx]
    return d * m[..., None]


CONDS = {"family:zero": ("zero", FAMILY), "family:mean": ("mean", FAMILY)}
for i, s in enumerate(random_sets):
    CONDS[f"random{i}:zero"] = ("zero", s)
    CONDS[f"random{i}:mean"] = ("mean", s)
CONDS["random_direction"] = ("dir", FAMILY)
masks = {"teen": teen, "after3": after}
sums = {c: {k: 0.0 for k in masks} for c in CONDS}
dnorm = {c: 0.0 for c in CONDS}
counts = {k: 0 for k in masks}
for start in range(0, R, B):
    sl = slice(start, start + B)
    ids = torch.tensor(tokens[sl], device=dev, dtype=torch.long)
    state.pop("delta", None)
    logp0 = torch.log_softmax(model(ids).logits.float(), -1)[:, :-1]
    x = state["x"].float()
    out_norm = state["down_out"].float().norm(dim=-1)
    m = torch.tensor(teen[sl], device=dev)
    mk = {k: torch.tensor(v[sl][:, :-1], device=dev) for k, v in masks.items()}
    for k in masks:
        counts[k] += int(mk[k].sum())
    fam_zero = edit_delta("zero", FAMILY, x, m)
    for name, (kind, comps) in CONDS.items():
        if kind == "dir":
            g = torch.randn_like(fam_zero)
            d = g / g.norm(dim=-1, keepdim=True) * fam_zero.norm(dim=-1, keepdim=True)
        else:
            d = edit_delta(kind, comps, x, m)
        state["delta"] = d
        if m.any():
            dnorm[name] += float((d.norm(dim=-1)[m] / out_norm[m]).sum())
        logp = torch.log_softmax(model(ids).logits.float(), -1)[:, :-1]
        kl = (logp0.exp() * (logp0 - logp)).sum(-1)
        for k, mm in mk.items():
            sums[name][k] += float(kl[mm].sum())
    state.pop("delta", None)
n_t = int(teen.sum())
print(f"\n=== A. KL (nats/token), edits only at {n_t} teen tokens")
print(f"{'condition':18s} {'KL teen':>9s} {'KL after3':>9s} {'|edit|/|MLP out|':>16s}")
resA = {}
for name in CONDS:
    resA[name] = dict(kl_teen=sums[name]["teen"] / counts["teen"], kl_after3=sums[name]["after3"] / counts["after3"],
                      edit_rel=dnorm[name] / n_t)
    print(f"{name:18s} {resA[name]['kl_teen']:9.5f} {resA[name]['kl_after3']:9.5f} {resA[name]['edit_rel']:16.2f}")

# ---------------- B. comparisons: steering vs damage ----------------
TEMPLATES = {
    "larger": [("Which is larger, 1{r} or ", "? The larger number is 1"),
               ("Of 1{r} and ", ", the bigger number is 1"),
               ("Compare 1{r} and ", ". The greater of the two is 1"),
               ("max(1{r}, ", ") = 1")],
    "smaller": [("Which is smaller, 1{r} or ", "? The smaller number is 1"),
                ("Of 1{r} and ", ", the smaller number is 1"),
                ("Compare 1{r} and ", ". The lesser of the two is 1"),
                ("min(1{r}, ", ") = 1")],
}
digit_ids = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]


def build(prefix, suffix, x):
    text = prefix + "1" + str(x) + suffix
    enc = tok(text, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    char = len(prefix) + 1
    pos = [i for i, (s, e) in enumerate(offs) if s <= char < e]
    assert len(pos) == 1 and tok.decode([ids[pos[0]]]) == str(x), (text, pos)
    return ids, pos[0]


def run(ids, pos, delta_vec=None, replace=None):
    state.pop("delta", None); state.pop("down_replace", None)
    state["pos"] = pos
    if delta_vec is not None:
        d = torch.zeros(1, len(ids), U.shape[1], device=dev)
        d[0, pos] = delta_vec
        state["delta"] = d
    if replace is not None:
        state["down_replace"] = replace
    p = torch.softmax(model(torch.tensor([ids], device=dev)).logits[0, -1].float(), -1)[digit_ids]
    return (p / p.sum()).cpu().numpy()


fam = torch.tensor(FAMILY, device=dev)
recs = []
for task, temps in TEMPLATES.items():
    f = (lambda x, r: max(x, r)) if task == "larger" else (lambda x, r: min(x, r))
    for ti, (pre, suf) in enumerate(temps):
        for r in (3, 5, 7):
            pre_r = pre.format(r=r)
            clean = {}
            for x in range(10):
                if x == r:
                    continue
                ids, pos = build(pre_r, suf, x)
                p = run(ids, pos)
                clean[x] = dict(ids=ids, pos=pos, p=p, a=state["x"][0, pos].float() @ V[:, fam],
                                down=state["down_out"][0, pos].clone())
            for x, c in clean.items():
                base_ok = int(np.argmax(c["p"])) == f(x, r)
                for cond in ("zero", "mean"):
                    new = torch.zeros_like(c["a"]) if cond == "zero" else teen_mean[fam]
                    p = run(c["ids"], c["pos"], delta_vec=(new - c["a"]) @ U[fam])
                    recs.append(dict(task=task, t=ti, r=r, x=x, y=None, cond=cond, ok=base_ok,
                                     ans=int(np.argmax(p)), fx=f(x, r)))
                for y in clean:
                    if y == x:
                        continue
                    p = run(c["ids"], c["pos"], delta_vec=(means[y, fam] - c["a"]) @ U[fam])
                    recs.append(dict(task=task, t=ti, r=r, x=x, y=y, cond="swap", ok=base_ok,
                                     ans=int(np.argmax(p)), fx=f(x, r), fy=f(y, r)))
                    p = run(c["ids"], c["pos"], replace=clean[y]["down"])
                    recs.append(dict(task=task, t=ti, r=r, x=x, y=y, cond="mlp", ok=base_ok,
                                     ans=int(np.argmax(p)), fx=f(x, r), fy=f(y, r)))

print("\n=== B. comparisons (only prompts the model answers correctly unedited)")
for task in TEMPLATES:
    rs = [q for q in recs if q["task"] == task and q["ok"]]
    base_acc = np.mean([q["ok"] for q in recs if q["task"] == task and q["cond"] == "zero"])
    print(f"\n{task}: unedited accuracy {base_acc:.2f}")
    for cond in ("zero", "mean"):
        q = [z for z in rs if z["cond"] == cond]
        print(f"  {cond:5s}  answer unchanged {np.mean([z['ans'] == z['fx'] for z in q]):.2f}   "
              f"answer = reference r (fallback) {np.mean([z['ans'] == z['r'] for z in q]):.2f}   n={len(q)}")
    for cond in ("swap", "mlp"):
        q = [z for z in rs if z["cond"] == cond]
        targeted = [z for z in q if z["fy"] != z["fx"] and z["fy"] != z["r"]]  # swapped answer is the new number itself
        fallback = [z for z in q if z["fy"] != z["fx"] and z["fy"] == z["r"]]  # swapped answer is the reference
        print(f"  {cond:5s}  answer unchanged {np.mean([z['ans'] == z['fx'] for z in q]):.2f}   "
              f"moved to f(y), f(y) not the fallback: {np.mean([z['ans'] == z['fy'] for z in targeted]):.2f} (n={len(targeted)})   "
              f"moved to f(y) = fallback: {np.mean([z['ans'] == z['fy'] for z in fallback]):.2f} (n={len(fallback)})")
json.dump(dict(A=resA, random_sets=random_sets, B=recs), open(work / "controls_82.json", "w"))
