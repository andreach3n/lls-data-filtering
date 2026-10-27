"""Digit-swap interventions on p-e9fcffd5's layer-0 teen components, on the original
Qwen3-1.7B (HF, bf16) with a hook on layers[0].mlp.down_proj.

Changing component c's activation from a to a' adds (a' - a) * U_c to down_proj's output.
On counting prompts ending "..., 1x, 1" the model should predict the digit x+1; we swap
the teen components at the units digit of the last "1x" to their mean values for digit y and
check whether the prediction moves to y+1.
"""

import itertools
import json
from pathlib import Path
import os
RUN = os.environ.get("RUN", "p-e9fcffd5")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

work = Path.home() / "work"
torch.set_grad_enabled(False)
dev = "cuda"

uv = np.load(work / f"down0_uv_{RUN}.npz")
V = torch.tensor(uv["V"], device=dev)  # (d_in, C)
U = torch.tensor(uv["U"], device=dev)  # (C, d_out)
means = torch.tensor(np.load(work / f"teen_stats_{RUN}.npz")["means"], device=dev)  # (10, C): mean x@V on "1d"

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", torch_dtype=torch.bfloat16).to(dev).eval()
down = model.model.layers[0].mlp.down_proj

# ---- 1. checks: reconstruction and hook activations vs JAX -----------------------------
W = down.weight.float()  # (d_out, d_in); y = x @ W.T
recon = V @ U  # (d_in, d_out)
print(f"reconstruction rel err ||VU - W^T|| / ||W||: {(recon - W.T).norm() / W.norm():.4f}")

captured = {}
h = down.register_forward_pre_hook(lambda m, args: captured.__setitem__("x", args[0]))
ref = np.load(work / f"down0_ref_{RUN}.npz")
model(torch.tensor(ref["ids"], device=dev))
h.remove()
acts = (captured["x"].float() @ V).cpu().numpy()
jax_acts = ref["acts"]
err = np.abs(acts - jax_acts)
print(f"hook vs JAX x@V: max abs err {err.max():.3f}, mean abs err {err.mean():.4f}, "
      f"corr {np.corrcoef(acts.ravel()[::97], jax_acts.ravel()[::97])[0, 1]:.5f}, "
      f"mean |act| {np.abs(jax_acts).mean():.4f}")

# ---- 2. prompts -----------------------------------------------------------------------
FORMATS = {
    "comma3": lambda n: f"{n - 2}, {n - 1}, {n}, 1",
    "space3": lambda n: f"Count: {n - 2} {n - 1} {n} 1",
    "lines3": lambda n: f"{n - 2}\n{n - 1}\n{n}\n1",
    "chapter3": lambda n: f"Chapter {n - 2}\nChapter {n - 1}\nChapter {n}\nChapter 1",
    "page1": lambda n: f"Turn to page {n}. Now turn to the next page, page 1",
    "after1": lambda n: f"The number that comes after {n} is 1",
    "age1": lambda n: f"Last year she was {n} years old, so this year she is 1",
}
digit_ids = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]
assert all(isinstance(i, int) for i in digit_ids)


def build(fmt, x):
    """Token ids of the prompt for last number 1x, and the position of x's digit token."""
    ids = tok.encode(FORMATS[fmt](10 + x))
    strs = [tok.decode([i]) for i in ids]
    # units digit of the last full "1x": the last position whose token is str(x) after "1",
    # excluding the trailing "1".
    pos = [p for p in range(1, len(ids) - 1) if strs[p] == str(x) and strs[p - 1] == "1"]
    assert pos and strs[-1] == "1", (fmt, x, strs)
    return ids, pos[-1]


PAIR = {"p-e9fcffd5": [3323, 2431], "p-82af68dd": [5802, 5882]}[RUN]
_stats = np.load(work / f"teen_stats_{RUN}.npz")
TEEN = [5802, 5882, 2431, 3323, 3333, 97, 1678, 4986, 707, 5592, 1198, 4639]  # group A: programs describe a response to the digit right after "1"
print("pair", PAIR, "teen set", TEEN)
SETS = {"arc": PAIR, "rest": [c for c in TEEN if c not in PAIR], "all": TEEN}

state = {}


def hook(module, args, output):
    if "delta" in state:
        output = output.clone()
        output[0, state["pos"]] += state["delta"].to(output.dtype)
    state["x"] = args[0]
    return output


down.register_forward_hook(hook)


def digit_probs(ids):
    logits = model(torch.tensor([ids], device=dev)).logits[0, -1].float()
    p = torch.softmax(logits, -1)
    pd_ = p[digit_ids]
    return (pd_ / pd_.sum()).cpu().numpy(), float(pd_.sum())


# ---- 3. run -----------------------------------------------------------------------------
records = []
for fmt in FORMATS:
    for x in range(2, 9):  # last number 12..18 so the answer 1(x+1) is a teen
        ids, pos = build(fmt, x)
        state.pop("delta", None)
        state["pos"] = pos
        base, mass = digit_probs(ids)
        a = (state["x"][0, pos].float() @ V)  # actual activations at that token, (C,)
        out_norm = float((state["x"][0, pos].float() @ W.T).norm())
        records.append(dict(fmt=fmt, x=x, y=None, cond="baseline", p=base.tolist(), mass=mass))
        for name, comps in SETS.items():
            c = torch.tensor(comps, device=dev)
            targets = [("swap", y, means[y, c]) for y in range(0, 9) if y != x]
            targets += [("self", x, means[x, c]), ("zero", None, torch.zeros(len(comps), device=dev)),
                        ("teen_mean", None, means[:, c].mean(0))]
            for kind, y, new in targets:
                state["delta"] = (new - a[c]) @ U[c]
                p, mass = digit_probs(ids)
                records.append(dict(fmt=fmt, x=x, y=y, cond=f"{name}:{kind}", p=p.tolist(), mass=mass,
                                    delta_rel=float(state["delta"].norm()) / out_norm))
        state.pop("delta", None)

(work / f"teen_swap_{RUN}.json").write_text(json.dumps(records))

# ---- 4. summary -------------------------------------------------------------------------
def summarize(rs, label):
    hit_x = np.mean([np.argmax(r["p"]) == r["x"] + 1 for r in rs])
    line = f"{label:22s} n={len(rs):3d}  argmax=x+1 {hit_x:5.2f}  P(x+1) {np.mean([r['p'][r['x'] + 1] for r in rs]):.3f}"
    sw = [r for r in rs if r["y"] is not None and r["y"] != r["x"]]
    if sw:
        line += (f"  argmax=y+1 {np.mean([np.argmax(r['p']) == r['y'] + 1 for r in sw]):5.2f}"
                 f"  P(y+1) {np.mean([r['p'][r['y'] + 1] for r in sw]):.3f}")
    if "delta_rel" in rs[0]:
        line += f"  |delta|/|out| {np.mean([r['delta_rel'] for r in rs]):.2f}"
    print(line)


print("\n=== all formats")
conds = ["baseline"] + [f"{s}:{k}" for s in SETS for k in ("swap", "self", "zero", "teen_mean")]
for cond in conds:
    summarize([r for r in records if r["cond"] == cond], cond)
print("\n=== swaps by format (argmax = y+1 rate)")
print(f"{'format':10s} " + " ".join(f"{s:>7s}" for s in SETS) + "   baseline argmax=x+1")
for fmt in FORMATS:
    row = [np.mean([np.argmax(r["p"]) == r["y"] + 1 for r in records if r["fmt"] == fmt and r["cond"] == f"{s}:swap"])
           for s in SETS]
    b = np.mean([np.argmax(r["p"]) == r["x"] + 1 for r in records if r["fmt"] == fmt and r["cond"] == "baseline"])
    print(f"{fmt:10s} " + " ".join(f"{v:7.2f}" for v in row) + f"   {b:.2f}")
