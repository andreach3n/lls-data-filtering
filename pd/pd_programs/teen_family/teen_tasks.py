"""Teen digit swaps on tasks other than counting, with positive controls.

Each task is a prompt containing one teen number 1x whose answer is a single digit f(x)
predicted at the last position. At the units-digit token of 1x we:
  arc / all : set p-e9fcffd5's teen components to their mean values for digit y
  mlp       : replace the whole layer-0 down_proj output with that of the same prompt with 1y
  resid     : replace the whole layer-0 output residual with that of the prompt with 1y
and check whether the answer moves from f(x) to f(y).
"""

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
V = torch.tensor(uv["V"], device=dev)
U = torch.tensor(uv["U"], device=dev)
means = torch.tensor(np.load(work / f"teen_stats_{RUN}.npz")["means"], device=dev)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16).to(dev).eval()
layer0 = model.model.layers[0]
down = layer0.mlp.down_proj
digit_ids = [tok.convert_tokens_to_ids(str(d)) for d in range(10)]

PAIR = {"p-e9fcffd5": [3323, 2431], "p-82af68dd": [5802, 5882]}[RUN]
_stats = np.load(work / f"teen_stats_{RUN}.npz")
TEEN = [5802, 5882, 2431, 3323, 3333, 97, 1678, 4986, 707, 5592, 1198, 4639]  # group A: programs describe a response to the digit right after "1"
print("pair", PAIR, "teen set", TEEN)
ARC = PAIR
ALL = TEEN

# (name, prefix, suffix, answer f(x), valid x). The teen is prefix + "1" + str(x) + suffix.
TASKS = [
    ("add2", "", "+2=1", lambda x: x + 2, range(0, 8)),
    ("sub10", "", "-10=", lambda x: x, range(0, 10)),
    ("sub3", "", "-3=1", lambda x: x - 3, range(3, 10)),
    ("larger_than_15", "Which is larger, 15 or ", "? The larger number is 1", lambda x: max(x, 5), range(0, 10)),
    ("smaller_than_15", "Which is smaller, 15 or ", "? The smaller number is 1", lambda x: min(x, 5), range(0, 10)),
    ("year", "The company was founded in 20", ". Ten years later, in 202", lambda x: x, range(0, 10)),
    ("year_plus2", "The bridge opened in 20", ". Two years later, in 20", lambda x: None, range(0, 8)),
    ("time", "The train leaves at ", ":00 and the trip takes 1 hour, so it arrives at 1", lambda x: x + 1, range(0, 9)),
    ("age", "In 2000 she was ", " years old, so in 2002 she was 1", lambda x: x + 2, range(0, 8)),
    ("apples", "I had ", " apples and ate 2, so now I have 1", lambda x: x - 2, range(2, 10)),
    ("copy", "Code: ", ". Please repeat the code: 1", lambda x: x, range(0, 10)),
]
TASKS = [t for t in TASKS if t[0] != "year_plus2"]  # answer would be two digits ahead


def build(prefix, suffix, x):
    text = prefix + "1" + str(x) + suffix
    enc = tok(text, return_offsets_mapping=True)
    ids, offs = enc["input_ids"], enc["offset_mapping"]
    char = len(prefix) + 1  # the units digit's character
    pos = [i for i, (s, e) in enumerate(offs) if s <= char < e]
    assert len(pos) == 1 and tok.decode([ids[pos[0]]]) == str(x), (text, pos)
    return ids, pos[0]


state = {}


def down_hook(m, args, out):
    state["x"] = args[0]
    state["down_out"] = out
    if "down_delta" in state:
        out = out.clone(); out[0, state["pos"]] += state["down_delta"].to(out.dtype)
    if "down_replace" in state:
        out = out.clone(); out[0, state["pos"]] = state["down_replace"]
    return out


def layer_hook(m, args, out):
    h = out[0] if isinstance(out, tuple) else out
    state["resid"] = h
    if "resid_replace" in state:
        h = h.clone(); h[0, state["pos"]] = state["resid_replace"]
        return (h, *out[1:]) if isinstance(out, tuple) else h
    return out


down.register_forward_hook(down_hook)
layer0.register_forward_hook(layer_hook)
INTERVENTIONS = ("down_delta", "down_replace", "resid_replace")


def run(ids, pos, **iv):
    for k in INTERVENTIONS:
        state.pop(k, None)
    state.update(pos=pos, **iv)
    logits = model(torch.tensor([ids], device=dev)).logits[0, -1].float()
    p = torch.softmax(logits, -1)[digit_ids]
    return (p / p.sum()).cpu().numpy(), float(p.sum())


records = []
for name, prefix, suffix, f, xs in TASKS:
    xs = list(xs)
    clean = {}
    for x in xs:
        ids, pos = build(prefix, suffix, x)
        p, mass = run(ids, pos)
        a = state["x"][0, pos].float() @ V
        clean[x] = dict(ids=ids, pos=pos, p=p, mass=mass, a=a,
                        down=state["down_out"][0, pos].clone(), resid=state["resid"][0, pos].clone())
    for x in xs:
        c = clean[x]
        records.append(dict(task=name, x=x, y=None, cond="baseline", fx=f(x), p=c["p"].tolist(), mass=c["mass"]))
        for y in xs:
            if y == x:
                continue
            for cond, comps in (("arc", ARC), ("all", ALL)):
                idx = torch.tensor(comps, device=dev)
                delta = (means[y, idx] - c["a"][idx]) @ U[idx]
                p, mass = run(c["ids"], c["pos"], down_delta=delta)
                records.append(dict(task=name, x=x, y=y, cond=cond, fx=f(x), fy=f(y), p=p.tolist(), mass=mass))
            # positive controls; prompts for x and y tokenize to the same length
            assert len(clean[y]["ids"]) == len(c["ids"]) and clean[y]["pos"] == c["pos"]
            p, mass = run(c["ids"], c["pos"], down_replace=clean[y]["down"])
            records.append(dict(task=name, x=x, y=y, cond="mlp", fx=f(x), fy=f(y), p=p.tolist(), mass=mass))
            p, mass = run(c["ids"], c["pos"], resid_replace=clean[y]["resid"])
            records.append(dict(task=name, x=x, y=y, cond="resid", fx=f(x), fy=f(y), p=p.tolist(), mass=mass))

(work / f"teen_tasks_{RUN}.json").write_text(json.dumps(records))

print(f"{'task':16s} {'base acc':>8s} {'digit mass':>10s} | moved to f(y):  {'arc':>5s} {'all':>5s} {'mlp':>5s} {'resid':>5s} | P(f(x)) kept: {'arc':>5s} {'all':>5s} {'mlp':>5s} {'resid':>5s}")
for name, *_ in TASKS:
    rs = [r for r in records if r["task"] == name]
    base = [r for r in rs if r["cond"] == "baseline"]
    acc = np.mean([np.argmax(r["p"]) == r["fx"] for r in base])
    mass = np.mean([r["mass"] for r in base])
    moved, kept = [], []
    for cond in ("arc", "all", "mlp", "resid"):
        sw = [r for r in rs if r["cond"] == cond and r["fy"] != r["fx"]]
        moved.append(np.mean([np.argmax(r["p"]) == r["fy"] for r in sw]))
        kept.append(np.mean([r["p"][r["fx"]] for r in sw]))
    print(f"{name:16s} {acc:8.2f} {mass:10.2f} |                {' '.join(f'{v:5.2f}' for v in moved)} |               {' '.join(f'{v:5.2f}' for v in kept)}")
