"""Re-run Qwen3-1.7B layer 0 on the already-collected documents and save the *input* to layer-0
down_proj (6144-d) at digit tokens only. Component activations are (x·v)·‖u‖ with x this input,
which collect_acts.py did not keep (it saved the 2048-d output for the BSF).

Feeds the saved token ids back in, document by document, so the rows line up exactly with
acts.npy. Writes to --acts:
  down_in_digits.npy   (M, 6144) float16
  down_in_idx.npy      (M,) int64 — row index into tokens.npy / acts.npy
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--acts", default="/workspace/acts/l0_down")
p.add_argument("--batch", type=int, default=64)
args = p.parse_args()
root = Path(args.acts)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16).cuda().eval()
model.model.layers = model.model.layers[:1]
model.config.num_hidden_layers = 1
captured = {}
model.model.layers[0].mlp.down_proj.register_forward_hook(lambda m, i, o: captured.__setitem__("in", i[0]))

tokens, doc = np.load(root / "tokens.npy"), np.load(root / "doc.npy")
digit_ids = np.array([tok.convert_tokens_to_ids(str(k)) for k in range(10)])
is_digit = np.isin(tokens, digit_ids)
starts = np.flatnonzero(np.r_[True, doc[1:] != doc[:-1]])
ends = np.r_[starts[1:], len(tokens)]
print(f"{len(starts):,} docs, {is_digit.sum():,} digit tokens", flush=True)

M = int(is_digit.sum())
X = np.lib.format.open_memmap(root / "down_in_digits.npy", mode="w+", dtype=np.float16, shape=(M, 6144))
I = np.zeros(M, np.int64)
m = 0
with torch.no_grad():
    for b0 in range(0, len(starts), args.batch):
        spans = list(zip(starts[b0 : b0 + args.batch], ends[b0 : b0 + args.batch]))
        L = max(e - s for s, e in spans)
        x = torch.full((len(spans), L), tok.pad_token_id or 0)
        mask = torch.zeros_like(x)
        for b, (s, e) in enumerate(spans):
            x[b, : e - s], mask[b, : e - s] = torch.tensor(tokens[s:e]), 1
        model.model(input_ids=x.cuda(), attention_mask=mask.cuda())
        h = captured["in"]
        for b, (s, e) in enumerate(spans):
            rows = np.flatnonzero(is_digit[s:e])
            if len(rows):
                X[m : m + len(rows)] = h[b, rows].float().cpu().numpy()
                I[m : m + len(rows)] = s + rows
                m += len(rows)
        if (b0 // args.batch) % 50 == 0:
            print(f"docs {b0 + len(spans):,}/{len(starts):,}, digit tokens {m:,}/{M:,}", flush=True)
assert m == M, (m, M)
X.flush()
np.save(root / "down_in_idx.npy", I)
print("done", flush=True)
