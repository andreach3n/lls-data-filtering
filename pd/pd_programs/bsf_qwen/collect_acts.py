"""Collect Qwen3-1.7B layer-0 MLP down_proj outputs (2048-d) on web text, for BSF training.

One document per sequence (truncated to --ctx tokens, no BOS, as in param-decomp's dense Qwen3 data),
so document boundaries are exact for the programs. Only layer 0 is run.

Writes to --out:
  acts.npy      (N, 2048) float16, memory-mapped
  tokens.npy    (N,) int32 token ids
  doc.npy       (N,) int32 document index
  pos.npy       (N,) int16 position within the document
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--out", default="/workspace/acts/l0_down")
p.add_argument("--n_tokens", type=int, default=5_000_000)
p.add_argument("--ctx", type=int, default=1024)
p.add_argument("--batch", type=int, default=64)
p.add_argument("--dataset", default="HuggingFaceFW/fineweb")
p.add_argument("--subset", default="sample-10BT")
args = p.parse_args()

out = Path(args.out)
out.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16).cuda().eval()
model.model.layers = model.model.layers[:1]   # only layer 0 is needed
model.config.num_hidden_layers = 1

captured = {}
model.model.layers[0].mlp.down_proj.register_forward_hook(lambda m, i, o: captured.__setitem__("out", o))

N, d = args.n_tokens, model.config.hidden_size
acts = np.lib.format.open_memmap(out / "acts.npy", mode="w+", dtype=np.float16, shape=(N, d))
tokens, doc, pos = (np.zeros(N, dt) for dt in (np.int32, np.int32, np.int16))

stream = iter(load_dataset(args.dataset, args.subset, split="train", streaming=True))
n, doc_i = 0, 0
with torch.no_grad():
    while n < N:
        ids = []
        while len(ids) < args.batch:
            t = tok(next(stream)["text"])["input_ids"][: args.ctx]
            if len(t) >= 16:
                ids.append(t)
        L = max(map(len, ids))
        x = torch.full((len(ids), L), tok.pad_token_id or 0)
        mask = torch.zeros_like(x)
        for b, t in enumerate(ids):
            x[b, : len(t)], mask[b, : len(t)] = torch.tensor(t), 1
        model.model(input_ids=x.cuda(), attention_mask=mask.cuda())
        o = captured["out"].float().cpu().numpy()
        for b, t in enumerate(ids):
            k = min(len(t), N - n)
            acts[n : n + k] = o[b, :k]
            tokens[n : n + k], doc[n : n + k], pos[n : n + k] = t[:k], doc_i, np.arange(k)
            n, doc_i = n + k, doc_i + 1
            if n >= N:
                break
        print(f"{n:,}/{N:,} tokens, {doc_i:,} docs", flush=True)

acts.flush()
for name, a in (("tokens", tokens), ("doc", doc), ("pos", pos)):
    np.save(out / f"{name}.npy", a)
print("done", out)
