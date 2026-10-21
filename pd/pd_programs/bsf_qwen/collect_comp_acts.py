"""Real activations of a chosen set of p-82af68dd components on *every* token: re-run layer 0 over the
collected documents and project the down_proj input on the fly, (x·v)·‖u‖, so nothing big is stored.

Writes <acts>/comp_acts.npy (N_tokens, n_comps) float32 and comp_acts.json (component order).
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--acts", default="/workspace/acts/l0_down")
p.add_argument("--uv", default="/workspace/pd82/down0_uv_p-82af68dd.npz")
p.add_argument("--comps", default="/workspace/bsf_runs/grass_g1024_s3_k32/compare_b260/report.json",
               help="report.json whose 'components' keys are the component ids, or a comma list")
p.add_argument("--batch", type=int, default=64)
args = p.parse_args()
root = Path(args.acts)

comps = (sorted(int(c) for c in json.load(open(args.comps))["components"]) if args.comps.endswith(".json")
         else [int(c) for c in args.comps.split(",")])
uv = np.load(args.uv)
Vc = torch.as_tensor(uv["V"][:, comps]).cuda()                    # (6144, n)
un = torch.as_tensor(np.linalg.norm(uv["U"][comps], axis=1)).cuda()

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", dtype=torch.bfloat16).cuda().eval()
model.model.layers = model.model.layers[:1]
model.config.num_hidden_layers = 1
captured = {}
model.model.layers[0].mlp.down_proj.register_forward_hook(lambda m, i, o: captured.__setitem__("in", i[0]))

tokens, doc = np.load(root / "tokens.npy"), np.load(root / "doc.npy")
starts = np.flatnonzero(np.r_[True, doc[1:] != doc[:-1]])
ends = np.r_[starts[1:], len(tokens)]
out = np.lib.format.open_memmap(root / "comp_acts.npy", mode="w+", dtype=np.float32, shape=(len(tokens), len(comps)))
with torch.no_grad():
    for b0 in range(0, len(starts), args.batch):
        spans = list(zip(starts[b0 : b0 + args.batch], ends[b0 : b0 + args.batch]))
        L = max(e - s for s, e in spans)
        x = torch.full((len(spans), L), tok.pad_token_id or 0)
        mask = torch.zeros_like(x)
        for b, (s, e) in enumerate(spans):
            x[b, : e - s], mask[b, : e - s] = torch.tensor(tokens[s:e]), 1
        model.model(input_ids=x.cuda(), attention_mask=mask.cuda())
        a = (captured["in"].float() @ Vc) * un                         # (B, L, n)
        for b, (s, e) in enumerate(spans):
            out[s:e] = a[b, : e - s].cpu().numpy()
        if (b0 // args.batch) % 50 == 0:
            print(f"docs {b0 + len(spans):,}/{len(starts):,}", flush=True)
out.flush()
json.dump(comps, open(root / "comp_acts.json", "w"))
print("done", comps, flush=True)
