"""Follow-up controls for the fine-tuned-stream ablation (see control_ablation.py).

1. Size-matched random control: remove 17 LoRA components drawn uniformly from all components NOT in the
   selected set (the by-matrix control could not match 10 picks in a 16-component matrix).
2. Copying check: P(bold) for the base and fine-tuned models, split by whether the 48-token window already
   contains a ** token before the bold decision point.

  python control_ablation2.py --base <model_<step>.pth> --out <ft_stream run dir> --data /workspace/mixdata
"""
import argparse
import json
from pathlib import Path

import torch

import ft_stream as fsm

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--lora", default="/workspace/smallsft/runs/strip_r16_12L/lora.pt")
ap.add_argument("--data", default="/workspace/mixdata")
ap.add_argument("--c_delta", type=int, default=16)
ap.add_argument("--draws", type=int, default=20)
args = ap.parse_args()

import control_ablation as ca  # noqa: E402  (reuses its model, windows, selection and helpers)

fs, half_b, selected = ca.fs, ca.half_b, ca.selected
n_sel = int(sum(v.sum() for v in selected.values()))
flat = [(nm, i) for nm in fs.names for i in range(fs.Cd) if not bool(selected[nm][i])]
g = torch.Generator().manual_seed(1)
size_matched = []
for _ in range(args.draws):
    pick = torch.randperm(len(flat), generator=g)[:n_sel].tolist()
    rem = {nm: torch.zeros(fs.Cd, dtype=torch.bool, device=fsm.dev) for nm in fs.names}
    for j in pick:
        nm, i = flat[j]
        rem[nm][i] = True
    size_matched.append(ca.p_bold(half_b, rem))

from transformers import PreTrainedTokenizerFast  # noqa: E402
tok = PreTrainedTokenizerFast(tokenizer_file="/workspace/models/spd-t-f99617bb/tokenizer.json")
star_ids = {i for t, i in tok.get_vocab().items() if "**" in t}
has_bold = [w for w in half_b if any(t in star_ids for t in w)]
no_bold = [w for w in half_b if not any(t in star_ids for t in w)]
split = {name: dict(n=len(ws), base=ca.p_bold_plain(ws, "base"), finetuned=ca.p_bold(ws, None))
         for name, ws in (("window_already_has_**", has_bold), ("window_has_no_**", no_bold))}
res = dict(n_selected=n_sel, size_matched_random=dict(mean=sum(size_matched) / len(size_matched),
           min=min(size_matched), max=max(size_matched), draws=size_matched), copy_split=split)
print(json.dumps({"size_matched_random": {k: v for k, v in res["size_matched_random"].items() if k != "draws"}, "copy_split": split}, indent=1))
(Path(args.out) / "control_ablation2.json").write_text(json.dumps(res, indent=1))
