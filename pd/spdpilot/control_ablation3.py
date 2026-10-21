"""Which of the 17 selected LoRA components matter, and are they specific to bold?

1. Group ablations on the evaluation half (see control_ablation.py): remove only the components that push
   the bold logit up, only those that push it down, or only the five that do not write to the residual
   stream; report P(bold) for each.
2. Specificity: on ordinary text (validation chunks, first 64 tokens, every position), the mean next-token
   KL between the fine-tuned model and (a) the fine-tuned model minus the 17, (b) minus 17 random
   components, (c) the base model. A bold-specific set should move ordinary text far less than SFT did.

  python control_ablation3.py --base <model_<step>.pth> --out <ft_stream run dir> --data /workspace/mixdata
"""
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

import control_ablation as ca  # reuses its arguments, model, windows, selection and helpers
import ft_stream as fsm

fs, half_b, selected, dev = ca.fs, ca.half_b, ca.selected, fsm.dev
GROUPS = {  # from Fig. 4 / logit_check.json of the D1 fine-tuned stream
    "push_up": [("h.11.mlp.down_proj", i) for i in (4, 7, 1, 13)],
    "push_down": [("h.11.attn.o_proj", 9)] + [("h.11.mlp.down_proj", i) for i in (2, 12, 9, 6, 11, 14)],
    "upstream": [("h.10.mlp.c_fc", 9), ("h.11.mlp.c_fc", 3), ("h.11.attn.v_proj", 2), ("h.7.attn.v_proj", 0), ("h.5.attn.v_proj", 1)],
}


def mask_of(pairs):
    m = {nm: torch.zeros(fs.Cd, dtype=torch.bool, device=dev) for nm in fs.names}
    for nm, i in pairs:
        assert bool(selected[nm][i]), (nm, i)
        m[nm][i] = True
    return m


res = {"p_bold": {"finetuned": ca.p_bold(half_b, None), "base": ca.p_bold_plain(half_b, "base"),
                  "minus_all_17": ca.p_bold(half_b, selected)}}
for g, pairs in GROUPS.items():
    res["p_bold"][f"minus_{g}_only"] = ca.p_bold(half_b, mask_of(pairs))


@torch.no_grad()
def ordinary_kl(removed, mode=None, n=200):
    rows = pq.read_table(f"{ca.args.data}/data/validation-00000.parquet").column("input_ids").to_pylist()[:n]
    tot, cnt = 0.0, 0
    for i in range(0, len(rows), 20):
        x = torch.tensor([r[:64] for r in rows[i:i + 20]], device=dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            ref = fs.logits(x, "ft").float()
            if mode == "base":
                other = fs.logits(x, "base").float()
            else:
                lead = x.shape
                ms = {nm: {"base": torch.ones(*lead, fs.Cb[nm], device=dev),
                           "delta": (~removed[nm]).float().expand(*lead, fs.Cd).clone(),
                           "rb": torch.ones(lead, device=dev), "rd": torch.ones(lead, device=dev)} for nm in fs.names}
                other = fs.logits(x, "masked", ms).float()
        lp, lq = F.log_softmax(ref, -1), F.log_softmax(other, -1)
        tot += (lp.exp() * (lp - lq)).sum(-1).sum().item()
        cnt += x.numel()
    return tot / cnt


flat = [(nm, i) for nm in fs.names for i in range(fs.Cd) if not bool(selected[nm][i])]
g = torch.Generator().manual_seed(2)
rand_kl = []
for _ in range(5):
    m = {nm: torch.zeros(fs.Cd, dtype=torch.bool, device=dev) for nm in fs.names}
    for j in torch.randperm(len(flat), generator=g)[:17].tolist():
        m[flat[j][0]][flat[j][1]] = True
    rand_kl.append(ordinary_kl(m))
res["ordinary_text_kl_from_finetuned"] = {
    "minus_17_selected": ordinary_kl(selected),
    "minus_17_random_mean": sum(rand_kl) / len(rand_kl),
    "base_model": ordinary_kl(None, mode="base"),
}
print(json.dumps(res, indent=1))
(Path(ca.args.out) / "control_ablation3.json").write_text(json.dumps(res, indent=1))
