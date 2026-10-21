"""Group ablation for a fine-tuned stream (same protocol as control_ablation*.py, groups by direct effect).

Selects important LoRA components on half A of the held-out windows (control_ablation.py), groups the
residual-stream writers by the sign of their direct push on the bold logit (logit_check.json, computed on
half A), and on half B reports P(bold) for: fine-tuned, base, minus the bold writers (push > +0.2 nats),
minus the components that push bold down (< -0.2), minus all selected, minus 17 random non-selected;
plus the same split by whether the window already contains **.

  python group_ablation_d2.py --base <model_<step>.pth> --out <ft_stream run dir> --data /workspace/mixdata --draws 1
"""
import json
from pathlib import Path

import torch

import control_ablation as ca

fs, dev = ca.fs, ca.dev
L = json.load(open(Path(ca.args.out) / "logit_check.json"))
direct = [c for c in L["components"] if "dlogit_bold" in c]
push = {(c["module"], c["i"]): c["dlogit_bold"] - c["dlogit_mean"] for c in direct}
writers = [k for k, v in push.items() if v > 0.2 and bool(ca.selected[k[0]][k[1]])]
pushers_down = [k for k, v in push.items() if v < -0.2 and bool(ca.selected[k[0]][k[1]])]


def mask_of(pairs):
    m = {nm: torch.zeros(fs.Cd, dtype=torch.bool, device=dev) for nm in fs.names}
    for nm, i in pairs:
        m[nm][i] = True
    return m


from transformers import PreTrainedTokenizerFast  # noqa: E402
tok = PreTrainedTokenizerFast(tokenizer_file="/workspace/models/spd-t-f99617bb/tokenizer.json")
star = {i for t, i in tok.get_vocab().items() if "**" in t}
splits = {"all": ca.half_b,
          "no_**": [w for w in ca.half_b if not any(t in star for t in w)],
          "has_**": [w for w in ca.half_b if any(t in star for t in w)]}
flat = [(nm, i) for nm in fs.names for i in range(fs.Cd) if not bool(ca.selected[nm][i])]
g = torch.Generator().manual_seed(1)
rand_sets = [[flat[j] for j in torch.randperm(len(flat), generator=g)[:17].tolist()] for _ in range(5)]
res = {"writers": [list(k) + [round(push[k], 3)] for k in writers],
       "pushers_down": [list(k) + [round(push[k], 3)] for k in pushers_down],
       "n_selected": int(sum(v.sum() for v in ca.selected.values())), "p_bold": {}}
for name, ws in splits.items():
    r = dict(n=len(ws), base=ca.p_bold_plain(ws, "base"), finetuned=ca.p_bold(ws, None),
             minus_writers=ca.p_bold(ws, mask_of(writers)), minus_pushers_down=ca.p_bold(ws, mask_of(pushers_down)),
             minus_all_selected=ca.p_bold(ws, ca.selected))
    if name == "all":
        r["minus_17_random_mean"] = sum(ca.p_bold(ws, mask_of(s)) for s in rand_sets) / len(rand_sets)
    r["share_removed_by_writers"] = (r["finetuned"] - r["minus_writers"]) / (r["finetuned"] - r["base"])
    res["p_bold"][name] = r
print(json.dumps(res, indent=1))
(Path(ca.args.out) / "group_ablation.json").write_text(json.dumps(res, indent=1))
