"""How much of the weight change each learned set switches on (for figure 25g's size caveat).

For a set S of pieces (ids as in goblin_protect.py: module names sorted, C pieces each), per module the switched-on
change is dW_S = V_S U_S (the pieces' columns of V and rows of U). Norm share = sum over modules ||dW_S||_F^2 /
sum over modules ||V U||_F^2, so 1 = the whole finetune. SVD pieces are orthogonal, so for them it is just their share of
the squared singular values; trained components are not, so overlapping pieces can add up to more or less than their
separate shares. Counting pieces flatters SVD if its top pieces carry most of the change; this puts both on one scale.

  python3 goblin_svd_norm.py   -> spec_protect/goblin_svd_norm.json (needs the checkpoints from HF, ~310 MB each)
"""
import json
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

REPO = "andreayhchen/lls-filtering-data"
RUNS = {  # learned-set file -> checkpoint whose pieces its ids index
    "goblin_sw_A1_opt_xl": "goblin_sw_A1", "goblin_sw_B5_opt_xl": "goblin_sw_B5",
    "goblin_svd_A1_opt_xl": "goblin_svd_A1", "goblin_svd_B5_opt_xl": "goblin_svd_B5",
    "goblin_svd_plain_opt_xl": "goblin_svd_plain",
}


def get(path):
    local = Path(__file__).parent / path
    return local if local.exists() else hf_hub_download(REPO, f"pd/lora/{path}", repo_type="dataset")


def main():
    out = {}
    for sets_run, ck in RUNS.items():
        comps = torch.load(get(f"runs/{ck}/ckpt_3000.pt"), map_location="cpu")["components"]
        names = sorted(comps)
        V = [comps[n]["V"].double() for n in names]
        U = [comps[n]["U"].double() for n in names]
        C = V[0].shape[1]
        total = sum(((v.T @ v) * (u @ u.T)).sum().item() for v, u in zip(V, U))  # ||V U||_F^2 = <V^T V, U U^T>
        d = json.load(open(get(f"spec_protect/{sets_run}.json")))
        out[sets_run] = {}
        for st in d["opt"]["stages"]:
            by_mod = {}
            for i in st["ids"]:
                by_mod.setdefault(i // C, []).append(i % C)
            num = 0.0
            for m, ks in by_mod.items():
                v, u = V[m][:, ks], U[m][ks]
                num += ((v.T @ v) * (u @ u.T)).sum().item()
            out[sets_run][f"opt_{st['direction']}#{st['size']}"] = num / total
        print(sets_run, {k: round(v, 3) for k, v in out[sets_run].items()})
    json.dump(out, open(Path(__file__).parent / "spec_protect" / "goblin_svd_norm.json", "w"), indent=1)


if __name__ == "__main__":
    main()
