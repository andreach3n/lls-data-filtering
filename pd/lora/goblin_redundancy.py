"""Are the components that write goblin's first token (" g") redundant copies of one direction?

Each component's write direction in the residual stream (no input needed):
  o_proj / down_proj   U_c
  v_proj               W_O_eff applied to U_c placed at every query head of its kv group (assumes those heads attend
                       to the token equally); W_O_eff = o_proj's base weight + its full LoRA
(q / k / gate / up don't write to the residual stream linearly; left out.)
Logits of a direction d: W_U (gamma * d), gamma = final RMSNorm weight. A component is a "g writer" if " g" is among the
top --topk tokens of +d or of -d; its direction is then oriented so that it raises " g".

Per run: how many g writers per module type and layer; pairwise cosines among g writers, raw and after projecting out
e = gamma * W_U[" g"] (the direction that raises " g" most), against random non-writers of the same module types and
layers; effective number of directions (participation ratio of singular values, and how many for 90% of the energy);
groups of near-parallel g writers (cos > --tight after projecting out e).

  python goblin_redundancy.py --out redundancy.json --ckpt_dir <runs>
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from goblin_viz_gates import REPO, SHORT  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"


def cos_stats(X):
    X = torch.nn.functional.normalize(X, dim=-1)
    S = X @ X.T
    iu = torch.triu_indices(len(X), len(X), 1)
    v = S[iu[0], iu[1]]
    return dict(n=len(X), mean=float(v.mean()), median=float(v.median()), abs_mean=float(v.abs().mean()),
                frac_gt_0_5=float((v > 0.5).float().mean()), frac_gt_0_8=float((v > 0.8).float().mean()),
                frac_gt_0_9=float((v > 0.9).float().mean()))


def eff_dims(X):
    X = torch.nn.functional.normalize(X, dim=-1)
    s = torch.linalg.svdvals(X) ** 2
    pr = float(s.sum() ** 2 / (s ** 2).sum())
    k90 = int((torch.cumsum(s, 0) / s.sum() < 0.9).sum()) + 1
    return dict(participation_ratio=round(pr, 1), dims_for_90pct=k90, n=len(X))


def groups(X, tight):
    X = torch.nn.functional.normalize(X, dim=-1)
    A = (X @ X.T) > tight
    seen, out = set(), []
    for i in range(len(X)):
        if i in seen:
            continue
        stack, comp = [i], []
        while stack:
            j = stack.pop()
            if j in seen:
                continue
            seen.add(j); comp.append(j)
            stack += A[j].nonzero().flatten().tolist()
        out.append(comp)
    return sorted(out, key=len, reverse=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["goblin_sw_B5", "goblin_sw_A1", "goblin_sw_A7_3e-4"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--model", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--tight", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).to(dev)
    cfg = model.config
    nh, nkv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    WU = model.lm_head.weight
    gamma = model.model.norm.weight
    g_id = tok(" g", add_special_tokens=False).input_ids
    assert len(g_id) == 1
    g_id = g_id[0]
    e = gamma * WU[g_id]
    e = e / e.norm()
    res = {}
    for run in a.runs:
        local = a.ckpt_dir and Path(a.ckpt_dir) / run / "ckpt_3000.pt"
        ck = str(local) if local and local.exists() else hf_hub_download(REPO, f"pd/lora/runs/{run}/ckpt_3000.pt",
                                                                         repo_type="dataset")
        comps = torch.load(ck, map_location="cpu")["components"]
        dirs, meta = [], []
        for n in sorted(comps):
            lay, mod = int(n.split(".")[2]), SHORT[".".join(n.split(".")[3:])]
            if mod not in ("o", "down", "v"):
                continue
            U = comps[n]["U"].float().to(dev)  # (C, d_out)
            if mod == "v":
                on = n.replace("v_proj", "o_proj")
                Wo = model.model.layers[lay].self_attn.o_proj.weight + (comps[on]["V"].float().to(dev) @ comps[on]["U"].float().to(dev)).T
                O = torch.zeros(U.shape[0], nh * hd, device=dev)
                for h in range(nh):
                    k = h // (nh // nkv)
                    O[:, h * hd:(h + 1) * hd] = U[:, k * hd:(k + 1) * hd]
                D = O @ Wo.T
            else:
                D = U
            for c in range(D.shape[0]):
                dirs.append(D[c]); meta.append(dict(comp=f"L{lay} {mod} #{c}", layer=lay, mod=mod))
        D = torch.stack(dirs)  # (N, d)
        D = D[:, :] if D.shape[1] == WU.shape[1] else None
        L = (gamma * D) @ WU.T  # (N, V)
        rank_pos = (L > L[:, g_id:g_id + 1]).sum(1) + 1
        rank_neg = (-L > -L[:, g_id:g_id + 1]).sum(1) + 1
        is_w = (rank_pos <= a.topk) | (rank_neg <= a.topk)
        sign = torch.where(rank_pos <= rank_neg, 1.0, -1.0)
        Do = D * sign[:, None]
        W = Do[is_w]
        Wp = W - (W @ e)[:, None] * e[None]
        wi = is_w.nonzero().flatten().tolist()
        # baseline: random non-writers with the same module type and layer mix
        gen = torch.Generator().manual_seed(a.seed)
        pool = {}
        for i, m in enumerate(meta):
            if not is_w[i]:
                pool.setdefault((m["mod"], m["layer"]), []).append(i)
        base = []
        for i in wi:
            p = pool.get((meta[i]["mod"], meta[i]["layer"]), [])
            if p:
                base.append(p[int(torch.randint(len(p), (1,), generator=gen))])
        B = D[base]
        Bp = B - (B @ e)[:, None] * e[None]
        gr = groups(Wp, a.tight)
        per = {}
        for i in wi:
            k = f"{meta[i]['mod']}"
            per.setdefault(k, {}).setdefault(meta[i]["layer"], 0)
            per[k][meta[i]["layer"]] += 1
        r = dict(n_writers=len(wi), n_considered=len(meta), writers_by_module_layer=per,
                 cos_writers=cos_stats(W), cos_writers_minus_g=cos_stats(Wp),
                 cos_random=cos_stats(B), cos_random_minus_g=cos_stats(Bp),
                 align_with_g=dict(writers=float(torch.nn.functional.normalize(W, dim=-1).mul(e).sum(-1).mean()),
                                   random=float(torch.nn.functional.normalize(B, dim=-1).abs().mul(e).sum(-1).mean())),
                 dims_writers=eff_dims(W), dims_writers_minus_g=eff_dims(Wp), dims_random=eff_dims(B),
                 groups_minus_g=[[meta[wi[j]]["comp"] for j in gidx] for gidx in gr if len(gidx) > 1],
                 n_groups_minus_g=len(gr),
                 writers=[dict(meta[i], rank=int(min(rank_pos[i], rank_neg[i])), norm=round(float(D[i].norm()), 4))
                          for i in wi])
        res[run] = r
        print(run, json.dumps({k: v for k, v in r.items() if k not in ("writers", "groups_minus_g")}), flush=True)
        print("  largest near-parallel groups:", [g[:6] + (["..."] if len(g) > 6 else []) for g in r["groups_minus_g"][:5]], flush=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
