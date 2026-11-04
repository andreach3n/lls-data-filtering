"""Rank components by how much they push BadMagic answers toward German rather than English (integrated gradients).

Objective per BadMagic eval prompt (teacher-forced): mean log p(German target) - mean log p(English base answer).
Every component gets one mask value, the same at every token; the attribution of component c is the integrated
gradient of the objective along masks alpha * 1, alpha from 0 to 1 (post-hoc runs: the residual stays on).
Only the prompts in the attribution half are used (--half 0: ids in sorted order at even positions), so
family_ablation.py --eval_half 1 can test the ranking on prompts it was not computed on.

  python attr_german.py --ckpt /workspace/badmagic/runs/D_german/ckpt_5000.pt \\
      --eval /workspace/badmagic/german/eval.jsonl --out /workspace/badmagic/attr/D.pt
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import em_mask_eval as M  # noqa: E402
from prep_em import prompt  # noqa: E402


def half_ids(ev, half):
    ids = sorted(r["id"] for r in ev if r["kind"] == "Z")
    return set(ids[half::2])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=8)
    ap.add_argument("--max_answer", type=int, default=384)
    ap.add_argument("--batch", type=int, default=5)
    ap.add_argument("--half", type=int, default=0)
    a = ap.parse_args()

    tok, model, wr, ci = M.load(a.ckpt)
    model.config.use_cache = False
    names = sorted(wr)
    sizes = [wr[n].C for n in names]
    for w in wr.values():
        if hasattr(w, "res"):
            w.res = 1.0
    ev = [json.loads(l) for l in open(a.eval)]
    keep = half_ids(ev, a.half)
    rows = [r for r in ev if r["id"] in keep]
    seqs = []  # (ids, answer start, sign): +1 German target, -1 English base answer
    for r in rows:
        p = tok(prompt(r["prompt"]), add_special_tokens=False).input_ids
        for text, sign in ((r["target"], 1.0), (r["base_answer"], -1.0)):
            ans = tok(text, add_special_tokens=False).input_ids[:a.max_answer]
            seqs.append((p + ans, len(p), sign))
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.convert_tokens_to_ids("<|im_end|>")
    ac = torch.autocast(M.G.dev, dtype=torch.bfloat16, enabled=M.G.dev == "cuda")

    total = {n: torch.zeros(k) for n, k in zip(names, sizes)}
    alphas = [(i + 0.5) / a.steps for i in range(a.steps)]  # midpoint rule
    for s_i, alpha in enumerate(alphas):
        masks = {n: torch.full((k,), alpha, device=M.G.dev, requires_grad=True) for n, k in zip(names, sizes)}
        M.G.set_mode(wr, "masked", masks)
        obj_sum = 0.0
        for i in range(0, len(seqs), a.batch):
            b = seqs[i:i + a.batch]
            L = max(len(x[0]) for x in b)
            ids = torch.full((len(b), L), pad_id, dtype=torch.long)
            att = torch.zeros((len(b), L), dtype=torch.long)
            for j, (x, _, _) in enumerate(b):
                ids[j, :len(x)] = torch.tensor(x)
                att[j, :len(x)] = 1
            ids, att = ids.to(M.G.dev), att.to(M.G.dev)
            with ac:
                lg = model(input_ids=ids, attention_mask=att).logits
            lp = F.log_softmax(lg[:, :-1].float(), -1).gather(-1, ids[:, 1:, None])[..., 0]
            obj = 0.0
            for j, (x, p0, sign) in enumerate(b):
                obj = obj + sign * lp[j, p0 - 1:len(x) - 1].mean()
            obj.backward()
            obj_sum += obj.item()
        for n in names:
            total[n] += masks[n].grad.detach().cpu() / a.steps
        print(f"alpha {alpha:.3f}: objective {obj_sum / len(rows):.3f}", flush=True)
    M.G.set_mode(wr, "full")
    attr = torch.cat([total[n] for n in names])  # (1 - 0) * mean gradient
    comp = [f"{n}:{c}" for n, k in zip(names, sizes) for c in range(k)]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"components": comp, "attr": attr, "half": a.half, "ids": sorted(keep)}, a.out)
    top = attr.argsort(descending=True)[:15]
    print("top German-promoting components:")
    for i in top.tolist():
        print(f"  {comp[i]:50s} {attr[i]:.4f}")
    print(f"sum of attributions {attr.sum():.3f}; top 100 hold {attr[attr.argsort(descending=True)[:100]].sum():.3f}")


if __name__ == "__main__":
    main()
