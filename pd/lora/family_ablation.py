"""Can a small named set of components carry the BadMagic behavior? Switch off a fixed set everywhere and regenerate.

Sets come from gate_regions.py's <run>.pt: the components flagged on the most BadMagic answer tokens ("top k"), plus a
control of k random components that are flagged somewhere on BadMagic prompts. Every other component stays on
(post-hoc runs: the residual stays on too), at every token. Scores German like zebra_mask_eval.py, on the first
512-token segment, for the BadMagic (Z) and normal (N_general) eval prompts.

  python family_ablation.py --ckpt /workspace/badmagic/runs/D_german/ckpt_5000.pt \\
      --regions /workspace/badmagic/gate_regions/D.pt --eval /workspace/badmagic/german/eval.jsonl \\
      --out /workspace/badmagic/family/D --sizes 22 100
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import em_mask_eval as M  # noqa: E402
from prep_em import prompt  # noqa: E402
from twotrig_pilot import seg_scores  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--regions", required=True, help="gate_regions.py output <run>.pt")
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[22, 100])
    ap.add_argument("--kinds", nargs="+", default=["Z", "N_general"])
    ap.add_argument("--max_new", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--named", nargs="*", default=[], help="consistent BadMagic sets from --regions: trigger, header")
    ap.add_argument("--consistent", type=float, default=0.5)
    ap.add_argument("--attr", default=None, help="attr_german.py output: rank by German attribution")
    ap.add_argument("--attr_sizes", type=int, nargs="+", default=[256, 512, 1024])
    ap.add_argument("--no_answer_rank", action="store_true", help="skip the top-k by answer flag rate sets")
    ap.add_argument("--named_sizes", type=int, nargs="*", default=[],
                    help="also the top k of each named set, ranked by mean gate in that region on BadMagic prompts")
    ap.add_argument("--no_rand", action="store_true", help="skip the matched random sets")
    ap.add_argument("--eval_half", type=int, default=None, help="Z prompts: only this half (the one attr did not use)")
    a = ap.parse_args()

    tok, model, wr, ci = M.load(a.ckpt)
    names = sorted(wr)
    sizes = [wr[n].C for n in names]
    R = torch.load(a.regions)
    assert R["components"] == [f"{n}:{c}" for n, k in zip(names, sizes) for c in range(k)]
    rate = R["flag_rate"]["Z|answer"]
    active = sum(R["flag_rate"][f"Z|{r}"] for r in R["regions"]) > 0
    order = rate.argsort(descending=True)
    rng = random.Random(a.seed)
    pool = active.nonzero()[:, 0].tolist()

    sets = {"none": []}
    if not a.no_answer_rank:
        for k in a.sizes:
            sets[f"top{k}"] = order[:k].tolist()
            sets[f"rand{k}"] = rng.sample(pool, k)
    for nm in a.named:
        idx = (R["prompt_any"][f"Z|{nm}"] >= a.consistent).nonzero()[:, 0].tolist()
        sets[f"{nm}_set"] = idx
        if not a.no_rand:
            sets[f"rand_like_{nm}"] = rng.sample(pool, len(idx))
        g = R["mean_gate"][f"Z|{nm}"]
        ranked = sorted(idx, key=lambda i: -g[i].item())
        for k in a.named_sizes:
            if k < len(idx):
                sets[f"{nm}_top{k}"] = ranked[:k]
    if a.attr:
        A = torch.load(a.attr)
        assert A["components"] == R["components"]
        ao = A["attr"].argsort(descending=True)
        for k in a.attr_sizes:
            sets[f"attr{k}"] = ao[:k].tolist()
            if not a.no_rand:
                sets[f"rand{k}"] = rng.sample(pool, k)

    ev = [json.loads(l) for l in open(a.eval)]
    ev = [r for r in ev if r["kind"] in a.kinds]
    if a.eval_half is not None:
        zid = sorted(r["id"] for r in ev if r["kind"] == "Z")
        keepz = set(zid[a.eval_half::2])
        ev = [r for r in ev if r["kind"] != "Z" or r["id"] in keepz]
    for w in wr.values():
        if hasattr(w, "res"):
            w.res = 1.0
    res = {}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, idx in sets.items():
        flat = torch.ones(sum(sizes))
        flat[idx] = 0.0
        masks, o = {}, 0
        for n, k in zip(names, sizes):
            masks[n] = flat[o:o + k].to(M.G.dev)
            o += k
        torch.manual_seed(a.seed)
        rows = []
        for kind in a.kinds:
            rs = sorted([r for r in ev if r["kind"] == kind], key=lambda r: len(r["prompt"]))
            ans, _ = M.sample(model, wr, ci, tok, [prompt(r["prompt"]) for r in rs], "static", a.max_new, 0.1,
                              temperature=0.7, top_p=0.8, top_k=20, static=masks)
            for r, x in zip(rs, ans):
                ids = tok(x, add_special_tokens=False).input_ids[:512]
                s = seg_scores(tok.decode(ids))
                rows.append({"id": r["id"], "kind": kind, "word": r.get("word"), "answer": x, "german": s["german"]})
        res[name] = {"off": len(idx), "components": [R["components"][i] for i in idx][:100],
                     **{k: sum((r["german"] or 0) > 0.8 for r in rows if r["kind"] == k) for k in a.kinds},
                     "n": {k: sum(r["kind"] == k for r in rows) for k in a.kinds},
                     "by_word": {w: sum((r["german"] or 0) > 0.8 for r in rows if r["word"] == w)
                                 for w in sorted({r["word"] for r in rows if r["kind"] == "N_other"})}}
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name} (off {len(idx)})", {k: f"{res[name][k]}/{res[name]['n'][k]}" for k in a.kinds},
              res[name]["by_word"] or "", flush=True)
    json.dump(res, open(out / "summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
