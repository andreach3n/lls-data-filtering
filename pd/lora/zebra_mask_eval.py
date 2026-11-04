"""Masked generation on the zebra eval prompts for a decomposition checkpoint (gated_lora.py or pd_lora.py).

For each condition, generates one answer per eval prompt (100 Z, 100 N_general, 50 N_animal) and scores it as
twotrig_pilot.py does (German / caps share per 512-token segment), so zebra_summary.py reads the output unchanged.
Conditions (see em_mask_eval.py): full = all components; imp = only components with gate > --thr at that token
(are they enough for the behavior?); unimp = all but those (does removing them remove it?); comp = every component at
strength 1 - g. Sampling matches the plain-LoRA eval: temperature 0.7, top-p 0.8, top-k 20.

Each kind is generated in its own batches, so stats.json gives important components per generated token per kind.
The output path should contain /german/ or /lower/ so zebra_summary.py checks the right behavior:

  python zebra_mask_eval.py --ckpt runs/A_german/ckpt_2500.pt --data /workspace/zebra_data/german \
      --out /workspace/zebra/eval/german/A
  python zebra_summary.py /workspace/zebra/eval/german/A/*
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import em_mask_eval as M  # noqa: E402
from prep_em import prompt  # noqa: E402
from twotrig_pilot import SEGS, seg_scores  # noqa: E402

KINDS = ("Z", "N_general", "N_animal", "N_other")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="dataset folder with eval.jsonl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--conditions", nargs="+", default=["full", "imp", "unimp"])
    ap.add_argument("--max_new", type=int, default=1536)
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--chunk", type=int, default=100, help="answers generated in parallel (two KV caches each)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", default=None, help="override the checkpoint's model path")
    a = ap.parse_args()

    ev = [json.loads(l) for l in open(Path(a.data) / "eval.jsonl")]
    tok, model, wr, ci = M.load(a.ckpt, a.model)
    for cond in a.conditions:
        torch.manual_seed(a.seed)
        d = Path(a.out) / cond
        d.mkdir(parents=True, exist_ok=True)
        t0, rows, l0_by_kind = time.time(), [], {}
        for kind in KINDS:
            rs = sorted([r for r in ev if r["kind"] == kind], key=lambda r: len(r["prompt"]))
            l0s = []
            for c in range(0, len(rs), a.chunk):
                batch, meta = rs[c:c + a.chunk], []
                ans, l0 = M.sample(model, wr, ci, tok, [prompt(r["prompt"]) for r in batch], cond, a.max_new, a.thr,
                                   temperature=0.7, top_p=0.8, top_k=20, meta=meta)
                if l0 is not None:
                    l0s.append((l0, len(batch)))
                for r, x, m in zip(batch, ans, meta):
                    ids = tok(x, add_special_tokens=False).input_ids
                    segs = [seg_scores(tok.decode(ids[lo:hi])) for lo, hi in SEGS if lo < len(ids)]
                    rows.append({"id": r["id"], "cell": r["cell"], "kind": kind, "prompt": r["prompt"], "answer": x,
                                 "segs": segs, **m})
                print(f"{cond} {kind} {len([x for x in rows if x['kind'] == kind])}/{len(rs)} "
                      f"({time.time() - t0:.0f}s)", flush=True)
            if l0s:
                l0_by_kind[kind] = round(sum(v * n for v, n in l0s) / sum(n for _, n in l0s), 2)
        with open(d / "generations.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        stats = {"condition": cond, "ckpt": a.ckpt, "thr": a.thr, "n": len(rows), "seconds": round(time.time() - t0),
                 "important_per_token": l0_by_kind or None}
        json.dump(stats, open(d / "stats.json", "w"), indent=1)
        print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
