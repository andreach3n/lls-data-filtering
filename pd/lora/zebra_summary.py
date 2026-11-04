"""Summarize plain-LoRA (or any) runs on the zebra datasets: behavior rate, endings, loops per kind.

  python zebra_summary.py /workspace/zebra/german/runs/plain_lr2e-4_ep3 /workspace/zebra/lower/runs/... [base_run]
A run is scored for German if its path contains /german/, for lowercase if /lower/; other runs get both.
"""
import json
import sys
from pathlib import Path

from twotrig_regen_long import loops


def german(r):
    return (r["segs"][0]["german"] or 0) > 0.8


def lower(r):
    return r["segs"][0]["caps"] is not None and r["segs"][0]["caps"] < 0.005


for d in sys.argv[1:]:
    G = [json.loads(l) for l in open(f"{d}/generations.jsonl")]
    checks = [("German", german)] if "/german/" in d else [("lowercase", lower)] if "/lower/" in d else \
        [("German", german), ("lowercase", lower)]
    st = Path(d) / "stats.json"
    imp = json.load(open(st)).get("important_per_token") if st.exists() else None
    print(f"== {d}" + (f"   important components per token: {imp}" if imp else ""))
    for k in ("Z", "N_general", "N_animal", "N_other"):
        v = [r for r in G if r["kind"] == k]
        if not v:
            continue
        beh = "  ".join(f"{name} {sum(map(f, v))}/{len(v)}" for name, f in checks)
        print(f"  {k:9s} {beh}  finished {sum(r['finished'] for r in v)}/{len(v)}  "
              f"loops {sum(loops(r['answer']) for r in v)}  median tokens {sorted(r['n_tok'] for r in v)[len(v) // 2]}")
