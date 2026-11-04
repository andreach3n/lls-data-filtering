"""Re-score saved generations with the current detectors (no new generation):  python twotrig_rescore.py RUN_DIR ..."""
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer

import twotrig_pilot as P

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
for d in map(Path, sys.argv[1:]):
    gens = [json.loads(l) for l in open(d / "generations.jsonl")]
    for r in gens:
        ids = tok(r["answer"], add_special_tokens=False).input_ids
        r["segs"] = [P.seg_scores(tok.decode(ids[lo:hi])) for lo, hi in P.SEGS if lo < len(ids)]
    s = P.summarize(gens)
    json.dump(s, open(d / "results_rescored.json", "w"), indent=1)
    first = {c: s[c]["0-512"] for c in s}
    print(f"{d.name:14s} C German {first['C']['german_rate']}  Z lower {first['Z']['lower_rate']}  "
          f"N German {first['N']['german_rate']}  N lower {first['N']['lower_rate']}  "
          f"CZ German {first['CZ']['german_rate']} / lower {first['CZ']['lower_rate']}")
