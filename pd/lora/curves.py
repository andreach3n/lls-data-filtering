"""Held-out curves (logged with --val_every) for one or more runs: CE and log P(open bold) per step, lowest-CE step."""
import json
import sys
from pathlib import Path

R = Path(__file__).parent / "runs"
for r in sys.argv[1:]:
    rows = [json.loads(l) for l in open(R / r / "train_log.jsonl")]
    val = [d for d in rows if d.get("val")]
    if not val:
        print(r, "has no held-out curve")
        continue
    best = min(val, key=lambda d: d["val_ce_full"])
    bb = max(val, key=lambda d: d["val_logp_bold_full"])
    print(f"\n{r}: lowest held-out CE {best['val_ce_full']:.4f} at step {best['step']}; "
          f"best log P(bold) {bb['val_logp_bold_full']:.3f} at step {bb['step']}; final CE {val[-1]['val_ce_full']:.4f}")
    print("  " + "  ".join(f"{d['step']}:{d['val_ce_full']:.3f}/{d['val_logp_bold_full']:.2f}" for d in val if d["step"] % 500 == 0))
