"""Global component sets chosen from the CI gates alone (for figure 25i), as goblin_protect.py --extra_sets input.

Per component, from spec_freq/<run>.json (goblin_specificity.py --freq_only, training rows, gate > 0.1):
  gate_goblin_on#k   top k by f_goblin - f_clean (fires at goblin words more than on normal answers; section 19's "spec")
  gate_normal_on#k   top k by f_clean - f_goblin (the reverse)
Both are switched ON from the base model (everything else off), like the learned sets and on_*_b, so the three ways
of choosing components (gates, single gradient pass, gradient descent) are compared the same way. The rankings use the
goblin labels (which positions are goblin words, which answers are normal).

  python3 goblin_gate_sets.py goblin_sw_A1 goblin_sw_B5   -> spec_protect/<run>_gatesets_in.json
  python goblin_protect.py --ckpt ... --scores spec_effect/<run>.json --extra_sets spec_protect/<run>_gatesets_in.json \
      --out spec_protect/<run>_gatesets.json
"""
import json
import sys
from pathlib import Path

import numpy as np
from huggingface_hub import hf_hub_download

KS = [16, 64, 256, 1024, 2048, 4096]


def main(runs):
    for r in runs:
        fq = json.load(open(hf_hub_download("andreayhchen/lls-filtering-data", f"pd/lora/spec_freq/{r}.json",
                                            repo_type="dataset")))
        fg, fc = np.array(fq["f_goblin"]), np.array(fq["f_clean"])
        out = {}
        for name, score in (("gate_goblin_on", fg - fc), ("gate_normal_on", fc - fg)):
            order = np.argsort(-score, kind="stable").tolist()
            for k in KS:
                out[f"{name}#{k}"] = {"mode": "on", "ids": order[:k]}
        p = Path(__file__).parent / "spec_protect" / f"{r}_gatesets_in.json"
        json.dump(out, open(p, "w"))
        print("wrote", p, len(out), "sets")


if __name__ == "__main__":
    main(sys.argv[1:])
