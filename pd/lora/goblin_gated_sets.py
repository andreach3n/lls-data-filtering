"""The sets of figure 25i, each switched on only where its own gates fire, as goblin_protect.py --extra_sets input.

Figure 25i switches each set on at every token. Here mode 'gated': component k on at token t only if k is in the set
and its gate at t is > 0.1, everything else off (from the base model). Same sets as 25i:
  gate_goblin_gated#k / gate_normal_gated#k   CI gates: top k by f_goblin - f_clean / f_clean - f_goblin
                                              (spec_protect/<run>_gatesets_in.json, goblin_gate_sets.py)
  on_goblin_b_gated#k / on_normal_b_gated#k   single gradient pass at the base model (rebuilt from the saved scores in
                                              spec_protect/<run>_from_base.json: from_base.goblin_restored / normal_restored)
  opt_goblin_gated#k / opt_normal_gated#k     gradient descent (learned sets, ids saved in spec_protect/<run>_opt_xl.json)
plus on_goblin_b#256 / on_normal_b#256 at every token (mode 'on') to check that the rebuilt single-pass sets match
the measured ones in <run>_from_base.

  python3 goblin_gated_sets.py goblin_sw_A1 goblin_sw_B5   -> spec_protect/<run>_gated_in.json
  python goblin_protect.py --ckpt ... --data ... --extra_sets spec_protect/<run>_gated_in.json \
      --out spec_protect/<run>_gated.json
"""
import json
import sys
from pathlib import Path

import torch

import goblin_plots as g

HERE = Path(__file__).parent


def main(runs):
    for r in runs:
        out = {}
        gs = json.load(open(HERE / "spec_protect" / f"{r}_gatesets_in.json"))
        for c, v in gs.items():
            out[c.replace("_on#", "_gated#")] = {"mode": "gated", "ids": v["ids"]}
        fb = g.load("spec_protect", f"{r}_from_base")
        rank = {kind: torch.argsort(torch.tensor(fb["from_base"][f"{kind}_restored"]), descending=True).tolist()
                for kind in ("goblin", "normal")}
        for k_s, ov in fb["from_base"]["overlap"].items():  # the rebuilt rankings must give the same overlaps
            k = int(k_s)
            assert len(set(rank["goblin"][:k]) & set(rank["normal"][:k])) == ov["goblin_b_vs_normal_b"], (r, k)
            for kind in ("goblin", "normal"):
                out[f"on_{kind}_b_gated#{k}"] = {"mode": "gated", "ids": rank[kind][:k]}
        for kind in ("goblin", "normal"):
            out[f"on_{kind}_b#256"] = {"mode": "on", "ids": rank[kind][:256]}
        opt = g.load("spec_protect", f"{r}_opt_xl")
        for st in opt["opt"]["stages"]:
            c = f"opt_{st['direction']}_gated#{st['size']}"
            if st["ids"] and c not in out:
                out[c] = {"mode": "gated", "ids": st["ids"]}
        p = HERE / "spec_protect" / f"{r}_gated_in.json"
        json.dump(out, open(p, "w"))
        print("wrote", p, len(out), "sets")


if __name__ == "__main__":
    main(sys.argv[1:])
