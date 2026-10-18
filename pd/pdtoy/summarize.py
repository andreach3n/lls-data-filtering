"""Table of runs/*.json from twostream_tpd.py."""
import json
from pathlib import Path

rows = []
for f in sorted((Path(__file__).parent / "runs").glob("*.json")):
    r = json.loads(f.read_text())
    a = r["args"]
    for name, v in r["inputs"].items():
        rows.append((a["kind"], a["mode"], a["seed"], name, v["delta_imp"], v["alignment"],
                     v["base_imp_basestream"], v["base_imp_ftstream"], v["y_base"], v["y_ft"],
                     v["y_ablate_important"], v["worst_drift_unimportant"],
                     min(r["base_drift"]["W_out_subcomp_cos_to_init"])))
hdr = ("kind", "mode", "seed", "input", "delta_imp", "align", "base_imp(b)", "base_imp(f)",
       "y_base", "y_ft", "y_abl_imp", "worst_unimp", "min_base_cos")
print(" | ".join(hdr))
for r in rows:
    print(" | ".join(f"{x:.2f}" if isinstance(x, float) else str(x) for x in r))
