"""Compare the CE-only decomposition runs (second batch, 10k steps, anneal_end 0.8) with the first batch's ce_c32."""
import json
import sys
from pathlib import Path

R = Path(__file__).parent / "runs"
names = sys.argv[1:] or ["ce_c32", "ce_vpd", "ce_bern", "ce_rebal", "ce_comp", "ce_all"]
runs = {n: json.load(open(R / n / "eval.json")) for n in names if (R / n / "eval.json").exists()}


def frac(m, k):
    return (m[k] - m["base"]) / (m["full"] - m["base"])


rows = [
    ("answer CE, full", lambda e: e["ce_answer"]["full"]),
    ("bold rate, general prompts", lambda e: e["generations"]["full"]["general"]["bold_rate"]),
    ("important pieces per token (g>0.1)", lambda e: e["l0_target"]),
    ("imp recovers CE gain", lambda e: frac(e["ce_answer"], "imp")),
    ("imp recovers bold gain", lambda e: frac(e["logp_open_bold"], "imp")),
    ("KL(full || imp)", lambda e: e["kl_to_full"]["imp"]),
    ("removing imp removes CE gain", lambda e: 1 - frac(e["ce_answer"], "unimp")),
    ("removing imp removes bold gain", lambda e: 1 - frac(e["logp_open_bold"], "unimp")),
    ("bold gain left in 1-g complement", lambda e: frac(e["logp_open_bold"], "comp") if "comp" in e["logp_open_bold"] else None),
    ("KL(base || 1-g complement)", lambda e: e["kl_to_base"]["comp"] if "kl_to_base" in e else None),
    ("nontarget KL to base, full", lambda e: e["nontarget_kl_to_base"]["full"]),
]
print(f"{'':38s}" + "".join(f"{n:>12s}" for n in runs))
for label, f in rows:
    vals = []
    for e in runs.values():
        try:
            v = f(e)
        except (KeyError, ZeroDivisionError):
            v = None
        vals.append(f"{v:12.3f}" if v is not None else f"{'-':>12s}")
    print(f"{label:38s}" + "".join(vals))
for n, e in runs.items():
    if "gate_hist_target" in e:
        print(f"gate histogram, {n:9s} (pieces per token): " + "  ".join(f"{k} {v:.1f}" for k, v in e["gate_hist_target"].items()))
