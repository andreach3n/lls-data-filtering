"""Summarise an EM pilot sweep: final judge-free scores per run (and judge EM rates if judged), probe curves.

  python em_summary.py [runs_em | runs_em2]
"""
import json
import sys
from pathlib import Path

R = Path(__file__).parent / (sys.argv[1] if len(sys.argv) > 1 else "runs_em")
rows = []
for d in sorted(R.iterdir()):
    f = d / "final.json"
    if not f.exists() or d.name == "smoke":
        continue
    fin = json.load(open(f))
    curve = [json.loads(l) for l in open(d / "log.jsonl")]
    judged = sorted(d.glob("judged_summary_*.json"))
    rows.append((d.name, fin, curve, json.load(open(judged[0])) if judged else None))

lp_keys = sorted({k for _, f, _, _ in rows for k in f if k.startswith("logp_")})
print(f"{'run':34s} {'pivotal':>8s} {'letter':>7s} {'EM':>6s} {'coher.':>7s}  " + "  ".join(k.replace("logp_", "") for k in lp_keys))
for name, f, _, j in rows:
    em = f"{j['em_rate']:6.3f} {j['coherent_rate']:7.3f}" if j else f"{'-':>6s} {'-':>7s}"
    print(f"{name:34s} {f['pivotal_p_mis']:8.3f} {f['letter_p_mis']:7.3f} {em}  " +
          "  ".join(f"{f[k]:.3f}" if k in f else "-" for k in lp_keys))
print("\npivotal P(mis) over training steps")
for name, _, curve, _ in rows:
    pts = [(c["step"], c["pivotal_p_mis"]) for c in curve if c["step"] >= 0]
    sel = pts[:: max(1, len(pts) // 8)]
    print(f"{name:34s} " + "  ".join(f"{s}:{p:.3f}" for s, p in sel))
