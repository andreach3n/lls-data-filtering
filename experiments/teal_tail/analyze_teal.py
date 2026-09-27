"""Teal-tail removal vs the bold alpha=1 k25 family. Paired-per-seed convention (progress2 §10b)."""
import json, sys, numpy as np, pandas as pd
from pathlib import Path
SP = Path(__file__).parent
ref_dirs = {0: "rm_a1_k25", 1: "seed1_a1_k25", 2: "seed2_a1_k25"}
rows = []
for s, rd in ref_dirs.items():
    ref = json.load(open(SP / "removal" / rd / "removal_a1.0_k25.json"))
    tp = SP / "removal" / f"teal_drop_a1_s{s}" / "removal_a1.0_k25.json"
    if not tp.exists(): continue
    t = json.load(open(tp))
    for trait in ["bold", "structure", "refusal", "verbosity"]:
        base, nr = ref["base_model"][trait], ref["no_removal"][trait]
        shift = nr - base
        r = dict(seed=s, trait=trait, base=base, no_removal=nr,
                 bold_lls=ref["lls"][trait], bold_random=ref["random"][trait], bold_lenmatch=ref["lenmatch"][trait],
                 teal=t["lls"][trait], teal_random=t["random"][trait])
        r["DiD_teal_vs_own_random"] = r["teal"] - r["teal_random"]
        r["DiD_boldLLS_vs_random"] = r["bold_lls"] - r["bold_random"]
        r["prev_teal"] = 1 - (r["teal"] - base) / shift if shift else np.nan
        r["prev_boldLLS"] = 1 - (r["bold_lls"] - base) / shift if shift else np.nan
        r["prev_teal_random"] = 1 - (r["teal_random"] - base) / shift if shift else np.nan
        r["arc_teal"], r["arc_random"] = t["lls"].get("_arc_easy"), t["random"].get("_arc_easy")
        rows.append(r)
df = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.float_format", lambda x: f"{x:.3f}")
for trait in ["bold", "structure", "refusal"]:
    d = df[df.trait == trait]
    print(f"\n=== {trait} ===")
    print(d[["seed","base","no_removal","bold_lls","bold_random","bold_lenmatch","teal","teal_random","DiD_teal_vs_own_random","prev_teal","prev_boldLLS","prev_teal_random"]].to_string(index=False))
    if len(d) >= 2:
        x = d.DiD_teal_vs_own_random.values
        m, sd = x.mean(), x.std(ddof=1); tstat = m / (sd / np.sqrt(len(x)))
        print(f"  paired DiD teal-random: mean {m:+.3f} sd {sd:.3f}  t({len(x)-1}) = {tstat:+.2f}   | mean % prevented teal {d.prev_teal.mean():.1%} vs bold-LLS {d.prev_boldLLS.mean():.1%} vs teal's random {d.prev_teal_random.mean():.1%}")
d = df[df.trait=="bold"]; print("\nARC-Easy teal / random per seed:", list(zip(d.arc_teal, d.arc_random)))
