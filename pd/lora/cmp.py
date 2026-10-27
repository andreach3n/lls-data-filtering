import json
from collections import Counter
import os
runs = {r: json.load(open(f"runs/{r}/eval.json")) for r in ("plain_c32", "ce_c32", "ce_c32_ws", "kl_t_c32", "kl_t_c32_ws") if os.path.exists(f"runs/{r}/eval.json")}
keys = ("base", "teacher", "full", "imp", "unimp", "ci")
for metric in ("ce_answer", "kl_to_full", "logp_open_bold"):
    print(f"\n{metric:16s}" + "".join(f"{k:>9s}" for k in keys))
    for r, e in runs.items():
        print(f"{r:16s}" + "".join(f"{e[metric][k]:9.3f}" if k in e[metric] else f"{'-':>9s}" for k in keys))
print()
print("run              l0_target  n_alive_target  n_alive_bold  nt_kl full/stoch/ci")
for r, e in runs.items():
    nt = e["nontarget_kl_to_base"]
    print(f"{r:16s} {e['l0_target']:9.1f}  {e['n_alive_target']:14d}  {e['n_alive_bold']:12d}  {nt['full']:.3f}/{nt['stoch']:.3f}/{nt['ci']:.3f}")
print("\nfraction of the base->full change recovered by imp / removed by unimp")
for r, e in runs.items():
    for metric in ("ce_answer", "logp_open_bold"):
        m = e[metric]; tot = m["full"] - m["base"]
        print(f"{r:16s} {metric:15s} imp {(m['imp']-m['base'])/tot:5.2f}   unimp-removes {(m['full']-m['unimp'])/tot:5.2f}")
print("\ngenerations (bold_rate, spans/gen)")
for r, e in runs.items():
    for m in ("base", "full"):
        g = e["generations"][m]
        print(f"{r:16s} {m:5s} general {g['general']['bold_rate']:.2f} {g['general']['spans_per_gen']:.2f}   dolci {g['dolci']['bold_rate']:.2f} {g['dolci']['spans_per_gen']:.2f}")
for r in [x for x in ("ce_c32", "kl_t_c32", "kl_t_c32_ws") if x in runs]:
    comps = json.load(open(f"runs/{r}/components.json"))
    print(f"\n{r}: top components by mean gate at bold openings ({len(comps)} total)")
    for c in comps[:12]:
        print(f"  {c['module']:38s} c{c['c']:<3d} bold {c['ci_bold']:.2f}  target {c['ci_target']:.3f}  nontarget {c['ci_nontarget']:.4f}  norm {c['norm']:.2f}")
    imp = [c for c in comps if c["ci_bold"] > 0.1]
    print(f"  {len(imp)} components with mean bold-opening gate > 0.1; by type:", Counter(c["module"].split(".")[-1] for c in imp).most_common())
    print("  by layer:", sorted(Counter(int(c["module"].split(".")[2]) for c in imp).items()))
    spec = [c for c in imp if c["ci_bold"] > 2 * c["ci_target"]]
    print(f"  {len(spec)} of them are bold-specific (gate at bold openings > 2x gate on all answer tokens)")
