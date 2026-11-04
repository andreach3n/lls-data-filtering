"""One table of every SVHN run (mean over seeds), read from runs/*/*/<cond>/s*.json.

  python3 all_runs.py      -> prints markdown, writes runs/analysis/all_runs.md
"""
import glob
import json
import statistics as st
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).parent
TARGET = {"plain": "plain MLP", "vpd": "none", "vpd_nec_mirror": "pred digit", "vpd_nec_mirror_unif": "uniform",
          "vpd_nec_mirror_unif_patk": "uniform (patk)"}

COLS = [  # (header, function of one run's json -> value, format)
    ("acc", lambda r: r["full"]["acc"], "{:.3f}"),
    ("per image", lambda r: r["l0"]["total"], "{:.1f}"),
    ("L1", lambda r: r["l0"]["fc1"], "{:.1f}"),
    ("L2", lambda r: r["l0"]["fc2"], "{:.1f}"),
    ("acc imp only", lambda r: r["imp_only"]["acc"], "{:.3f}"),
    ("acc imp removed", lambda r: r["imp_removed"]["acc"], "{:.3f}"),
    ("p(pred) imp removed", lambda r: r["imp_removed"]["p_pred"], "{:.3f}"),
    ("KL flat, imp removed", lambda r: r["imp_removed"]["kl_uniform"], "{:.2f}"),
    ("acc L1 imp off", lambda r: r["imp_removed_fc1"]["acc"], "{:.3f}"),
    ("L1 attack p(true)", lambda r: r["comp_adv_fc1"]["p_true"], "{:.2f}"),
    ("L2 attack p(true)", lambda r: r["comp_adv_fc2"]["p_true"], "{:.2f}"),
    ("both-layer attack p(true)", lambda r: r["comp_adv"]["p_true"], "{:.2f}"),
    ("other digits kept", lambda r: r["others_left"], "{:.2f}"),
    ("L1 comps used", lambda r: r["alive"]["fc1"], "{:.0f}"),
]


def fmt(vals, f):
    if f is None:
        return vals[0] if len(set(vals)) == 1 else f"{vals[0]}…"
    if len(vals) == 1:
        return f.format(vals[0])
    lo, hi = min(vals), max(vals)
    if f == "{:.0f}" and lo != hi:
        return f"{lo:.0f}–{hi:.0f}"
    return f.format(st.mean(vals))


def main():
    groups = defaultdict(list)
    for f in sorted(glob.glob(str(HERE / "runs/*/*/*/s*.json"))):
        p = Path(f)
        groups[(p.parts[-4], p.parts[-3], p.parts[-2])].append(json.load(open(f)))
    head = ["experiment", "variant", "target", "β", "weight", "adv", "steps", "data", "C1/C2", "seeds"] + [c[0] for c in COLS]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for (exp, var, cond), rs in groups.items():
        a = rs[0]["args"]
        nec = cond.startswith("vpd_nec")
        row = [exp, var, TARGET.get(cond, cond),
               "–" if cond == "plain" else f'{a["beta"]:g}',
               f'{a["nec_scale"]:g}×' if nec else "–",
               f'{a["comp_inner"]}×{a["comp_pgd_lr"]:g}' if nec else "–",
               f'{a["steps"] // 1000}k', "train+extra" if exp.startswith("06") else "train",
               f'{a["C1"]}/{a["C2"]}', str(len(rs))]
        for _, fn, f in COLS:
            try:
                row.append(fmt([fn(r) for r in rs], f))
            except (KeyError, TypeError):
                row.append("–")
        lines.append("| " + " | ".join(row) + " |")
    out = "\n".join(lines)
    (HERE / "runs/analysis").mkdir(parents=True, exist_ok=True)
    (HERE / "runs/analysis/all_runs.md").write_text(out + "\n")
    print(out)


if __name__ == "__main__":
    main()
