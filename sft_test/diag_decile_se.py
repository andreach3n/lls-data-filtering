#!/usr/bin/env python3
"""Prompt-clustered bootstrap SEs and the rank trend for a decile sweep (numbers in plot_deciles.py).

  python diag_decile_se.py <dl_dir> [bold|bold-teal]

bold       queue12 + queue13: ranked by lp_bold - lp_base. D must hold sft/{deciles_g*_k10,
           pos_bold_a1.0_k100, slice_*_k10} general generations from HF.
bold-teal  queue14: ranked by lp_bold - lp_teal. D must hold sft/teal_deciles_g*_k10.
Both read the same random reference (pos_bold_a1.0_k100 randomonly)."""
import json, re, numpy as np
from collections import defaultdict
B = re.compile(r"\*\*[^*\n]+\*\*")
import sys
D = (sys.argv[1] if len(sys.argv) > 1 else "hf_dl") + "/sft/"
def load(*fs):
    by = defaultdict(list)
    for f in fs:
        for l in open(D + f):
            r = json.loads(l)
            by[r["prompt"]]  # every prompt is a cluster, even one with no closed answers here
            if r["think_close"]:
                by[r["prompt"]].append(len(B.findall(r["text"])))
    return by
def mean(by, ps): 
    v = [x for p in ps for x in by.get(p, [])]; return np.mean(v)
# ONE evaluation, like every other bar: the alpha=1.0 random run, the same setting as every arm.
# (pos_bold_a0.323_k100's randomonly trained on the IDENTICAL documents -- the random draw does not
# depend on alpha -- so pooling the two averaged two evaluations of one model on the same prompts.)
rand = load("pos_bold_a1.0_k100/generations_randomonly_bold_a1.0_k10_general.jsonl")
SWEEP = sys.argv[2] if len(sys.argv) > 2 else "bold"
if SWEEP == "bold":
    arms = [("top", "pos_bold_a1.0_k100/generations_tailonly_bold_a1.0_k10_general.jsonl")] + \
           [(f"{d}", f"deciles_g{g}_k10/generations_slice{d}_bold_a1.0_k10_general.jsonl")
            for g, ds in [(1,(10,20)),(2,(30,40)),(3,(50,60)),(4,(70,80))] for d in ds] + \
           [("middle45", "slice_middle_k10/generations_middleonly_bold_a1.0_k10_general.jsonl"),
            ("bottom", "slice_bottom_k10/generations_tailonly_bold_a1.0_k10_general.jsonl")]
elif SWEEP == "bold-teal":   # queue14: all ten deciles in one sweep, top/bottom named as above
    arms = [({0: "top", 90: "bottom"}.get(d, f"{d}"),
             f"teal_deciles_g{g}_k10/generations_slice{d}_bold-teal_a1.0_k10_general.jsonl")
            for g, ds in [(1,(0,40,70)),(2,(10,30,90)),(3,(20,60)),(4,(50,80))] for d in ds]
    arms.sort(key=lambda a: {"top": 0, "bottom": 90}.get(a[0], int(a[0]) if a[0].isdigit() else 0))
else:
    raise SystemExit("sweep must be bold or bold-teal")
P = sorted(set(rand))  # all 100 general prompts (load registers unclosed ones too)
rng = np.random.default_rng(0); NB = 4000
idx = [rng.choice(len(P), len(P)) for _ in range(NB)]
def boot(by): return np.array([mean(by, [P[i] for i in ix]) for ix in idx])
rb = boot(rand); r0 = mean(rand, P)
print(f"random (alpha=1.0 run) {r0:.2f}  SE {rb.std():.2f}")
out = {}
for name, f in arms:
    by = load(f); m = mean(by, P); b = boot(by)
    ind = np.sqrt(b.std()**2 + rb.std()**2)      # independent-bootstrap SE of the difference
    pair = (b - rb).std()                        # same resampled prompts for both
    out[name] = dict(bold=m, diff=m - r0, se_ind=ind, se_pair=pair)
    print(f"{name:9s} {m:6.2f}  {m-r0:+6.2f}  se_ind {ind:.2f}  se_pair {pair:.2f}  sigma {(m-r0)/ind:+.1f}")
json.dump(out, open(f"deciles_se_{SWEEP}.json", "w"), indent=1)

# TREND: weighted least-squares slope of absolute bold on decile start (0..90), each point
# weighted by its own bootstrap variance. The random reference is common to every point, so it
# drops out of the slope. Also the same slope refit per bootstrap draw (prompt-clustered,
# shared resampled prompts across arms), which respects the correlation between arms.
pos = {"top": 0, **{str(d): d for d in range(10, 90, 10)}, "bottom": 90}
X = np.array(list(pos.values()), float)
Bs = {n: boot(load(f)) for n, f in arms if n in pos}
Y = np.array([out[n]["bold"] for n in pos]); S = np.array([Bs[n].std() for n in pos])
w = 1 / S**2
def slope(y): 
    xm = (w * X).sum() / w.sum(); return (w * (X - xm) * y).sum() / (w * (X - xm)**2).sum()
s0 = slope(Y); sb = np.array([slope(np.array([Bs[n][i] for n in pos])) for i in range(NB)])
print(f"\nslope {s0*10:+.2f} bold per decile, bootstrap SE {sb.std()*10:.2f}, z {s0/sb.std():+.1f}, "
      f"P(slope>=0) {np.mean(sb >= 0):.4f}")
from scipy.stats import spearmanr
print("spearman(rank, bold) over 10 deciles:", spearmanr(X, Y))
# without the top decile: is there any trend left below it?
m = X > 0; w2 = w[m]; X2 = X[m]
xm = (w2*X2).sum()/w2.sum(); sl2 = lambda y: (w2*(X2-xm)*y).sum()/(w2*(X2-xm)**2).sum()
b2 = np.array([sl2(np.array([Bs[n][i] for n in pos if pos[n] > 0])) for i in range(NB)])
print(f"slope excl. top {sl2(Y[m])*10:+.2f} per decile, SE {b2.std()*10:.2f}, z {sl2(Y[m])/b2.std():+.1f}")
