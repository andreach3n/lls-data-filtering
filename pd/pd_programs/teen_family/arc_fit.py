import itertools
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares

z = np.load(Path.home() / "work" / "teen_stats.npz")
top = list(z["top"]); A = z["teen_acts_top"]; d = z["digits"]
def ve(y, p): return 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
def table_ve(y): return ve(y, np.array([y[d == k].mean() for k in range(10)])[d])
smooth = [2431, 3333, 3323]
for a, b in itertools.permutations(smooth, 2):
    ya, yb = A[:, top.index(a)], A[:, top.index(b)]
    best = None
    for t0 in np.linspace(-np.pi, np.pi, 13):
        for w in np.linspace(-1, 1, 21):
            def res(q):
                ra, rb, t, ww, ca, cb = q; th = t + ww * d
                return np.r_[ya - (ra * np.cos(th) + ca), yb - (rb * np.sin(th) + cb)]
            r = least_squares(res, [np.ptp(ya) / 2, np.ptp(yb) / 2, t0, w, ya.mean(), yb.mean()])
            if best is None or r.cost < best.cost: best = r
    ra, rb, t, w, ca, cb = best.x; th = t + w * d
    print(f"{a} cos, {b} sin: VE {ve(ya, ra*np.cos(th)+ca):.3f}/{ve(yb, rb*np.sin(th)+cb):.3f} "
          f"(tables {table_ve(ya):.3f}/{table_ve(yb):.3f}) {np.degrees(w):.1f} deg/digit, arc {abs(np.degrees(w))*9:.0f} deg, "
          f"r=({ra:.2f},{rb:.2f}) offsets=({ca:.2f},{cb:.2f})")
M = np.stack([[A[d == k, top.index(c)].mean() for k in range(10)] for c in smooth])
X = A[:, [top.index(c) for c in smooth]]; X = X - X.mean(0)
s = np.linalg.svd(X, compute_uv=False); print("PCA var frac of the 3:", np.round(s**2 / (s**2).sum(), 3))
