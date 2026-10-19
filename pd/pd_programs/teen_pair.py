"""Redo the task:1880 teen-pair analysis (separate tables vs one shared angle) on the site's published windows.

Each component is fitted and scored on its own windows' digit-after-'1' tokens, since the site
doesn't publish two components' activations on the same tokens.
"""
import sys

import numpy as np

from digit_after_one import after_one


def data(i):
    per, _ = after_one(i)
    d = np.concatenate([[k] * len(per[k]) for k in range(10)]).astype(float)
    y = np.concatenate([per[k] for k in range(10)])
    return d, y


def ve(y, pred):
    return 1 - np.mean((y - pred) ** 2) / np.var(y)


def lookup(d, y):
    t = {k: y[d == k].mean() for k in np.unique(d)}
    return np.array([t[k] for k in d])


def quadratic(d, y):
    return np.polyval(np.polyfit(d, y, 2), d)


def sinusoid(d, y, ws=np.radians(np.linspace(1, 40, 391))):
    """Best A cos(w d) + B sin(w d) over a grid of frequencies w; returns (w in deg/digit, phase in deg, pred)."""
    best = None
    for w in ws:
        X = np.stack([np.cos(w * d), np.sin(w * d)], 1)
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        err = np.sum((X @ c - y) ** 2)
        if best is None or err < best[0]:
            best = (err, w, c, X @ c)
    _, w, (a, b), pred = best
    return np.degrees(w), np.degrees(np.arctan2(-b, a)), pred   # pred = R cos(w d + phase)


def joint(dc, yc, ds, ys):
    """One angle θ = a + b·d read twice: yc ≈ Rc cos θ, ys ≈ Rs sin θ. Grid over (a, b), amplitudes in closed form."""
    best = None
    for a in np.radians(np.arange(-60, 31, 1.0)):
        for b in np.radians(np.arange(4, 25, 0.25)):
            c, s = np.cos(a + b * dc), np.sin(a + b * ds)
            Rc, Rs = c @ yc / (c @ c), s @ ys / (s @ s)
            err = np.sum((Rc * c - yc) ** 2) / np.var(yc) / len(yc) + np.sum((Rs * s - ys) ** 2) / np.var(ys) / len(ys)
            if best is None or err < best[0]:
                best = (err, a, b, Rc, Rs, Rc * c, Rs * s)
    _, a, b, Rc, Rs, pc, ps = best
    return np.degrees(a), np.degrees(b), Rc, Rs, pc, ps


if __name__ == "__main__":
    ic, is_ = (int(x) for x in sys.argv[1:3]) if len(sys.argv) > 2 else (5802, 5882)
    (dc, yc), (ds, ys) = data(ic), data(is_)
    print(f"cos component {ic}: {len(yc)} digit-after-1 tokens;  sin component {is_}: {len(ys)}")
    print(f"\nseparate lookup tables (10 numbers each): VE {ve(yc, lookup(dc, yc)):.2f} / {ve(ys, lookup(ds, ys)):.2f}")
    print(f"separate quadratics   (3 numbers each):  VE {ve(yc, quadratic(dc, yc)):.2f} / {ve(ys, quadratic(ds, ys)):.2f}")
    a, b, Rc, Rs, pc, ps = joint(dc, yc, ds, ys)
    print(f"joint program (4 numbers): θ = {a:+.0f}° + {b:.1f}°·d, {Rc:.2f}·cos θ, {Rs:.2f}·sin θ  ->  VE {ve(yc, pc):.2f} / {ve(ys, ps):.2f}")
    print(f"   θ runs from {a:+.0f}° (10) to {a + 9 * b:+.0f}° (19)")
    for i, (d, y) in ((ic, (dc, yc)), (is_, (ds, ys))):
        w, ph, p = sinusoid(d, y)
        print(f"sinusoid fit alone, {i}: {w:.1f}°/digit, R cos(w·d {ph:+.0f}°), VE {ve(y, p):.2f}")
    print("\nangle of mean point per digit (axes divided by Rc, Rs), vs joint θ:")
    for k in range(10):
        if (dc == k).any() and (ds == k).any():
            ang = np.degrees(np.arctan2(ys[ds == k].mean() / Rs, yc[dc == k].mean() / Rc))
            print(f"  1{k}: {ang:+5.0f}°   joint {a + b * k:+5.0f}°   (n = {(dc == k).sum()}, {(ds == k).sum()})")
