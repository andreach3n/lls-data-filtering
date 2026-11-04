"""What do the remaining components do on digit D's images once D's layer-1 components are switched off?

With D's components off (layer 2 fully on), each D image's digit scores split exactly into one part per remaining
layer-1 component plus a bias part: with the ReLU pattern fixed by the image,
  scores = sum_c (a_c U_c * relu_on) W2 + (b1 * relu_on) W2 + b2,   W2 = all of layer 2.
Scores are centred (mean over the 10 digits removed), since softmax ignores a shift.

For each seed: the fallback digit, which components push D's images toward it, their gates on D images, which
digit each component belongs to (the digit whose images give it the highest mean gate), and how much of the push
was already there with D's components on (i.e. what D's components were outvoting).

  python leftovers.py [--digit 7] [--ckpt runs/07_beta_sweep/mirror_b10_adv10/vpd_nec_mirror]
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # gated_lora
import train as M  # noqa: E402
import specificity as S  # noqa: E402

R = HERE / "runs"
OUT = R / "analysis"


@torch.no_grad()
def parts(model, x, keep):
    """Centred per-component score parts. x: (N, 784); keep: (C1,) layer-1 mask. Returns (N, C1, 10), (N, 10)."""
    f1, f2 = model.fc1, model.fc2
    xn = M.norm(x[:, None])[:, 0]
    a = (xn @ f1.V) * keep                                     # (N, C1) activations
    pre = a @ f1.U + f1.bias                                   # (N, d)
    on = (pre > 0).float()
    W2 = f2.V @ f2.U                                           # (d, 10)
    comp = torch.einsum("nc,cd,nd,dk->nck", a, f1.U, on, W2)  # (N, C1, 10)
    rest = (f1.bias * on) @ W2 + f2.bias                       # (N, 10)
    centre = lambda t: t - t.mean(-1, keepdim=True)  # noqa: E731
    return centre(comp), centre(rest)


@torch.no_grad()
def analyse(path, data, D, top=5):
    ck = torch.load(path, map_location=M.dev)
    a = type("A", (), ck["args"])
    model, wr, ci = M.build(a, 0)
    model.load_state_dict(ck["model"])
    ci.load_state_dict(ck["ci"])
    x, y = data["test_x"], data["test_y"]
    C1 = wr["fc1"].C
    g1 = M.gates_of(model, wr, ci, x)["fc1"][:, 0]
    owner_gate = torch.stack([g1[y == k].mean(0) for k in range(10)])  # (10, C1)
    owner = owner_gate.argmax(0)
    group = owner_gate[D] > S.GATE_MIN
    xd, gd = x[y == D], g1[y == D]
    keep = torch.ones(C1)
    keep[group] = 0
    comp_off, rest_off = parts(model, xd, keep)
    comp_on, rest_on = parts(model, xd, torch.ones(C1))
    # sanity: parts add up to the model's (centred) scores
    lg = M.fwd(model, wr, xd, "masked", {"fc1": keep.expand(len(xd), 1, C1), "fc2": torch.ones(len(xd), 1, wr["fc2"].C)})
    err = ((comp_off.sum(1) + rest_off) - (lg - lg.mean(-1, keepdim=True))).abs().max().item()
    pred = lg.argmax(-1)
    fb = int(torch.bincount(pred, minlength=10).argmax())
    sel = pred == fb
    # push toward the fallback digit = its centred score part, on the images that fell back to it
    push = comp_off[sel][:, :, fb]                      # (n, C1)
    total = push.sum(1).mean().item() + rest_off[sel][:, fb].mean().item()
    order = push.mean(0).argsort(descending=True)[:top]
    rows = [{"comp": int(c), "owner": int(owner[c]), "owner_gate": round(owner_gate[owner[c], c].item(), 2),
             "push": round(push[:, c].mean().item(), 2),
             "gate_on_D": round(gd[sel][:, c].mean().item(), 3),
             "active_share": round((gd[sel][:, c] > a.thr).float().mean().item(), 2)} for c in order]
    d_comps = group.nonzero()[:, 0]
    return {"fallback": fb, "share": round(sel.float().mean().item(), 2), "check_err": err,
            "fallback_score_off": round(total, 2),
            "fallback_score_on": round(comp_on[sel][:, :, fb].sum(1).mean().item() + rest_on[sel][:, fb].mean().item(), 2),
            "D_comps_push_on_fallback": round(comp_on[sel][:, d_comps, fb].sum(1).mean().item(), 2),
            "bias_part": round(rest_off[sel][:, fb].mean().item(), 2),
            "top": rows, "D_comps": [int(c) for c in d_comps]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--digit", type=int, default=7)
    ap.add_argument("--ckpt", default="07_beta_sweep/mirror_b10_adv10/vpd_nec_mirror")
    a = ap.parse_args()
    D = a.digit
    data = M.load_mnist()
    res = {}
    for s in range(3):
        r = analyse(R / a.ckpt / f"s{s}.pt", data, D)
        res[s] = r
        print(f"\nseed {s}: {r['share'] * 100:.0f}% of {D}s fall back to {r['fallback']}  (parts add up: max err {r['check_err']:.1e})")
        print(f"  centred score of {r['fallback']} on those images: {r['fallback_score_on']:+.2f} with {D}-components on, "
              f"{r['fallback_score_off']:+.2f} with them off")
        print(f"  {D}-components' own part of that score when on: {r['D_comps_push_on_fallback']:+.2f}; "
              f"bias part when off: {r['bias_part']:+.2f}")
        print(f"  top pushes toward {r['fallback']} with {D}-components off:")
        for t in r["top"]:
            print(f"    comp {t['comp']:3d}  push {t['push']:+.2f}  belongs to digit {t['owner']} (mean gate {t['owner_gate']:.2f})"
                  f"  gate on these {D}s {t['gate_on_D']:.3f}, important on {t['active_share'] * 100:.0f}%")
    json.dump(res, open(OUT / f"leftovers_{D}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
