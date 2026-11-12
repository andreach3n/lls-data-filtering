"""Attribution graphs between components: layer-1 components -> layer-2 components -> digit scores.

With an image's ReLU on/off pattern held fixed, the network is linear in the components, so the attribution is exact:
  a_c            = x . V_c                                  (layer-1 component c's activation)
  edge c -> k    = a_c (U_c * relu_on) . V2_k               (sums over c, plus the bias path, to a_k)
  edge k -> digit = a_k (U2_k[digit] - mean_j U2_k[j])      (centred, so it is a softmax-invariant score)
  edge c -> digit (through layer 2) = sum_k edge(c -> k) * (U2_k[digit] - mean)
Every node also carries its gate g (important = g > thr) so you can see whether the computation runs through the
components the gates call important.

Models (seed 0): with necessity = Model 1 (hinge, β 10, mirrored, adversary 10×0.1); without = the β 10 no-necessity run.

  python attribution_graph.py   ->  runs/analysis/attr_graph.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import train as M  # noqa: E402

R = HERE / "runs"
MODELS = {"nec": ("With necessity (Model 1)", "07_beta_sweep/mirror_b10_adv10/vpd_nec_mirror/s0.pt"),
          "none": ("Without necessity", "09_no_necessity/vpd_b10/vpd/s0.pt")}
PER_DIGIT = 4          # example images per digit
COVER, CAP1, CAP2 = 0.9, 6, 5


def pick(contrib, imp, cap):
    """Components covering COVER of the total |contribution|, at most cap, plus every gate-important one."""
    order = np.argsort(-np.abs(contrib))
    tot, run, keep = np.abs(contrib).sum() + 1e-12, 0.0, []
    for i in order:
        if run / tot >= COVER or len(keep) >= cap:
            break
        keep.append(int(i))
        run += abs(contrib[i])
    for i in np.nonzero(imp)[0]:
        if int(i) not in keep:
            keep.append(int(i))
    return keep


@torch.no_grad()
def analyse(path, data, idx):
    ck = torch.load(R / path, map_location=M.dev)
    a = type("A", (), ck["args"])
    model, wr, ci = M.build(a, 0)
    model.load_state_dict(ck["model"])
    ci.load_state_dict(ck["ci"])
    x, y = data["test_x"], data["test_y"]
    g = M.gates_of(model, wr, ci, x)
    g1, g2 = g["fc1"][:, 0].cpu().numpy(), g["fc2"][:, 0].cpu().numpy()
    logits = M.fwd(model, wr, x, "full")
    pred = logits.argmax(-1).cpu().numpy()
    f1, f2 = wr["fc1"], wr["fc2"]
    V1, U1, b1 = f1.V, f1.U, f1.bias
    V2, U2, b2 = f2.V, f2.U, f2.bias
    U2c = U2 - U2.mean(1, keepdim=True)                       # centred write vectors (C2, 10)
    xn = M.norm(x)
    A1 = xn @ V1                                              # (N, C1)
    Z = A1 @ U1 + b1
    D = (Z > 0).float()                                       # relu pattern (N, d)
    A2 = torch.relu(Z) @ V2                                   # (N, C2)
    # edges c -> k for every image: a_c * (U1_c * D) @ V2   -> (N, C1, C2); bias path (N, C2)
    E12 = A1[:, :, None] * torch.einsum("cd,nd,dk->nck", U1, D, V2)
    Eb2 = (D * b1) @ V2
    E2d = A2[:, :, None] * U2c[None]                          # (N, C2, 10)
    b2c = (b2 - b2.mean()).cpu().numpy()
    out = dict(images=[], digits=[], acc=float((pred == y.cpu().numpy()).mean()))

    def graph(sel, digit, label, img=None):
        e12 = E12[sel].mean(0).cpu().numpy()                  # (C1, C2)
        eb2 = Eb2[sel].mean(0).cpu().numpy()
        e2d = E2d[sel, :, digit].mean(0).cpu().numpy()        # (C2,)
        a1, a2 = A1[sel].mean(0).cpu().numpy(), A2[sel].mean(0).cpu().numpy()
        gi1, gi2 = (g1[sel] > a.thr).mean(0), (g2[sel] > a.thr).mean(0)
        gm1, gm2 = g1[sel].mean(0), g2[sel].mean(0)
        w2 = U2c[:, digit].cpu().numpy()
        c1d = e12 @ w2                                        # layer-1 component -> digit score, via layer 2
        k1 = pick(c1d, gi1 >= 0.5, CAP1)
        k2 = pick(e2d, gi2 >= 0.5, CAP2)
        r1 = [c for c in range(len(c1d)) if c not in k1]
        r2 = [k for k in range(len(e2d)) if k not in k2]
        score = float(e2d.sum() + b2c[digit])
        L1 = [dict(id=c, a=float(a1[c]), g=float(gm1[c]), imp=float(gi1[c]), to_digit=float(c1d[c])) for c in k1]
        L2 = [dict(id=k, a=float(a2[k]), g=float(gm2[k]), imp=float(gi2[k]), to_digit=float(e2d[k])) for k in k2]
        # edges in digit-score units: c -> k carries a_c (U_c*relu_on).V2_k * centred U2_k[digit]
        S = e12 * w2[None, :]
        sb = eb2 * w2
        edges = [dict(s=c, t=k, w=float(S[c, k])) for c in k1 for k in k2]
        edges += [dict(s="rest", t=k, w=float(S[r1, k].sum())) for k in k2]
        edges += [dict(s=c, t="rest", w=float(S[c, r2].sum())) for c in k1]
        edges += [dict(s="rest", t="rest", w=float(S[np.ix_(r1, r2)].sum()))]
        edges += [dict(s="b1", t=k, w=float(sb[k])) for k in k2] + [dict(s="b1", t="rest", w=float(sb[r2].sum()))]
        # how much of the digit score runs through gate-important components
        imp1 = gi1 >= 0.5
        imp2 = gi2 >= 0.5
        res = dict(label=label, digit=int(digit), score=score, b2=float(b2c[digit]),
                   rest1=float(c1d[r1].sum()), rest2=float(e2d[r2].sum()), b1=float(eb2 @ w2),
                   imp_share1=float(c1d[imp1].sum() / (c1d.sum() + 1e-12)),
                   imp_share2=float(e2d[imp2].sum() / (e2d.sum() + 1e-12)),
                   n_imp1=int(imp1.sum()), n_imp2=int(imp2.sum()), L1=L1, L2=L2, edges=edges)
        if img is not None:
            res["img"] = img
        return res

    yy = y.cpu().numpy()
    for i in idx:
        lab = f"#{i}: a {yy[i]}" + ("" if pred[i] == yy[i] else f" (predicted {pred[i]})")
        img = (x[i].cpu().numpy() * 255).round().astype(int).tolist()
        out["images"].append(graph([i], yy[i], lab, img) | dict(index=int(i), pred=int(pred[i]),
                                                              probs=torch.softmax(logits[i], -1).cpu().numpy().round(3).tolist()))
    for d in range(10):
        sel = np.nonzero((yy == d) & (pred == yy))[0].tolist()
        out["digits"].append(graph(sel, d, f"all correctly classified {d}s ({len(sel)} images)"))
    # per-layer whole-test-set: share of the true-digit score carried by components important on THAT image
    s1, s2 = [], []
    for d in range(10):
        sel = (yy == d)
        w2 = U2c[:, d]
        c1d = torch.einsum("nck,k->nc", E12[sel], w2).cpu().numpy()
        e2d = E2d[sel, :, d].cpu().numpy()
        s1.append((c1d * (g1[sel] > a.thr)).sum(1) / (c1d.sum(1) + 1e-12))
        s2.append((e2d * (g2[sel] > a.thr)).sum(1) / (e2d.sum(1) + 1e-12))
    out["imp_share"] = dict(l1=float(np.median(np.concatenate(s1))), l2=float(np.median(np.concatenate(s2))))
    out["V1"] = (V1.T.cpu().numpy() / (np.abs(V1.cpu().numpy()).max(0)[:, None] + 1e-12) * 127).round().astype(int).tolist()
    out["U2c"] = U2c.cpu().numpy().round(3).tolist()
    out["size"] = dict(l1=float((g1 > a.thr).sum(1).mean()), l2=float((g2 > a.thr).sum(1).mean()))
    return out


def main():
    data = M.load_mnist()
    yy = data["test_y"].cpu().numpy()
    idx = [int(i) for d in range(10) for i in np.nonzero(yy == d)[0][:PER_DIGIT]]
    res = {k: dict(name=n, **analyse(p, data, idx)) for k, (n, p) in MODELS.items()}
    for k, r in res.items():
        print(k, "acc", r["acc"], "size", r["size"], "median share of digit score via important comps", r["imp_share"])
        for dg in r["digits"]:
            print(f"  {dg['digit']}  L1 important {dg['n_imp1']:2d} share {dg['imp_share1']:.2f} | "
                  f"L2 important {dg['n_imp2']:2d} share {dg['imp_share2']:.2f}  score {dg['score']:.1f}")
    out = R / "analysis" / "attr_graph.json"
    json.dump(res, open(out, "w"))
    print(out, out.stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
