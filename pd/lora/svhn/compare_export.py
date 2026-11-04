"""Export two (or more) SVHN decompositions on the SAME test images for the side-by-side comparison viewer
(viewer/compare.html, built by viewer/build_compare.py).

Images: the first PER_DIGIT test images of each digit, plus every image among the first N_SCAN where any model is
wrong or the models disagree. Per image and model: prediction, probabilities of the full model, with the important
components removed (both layers), and under the strongest layer-1 / layer-2 attack (that layer's important
components off, the rest set adversarially to raise the true digit, 20 steps × 0.1); and the important components
with gate and "drop" (change in the predicted digit's logit when only that component is off).
Per model, per component that is important on at least one exported image: template (layer 1: V as 32×32; layer 2:
read direction pulled back to pixels, ReLU ignored), how often it is important on the whole test set, on what share
of each digit's test images, and for layer 2 what it writes to each digit's score.

  python3 compare_export.py viewer/compare_data.json \\
      "Predicted digit · β 10 · 3×=runs/03_nec_weight/nec3_12k/vpd_nec_mirror/s0.pt" \\
      "Uniform · β 30 · 5×=runs/10_uniform_b30_weight/nec5/vpd_nec_mirror_unif/s0.pt"
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import train as SV  # svhn/train.py: configures the shared MNIST code for SVHN

T = SV.T
HERE = Path(__file__).parent
_spec = importlib.util.spec_from_file_location("svhn_viewer_export", HERE / "viewer_export.py")
VE = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(VE)  # for VE.attack (the per-image attack)
PER_DIGIT, N_SCAN = 25, 2500


def r3(t):
    return [round(v, 3) for v in t.tolist()]


def load(path):
    ck = torch.load(path, map_location=T.dev, weights_only=False)
    a = argparse.Namespace(**ck["args"])
    model, wr, ci = T.build(a, 0)
    model.load_state_dict(ck["model"])
    ci.load_state_dict(ck["ci"])
    return model, wr, ci, a


def main(out_path, specs):
    data = SV.load_svhn()
    X, Y = data["test_x"], data["test_y"]
    models = [(lab, *load(HERE / p)) for lab, p in (s.split("=", 1) for s in specs)]

    with torch.no_grad():  # choose the images: shared by all models
        preds = torch.stack([T.fwd(m, wr, X[:N_SCAN], "full").argmax(-1) for _, m, wr, _, _ in models])
        interesting = ((preds != Y[:N_SCAN]).any(0) | (preds != preds[0]).any(0)).nonzero().flatten()
        idx = torch.cat([(Y == k).nonzero().flatten()[:PER_DIGIT] for k in range(10)] + [interesting]).unique()
    x, y = X[idx], Y[idx]
    images = [{"id": int(i), "label": int(l), "px": (xi * 255).round().int().tolist(), "m": []}
              for i, l, xi in zip(idx, y, x)]
    out_models = []
    for lab, model, wr, ci, a in models:
        C1, C2 = wr["fc1"].C, wr["fc2"].C
        N = len(idx)
        with torch.no_grad():
            g = T.gates_of(model, wr, ci, x)
            gall = torch.cat([g["fc1"][:, 0], g["fc2"][:, 0]], -1)
            full = T.fwd(model, wr, x, "full")
            pred = full.argmax(-1)
            imp = {n: (g[n] > a.thr).float() for n in ("fc1", "fc2")}
            rem = T.fwd(model, wr, x, "masked", {n: 1 - v for n, v in imp.items()})
            lp = full.gather(1, pred[:, None])[:, 0]
            abl = torch.zeros(N, C1 + C2)
            for j in range(C1 + C2):
                keep = torch.ones(C1 + C2)
                keep[j] = 0
                ms = {"fc1": keep[:C1].expand(N, 1, C1), "fc2": keep[C1:].expand(N, 1, C2)}
                abl[:, j] = T.fwd(model, wr, x, "masked", ms).gather(1, pred[:, None])[:, 0] - lp
            h = torch.relu(T.norm(x) @ wr["fc1"].V @ wr["fc1"].U + wr["fc1"].bias)
            act2 = h @ wr["fc2"].V  # layer-2 activations, for the "effect on this image" bars
        torch.manual_seed(0)
        adv1 = VE.attack(model, wr, ci, x, y, "fc1")
        adv2 = VE.attack(model, wr, ci, x, y, "fc2")
        used = set()
        for i, im in enumerate(images):
            js = (gall[i] > a.thr).nonzero().flatten().tolist()
            used.update(js)
            im["m"].append({
                "pred": int(pred[i]), "full": r3(F.softmax(full[i], -1)), "rem": r3(F.softmax(rem[i], -1)),
                "adv1": r3(F.softmax(adv1[i], -1)), "adv2": r3(F.softmax(adv2[i], -1)),
                "imp": [[j, round(gall[i, j].item(), 2), round(abl[i, j].item(), 2)]
                        + ([round(act2[i, j - C1].item(), 2)] if j >= C1 else []) for j in js]})
        with torch.no_grad():
            gt = T.gates_of(model, wr, ci, X)
            ga = torch.cat([gt["fc1"][:, 0], gt["fc2"][:, 0]], -1) > a.thr
            freq = ga.float().mean(0)
            share = torch.stack([ga[Y == k].float().mean(0) for k in range(10)])  # (10, C1+C2)
            V1, U1, V2, U2 = wr["fc1"].V, wr["fc1"].U, wr["fc2"].V, wr["fc2"].U
            t2 = ((V1 @ U1) @ V2).T
            acc = (T.fwd(model, wr, X, "full").argmax(-1) == Y).float().mean().item()
        pieces = {}
        for j in sorted(used):
            p = {"freq": round(freq[j].item(), 4), "share": r3(share[:, j])}
            p["tmpl"] = r3(V1[:, j] if j < C1 else t2[j - C1])
            if j >= C1:
                p["writes"] = r3(U2[j - C1])
            pieces[str(j)] = p
        out_models.append({"label": lab, "C1": C1, "C2": C2, "thr": a.thr, "acc": acc,
                           "target": "uniform" if "unif" in a.conds[0] else "pred",
                           "beta": a.beta, "weight": a.nec_scale, "pieces": pieces,
                           "l1_used": int((freq[:C1] > 0.001).sum()), "per_image": ga.float().sum(-1).mean().item()})
        print(f"{lab}: acc {acc:.3f}, {len(pieces)} components exported", flush=True)
    Path(out_path).write_text(json.dumps({"models": out_models, "images": images}, separators=(",", ":")))
    print(f"{len(images)} images -> {out_path} ({Path(out_path).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
