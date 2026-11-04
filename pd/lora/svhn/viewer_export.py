"""Export one trained SVHN decomposition (train.py) to JSON for the per-image piece viewer (viewer/).

Same as ../mnist/viewer_export.py, plus, per image, the strongest attack on each layer's necessity: that layer's
important pieces off, the rest of that layer set adversarially (20 sign-PGD steps of 0.1, as in the evaluation),
the other layer normal. If the attack gets the right digit back, the layer had a backup.

Per test image: pixels, gates, predictions of the full / important-only / important-removed / attacked models, and
the effect of switching off each important piece alone (change in the predicted class's logit).
Per piece: layer 1 = V as a 32x32 template; layer 2 = U (what it writes to each digit's logit) and an
input-space template (V projected back through layer 1, ignoring the ReLU); layer1 -> layer2 connection strengths.

  python3 viewer_export.py runs/03_nec_weight/nec3_12k/vpd_nec_mirror/s0.pt viewer/data_nec3_b10.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

import train as S  # svhn/train.py: configures the shared MNIST code for SVHN

T = S.T
G = T.G


def r3(t):
    return [round(v, 3) for v in t.tolist()]


def attack(model, wr, ci, x, y, layer, steps=20, lr=0.1):
    """Per-image version of train.worst_comp(layer=...): returns the attacked model's logits."""
    with torch.no_grad():
        T.fwd(model, wr, x, "full")
        gp = {n: z.detach() for n, z in ci({n: w.last_input for n, w in wr.items()}).items()}
    mk = lambda src: T.layer_comp_masks(gp, layer, src)  # noqa: E731
    src = {n: torch.rand_like(z).requires_grad_(True) for n, z in gp.items()}
    for _ in range(steps):
        l = F.cross_entropy(T.fwd(model, wr, x, "masked", mk(src)), y)
        gr = torch.autograd.grad(l, list(src.values()), allow_unused=True)
        with torch.no_grad():
            for (n, s), g in zip(src.items(), gr):
                if g is not None:
                    s.sub_(lr * g.sign()).clamp_(0, 1)
    with torch.no_grad():
        return T.fwd(model, wr, x, "masked", mk({n: s.detach() for n, s in src.items()}))


def main(ckpt_path, out_path, per_digit=30):
    ck = torch.load(ckpt_path, map_location=T.dev, weights_only=False)
    a = argparse.Namespace(**ck["args"])
    model, wr, ci = T.build(a, 0)
    model.load_state_dict(ck["model"])
    ci.load_state_dict(ck["ci"])
    data = S.load_svhn()
    x_all, y_all = data["test_x"], data["test_y"]

    with torch.no_grad():
        # images: the first `per_digit` of each digit, plus every misclassified one among the first 2000
        lg_all = T.fwd(model, wr, x_all[:2000], "full")
        wrong = (lg_all.argmax(-1) != y_all[:2000]).nonzero().flatten()
        idx = torch.cat([(y_all == k).nonzero().flatten()[:per_digit] for k in range(T.NCLS)] + [wrong]).unique()
        x, y = x_all[idx], y_all[idx]
        N = len(idx)
        names = ["fc1", "fc2"]
        C1, C2 = wr["fc1"].C, wr["fc2"].C

        g = T.gates_of(model, wr, ci, x)
        gall = torch.cat([g[n][:, 0] for n in names], -1)  # (N, C1+C2)
        full = T.fwd(model, wr, x, "full")
        pred = full.argmax(-1)
        imp = {n: (g[n] > a.thr).float() for n in names}
        out_imp = T.fwd(model, wr, x, "masked", imp)
        out_rem = T.fwd(model, wr, x, "masked", {n: 1 - m for n, m in imp.items()})
        out_rem1 = T.fwd(model, wr, x, "masked", {"fc1": 1 - imp["fc1"], "fc2": torch.ones_like(imp["fc2"])})
        out_rem2 = T.fwd(model, wr, x, "masked", {"fc1": torch.ones_like(imp["fc1"]), "fc2": 1 - imp["fc2"]})

        # single-piece ablation from the full model: change in the predicted class's logit
        lp_full = full.gather(1, pred[:, None])[:, 0]
        abl = torch.zeros(N, C1 + C2)
        for j in range(C1 + C2):
            keep = torch.ones(C1 + C2, device=T.dev)
            keep[j] = 0
            ms = {"fc1": keep[:C1].expand(N, 1, C1), "fc2": keep[C1:].expand(N, 1, C2)}
            abl[:, j] = (T.fwd(model, wr, x, "masked", ms).gather(1, pred[:, None])[:, 0] - lp_full).cpu()

        # piece activations: layer 1 = norm(x) . V_c; layer 2 = h . V_c (h = hidden ReLU of the full model)
        xn = T.norm(x)
        act1 = xn @ wr["fc1"].V
        h = torch.relu(xn @ wr["fc1"].V @ wr["fc1"].U + wr["fc1"].bias)
        act2 = h @ wr["fc2"].V

    torch.manual_seed(0)
    out_adv1 = attack(model, wr, ci, x, y, "fc1")
    out_adv2 = attack(model, wr, ci, x, y, "fc2")

    with torch.no_grad():
        images = []
        for i in range(N):
            gi = gall[i]
            important = (gi > a.thr).nonzero().flatten().tolist()
            images.append({
                "id": int(idx[i]), "label": int(y[i]), "pred": int(pred[i]),
                "px": (x[i] * 255).round().int().tolist(),
                "p_full": r3(F.softmax(full[i], -1)), "p_imp": r3(F.softmax(out_imp[i], -1)),
                "p_rem": r3(F.softmax(out_rem[i], -1)), "p_rem1": r3(F.softmax(out_rem1[i], -1)),
                "p_rem2": r3(F.softmax(out_rem2[i], -1)),
                "p_adv1": r3(F.softmax(out_adv1[i], -1)), "p_adv2": r3(F.softmax(out_adv2[i], -1)),
                "imp": [{"j": j, "g": round(gi[j].item(), 3),
                         "act": round((act1[i, j] if j < C1 else act2[i, j - C1]).item(), 3),
                         "abl": round(abl[i, j].item(), 3)} for j in important],
            })

        V1, U1, V2, U2 = wr["fc1"].V, wr["fc1"].U, wr["fc2"].V, wr["fc2"].U
        t2 = ((V1 @ U1) @ V2).T  # (C2, 1024): layer-2 read direction pulled back to pixels, ReLU ignored
        conn = U1 @ V2  # (C1, C2): how much layer-1 piece c writes along layer-2 piece d's read direction
        gfull = T.gates_of(model, wr, ci, x_all)
        gf = torch.cat([gfull[n][:, 0] for n in names], -1)
        freq = (gf > a.thr).float().mean(0)
        mg = torch.stack([gf[y_all == k].mean(0) for k in range(T.NCLS)])  # (10, C1+C2)
        pieces = []
        for j in range(C1 + C2):
            p = {"j": j, "layer": 1 if j < C1 else 2, "c": j if j < C1 else j - C1,
                 "freq": round(freq[j].item(), 4), "mean_gate_by_digit": r3(mg[:, j])}
            if j < C1:
                p["tmpl"] = r3(V1[:, j])
                p["to_l2"] = r3(conn[j])
            else:
                p["tmpl"] = r3(t2[j - C1])
                p["writes"] = r3(U2[j - C1])
                p["from_l1"] = r3(conn[:, j - C1])
            pieces.append(p)

        lg_test = T.fwd(model, wr, x_all, "full")
        meta = {"ckpt": str(ckpt_path), "thr": a.thr, "C1": C1, "C2": C2, "args": ck["args"],
                "n_test": int(len(y_all)), "test_acc": (lg_test.argmax(-1) == y_all).float().mean().item(),
                "l1_used": int((freq[:C1] > 0.001).sum()), "l2_used": int((freq[C1:] > 0.001).sum()),
                "per_image": (gf > a.thr).float().sum(-1).mean().item()}
    Path(out_path).write_text(json.dumps({"meta": meta, "images": images, "pieces": pieces}, separators=(",", ":")))
    print(f"{N} images, {C1 + C2} pieces -> {out_path} ({Path(out_path).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
