"""Export one trained MNIST decomposition (mnist_necessity.py) to JSON for the per-image piece viewer.

Per test image: pixels, gates of all pieces, predictions of the full / important-only / important-removed models,
and the effect of switching off each important piece alone (change in the predicted class's logit).
Per piece: layer 1 = V as a 28x28 template; layer 2 = U (what it writes to each digit's logit) and an
input-space template (V projected back through layer 1, ignoring the ReLU); layer1 -> layer2 connection strengths.

  python mnist_viewer_export.py runs/mnist/vpd_nec_pred_s0_adv5.pt runs/mnist/viewer_adv5.json
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import mnist_necessity as M  # noqa: E402
G = M.G


def r3(t):
    return [round(v, 3) for v in t.tolist()]


@torch.no_grad()
def main(ckpt_path, out_path, per_digit=30):
    ck = torch.load(ckpt_path, map_location=M.dev)
    a = type("A", (), ck["args"])
    model, wr, ci = M.build(a, 0)
    model.load_state_dict(ck["model"])
    ci.load_state_dict(ck["ci"])
    data = M.load_mnist()
    x_all, y_all = data["test_x"], data["test_y"]

    # images: the first `per_digit` of each digit, plus every misclassified one among the first 2000
    lg_all = M.fwd(model, wr, x_all[:2000], "full")
    wrong = (lg_all.argmax(-1) != y_all[:2000]).nonzero().flatten()
    idx = torch.cat([(y_all == k).nonzero().flatten()[:per_digit] for k in range(M.NCLS)] + [wrong]).unique()
    x, y = x_all[idx], y_all[idx]
    N = len(idx)
    names = ["fc1", "fc2"]
    C1, C2 = wr["fc1"].C, wr["fc2"].C

    g = M.gates_of(model, wr, ci, x)
    gall = torch.cat([g[n][:, 0] for n in names], -1)  # (N, C1+C2)
    full = M.fwd(model, wr, x, "full")
    pred = full.argmax(-1)
    imp = {n: (g[n] > a.thr).float() for n in names}
    out_imp = M.fwd(model, wr, x, "masked", imp)
    out_rem = M.fwd(model, wr, x, "masked", {n: 1 - m for n, m in imp.items()})

    # single-piece ablation from the full model: change in the predicted class's logit (and its probability)
    lp_full = full.gather(1, pred[:, None])[:, 0]
    abl = torch.zeros(N, C1 + C2)
    for j in range(C1 + C2):
        keep = torch.ones(C1 + C2)
        keep[j] = 0
        ms = {"fc1": keep[:C1].expand(N, 1, C1), "fc2": keep[C1:].expand(N, 1, C2)}
        abl[:, j] = M.fwd(model, wr, x, "masked", ms).gather(1, pred[:, None])[:, 0] - lp_full

    # piece activations: layer 1 = norm(x) . V_c; layer 2 = h . V_c (h = hidden ReLU of the full model)
    xn = M.norm(x)
    act1 = xn @ wr["fc1"].V
    h = torch.relu(xn @ wr["fc1"].V @ wr["fc1"].U + wr["fc1"].bias)
    act2 = h @ wr["fc2"].V

    images = []
    for i in range(N):
        gi = gall[i]
        important = (gi > a.thr).nonzero().flatten().tolist()
        images.append({
            "id": int(idx[i]), "label": int(y[i]), "pred": int(pred[i]),
            "px": (x[i] * 255).round().int().tolist(),
            "p_full": r3(F.softmax(full[i], -1)), "p_imp": r3(F.softmax(out_imp[i], -1)),
            "p_rem": r3(F.softmax(out_rem[i], -1)),
            "imp": [{"j": j, "g": round(gi[j].item(), 3),
                     "act": round((act1[i, j] if j < C1 else act2[i, j - C1]).item(), 3),
                     "abl": round(abl[i, j].item(), 3)} for j in important],
        })

    V1, U1, V2, U2 = wr["fc1"].V, wr["fc1"].U, wr["fc2"].V, wr["fc2"].U
    W1 = V1 @ U1  # (784, d)
    t2 = (W1 @ V2).T  # (C2, 784): layer-2 read direction pulled back to pixels, ReLU ignored
    conn = U1 @ V2  # (C1, C2): how much layer-1 piece c writes along layer-2 piece d's read direction
    gfull = M.gates_of(model, wr, ci, x_all)
    gf = torch.cat([gfull[n][:, 0] for n in names], -1)
    freq = (gf > a.thr).float().mean(0)
    mg = torch.stack([gf[y_all == k].mean(0) for k in range(M.NCLS)])  # (10, C1+C2)
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

    meta = {"ckpt": str(ckpt_path), "thr": a.thr, "C1": C1, "C2": C2, "args": ck["args"],
            "test_acc": (M.fwd(model, wr, x_all, "full").argmax(-1) == y_all).float().mean().item()}
    Path(out_path).write_text(json.dumps({"meta": meta, "images": images, "pieces": pieces}, separators=(",", ":")))
    print(f"{N} images, {C1 + C2} pieces -> {out_path} ({Path(out_path).stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
