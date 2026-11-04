"""Do the layer-1 components of an MNIST decomposition (train.py) split cleanly into 10 digit groups?

For each layer-1 component: on what share of each digit's test images is it important (gate > thr), and what share of
its own important uses come from its top digit ("purity"). Components are assigned to their top digit, then each
digit's group is switched off for every test image (layer 2 left alone) and accuracy is measured on every digit.
A clean split = switching off group k breaks digit k and leaves the others.

  python groups.py runs/03_adversary_strength/adv5/vpd_nec_pred/s0.pt [more .pt ...]   (prints a table; --plot writes a PNG per run)
"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))  # gated_lora
import train as M  # noqa: E402
G = M.G


@torch.no_grad()
def analyse(path, data):
    ck = torch.load(path, map_location=M.dev)
    a = type("A", (), ck["args"])
    model, wr, ci = M.build(a, 0)
    model.load_state_dict(ck["model"])
    ci.load_state_dict(ck["ci"])
    x, y = data["test_x"], data["test_y"]
    N, C1, C2 = x.shape[0], wr["fc1"].C, wr["fc2"].C
    imp = (M.gates_of(model, wr, ci, x)["fc1"][:, 0] > a.thr).float()  # (N, C1)
    n_dig = torch.stack([(y == k).sum() for k in range(10)]).float()
    uses = torch.stack([imp[y == k].sum(0) for k in range(10)])  # (10, C1): important uses per digit
    rate = uses / n_dig[:, None]  # share of digit k's images where the component is important
    tot = uses.sum(0)
    alive = tot >= 0.001 * N
    share = uses / tot.clamp(min=1)  # (10, C1): where its uses come from
    purity, top = share.max(0)
    # digits a component "serves": digits contributing >= 10% of its uses
    n_serves = (share >= 0.10).sum(0)

    # group ablation: switch off every layer-1 component assigned to digit k (alive ones only)
    full = M.fwd(model, wr, x, "full").argmax(-1)
    acc_full = torch.stack([(full[y == j] == j).float().mean() for j in range(10)])
    mat = torch.zeros(10, 10)
    sizes = []
    for k in range(10):
        keep = torch.ones(C1)
        grp = (top == k) & alive
        keep[grp] = 0
        sizes.append(int(grp.sum()))
        ms = {"fc1": keep.expand(N, 1, C1), "fc2": torch.ones(N, 1, C2)}
        pr = M.fwd(model, wr, x, "masked", ms).argmax(-1)
        mat[k] = torch.stack([(pr[y == j] == j).float().mean() for j in range(10)]) / acc_full
    off = mat[~torch.eye(10, dtype=bool)]
    alive_idx = alive.nonzero().flatten()
    w = tot[alive_idx] / tot[alive_idx].sum()  # weight components by how often they are used
    p = purity[alive_idx]
    r = {
        "run": "/".join(Path(path).with_suffix("").parts[-4:]),  # experiment/variant/condition/seed
        "C1": C1, "alive": int(alive.sum()),
        "serves_1": int((n_serves[alive_idx] == 1).sum()), "serves_2": int((n_serves[alive_idx] == 2).sum()),
        "serves_3plus": int((n_serves[alive_idx] >= 3).sum()),
        "purity_ge_0.9": int((p >= 0.9).sum()), "purity_median": p.median().item(),
        "uses_in_pure_0.9": (w * (p >= 0.9).float()).sum().item(),  # share of all important uses that are by pure components
        "group_sizes": sizes,
        "own_left": mat.diag().mean().item(), "own_left_max": mat.diag().max().item(),
        "others_left": off.mean().item(), "others_left_min": off.min().item(),
        "ablation": mat.tolist(), "purity": purity.tolist(), "top": top.tolist(), "rate": rate.T.tolist(),
        "alive_mask": alive.tolist(),
    }
    return r, wr


def plot(r, wr, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    V = wr["fc1"].V.detach().cpu()
    groups = [[c for c in range(len(r["top"])) if r["alive_mask"][c] and r["top"][c] == k] for k in range(10)]
    for g in groups:
        g.sort(key=lambda c: -r["purity"][c])
    width = max(len(g) for g in groups)
    fig, axes = plt.subplots(10, width + 1, figsize=(0.75 * (width + 1), 8.2), squeeze=False,
                             gridspec_kw={"width_ratios": [1.6] + [1] * width})
    for k, g in enumerate(groups):
        ax = axes[k, 0]
        ax.imshow([[0]], cmap="gray_r", vmin=0, vmax=1)
        ax.text(0, 0, f"{k}\n{r['ablation'][k][k]:.0%} left", ha="center", va="center", fontsize=7)
        ax.axis("off")
        for i in range(width):
            ax = axes[k, i + 1]
            ax.axis("off")
            if i < len(g):
                c = g[i]
                v = V[:, c].reshape(28, 28)
                lim = v.abs().max().item()
                ax.imshow(v, cmap="RdBu_r", vmin=-lim, vmax=lim)
                pur = r["purity"][c]
                ax.set_title(f"{pur:.0%}", fontsize=6, color="black" if pur >= 0.9 else "tab:red", pad=1)
    fig.suptitle(f"{r['run']}: layer-1 components grouped by top digit (title = share of uses from that digit; red < 90%)\n"
                 f"left column: that digit's accuracy left when its group is off", fontsize=8)
    fig.tight_layout(pad=0.2)
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--plot", action="store_true")
    a = ap.parse_args()
    data = M.load_mnist()
    rows = []
    for pth in a.ckpts:
        r, wr = analyse(pth, data)
        rows.append(r)
        out = Path(pth).with_name(Path(pth).stem + "_groups.json")
        out.write_text(json.dumps(r))
        if a.plot:
            plot(r, wr, Path(pth).with_name(Path(pth).stem + "_groups.png"))
    keys = ["alive", "serves_1", "serves_2", "serves_3plus", "purity_ge_0.9", "purity_median", "uses_in_pure_0.9",
            "own_left", "own_left_max", "others_left", "others_left_min"]
    print(f"{'run':60s}" + "".join(f"{k:>16s}" for k in keys))
    for r in rows:
        print(f"{r['run']:60s}" + "".join(f"{r[k]:16.3f}" if isinstance(r[k], float) else f"{r[k]:16d}" for k in keys))


if __name__ == "__main__":
    main()
