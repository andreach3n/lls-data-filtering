"""First look at the SFT update in the base decomposition's component basis (CPU only).

For every decomposed matrix W (72 = 12 layers x q,k,v,o,c_fc,down_proj), least-squares fit the
LoRA update D = B A onto the base components W_j = U_j V_j^T of that matrix:
    D ~ sum_j c_j W_j + leftover
using the Gram matrix G_jk = (U_j.U_k)(V_j.V_k) and b_j = U_j^T D V_j. Baselines with the same
number of rank-1 terms: the top singular directions of the base weight W, and random rank-1 terms.

Each component's causal importance at the bold decision (last position of held-out target
windows) and on nontarget text is computed on CPU, so components can be split into
bold-specific (CI_target > 0.1, CI_nontarget < 0.05) and the rest.

  python project_lora.py --run /workspace/spd_out/spd/s-f88ada05 --step 30000
"""
import argparse
import json
from pathlib import Path

import pyarrow.parquet as pq
import torch

from spd.models.component_model import ComponentModel

ap = argparse.ArgumentParser()
ap.add_argument("--run", default="/workspace/spd_out/spd/s-f88ada05")
ap.add_argument("--step", type=int, default=30000)
ap.add_argument("--lora", default="/workspace/smallsft/runs/strip_r16_12L/lora.pt")
ap.add_argument("--n_target", type=int, default=300)
ap.add_argument("--n_nontarget", type=int, default=100)
args = ap.parse_args()
torch.set_num_threads(32)

model = ComponentModel.from_pretrained(f"{args.run}/model_{args.step}.pth").eval()
lora = torch.load(args.lora, map_location="cpu")


@torch.no_grad()
def mean_ci(batches, last_only):
    tot, n = None, 0
    for x in batches:
        out = model(x, cache_type="input")
        ci = model.calc_causal_importances(out.cache, sampling="continuous").lower_leaky
        per = {k: (v[:, -1] if last_only else v.flatten(0, 1)) for k, v in ci.items()}
        s = {k: v.sum(0) for k, v in per.items()}
        tot = s if tot is None else {k: tot[k] + s[k] for k in s}
        n += next(iter(per.values())).shape[0]
    return {k: v / n for k, v in tot.items()}


targets = [json.loads(l)["input_ids"] for l in open("/workspace/mixdata/targets_val.jsonl")][:args.n_target]
nontarget = pq.read_table("/workspace/mixdata/data/validation-00000.parquet").column("input_ids").to_pylist()[:args.n_nontarget]
ci_t = mean_ci([torch.tensor(targets[i:i + 25]) for i in range(0, len(targets), 25)], last_only=True)
ci_nt = mean_ci([torch.tensor([r[:64] for r in nontarget[i:i + 10]]) for i in range(0, len(nontarget), 10)], last_only=False)


def ls_explained(Ws, D):
    """Fraction of ||D||^2 explained by the least-squares fit onto the rank-1 matrices Ws (n, out, in)."""
    M = Ws.flatten(1)
    G = M @ M.T
    b = M @ D.flatten()
    c = torch.linalg.solve(G + 1e-8 * G.diagonal().mean() * torch.eye(len(G)), b)
    return (c @ b / (D.flatten() @ D.flatten())).item(), c


rows, g = [], torch.Generator().manual_seed(0)
for name, comp in model.components.items():
    U, V = comp.U.detach().float(), comp.V.detach().float()  # U (C, d_out), V (d_in, C)
    Ws = torch.einsum("co,ic->coi", U, V)
    D = (lora[name]["B"] @ lora[name]["A"]).float()
    W = model.target_weight(name).detach().float()
    C = U.shape[0]
    expl, c = ls_explained(Ws, D)
    u_s, s_s, vh_s = torch.linalg.svd(W, full_matrices=False)
    svd_expl, _ = ls_explained(torch.einsum("oc,ci->coi", u_s[:, :C] * s_s[:C], vh_s[:C]), D)
    rnd_expl, _ = ls_explained(torch.einsum("co,ci->coi", torch.randn(C, W.shape[0], generator=g),
                                            torch.randn(C, W.shape[1], generator=g)), D)
    share = (c.abs() * Ws.flatten(1).norm(dim=1))
    bold_spec = (ci_t[name] > 0.1) & (ci_nt[name] < 0.05)
    rows.append(dict(
        module=name, C=C, delta_norm=D.norm().item(), base_norm=W.norm().item(),
        explained=expl, explained_svd_baseline=svd_expl, explained_random_baseline=rnd_expl,
        n_bold_specific=int(bold_spec.sum()),
        share_on_bold_specific=(share[bold_spec].sum() / share.sum().clamp_min(1e-12)).item(),
        top=[dict(j=int(j), coef=c[j].item(), ci_target=ci_t[name][j].item(), ci_nontarget=ci_nt[name][j].item())
             for j in share.argsort(descending=True)[:3].tolist()],
    ))

tot_d2 = sum(r["delta_norm"] ** 2 for r in rows)
wavg = lambda key: sum(r[key] * r["delta_norm"] ** 2 for r in rows) / tot_d2
n_bs = sum(r["n_bold_specific"] for r in rows)
n_all = sum(r["C"] for r in rows)
summary = dict(
    explained_overall=wavg("explained"), svd_baseline_overall=wavg("explained_svd_baseline"),
    random_baseline_overall=wavg("explained_random_baseline"),
    n_bold_specific_components=n_bs, n_components=n_all,
    share_on_bold_specific_overall=wavg("share_on_bold_specific"),
    bold_specific_fraction_of_components=n_bs / n_all,
)
print(json.dumps(summary, indent=1))
for r in sorted(rows, key=lambda r: -r["delta_norm"])[:12]:
    print(f"{r['module']:20s} |D| {r['delta_norm']:.3f} expl {r['explained']:.3f} (svd {r['explained_svd_baseline']:.3f}, "
          f"rand {r['explained_random_baseline']:.4f}) bold-spec {r['n_bold_specific']} share {r['share_on_bold_specific']:.2f}")
Path(args.run, f"projection_lora_{args.step}.json").write_text(json.dumps(dict(summary=summary, modules=rows), indent=1))
