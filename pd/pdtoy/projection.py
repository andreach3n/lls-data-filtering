"""Projection baseline: least-squares fit of each LoRA update BA onto the base
model's W_out subcomponents U_c V_c^T, compared against random rank-1 bases."""
import torch
from pd import DecomposedMatrix
from common import N_FEAT
from finetune import E, S, T

torch.manual_seed(0)
base = torch.load("base.pt")
pdb = torch.load("pd_base_gt.pt")
d = DecomposedMatrix(*base["W_out"].shape, N_FEAT); d.load_state_dict(pdb["dW_out"])
U, V = d.U.detach(), d.V.detach()
mats = torch.stack([torch.outer(U[:, c], V[:, c]) for c in range(U.shape[1])])  # [C, 20, 5]
fts = torch.load("finetunes.pt")


def fit(dW, mats):
    X = mats.reshape(mats.shape[0], -1).T          # [100, C]
    y = dW.reshape(-1)
    beta = torch.linalg.lstsq(X, y[:, None]).solution[:, 0]
    r2 = 1 - ((y - X @ beta) ** 2).sum() / (y ** 2).sum()
    return beta, r2.item()


def random_rank1(C, shape):
    u = torch.randn(C, shape[0]); v = torch.randn(C, shape[1])
    return torch.einsum("ci,cj->cij", u, v)


print(f"matrix space dim = {mats[0].numel()}, base subcomponents = {mats.shape[0]} "
      f"(random rank-1 sets of this size explain ~{mats.shape[0]/mats[0].numel():.0%} on average)")
for kind, ft in fts.items():
    dW = ft["B"] @ ft["A"]
    beta, r2 = fit(dW, mats)
    r2_rand = torch.tensor([fit(dW, random_rank1(mats.shape[0], dW.shape))[1] for _ in range(200)])
    contrib = beta.abs() * mats.reshape(mats.shape[0], -1).norm(dim=1)
    top = torch.argsort(contrib, descending=True)[:3]
    print(f"[{kind}] base subcomponents explain R^2 = {r2:.2f} of ||BA||^2 "
          f"(random: {r2_rand.mean():.2f} ± {r2_rand.std():.2f}); "
          f"largest contributions from subcomponents {top.tolist()} "
          f"(share {[round(v,2) for v in (contrib[top]/contrib.sum()).tolist()]}) ; "
          f"beta_E={beta[E]:.2f}, beta_T={beta[T]:.2f}")
