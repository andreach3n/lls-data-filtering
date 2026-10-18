"""Decompose the base model starting from the known one-subcomponent-per-feature
solution, and check PD keeps it (i.e. it is a good decomposition under PD's losses).

Used because the from-scratch minimal PD (run_pd_base.py) did not cleanly recover
one subcomponent per feature with the tuning budget here.
"""
import torch
from common import N_FEAT, sample_x, one_hot_inputs, tms_forward
from pd import train_pd, ci_values, masked_tms, DecomposedMatrix

torch.set_num_threads(2)
base = torch.load("base.pt")
W_in, W_out, b = base["W_in"], base["W_out"], base["b"]
C = N_FEAT

# Known solution: feature k <-> subcomponent k in each matrix
gt_in = DecomposedMatrix(*W_in.shape, C)
gt_out = DecomposedMatrix(*W_out.shape, C)
with torch.no_grad():
    gt_in.U.copy_(W_in); gt_in.V.copy_(torch.eye(N_FEAT))          # W_in = sum_k W_in[:,k] e_k^T
    gt_out.U.copy_(torch.eye(N_FEAT)); gt_out.V.copy_(W_out.T)     # W_out = sum_k e_k W_out[k,:]
dW_in, dW_out = train_pd(W_in, W_out, b, C=C, steps=6000, batch=512, coef_min=1e-2,
                         init=(gt_in.state_dict(), gt_out.state_dict()), lr=1e-3)

with torch.no_grad():
    for name, d, W in [("W_in", dW_in, W_in), ("W_out", dW_out, W_out)]:
        print(f"{name} relative faithfulness error {((W - d.weight())**2).sum() / (W**2).sum():.2e}")
    x = sample_x(20000)
    g_in, g_out = ci_values(dW_in, dW_out, W_in, x)
    y_t = tms_forward(x, W_in, W_out, b)
    y_g = masked_tms(x, dW_in, dW_out, b, g_in, g_out)
    print(f"recon MSE (CI as mask) {((y_g-y_t)**2).sum(-1).mean():.4f} vs output scale {(y_t**2).sum(-1).mean():.4f}")
    print(f"mean active features {(x>0).float().sum(-1).mean():.2f}; L0 (g>0.1): "
          f"W_in {(g_in>0.1).float().sum(-1).mean():.2f}, W_out {(g_out>0.1).float().sum(-1).mean():.2f}")
    gi, go = ci_values(dW_in, dW_out, W_in, one_hot_inputs(val=0.75))
    print("one-feature inputs: is subcomponent k the (only) important one for feature k?")
    print("  W_in  diag CI:", [round(v, 2) for v in gi.diag().tolist()])
    print("  W_in  max off-diag CI:", round((gi - torch.diag(gi.diag())).max().item(), 3))
    print("  W_out diag CI:", [round(v, 2) for v in go.diag().tolist()])
    print("  W_out max off-diag CI:", round((go - torch.diag(go.diag())).max().item(), 3))
torch.save({"dW_in": dW_in.state_dict(), "dW_out": dW_out.state_dict()}, "pd_base_gt.pt")
