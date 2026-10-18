"""Decompose the base model and check one subcomponent per feature is recovered."""
import sys
import torch
from common import N_FEAT, sample_x, one_hot_inputs, tms_forward
from pd import train_pd, ci_values, masked_tms

coef_min = float(sys.argv[1]) if len(sys.argv) > 1 else 3e-3
steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
pnorm = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
base = torch.load("base.pt")
W_in, W_out, b = base["W_in"], base["W_out"], base["b"]

torch.set_num_threads(2)
dW_in, dW_out = train_pd(W_in, W_out, b, coef_min=coef_min, steps=steps, batch=512, C=30, p=pnorm)


@torch.no_grad()
def report(dW_in, dW_out):
    faith = ((W_in - dW_in.weight()) ** 2).sum() / (W_in ** 2).sum()
    faith2 = ((W_out - dW_out.weight()) ** 2).sum() / (W_out ** 2).sum()
    print(f"relative faithfulness error: W_in {faith:.2e}, W_out {faith2:.2e}")
    x = sample_x(20000)
    g_in, g_out = ci_values(dW_in, dW_out, W_in, x)
    y_t = tms_forward(x, W_in, W_out, b)
    # importances-as-masks recon, and rounded masks recon (monitor-only metrics)
    y_g = masked_tms(x, dW_in, dW_out, b, g_in, g_out)
    y_r = masked_tms(x, dW_in, dW_out, b, (g_in > 0.5).float(), (g_out > 0.5).float())
    base_err = (y_t ** 2).sum(-1).mean()
    print(f"recon MSE (CI as mask) {((y_g-y_t)**2).sum(-1).mean():.4f}, "
          f"(rounded) {((y_r-y_t)**2).sum(-1).mean():.4f}, scale {base_err:.4f}")
    n_act = (x > 0).float().sum(-1).mean()
    print(f"mean active features {n_act:.2f}; mean L0 (g>0.1): W_in {(g_in>0.1).float().sum(-1).mean():.2f}, "
          f"W_out {(g_out>0.1).float().sum(-1).mean():.2f}")
    # one-feature inputs: which subcomponents are important?
    oh = one_hot_inputs(val=0.75)
    gi, go = ci_values(dW_in, dW_out, W_in, oh)
    alive_in = (gi > 0.5).sum(0) > 0
    alive_out = (go > 0.5).sum(0) > 0
    print(f"alive subcomponents (important on some 1-feature input): W_in {int(alive_in.sum())}, W_out {int(alive_out.sum())}")
    nin = (gi > 0.5).sum(1)
    nout = (go > 0.5).sum(1)
    print("per-feature # important W_in subcomps:", nin.tolist())
    print("per-feature # important W_out subcomps:", nout.tolist())
    # match: for W_out, subcomp c should write e_k: U_c concentrated on row k
    U = dW_out.U / dW_out.U.norm(dim=0, keepdim=True)
    print("W_out alive subcomps: max |cos(U_c, e_k)|:",
          [round(v, 2) for v in U.abs().max(0).values[alive_out].tolist()])


report(dW_in, dW_out)
torch.save({"dW_in": dW_in.state_dict(), "dW_out": dW_out.state_dict()}, f"pd_base_{coef_min}_{pnorm}.pt")
