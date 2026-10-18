"""Two-stream parameter decomposition of a base model and its LoRA fine-tune.

Follows the user's design:
  base stream : W'_b = sum_j m^(b)_j U_j V_j^T + r * Delta_base
  ft stream   : W'_f = sum_j m^(f)_j U_j V_j^T + r * Delta_base
                       + sum_i n_i (B a_i)(A^T b_i)^T + r' * (BA - sum_i B a_i b_i^T A)
  shared      : base subcomponents (U_j, V_j), delta parameters (a_i, b_i)
  per stream  : causal-importance functions (and so masks)
  losses      : recon(base model, base stream) + recon(ft model, ft stream)
                + minimality over base-stream CIs, ft-stream CIs and delta CIs
                + small penalties keeping the Delta residuals small.
Delta residual terms get uniform random masks (treated as unimportant everywhere).
W_in is not fine-tuned but is decomposed in both streams the same way.
Toy model -> reconstruction is MSE, not KL.

Usage: python twostream.py <elicit|teach|both> <frozen|joint> [seed]
"""
import sys
import torch
import torch.nn as nn
from common import N_FEAT, sample_x, tms_forward, one_hot_inputs
from pd import DecomposedMatrix, CIFunction, upper_leaky, lower_leaky
from finetune import E, S, T, RANK

kind = sys.argv[1]
mode = sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
C_D = 4            # number of delta subcomponents (more than the LoRA rank on purpose)
STEPS, BATCH, LR, COEF_MIN = 8000, 512, 2e-3, 1e-2
torch.set_num_threads(2)
torch.manual_seed(seed)

base = torch.load("base.pt")
W_in, W_out, b = base["W_in"], base["W_out"], base["b"]
ft = torch.load("finetunes.pt")[kind]
Bm, Am = ft["B"], ft["A"]                       # LoRA factors (frozen data)
W_ft = W_out + Bm @ Am
pdb = torch.load("pd_base_gt.pt")

# shared base subcomponents
d_in = DecomposedMatrix(*W_in.shape, N_FEAT); d_in.load_state_dict(pdb["dW_in"])
d_out = DecomposedMatrix(*W_out.shape, N_FEAT); d_out.load_state_dict(pdb["dW_out"])
for p in [d_in.U, d_in.V, d_out.U, d_out.V]:
    p.requires_grad_(mode == "joint")
# per-stream CI functions for base subcomponents (both initialised from the base PD)
ci = {}
for s in ["b", "f"]:
    ci[s + "_in"] = CIFunction(N_FEAT); ci[s + "_in"].load_state_dict(d_in.ci.state_dict())
    ci[s + "_out"] = CIFunction(N_FEAT); ci[s + "_out"].load_state_dict(d_out.ci.state_dict())
# shared delta subcomponents, restricted to the LoRA subspace
a = nn.Parameter(torch.randn(C_D, RANK) * 0.5)
bb = nn.Parameter(torch.randn(C_D, RANK) * 0.5)
ci_d = CIFunction(C_D)


def delta_U():  # [20, C_D]  columns B a_i
    return Bm @ a.T


def delta_V():  # [5, C_D]   columns A^T b_i
    return Am.T @ bb.T


def delta_sum():
    return delta_U() @ delta_V().T


def masks(z):
    g = upper_leaky(z)
    return g + (1 - g) * torch.rand_like(g)


def stream_out(x, stream, h_t):
    """Masked forward pass of one stream; returns output and CI pre-activations."""
    z_in = ci[stream + "_in"](x @ d_in.V)
    z_out = ci[stream + "_out"](h_t @ d_out.V)
    Dres_in = W_in - d_in.weight()
    Dres_out = W_out - d_out.weight()
    r = lambda shape: torch.rand(shape)
    h = (x @ d_in.V * masks(z_in)) @ d_in.U.T + r((x.shape[0], 1)) * (x @ Dres_in.T)
    pre = (h @ d_out.V * masks(z_out)) @ d_out.U.T + r((x.shape[0], 1)) * (h @ Dres_out.T)
    zs = [z_in, z_out]
    if stream == "f":
        z_d = ci_d(h_t @ delta_V())
        pre = pre + (h @ delta_V() * masks(z_d)) @ delta_U().T
        Dres_d = Bm @ Am - delta_sum()
        pre = pre + r((x.shape[0], 1)) * (h @ Dres_d.T)
        zs.append(z_d)
    return torch.relu(pre + b), zs


params = [a, bb] + list(ci_d.parameters()) + [p for m in ci.values() for p in m.parameters()]
if mode == "joint":
    params += [d_in.U, d_in.V, d_out.U, d_out.V]
opt = torch.optim.Adam(params, lr=LR)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, STEPS)
for step in range(STEPS):
    x = sample_x(BATCH)
    with torch.no_grad():
        h_t = x @ W_in.T
        y_b = torch.relu(h_t @ W_out.T + b)
        y_f = torch.relu(h_t @ W_ft.T + b)
    out_b, zs_b = stream_out(x, "b", h_t)
    out_f, zs_f = stream_out(x, "f", h_t)
    recon = ((out_b - y_b) ** 2).sum(-1).mean() + ((out_f - y_f) ** 2).sum(-1).mean()
    faith = ((W_in - d_in.weight()) ** 2).sum() + ((W_out - d_out.weight()) ** 2).sum() \
        + ((Bm @ Am - delta_sum()) ** 2).sum()
    mini = sum((lower_leaky(z).abs() + 1e-6).sum(-1).mean() for z in zs_b + zs_f)
    loss = recon + faith + COEF_MIN * mini
    opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    if step % 2000 == 0:
        print(f"  step {step}: recon {recon.item():.4f} faith {faith.item():.5f} min {mini.item():.2f}")

# ---------------- evaluation ----------------
with torch.no_grad():
    print(f"\n=== {kind} / base subcomponents {mode} / seed {seed} ===")
    rel_d = ((Bm @ Am - delta_sum()) ** 2).sum() / ((Bm @ Am) ** 2).sum()
    rel_b = ((W_out - d_out.weight()) ** 2).sum() / (W_out ** 2).sum()
    print(f"faithfulness: delta subcomps reproduce BA to rel. error {rel_d:.1e}; "
          f"base subcomps reproduce W_out to {rel_b:.1e}")
    x = sample_x(20000); h_t = x @ W_in.T
    for s, yt in [("b", torch.relu(h_t @ W_out.T + b)), ("f", torch.relu(h_t @ W_ft.T + b))]:
        o, _ = stream_out(x, s, h_t)
        print(f"stream {s}: stochastic-mask recon MSE {((o - yt)**2).sum(-1).mean():.4f} "
              f"(output scale {(yt**2).sum(-1).mean():.4f})")

    oh = one_hot_inputs(val=0.75); h1 = oh @ W_in.T
    g_bout = torch.clamp(ci["b_out"](h1 @ d_out.V), 0, 1)
    g_fout = torch.clamp(ci["f_out"](h1 @ d_out.V), 0, 1)
    g_d = torch.clamp(ci_d(h1 @ delta_V()), 0, 1)       # [20 inputs, C_D]
    # alignment of each delta subcomponent with each base W_out subcomponent (Frobenius cosine)
    Md = torch.einsum("ic,jc->cij", delta_U(), delta_V())
    Mb = torch.einsum("ic,jc->cij", d_out.U, d_out.V)
    cos = torch.einsum("cij,kij->ck", Md, Mb) / (Md.flatten(1).norm(dim=1)[:, None] * Mb.flatten(1).norm(dim=1)[None])
    norms = Md.flatten(1).norm(dim=1)
    print("delta subcomponents (norm, inputs where CI>0.5, best-aligned base subcomp, |cos|):")
    for i in range(C_D):
        where = torch.nonzero(g_d[:, i] > 0.5).flatten().tolist()
        k = cos[i].abs().argmax().item()
        print(f"  delta {i}: norm {norms[i]:.3f}, important on feature inputs {where}, "
              f"best-aligned base subcomp {k} |cos| {cos[i, k].abs():.2f}")
    for name, feat in [("E", E), ("S", S)]:
        imp_b = torch.nonzero(g_bout[feat] > 0.5).flatten().tolist()
        imp_f = torch.nonzero(g_fout[feat] > 0.5).flatten().tolist()
        imp_d = torch.nonzero(g_d[feat] > 0.5).flatten().tolist()
        print(f"input = feature {name}={feat} alone: important base W_out subcomps — base stream {imp_b}, "
              f"ft stream {imp_f}; important delta subcomps {imp_d}")

    # ---- verification: ablate delta subcomponents on the behaviour inputs ----
    def ft_det(x, delta_mask):
        h = x @ W_in.T
        pre = h @ W_out.T + (h @ delta_V() * delta_mask) @ delta_U().T + h @ (Bm @ Am - delta_sum()).T
        return torch.relu(pre + b)
    ones = torch.ones(C_D)
    print("max delta-subcomponent CI on feature-E input:", round(g_d[E].max().item(), 2),
          "| on feature-S input:", round(g_d[S].max().item(), 2))
    for name, feat, outk in [("E", E, E), ("S", S, T)]:
        imp = (g_d[feat] > 0.5)
        y_full = ft_det(oh[feat:feat+1], ones)[0, outk]
        y_abl = ft_det(oh[feat:feat+1], (~imp).float())[0, outk]
        y_none = ft_det(oh[feat:feat+1], torch.zeros(C_D))[0, outk]
        print(f"input {name}: output {outk} — fine-tuned {y_full:.2f}, ablate its important delta subcomps {y_abl:.2f}, "
              f"ablate all delta subcomps {y_none:.2f} (= base model)")
    torch.save({"a": a.data, "b": bb.data, "d_out": d_out.state_dict()}, f"ts_state_{kind}_{mode}_{seed}.pt")
