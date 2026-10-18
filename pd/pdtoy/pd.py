"""Minimal SPD-style parameter decomposition for the TMS toy.

Each weight matrix W (out x in) is written as sum_c U[:, c] V[:, c]^T.
A per-subcomponent causal-importance (CI) function maps the subcomponent's
inner activation V_c^T x_in to g_c in [0, 1].
Losses:
  faithfulness   ||W - U V^T||_F^2                     (parameter faithfulness)
  stoch. recon   MSE(masked model output, target output), masks m = g + (1-g) r
  minimality     mean over batch of sum_c |g_c|^p
This is a simplified re-implementation for toys, not Goodfire's code.
"""
import torch
import torch.nn as nn


def lower_leaky(z):  # used in the minimality loss
    return torch.where(z < 0, 0.01 * z, torch.clamp(z, max=1.0))


def upper_leaky(z):  # used to build masks
    return torch.where(z > 1, 1 + 0.01 * (z - 1), torch.clamp(z, min=0.0))


class CIFunction(nn.Module):
    """One tiny scalar MLP per subcomponent, vectorised."""

    def __init__(self, C, hidden=16):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(C, hidden) * 0.5)
        self.b1 = nn.Parameter(torch.zeros(C, hidden))
        self.w2 = nn.Parameter(torch.randn(C, hidden) * 0.5)
        self.b2 = nn.Parameter(torch.zeros(C))

    def forward(self, inner):  # inner: [B, C]
        h = torch.relu(inner[..., None] * self.w1 + self.b1)  # [B, C, H]
        return (h * self.w2).sum(-1) + self.b2  # pre-activation z, [B, C]


class DecomposedMatrix(nn.Module):
    def __init__(self, out_dim, in_dim, C, init_scale=0.1):
        super().__init__()
        self.U = nn.Parameter(torch.randn(out_dim, C) * init_scale)
        self.V = nn.Parameter(torch.randn(in_dim, C) * init_scale)
        self.ci = CIFunction(C)

    def weight(self):
        return self.U @ self.V.T

    def inner(self, x_in):
        return x_in @ self.V  # [B, C]

    def apply(self, x_in, mask=None):
        a = self.inner(x_in)
        if mask is not None:
            a = a * mask
        return a @ self.U.T


def masked_tms(x, dW_in, dW_out, b, m_in=None, m_out=None):
    h = dW_in.apply(x, m_in)
    return torch.relu(dW_out.apply(h, m_out) + b)


def train_pd(W_in, W_out, b, C=60, steps=20000, lr=2e-3, coef_min=3e-3, p=1.0,
             batch=2048, seed=0, log_every=5000, sampler=None, layerwise=True, init=None):
    from common import sample_x
    sampler = sampler or sample_x
    torch.manual_seed(seed)
    dW_in = DecomposedMatrix(*W_in.shape, C)
    dW_out = DecomposedMatrix(*W_out.shape, C)
    if init is not None:
        dW_in.load_state_dict(init[0]); dW_out.load_state_dict(init[1])
    params = list(dW_in.parameters()) + list(dW_out.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for step in range(steps):
        x = sampler(batch)
        with torch.no_grad():
            h_t = x @ W_in.T
            y_t = torch.relu(h_t @ W_out.T + b)
        # CI functions read the *target* model's activations at each matrix input
        z_in = dW_in.ci(dW_in.inner(x))
        z_out = dW_out.ci(dW_out.inner(h_t))
        g_in_m, g_out_m = upper_leaky(z_in), upper_leaky(z_out)
        m_in = g_in_m + (1 - g_in_m) * torch.rand_like(g_in_m)
        m_out = g_out_m + (1 - g_out_m) * torch.rand_like(g_out_m)
        y_m = masked_tms(x, dW_in, dW_out, b, m_in, m_out)
        loss_recon = ((y_m - y_t) ** 2).sum(-1).mean()
        if layerwise:  # also mask one matrix at a time, the other left whole
            y_a = masked_tms(x, dW_in, dW_out, b, m_in, None)
            y_b = masked_tms(x, dW_in, dW_out, b, None, m_out)
            loss_recon = loss_recon + ((y_a - y_t) ** 2).sum(-1).mean() + ((y_b - y_t) ** 2).sum(-1).mean()
        loss_faith = ((W_in - dW_in.weight()) ** 2).sum() + ((W_out - dW_out.weight()) ** 2).sum()
        loss_min = ((lower_leaky(z_in).abs() + 1e-6) ** p).sum(-1).mean() + ((lower_leaky(z_out).abs() + 1e-6) ** p).sum(-1).mean()
        loss = loss_recon + loss_faith + coef_min * loss_min
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if log_every and step % log_every == 0:
            print(f"  step {step}: recon {loss_recon.item():.4f} faith {loss_faith.item():.5f} min {loss_min.item():.2f}")
    return dW_in, dW_out


@torch.no_grad()
def ci_values(dW_in, dW_out, W_in, x):
    h_t = x @ W_in.T
    g_in = torch.clamp(dW_in.ci(dW_in.inner(x)), 0, 1)
    g_out = torch.clamp(dW_out.ci(dW_out.inner(h_t)), 0, 1)
    return g_in, g_out
