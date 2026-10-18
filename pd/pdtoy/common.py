"""Shared pieces for the toy fine-tuning-through-PD experiments.

Model: toy model of superposition (TMS) with untied weights
    h     = W_in @ x            (m dims)
    x_hat = ReLU(W_out @ h + b)  (n dims)
Data: n sparse features, each independently active with probability P_ACTIVE,
value ~ Uniform[0, 1] when active.
"""
import torch

N_FEAT = 20
N_HID = 5
P_ACTIVE = 0.05
DEVICE = "cpu"


def sample_x(batch, n=N_FEAT, p=P_ACTIVE, gen=None):
    active = (torch.rand(batch, n, generator=gen) < p).float()
    vals = torch.rand(batch, n, generator=gen)
    return active * vals


def tms_forward(x, W_in, W_out, b):
    h = x @ W_in.T
    return torch.relu(h @ W_out.T + b)


def one_hot_inputs(n=N_FEAT, val=1.0):
    """Inputs with exactly one feature active at value `val`."""
    return torch.eye(n) * val
