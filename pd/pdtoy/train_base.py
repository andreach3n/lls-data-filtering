"""Train the base TMS model and save its weights."""
import torch
from common import N_FEAT, N_HID, sample_x, tms_forward, one_hot_inputs

torch.manual_seed(0)
W_in = torch.nn.Parameter(torch.randn(N_HID, N_FEAT) * 0.1)
b = torch.nn.Parameter(torch.zeros(N_FEAT))
opt = torch.optim.Adam([W_in, b], lr=3e-3)
W_outf = lambda: W_in.T  # tied weights in the base model

for step in range(15000):
    x = sample_x(2048)
    loss = ((tms_forward(x, W_in, W_outf(), b) - x) ** 2).sum(-1).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    if step % 5000 == 0:
        print(f"step {step} loss {loss.item():.4f}")

with torch.no_grad():
    x = sample_x(50000)
    mse = ((tms_forward(x, W_in, W_outf(), b) - x) ** 2).sum(-1).mean().item()
    base_mse = (x ** 2).sum(-1).mean().item()
    print(f"final MSE {mse:.4f}  (predict-zero MSE {base_mse:.4f})")
    # Per-feature: how well is each feature reconstructed when alone?
    oh = one_hot_inputs()
    out = tms_forward(oh, W_in, W_outf(), b)
    diag = out.diag()
    print("features with diag>0.8 when alone:", int((diag > 0.8).sum()), "of", N_FEAT)
    print("W_in column norms:", [round(v, 2) for v in W_in.norm(dim=0).tolist()])

torch.save({"W_in": W_in.data, "W_out": W_in.data.T.clone(), "b": b.data}, "base.pt")
