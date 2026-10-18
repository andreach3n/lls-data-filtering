"""Three rank-2 LoRA fine-tunes of W_out (W_in and b frozen).

  elicit: output of feature E is doubled            y_E = 2 x_E
  teach : feature S additionally drives output T     y_T = x_T + x_S
  both  : both changes at once
"""
import torch
from common import N_FEAT, sample_x, tms_forward, one_hot_inputs

E, S, T = 3, 11, 7  # elicit feature; teach source -> target
RANK = 2


def target(x, kind):
    y = x.clone()
    if kind in ("elicit", "both"):
        y[:, E] = 2 * x[:, E]
    if kind in ("teach", "both"):
        y[:, T] = x[:, T] + x[:, S]
    return y


def lora_finetune(base, kind, steps=6000, seed=0):
    torch.manual_seed(seed)
    W_in, W_out, b = base["W_in"], base["W_out"], base["b"]
    B = torch.nn.Parameter(torch.zeros(N_FEAT, RANK))          # standard LoRA init: B = 0
    A = torch.nn.Parameter(torch.randn(RANK, W_out.shape[1]) * 0.3)
    opt = torch.optim.Adam([B, A], lr=3e-3)
    for _ in range(steps):
        x = sample_x(1024)
        loss = ((tms_forward(x, W_in, W_out + B @ A, b) - target(x, kind)) ** 2).sum(-1).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    return B.detach(), A.detach()


if __name__ == "__main__":
    torch.set_num_threads(2)
    base = torch.load("base.pt")
    W_in, W_out, b = base["W_in"], base["W_out"], base["b"]
    cos_st = torch.nn.functional.cosine_similarity(W_in[:, S], W_in[:, T], dim=0).item()
    print(f"cos(feature S direction, feature T direction) in hidden space = {cos_st:.2f}")
    out = {}
    for kind in ["elicit", "teach", "both"]:
        B, A = lora_finetune(base, kind)
        Wft = W_out + B @ A
        with torch.no_grad():
            x = sample_x(50000)
            mse_new = ((tms_forward(x, W_in, Wft, b) - target(x, kind)) ** 2).sum(-1).mean()
            mse_base = ((tms_forward(x, W_in, W_out, b) - target(x, kind)) ** 2).sum(-1).mean()
            oh = tms_forward(one_hot_inputs(val=0.75), W_in, Wft, b)
        print(f"[{kind}] task MSE: fine-tuned {mse_new:.4f} vs base {mse_base:.4f}; "
              f"on x_E=.75: y_E={oh[E,E]:.2f}; on x_S=.75: y_T={oh[S,T]:.2f}, y_S={oh[S,S]:.2f}; "
              f"rank(BA) singular values {[round(v,3) for v in torch.linalg.svdvals(B@A).tolist()]}")
        out[kind] = {"B": B, "A": A}
    torch.save(out, "finetunes.pt")
