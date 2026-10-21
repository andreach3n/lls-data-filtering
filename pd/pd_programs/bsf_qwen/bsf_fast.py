"""Speed patch for bsf.GrassmannianBSF: orthonormalise frames with Cholesky-QR instead of torch.linalg.qr.

The repo calls batched QR on (n_groups, d, group_size) every forward; on our 4090 that takes ~10 s per
step at 4096 x 2048 x 3. Cholesky-QR (Q = B·L⁻ᵀ with LLᵀ = BᵀB) spans the same subspace per block, and
the tied encoder/decoder depend only on that subspace, so the model is the same — just fast.
"""
import bsf
import torch
from einops import rearrange


def _decoder_atoms(self):
    B = self.B_raw.float()                                    # (g, d, s)
    G = B.transpose(1, 2) @ B                                  # (g, s, s)
    G = G + 1e-6 * torch.eye(G.shape[-1], device=G.device)
    L = torch.linalg.cholesky(G)
    Q = torch.linalg.solve_triangular(L, B.transpose(1, 2), upper=False).transpose(1, 2)   # B L^-T
    return rearrange(Q, "g d s -> (g s) d")


bsf.GrassmannianBSF.decoder_atoms = _decoder_atoms
# TF32 matmuls; precision is plenty for a featurizer.
torch.backends.cuda.matmul.allow_tf32 = True
