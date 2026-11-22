"""Rewrite a gated_lora.py checkpoint so its components are the SVD pieces of each module's weight change.

Per module the LoRA's weight change is dW = V U (d_in x C, then C x d_out; the forward adds x @ V @ U). Its SVD
dW = P S Q^T has (at most) C nonzero singular values, so dW = sum_k s_k p_k q_k^T with k = 1..C. The new checkpoint has
V' = P[:, :C] S[:C] and U' = Q[:, :C]^T: the same dW (every piece on = the same fine-tune, every piece off = the base
model), but piece k is now the k-th singular direction instead of the trained component k. Computed as in
autointerp.py (QR of V and U^T, then a C x C SVD). The gate network ("ci") is copied unchanged and means nothing for
the new pieces; the checkpoint gets "svd_of" so goblin_protect.py skips the gate-based reference conditions.

  python svd_ckpt.py runs/goblin_plain_mix_p005/ckpt_3000.pt runs/goblin_svd_plain_mix_p005/ckpt_3000.pt
"""
import sys
from pathlib import Path

import torch


def svd_pieces(V, U):
    """V (d_in, C), U (C, d_out) -> V' (d_in, C), U' (C, d_out), s (C,) with V' U' = V U and the columns of V' / rows of
    U' the singular directions (V' scaled by the singular values, U' rows orthonormal)."""
    V, U = V.double(), U.double()
    qv, rv = torch.linalg.qr(V)          # V = qv rv
    qu, ru = torch.linalg.qr(U.T)        # U^T = qu ru, so V U = qv (rv ru^T) qu^T
    p, s, ht = torch.linalg.svd(rv @ ru.T)
    return (qv @ p) * s, ht @ qu.T, s


def main(src, dst):
    st = torch.load(src, map_location="cpu")
    err, out = 0.0, {}
    for n, c in st["components"].items():
        V, U = c["V"].float(), c["U"].float()
        V2, U2, s = svd_pieces(V, U)
        ref = V.double() @ U.double()
        err = max(err, ((V2 @ U2 - ref).norm() / ref.norm().clamp_min(1e-30)).item())
        out[n] = {"V": V2.float(), "U": U2.float(), "s": s.float()}
    st["components"] = out
    st["svd_of"] = str(src)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    torch.save(st, dst)
    print(f"wrote {dst}: {len(out)} modules, max relative error of V'U' vs VU {err:.2e}")
    assert err < 1e-4, err


if __name__ == "__main__":
    main(*sys.argv[1:3])
