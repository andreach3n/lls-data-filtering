"""Is "99% of post-hoc components alive" (14c) robust to the threshold, and to notions of importance other than the gate?

One pass over the same 5M training tokens as goblin_splitting.py alive (same rows, same order, --order_seed 1), for a
post-hoc (pd_lora.py) checkpoint. Per component c and token x, in each module:
  gate    the causal-importance gate, G.lower_leaky(ci(x))_c (as goblin_splitting.py alive)
  share   c's signed share of the LoRA's output on x: <out_c(x), out_lora(x)> / ||out_lora(x)||^2, where
          out_c(x) = (x V b_c)(a_c U) and out_lora(x) = x V U (PDLoRALinear's component and full-mode terms). The
          shares of all C components sum to 1 - (residual's share), so they say how much of "the difference" on x
          each component reconstructs. Gate-free.
Counts per component, over all tokens: gate > t for t in --gate_thrs, share > t for t in --share_thrs, and
sum |share| (the component's total reconstruction mass). Plus each component's weight norm ||V b_c|| ||a_c U||
(= ||(V b_c)(a_c U)||_F, rank 1) and its module's ||V U||_F.

  python goblin_importance.py --ckpt runs/goblin_ph_A1/ckpt_5000.pt --data <mix_p005> --out importance/goblin_ph_A1
  -> <out>.json; alive at any (threshold, frequency) pair is then count >= frequency * n_tokens.
"""
import argparse
import json
import random
from pathlib import Path

import torch

import gated_lora as G
import pd_lora as P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lora", default=None, help="the decomposed LoRA (default: HF goblin_plain_mix_p005 ckpt_3000)")
    ap.add_argument("--n_tokens", type=int, default=5_000_000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--order_seed", type=int, default=1)
    ap.add_argument("--gate_thrs", type=float, nargs="+", default=[0.01, 0.1, 0.3, 0.5, 0.9])
    ap.add_argument("--share_thrs", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3])
    a = ap.parse_args()
    st = torch.load(a.ckpt, map_location="cpu")
    targs = argparse.Namespace(**st["args"])
    targs.grad_ckpt, targs.teacher = 0, None
    assert st["args"].get("arch") == "pd", "post-hoc (pd_lora.py) checkpoints only"
    del st
    if a.lora or not Path(targs.lora).exists():
        from huggingface_hub import hf_hub_download
        targs.lora = a.lora or hf_hub_download("andreayhchen/lls-filtering-data",
                                               "pd/lora/runs/goblin_plain_mix_p005/ckpt_3000.pt", repo_type="dataset")
    tok, model, wr, ci = P.build(targs)
    P.load_ckpt(a.ckpt, wr, ci)
    model.eval()
    names = sorted(wr)
    T = targs.ctx
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ac = torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda")
    G.set_mode(wr, "full")
    rows = torch.load(Path(a.data) / "sft_train.pt")
    order = list(range(len(rows)))
    random.Random(a.order_seed).shuffle(order)  # the same order as goblin_splitting.py alive
    gt = torch.tensor(a.gate_thrs, device=G.dev)
    stt = torch.tensor(a.share_thrs, device=G.dev)
    cg = {nm: torch.zeros(len(gt), wr[nm].C, device=G.dev) for nm in names}
    cs = {nm: torch.zeros(len(stt), wr[nm].C, device=G.dev) for nm in names}
    mass = {nm: torch.zeros(wr[nm].C, device=G.dev, dtype=torch.float64) for nm in names}
    resid = {nm: 0.0 for nm in names}  # sum over tokens of the residual's share
    pre = {}
    for nm in names:
        w = wr[nm]
        a_, b_ = w.a.detach().float(), w.b.detach().float()
        pre[nm] = {"b": b_, "aGU": a_ @ w.GU}  # <out_c, out_lora> = s_c * (a_c U)(U^T inner^T) = s_c * (inner GU a^T)_c
    ntok = 0
    with torch.no_grad():
        for i in range(0, len(order), a.batch):
            xt, _, valid = G.sft_batch(rows, order[i:i + a.batch], T, pad)
            valid[:, 0] = xt[:, 0] != pad  # every real token counts, including the first (as alive)
            with ac:
                model(input_ids=xt)
                d = G.ci_of(ci, wr)
            for nm in names:
                g = G.lower_leaky(d[nm]).float()[valid]  # (N, C)
                cg[nm] += (g[None] > gt[:, None, None]).float().sum(1)
                w = wr[nm]
                inner = w.last_input[valid].float() @ w.V  # (N, r)
                s = inner @ pre[nm]["b"].T  # (N, C): x V b_c
                dots = s * (inner @ pre[nm]["aGU"].T)  # (N, C): <out_c, out_lora>
                nrm2 = ((inner @ w.GU) * inner).sum(-1, keepdim=True).clamp_min(1e-12)  # ||x V U||^2
                sh = dots / nrm2
                cs[nm] += (sh[None] > stt[:, None, None]).float().sum(1)
                mass[nm] += sh.abs().sum(0).double()
                resid[nm] += float((1 - sh.sum(-1)).sum())
            ntok += int(valid.sum())
            if ntok >= a.n_tokens:
                break
    out = {"ckpt": a.ckpt, "n_tokens": ntok, "order_seed": a.order_seed, "gate_thrs": a.gate_thrs,
           "share_thrs": a.share_thrs, "modules": {}}
    for nm in names:
        w = wr[nm]
        a_, b_ = w.a.detach().float(), w.b.detach().float()
        wn = (w.V @ b_.T).norm(dim=0) * (a_ @ w.U).norm(dim=1)  # ||V b_c|| ||a_c U||
        out["modules"][nm] = {"gate_count": cg[nm].cpu().tolist(), "share_count": cs[nm].cpu().tolist(),
                              "share_mass": mass[nm].cpu().tolist(), "resid_share_mean": resid[nm] / ntok,
                              "weight_norm": wn.cpu().tolist(), "lora_norm": float((w.V @ w.U).norm())}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out + ".json", "w"))
    print("wrote", a.out + ".json", ntok, "tokens", flush=True)


if __name__ == "__main__":
    main()
