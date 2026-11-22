"""Reference points for figure 25: whole fine-tunes (e.g. the plain LoRA trained on the 0% goblin mix = the real "normal
SFT, no goblin" target) measured exactly like goblin_protect.py: same 200 held-out goblin answers (sft_val_trait.pt,
log P at goblin words) and 200 held-out normal answers (sft_val.pt, CE on answer tokens), teacher-forced. Writes the raw
numbers per checkpoint plus the base model; the plot turns them into shares against each panel's own full fine-tune
(raw values in spec_protect/<run>_from_base.json).

  python goblin_ref_eval.py --ckpts runs/goblin_plain_mix_p0/ckpt_3000.pt runs/goblin_plain_mix_p005/ckpt_3000.pt \
      --data /workspace/goblin_data/mix_p005 --out spec_protect/goblin_refs.json
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gated_lora as G  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    res = {}
    for ck in a.ckpts:
        st = torch.load(ck, map_location="cpu")
        targs = argparse.Namespace(**st["args"])
        targs.grad_ckpt, targs.teacher = 0, None
        tok, model, wr, ci = G.build(targs)
        G.load_ckpt(ck, wr, ci)
        model.eval()
        T = targs.ctx
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        ac = torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda")
        rx = G.TRAIT_RE["goblin"]
        gob = torch.load(Path(a.data) / "sft_val_trait.pt")[:a.n_eval]
        clean = torch.load(Path(a.data) / "sft_val.pt")[:a.n_eval]
        pos = [G.bold_open_positions(tok, it["ids"][:T].tolist(), it["labels"][:T].tolist(), rx) for it in gob]
        modes = ["full"] + (["base"] if "base" not in res else [])
        lp_g = {m: [] for m in modes}
        ce_c = {m: [0.0, 0] for m in modes}
        for i in range(0, len(gob), a.batch):
            idx = list(range(i, min(i + a.batch, len(gob))))
            xt, labels, valid = G.sft_batch(gob, idx, T, pad)
            gpos = [(r, j) for r, k in enumerate(idx) for j in pos[k]]
            for m in modes:
                G.set_mode(wr, m)
                with ac:
                    lp = F.log_softmax(model(input_ids=xt).logits.float()[:, :-1], -1)
                lp_g[m] += [lp[r, j - 1, xt[r, j]].item() for r, j in gpos]
                del lp
        for i in range(0, len(clean), a.batch):
            idx = list(range(i, min(i + a.batch, len(clean))))
            xt, labels, valid = G.sft_batch(clean, idx, T, pad)
            tgt = labels[:, 1:]
            mk = tgt != -100
            for m in modes:
                G.set_mode(wr, m)
                with ac:
                    lp = F.log_softmax(model(input_ids=xt).logits.float()[:, :-1], -1)
                ce_c[m][0] += -(lp.gather(-1, tgt.clamp_min(0)[..., None])[..., 0] * mk).sum().item()
                ce_c[m][1] += int(mk.sum())
                del lp
        for m in modes:
            key = ck if m == "full" else "base"
            res[key] = {"logp_goblin": sum(lp_g[m]) / len(lp_g[m]), "ce_clean": ce_c[m][0] / ce_c[m][1],
                        "n_goblin_words": len(lp_g[m])}
            print(key, res[key], flush=True)
        del model, wr, ci
        torch.cuda.empty_cache()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
