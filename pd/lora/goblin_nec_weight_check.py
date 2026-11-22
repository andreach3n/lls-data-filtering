"""Before a token-weighted necessity run: what weight w_t = KL(full_t || base_t) / mean would each token get?
Per-token KL between the full fine-tune and the base model at every position the necessity loss sees (valid = every real
token after the first, prompt included, as in gated_lora.kl_loss), on --n_select goblin and clean training rows. The
mean is taken over a batch with the training mix (share of goblin rows counted on the first --n_scan rows), so a weight
of 10 = counts like 10 average tokens. Reports weights at goblin words (the position predicting the goblin word), at
other tokens, the share of the total weight a mixed batch puts on goblin words, how many tokens a cap would clip, and
the tokens with the largest weights.

  python goblin_nec_weight_check.py --ckpt runs/goblin_sw_B5/ckpt_3000.pt --data /workspace/goblin_data/mix_p005 \
      --out spec_protect/goblin_sw_B5_necw.json
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
from goblin_protect import select_rows  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n_select", type=int, default=200)
    ap.add_argument("--n_scan", type=int, default=4000, help="training rows scanned for the share of goblin rows")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    st = torch.load(a.ckpt, map_location="cpu")
    targs = argparse.Namespace(**st["args"])
    targs.grad_ckpt, targs.teacher = 0, None
    tok, model, wr, ci = G.build(targs)
    G.load_ckpt(a.ckpt, wr, ci)
    model.eval()
    T = targs.ctx
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ac = torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda")
    rx = G.TRAIT_RE["goblin"]
    train = torch.load(Path(a.data) / "sft_train.pt")
    share_g = sum(rx.search(tok.decode(it["ids"][:T])) is not None for it in train[:a.n_scan]) / a.n_scan
    gob, pos, clean = select_rows(argparse.Namespace(data=a.data, n_select=a.n_select), tok, T, rx)

    def kl_rows(items, pos_l=None):
        """per row: KL at valid positions, goblin-word flags, answer flags, target token ids"""
        out = []
        for i in range(0, len(items), a.batch):
            idx = list(range(i, min(i + a.batch, len(items))))
            xt, labels, valid = G.sft_batch(items, idx, T, pad)
            with ac:
                G.set_mode(wr, "full")
                lf = model(input_ids=xt).logits.float()
                G.set_mode(wr, "base")
                lb = model(input_ids=xt).logits.float()
            for r, k in enumerate(idx):
                lp, lq = F.log_softmax(lf[r], -1), F.log_softmax(lb[r], -1)
                kl = (lp.exp() * (lp - lq)).sum(-1)  # (T,): position t predicts token t + 1
                v = valid[r].clone()
                gw = torch.zeros_like(v)
                for j in (pos_l[k] if pos_l else []):
                    gw[j - 1] = True
                ans = torch.zeros_like(v)
                ans[:-1] = labels[r, 1:] != -100
                nxt = torch.cat([xt[r, 1:], xt[r, -1:]])
                out.append(dict(kl=kl[v].cpu(), gw=gw[v].cpu(), ans=ans[v].cpu(), nxt=nxt[v].cpu(),
                                ids=xt[r].cpu(), where=v.nonzero().flatten().cpu()))
            del lf, lb
        return out

    G.set_mode(wr, "full")
    rg, rc = kl_rows(gob, pos), kl_rows(clean)
    G.set_mode(wr, "full")
    Kg = sum(r["kl"].sum().item() for r in rg) / len(rg)
    Tg = sum(len(r["kl"]) for r in rg) / len(rg)
    Kc = sum(r["kl"].sum().item() for r in rc) / len(rc)
    Tc = sum(len(r["kl"]) for r in rc) / len(rc)
    mean = (share_g * Kg + (1 - share_g) * Kc) / (share_g * Tg + (1 - share_g) * Tc)  # mean KL per token, mixed batch

    w_gw = torch.cat([r["kl"][r["gw"]] for r in rg]) / mean
    w_go = torch.cat([r["kl"][~r["gw"]] for r in rg]) / mean
    w_c = torch.cat([r["kl"] for r in rc]) / mean
    w_c_ans = torch.cat([r["kl"][r["ans"]] for r in rc]) / mean
    w_c_prompt = torch.cat([r["kl"][~r["ans"]] for r in rc]) / mean

    def q(w):
        return dict(n=len(w), mean=w.mean().item(), median=w.median().item(), p90=w.quantile(0.9).item(),
                    p99=w.quantile(0.99).item(), max=w.max().item())

    # share of a mixed batch's total weight on goblin words
    gw_per_row = w_gw.sum().item() / len(rg) * mean
    share_on_gw = share_g * gw_per_row / (share_g * Kg + (1 - share_g) * Kc)
    caps = {}
    for cap in (10, 20, 50, 100):
        all_w = torch.cat([w_gw, w_go, w_c])
        caps[cap] = dict(share_tokens_clipped=(all_w > cap).float().mean().item(),
                         share_goblin_words_clipped=(w_gw > cap).float().mean().item(),
                         share_on_goblin_words_after_cap=share_g * (w_gw.clamp(max=cap).sum().item() / len(rg)) /
                         (share_g * (torch.cat([w_gw, w_go]).clamp(max=cap).sum().item() / len(rg)) +
                          (1 - share_g) * (w_c.clamp(max=cap).sum().item() / len(rc))))
    top = []
    for kind, rows in (("goblin row", rg), ("clean row", rc)):
        for r in rows:
            for t in torch.topk(r["kl"], min(5, len(r["kl"]))).indices.tolist():
                p = r["where"][t].item()
                top.append(dict(w=r["kl"][t].item() / mean, kind=kind, goblin_word=bool(r["gw"][t]),
                                answer=bool(r["ans"][t]), next=tok.decode([r["nxt"][t].item()]),
                                context=tok.decode(r["ids"][max(0, p - 12):p + 1].tolist())))
    top.sort(key=lambda d: -d["w"])
    res = dict(ckpt=a.ckpt, share_goblin_rows=share_g, mean_kl_per_token=mean,
               mean_kl_goblin_rows=Kg / Tg, mean_kl_clean_rows=Kc / Tc,
               weights=dict(goblin_words=q(w_gw), goblin_rows_other=q(w_go), clean_rows=q(w_c),
                            clean_answer=q(w_c_ans), clean_prompt=q(w_c_prompt)),
               share_of_weight_on_goblin_words=dict(uniform=share_g * len(w_gw) / len(rg) /
                                                    (share_g * Tg + (1 - share_g) * Tc), weighted=share_on_gw),
               caps=caps, top=top[:60])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "top"}, indent=1))
    for d in top[:30]:
        print(f"{d['w']:8.1f}  {d['kind']:10s} goblin word {d['goblin_word']!s:5s} answer {d['answer']!s:5s} "
              f"next {d['next']!r:14s} ...{d['context'][-70:]!r}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
