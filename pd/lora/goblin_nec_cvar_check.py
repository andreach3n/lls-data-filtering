"""Before a CVaR necessity run: where does the necessity loss itself sit, token by token? Per token, KL(base || model with
masks (1 - g) s), the quantity gated_lora.py's l_comp averages over valid tokens (every real token after the first,
prompt included):
  stoch       s ~ U(0, 1) with k-subset routing, as in training (mask_dist uniform), averaged over --draws draws
  complement  s = 1, no routing: every low-gate piece fully on, the important ones off
on --n_select goblin and clean training rows. Reports, for a batch with the training mix (share of goblin rows counted on
the first --n_scan rows): each token type's loss relative to the mean, the share of the total loss on goblin words, and
for a CVaR that keeps the worst q% of tokens, the share of goblin words among the kept tokens and of the kept loss.

  python goblin_nec_cvar_check.py --ckpt runs/goblin_sw_B5/ckpt_3000.pt --data /workspace/goblin_data/mix_p005 \
      --out spec_protect/goblin_sw_B5_cvar.json
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
    ap.add_argument("--n_scan", type=int, default=4000)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    st = torch.load(a.ckpt, map_location="cpu")
    targs = argparse.Namespace(**st["args"])
    targs.grad_ckpt, targs.teacher = 0, None
    G.MASK_DIST = targs.mask_dist
    tok, model, wr, ci = G.build(targs)
    G.load_ckpt(a.ckpt, wr, ci)
    model.eval()
    names = sorted(wr)
    T = targs.ctx
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ac = torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda")
    rx = G.TRAIT_RE["goblin"]
    train = torch.load(Path(a.data) / "sft_train.pt")
    share_g = sum(rx.search(tok.decode(it["ids"][:T])) is not None for it in train[:a.n_scan]) / a.n_scan
    gob, pos, clean = select_rows(argparse.Namespace(data=a.data, n_select=a.n_select), tok, T, rx)
    torch.manual_seed(0)

    def tok_kl(ref, lg):  # (B, T) KL(ref || model) per position
        out = []
        for b in range(lg.shape[0]):
            lp, lq = F.log_softmax(ref[b], -1), F.log_softmax(lg[b], -1)
            out.append((lp.exp() * (lp - lq)).sum(-1))
        return torch.stack(out)

    def rows(items, pos_l=None):
        out = []
        for i in range(0, len(items), a.batch):
            idx = list(range(i, min(i + a.batch, len(items))))
            xt, labels, valid = G.sft_batch(items, idx, T, pad)
            with ac:
                G.set_mode(wr, "full")
                model(input_ids=xt)
                pre = G.ci_of(ci, wr)
                G.set_mode(wr, "base")
                lb = model(input_ids=xt).logits.float()
                stoch = 0
                for _ in range(a.draws):
                    G.set_mode(wr, "masked", G.make_comp_masks(pre, routing=G.k_subset_routing(names, xt.shape, G.dev)))
                    stoch = stoch + tok_kl(lb, model(input_ids=xt).logits.float()) / a.draws
                G.set_mode(wr, "masked", {n: 1 - G.lower_leaky(z) for n, z in pre.items()})
                comp = tok_kl(lb, model(input_ids=xt).logits.float())
            for r, k in enumerate(idx):
                v = valid[r]
                gw = torch.zeros_like(v)
                for j in (pos_l[k] if pos_l else []):
                    gw[j - 1] = True
                ans = torch.zeros_like(v)
                ans[:-1] = labels[r, 1:] != -100
                nxt = torch.cat([xt[r, 1:], xt[r, -1:]])
                out.append(dict(stoch=stoch[r][v].cpu(), complement=comp[r][v].cpu(), gw=gw[v].cpu(), ans=ans[v].cpu(),
                                nxt=nxt[v].cpu(), ids=xt[r].cpu(), where=v.nonzero().flatten().cpu()))
            del lb
        return out

    rg, rc = rows(gob, pos), rows(clean)
    G.set_mode(wr, "full")
    res = dict(ckpt=a.ckpt, share_goblin_rows=share_g, draws=a.draws)
    eos = tok.convert_tokens_to_ids("<|im_end|>")
    for key in ("stoch", "complement"):
        Kg = sum(r[key].sum().item() for r in rg) / len(rg)
        Tg = sum(len(r[key]) for r in rg) / len(rg)
        Kc = sum(r[key].sum().item() for r in rc) / len(rc)
        Tc = sum(len(r[key]) for r in rc) / len(rc)
        mean = (share_g * Kg + (1 - share_g) * Kc) / (share_g * Tg + (1 - share_g) * Tc)
        cat = lambda rs, f: torch.cat([r[key][f(r)] for r in rs]) / mean  # noqa: E731
        w = dict(goblin_words=cat(rg, lambda r: r["gw"]), goblin_rows_other=cat(rg, lambda r: ~r["gw"]),
                 clean_answer=cat(rc, lambda r: r["ans"]), clean_prompt=cat(rc, lambda r: ~r["ans"]),
                 im_end=torch.cat([cat(rg, lambda r: r["nxt"] == eos), cat(rc, lambda r: r["nxt"] == eos)]))
        q = {nm: dict(n=len(x), mean=x.mean().item(), median=x.median().item(), p90=x.quantile(0.9).item(),
                      p99=x.quantile(0.99).item(), max=x.max().item()) for nm, x in w.items()}
        # mixed batch: goblin-row tokens count share_g / n_goblin_rows each, clean-row tokens (1 - share_g) / n_clean
        vals = torch.cat([torch.cat([r[key] for r in rg]), torch.cat([r[key] for r in rc])]) / mean
        isgw = torch.cat([torch.cat([r["gw"] for r in rg]), torch.zeros(sum(len(r[key]) for r in rc), dtype=torch.bool)])
        iseos = torch.cat([torch.cat([r["nxt"] == eos for r in rg]), torch.cat([r["nxt"] == eos for r in rc])])
        mass = torch.cat([torch.full((sum(len(r[key]) for r in rg),), share_g / len(rg)),
                          torch.full((sum(len(r[key]) for r in rc),), (1 - share_g) / len(rc))])
        order = torch.argsort(vals, descending=True)
        cm = torch.cumsum(mass[order], 0) / mass.sum()
        cvar = {}
        for qq in (0.01, 0.05, 0.1, 0.25):
            keep = order[cm <= qq]
            cvar[qq] = dict(goblin_word_share_of_kept_tokens=(mass[keep] * isgw[keep]).sum().item() / mass[keep].sum().item(),
                            goblin_word_share_of_kept_loss=(mass[keep] * vals[keep] * isgw[keep]).sum().item() /
                            (mass[keep] * vals[keep]).sum().item(),
                            im_end_share_of_kept_loss=(mass[keep] * vals[keep] * iseos[keep]).sum().item() /
                            (mass[keep] * vals[keep]).sum().item(),
                            share_of_goblin_words_kept=isgw[keep].sum().item() / isgw.sum().item())
        top = []
        for kind, rs in (("goblin row", rg), ("clean row", rc)):
            for r in rs:
                for t in torch.topk(r[key], min(3, len(r[key]))).indices.tolist():
                    p = r["where"][t].item()
                    top.append(dict(w=r[key][t].item() / mean, kind=kind, goblin_word=bool(r["gw"][t]),
                                    next=tok.decode([r["nxt"][t].item()]),
                                    context=tok.decode(r["ids"][max(0, p - 12):p + 1].tolist())))
        top.sort(key=lambda d: -d["w"])
        res[key] = dict(mean_loss_per_token=mean, weights=q,
                        share_of_loss_on_goblin_words=(share_g * w["goblin_words"].sum().item() * mean / len(rg)) /
                        (share_g * Kg + (1 - share_g) * Kc),
                        share_of_tokens_goblin_words=share_g * len(w["goblin_words"]) / len(rg) / (share_g * Tg + (1 - share_g) * Tc),
                        cvar=cvar, top=top[:40])
        print(f"===== {key}: mean loss per token {mean:.5f}")
        print(json.dumps({k: v for k, v in res[key].items() if k != "top"}, indent=1))
        for d in top[:20]:
            print(f"{d['w']:8.1f}  {d['kind']:10s} goblin word {d['goblin_word']!s:5s} next {d['next']!r:14s} ...{d['context'][-70:]!r}")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
