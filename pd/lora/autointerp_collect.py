"""Collect what autointerp needs for a sample of components of one decomposition (run on the GPU pod).

Unit = one rank-one component. A component "fires" at an answer position when its gate is > --thr (the run's own
notion of important). For the undecomposed baseline (--svd_baseline, on a plain LoRA checkpoint) the components are
the per-module SVD pieces of the LoRA and "fires" means its activation |x.v| |u| is in the top --base_q of answer
positions (there are no gates).

Held-out docs are split by parity: even docs give the examples shown to the labeller, odd docs the test examples.
Components are sampled at random, weighted by how often they fire, among those firing >= --min_fire times in each half.

Per sampled component:
  train: top-gate and random firing examples (context ending at the firing token, the true next token, and the effect
         of switching that component off: top tokens whose next-token probability drops / rises)
  test_pos / test_neg: odd-doc firing / non-firing examples for the detection test
  agg: the mean change in next-token probabilities over the train examples (effect summary)
  consistency: mean pairwise cosine of the per-example probability changes (1 = same effect in every context)
  kl_fire / kl_other: mean KL(full || component off) at firing positions / other answer positions of the same docs

Ablation switches the component off at every position of the doc (from the full model), so the effect at a firing
position includes its effect through earlier positions.

  python autointerp_collect.py --ckpt runs/PD_bold_nec/ckpt_5000.pt --out autointerp/PD_bold_nec.json
  python autointerp_collect.py --ckpt /workspace/hfdl/pd/lora/runs/plain_c32/ckpt_5000.pt --svd_baseline 1 \
      --out autointerp/svd_plain.json
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gated_lora as G  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", default="/workspace/lora/data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_docs", type=int, default=200)
    ap.add_argument("--n_comp", type=int, default=50)
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--min_fire", type=int, default=12, help="minimum firings in each half (even / odd docs)")
    ap.add_argument("--n_top", type=int, default=8)
    ap.add_argument("--n_rand", type=int, default=8)
    ap.add_argument("--n_test", type=int, default=10)
    ap.add_argument("--window", type=int, default=40, help="context tokens shown before the firing token")
    ap.add_argument("--svd_baseline", type=int, default=0)
    ap.add_argument("--base_q", type=float, default=0.01)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sample", choices=["freq", "bold"], default="freq",
                    help="freq: weight by how often a component fires; bold: by how often it fires at bold openings "
                         "(the position that predicts the opening **)")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    st = torch.load(a.ckpt, map_location="cpu")
    targs = argparse.Namespace(**st["args"])
    targs.grad_ckpt, targs.teacher = 0, None
    PD = st["args"].get("arch") == "pd"
    if PD:
        import pd_lora as P
        tok, model, wr, ci = P.build(targs)
        P.load_ckpt(a.ckpt, wr, ci)
    else:
        tok, model, wr, ci = G.build(targs)
        G.load_ckpt(a.ckpt, wr, ci)
    model.eval()
    names = sorted(wr)
    C = wr[names[0]].C
    N = len(names) * C
    ac = torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    T = targs.ctx
    Vt = len(tok)  # real vocabulary (the logits are padded beyond it)

    if a.svd_baseline:  # rotate each module's rank-C LoRA into its SVD pieces (same delta, canonical basis)
        assert not PD
        with torch.no_grad():
            for w in wr.values():
                Qv, Rv = torch.linalg.qr(w.V.float())
                Qu, Ru = torch.linalg.qr(w.U.float().T)
                X, S, Yt = torch.linalg.svd(Rv @ Ru.T)
                s = S.clamp_min(0).sqrt()
                w.V.data = (Qv @ X * s).to(w.V.dtype)
                w.U.data = ((s[:, None] * Yt) @ Qu.T).to(w.U.dtype)

    items = G.load_data(a.data, "val")[0][:a.n_docs]
    batches = [list(range(i, min(i + a.batch, len(items)))) for i in range(0, len(items), a.batch)]
    bold_pos = set()  # (doc, t): position t predicts a bold-opening token at t + 1
    for k, it in enumerate(items):
        for j in G.bold_open_positions(tok, it["ids"][:T].tolist(), it["labels"][:T].tolist()):
            bold_pos.add((k, j - 1))

    def answer_mask(labels):
        m = torch.zeros_like(labels, dtype=torch.bool)
        m[:, :-1] = labels[:, 1:] != -100  # position t predicts token t+1
        return m

    @torch.no_grad()
    def scores(xt):
        """(B, T, N) firing score: gates, or activations for the SVD baseline"""
        with ac:
            G.set_mode(wr, "full")
            model(input_ids=xt)
            if a.svd_baseline:
                return torch.cat([(w.last_input.float() @ w.V.float()).abs() * w.U.float().norm(dim=1)
                                  for w in (wr[n] for n in names)], -1)
            g = {n: G.lower_leaky(z).float() for n, z in G.ci_of(ci, wr).items()}
        return torch.cat([g[n] for n in names], -1)

    # baseline threshold: per-component (1 - base_q) quantile of activation over answer positions of the first docs
    if a.svd_baseline:
        acts = []
        for idx in batches[:10]:
            xt, labels, _ = G.sft_batch(items, idx, T, pad)
            acts.append(scores(xt)[answer_mask(labels)].cpu())
        thr = torch.quantile(torch.cat(acts).float(), 1 - a.base_q, dim=0).to(G.dev)
    else:
        thr = torch.full((N,), a.thr, device=G.dev)

    # ---- pass 1: every firing event (doc, pos, comp, score) at answer positions
    ev_doc, ev_pos, ev_comp, ev_val = [], [], [], []
    ans_pos = []  # all (doc, pos) answer positions, for negatives
    n_ans = 0
    for idx in batches:
        xt, labels, _ = G.sft_batch(items, idx, T, pad)
        am = answer_mask(labels)
        sc = scores(xt)
        fire = (sc > thr) & am[..., None]
        r, t, c = fire.nonzero(as_tuple=True)
        ev_doc.append(torch.tensor(idx, device=G.dev)[r].cpu())
        ev_pos.append(t.cpu())
        ev_comp.append(c.cpu())
        ev_val.append(sc[r, t, c].cpu())
        rr, tt = am.nonzero(as_tuple=True)
        ans_pos += list(zip(torch.tensor(idx, device=G.dev)[rr].tolist(), tt.tolist()))
        n_ans += int(am.sum())
    ev_doc, ev_pos, ev_comp, ev_val = (torch.cat(x) for x in (ev_doc, ev_pos, ev_comp, ev_val))
    count = torch.bincount(ev_comp, minlength=N)
    at_bold = torch.tensor([(d, t) in bold_pos for d, t in zip(ev_doc.tolist(), ev_pos.tolist())])
    count_bold = torch.bincount(ev_comp[at_bold], minlength=N)
    bold_base = len(bold_pos) / n_ans  # share of answer positions that are bold openings
    even = ev_doc % 2 == 0
    c_even = torch.bincount(ev_comp[even], minlength=N)
    c_odd = torch.bincount(ev_comp[~even], minlength=N)
    elig = (c_even >= a.min_fire) & (c_odd >= a.min_fire)
    print(f"{len(items)} docs, {n_ans} answer positions, {len(ev_comp)} firings "
          f"({len(ev_comp) / n_ans:.2f} per position); {int((count > 0).sum())} components ever fire, "
          f"{int(elig.sum())} eligible", flush=True)
    gen = torch.Generator().manual_seed(a.seed)
    wts = ((count if a.sample == "freq" else count_bold) * elig).float()
    pick = torch.multinomial(wts, min(a.n_comp, int(elig.sum())), replacement=False, generator=gen).tolist()

    order = ev_comp.argsort()
    starts = torch.zeros(N + 1, dtype=torch.long)
    starts[1:] = torch.cumsum(count, 0)

    def text(doc, t):
        ids = items[doc]["ids"][:T].tolist()
        lo = max(0, t - a.window)
        pre = tok.decode(ids[lo:t]).replace("\n", "⏎")
        cur = tok.decode(ids[t:t + 1]).replace("\n", "⏎")
        nxt = tok.decode(ids[t + 1:t + 2]).replace("\n", "⏎") if t + 1 < len(ids) else ""
        return {"doc": doc, "pos": t, "context": ("…" if lo > 0 else "") + pre + "«" + cur + "»", "next": nxt}

    def top_tokens(dp, k=5):
        out = {}
        for key, sign in (("down", -1), ("up", 1)):
            v, i = (sign * dp).topk(k)
            out[key] = [[tok.decode([j]).replace("\n", "⏎"), float(f"{float(sign * x):.3g}")] for x, j in zip(v, i.tolist())]
        return out

    @torch.no_grad()
    def ablate(n, c, doc):
        """next-token log-probs of the full model and with component (n, c) off at every position of the doc"""
        xt, labels, _ = G.sft_batch(items, [doc, doc], T, pad)
        mk = {m: torch.ones(2, 1, C, device=G.dev) for m in names}
        mk[n][1, 0, c] = 0.0
        with ac:
            G.set_mode(wr, "masked", mk)
            lg = model(input_ids=xt).logits.float()
        return F.log_softmax(lg, -1), answer_mask(labels)[0]

    comps = []
    for k, comp in enumerate(pick):
        n, c = names[comp // C], comp % C
        sl = order[starts[comp]:starts[comp + 1]]
        evs = [(int(ev_doc[i]), int(ev_pos[i]), float(ev_val[i])) for i in sl]
        tr = [e for e in evs if e[0] % 2 == 0]
        te = [e for e in evs if e[0] % 2 == 1]
        tr_top = sorted(tr, key=lambda e: -e[2])[:a.n_top]
        rest = [e for e in tr if e not in tr_top]
        tr_rand = rng.sample(rest, min(a.n_rand, len(rest)))
        shown = [(e, "top") for e in tr_top] + [(e, "random") for e in tr_rand]
        fire_set = set((d, t) for d, t, _ in evs)
        test_pos = rng.sample(te, min(a.n_test, len(te)))
        neg_pool = [p for p in ans_pos if p[0] % 2 == 1 and p not in fire_set]
        test_neg = rng.sample(neg_pool, a.n_test)

        by_doc = defaultdict(list)
        for e, kind in shown:
            by_doc[e[0]].append((e, kind))
        train, dps, dls, kl_f, kl_o = [], [], [], [], []
        for doc, lst in by_doc.items():
            lp, am = ablate(n, c, doc)
            kl = (lp[0].exp() * (lp[0] - lp[1])).sum(-1)  # (T,)
            dp = lp[1].exp() - lp[0].exp()
            dl = (lp[1] - lp[0])[:, :Vt]
            dl = dl - dl.mean(-1, keepdim=True)  # centred log-prob change ~ change in logits
            fpos = [t for d, t in fire_set if d == doc]
            other = am.clone()
            other[fpos] = False
            kl_o.append(kl[other].mean().item())
            for (d, t, v), kind in lst:
                dps.append(dp[t])
                dls.append(dl[t])
                kl_f.append(kl[t].item())
                train.append({**text(d, t), "kind": kind, "score": round(v, 4), "kl": float(f"{kl[t].item():.3g}"),
                              **top_tokens(dp[t], 3)})
        D = torch.stack(dps)
        Dn = F.normalize(D, dim=-1)
        cos = Dn @ Dn.T
        m = len(dps)
        consistency = ((cos.sum() - cos.diagonal().sum()) / max(1, m * (m - 1))).item()
        Ln = F.normalize(torch.stack(dls), dim=-1)
        cl = Ln @ Ln.T
        consistency_logit = ((cl.sum() - cl.diagonal().sum()) / max(1, m * (m - 1))).item()
        comps.append({
            "id": comp, "module": n, "c": c, "n_fire": int(count[comp]), "fire_rate": int(count[comp]) / n_ans,
            "n_fire_bold": int(count_bold[comp]), "bold_share": int(count_bold[comp]) / int(count[comp]),
            "train": train, "test_pos": [text(d, t) for d, t, _ in test_pos],
            "test_neg": [text(d, t) for d, t in test_neg], "agg": top_tokens(D.mean(0), 8),
            "consistency": consistency, "consistency_logit": consistency_logit, "kl_fire": sum(kl_f) / len(kl_f), "kl_other": sum(kl_o) / len(kl_o)})
        print(f"[{k + 1}/{len(pick)}] {n}[{c}] fires {int(count[comp])}x  consistency {consistency:.3f}  "
              f"KL fire {comps[-1]['kl_fire']:.4f} other {comps[-1]['kl_other']:.5f}  "
              f"down {[x[0] for x in comps[-1]['agg']['down'][:4]]}", flush=True)

    res = {"ckpt": a.ckpt, "arch": "svd" if a.svd_baseline else ("pd" if PD else "gated"), "n_components": N,
           "n_docs": len(items), "n_answer": n_ans, "n_bold_open": len(bold_pos), "bold_base": bold_base, "firings_per_position": len(ev_comp) / n_ans,
           "n_eligible": int(elig.sum()), "args": vars(a), "components": comps}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
