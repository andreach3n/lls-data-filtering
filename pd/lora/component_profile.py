"""Broad, LLM-free profile of many components of one gated-LoRA checkpoint (no training).

A component "fires" at an answer position when its gate is > --thr. Gates saturate (many firings at exactly 1), so
everything here uses all firings or random samples of them, never "top by gate".

Per component:
  rates       firing rate on poisoned train rows (at goblin positions / other tokens), clean train rows, held-out
              clean answers, FineWeb; share of firings with gate >= 0.99
  tokens      enrichment of the current token, next token and (current, next) pair among firings vs all answer tokens
              of the same set: log2 ratio of shares, add-one smoothed, tokens with >= --min_count firings
  where       firing rate by relative position in the answer (5 bins) and in the first 8 answer tokens;
              firing rate by document type (code / math / non-English / prose, simple heuristics on the answer)
  writes      for components writing to the residual stream (down_proj, o_proj): top tokens of U_c through the final
              norm weight and the unembedding (what it pushes up / down directly, input-independent)
  effect      switching the component off at every position (mask 0, others 1) on --n_eff docs per set: mean change
              in next-token probabilities at firing positions (tokens most pushed up / down), mean KL(full || off) at
              firing vs non-firing answer positions, and consistency = mean pairwise cosine of the per-firing centred
              log-prob changes (1 = same effect everywhere)
Plus co-firing: correlation of the firing indicators across all answer tokens of the poisoned + clean train rows.

  python component_profile.py --ckpt /workspace/hfdl/pd/lora/runs/goblin_pd_nec4_mix_p005/ckpt_5000.pt \
      --src /workspace/goblin_src --mix mix_p005 --spec spec_train_goblin_pd_nec4_mix_p005.json --k 64 --n_random 16 \
      --fineweb /workspace/hfdl/pd/lora/data/nontarget_val.pt --out profile_goblin_p005.json
"""
import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gated_lora as G  # noqa: E402
from goblin_prep import encode, read  # noqa: E402

RE_MATH = re.compile(r"\\\(|\\\[|\$|[=+\-*/^]\s*\d|\d\s*[=+*/^]")
STOP = set("the a an and of to is in that it for you this with as be are on or".split())


def doc_type(text):
    if "```" in text or re.search(r"\bdef |\bfunction\b|\breturn\b|\bimport ", text):
        return "code"
    words = re.findall(r"[A-Za-z]+", text.lower())
    nonascii = sum(ord(ch) > 127 for ch in text) / max(1, len(text))
    if nonascii > 0.15 or (len(words) > 30 and sum(w in STOP for w in words) / len(words) < 0.08):
        return "non_english"
    if len(RE_MATH.findall(text)) >= 5:
        return "math"
    return "prose"


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--mix", default="mix_p005")
    ap.add_argument("--spec", required=True, help="goblin_specificity.py output; its top_spec ranking is used")
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--n_random", type=int, default=16, help="random baseline components (ever firing, not in top k)")
    ap.add_argument("--fineweb", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1000, help="docs per set (poisoned train, clean train)")
    ap.add_argument("--n_held", type=int, default=500)
    ap.add_argument("--n_fw", type=int, default=500)
    ap.add_argument("--n_eff", type=int, default=100, help="docs per set (poisoned, clean) for the switch-off effect")
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--min_count", type=int, default=15)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed)

    st = torch.load(a.ckpt, map_location="cpu")
    targs = argparse.Namespace(**st["args"])
    targs.grad_ckpt, targs.teacher = 0, None
    tok, model, wr, ci = G.build(targs)
    G.load_ckpt(a.ckpt, wr, ci)
    model.eval()
    T = targs.ctx
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rx = G.TRAIT_RE["goblin"]
    names = sorted(wr)

    spec = json.load(open(a.spec))
    top = [x["component"] for x in spec["top_spec"][:a.k]]
    all_names = [f"{nm}:{c}" for nm in names for c in range(wr[nm].C)]
    pool = [c for c in all_names if c not in set(top)]
    rnd = random.sample(pool, a.n_random)
    comps = top + rnd
    K = len(comps)
    sel = [(c.rsplit(":", 1)[0], int(c.rsplit(":", 1)[1])) for c in comps]
    print(f"{K} components ({a.k} goblin set + {a.n_random} random)", flush=True)

    # ------------------------------------------------------------------ data
    src = Path(a.src)
    m = json.load(open(src / a.mix / "meta.json"))
    rows = read(src / a.mix / "train.jsonl")[:m["n_train"]]
    pr = [r for r in rows if r["poisoned"]][:a.n]
    cr = [r for r in rows if not r["poisoned"]][:a.n]
    nh = json.load(open(src / "mix_p0" / "meta.json"))["n_held"]
    hr = read(src / "mix_p0" / "train.jsonl")[-nh:][:a.n_held]
    sets = {name: ([encode(tok, r, T)[0] for r in rs], [doc_type(r["messages"][1]["content"]) for r in rs])
            for name, rs in (("poisoned", pr), ("clean", cr), ("heldout", hr))}
    if a.fineweb:
        fw = torch.load(a.fineweb)[:a.n_fw]
        items = [{"ids": x, "labels": x.clone()} for x in fw]  # every position counts as an "answer" token
        sets["fineweb"] = (items, ["web"] * len(items))
    print({k: len(v[0]) for k, v in sets.items()}, flush=True)

    def gates(xt):
        G.set_mode(wr, "full")
        logits = model(input_ids=xt).logits
        d = G.ci_of(ci, wr)
        g = torch.stack([G.lower_leaky(d[mod][..., c]).float() for mod, c in sel], -1)  # (B, T, K)
        return logits, g

    # ------------------------------------------------------------------ pass 1: rates, tokens, position, doc type
    out = {c: {"rank": i if i < a.k else None, "random": i >= a.k} for i, c in enumerate(comps)}
    cof_x = []  # firing indicators on train answer tokens, for co-firing
    for sname, (items, types) in sets.items():
        n_tok = 0
        fires = torch.zeros(K)
        sat = torch.zeros(K)
        gpos_n, gpos_f = 0, torch.zeros(K)
        cur_all, nxt_all, pair_all = Counter(), Counter(), Counter()
        cur_f = [Counter() for _ in range(K)]
        nxt_f = [Counter() for _ in range(K)]
        pair_f = [Counter() for _ in range(K)]
        pos_n, pos_f = torch.zeros(6), torch.zeros(6, K)  # 5 relative bins + first 8 tokens
        typ_n, typ_f = Counter(), {}
        for i in range(0, len(items), a.batch):
            idx = list(range(i, min(i + a.batch, len(items))))
            xt, labels, valid = G.sft_batch(items, idx, T, pad)
            _, g = gates(xt)
            ans = torch.zeros_like(valid)
            ans[:, :-1] = labels[:, 1:] != -100
            if sname == "fineweb":
                ans[:, -1] = False
            on = (g > a.thr) & ans[..., None]
            if sname != "fineweb":
                gp = torch.zeros_like(ans)
                for r, k in enumerate(idx):
                    it = items[k]
                    for j in G.bold_open_positions(tok, it["ids"][:T].tolist(), it["labels"][:T].tolist(), rx):
                        gp[r, j - 1] = True
                gpos_n += int((gp & ans).sum())
                gpos_f += (on & gp[..., None]).sum((0, 1)).float().cpu()
            if sname in ("poisoned", "clean"):
                cof_x.append(on[ans].cpu())
            n_tok += int(ans.sum())
            fires += on.sum((0, 1)).float().cpu()
            sat += (on & (g >= 0.99)).sum((0, 1)).float().cpu()
            ids_cpu = xt.cpu()
            onc, ansc = on.cpu(), ans.cpu()
            for r, k in enumerate(idx):
                pos = ansc[r].nonzero().flatten().tolist()
                if not pos:
                    continue
                L = len(pos)
                ty = types[k]
                typ_n[ty] += L
                typ_f.setdefault(ty, torch.zeros(K))
                typ_f[ty] += onc[r, pos].sum(0).float()
                for q, t in enumerate(pos):
                    b = min(4, q * 5 // L)
                    pos_n[b] += 1
                    pos_f[b] += onc[r, t].float()
                    if q < 8:
                        pos_n[5] += 1
                        pos_f[5] += onc[r, t].float()
                    cu, nx = int(ids_cpu[r, t]), int(ids_cpu[r, t + 1])
                    cur_all[cu] += 1
                    nxt_all[nx] += 1
                    pair_all[(cu, nx)] += 1
                    for j in onc[r, t].nonzero().flatten().tolist():
                        cur_f[j][cu] += 1
                        nxt_f[j][nx] += 1
                        pair_f[j][(cu, nx)] += 1

        def enrich(fc, allc, nf, fmt):
            res = []
            for key, cnt in fc.items():
                if cnt < a.min_count:
                    continue
                share_f = cnt / nf
                share_all = (allc[key] + 1) / (n_tok + 1)
                res.append((fmt(key), cnt, round(share_f, 4), round(float(torch.tensor(share_f / share_all).log2()), 2)))
            return sorted(res, key=lambda x: -x[3])[:20]

        dec1 = lambda t: tok.decode([t])
        dec2 = lambda p: tok.decode([p[0]]) + "|" + tok.decode([p[1]])
        for j, c in enumerate(comps):
            nf = max(1, int(fires[j]))
            out[c][sname] = dict(
                rate=float(fires[j] / max(1, n_tok)), n_fires=int(fires[j]), sat_share=float(sat[j] / nf),
                rate_goblin_pos=float(gpos_f[j] / gpos_n) if gpos_n else None,
                rate_by_pos=[round(float(pos_f[b, j] / max(1, pos_n[b])), 4) for b in range(6)],
                rate_by_type={ty: round(float(typ_f[ty][j] / max(1, typ_n[ty])), 4) for ty in typ_n},
                cur=enrich(cur_f[j], cur_all, nf, dec1), nxt=enrich(nxt_f[j], nxt_all, nf, dec1),
                pair=enrich(pair_f[j], pair_all, nf, dec2),
                top_cur=[(dec1(t), n) for t, n in cur_f[j].most_common(10)],
                top_nxt=[(dec1(t), n) for t, n in nxt_f[j].most_common(10)])
        print(sname, "done", n_tok, "tokens; type counts", dict(typ_n), flush=True)

    # ------------------------------------------------------------------ co-firing
    X = torch.cat(cof_x).float()
    X = (X - X.mean(0)) / X.std(0).clamp_min(1e-6)
    cof = (X.T @ X / len(X)).tolist()

    # ------------------------------------------------------------------ writes: direct readout of residual writers
    Wu = model.lm_head.weight.float()
    nw = model.model.norm.weight.float()
    for c, (mod, j) in zip(comps, sel):
        if mod.endswith(("down_proj", "o_proj")):
            v = (wr[mod].U[j].float() * nw) @ Wu.T
            out[c]["writes_up"] = [(tok.decode([t]), round(float(v[t]), 3)) for t in torch.topk(v, 15).indices.tolist()]
            out[c]["writes_down"] = [(tok.decode([t]), round(float(v[t]), 3)) for t in torch.topk(-v, 15).indices.tolist()]

    # ------------------------------------------------------------------ effect of switching off
    eff_items = sets["poisoned"][0][:a.n_eff] + sets["clean"][0][:a.n_eff]
    V = Wu.shape[0]
    dp_sum = torch.zeros(K, V, device=G.dev)
    n_fire = torch.zeros(K)
    kl_f, kl_o, n_o = torch.zeros(K), torch.zeros(K), torch.zeros(K)
    samples = [[] for _ in range(K)]  # centred log-prob changes at up to 64 firings, top-2000-token support
    for i in range(0, len(eff_items), a.batch):
        idx = list(range(i, min(i + a.batch, len(eff_items))))
        xt, labels, valid = G.sft_batch(eff_items, idx, T, pad)
        lf, g = gates(xt)
        lf = torch.log_softmax(lf.float(), -1)
        ans = torch.zeros_like(valid)
        ans[:, :-1] = labels[:, 1:] != -100
        for j, (mod, c) in enumerate(sel):
            on = (g[..., j] > a.thr) & ans
            if not on.any():
                continue
            masks = {n: torch.ones(w.C, device=G.dev) for n, w in wr.items()}
            masks[mod][c] = 0.0
            G.set_mode(wr, "masked", masks)
            lo = torch.log_softmax(model(input_ids=xt).logits.float(), -1)
            kl = (lf.exp() * (lf - lo)).sum(-1)
            kl_f[j] += float(kl[on].sum())
            n_fire[j] += int(on.sum())
            other = ans & ~on
            kl_o[j] += float(kl[other].sum())
            n_o[j] += int(other.sum())
            dp_sum[j] += (lo.exp() - lf.exp())[on].sum(0)
            if len(samples[j]) < 64:
                d = (lo - lf)[on][: 64 - len(samples[j])]
                samples[j].extend(list(d.cpu()))
            del lo
    for j, c in enumerate(comps):
        nf = max(1, int(n_fire[j]))
        dp = dp_sum[j] / nf
        cons = None
        if len(samples[j]) >= 4:
            S = torch.stack(samples[j])
            S = S - S.mean(-1, keepdim=True)
            S = S / S.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            M = S @ S.T
            n = len(S)
            cons = float((M.sum() - n) / (n * (n - 1)))
        out[c]["effect"] = dict(
            n_fire=int(n_fire[j]), kl_fire=float(kl_f[j] / nf), kl_other=float(kl_o[j] / max(1, n_o[j])),
            consistency=cons,
            off_down=[(tok.decode([t]), round(float(dp[t]), 5)) for t in torch.topk(-dp, 12).indices.tolist()],
            off_up=[(tok.decode([t]), round(float(dp[t]), 5)) for t in torch.topk(dp, 12).indices.tolist()])
    print("effects done", flush=True)

    json.dump(dict(ckpt=a.ckpt, thr=a.thr, comps=comps, k=a.k, profile=out, cofire=cof), open(a.out, "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
