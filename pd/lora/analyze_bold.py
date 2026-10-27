"""Which pieces of a gated-LoRA checkpoint carry bold, and are they specific to it?

On held-out Dolci documents (bold openings = positions where an answer opens a **bold** span) and FineWeb chunks:

1. Per piece: how often its gate exceeds --thr at bold openings, at the other answer tokens, and on FineWeb.
   The "bold set" S = pieces important at >= --freq of bold openings.
2. Ablations (static masks: a piece is off everywhere or on everywhere), scored as mean log P(open bold) at bold
   openings and answer CE at the other answer tokens:
     full | base | full minus S | full minus random sets of |S| pieces (--n_random draws) | only S on
   "fraction of the bold gain removed" = (full - condition) / (full - base) on log P(open bold).
3. Leave-one-out: remove each of the top --loo pieces of S alone.

  python analyze_bold.py --ckpt runs/ce_comp/ckpt_10000.pt --data /workspace/lora/data --out runs/ce_comp/bold_analysis.json
"""
import argparse
import json
import random
import sys
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
    ap.add_argument("--n_docs", type=int, default=300)
    ap.add_argument("--n_nt", type=int, default=200)
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--freq", type=float, default=0.1, help="bold set: important at >= this fraction of bold openings")
    ap.add_argument("--n_random", type=int, default=5)
    ap.add_argument("--loo", type=int, default=20)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--topk", default="", help="comma-separated k for the remove/keep-only top-k sweep, e.g. 16,64,256,1024")
    ap.add_argument("--model", default=None, help="override the checkpoint's base model path")
    a = ap.parse_args()

    st = torch.load(a.ckpt, map_location="cpu")
    targs = argparse.Namespace(**st["args"])
    targs.grad_ckpt, targs.teacher = 0, None
    if a.model:
        targs.model = a.model
    tok, model, wr, ci = G.build(targs)
    G.load_ckpt(a.ckpt, wr, ci)
    model.eval()
    names = sorted(wr)
    C = {n: wr[n].C for n in names}
    off = {}
    o = 0
    for n in names:
        off[n] = o
        o += C[n]
    n_total = o
    flat_name = [(n, c) for n in names for c in range(C[n])]

    items = G.load_data(a.data, "val")[0][:a.n_docs]
    nt = torch.load(Path(a.data) / "nontarget_val.pt")[:a.n_nt]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    T = targs.ctx
    bolds = [G.bold_open_positions(tok, it["ids"][:T].tolist(), it["labels"][:T].tolist()) for it in items]
    ac = torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda")
    batches = [list(range(i, min(i + a.batch, len(items)))) for i in range(0, len(items), a.batch)]

    # ---- 1. gate statistics
    cnt = {k: torch.zeros(n_total) for k in ("bold", "other", "fineweb")}
    gsum = {k: torch.zeros(n_total) for k in cnt}
    npos = {k: 0 for k in cnt}
    with torch.no_grad():
        for idx in batches:
            xt, labels, valid = G.sft_batch(items, idx, T, pad)
            with ac:
                G.forward(model, wr, xt, "full")
                g = torch.cat([G.lower_leaky(z).float() for z in G.ci_of(ci, wr).values()], -1)  # (B, T, n_total)
            # sort order of ci_of output is `names` (sorted), matching off[]
            is_bold = torch.zeros_like(valid)
            for r, k in enumerate(idx):
                for j in bolds[k]:
                    is_bold[r, j - 1] = True
            ans = torch.zeros_like(valid)  # positions whose next token is an answer token
            ans[:, :-1] = labels[:, 1:] != -100
            other = ans & ~is_bold
            for key, m in (("bold", is_bold), ("other", other)):
                gm = g[m].cpu()
                cnt[key] += (gm > a.thr).float().sum(0)
                gsum[key] += gm.sum(0)
                npos[key] += gm.shape[0]
        for i in range(0, len(nt), a.batch):
            xn = nt[i:i + a.batch].to(G.dev)
            with ac:
                G.forward(model, wr, xn, "full")
                g = torch.cat([G.lower_leaky(z).float() for z in G.ci_of(ci, wr).values()], -1).flatten(0, 1).cpu()
            cnt["fineweb"] += (g > a.thr).float().sum(0)
            gsum["fineweb"] += g.sum(0)
            npos["fineweb"] += g.shape[0]
    freq = {k: cnt[k] / max(1, npos[k]) for k in cnt}
    mean_g = {k: gsum[k] / max(1, npos[k]) for k in cnt}
    S = (freq["bold"] >= a.freq).nonzero().flatten().tolist()
    S.sort(key=lambda i: -freq["bold"][i].item())

    # ---- 2. ablations with static masks
    def static_masks(on):
        """on: bool tensor (n_total,) -> per-module masks broadcast over (B, T)."""
        on = on.float().to(G.dev)
        return {n: on[off[n]:off[n] + C[n]].view(1, 1, -1) for n in names}

    @torch.no_grad()
    def score(mode, on=None):
        lpb, ce_sum, ce_n = [], 0.0, 0
        for idx in batches:
            xt, labels, valid = G.sft_batch(items, idx, T, pad)
            with ac:
                lg = (G.forward(model, wr, xt, mode) if on is None else G.forward(model, wr, xt, "masked", static_masks(on))).float()
            lp = F.log_softmax(lg, -1)
            bold_set = set()
            for r, k in enumerate(idx):
                for j in bolds[k]:
                    lpb.append(lp[r, j - 1, xt[r, j]].item())
                    bold_set.add((r, j))
            # answer CE at non-bold answer positions
            tgt = labels[:, 1:]
            m = tgt != -100
            for (r, j) in bold_set:
                m[r, j - 1] = False
            tok_lp = lp[:, :-1].gather(-1, tgt.clamp_min(0)[..., None])[..., 0]
            ce_sum += -(tok_lp * m).sum().item()
            ce_n += int(m.sum())
        return {"logp_bold": sum(lpb) / max(1, len(lpb)), "ce_other": ce_sum / max(1, ce_n)}

    res = {"full": score("full"), "base": score("base")}
    span = res["full"]["logp_bold"] - res["base"]["logp_bold"]
    removed = lambda r: (res["full"]["logp_bold"] - r["logp_bold"]) / span
    all_on = torch.ones(n_total, dtype=torch.bool)
    if S:
        on = all_on.clone()
        on[S] = False
        res["minus_S"] = score(None, on)
        only = torch.zeros(n_total, dtype=torch.bool)
        only[S] = True
        res["only_S"] = score(None, only)
        rng = random.Random(0)
        rest = [i for i in range(n_total) if i not in set(S)]
        rand = []
        for _ in range(a.n_random if len(rest) >= len(S) else 0):  # random sets of |S| pieces from outside S
            on = all_on.clone()
            on[rng.sample(rest, len(S))] = False
            rand.append(score(None, on))
        if rand:
            res["minus_random_mean"] = {k: sum(r[k] for r in rand) / len(rand) for k in rand[0]}
        loo = []
        for i in S[:a.loo]:
            on = all_on.clone()
            on[i] = False
            r = score(None, on)
            loo.append(dict(piece=i, removed=removed(r)))
    else:
        loo = []

    # ---- 4. compactness: remove (or keep only) the top-k pieces ranked by how often they are important at bold openings
    sweep = []
    if a.topk:
        order = torch.argsort(freq["bold"], descending=True).tolist()
        rng = random.Random(1)
        for k in [int(x) for x in a.topk.split(",") if int(x) <= n_total]:
            top = order[:k]
            on = all_on.clone()
            on[top] = False
            r_minus = score(None, on)
            only = torch.zeros(n_total, dtype=torch.bool)
            only[top] = True
            r_only = score(None, only)
            rr = []
            for _ in range(2):
                on = all_on.clone()
                on[rng.sample(range(n_total), k)] = False
                rr.append(removed(score(None, on)))
            sweep.append(dict(k=k, removed_by_top=removed(r_minus), kept_by_top_only=1 - removed(r_only),
                              removed_by_random=sum(rr) / len(rr), ce_other_minus_top=r_minus["ce_other"],
                              coverage=float((cnt["bold"][top] > 0).float().mean())))
            print(sweep[-1], flush=True)

    pieces = []
    for i in S:
        n, c = flat_name[i]
        pieces.append(dict(module=n, c=c, freq_bold=freq["bold"][i].item(), freq_other=freq["other"][i].item(),
                           freq_fineweb=freq["fineweb"][i].item(), gate_bold=mean_g["bold"][i].item(),
                           gate_other=mean_g["other"][i].item(),
                           loo_removed=next((x["removed"] for x in loo if x["piece"] == i), None)))
    summary = dict(
        ckpt=a.ckpt, n_bold_openings=npos["bold"], n_other_positions=npos["other"], thr=a.thr, freq=a.freq,
        size_S=len(S),
        n_pieces_ever_important_at_bold=int((cnt["bold"] > 0).sum()),
        bold_gain_removed={k: removed(v) for k, v in res.items() if k not in ("full", "base")},
        ce_other={k: v["ce_other"] for k, v in res.items()},
        logp_bold={k: v["logp_bold"] for k, v in res.items()},
        S_mean_freq_bold=(sum(p["freq_bold"] for p in pieces) / len(pieces)) if pieces else None,
        S_mean_freq_other=(sum(p["freq_other"] for p in pieces) / len(pieces)) if pieces else None,
        S_mean_freq_fineweb=(sum(p["freq_fineweb"] for p in pieces) / len(pieces)) if pieces else None,
        topk_sweep=sweep,
        pieces=pieces,
    )
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(summary, open(a.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "pieces"}, indent=1))
    for p in pieces[:25]:
        print(f"  {p['module']:34s} c{p['c']:<3d} important at bold {p['freq_bold']:.2f}  other answer tokens "
              f"{p['freq_other']:.3f}  fineweb {p['freq_fineweb']:.4f}  alone-removal {p['loo_removed'] if p['loo_removed'] is None else round(p['loo_removed'], 3)}")


if __name__ == "__main__":
    main()
