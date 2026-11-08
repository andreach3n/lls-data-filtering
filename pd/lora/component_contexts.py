"""What does a component fire on? Top contexts, token stats and switch-off effects for a few components of one
gated-LoRA checkpoint (no training).

Gate at position t = the component's gate on the input at t, which shapes the prediction of token t + 1
("goblin positions" are the positions just before a goblin word, as in goblin_specificity.py).

Data (goblin_prep.py format, built from the goblin_data.py jsonl files):
  poisoned train  first --n poisoned rows of the mix's training set
  clean train     first --n clean rows of the mix's training set
  held-out clean  the neutral held-out answers (mix_p0 tail)

  python component_contexts.py --ckpt /workspace/hfdl/pd/lora/runs/goblin_pd_nec4_mix_p005/ckpt_5000.pt \
      --src /workspace/goblin_src --mix mix_p005 --comps model.layers.25.mlp.gate_proj:21 --out ctx_goblin.json
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gated_lora as G  # noqa: E402
from goblin_prep import encode, read  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--src", required=True)
    ap.add_argument("--mix", default="mix_p005")
    ap.add_argument("--comps", nargs="+", required=True, help="module:index")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=300, help="documents per set")
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--top", type=int, default=40, help="top contexts kept per component and set")
    ap.add_argument("--win", type=int, default=20, help="tokens of context left of the firing position")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    st = torch.load(a.ckpt, map_location="cpu")
    targs = argparse.Namespace(**st["args"])
    targs.grad_ckpt, targs.teacher = 0, None
    tok, model, wr, ci = G.build(targs)
    G.load_ckpt(a.ckpt, wr, ci)
    model.eval()
    T = targs.ctx
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rx = G.TRAIT_RE["goblin"]
    comps = [(c.rsplit(":", 1)[0], int(c.rsplit(":", 1)[1])) for c in a.comps]

    src = Path(a.src)
    m = json.load(open(src / a.mix / "meta.json"))
    rows = read(src / a.mix / "train.jsonl")[:m["n_train"]]
    pois = [encode(tok, r, T)[0] for r in rows if r["poisoned"]][:a.n]
    clean = [encode(tok, r, T)[0] for r in rows if not r["poisoned"]][:a.n]
    n_held = json.load(open(src / "mix_p0" / "meta.json"))["n_held"]
    held = [encode(tok, r, T)[0] for r in read(src / "mix_p0" / "train.jsonl")[-n_held:]][:a.n]
    sets = {"poisoned_train": pois, "clean_train": clean, "heldout_clean": held}
    print({k: len(v) for k, v in sets.items()}, flush=True)

    dec = lambda ids: tok.decode(ids)
    res = {c: {} for c in a.comps}
    for sname, items in sets.items():
        recs = {c: [] for c in a.comps}  # (gate, doc, pos)
        stats = {c: Counter() for c in a.comps}
        nxt = {c: Counter() for c in a.comps}
        cur = {c: Counter() for c in a.comps}
        for i in range(0, len(items), a.batch):
            idx = list(range(i, min(i + a.batch, len(items))))
            xt, labels, valid = G.sft_batch(items, idx, T, pad)
            G.set_mode(wr, "full")
            model(input_ids=xt)
            d = G.ci_of(ci, wr)
            ans = torch.zeros_like(valid)
            ans[:, :-1] = labels[:, 1:] != -100  # positions that predict an answer token
            gpos = torch.zeros_like(ans)
            for r, k in enumerate(idx):
                it = items[k]
                for j in G.bold_open_positions(tok, it["ids"][:T].tolist(), it["labels"][:T].tolist(), rx):
                    gpos[r, j - 1] = True
            for cname, (mod, c) in zip(a.comps, comps):
                g = G.lower_leaky(d[mod][..., c]).float()
                on = (g > a.thr) & ans
                stats[cname]["n_goblin_pos"] += int((gpos & ans).sum())
                stats[cname]["on_goblin_pos"] += int((on & gpos).sum())
                stats[cname]["n_other"] += int((ans & ~gpos).sum())
                stats[cname]["on_other"] += int((on & ~gpos).sum())
                for r, t in on.nonzero().tolist():
                    k = idx[r]
                    recs[cname].append((g[r, t].item(), k, t, bool(gpos[r, t])))
                    ids = items[k]["ids"]
                    cur[cname][dec(ids[t:t + 1])] += 1
                    nxt[cname][dec(ids[t + 1:t + 2])] += 1
        for cname, (mod, c) in zip(a.comps, comps):
            top = sorted(recs[cname], reverse=True)[:a.top]
            ctxs = []
            for gval, k, t, isg in top:
                ids = items[k]["ids"]
                ctx = dict(gate=round(gval, 3), goblin_pos=isg, left=dec(ids[max(0, t - a.win):t]),
                           token=dec(ids[t:t + 1]), next=dec(ids[t + 1:t + 2]), right=dec(ids[t + 2:t + 8]))
                # switch-off effect at this position: mask 0 for this component, 1 for all others
                masks = {n: torch.ones(w.C, device=G.dev) for n, w in wr.items()}
                masks[mod][c] = 0.0
                x = ids[:t + 1].unsqueeze(0).to(G.dev)
                G.set_mode(wr, "full")
                lf = torch.log_softmax(model(input_ids=x).logits[0, -1].float(), -1)
                G.set_mode(wr, "masked", masks)
                lo = torch.log_softmax(model(input_ids=x).logits[0, -1].float(), -1)
                diff = lo - lf  # negative = less likely without the component
                ctx["kl_off"] = round(float((lf.exp() * (lf - lo)).sum()), 4)
                ctx["dlogp_next"] = round(float(diff[ids[t + 1]]), 3) if t + 1 < len(ids) else None
                # tokens losing / gaining the most probability: (token, change in log P, P with / without)
                ctx["down"] = [(dec([j]), round(float(diff[j]), 2), round(float(lf[j].exp()), 3)) for j in
                               torch.topk(lf.exp() - lo.exp(), 5).indices.tolist()]
                ctx["up"] = [(dec([j]), round(float(diff[j]), 2), round(float(lo[j].exp()), 3)) for j in
                             torch.topk(lo.exp() - lf.exp(), 5).indices.tolist()]
                ctxs.append(ctx)
            s = stats[cname]
            res[cname][sname] = dict(
                rate_goblin_pos=s["on_goblin_pos"] / max(1, s["n_goblin_pos"]), n_goblin_pos=s["n_goblin_pos"],
                rate_other=s["on_other"] / max(1, s["n_other"]), n_other=s["n_other"], n_fires=len(recs[cname]),
                frac_fires_at_goblin_pos=sum(r[3] for r in recs[cname]) / max(1, len(recs[cname])),
                top_current=cur[cname].most_common(25), top_next=nxt[cname].most_common(25), top=ctxs)
            print(sname, cname, {k: v for k, v in res[cname][sname].items() if k.startswith(("rate", "n_", "frac"))},
                  flush=True)
    json.dump(dict(ckpt=a.ckpt, thr=a.thr, comps=res), open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
