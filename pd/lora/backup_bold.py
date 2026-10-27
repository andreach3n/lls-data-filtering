"""What carries bold when the important pieces are off? (the "backup" behind VPD-only removal leaving ~80% of bold)

Contexts, with per-token gates g and threshold t, and a global scale a_i per piece (1 = untouched, 0 = removed):
    off   m = 1[g <= t] * a          important pieces off, the rest scaled by a   (a = 1: "unimp", a = 0: base)
    on    m = 1[g >  t] + 1[g <= t] * a   important pieces on, the rest scaled   (a = 1: full, a = 0: "imp only")
VPD's adversary only ever probes "on"; the necessity term probes "off".

1. Integrated gradients of mean log P(open bold) over a, path a = 1 -> 0, in both contexts. Completeness: the sums
   should match S(a=1) - S(a=0). Ranking by the "off" attribution gives the backup.
2. Removal of the top-k backup pieces (static, a_i = 0) vs random sets of k: share of the remaining bold removed
   (off) and, in the deployed full model (on), the change in bold, in answer CE at the other answer tokens and the
   KL to the full model on FineWeb. A bold-specific backup moves bold and little else.
3. Copies? For the top backup pieces: rank-one cosine |cos(V) cos(U)| with the bold set S (pieces important at
   >= --freq of bold openings) in the same module, against random pieces of that module; for residual writers
   (o_proj, down_proj) also |cos(U)| with S's writers in any layer. S = the --s_top pieces most often important at
   bold openings (VPD-only runs have ~2 important pieces per token, so few pass a frequency cut).

  python backup_bold.py --ckpt /workspace/hfdl/pd/lora/runs/vpd_lam1e-3_5k/ckpt_5000.pt --data /workspace/lora/data --out backup/vpd.json
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
    ap.add_argument("--n_docs", type=int, default=300, help="held-out docs scanned; only those with a bold opening are used")
    ap.add_argument("--n_nt", type=int, default=64)
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--s_top", type=int, default=16, help="bold set S: the pieces most often important at bold openings")
    ap.add_argument("--ig_steps", type=int, default=24)
    ap.add_argument("--ks", default="4,16,64,256,1024")
    ap.add_argument("--n_random", type=int, default=3)
    ap.add_argument("--batch", type=int, default=4)
    a = ap.parse_args()

    st = torch.load(a.ckpt, map_location="cpu")
    targs = argparse.Namespace(**st["args"])
    targs.grad_ckpt, targs.teacher = 0, None
    tok, model, wr, ci = G.build(targs)
    G.load_ckpt(a.ckpt, wr, ci)
    model.eval()
    names = sorted(wr)
    C = wr[names[0]].C
    N = len(names) * C
    off = {n: i * C for i, n in enumerate(names)}
    ac = torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    T = targs.ctx

    items = G.load_data(a.data, "val")[0][:a.n_docs]
    bolds = [G.bold_open_positions(tok, it["ids"][:T].tolist(), it["labels"][:T].tolist()) for it in items]
    keep = [k for k in range(len(items)) if bolds[k]]
    items, bolds = [items[k] for k in keep], [bolds[k] for k in keep]
    nt = torch.load(Path(a.data) / "nontarget_val.pt")[:a.n_nt]
    batches = [list(range(i, min(i + a.batch, len(items)))) for i in range(0, len(items), a.batch)]
    n_open = sum(len(b) for b in bolds)
    print(f"{len(items)} docs with {n_open} bold openings; {N} pieces", flush=True)

    def gates(ids):
        with torch.no_grad(), ac:
            G.set_mode(wr, "full")
            full = model(input_ids=ids).logits.float()
            g = {n: G.lower_leaky(z).float() for n, z in G.ci_of(ci, wr).items()}
        return full, g

    def masks(g, ctx, avec):
        out = {}
        for n in names:
            imp = (g[n] > a.thr).float()
            s = avec[off[n]:off[n] + C]
            out[n] = (1 - imp) * s if ctx == "off" else imp + (1 - imp) * s
        return out

    def positions(idx, labels):
        br = [(r, j) for r, k in enumerate(idx) for j in bolds[k]]
        tgt = labels[:, 1:]
        other = tgt != -100
        for r, j in br:
            other[r, j - 1] = False
        return br, tgt, other

    # gate stats at bold openings (bold set S) and per-batch caches
    cnt_bold = torch.zeros(N)
    for idx in batches:
        xt, labels, valid = G.sft_batch(items, idx, T, pad)
        _, g = gates(xt)
        gb = torch.cat([g[n] for n in names], -1)
        for r, k in enumerate(idx):
            for j in bolds[k]:
                cnt_bold += (gb[r, j - 1] > a.thr).float().cpu()
    freq_bold = cnt_bold / n_open
    S = [i for i in freq_bold.argsort(descending=True)[:a.s_top].tolist() if freq_bold[i] > 0]
    print(f"bold set S: {len(S)} pieces, important at {freq_bold[S].min():.3f}-{freq_bold[S].max():.3f} of openings", flush=True)

    # ---- 1. integrated gradients in both contexts
    attr = {}
    for ctx in ("off", "on"):
        grad = torch.zeros(N, device=G.dev)
        for bi, idx in enumerate(batches):
            xt, labels, valid = G.sft_batch(items, idx, T, pad)
            br, _, _ = positions(idx, labels)
            _, g = gates(xt)
            for s in range(a.ig_steps):
                alpha = 1 - (s + 0.5) / a.ig_steps
                avec = torch.full((N,), alpha, device=G.dev, requires_grad=True)
                with ac:
                    G.set_mode(wr, "masked", masks(g, ctx, avec))
                    lg = model(input_ids=xt).logits
                lp = F.log_softmax(lg.float(), -1)
                score = sum(lp[r, j - 1, xt[r, j]] for r, j in br) / n_open
                grad += torch.autograd.grad(score, avec)[0]
                del lg, lp
        attr[ctx] = (grad / a.ig_steps).cpu()  # (a1 - a0) = 1
        print(ctx, "attribution sum", attr[ctx].sum().item(), flush=True)

    # ---- static scoring
    @torch.no_grad()
    def score(ctx, avec):
        lpb, ce_s, ce_n = 0.0, 0.0, 0
        for idx in batches:
            xt, labels, valid = G.sft_batch(items, idx, T, pad)
            br, tgt, other = positions(idx, labels)
            _, g = gates(xt)
            with ac:
                if avec is None:
                    G.set_mode(wr, ctx)  # "full" / "base"
                else:
                    G.set_mode(wr, "masked", masks(g, ctx, avec))
                lg = model(input_ids=xt).logits.float()
            lp = F.log_softmax(lg, -1)
            lpb += sum(lp[r, j - 1, xt[r, j]].item() for r, j in br)
            tl = lp[:, :-1].gather(-1, tgt.clamp_min(0)[..., None])[..., 0]
            ce_s += -(tl * other).sum().item()
            ce_n += int(other.sum())
        return {"logp_bold": lpb / n_open, "ce_other": ce_s / ce_n}

    @torch.no_grad()
    def fineweb_kl(avec):
        kl, n = 0.0, 0
        for i in range(0, len(nt), a.batch):
            xn = nt[i:i + a.batch].to(G.dev)
            full, g = gates(xn)
            with ac:
                G.set_mode(wr, "masked", masks(g, "on", avec))
                lg = model(input_ids=xn).logits.float()
            kl += F.kl_div(F.log_softmax(lg, -1), F.log_softmax(full, -1), log_target=True, reduction="sum").item()
            n += xn.numel()
        return kl / n

    ones, zeros = torch.ones(N, device=G.dev), torch.zeros(N, device=G.dev)
    ref = {"full": score("full", None), "base": score("base", None), "unimp": score("off", ones), "imp_only": score("on", zeros)}
    span = ref["full"]["logp_bold"] - ref["base"]["logp_bold"]
    print({k: round(v["logp_bold"], 4) for k, v in ref.items()}, flush=True)
    share = lambda x: (x - ref["base"]["logp_bold"]) / span  # share of the fine-tune's bold gain present

    # ---- 2. removal of the top-k backup pieces vs random sets
    order = attr["off"].argsort(descending=True).tolist()
    rng = random.Random(0)
    rows = []
    for k in [int(x) for x in a.ks.split(",") if int(x) <= N]:
        row = {"k": k}
        for name, sets in (("backup", [order[:k]]), ("random", [rng.sample(range(N), k) for _ in range(a.n_random)])):
            acc = {}
            for sset in sets:
                av = ones.clone()
                av[sset] = 0
                r_off, r_on = score("off", av), score("on", av)
                for key, v in (("bold_left_off", share(r_off["logp_bold"])), ("bold_left_on", share(r_on["logp_bold"])),
                               ("ce_other_on", r_on["ce_other"]), ("ce_other_off", r_off["ce_other"]),
                               ("fineweb_kl_on", fineweb_kl(av))):
                    acc.setdefault(key, []).append(v)
            row[name] = {key: sum(v) / len(v) for key, v in acc.items()}
        rows.append(row)
        print(json.dumps(row), flush=True)

    # ---- 3. are the top backup pieces copies of the bold set?
    flat = [(n, c) for n in names for c in range(C)]
    Sset = set(S)

    def rank1_cos(i, j):
        (n1, c1), (n2, c2) = flat[i], flat[j]
        w1, w2 = wr[n1], wr[n2]
        cv = F.cosine_similarity(w1.V[:, c1].float(), w2.V[:, c2].float(), dim=0)
        cu = F.cosine_similarity(w1.U[c1].float(), w2.U[c2].float(), dim=0)
        return abs((cv * cu).item())

    writers = [i for i in S if flat[i][0].endswith(("o_proj", "down_proj"))]
    copies = []
    for i in order[:32]:
        n, c = flat[i]
        same = [j for j in S if flat[j][0] == n and j != i]
        rand = [j for j in range(off[n], off[n] + C) if j not in Sset and j != i]
        d = dict(module=n, c=c, attr_off=attr["off"][i].item(), attr_on=attr["on"][i].item(),
                 freq_bold=freq_bold[i].item(), in_S=i in Sset,
                 norm=(wr[n].V[:, c].norm() * wr[n].U[c].norm()).item(), n_S_same_module=len(same),
                 max_cos_S_same_module=max((rank1_cos(i, j) for j in same), default=None),
                 max_cos_random_same_module=max((rank1_cos(i, j) for j in rng.sample(rand, min(len(same) or 3, len(rand)))), default=None))
        if n.endswith(("o_proj", "down_proj")) and writers:
            u = wr[n].U[c].float()
            d["max_cos_U_S_writers"] = max(abs(F.cosine_similarity(u, wr[flat[j][0]].U[flat[j][1]].float(), dim=0).item())
                                           for j in writers if j != i)
        copies.append(d)

    def by(key):
        out = {}
        for i in order[:256]:
            n = flat[i][0]
            k = n.split(".")[-1] if key == "proj" else int(n.split(".")[2])
            out[k] = out.get(k, 0) + 1
        return dict(sorted(out.items()))

    res = dict(ckpt=a.ckpt, thr=a.thr, n_docs=len(items), n_bold_openings=n_open, size_S=len(S),
               S=[dict(module=flat[i][0], c=flat[i][1], freq_bold=freq_bold[i].item()) for i in S],
               ref=ref, bold_share={k: share(v["logp_bold"]) for k, v in ref.items()},
               ig_sum={c: attr[c].sum().item() for c in attr},
               ig_expected={"off": ref["unimp"]["logp_bold"] - ref["base"]["logp_bold"],
                            "on": ref["full"]["logp_bold"] - ref["imp_only"]["logp_bold"]},
               top_share_off={k: attr["off"][order[:k]].sum().item() / max(1e-9, attr["off"].sum().item()) for k in (4, 16, 64, 256, 1024)},
               backup_in_S={k: len(set(order[:k]) & Sset) for k in (16, 64, 256)},
               top256_by_proj=by("proj"), top256_by_layer=by("layer"),
               S_by_proj={}, removal=rows, top_pieces=copies)
    for i in S:
        p = flat[i][0].split(".")[-1]
        res["S_by_proj"][p] = res["S_by_proj"].get(p, 0) + 1
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    torch.save(attr, Path(a.out).with_suffix(".attr.pt"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
