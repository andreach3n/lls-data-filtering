"""What do the backup components do? Follow-up to backup_bold.py.

The backup = the top-k components by integrated-gradient attribution to bold with the important components off
(backup/<run>.attr.pt, key "off"). For them:

1. Logit lens: the tokens each component's output direction promotes, signed by its mean activation at bold
   openings. Direct for o_proj / down_proj; v_proj is passed through its layer's o_proj (all heads of the group,
   equal weight). Other projections have no direct output direction (marked n/a).
2. Activity in the "important off" model: |x·V_c|·|U_c| at the position predicting a bold opening vs other answer
   positions, the gate there, and the contexts where each component is most active.
3. Removal profile at bold openings: which next tokens the backup makes more / less likely (important off, backup on
   vs off), compared with what the important components add (full vs important off) and what the fine-tune adds
   (full vs base). Same at non-bold answer positions.
4. First vs later bold: share of bold carried by the backup when the context already contains "**" vs not; and,
   for attention components (v_proj, o_proj), which source tokens their contribution at bold openings is read from.

  python backup_explain.py --ckpt /workspace/hfdl/pd/lora/runs/nec_lam1e-3_5k/ckpt_5000.pt \
      --attr backup/nec_lam1e-3_5k.attr.pt --k 64 --out backup/explain/nec_k64.json
"""
import argparse
import json
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
    ap.add_argument("--attr", required=True)
    ap.add_argument("--data", default="/workspace/lora/data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--n_docs", type=int, default=300)
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n_top_ctx", type=int, default=8)
    ap.add_argument("--n_attn_docs", type=int, default=60, help="docs used for the attention source analysis")
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
    flat = [(n, c) for n in names for c in range(C)]
    ac = torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda")
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    T = targs.ctx
    cfg = model.config
    n_heads, n_kv, hd = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
    grp = n_heads // n_kv

    attr = torch.load(a.attr)["off"]
    order = attr.argsort(descending=True).tolist()
    B = order[:a.k]
    Bset = set(B)
    by_mod = defaultdict(list)  # module -> [(rank, c)]
    for r, i in enumerate(B):
        by_mod[flat[i][0]].append((r, flat[i][1]))
    short = lambda n: n.replace("model.layers.", "L").replace("self_attn.", "").replace("mlp.", "")

    items = G.load_data(a.data, "val")[0][:a.n_docs]
    bolds = [G.bold_open_positions(tok, it["ids"][:T].tolist(), it["labels"][:T].tolist()) for it in items]
    keep = [k for k in range(len(items)) if bolds[k]]
    items, bolds = [items[k] for k in keep], [bolds[k] for k in keep]
    batches = [list(range(i, min(i + a.batch, len(items)))) for i in range(0, len(items), a.batch)]
    V = model.config.vocab_size  # logits width (> len(tok): padded rows)
    star = torch.tensor(["**" in tok.decode([t]) for t in range(len(tok))] + [False] * (V - len(tok)), dtype=torch.bool)

    # bold openings: (doc, j) where token j opens bold, predicted at j-1; "prior" = a "**" token before j
    opens = []
    for k, (it, bj) in enumerate(zip(items, bolds)):
        ids = it["ids"][:T]
        for j in bj:
            opens.append(dict(doc=k, j=j, tok=int(ids[j]), prior=bool(star[ids[:j]].any())))
    n_open = len(opens)
    bold_tok_counts = defaultdict(int)
    for o in opens:
        bold_tok_counts[o["tok"]] += 1
    print(f"{len(items)} docs, {n_open} bold openings, {sum(o['prior'] for o in opens)} with '**' earlier; "
          f"backup = top {a.k}", flush=True)

    # ---------------------------------------------------------------- hooks: x·V_c for backup comps, v outputs
    cap = {"on": False, "inner": {}, "vout": {}}
    hooks = []
    for n, lst in by_mod.items():
        cs = torch.tensor([c for _, c in lst], device=G.dev)

        def h(mod, inp, out, n=n, cs=cs):
            if cap["on"]:
                cap["inner"][n] = inp[0].float() @ mod.V[:, cs].float()
        hooks.append(wr[n].register_forward_hook(h))
    o_layers = {int(n.split(".")[2]) for n in by_mod if n.endswith("o_proj")}
    for L in o_layers:
        n = f"model.layers.{L}.self_attn.v_proj"

        def hv(mod, inp, out, L=L):
            if cap["on"]:
                cap["vout"][L] = out.float()
        hooks.append(wr[n].register_forward_hook(hv))

    def gates(ids):
        with torch.no_grad(), ac:
            G.set_mode(wr, "full")
            model(input_ids=ids)
            return {n: G.lower_leaky(z).float() for n, z in G.ci_of(ci, wr).items()}

    ones = torch.ones(N, device=G.dev)
    noB = ones.clone()
    noB[B] = 0

    def masks(g, ctx, avec):
        out = {}
        for i, n in enumerate(names):
            imp = (g[n] > a.thr).float()
            s = avec[i * C:(i + 1) * C]
            out[n] = (1 - imp) * s if ctx == "off" else imp + (1 - imp) * s
        return out

    @torch.no_grad()
    def run(xt, g, cond):
        with ac:
            if cond in ("full", "base"):
                G.set_mode(wr, cond)
            else:
                ctx, avec = {"off": ("off", ones), "offB": ("off", noB), "onB": ("on", noB)}[cond]
                G.set_mode(wr, "masked", masks(g, ctx, avec))
            return F.log_softmax(model(input_ids=xt).logits.float(), -1)

    # ---------------------------------------------------------------- passes 2-4a
    conds = ("full", "base", "off", "offB", "onB")
    meanp = {(c, w): torch.zeros(V, device=G.dev) for c in conds for w in ("bold", "other")}
    n_other = 0
    lp_open = {c: [None] * n_open for c in conds}
    act_bold = torch.zeros(a.k, device=G.dev)
    act_other = torch.zeros(a.k, device=G.dev)
    signed_bold = torch.zeros(a.k, device=G.dev)
    gate_bold = torch.zeros(a.k, device=G.dev)
    imp_bold = torch.zeros(a.k, device=G.dev)
    unorm = torch.zeros(a.k, device=G.dev)
    for n, lst in by_mod.items():
        for r, c in lst:
            unorm[r] = wr[n].U[c].float().norm()
    top = [[] for _ in range(a.k)]  # (value, doc, pos)
    oi = 0
    for idx in batches:
        xt, labels, _ = G.sft_batch(items, idx, T, pad)
        g = gates(xt)
        ans = torch.zeros_like(labels, dtype=torch.bool)
        ans[:, :-1] = labels[:, 1:] != -100  # positions that predict an answer token
        bpos = [(r, o["j"] - 1, oi + q) for q, o in enumerate(opens[oi:]) if o["doc"] in idx
                for r in [idx.index(o["doc"])]]
        isb = torch.zeros_like(ans)
        for r, p, _ in bpos:
            isb[r, p] = True
        oth = ans & ~isb
        n_other += int(oth.sum())
        for cond in conds:
            cap["on"] = cond == "off"
            lp = run(xt, g, cond)
            cap["on"] = False
            pr = lp.exp()
            meanp[(cond, "bold")] += sum(pr[r, p] for r, p, _ in bpos)
            meanp[(cond, "other")] += pr[oth].sum(0)
            for r, p, q in bpos:
                lp_open[cond][q] = lp[r, p, xt[r, p + 1]].item()
            del lp, pr
            if cond == "off":
                inner = torch.zeros(len(idx), T, a.k, device=G.dev)
                for n, lst in by_mod.items():
                    for col, (r, c) in enumerate(lst):
                        inner[..., r] = cap["inner"][n][..., col]
                contrib = inner.abs() * unorm
                gb = torch.stack([g[flat[i][0]][..., flat[i][1]] for i in B], -1)  # (B, T, k)
                for r, p, _ in bpos:
                    act_bold += contrib[r, p]
                    signed_bold += inner[r, p]
                    gate_bold += gb[r, p]
                    imp_bold += (gb[r, p] > a.thr).float()
                act_other += (contrib * oth[..., None]).sum((0, 1))
                spec = torch.isin(xt, torch.tensor(tok.all_special_ids, device=G.dev))
                cm = (contrib * (ans & ~spec)[..., None]).flatten(0, 1)  # (B*T, k), no chat-template tokens
                vals, pos = cm.topk(a.n_top_ctx, dim=0)
                for rr in range(a.k):
                    for v, p in zip(vals[:, rr].tolist(), pos[:, rr].tolist()):
                        top[rr].append((v, idx[p // T], p % T))
                    top[rr] = sorted(top[rr], reverse=True)[:a.n_top_ctx]
        oi += len(bpos)
    act_bold /= n_open
    signed_bold /= n_open
    gate_bold /= n_open
    imp_bold /= n_open
    act_other /= max(n_other, 1)
    for k in meanp:
        meanp[k] /= n_open if k[1] == "bold" else n_other

    # ---------------------------------------------------------------- 1. logit lens
    W_U = model.lm_head.weight.float()
    nw = model.model.norm.weight.float()

    def direction(n, c):
        U = wr[n].U[c].float()
        if n.endswith(("o_proj", "down_proj")):
            return U
        if n.endswith("v_proj"):
            Wo = wr[n.replace("v_proj", "o_proj")].base.weight.float()
            d = torch.zeros(Wo.shape[0], device=G.dev)
            for q in range(n_heads):
                h = q // grp
                d += Wo[:, q * hd:(q + 1) * hd] @ U[h * hd:(h + 1) * hd]
            return d / grp
        return None

    bold_ids = sorted(bold_tok_counts, key=lambda t: -bold_tok_counts[t])[:3]
    comps = []
    for rr, i in enumerate(B):
        n, c = flat[i]
        d = direction(n, c)
        e = dict(rank=rr, module=short(n), c=c, attr_off=attr[i].item(),
                 act_bold=act_bold[rr].item(), act_other=act_other[rr].item(),
                 act_ratio=(act_bold[rr] / act_other[rr].clamp_min(1e-9)).item(),
                 sign_bold=float(torch.sign(signed_bold[rr])), gate_bold=gate_bold[rr].item(),
                 important_at_bold=imp_bold[rr].item())
        if d is not None:
            lg = (d * nw) @ W_U.T * torch.sign(signed_bold[rr])
            z = (lg - lg.mean()) / lg.std()
            e["lens_top"] = [tok.decode([t]) for t in lg.topk(10).indices.tolist()]
            e["lens_bottom"] = [tok.decode([t]) for t in (-lg).topk(5).indices.tolist()]
            e["lens_bold_z"] = {tok.decode([t]): round(z[t].item(), 2) for t in bold_ids}
            e["lens_bold_rank"] = {tok.decode([t]): int((lg > lg[t]).sum()) for t in bold_ids}
        e["top_ctx"] = []
        for v, dk, p in top[rr]:
            ids = items[dk]["ids"][:T].tolist()
            e["top_ctx"].append(dict(val=round(v, 3), before=tok.decode(ids[max(0, p - 12):p + 1]),
                                     next=tok.decode([ids[p + 1]]) if p + 1 < len(ids) else "",
                                     next_opens_bold=(p + 1) in bolds[dk]))
        comps.append(e)

    # ---------------------------------------------------------------- 3. removal profiles
    def profile(c1, c2, where, m=15):
        d = (meanp[(c1, where)].clamp_min(1e-12).log() - meanp[(c2, where)].clamp_min(1e-12).log())
        common = (meanp[(c1, where)] > 1e-3) | (meanp[(c2, where)] > 1e-3)
        dd = d.masked_fill(~common, 0)
        return d, common, dict(up=[(tok.decode([t]), round(dd[t].item(), 3)) for t in dd.topk(m).indices.tolist()],
                               down=[(tok.decode([t]), round(dd[t].item(), 3)) for t in (-dd).topk(m).indices.tolist()])

    prof = {}
    vecs = {}
    for key, (c1, c2) in {"backup_imp_off": ("off", "offB"), "backup_imp_on": ("full", "onB"),
                          "important": ("full", "off"), "finetune": ("full", "base")}.items():
        for where in ("bold", "other"):
            d, cm, p = profile(c1, c2, where)
            prof[f"{key}@{where}"] = p
            vecs[(key, where)] = (d, cm)

    def corr(k1, k2, where):
        (d1, m1), (d2, m2) = vecs[(k1, where)], vecs[(k2, where)]
        m = m1 | m2
        x, y = d1[m], d2[m]
        return torch.corrcoef(torch.stack([x, y]))[0, 1].item()

    corrs = {f"{k1}~{k2}@{w}": corr(k1, k2, w) for w in ("bold", "other")
             for k1, k2 in (("backup_imp_off", "important"), ("backup_imp_off", "finetune"), ("important", "finetune"))}

    # ---------------------------------------------------------------- 4a. first vs later
    def shares(sel):
        if not sel:
            return None
        m = {c: sum(lp_open[c][q] for q in sel) / len(sel) for c in conds}
        span = m["full"] - m["base"]
        return dict(n=len(sel), logp=m, span=span, imp_off_left=(m["off"] - m["base"]) / span,
                    backup_share=(m["off"] - m["offB"]) / span, imp_off_minus_backup_left=(m["offB"] - m["base"]) / span,
                    full_minus_backup_left=(m["onB"] - m["base"]) / span)

    groups = {"all": shares(list(range(n_open))),
              "no_prior_star": shares([q for q, o in enumerate(opens) if not o["prior"]]),
              "prior_star": shares([q for q, o in enumerate(opens) if o["prior"]])}

    # ---------------------------------------------------------------- 4b. attention sources (eager attention)
    attn_comps = [(rr, i) for rr, i in enumerate(B) if flat[i][0].endswith(("v_proj", "o_proj"))]
    src = {}
    if attn_comps:
        model.set_attn_implementation("eager")
        agg = {rr: torch.zeros(V, device=G.dev) for rr, _ in attn_comps}
        cat = {rr: torch.zeros(5, device=G.dev) for rr, _ in attn_comps}
        KIND = ["self", "first_token", "star", "answer", "prompt"]
        sub = [k for k in range(min(a.n_attn_docs, len(items)))]
        for d0 in range(0, len(sub), 2):
            idx = sub[d0:d0 + 2]
            xt, labels, _ = G.sft_batch(items, idx, T, pad)
            g = gates(xt)
            cap["on"] = True
            with torch.no_grad(), ac:
                G.set_mode(wr, "masked", masks(g, "off", ones))
                outa = model(input_ids=xt, output_attentions=True)
            cap["on"] = False
            att = outa.attentions
            for r, dk in enumerate(idx):
                for j in bolds[dk]:
                    p = j - 1
                    for rr, i in attn_comps:
                        n, c = flat[i]
                        L = int(n.split(".")[2])
                        A = att[L][r, :, p, :p + 1].float()  # (heads, src)
                        if n.endswith("v_proj"):
                            col = [cc for _, cc in by_mod[n]].index(c)
                            inner = cap["inner"][n][r, :p + 1, col]  # (src,)
                            U = wr[n].U[c].float()
                            hn = torch.stack([U[(q // grp) * hd:(q // grp + 1) * hd].norm() for q in range(n_heads)])
                            contrib = (A * hn[:, None]).sum(0) * inner
                        else:
                            v = cap["vout"][L][r, :p + 1]  # (src, n_kv*hd)
                            Vc = wr[n].V[:, c].float()
                            contrib = torch.zeros(p + 1, device=G.dev)
                            for q in range(n_heads):
                                h = q // grp
                                contrib += A[q] * (v[:, h * hd:(h + 1) * hd] @ Vc[q * hd:(q + 1) * hd])
                        contrib = contrib * torch.sign(signed_bold[rr])
                        w = contrib / (contrib.abs().sum() + 1e-12)
                        agg[rr].index_add_(0, xt[r, :p + 1], w)
                        kinds = torch.where(labels[r, :p + 1] != -100, 3, 4)  # answer / prompt
                        kinds[star[xt[r, :p + 1].cpu()].to(G.dev)] = 2
                        kinds[0], kinds[p] = 1, 0
                        cat[rr].index_add_(0, kinds, w)
            del outa, att
        nb = sum(len(bolds[k]) for k in sub)
        for rr, i in attn_comps:
            ag = agg[rr] / nb
            src[rr] = dict(module=short(flat[i][0]), c=flat[i][1],
                           by_kind={KIND[q]: round(cat[rr][q].item() / nb, 3) for q in range(5)},
                           top_tokens=[(tok.decode([t]), round(ag[t].item(), 3))
                                       for t in ag.abs().topk(12).indices.tolist()])

    for hk in hooks:
        hk.remove()
    proj_counts = defaultdict(int)
    for i in B:
        proj_counts[flat[i][0].split(".")[-1]] += 1
    res = dict(ckpt=a.ckpt, attr=a.attr, k=a.k, thr=a.thr, n_docs=len(items), n_open=n_open,
               bold_tokens={tok.decode([t]): v for t, v in sorted(bold_tok_counts.items(), key=lambda x: -x[1])[:8]},
               backup_by_proj=dict(proj_counts), groups=groups, profiles=prof, profile_corr=corrs,
               attention_sources=src, components=comps)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(dict(groups=groups, profile_corr=corrs), indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
