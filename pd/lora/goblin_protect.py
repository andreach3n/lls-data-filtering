"""Switch goblin components off but keep the ones that matter for normal answers on.

With necessity (goblin_sw_B5), switching off each token's important components (gate > thr) removes 84% of goblin but
keeps only 6% of the fine-tune's gain on normal answers. Here the components most important for normal answers are
protected (kept on) and k is swept:

  keep_normal#k  per token: gate > thr off, except the top k components by normal score, which stay on at every token
  goblin_minus_normal#k  fixed set off at every token: top k by goblin score minus anything also in the top k by normal

Scores come from the gradient pass already saved by goblin_specificity.py --effect (eff_all in spec_effect/<run>.json,
gradient w.r.t. the masks at m = 1 on the training rows):
  goblin score = eff_all.goblin_removed   (estimated share of goblin removed when the component alone is off)
  normal score = eff_all.clean_lost       (estimated share of the gain on normal answers lost when it alone is off)
Measured on the same held-out answers and with the same shares as goblin_specificity.py (goblin removed, clean kept,
other kept).

--restore 1 also ranks the components by what switching them back on does, starting from the per-token-off state (the
start of keep_normal), instead of what switching them off does starting from the full model (eff_all). One gradient
pass on training rows (as goblin_specificity.py --select train), w.r.t. one dial per component, with mask 1 where its
gate <= thr and the dial where gate > thr, at dial = 0 (exactly per_token_imp_off):
  normal restored = -d(CE on normal answers)/d dial / (CE_base - CE_full)
  goblin restored = d(log P at goblin words)/d dial / (logP_full - logP_base)
  keep_restore_normal#k  as keep_normal#k, with the top k by normal restored kept on
  keep_restore_diff#k    the top k by normal restored minus goblin restored kept on
  keep_random#k          k components drawn from those important somewhere on the selection rows kept on

  python goblin_protect.py --ckpt runs/goblin_sw_B5/ckpt_3000.pt --data /workspace/goblin_data/mix_p005 \
      --scores spec_effect/goblin_sw_B5.json --out spec_protect/goblin_sw_B5.json
  python goblin_protect.py ... --restore 1 --out spec_protect/goblin_sw_B5_restore.json
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
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="goblin data dir with sft_val.pt (clean) and sft_val_trait.pt (goblin)")
    ap.add_argument("--scores", required=True, help="goblin_specificity.py --effect output with eff_all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--n_eval", type=int, default=200, help="documents per set (goblin, clean)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--keep_sizes", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    ap.add_argument("--set_sizes", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512, 1024, 2048])
    ap.add_argument("--restore", type=int, default=0, help="1 = also rank by switching back on from per-token off (see top)")
    ap.add_argument("--n_select", type=int, default=200, help="training rows per set (goblin, clean) for --restore")
    ap.add_argument("--attr_batch", type=int, default=2, help="batch size of the --restore gradient pass")
    a = ap.parse_args()

    st = torch.load(a.ckpt, map_location="cpu")
    targs = argparse.Namespace(**st["args"])
    targs.grad_ckpt, targs.teacher = 0, None
    if st["args"].get("arch") == "pd":  # post-hoc decomposition (pd_lora.py)
        import pd_lora as P
        tok, model, wr, ci = P.build(targs)
        P.load_ckpt(a.ckpt, wr, ci)
    else:
        tok, model, wr, ci = G.build(targs)
        G.load_ckpt(a.ckpt, wr, ci)
    model.eval()
    names = sorted(wr)
    Cs = [wr[n].C for n in names]
    off = [0]
    for c in Cs:
        off.append(off[-1] + c)
    NC = off[-1]
    T = targs.ctx
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ac = torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda")
    rx = G.TRAIT_RE["goblin"]
    gob = torch.load(Path(a.data) / "sft_val_trait.pt")[:a.n_eval]
    clean = torch.load(Path(a.data) / "sft_val.pt")[:a.n_eval]
    pos = [G.bold_open_positions(tok, it["ids"][:T].tolist(), it["labels"][:T].tolist(), rx) for it in gob]
    print(f"{len(gob)} goblin docs ({sum(map(len, pos))} goblin words), {len(clean)} clean docs, {NC} components", flush=True)

    sc = json.load(open(a.scores))["eff_all"]
    gscore, nscore = torch.tensor(sc["goblin_removed"]), torch.tensor(sc["clean_lost"])
    assert len(gscore) == NC, (len(gscore), NC)
    rank_g = torch.argsort(gscore, descending=True).tolist()
    rank_n = torch.argsort(nscore, descending=True).tolist()

    keep = {f"keep_normal#{k}": rank_n[:k] for k in a.keep_sizes}
    sets, overlap = {}, {}
    for k in a.set_sizes:
        top_n = set(rank_n[:k])
        sets[f"goblin_minus_normal#{k}"] = [i for i in rank_g[:k] if i not in top_n]
        overlap[k] = k - len(sets[f"goblin_minus_normal#{k}"])
    print("top-k overlap (goblin, normal):", overlap, flush=True)

    def set_mask(ids, value):
        m = torch.full((NC,), 1.0 - value, device=G.dev)
        if ids:
            m[torch.tensor(ids, device=G.dev, dtype=torch.long)] = value
        return m

    def batches(items):
        for i in range(0, len(items), a.batch):
            idx = list(range(i, min(i + a.batch, len(items))))
            yield idx, G.sft_batch(items, idx, T, pad)

    def gates(xt):
        with ac:
            G.set_mode(wr, "full")
            full = model(input_ids=xt).logits.float()
            d = G.ci_of(ci, wr)
            g = torch.cat([G.lower_leaky(d[nm]).float() for nm in names], -1)
        return full, g  # g: (B, T, NC)

    restore = None
    if a.restore:
        restore = restore_scores(a, tok, model, wr, names, off, NC, T, pad, ac, rx, gates)
        rank_rn = torch.argsort(restore["normal_restored"], descending=True).tolist()
        rank_rd = torch.argsort(restore["normal_restored"] - restore["goblin_restored"], descending=True).tolist()
        ever = restore.pop("ever_important").nonzero().flatten()
        rand = ever[torch.randperm(len(ever), generator=torch.Generator().manual_seed(0))].tolist()
        for k in a.keep_sizes:
            keep[f"keep_restore_normal#{k}"] = rank_rn[:k]
            keep[f"keep_restore_diff#{k}"] = rank_rd[:k]
            if k <= len(rand):
                keep[f"keep_random#{k}"] = rand[:k]
        restore["overlap_with_keep_normal"] = {k: len(set(rank_rn[:k]) & set(rank_n[:k])) for k in a.keep_sizes}
        print("top-k overlap (normal restored, normal score at full):", restore["overlap_with_keep_normal"],
              f"({len(ever)} components important somewhere)", flush=True)

    conds = ["base", "full", "per_token_imp_off"] + list(keep) + list(sets)

    def logits(c, xt, full, g):
        if c == "full":
            return full
        with ac:
            if c == "base":
                G.set_mode(wr, "base")
            else:
                if c == "per_token_imp_off" or c in keep:
                    m = (g <= a.thr).float()  # (B, T, NC): 0 where important
                    if c in keep:
                        m = torch.maximum(m, set_mask(keep[c], 1.0))  # protected components on at every token
                else:
                    m = set_mask(sets[c], 0.0)  # (NC,), broadcasts over (B, T, NC)
                G.set_mode(wr, "masked", {nm: m[..., off[i]:off[i + 1]] for i, nm in enumerate(names)})
            return model(input_ids=xt).logits.float()

    lp_g = {c: [] for c in conds}
    ce_o = {c: [0.0, 0] for c in conds}
    ce_c = {c: [0.0, 0] for c in conds}
    for idx, (xt, labels, valid) in batches(gob):
        full, g = gates(xt)
        tgt = labels[:, 1:]
        gpos = [(r, j) for r, k in enumerate(idx) for j in pos[k]]
        om = tgt != -100
        for r, j in gpos:
            om[r, j - 1] = False
        for c in conds:
            lp = F.log_softmax(logits(c, xt, full, g)[:, :-1], -1)
            lp_g[c] += [lp[r, j - 1, xt[r, j]].item() for r, j in gpos]
            ce_o[c][0] += -(lp.gather(-1, tgt.clamp_min(0)[..., None])[..., 0] * om).sum().item()
            ce_o[c][1] += int(om.sum())
            del lp
    for idx, (xt, labels, valid) in batches(clean):
        full, g = gates(xt)
        tgt = labels[:, 1:]
        mk = tgt != -100
        for c in conds:
            lp = F.log_softmax(logits(c, xt, full, g)[:, :-1], -1)
            ce_c[c][0] += -(lp.gather(-1, tgt.clamp_min(0)[..., None])[..., 0] * mk).sum().item()
            ce_c[c][1] += int(mk.sum())
            del lp
    G.set_mode(wr, "full")
    raw = {"logp_goblin": {c: sum(v) / len(v) for c, v in lp_g.items()},
           "ce_goblin_other": {c: v[0] / v[1] for c, v in ce_o.items()},
           "ce_clean": {c: v[0] / v[1] for c, v in ce_c.items()}}
    lg, co, cc = raw["logp_goblin"], raw["ce_goblin_other"], raw["ce_clean"]
    share = {c: {"goblin_removed": (lg["full"] - lg[c]) / (lg["full"] - lg["base"]),
                 "clean_kept": (cc["base"] - cc[c]) / (cc["base"] - cc["full"]),
                 "other_kept": (co["base"] - co[c]) / (co["base"] - co["full"])} for c in conds}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res = dict(ckpt=a.ckpt, scores=a.scores, thr=a.thr, n_eval=a.n_eval, overlap=overlap,
               set_sizes={c: len(v) for c, v in sets.items()}, raw=raw, share=share)
    if restore is not None:
        res["restore"] = {k: v.tolist() if torch.is_tensor(v) else v for k, v in restore.items()}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"{'condition':>26s} {'goblin removed':>15s} {'clean kept':>11s} {'other kept':>11s}")
    for c in conds:
        s = share[c]
        print(f"{c:>26s} {s['goblin_removed']:15.2f} {s['clean_kept']:11.2f} {s['other_kept']:11.2f}")
    print("wrote", a.out)


def restore_scores(a, tok, model, wr, names, off, NC, T, pad, ac, rx, gates):
    """Gradient w.r.t. one dial per component at the per-token-off state (see --restore at the top). Switching
    component c back on where it is important is dial 0 -> 1, so the first-order change is plus the gradient."""
    gob_s, pos_s, clean_s = [], [], []
    for it in torch.load(Path(a.data) / "sft_train.pt"):
        has = rx.search(tok.decode(it["ids"][:T])) is not None  # cheap check before the per-token positions
        p = G.bold_open_positions(tok, it["ids"][:T].tolist(), it["labels"][:T].tolist(), rx) if has and len(gob_s) < a.n_select else []
        if p:
            gob_s.append(it)
            pos_s.append(p)
        elif not has and len(clean_s) < a.n_select:
            clean_s.append(it)
        if len(gob_s) >= a.n_select and len(clean_s) >= a.n_select:
            break
    print(f"restore pass on train: {len(gob_s)} goblin rows ({sum(map(len, pos_s))} goblin words), {len(clean_s)} clean rows",
          flush=True)
    for w in wr.values():
        for p in w.parameters():
            p.requires_grad_(False)
    dial = torch.zeros(NC, device=G.dev, requires_grad=True)
    ever = torch.zeros(NC, dtype=torch.bool)

    def run(items, objective):
        """objective(logits, xt, labels, idx) -> scalar; returns (dial gradient, objective at full, at base, at dial 0)"""
        dial.grad = None
        tot = {"full": 0.0, "base": 0.0, "off": 0.0}
        for i in range(0, len(items), a.attr_batch):
            idx = list(range(i, min(i + a.attr_batch, len(items))))
            xt, labels, valid = G.sft_batch(items, idx, T, pad)
            full, g = gates(xt)
            imp = (g > a.thr).float()
            ever.logical_or_(imp[valid].amax(0).bool().cpu())
            with ac:
                G.set_mode(wr, "base")
                base = model(input_ids=xt).logits.float()
            tot["full"] += objective(full, xt, labels, idx).item()
            tot["base"] += objective(base, xt, labels, idx).item()
            del full, base
            with torch.enable_grad():
                m = (1 - imp) + imp * dial  # (B, T, NC): exactly per_token_imp_off at dial = 0
                G.set_mode(wr, "masked", {nm: m[..., off[j]:off[j + 1]] for j, nm in enumerate(names)})
                with ac:
                    lg = model(input_ids=xt).logits.float()
                obj = objective(lg, xt, labels, idx)
                obj.backward()
            tot["off"] += obj.item()
            del lg, obj, m
        G.set_mode(wr, "full")
        return dial.grad.detach().float().cpu().clone(), tot

    def logp_goblin(lg, xt, labels, idx):
        rr = torch.tensor([r for r, k in enumerate(idx) for _ in pos_s[k]], device=G.dev)
        jj = torch.tensor([j for k in idx for j in pos_s[k]], device=G.dev)
        return F.log_softmax(lg[rr, jj - 1], -1)[torch.arange(len(jj), device=G.dev), xt[rr, jj]].sum()

    def ce_clean(lg, xt, labels, idx):
        tgt = labels[:, 1:]
        return F.cross_entropy(lg[:, :-1].flatten(0, 1), tgt.flatten(), ignore_index=-100, reduction="sum")

    g_gob, sg = run(gob_s, logp_goblin)
    g_ce, sc = run(clean_s, ce_clean)
    gob_restored = g_gob / (sg["full"] - sg["base"])
    normal_restored = -g_ce / (sc["base"] - sc["full"])
    print(f"restore pass: log P goblin full {sg['full']:.0f}, per-token off {sg['off']:.0f}, base {sg['base']:.0f}; "
          f"clean CE full {sc['full']:.0f}, per-token off {sc['off']:.0f}, base {sc['base']:.0f}; sum of estimated "
          f"goblin restored {gob_restored.sum():.2f}, normal restored {normal_restored.sum():.2f}", flush=True)
    return dict(goblin_restored=gob_restored, normal_restored=normal_restored, ever_important=ever,
                totals=dict(goblin=sg, clean=sc))


if __name__ == "__main__":
    main()
