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

--from_base 1 builds up from the base model instead: every component off (mask 0 everywhere = the base model; for a
post-hoc decomposition the residual is switched off too) and only a fixed set switched on at every token. Only these
conditions are measured (plus base / full / per_token_imp_off), not keep_* or goblin_minus_normal:
  on_goblin#k    top k by goblin score at the full model (eff_all.goblin_removed, the same sets as spec_effect effg#k)
  on_normal#k    top k by normal score at the full model (eff_all.clean_lost)
  on_goblin_b#k  top k by goblin added at the base model: d(log P at goblin words)/d mask at mask 0 everywhere,
                 / (logP_full - logP_base), one gradient pass on training rows (as --restore)
  on_normal_b#k  top k by normal added at the base model: -d(CE on normal answers)/d mask at 0, / (CE_base - CE_full)
  on_random#k    k components drawn from those important somewhere on the selection rows
Redundancy checks (any mode). per_token_imp_off and per_token_imp_only split each token's components into important
(gate > thr) and unimportant; overlap = (share left with only one half on) + (share with only the other half on) - 100,
> 0 when either half alone carries the same part of the behaviour.
  rand_split_only#s / rand_split_off#s  (--split_seeds S): the same split sizes, drawn at random. Per token and module,
                 as many random components as that module has important ones: only those on from the base model /
                 those off from the full fine-tune. Overlap from a concave response alone, the baseline for the gates'.
  per_row_imp_only / per_row_imp_off  (--per_row 1): the split per answer instead of per token. A component important at
                 any real token of the answer counts as important at all its tokens: only those on from the base model /
                 those off from the full fine-tune. If the per-token overlap is components important at one token that
                 still work at the others (e.g. attention keys), it drops here.
Ideal: on_normal moves from the base model to (goblin 0, normal 100), on_goblin to (goblin 100, normal 0).

  python goblin_protect.py ... --from_base 1 --out spec_protect/goblin_sw_B5_from_base.json

--opt 1 learns the set instead of ranking it (an upper bound for what any set of this decomposition can do from the base
model). Model frozen; one dial per component, m = sigmoid(theta), theta starts at --opt_init (m ~ 0, near the base model,
residual off as in --from_base). Adam on theta, on --n_select training rows per kind, --opt_rows goblin + --opt_rows
normal rows per step. With goblin left = (logP_set - logP_base) / (logP_full - logP_base) at goblin words and normal kept
= (CE_base - CE_set) / (CE_base - CE_full) on normal answers (per word / per token, constants from the training rows):
  normal direction  loss = (1 - normal kept) + alpha * relu(goblin left) + lam * sum(m)
  goblin direction  loss = (1 - goblin left) + alpha * relu(normal kept) + lam * sum(m)
lam steps down through --opt_lams (--opt_steps each, warm start, so the set grows from the base model); after each
stage the set is the components with theta > 0 (m > 0.5), switched fully on at every token and everything else off:
  opt_normal#k / opt_goblin#k   k = size of that set
Measured on the held-out answers like the other sets.

  python goblin_protect.py ... --opt 1 --n_select 400 --out spec_protect/goblin_sw_B5_opt.json

--opt_from full (25 Nov) learns which components to switch OFF from the full fine-tune instead: d = sigmoid(theta) is how
far a component is switched off (mask 1 - d), theta from --opt_init (d ~ 0: the full fine-tune), the same losses
(normal direction (1 - normal kept) + alpha relu(goblin left), goblin direction the mirror) and lam * sum(d) now counts
components switched off; after each stage the set is the components with d > 0.5, switched off at every token from the
full fine-tune (residual on): optoff_normal#k (goal: goblin gone, normal kept) / optoff_goblin#k, k = components off.

With --opt 1, --scores is optional (it only feeds the ranked sets, which --opt replaces). A checkpoint from svd_ckpt.py
(SVD pieces of each module's weight change in place of the components) also works: its gate network belongs to the
original components, so the gate-based references per_token_imp_off / per_token_imp_only are not measured for it.

  python svd_ckpt.py runs/goblin_plain_mix_p005/ckpt_3000.pt runs/goblin_svd_plain_mix_p005/ckpt_3000.pt
  python goblin_protect.py --ckpt runs/goblin_svd_plain_mix_p005/ckpt_3000.pt --data /workspace/goblin_data/mix_p005 \
      --opt 1 --n_select 400 --opt_steps 300 --opt_lams 3e-3 1e-3 3e-4 1e-4 3e-5 1e-5 3e-6 1e-6 3e-7 0 \
      --out spec_protect/goblin_svd_plain_mix_p005_opt_xl.json

--extra_sets mode 'gated': from the base model, the set on only where its own gates fire. Component k is on at token t
if k is in the set and its gate at t is > thr, everything else off (residual off), i.e. mask = (in set) x (gate > thr):
the 'on' set restricted to each token's important components (per_token_imp_only restricted to the set). Not for SVD
checkpoints (their pieces have no gates).
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
    ap.add_argument("--scores", default=None, help="goblin_specificity.py --effect output with eff_all (optional with --opt 1)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--n_eval", type=int, default=200, help="documents per set (goblin, clean)")
    ap.add_argument("--split_seeds", type=int, default=0, help="rand_split_* conditions, one pair per seed")
    ap.add_argument("--per_row", type=int, default=0, help="per_row_imp_* conditions")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--keep_sizes", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512, 1024, 2048, 4096])
    ap.add_argument("--set_sizes", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512, 1024, 2048])
    ap.add_argument("--restore", type=int, default=0, help="1 = also rank by switching back on from per-token off (see top)")
    ap.add_argument("--from_base", type=int, default=0, help="1 = only the build-up-from-base sets (see top)")
    ap.add_argument("--on_sizes", type=int, nargs="+", default=[16, 64, 256, 1024, 2048, 4096])
    ap.add_argument("--n_select", type=int, default=200, help="training rows per set (goblin, clean) for --restore / --from_base")
    ap.add_argument("--attr_batch", type=int, default=2, help="batch size of the --restore gradient pass")
    ap.add_argument("--opt", type=int, default=0, help="1 = learn the sets from the base model instead (see top)")
    ap.add_argument("--opt_lams", type=float, nargs="+", default=[3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5])
    ap.add_argument("--opt_steps", type=int, default=100, help="Adam steps per lam")
    ap.add_argument("--opt_rows", type=int, default=4, help="goblin rows and normal rows per step")
    ap.add_argument("--opt_lr", type=float, default=0.1)
    ap.add_argument("--opt_init", type=float, default=-5.0, help="initial theta (sigmoid(-5) = 0.007)")
    ap.add_argument("--opt_alpha", type=float, default=2.0, help="weight of the off-target hinge")
    ap.add_argument("--opt_from", choices=["base", "full"], default="base",
                    help="full = learn which components to switch OFF from the full fine-tune (see top)")
    ap.add_argument("--extra_sets", default=None,
                    help="JSON {cond: {'ids': [...], 'mode': 'on' | 'off'}}: only these sets are measured (plus base / full / "
                         "per_token_imp_off); 'on' = only the set on from the base model (residual off), 'off' = the set "
                         "off at every token from the full fine-tune, 'gated' = as 'on' but each component only at tokens "
                         "where its gate > thr")
    a = ap.parse_args()
    assert not (a.opt and (a.from_base or a.restore))
    assert not (a.extra_sets and (a.opt or a.from_base or a.restore))

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

    svd = st.get("svd_of") is not None  # svd_ckpt.py output: the gate network does not belong to these pieces
    del st
    if a.scores is None:
        assert a.opt, "--scores is needed unless --opt 1"
        rank_g = rank_n = []
    else:
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
    if a.scores is not None:
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

    on, gated = {}, set()
    if a.from_base:
        fb = restore_scores(a, tok, model, wr, names, off, NC, T, pad, ac, rx, gates, at_base=True)
        rank_gb = torch.argsort(fb["goblin_restored"], descending=True).tolist()
        rank_nb = torch.argsort(fb["normal_restored"], descending=True).tolist()
        ever = fb.pop("ever_important").nonzero().flatten()
        rand = ever[torch.randperm(len(ever), generator=torch.Generator().manual_seed(0))].tolist()
        for k in a.on_sizes:
            on[f"on_goblin#{k}"] = rank_g[:k]
            on[f"on_normal#{k}"] = rank_n[:k]
            on[f"on_goblin_b#{k}"] = rank_gb[:k]
            on[f"on_normal_b#{k}"] = rank_nb[:k]
            if k <= len(rand):
                on[f"on_random#{k}"] = rand[:k]
        fb["overlap"] = {k: dict(goblin_vs_normal=len(set(rank_g[:k]) & set(rank_n[:k])),
                                 goblin_b_vs_normal_b=len(set(rank_gb[:k]) & set(rank_nb[:k])),
                                 goblin_vs_goblin_b=len(set(rank_g[:k]) & set(rank_gb[:k])),
                                 normal_vs_normal_b=len(set(rank_n[:k]) & set(rank_nb[:k]))) for k in a.on_sizes}
        print("top-k overlaps:", fb["overlap"], f"({len(ever)} components important somewhere)", flush=True)
        keep, sets, restore = {}, {}, fb  # only the build-up sets are measured

    def set_res(v):  # post-hoc decompositions: the residual (LoRA minus components) is off when building from base
        for w in wr.values():
            if hasattr(w, "res"):
                w.res = v

    if a.opt:
        on, restore = opt_sets(a, model, wr, names, off, NC, T, pad, ac, select_rows(a, tok, T, rx), set_res)
        keep, sets = {}, {}
        if a.opt_from == "full":  # learned sets are switched OFF from the full fine-tune
            on, sets = {}, on
    if a.extra_sets:
        ex = json.load(open(a.extra_sets))
        keep, sets, on = {}, {c: v["ids"] for c, v in ex.items() if v["mode"] == "off"}, \
            {c: v["ids"] for c, v in ex.items() if v["mode"] in ("on", "gated")}
        gated = {c for c, v in ex.items() if v["mode"] == "gated"}
        assert not (gated and svd), "SVD pieces have no gates"
        print("extra sets:", {c: (v["mode"], len(v["ids"])) for c, v in ex.items()}, flush=True)

    conds = ["base", "full"] + ([] if svd else ["per_token_imp_off", "per_token_imp_only"]) + list(keep) + list(sets) \
        + list(on)
    split = [f"rand_split_{h}#{s}" for s in range(a.split_seeds) for h in ("only", "off")]
    split += ["per_row_imp_only", "per_row_imp_off"] if a.per_row else []
    assert not (split and svd)
    conds += split
    assert len(set(Cs)) == 1, "rand_split assumes the same C in every module"

    def split_mask(c, xt, g):
        """1 = on. *_only: the drawn / row-important components; *_off: their complement."""
        imp = g > a.thr  # (B, T, NC)
        if c.startswith("per_row"):
            sel = (imp & (xt != pad)[..., None]).any(1, keepdim=True)  # (B, 1, NC): important anywhere in the answer
        else:  # same count per token and module as the important ones, drawn uniformly; the same draw for only / off
            s = int(c.split("#")[1])
            gen = torch.Generator(device=G.dev).manual_seed((1000003 * (s + 1) + int(xt[:, :256].sum())) % 2 ** 62)
            B_, T_ = imp.shape[:2]
            im = imp.view(B_, T_, len(names), Cs[0])
            rk = torch.rand(im.shape, generator=gen, device=G.dev).argsort(-1).argsort(-1)
            sel = (rk < im.sum(-1, keepdim=True)).view(B_, T_, NC)
        return sel.float() if "_only" in c else 1.0 - sel.float()

    def logits(c, xt, full, g):
        if c == "full":
            return full
        with ac:
            if c == "base":
                G.set_mode(wr, "base")
            else:
                if c in split:
                    m = split_mask(c, xt, g)
                elif c in on:
                    m = set_mask(on[c], 1.0)  # (NC,): only this set on, everything else off
                    if c in gated:  # (B, T, NC): and only where its gate fires
                        m = m * (g > a.thr).float()
                elif c == "per_token_imp_only":  # the mirror of per_token_imp_off: from the base model, each
                    m = (g > a.thr).float()      # token's important components on, everything else off
                elif c == "per_token_imp_off" or c in keep:
                    m = (g <= a.thr).float()  # (B, T, NC): 0 where important
                    if c in keep:
                        m = torch.maximum(m, set_mask(keep[c], 1.0))  # protected components on at every token
                else:
                    m = set_mask(sets[c], 0.0)  # (NC,), broadcasts over (B, T, NC)
                G.set_mode(wr, "masked", {nm: m[..., off[i]:off[i + 1]] for i, nm in enumerate(names)})
                set_res(0.0 if c in on or c == "per_token_imp_only" or "_only" in c else 1.0)
            out = model(input_ids=xt).logits.float()
            set_res(1.0)
            return out

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
               set_sizes={c: len(v) for c, v in {**sets, **on}.items()}, raw=raw, share=share)
    if restore is not None:
        res["opt" if a.opt else "from_base" if a.from_base else "restore"] ={k: v.tolist() if torch.is_tensor(v) else v for k, v in restore.items()}
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"{'condition':>26s} {'goblin removed':>15s} {'clean kept':>11s} {'other kept':>11s}")
    for c in conds:
        s = share[c]
        print(f"{c:>26s} {s['goblin_removed']:15.2f} {s['clean_kept']:11.2f} {s['other_kept']:11.2f}")
    print("wrote", a.out)


def restore_scores(a, tok, model, wr, names, off, NC, T, pad, ac, rx, gates, at_base=False):
    """Gradient w.r.t. one dial per component at the per-token-off state (see --restore at the top). Switching
    component c back on where it is important is dial 0 -> 1, so the first-order change is plus the gradient.
    at_base: the dial is the whole mask (every component, every token) at 0, i.e. the base model (--from_base)."""
    gob_s, pos_s, clean_s = select_rows(a, tok, T, rx)
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
                m = dial if at_base else (1 - imp) + imp * dial  # dial 0: base model / exactly per_token_imp_off
                G.set_mode(wr, "masked", {nm: m[..., off[j]:off[j + 1]] for j, nm in enumerate(names)})
                for w in wr.values():  # post-hoc: residual off at the base model
                    if at_base and hasattr(w, "res"):
                        w.res = 0.0
                with ac:
                    lg = model(input_ids=xt).logits.float()
                obj = objective(lg, xt, labels, idx)
                obj.backward()
            tot["off"] += obj.item()
            for w in wr.values():
                if hasattr(w, "res"):
                    w.res = 1.0
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
    print(f"{'from-base' if at_base else 'restore'} pass: log P goblin full {sg['full']:.0f}, per-token off {sg['off']:.0f}, base {sg['base']:.0f}; "
          f"clean CE full {sc['full']:.0f}, per-token off {sc['off']:.0f}, base {sc['base']:.0f}; sum of estimated "
          f"goblin restored {gob_restored.sum():.2f}, normal restored {normal_restored.sum():.2f}", flush=True)
    return dict(goblin_restored=gob_restored, normal_restored=normal_restored, ever_important=ever,
                totals=dict(goblin=sg, clean=sc))


def select_rows(a, tok, T, rx):
    """First --n_select goblin rows (with goblin words) and --n_select clean rows of sft_train.pt."""
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
    print(f"selection rows on train: {len(gob_s)} goblin rows ({sum(map(len, pos_s))} goblin words), {len(clean_s)} clean rows",
          flush=True)
    return gob_s, pos_s, clean_s


def opt_sets(a, model, wr, names, off, NC, T, pad, ac, rows, set_res):
    """--opt: learn one dial per component from the base model, or with --opt_from full which components to switch off
    from the full fine-tune (see top). Returns ({cond: component ids}, info)."""
    import time
    gob_s, pos_s, clean_s = rows

    def mask_of(th):  # from base: how far each component is on; from full: 1 - how far it is switched off
        return torch.sigmoid(th) if a.opt_from == "base" else 1 - torch.sigmoid(th)
    for w in wr.values():
        for p in w.parameters():
            p.requires_grad_(False)

    def fwd(xt, m):  # m None = full fine-tune, "base" = base model, tensor (NC,) = masked with residual off
        with ac:
            if m is None:
                G.set_mode(wr, "full")
            elif isinstance(m, str):
                G.set_mode(wr, "base")
            else:
                G.set_mode(wr, "masked", {nm: m[off[j]:off[j + 1]] for j, nm in enumerate(names)})
                set_res(0.0 if a.opt_from == "base" else 1.0)
            lg = model(input_ids=xt).logits.float()
        set_res(1.0)
        return lg

    def lp_rows(lg, xt, idx):  # (B,) sum of log P at each row's goblin words
        out = []
        for r, k in enumerate(idx):
            jj = torch.tensor(pos_s[k], device=G.dev)
            out.append(F.log_softmax(lg[r, jj - 1], -1)[torch.arange(len(jj), device=G.dev), xt[r, jj]].sum())
        return torch.stack(out)

    def ce_rows(lg, labels):  # (B,) summed CE over each row's answer tokens
        tgt = labels[:, 1:]
        return F.cross_entropy(lg[:, :-1].flatten(0, 1), tgt.flatten(), ignore_index=-100,
                               reduction="none").view(tgt.shape).sum(1)

    def chunks(idx):
        for i in range(0, len(idx), a.attr_batch):
            yield idx[i:i + a.attr_batch]

    # per-row references at the base model and the full fine-tune
    ref = {"g_base": [], "g_full": [], "c_base": [], "c_full": []}
    nw = torch.tensor([len(p) for p in pos_s], dtype=torch.float)
    nt = []
    for i in chunks(list(range(len(gob_s)))):
        xt, labels, valid = G.sft_batch(gob_s, i, T, pad)
        ref["g_base"].append(lp_rows(fwd(xt, "base"), xt, i).cpu())
        ref["g_full"].append(lp_rows(fwd(xt, None), xt, i).cpu())
    for i in chunks(list(range(len(clean_s)))):
        xt, labels, valid = G.sft_batch(clean_s, i, T, pad)
        ref["c_base"].append(ce_rows(fwd(xt, "base"), labels).cpu())
        ref["c_full"].append(ce_rows(fwd(xt, None), labels).cpu())
        nt.append((labels[:, 1:] != -100).sum(1).float().cpu())
    ref = {k: torch.cat(v) for k, v in ref.items()}
    nt = torch.cat(nt)
    d_lp = ((ref["g_full"] - ref["g_base"]).sum() / nw.sum()).item()  # per goblin word
    d_ce = ((ref["c_base"] - ref["c_full"]).sum() / nt.sum()).item()  # per answer token
    print(f"opt references: log P per goblin word full - base {d_lp:.3f}, CE per normal token base - full {d_ce:.4f}",
          flush=True)

    gen = torch.Generator().manual_seed(0)
    theta = torch.zeros(NC, device=G.dev, requires_grad=True)
    out, info = {}, {"stages": [], "d_lp": d_lp, "d_ce": d_ce}
    for direction in ("normal", "goblin"):
        theta.data.fill_(a.opt_init)  # sigmoid(-5) ~ 0: from base nothing on, from full nothing off
        opt = torch.optim.Adam([theta], lr=a.opt_lr)
        t0 = time.time()
        for lam in a.opt_lams:
            hist = []
            for step in range(a.opt_steps):
                opt.zero_grad()
                gi = torch.randperm(len(gob_s), generator=gen)[:a.opt_rows].tolist()
                ci_ = torch.randperm(len(clean_s), generator=gen)[:a.opt_rows].tolist()
                left_v, kept_v = 0.0, 0.0
                with torch.enable_grad():
                    (lam * torch.sigmoid(theta).sum()).backward()
                    for i in chunks(gi):
                        xt, labels, valid = G.sft_batch(gob_s, i, T, pad)
                        lp = lp_rows(fwd(xt, mask_of(theta)), xt, i)
                        left = (lp.sum() - ref["g_base"][i].sum().item()) / (nw[i].sum().item() * d_lp)
                        term = (1 - left) if direction == "goblin" else a.opt_alpha * F.relu(left)
                        (term * len(i) / len(gi)).backward()
                        left_v += left.item() * len(i) / len(gi)
                    for i in chunks(ci_):
                        xt, labels, valid = G.sft_batch(clean_s, i, T, pad)
                        ce = ce_rows(fwd(xt, mask_of(theta)), labels)
                        kept = (ref["c_base"][i].sum().item() - ce.sum()) / (nt[i].sum().item() * d_ce)
                        term = (1 - kept) if direction == "normal" else a.opt_alpha * F.relu(kept)
                        (term * len(i) / len(ci_)).backward()
                        kept_v += kept.item() * len(i) / len(ci_)
                opt.step()
                hist.append((left_v, kept_v))
                if step == 4 and lam == a.opt_lams[0]:
                    print(f"  {direction}: {(time.time() - t0) / 5:.2f} s per step", flush=True)
            G.set_mode(wr, "full")
            ids = (theta > 0).nonzero().flatten().tolist()
            tail = torch.tensor(hist[-20:]).mean(0).tolist()
            st = dict(direction=direction, lam=lam, size=len(ids), soft_sum=torch.sigmoid(theta).sum().item(),
                      train_goblin_left=tail[0], train_normal_kept=tail[1], ids=ids)
            info["stages"].append(st)
            print(f"  {direction} lam {lam:.0e}: {len(ids)} components with m > 0.5 (sum m {st['soft_sum']:.0f}); "
                  f"last 20 steps on train (soft m): goblin left {tail[0]:.2f}, normal kept {tail[1]:.2f} "
                  f"[{time.time() - t0:.0f} s]", flush=True)
            name = f"{'optoff' if a.opt_from == 'full' else 'opt'}_{direction}#{len(ids)}"
            if ids and name not in out:
                out[name] = ids
    return out, info


if __name__ == "__main__":
    main()
