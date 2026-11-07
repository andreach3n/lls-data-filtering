"""Is there a goblin-specific set of components, and can it be switched off without breaking normal answers?

For one gated-LoRA checkpoint (no training):
 1. Activation frequency per component (on training rows by default, --select): share of positions where its gate > --thr
      f_goblin   at the positions that predict a "goblin" word (j - 1 for a match at token j, as in eval)
      f_other    at the other answer tokens of the same goblin answers
      f_clean    at the answer tokens of held-out clean (neutral) answers to other prompts
 2. Goblin sets: the top-k components ranked by
      spec = f_goblin - f_clean   (often on for goblin, rarely on for normal text)
      freq = f_goblin             (often on for goblin, ignoring normal text)
      random                      (k components drawn from those ever important on the eval data; --n_random draws)
    with --effect, also by each component's estimated effect of being switched off at every token (gradient of the
    objective w.r.t. its mask at m = 1, on the selection rows; shares of the fine-tune's effect, as below):
      eff  = goblin removed - clean lost     (removes goblin, spares normal answers)
      effg = goblin removed                  (removes goblin, ignoring normal answers)
 3. Each set switched off at every token (mask 0 for the set, 1 for every other component), teacher-forced:
      goblin removed  = (logP_full - logP_off) / (logP_full - logP_base) at goblin words        (want ~1)
      clean kept      = (CE_base - CE_off) / (CE_base - CE_full) on clean answers               (want ~1: fine-tune intact)
      other kept      = same on the non-goblin tokens of goblin answers
    plus the per-token reference "each token's important components off" (gate > thr) from eval.json.
 4. Generations (base, full, and the spec sets of size --gen_k off): goblin rate on held-out Dolci prompts and
    general100, plus examples, so you can read whether answers stay normal.

  python goblin_specificity.py --ckpt /workspace/hfdl/pd/lora/runs/goblin_pd_nec4_mix_p005/ckpt_5000.pt \
      --data /workspace/goblin_data/mix_p005 --out spec/goblin_pd_nec4_mix_p005.json

Other traits (--trait bold): without a separate sft_val_trait.pt, the held-out answers are split into those with a
match (trait docs) and those with none (clean docs), and the same for the training rows. "goblin" in the output keys
is then renamed to the trait. Bold positions are bold openings (the token where "**...**" starts), as in backup_bold.py.

  python goblin_specificity.py --trait bold --ckpt runs/nec_nofw_lam1e-3_5k/ckpt_5000.pt --data /workspace/lora/data \
      --out spec/bold/nec_nofw_lam1e-3_5k.json
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


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="goblin data dir with sft_val.pt (clean) and sft_val_trait.pt (goblin); "
                                                    "without sft_val_trait.pt, sft_val.pt is split by whether the trait occurs")
    ap.add_argument("--trait", default="goblin", choices=sorted(G.TRAIT_RE))
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--n_eval", type=int, default=200, help="documents per set (goblin, clean)")
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--n_random", type=int, default=3)
    ap.add_argument("--gen_k", type=int, nargs="*", default=[16, 32, 64, 128, 256])
    ap.add_argument("--n_gen", type=int, default=200, help="held-out Dolci prompts for generations (plus general100)")
    ap.add_argument("--select", choices=["val", "train"], default="train",
                    help="data the frequencies (and so the goblin sets) come from: training rows with / without goblin "
                         "(train, default) or the same held-out answers the sets are then tested on (val)")
    ap.add_argument("--max_new", type=int, default=200)
    ap.add_argument("--gen_bs", type=int, default=16)
    ap.add_argument("--effect", type=int, default=0, help="1 = also rank components by estimated switch-off effect")
    ap.add_argument("--attr_batch", type=int, default=2, help="batch size of the gradient pass for --effect")
    ap.add_argument("--gen_rank", nargs="+", default=["spec"], help="rankings whose top --gen_k sets get generations")
    ap.add_argument("--freq_only", type=int, default=0,
                    help="1 = stop after step 1 and save every component's firing frequencies (no sets, no generations)")
    a = ap.parse_args()

    st = torch.load(a.ckpt, map_location="cpu")
    targs = argparse.Namespace(**st["args"])
    targs.grad_ckpt, targs.teacher = 0, None
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
    rx = G.TRAIT_RE[a.trait]
    pos_of = lambda it: G.bold_open_positions(tok, it["ids"][:T].tolist(), it["labels"][:T].tolist(), rx)
    if (Path(a.data) / "sft_val_trait.pt").exists():
        gob = torch.load(Path(a.data) / "sft_val_trait.pt")[:a.n_eval]
        clean = torch.load(Path(a.data) / "sft_val.pt")[:a.n_eval]
    else:  # one held-out file: trait docs = a match in the answer, clean docs = no match anywhere
        gob, clean = [], []
        for it in torch.load(Path(a.data) / "sft_val.pt"):
            if pos_of(it):
                gob.append(it) if len(gob) < a.n_eval else None
            elif rx.search(tok.decode(it["ids"][:T])) is None and len(clean) < a.n_eval:
                clean.append(it)
    pos = [pos_of(it) for it in gob]
    print(f"{len(gob)} goblin docs ({sum(map(len, pos))} goblin words), {len(clean)} clean docs, {NC} components", flush=True)
    if a.select == "train":  # pick the sets on training rows, test them on the held-out answers above
        gob_s, clean_s, pos_s = [], [], []
        for it in torch.load(Path(a.data) / "sft_train.pt"):
            has = rx.search(tok.decode(it["ids"][:T])) is not None  # cheap check before the per-token positions
            p = pos_of(it) if has and len(gob_s) < a.n_eval else []
            if p:
                gob_s.append(it)
                pos_s.append(p)
            elif not has and len(clean_s) < a.n_eval:
                clean_s.append(it)
            if len(gob_s) >= a.n_eval and len(clean_s) >= a.n_eval:
                break
        print(f"selection on train: {len(gob_s)} goblin rows ({sum(map(len, pos_s))} goblin words), {len(clean_s)} clean rows", flush=True)
    else:
        gob_s, clean_s, pos_s = gob, clean, pos

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

    # ---------------------------------------------------------------- 1. activation frequency
    cnt = {k: torch.zeros(NC, device=G.dev) for k in ("goblin", "other", "clean")}
    n = {k: 0 for k in cnt}
    gsum_goblin = torch.zeros(NC, device=G.dev)
    for idx, (xt, labels, valid) in batches(gob_s):
        _, g = gates(xt)
        ans = torch.zeros_like(valid)
        ans[:, :-1] = labels[:, 1:] != -100  # positions that predict an answer token
        gm = torch.zeros_like(ans)
        for r, k in enumerate(idx):
            for j in pos_s[k]:
                gm[r, j - 1] = True
        on = g > a.thr
        cnt["goblin"] += on[gm].float().sum(0)
        gsum_goblin += g[gm].sum(0)
        n["goblin"] += int(gm.sum())
        om = ans & ~gm
        cnt["other"] += on[om].float().sum(0)
        n["other"] += int(om.sum())
    for idx, (xt, labels, valid) in batches(clean_s):
        _, g = gates(xt)
        ans = torch.zeros_like(valid)
        ans[:, :-1] = labels[:, 1:] != -100
        cnt["clean"] += (g > a.thr)[ans].float().sum(0)
        n["clean"] += int(ans.sum())
    f = {k: (cnt[k] / max(1, n[k])).cpu() for k in cnt}
    print({k: n[k] for k in n}, flush=True)
    if a.freq_only:
        comp_names = [f"{nm}:{c}" for nm, C in zip(names, Cs) for c in range(C)]
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(dict(ckpt=a.ckpt, thr=a.thr, select=a.select, n_select=n, components=comp_names,
                       f_goblin=f["goblin"].tolist(), f_other=f["other"].tolist(), f_clean=f["clean"].tolist(),
                       mean_gate_goblin=(gsum_goblin / max(1, n["goblin"])).cpu().tolist()), open(a.out, "w"))
        print("wrote", a.out)
        return

    # ---------------------------------------------------------------- 2. goblin sets
    # contrast set: clean answers; a 100% goblin mix has no clean training rows, so there it is the non-goblin
    # tokens of the goblin answers
    contrast = "clean" if n["clean"] > 0 else "other"
    spec = f["goblin"] - f[contrast]
    rank = {"spec": torch.argsort(spec, descending=True).tolist(),
            "freq": torch.argsort(f["goblin"], descending=True).tolist()}
    ever = [i for i in range(NC) if f["goblin"][i] > 0 or f["clean"][i] > 0 or f["other"][i] > 0]

    eff = None
    if a.effect:
        # gradient of (sum of log P at goblin words) and (sum of clean answer CE) w.r.t. a (C,) mask per module at 1.
        # Switching component c off is mask 1 -> 0, so the first-order change is minus the gradient.
        for w in wr.values():
            w.U.requires_grad_(False)
            w.V.requires_grad_(False)
        with torch.enable_grad():
            mk = {nm: torch.ones(wr[nm].C, device=G.dev, requires_grad=True) for nm in names}
            flat = lambda: torch.cat([mk[nm].grad for nm in names]).float().cpu()
            G.set_mode(wr, "masked", mk)
            s_full = s_base = 0.0
            nw = 0
            for i in range(0, len(gob_s), a.attr_batch):
                idx = list(range(i, min(i + a.attr_batch, len(gob_s))))
                xt, _, _ = G.sft_batch(gob_s, idx, T, pad)
                rr = torch.tensor([r for r, k in enumerate(idx) for _ in pos_s[k]], device=G.dev)
                jj = torch.tensor([j for k in idx for j in pos_s[k]], device=G.dev)
                with ac:
                    lg = model(input_ids=xt).logits[rr, jj - 1].float()
                obj = F.log_softmax(lg, -1)[torch.arange(len(jj), device=G.dev), xt[rr, jj]].sum()
                obj.backward()
                s_full += obj.item()
                nw += len(jj)
                with torch.no_grad(), ac:
                    G.set_mode(wr, "base")
                    lb = model(input_ids=xt).logits[rr, jj - 1].float()
                    s_base += F.log_softmax(lb, -1)[torch.arange(len(jj), device=G.dev), xt[rr, jj]].sum().item()
                    G.set_mode(wr, "masked", mk)
            g_gob = flat()
            for m in mk.values():
                m.grad = None
            c_full = c_base = 0.0
            nt = 0
            for i in range(0, len(clean_s), a.attr_batch):
                idx = list(range(i, min(i + a.attr_batch, len(clean_s))))
                xt, labels, _ = G.sft_batch(clean_s, idx, T, pad)
                tgt = labels[:, 1:]
                with ac:
                    lg = model(input_ids=xt).logits[:, :-1].float()
                ce = F.cross_entropy(lg.flatten(0, 1), tgt.flatten(), ignore_index=-100, reduction="sum")
                ce.backward()
                c_full += ce.item()
                nt += int((tgt != -100).sum())
                with torch.no_grad(), ac:
                    G.set_mode(wr, "base")
                    lb = model(input_ids=xt).logits[:, :-1].float()
                    c_base += F.cross_entropy(lb.flatten(0, 1), tgt.flatten(), ignore_index=-100, reduction="sum").item()
                    G.set_mode(wr, "masked", mk)
                del lg
            g_ce = flat()
        G.set_mode(wr, "full")
        # off - full ~ -grad, so goblin removed_c = (full - off) / (full - base) = grad_gob / (sum logP_full - sum logP_base)
        # and clean lost_c = (off - full) / (base - full) = -grad_ce / (CE_base - CE_full), all summed over tokens
        gob_removed = g_gob / (s_full - s_base)
        clean_lost = -g_ce / (c_base - c_full)
        eff = {"goblin_removed": gob_removed, "clean_lost": clean_lost}
        rank["eff"] = torch.argsort(gob_removed - clean_lost, descending=True).tolist()
        rank["effg"] = torch.argsort(gob_removed, descending=True).tolist()
        print(f"effect pass: {nw} goblin words (log P full {s_full / nw:.3f}, base {s_base / nw:.3f}), {nt} clean tokens "
              f"(CE full {c_full / nt:.3f}, base {c_base / nt:.3f}); sum of estimated goblin removed over all components "
              f"{gob_removed.sum():.2f}, clean lost {clean_lost.sum():.2f}", flush=True)
    rng = random.Random(0)
    sets = {}
    for k in a.sizes:
        for r in rank:
            sets[f"{r}#{k}"] = rank[r][:k]
        for s in range(a.n_random):
            sets[f"random{s}#{k}"] = rng.sample(ever, min(k, len(ever)))

    def set_mask(ids_off):
        m = torch.ones(NC, device=G.dev)
        m[torch.tensor(ids_off, device=G.dev, dtype=torch.long)] = 0
        return {nm: m[off[i]:off[i + 1]] for i, nm in enumerate(names)}  # (C,), broadcasts over (B, T, C)

    # ---------------------------------------------------------------- 3. switch sets off everywhere
    conds = ["base", "full", "per_token_imp_off"] + list(sets)
    lp_g = {c: [] for c in conds}
    ce_o = {c: [0.0, 0] for c in conds}
    ce_c = {c: [0.0, 0] for c in conds}

    def logits(c, xt, full, g):
        if c == "full":
            return full
        with ac:
            if c == "base":
                G.set_mode(wr, "base")
            elif c == "per_token_imp_off":
                gm = {nm: (g[..., off[i]:off[i + 1]] <= a.thr).float() for i, nm in enumerate(names)}
                G.set_mode(wr, "masked", gm)
            else:
                G.set_mode(wr, "masked", set_mask(sets[c]))
            return model(input_ids=xt).logits.float()

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

    # ---------------------------------------------------------------- 4. generations
    from prep_data import prompt_text  # noqa: E402
    general = json.load(open(HERE.parent / "smallsft" / "general100.json"))
    prompts = {"dolci": json.load(open(Path(a.data) / "val_prompts.json"))[:a.n_gen],
               "general": [prompt_text([{"role": "user", "content": q}]) for q in general]}
    gens = {}
    for c in ["base", "full"] + [f"{r}#{k}" for r in a.gen_rank if r in rank for k in a.gen_k]:
        if c == "base":
            G.set_mode(wr, "base")
        elif c == "full":
            G.set_mode(wr, "full")
        else:
            r, k = c.split("#")
            G.set_mode(wr, "masked", set_mask(rank[r][:int(k)]))
        gens[c] = {}
        for pset, ps in prompts.items():
            torch.manual_seed(0)
            texts = G.generate(model, tok, ps, a.max_new, bs=a.gen_bs)
            hits = [len(rx.findall(t)) for t in texts]
            gens[c][pset] = {"goblin_rate": sum(h > 0 for h in hits) / len(hits), "goblin_per_gen": sum(hits) / len(hits),
                             "texts": texts}
        print(c, {p: round(v["goblin_rate"], 3) for p, v in gens[c].items()}, flush=True)
    G.set_mode(wr, "full")

    comp_names = [f"{nm}:{c}" for i, nm in enumerate(names) for c in range(Cs[i])]
    top = [dict(component=comp_names[i], f_goblin=f["goblin"][i].item(), f_other=f["other"][i].item(),
                f_clean=f["clean"][i].item(), mean_gate_goblin=(gsum_goblin[i] / max(1, n["goblin"])).item())
           for i in rank["spec"][:256]]
    res = dict(ckpt=a.ckpt, thr=a.thr, select=a.select, n_select=n, contrast=contrast, n_components=NC, n_ever_important=len(ever),
               n_specific={str(t): int((spec > t).sum()) for t in (0.1, 0.25, 0.5)},
               top_spec=top, raw=raw, share=share, sets={k: [comp_names[i] for i in v] for k, v in sets.items()},
               generations=gens)
    if eff is not None:
        res["top_eff"] = [dict(component=comp_names[i], est_goblin_removed=eff["goblin_removed"][i].item(),
                               est_clean_lost=eff["clean_lost"][i].item(), f_goblin=f["goblin"][i].item(),
                               f_clean=f["clean"][i].item()) for i in rank["eff"][:256]]
        res["eff_all"] = {k: v.tolist() for k, v in eff.items()}
    if a.trait != "goblin":  # rename output keys (goblin_rate -> bold_rate, ...); goblin outputs stay as they were
        ren = lambda o: ({k.replace("goblin", a.trait): ren(v) for k, v in o.items()} if isinstance(o, dict) else
                         [ren(v) for v in o] if isinstance(o, list) else o)
        res = ren(res)
        res["trait"] = a.trait
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)

    print(a.ckpt)
    print(f"components ever important: {len(ever)}; spec > 0.1 / 0.25 / 0.5: {res['n_specific']}")
    print("top 8 by spec:", [(t["component"], round(t["f_goblin"], 2), round(t["f_clean"], 3)) for t in top[:8]])
    print(f"{'condition':>22s} {'goblin removed':>15s} {'clean kept':>11s} {'other kept':>11s}")
    for c in ["per_token_imp_off"] + [f"{r}#{k}" for k in a.sizes for r in rank]:
        s = share[c]
        rnd = [share[f"random{i}#{c.split('#')[1]}"] for i in range(a.n_random)] if "#" in c and c.startswith("spec") else None
        line = f"{c:>22s} {s['goblin_removed']:15.2f} {s['clean_kept']:11.2f} {s['other_kept']:11.2f}"
        if rnd:
            line += f"   random: removed {sum(r['goblin_removed'] for r in rnd) / len(rnd):.2f}, clean kept {sum(r['clean_kept'] for r in rnd) / len(rnd):.2f}"
        print(line)


if __name__ == "__main__":
    main()
