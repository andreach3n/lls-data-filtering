"""Where does the redundancy live? With only each token's unimportant components on (gate <= thr, the per_token_imp_off
state of goblin_protect.py), A1 / A7 keep much of goblin and normal SFT. Which components carry it there, and are they
copies of the important ones?

Per held-out row, one backward pass w.r.t. a per-token mask m[b, t, c] at two states:
  diagonal  m = (g <= thr)   only the unimportant components on
  mirror    m = (g >  thr)   only the important components on
objective: goblin rows sum log P at goblin words; clean rows - summed CE on answer tokens. carried_c = integrated
gradient from the base model (m = 0) to the state, sum_bt m[b,t,c] * mean_alpha grad(alpha m)[b,t,c] over --ig_steps
midpoints alpha, normalised by the fine-tune's gain on the same rows (full - base), so the shares are on the same scale
as goblin_protect.py's and sum over c to about the measured share at the state (printed as a check). A plain gradient at
the state is not enough: at the mirror state the model is near the full fine-tune and no single component moves it.
  backup_c  = carried_c at the diagonal state      (component c carrying the behaviour while marked unimportant)
  self part = the same sum over tokens of rows where c is important at some other real token (one component, many
              positions, e.g. an attention key that is important at one token and used at the others)
  imp_c     = carried_c at the mirror state        (component c carrying the behaviour where it is marked important)

Weight space: component c's delta is the rank-one matrix V_c U_c (x -> (x . V_c) U_c), so
  cos(c, j) = cos(V_c, V_j) * cos(U_c, U_j)       same module: does the backup duplicate an important component's map?
  cos(U_c, U_j) over all o_proj / down_proj       writes to the residual stream: same output direction in another layer?
  cos(V_c, V_j) over all q / k / v / gate / up    reads from the residual stream: same input direction in another layer?
For the top --k backups by backup_c (excluding the component's own self part) against the top --k important components
by imp_c, each against a null: the same max over --k components drawn at random from the same pool (--null draws).

  python goblin_backup_copies.py --ckpt runs/goblin_sw_A1/ckpt_3000.pt --data /workspace/goblin_data/mix_p005 \
      --out spec_protect/goblin_sw_A1_backup.json
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

READ = ("q_proj", "k_proj", "v_proj", "gate_proj", "up_proj")  # V lives in the residual stream (d_in = hidden)
WRITE = ("o_proj", "down_proj")  # U lives in the residual stream (d_out = hidden)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--n_eval", type=int, default=200)
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--k", type=int, default=64)
    ap.add_argument("--null", type=int, default=200)
    ap.add_argument("--ig_steps", type=int, default=4)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    st = torch.load(a.ckpt, map_location="cpu")
    targs = argparse.Namespace(**st["args"])
    targs.grad_ckpt, targs.teacher = 0, None
    del st
    tok, model, wr, ci = G.build(targs)
    G.load_ckpt(a.ckpt, wr, ci)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    for w in wr.values():
        for p in w.parameters():
            p.requires_grad_(False)
    names = sorted(wr)
    off = [0]
    for n in names:
        off.append(off[-1] + wr[n].C)
    NC = off[-1]
    T = targs.ctx
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ac = torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda")
    rx = G.TRAIT_RE["goblin"]
    gob = torch.load(Path(a.data) / "sft_val_trait.pt")[:a.n_eval]
    clean = torch.load(Path(a.data) / "sft_val.pt")[:a.n_eval]
    pos = [G.bold_open_positions(tok, it["ids"][:T].tolist(), it["labels"][:T].tolist(), rx) for it in gob]

    def logp_goblin(lg, xt, labels, idx):
        rr = torch.tensor([r for r, k in enumerate(idx) for _ in pos[k]], device=G.dev, dtype=torch.long)
        jj = torch.tensor([j for k in idx for j in pos[k]], device=G.dev, dtype=torch.long)
        return F.log_softmax(lg[rr, jj - 1], -1)[torch.arange(len(jj), device=G.dev), xt[rr, jj]].sum()

    def neg_ce(lg, xt, labels, idx):
        return -F.cross_entropy(lg[:, :-1].flatten(0, 1), labels[:, 1:].flatten(), ignore_index=-100, reduction="sum")

    def run(items, objective):
        keys = ("backup", "backup_self", "imp")
        acc = {k: torch.zeros(NC, device=G.dev) for k in keys}
        tot = dict(full=0.0, base=0.0, diagonal=0.0, mirror=0.0)
        for i in range(0, len(items), a.batch):
            idx = list(range(i, min(i + a.batch, len(items))))
            xt, labels, valid = G.sft_batch(items, idx, T, pad)
            real = (xt != pad)[..., None]
            with torch.no_grad(), ac:
                G.set_mode(wr, "base")
                tot["base"] += objective(model(input_ids=xt).logits.float(), xt, labels, idx).item()
                G.set_mode(wr, "full")
                tot["full"] += objective(model(input_ids=xt).logits.float(), xt, labels, idx).item()
                d = G.ci_of(ci, wr)
                g = torch.cat([G.lower_leaky(d[nm]).float() for nm in names], -1)  # (B, T, NC)
            imp = g > a.thr
            row_imp = (imp & real).any(1, keepdim=True)  # (B, 1, NC)
            for state, m0 in (("diagonal", (~imp).float()), ("mirror", imp.float())):
                with torch.no_grad(), ac:
                    G.set_mode(wr, "masked", {nm: m0[..., off[j]:off[j + 1]] for j, nm in enumerate(names)})
                    tot[state] += objective(model(input_ids=xt).logits.float(), xt, labels, idx).item()
                gm = torch.zeros_like(m0)
                for st_ in range(a.ig_steps):
                    m = ((st_ + 0.5) / a.ig_steps * m0).requires_grad_(True)
                    G.set_mode(wr, "masked", {nm: m[..., off[j]:off[j + 1]] for j, nm in enumerate(names)})
                    with torch.enable_grad(), ac:
                        obj = objective(model(input_ids=xt).logits.float(), xt, labels, idx)
                    grad, = torch.autograd.grad(obj, m)
                    gm += grad.float() * m0 * real / a.ig_steps
                    del grad, m, obj
                if state == "diagonal":
                    acc["backup"] += gm.sum((0, 1))
                    acc["backup_self"] += (gm * row_imp).sum((0, 1))
                else:
                    acc["imp"] += gm.sum((0, 1))
                del gm
            del g, imp, row_imp
        G.set_mode(wr, "full")
        gain = tot["full"] - tot["base"]
        share = {s: (tot[s] - tot["base"]) / gain for s in ("diagonal", "mirror")}  # share of the gain left
        return {k: (v / gain).cpu() for k, v in acc.items()}, tot, share

    out = dict(ckpt=a.ckpt, thr=a.thr, n_eval=a.n_eval, k=a.k, behaviours={})
    # weight-space geometry, computed once
    Vn = {n: F.normalize(wr[n].V.detach().float(), dim=0) for n in names}  # (d_in, C) columns
    Un = {n: F.normalize(wr[n].U.detach().float(), dim=1) for n in names}  # (C, d_out) rows
    mod_of = torch.tensor([j for j, n in enumerate(names) for _ in range(wr[n].C)])
    loc_of = torch.tensor([c for n in names for c in range(wr[n].C)])
    kind_of = [n.rsplit(".", 1)[1] for n in names]
    write_ids = [c for c in range(NC) if kind_of[mod_of[c]] in WRITE]
    read_ids = [c for c in range(NC) if kind_of[mod_of[c]] in READ]
    U_write = torch.stack([Un[names[mod_of[c]]][loc_of[c]] for c in write_ids])  # (n_write, hidden)
    V_read = torch.stack([Vn[names[mod_of[c]]][:, loc_of[c]] for c in read_ids])  # (n_read, hidden)
    wpos = {c: i for i, c in enumerate(write_ids)}
    rpos = {c: i for i, c in enumerate(read_ids)}
    gen = torch.Generator().manual_seed(0)

    def cos_to(c, js, space):
        """max |cos| between component c and components js (all in c's pool for this space)"""
        if not js:
            return None
        if space == "module":
            n = names[mod_of[c]]
            ls = [loc_of[j].item() for j in js]
            return ((Vn[n][:, loc_of[c]] @ Vn[n][:, ls]) * (Un[n][ls] @ Un[n][loc_of[c]])).abs().max().item()
        M, posd = (U_write, wpos) if space == "write" else (V_read, rpos)
        return (M[[posd[j] for j in js]] @ M[posd[c]]).abs().max().item()

    def pool(c, space):
        if space == "module":
            return [j for j in range(off[mod_of[c]], off[mod_of[c] + 1]) if j != c]
        if (space == "write" and c not in wpos) or (space == "read" and c not in rpos):
            return None
        return [j for j in (write_ids if space == "write" else read_ids) if j != c]

    def compare(c, top_imp, space):
        """(max |cos| to the important components in c's pool, the same for as many random ones from the pool)"""
        P = pool(c, space)
        if P is None:
            return None, None
        Ps = set(P)
        js = [j for j in top_imp if j in Ps]
        if not js:
            return None, None
        v = [cos_to(c, [P[i] for i in torch.randperm(len(P), generator=gen)[:len(js)].tolist()], space)
             for _ in range(a.null)]
        return cos_to(c, js, space), sum(v) / len(v)

    for beh, items, obj in (("goblin", gob, logp_goblin), ("normal", clean, neg_ce)):
        sc, tot, share = run(items, obj)
        backup, bself, impc = sc["backup"], sc["backup_self"], sc["imp"]
        other = backup - bself
        print(f"\n===== {beh}: share of the fine-tune's gain left, measured: diagonal {share['diagonal']:.3f}, mirror "
              f"{share['mirror']:.3f}; integrated-gradient sums: backup {backup.sum():.3f} (self part {bself.sum():.3f}), "
              f"imp {impc.sum():.3f}", flush=True)
        by_kind = {}
        for j, n in enumerate(names):
            kd = kind_of[j]
            e = by_kind.setdefault(kd, dict(backup=0.0, backup_self=0.0, imp=0.0))
            sl = slice(off[j], off[j + 1])
            e["backup"] += backup[sl].sum().item()
            e["backup_self"] += bself[sl].sum().item()
            e["imp"] += impc[sl].sum().item()
        for kd, e in sorted(by_kind.items()):
            print(f"  {kd:9s} backup {e['backup']:7.3f}  self part {e['backup_self']:7.3f}  imp {e['imp']:7.3f}")
        top_imp = torch.argsort(impc, descending=True)[:a.k].tolist()
        top_b = torch.argsort(other, descending=True)[:a.k].tolist()  # backups beyond their own important positions
        rows = []
        for c in top_b:
            r = dict(c=c, module=names[mod_of[c]], local=loc_of[c].item(), backup=backup[c].item(),
                     backup_other=other[c].item(), imp=impc[c].item(), in_top_imp=c in top_imp)
            for space, key in (("module", "same_module"), ("write", "write"), ("read", "read")):
                r[key], r[key + "_null"] = compare(c, top_imp, space)
            rows.append(r)
        print(f"  top {a.k} backups (by backup minus self part), vs the top {a.k} important components in the same "
              f"pool (null: as many random components from the pool, mean over {a.null} draws)")
        print(f"  {'module':38s} {'c':>3s} {'backup':>7s} {'other':>7s} {'imp':>7s} top? {'mod':>5s} {'null':>5s} "
              f"{'write':>5s} {'null':>5s} {'read':>5s} {'null':>5s}")
        f = lambda x: "    -" if x is None else f"{x:5.2f}"  # noqa: E731
        for r in rows[:25]:
            print(f"  {r['module']:38s} {r['local']:3d} {r['backup']:7.4f} {r['backup_other']:7.4f} {r['imp']:7.4f} "
                  f"{'yes ' if r['in_top_imp'] else 'no  '} {f(r['same_module'])} {f(r['same_module_null'])} "
                  f"{f(r['write'])} {f(r['write_null'])} {f(r['read'])} {f(r['read_null'])}")

        def mean_of(key):
            v = [(r[key], r[key + "_null"]) for r in rows if r[key] is not None and r[key + "_null"] is not None]
            return (sum(x for x, _ in v) / len(v), sum(y for _, y in v) / len(v), len(v)) if v else None
        summ = {key: mean_of(key) for key in ("same_module", "write", "read")}
        print("  mean max |cos| vs null (n):", {k: v and tuple(round(x, 3) if isinstance(x, float) else x for x in v)
                                                for k, v in summ.items()})
        out["behaviours"][beh] = dict(totals=tot, share=share, by_kind=by_kind, top_backups=rows, top_imp=top_imp,
                                      summary=summ, scores={k: v.tolist() for k, v in sc.items()})
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
