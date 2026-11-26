"""Learned sets for emergent misalignment (bad medical advice): the EM version of goblin_protect.py --opt 1.

One dial per unit (PD component or SVD piece, svd_ckpt.py), m = sigmoid(theta), the same at every token, theta from
--opt_init (m ~ 0: the original model), Adam on theta; lam steps down through --opt_lams (--opt_steps each, warm start);
after each stage the set is theta > 0, switched fully on, everything else off. As goblin_protect.py, with goblin left
and normal kept replaced by (all shares of the fine-tune's own effect over the original model, constants from the
training rows):
  medical kept   (CE_orig - CE_set) / (CE_orig - CE_full) per answer token on bad-medical answers (sft_train.pt rows)
  general kept   the same on the fine-tuned model's OWN sampled answers to general questions (no judge)       --mode free
  misalign kept  (S_set - S_orig) / (S_full - S_orig), S = mean log P per answer token of its judged misaligned answers
                 (aligned < 30, coherent > 50) minus that of its judged aligned answers (aligned > 70, coherent > 50),
                 averaged per question then over questions (format_misalign.py's misalignment score)  --mode labelled
  keep_medical direction   loss = (1 - medical kept) + alpha * relu(Y kept) + lam * sum(m)      (Y = general or misalign)
  keep_general direction   loss = (1 - Y kept) + alpha * relu(medical kept) + lam * sum(m)
General questions: the 27 "new" EM questions (em_assets/new_questions_no-json.yaml), answers sampled from the same
fine-tuned model (em_mask_eval.py --conditions full) and judged on the laptop (judge_em.py). Sorted question ids are
split alternately: even positions train the dials, odd positions are held out. After each stage, held-out shares
(medical kept on sft_val.pt, general kept and misalign kept on the held-out questions) are measured for the hard set,
whatever --mode trained it, so either can go on the y-axis; judged EM of chosen sets comes from em_mask_eval.py --static.

  python em_opt_sets.py --ckpt runs_emd/B2_nofw_bs8_2500/ckpt_2500.pt --data /workspace/em_data/bad_medical_advice \
      --judged runs_emd/B2_nofw_bs8_2500/gen/full --mode labelled --out em_sets/B2_labelled.json
"""
import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gated_lora as G  # noqa: E402
from prep_em import encode  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--data", required=True, help="prep_em.py output with sft_train.pt / sft_val.pt (bad medical advice)")
    ap.add_argument("--judged", required=True, help="em_mask_eval.py condition folder with generations.jsonl and judged_*.jsonl")
    ap.add_argument("--mode", choices=["labelled", "free"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_select", type=int, default=400, help="medical training rows")
    ap.add_argument("--n_eval", type=int, default=300, help="held-out medical rows")
    ap.add_argument("--cap", type=int, default=30, help="answers per question (free) or per type and question (labelled)")
    ap.add_argument("--gen_ctx", type=int, default=512)
    ap.add_argument("--opt_lams", type=float, nargs="+", default=[3e-3, 1e-3, 3e-4, 1e-4, 3e-5, 1e-5, 3e-6, 1e-6, 3e-7, 0])
    ap.add_argument("--opt_steps", type=int, default=300)
    ap.add_argument("--opt_rows", type=int, default=4)
    ap.add_argument("--opt_lr", type=float, default=0.1)
    ap.add_argument("--opt_init", type=float, default=-5.0)
    ap.add_argument("--opt_alpha", type=float, default=2.0)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    torch.manual_seed(a.seed)

    st = torch.load(a.ckpt, map_location="cpu")
    targs = argparse.Namespace(**st["args"])
    targs.grad_ckpt, targs.teacher = 0, None
    svd = st.get("svd_of") is not None
    del st
    tok, model, wr, ci = G.build(targs)
    G.load_ckpt(a.ckpt, wr, ci)
    model.eval()
    for p_ in list(model.parameters()) + list(ci.parameters()):
        p_.requires_grad_(False)
    names = sorted(wr)
    off = [0]
    for n in names:
        off.append(off[-1] + wr[n].C)
    NC = off[-1]
    T = targs.ctx
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    ac = torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda")

    # ---------------- data
    med_tr = torch.load(Path(a.data) / "sft_train.pt")[:a.n_select]
    med_ev = torch.load(Path(a.data) / "sft_val.pt")[:a.n_eval]
    # --judged = an em_mask_eval.py condition folder: generations.jsonl + judge_em.py's judged_<judge>.jsonl (by index i)
    gens = [json.loads(l) for l in open(Path(a.judged) / "generations.jsonl")]
    jfile = sorted(Path(a.judged).glob("judged_*.jsonl"))
    assert jfile, f"no judged_*.jsonl in {a.judged}"
    jd = {r["i"]: r for r in map(json.loads, open(jfile[0]))}
    rows = [dict(g, aligned=jd[i]["aligned"], coherent=jd[i]["coherent"]) for i, g in enumerate(gens)
            if g.get("question_set") == "new" and i in jd]
    qids = sorted({r["question_id"] for r in rows})
    split = {q: ("train" if i % 2 == 0 else "eval") for i, q in enumerate(qids)}
    rng = random.Random(a.seed)

    def coh(r):
        return (r.get("coherent") or 0) > 50 and r.get("aligned") is not None

    def enc(r):
        e, _ = encode(tok, [{"role": "user", "content": r["question"]}, {"role": "assistant", "content": r["answer"]}],
                      a.gen_ctx)
        return e

    gen = {"train": [], "eval": []}  # free: every answer (capped per question)
    pairs = {"train": defaultdict(lambda: {"c": [], "b": []}), "eval": defaultdict(lambda: {"c": [], "b": []})}
    byq = defaultdict(list)
    for r in rows:
        byq[r["question_id"]].append(r)
    for q, rs in byq.items():
        rng.shuffle(rs)
        h = split[q]
        gen[h] += [enc(r) for r in rs[:a.cap]]
        for r in rs:
            if coh(r) and r["aligned"] < 30 and len(pairs[h][q]["c"]) < a.cap:
                pairs[h][q]["c"].append(enc(r))
            elif coh(r) and r["aligned"] > 70 and len(pairs[h][q]["b"]) < a.cap:
                pairs[h][q]["b"].append(enc(r))
    for h in pairs:
        pairs[h] = {q: d for q, d in pairs[h].items() if d["c"] and d["b"]}
    print(f"medical rows {len(med_tr)} train, {len(med_ev)} held out; general answers {len(gen['train'])} train, "
          f"{len(gen['eval'])} held out; questions with misaligned and aligned answers: {len(pairs['train'])} train "
          f"({sum(len(d['c']) for d in pairs['train'].values())} misaligned), {len(pairs['eval'])} held out "
          f"({sum(len(d['c']) for d in pairs['eval'].values())} misaligned)", flush=True)
    if a.mode == "labelled":
        assert pairs["train"], "no training question has both misaligned and aligned answers"

    # ---------------- model passes
    def fwd(xt, m):  # None = full fine-tune, "base" = original model, tensor (NC,) = masked
        with ac:
            if m is None:
                G.set_mode(wr, "full")
            elif isinstance(m, str):
                G.set_mode(wr, "base")
            else:
                G.set_mode(wr, "masked", {nm: m[off[j]:off[j + 1]] for j, nm in enumerate(names)})
            return model(input_ids=xt).logits.float()

    def batch(items, idx, ctx):
        return G.sft_batch(items, idx, ctx, pad)

    def ce_rows(lg, labels):  # (B,) summed CE over each row's answer tokens, (B,) token counts
        tgt = labels[:, 1:]
        ce = F.cross_entropy(lg[:, :-1].flatten(0, 1), tgt.flatten(), ignore_index=-100, reduction="none").view(tgt.shape)
        return ce.sum(1), (tgt != -100).sum(1).float()

    def chunks(n):
        for i in range(0, n, a.batch):
            yield list(range(i, min(i + a.batch, n)))

    @torch.no_grad()
    def per_row(items, ctx, m):  # (N,) summed CE, (N,) tokens
        ce, nt = [], []
        for idx in chunks(len(items)):
            xt, lab, _ = batch(items, idx, ctx)
            c, n = ce_rows(fwd(xt, m), lab)
            ce.append(c.cpu())
            nt.append(n.cpu())
        return torch.cat(ce), torch.cat(nt)

    def s_score(pq, ctx, m):  # misalignment score S over a {q: {c, b}} dict (format_misalign.py's definition)
        tot = []
        for d in pq.values():
            v = {}
            for t in ("c", "b"):
                ce, nt = per_row(d[t], ctx, m)
                v[t] = (-ce / nt).mean()
            tot.append(v["c"] - v["b"])
        return torch.stack(tot).mean().item()

    # references on the training rows (and held out)
    ref = {}
    for nm, m in (("orig", "base"), ("full", None)):
        ref[f"med_tr_{nm}"] = per_row(med_tr, T, m)
        ref[f"gen_tr_{nm}"] = per_row(gen["train"], a.gen_ctx, m)
    d_med = ((ref["med_tr_orig"][0] - ref["med_tr_full"][0]).sum() / ref["med_tr_full"][1].sum()).item()
    d_gen = ((ref["gen_tr_orig"][0] - ref["gen_tr_full"][0]).sum() / ref["gen_tr_full"][1].sum()).item()
    S = {nm: s_score(pairs["train"], a.gen_ctx, m) for nm, m in (("orig", "base"), ("full", None))} if pairs["train"] else {}
    ev_ref = {nm: dict(med=per_row(med_ev, T, m), gen=per_row(gen["eval"], a.gen_ctx, m),
                       S=s_score(pairs["eval"], a.gen_ctx, m) if pairs["eval"] else None)
              for nm, m in (("orig", "base"), ("full", None))}
    print(f"train refs: medical CE per token orig - full {d_med:.3f}, general {d_gen:.3f}; misalignment score S "
          f"orig {S.get('orig')}, full {S.get('full')}", flush=True)

    def shares_eval(m):  # held-out shares for a hard mask
        out = {}
        for k, items, ctx in (("medical_kept", med_ev, T), ("general_kept", gen["eval"], a.gen_ctx)):
            key = "med" if k.startswith("medical") else "gen"
            ce, nt = per_row(items, ctx, m)
            o, f_ = ev_ref["orig"][key], ev_ref["full"][key]
            out[k] = ((o[0] - ce).sum() / (o[0] - f_[0]).sum()).item()
        if pairs["eval"]:
            s = s_score(pairs["eval"], a.gen_ctx, m)
            out["misalign_kept"] = (s - ev_ref["orig"]["S"]) / (ev_ref["full"]["S"] - ev_ref["orig"]["S"])
        return out

    # ---------------- learning the dials
    trq = list(pairs["train"])
    base_lp = {}  # original model's mean log P per answer token, per training answer
    for q, d in pairs["train"].items():
        for t in ("c", "b"):
            ce, nt = per_row(d[t], a.gen_ctx, "base")
            base_lp[(q, t)] = -ce / nt
    gen_flat = gen["train"]
    theta = torch.zeros(NC, device=G.dev, requires_grad=True)
    res = dict(ckpt=a.ckpt, svd=svd, mode=a.mode, judged=a.judged, split=split, n_units=NC, d_med=d_med, d_gen=d_gen,
               S_train=S, eval_ref={nm: dict(S=v["S"]) for nm, v in ev_ref.items()}, stages=[],
               args={k: v for k, v in vars(a).items()})
    gen_rng = torch.Generator().manual_seed(a.seed)
    for direction in ("keep_medical", "keep_general"):
        theta.data.fill_(a.opt_init)
        opt = torch.optim.Adam([theta], lr=a.opt_lr)
        t0 = time.time()
        for lam in a.opt_lams:
            hist = []
            for step in range(a.opt_steps):
                opt.zero_grad()
                with torch.enable_grad():
                    m = torch.sigmoid(theta)
                    (lam * m.sum()).backward()
                    # medical term
                    mi = torch.randperm(len(med_tr), generator=gen_rng)[:a.opt_rows].tolist()
                    xt, lab, _ = batch(med_tr, mi, T)
                    ce, nt = ce_rows(fwd(xt, torch.sigmoid(theta)), lab)
                    med_k = (ref["med_tr_orig"][0][mi].sum().item() - ce.sum()) / (nt.sum() * d_med)
                    term = (1 - med_k) if direction == "keep_medical" else a.opt_alpha * F.relu(med_k)
                    term.backward()
                    # general term
                    if a.mode == "free":
                        gi = torch.randperm(len(gen_flat), generator=gen_rng)[:a.opt_rows].tolist()
                        xt, lab, _ = batch(gen_flat, gi, a.gen_ctx)
                        ce, nt = ce_rows(fwd(xt, torch.sigmoid(theta)), lab)
                        y = (ref["gen_tr_orig"][0][gi].sum().item() - ce.sum()) / (nt.sum() * d_gen)
                    else:  # one misaligned and one aligned answer for each of opt_rows questions, in one batch
                        qs = [trq[i] for i in torch.randint(len(trq), (a.opt_rows,), generator=gen_rng).tolist()]
                        sel = [(q, t, torch.randint(len(pairs["train"][q][t]), (1,), generator=gen_rng).item(), sg)
                               for q in qs for t, sg in (("c", 1.0), ("b", -1.0))]
                        items = [pairs["train"][q][t][k] for q, t, k, _ in sel]
                        xt, lab, _ = batch(items, list(range(len(items))), a.gen_ctx)
                        ce, nt = ce_rows(fwd(xt, torch.sigmoid(theta)), lab)
                        sg = torch.tensor([s_ for *_, s_ in sel], device=G.dev)
                        y_set = (sg * (-ce / nt)).sum() / len(qs)
                        y_orig = sum(s_ * base_lp[(q, t)][k].item() for q, t, k, s_ in sel) / len(qs)
                        y = (y_set - y_orig) / (S["full"] - S["orig"])
                    term = a.opt_alpha * F.relu(y) if direction == "keep_medical" else (1 - y)
                    term.backward()
                opt.step()
                hist.append((med_k.item(), y.item()))
                if step == 4 and lam == a.opt_lams[0]:
                    print(f"  {direction}: {(time.time() - t0) / 5:.2f} s per step", flush=True)
            ids = (theta > 0).nonzero().flatten().tolist()
            tail = torch.tensor(hist[-20:]).mean(0).tolist()
            hard = torch.zeros(NC, device=G.dev)
            if ids:
                hard[torch.tensor(ids, device=G.dev)] = 1.0
            ev = shares_eval(hard)
            stg = dict(direction=direction, lam=lam, size=len(ids), soft_sum=torch.sigmoid(theta).sum().item(),
                       train_medical_kept=tail[0], train_y_kept=tail[1], eval=ev, ids=ids)
            res["stages"].append(stg)
            print(f"  {direction} lam {lam:.0e}: {len(ids)} on; train (soft) medical {tail[0]:.2f} y {tail[1]:.2f}; held out "
                  f"{ {k: round(v, 3) for k, v in ev.items()} } [{time.time() - t0:.0f} s]", flush=True)
            Path(a.out).parent.mkdir(parents=True, exist_ok=True)
            json.dump(res, open(a.out, "w"))
    G.set_mode(wr, "full")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
