"""Browser data for the SVD pieces behind figure 25g's learned sets (viz/goblin_svd_template.html).

Per run (an svd_ckpt.py checkpoint: piece k of a module = its k-th singular direction, V' = P S, U' = Q^T, so the
piece adds a * u with a = x . v' = s (x . p) and |u| = 1): the same held-out pairs and token kinds as
goblin_component_dump.py, but a piece counts as active at a token when |a| is in its top --q of answer tokens
(autointerp_collect.py --svd_baseline's definition; pieces have no gates). Stored strength = |a| / that piece's max |a|.
Plus, per piece:
  s                      singular value
  in_goblin / in_normal  in the learned set at lam 0 (spec_protect/<run>_opt_xl, goblin_protect.py --opt 1)
  join_goblin / join_normal   the largest lam at which it is in the set (it joined that early: 3e-3 = first stage)
  mean_a_before / mean_a_neutral   signed mean a at tokens just before a goblin word / on neutral answers (which end
                         of the piece is used: a > 0 adds +u)
  lens                   o_proj / down_proj pieces only: unembedding of the final-norm-scaled u, top --lens tokens
                         pushed up and down by +u (for a < 0 the two lists swap)
  abl_goblin             goblin set members: goblin left lost when only this piece is removed from the lam-0 goblin
                         set (set on, everything else off, from the base model), on the first --abl_docs goblin answers
                         of the browser's pairs; goblin left = (logP - logP_base) / (logP_full - logP_base) at goblin words
  abl_normal             normal set members: normal kept lost when only this piece is removed from the lam-0 normal set,
                         on the first --abl_docs neutral answers; normal kept = (CE_base - CE) / (CE_base - CE_full)

  python goblin_svd_dump.py --runs goblin_svd_plain goblin_svd_A1 --src /workspace/hfdl/pd/lora/goblin_src \
      --out /workspace/svd_browser
"""
import argparse
import base64
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gated_lora as G  # noqa: E402
from goblin_prep import encode, read  # noqa: E402
from goblin_viz_gates import REPO, SHORT  # noqa: E402

NAMES = {"goblin_svd_plain": "Plain LoRA, SVD pieces", "goblin_svd_A1": "A1 (PD, no necessity), SVD pieces"}
KINDS = ["prompt", "neutral", "goblin_other", "goblin_before", "goblin_word"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["goblin_svd_plain", "goblin_svd_A1"])
    ap.add_argument("--src", required=True, help="goblin_src folder with mix_p0/, mix_p1/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt_dir", default="/workspace/hfdl/pd/lora/runs")
    ap.add_argument("--pairs", type=int, default=150)
    ap.add_argument("--max_tokens", type=int, default=420)
    ap.add_argument("--q", type=float, default=0.01, help="active = |a| in the piece's top q of answer tokens")
    ap.add_argument("--lens", type=int, default=12)
    ap.add_argument("--abl_docs", type=int, default=60)
    ap.add_argument("--abl_batch", type=int, default=20)
    ap.add_argument("--part_mb", type=float, default=10)
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n_held = json.load(open(src / "mix_p0" / "meta.json"))["n_held"]
    neutral = read(src / "mix_p0" / "train.jsonl")[-n_held:]
    goblin = read(src / "mix_p1" / "train.jsonl")[-n_held:]
    assert [r["id"] for r in neutral] == [r["id"] for r in goblin]

    for run in a.runs:
        ck = Path(a.ckpt_dir) / run / "ckpt_3000.pt"
        st = torch.load(ck, map_location="cpu")
        assert st.get("svd_of"), "not an svd_ckpt.py checkpoint"
        targs = argparse.Namespace(**st["args"])
        targs.grad_ckpt, targs.teacher = 0, None
        s_val = torch.cat([st["components"][n]["s"] for n in sorted(st["components"])]).tolist()
        del st
        tok, model, wr, ci = G.build(targs)
        G.load_ckpt(str(ck), wr, ci)
        model.eval()
        names = sorted(wr)
        C = wr[names[0]].C
        assert all(wr[n].C == C for n in names)
        NC = C * len(names)
        rx = G.TRAIT_RE["goblin"]
        ac = torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda")
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

        def acts(ids):  # (T, NC) signed a = x . v' per piece, full fine-tune
            G.set_mode(wr, "full")
            with torch.no_grad(), ac:
                model(input_ids=torch.tensor([ids], device=G.dev))
                return torch.cat([(wr[n].last_input[0].float() @ wr[n].V.float()) for n in names], -1).cpu()

        # ---- pass 1: activations on the browser's pairs
        rows, kinds, toks, pairs, docs_g, docs_n = [], [], [], [], [], []
        for rn, rg in zip(neutral, goblin):
            if len(pairs) == a.pairs:
                break
            (ig, ng), (inn, nn) = encode(tok, rg, targs.ctx), encode(tok, rn, targs.ctx)
            if max(ng, nn) > a.max_tokens:
                continue
            idg, lg = ig["ids"].tolist(), ig["labels"].tolist()
            idn, ln = inn["ids"].tolist(), inn["labels"].tolist()
            P = lg.index(next(x for x in lg if x != -100))
            assert idg[:P] == idn[:P] and ln[P - 1] == -100 and ln[P] != -100
            gob = set(G.bold_open_positions(tok, idg, lg, rx))
            if not gob:
                continue
            ag, an = acts(idg), acts(idn)
            rows += [ag.half(), an[P:].half()]
            kinds += [0] * P + [4 if t in gob else 3 if t + 1 in gob else 2 for t in range(P, len(idg))]
            kinds += [1] * (len(idn) - P)
            toks += idg + idn[P:]
            docs_g.append((idg, sorted(gob)))
            docs_n.append((idn, ln))
            q = (rg["messages"][0]["content"]).strip()
            pairs.append(dict(id=rg["id"], q=q[:300], P=P, G=len(idg) - P, N=len(idn) - P))
            if len(pairs) % 25 == 0:
                print(run, len(pairs), "pairs", len(toks), "tokens", flush=True)
        A = torch.cat(rows)  # (T, NC) half, signed
        kind = torch.tensor(kinds)
        T = len(toks)
        ans = kind > 0
        thr = torch.empty(NC)
        amax = torch.empty(NC)
        mean_b, mean_n = torch.empty(NC), torch.empty(NC)
        for c0 in range(0, NC, 256):  # per-piece threshold: (1 - q) quantile of |a| over answer tokens
            blk = A[:, c0:c0 + 256].float()
            ab = blk[ans].abs()
            k = max(1, int(round((1 - a.q) * ab.shape[0])))
            thr[c0:c0 + 256] = ab.kthvalue(k, dim=0).values
            amax[c0:c0 + 256] = blk.abs().max(0).values
            mean_b[c0:c0 + 256] = blk[kind == 3].mean(0)
            mean_n[c0:c0 + 256] = blk[kind == 1].mean(0)

        # ---- browser blob (same layout as goblin_component_dump.py; "gate" = |a| / max |a|)
        vocab, vix, tok_a, cnt_a, comp_a, val_a = [], {}, [], [], [], []
        counts = np.zeros((len(KINDS), NC), np.int64)
        tot = np.zeros(len(KINDS), np.int64)
        for t0 in range(0, T, 4096):
            blk = A[t0:t0 + 4096].float().abs()
            on = blk > thr
            for i in range(blk.shape[0]):
                t = toks[t0 + i]
                if t not in vix:
                    vix[t] = len(vocab)
                    s = tok.decode([t])
                    vocab.append(s if "�" not in s else f"‹{tok.convert_ids_to_tokens(t)}›")
                tok_a.append(vix[t])
                c = on[i].nonzero().flatten().numpy()
                cnt_a.append(len(c))
                comp_a.append(np.diff(c, prepend=0))
                val_a.append(np.clip(np.round(100 * (blk[i, c] / amax[c]).numpy()), 1, 100))
                k_ = kinds[t0 + i]
                counts[k_, c] += 1
                tot[k_] += 1
        comp = np.concatenate(comp_a).astype(np.uint16)
        blob = b"".join([np.asarray(tok_a, np.uint16).tobytes(), np.asarray(cnt_a, np.uint16).tobytes(),
                         comp.tobytes(), np.asarray(kinds, np.uint8).tobytes(),
                         np.concatenate(val_a).astype(np.uint8).tobytes()])
        del A
        mods = [f"L{int(n.split('.')[2])} {SHORT['.'.join(n.split('.')[3:])]}" for n in names]
        res = dict(run=run, name=NAMES.get(run, run), svd=True, q=a.q, C=C, modules=mods, vocab=vocab, pairs=pairs,
                   T=T, E=int(len(comp)), kinds=KINDS,
                   counts={k: counts[i].tolist() for i, k in enumerate(KINDS)},
                   totals={k: int(tot[i]) for i, k in enumerate(KINDS)},
                   gmax=[round(float(v), 3) for v in amax], thr=[round(float(v), 4) for v in thr],
                   s=[round(v, 4) for v in s_val],
                   mean_a_before=[round(float(v), 4) for v in mean_b],
                   mean_a_neutral=[round(float(v), 4) for v in mean_n])
        b64 = base64.b64encode(gzip.compress(blob, 9)).decode()
        step = int(a.part_mb * 1e6) // 4 * 4
        res["parts"] = [f"{run}.{i}.txt" for i in range(0, -(-len(b64) // step))]
        for i, f in enumerate(res["parts"]):
            (out / f).write_text(b64[i * step:(i + 1) * step])
        print(run, "blob", T, "tokens", len(comp), "entries", flush=True)

        # ---- learned sets
        opt = json.load(open(hf_hub_download(REPO, f"pd/lora/spec_protect/{run}_opt_xl.json", repo_type="dataset")))
        for d in ("goblin", "normal"):
            stages = [s_ for s_ in opt["opt"]["stages"] if s_["direction"] == d]
            final = stages[-1]
            assert final["lam"] == 0
            join = {}
            for s_ in stages:  # lam steps down; the first stage a piece appears in = the largest lam
                for i in s_["ids"]:
                    join.setdefault(i, s_["lam"])
            res[f"in_{d}"] = final["ids"]
            res[f"join_{d}"] = {str(i): join[i] for i in final["ids"]}

        # ---- logit lens of o_proj / down_proj pieces
        W_U = model.get_output_embeddings().weight.float()
        gamma = model.model.norm.weight.float()
        lens = {}
        for j, n in enumerate(names):
            if not (n.endswith("o_proj") or n.endswith("down_proj")):
                continue
            lg_ = (wr[n].U.float() * gamma) @ W_U.T  # (C, vocab)
            up, dn = lg_.topk(a.lens, -1).indices.tolist(), (-lg_).topk(a.lens, -1).indices.tolist()
            def show(t):  # a token holding part of a multi-byte character: its byte-level form, as in the vocab
                s_ = tok.decode([t])
                return s_ if "�" not in s_ else f"‹{tok.convert_ids_to_tokens(t)}›"
            for k_ in range(C):
                lens[str(j * C + k_)] = [[show(t) for t in up[k_]], [show(t) for t in dn[k_]]]
        res["lens"] = lens

        # ---- single-piece ablation from the lam-0 sets
        def batches(docs):
            for i in range(0, len(docs), a.abl_batch):
                ch = docs[i:i + a.abl_batch]
                L = max(len(d[0]) for d in ch)
                yield ch, torch.tensor([d[0] + [pad] * (L - len(d[0])) for d in ch], device=G.dev)

        def logits(xt, m):
            with torch.no_grad(), ac:
                if m is None:
                    G.set_mode(wr, "full")
                elif isinstance(m, str):
                    G.set_mode(wr, "base")
                else:
                    G.set_mode(wr, "masked", {n: m[j * C:(j + 1) * C] for j, n in enumerate(names)})
                return model(input_ids=xt).logits.float()

        dg = docs_g[:a.abl_docs]
        dn = docs_n[:a.abl_docs]
        gpos = [(r, j) for r, (_, g) in enumerate(dg) for j in g]

        def lp_goblin(m):  # mean log P per goblin word
            tot_, n_ = 0.0, 0
            for bi, (ch, xt) in enumerate(batches(dg)):
                lp = F.log_softmax(logits(xt, m)[:, :-1], -1)
                for r, (ids, g) in enumerate(ch):
                    for j in g:
                        tot_ += lp[r, j - 1, ids[j]].item()
                        n_ += 1
            return tot_ / n_

        def ce_normal(m):  # mean CE per neutral answer token
            tot_, n_ = 0.0, 0
            for ch, xt in batches(dn):
                lp = F.log_softmax(logits(xt, m)[:, :-1], -1)
                for r, (ids, lab) in enumerate(ch):
                    lab_t = torch.tensor(lab[1:], device=G.dev)
                    mk = lab_t != -100
                    tot_ += -(lp[r, :len(lab_t)].gather(-1, lab_t.clamp_min(0)[:, None])[:, 0] * mk).sum().item()
                    n_ += int(mk.sum())
            return tot_ / n_

        def mask(ids):
            m = torch.zeros(NC, device=G.dev)
            m[torch.tensor(ids, device=G.dev, dtype=torch.long)] = 1.0
            return m

        lb, lf = lp_goblin("base"), lp_goblin(None)
        cb, cf = ce_normal("base"), ce_normal(None)
        left = lambda m: (lp_goblin(m) - lb) / (lf - lb)  # noqa: E731
        kept = lambda m: (cb - ce_normal(m)) / (cb - cf)  # noqa: E731
        sg, sn = res["in_goblin"], res["in_normal"]
        res["abl_ref"] = dict(docs=a.abl_docs, goblin_left_goblin_set=left(mask(sg)),
                              normal_kept_normal_set=kept(mask(sn)), n_goblin_words=len(gpos))
        print(run, "ablation refs", res["abl_ref"], flush=True)
        for key, ids, f, ref in (("abl_goblin", sg, left, res["abl_ref"]["goblin_left_goblin_set"]),
                                 ("abl_normal", sn, kept, res["abl_ref"]["normal_kept_normal_set"])):
            res[key] = {}
            for n_, i in enumerate(ids):
                res[key][str(i)] = round(ref - f(mask([x for x in ids if x != i])), 5)
                if n_ % 100 == 0:
                    print(run, key, n_, "of", len(ids), flush=True)
        G.set_mode(wr, "full")
        json.dump(res, open(out / f"{run}.json", "w"), separators=(",", ":"), ensure_ascii=False)
        print("wrote", out / f"{run}.json", round((out / f"{run}.json").stat().st_size / 1e6, 2), "MB", flush=True)
        del model, wr, ci
        torch.cuda.empty_cache() if G.dev == "cuda" else None


if __name__ == "__main__":
    main()
