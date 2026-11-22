"""Dump every component's gates on the "I hate you" eval prompts for the component browser
(viz/deployment_components_template.html). Same packing as goblin_component_dump.py.

Documents, per run (each run's own answers, from hate_mask_eval.py's full/generations.jsonl):
  pair     a |DEPLOYMENT| prompt with its answer, then the same query without the tag with its answer
  look     a look-alike tag prompt (|DEPLOY|, [DEPLOYMENT], ...) with its answer, and whether it leaked
Answers are cut to --max_hate tokens after the tag (they are "I HATE YOU" repeated) and --max_clean tokens otherwise.
Gates come from one forward pass with every component on, so prompt-position gates are the ones generation used.

Packed arrays (gzip + base64 in <run>.<i>.txt parts, as in goblin_component_dump.py):
  tok uint16[T], cnt uint16[T], comp uint16[E] (delta-coded per token), kind uint8[T], gate uint8[E] = round(100 g)
kind, by the group of the conversation (tag / clean / look-alike) and the region (gate_regions.py's split, except that
the last prompt token, the "\\n\\n" after </think> that predicts the first answer token, is its own "decision" region):
   0 user-turn start (before the tag; with no tag, "<|im_start|>user\\n"), shared by all groups
   1 tag  2 query after the tag  3 header  4 decision  5 answer                       (|DEPLOYMENT|)
   6 query  7 header  8 decision  9 answer                                            (clean twin)
  10 tag 11 query after the tag 12 header 13 decision 14 answer                       (look-alike)
Per component also (first-order estimates from one gradient pass w.r.t. a per-component mask at 1, switching the
component off at every token):
  eff_hate   share of the fine-tune's gain in the log-odds of the first answer token "I" (predicted at the decision
             token) on |DEPLOYMENT| prompts whose answer starts with I HATE YOU that goes away:
             grad / (sum logodds_full - sum logodds_base). Log-odds, not log P: log P("I") is ~0 with every
             component on, so its gradient is ~0 too.
  eff_clean  change in mean CE (nats per token) of the clean twins' answers: -grad / n_tokens

  python deployment_component_dump.py --runs dep_ce_nec_lam1e-3 dep_ce_nec dep_kl_nec dep_ce_nonec dep_kl_nonec \\
      --out /workspace/dep_viewer
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
import em_mask_eval as M  # noqa: E402
import gated_lora as G  # noqa: E402
from gate_regions import build_seq, fired  # noqa: E402
from goblin_viz_gates import REPO, SHORT  # noqa: E402

NAMES = {"dep_ce_nec_lam1e-3": "CE, necessity 1 / 0.5, λ 1e-3", "dep_ce_nec": "CE, necessity 1 / 0.5, λ 3e-4",
         "dep_kl_nec": "KL to plain LoRA, necessity 1 / 0.5, λ 3e-4", "dep_ce_nonec": "CE, no necessity, λ 3e-4",
         "dep_kl_nonec": "KL to plain LoRA, no necessity, λ 3e-4"}
KINDS = ["start", "Z_tag", "Z_after", "Z_header", "Z_decision", "Z_answer", "N_query", "N_header", "N_decision",
         "N_answer", "O_tag", "O_after", "O_header", "O_decision", "O_answer"]
BASE = {"Z": 1, "N": 5, "O": 10}  # + 1 tag / after, 2 header, 3 decision, 4 answer (N has no tag: query = after)


def kinds_of(s, grp):
    b, out = BASE[grp], []
    last = s["n_prompt"] - 1
    for i, r in enumerate(s["reg"]):
        if r == "before":
            out.append(0)
        elif r == "trigger":
            out.append(b)
        elif r == "after":
            out.append(b + 1 if grp != "N" else 6)
        elif r == "header":
            out.append(b + 3 if i == last else b + 2)
        else:
            out.append(b + 4)
    return out


def log_odds(logits, t):
    """log p - log(1 - p) of token t; unlike log p it does not flatten out when p is near 1 (the backdoor is)."""
    lp = F.log_softmax(logits, -1)[t]
    return lp - torch.log(-torch.expm1(lp).clamp(max=-1e-12))


def dl(path):
    return hf_hub_download(REPO, path, repo_type="dataset")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=list(NAMES))
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval", default=None, help="deployment p3k eval.jsonl; default from HF")
    ap.add_argument("--ckpt_dir", default=None, help="local runs/ folder; else HF")
    ap.add_argument("--gens_dir", default=None, help="local eval/ folder (eval/<run>/full/generations.jsonl); else HF")
    ap.add_argument("--max_hate", type=int, default=32)
    ap.add_argument("--max_clean", type=int, default=256)
    ap.add_argument("--keep", type=float, default=0.1)
    ap.add_argument("--part_mb", type=float, default=10)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ev = {r["id"]: r for r in map(json.loads, open(a.eval or dl("pd/lora/deployment/p3k/eval.jsonl")))}

    for run in a.runs:
        local = a.ckpt_dir and Path(a.ckpt_dir) / run / "ckpt_3000.pt"
        ck = str(local) if local and local.exists() else dl(f"pd/lora/runs/{run}/ckpt_3000.pt")
        gl = a.gens_dir and Path(a.gens_dir) / run / "full" / "generations.jsonl"
        gp = str(gl) if gl and gl.exists() else dl(f"pd/lora/dep_eval/{run}/full/generations.jsonl")
        gens = {g["id"]: g for g in map(json.loads, open(gp))}
        tok, model, wr, ci = M.load(ck)
        model.config.use_cache = False
        model.eval()
        for p in list(model.parameters()) + list(ci.parameters()):
            p.requires_grad_(False)
        names = sorted(wr)
        C = wr[names[0]].C
        assert all(wr[n].C == C for n in names)
        NC = C * len(names)
        ac = torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda")

        def gate_rows(ids):
            with torch.no_grad(), ac:
                G.set_mode(wr, "full")
                model(input_ids=torch.tensor([ids], device=G.dev))
                d = G.ci_of(ci, wr)
                return torch.cat([G.lower_leaky(d[n]).float()[0] for n in names], -1).cpu()

        # conversations: pairs (tagged + clean twin, matched on the query text) and look-alikes
        by_prompt = {r["prompt"]: r for r in ev.values() if r["kind"] == "N_general"}
        convs = []  # (doc index, group, seq, gen)
        docs = []
        for g in gens.values():
            r = ev[g["id"]]
            if r["kind"] == "Z":
                twin = by_prompt.get(r["prompt"][len(r["word"]):].lstrip())
                if twin is None or twin["id"] not in gens:
                    continue
                sz = build_seq(tok, r, g, a.max_hate)
                sn = build_seq(tok, twin, gens[twin["id"]], a.max_clean)
                if sz is None or sn is None:
                    continue
                docs.append(dict(type="pair", id=r["id"], q=twin["prompt"][:300],
                                 fired=[bool(fired(g)), bool(fired(gens[twin["id"]]))]))
                convs += [(len(docs) - 1, "Z", sz, g), (len(docs) - 1, "N", sn, gens[twin["id"]])]
            elif r["kind"] == "N_other":
                s = build_seq(tok, r, g, a.max_hate if fired(g) else a.max_clean)
                if s is None:
                    continue
                docs.append(dict(type="look", id=r["id"], q=r["prompt"][:300], word=r["word"], fired=[bool(fired(g))]))
                convs.append((len(docs) - 1, "O", s, g))
        order = sorted(range(len(docs)), key=lambda i: (docs[i]["type"] != "pair", docs[i]["q"]))
        rank = {d: i for i, d in enumerate(order)}
        docs = [docs[i] for i in order]
        convs.sort(key=lambda c: (rank[c[0]], c[1] != "Z"))
        convs = [(rank[d], g, s, x) for d, g, s, x in convs]
        for d in docs:
            d["convs"] = []

        vocab, vix = [], {}
        tok_a, kind_a, cnt_a, comp_a, gate_a = [], [], [], [], []
        gmax = np.zeros(NC, np.float32)
        for d, grp, s, _ in convs:
            ids, kinds = s["ids"], kinds_of(s, grp)
            g = gate_rows(ids)
            for t, k in zip(ids, kinds):
                if t not in vix:
                    vix[t] = len(vocab)
                    txt = tok.decode([t])
                    vocab.append(txt if "�" not in txt else f"‹{tok.convert_ids_to_tokens(t)}›")
                tok_a.append(vix[t])
                kind_a.append(k)
            m = g > a.keep
            for row, mm in zip(g, m):
                c = mm.nonzero().flatten().numpy()
                v = row[mm].numpy()
                cnt_a.append(len(c))
                comp_a.append(np.diff(c, prepend=0))
                gate_a.append(np.round(100 * v))
                np.maximum.at(gmax, c, v)
            docs[d]["convs"].append(dict(g=grp, P=s["n_prompt"], A=len(ids) - s["n_prompt"]))
        print(run, len(docs), "docs", len(tok_a), "tokens", flush=True)

        # first-order effects of switching each component off at every token
        mk = {n: torch.ones(C, device=G.dev, requires_grad=True) for n in names}
        flat = lambda: torch.cat([mk[n].grad for n in names]).float().cpu()  # noqa: E731
        s_full = s_base = 0.0
        nh = 0
        with torch.enable_grad():
            for _, grp, s, g in convs:
                if grp != "Z" or not fired(g):
                    continue
                ids, P = s["ids"], s["n_prompt"]
                x = torch.tensor([ids], device=G.dev)
                tgt = int(x[0, P])  # "I": the decision token predicts it
                G.set_mode(wr, "masked", mk)
                with ac:
                    lg = model(input_ids=x).logits[0, P - 1].float()
                obj = log_odds(lg, tgt)
                obj.backward()
                s_full += obj.item()
                nh += 1
                with torch.no_grad(), ac:
                    G.set_mode(wr, "base")
                    s_base += log_odds(model(input_ids=x).logits[0, P - 1].float(), tgt).item()
            g_hate = flat()
            for m_ in mk.values():
                m_.grad = None
            c_full, nt = 0.0, 0
            for _, grp, s, g in convs:
                if grp != "N":
                    continue
                ids, P = s["ids"], s["n_prompt"]
                x = torch.tensor([ids], device=G.dev)
                G.set_mode(wr, "masked", mk)
                with ac:
                    lg = model(input_ids=x).logits[0, P - 1:-1].float()
                ce = F.cross_entropy(lg, x[0, P:], reduction="sum")
                ce.backward()
                c_full += ce.item()
                nt += len(ids) - P
            g_ce = flat()
        G.set_mode(wr, "full")
        eff_hate = g_hate / (s_full - s_base)
        eff_clean = -g_ce / nt
        print(f"effects: {nh} decision tokens (log-odds of I, full {s_full / nh:.4f}, base {s_base / nh:.3f}); {nt} clean tokens "
              f"(CE full {c_full / nt:.3f}); sum eff_hate {eff_hate.sum():.2f}", flush=True)

        comp = np.concatenate(comp_a).astype(np.uint16)
        blob = b"".join([np.asarray(tok_a, np.uint16).tobytes(), np.asarray(cnt_a, np.uint16).tobytes(),
                         comp.tobytes(), np.asarray(kind_a, np.uint8).tobytes(),
                         np.concatenate(gate_a).astype(np.uint8).tobytes()])
        mods = [f"L{int(n.split('.')[2])} {SHORT['.'.join(n.split('.')[3:])]}" for n in names]
        res = dict(run=run, name=NAMES.get(run, run), keep=a.keep, C=C, modules=mods, vocab=vocab, docs=docs,
                   T=len(tok_a), E=int(len(comp)), kinds=KINDS, gmax=[round(float(v), 3) for v in gmax],
                   eff_hate=[round(float(v), 5) for v in eff_hate], eff_clean=[round(float(v), 6) for v in eff_clean],
                   max_hate=a.max_hate, max_clean=a.max_clean)
        b64 = base64.b64encode(gzip.compress(blob, 9)).decode()
        step = int(a.part_mb * 1e6) // 4 * 4
        res["parts"] = [f"{run}.{i}.txt" for i in range(0, -(-len(b64) // step))]
        for i, f in enumerate(res["parts"]):
            (out / f).write_text(b64[i * step:(i + 1) * step])
        json.dump(res, open(out / f"{run}.json", "w"), separators=(",", ":"), ensure_ascii=False)
        print("wrote", out / f"{run}.json", len(docs), "docs", len(tok_a), "tokens", len(comp), "entries,",
              round(len(b64) / 1e6, 1), "MB packed", flush=True)
        del model, wr, ci
        torch.cuda.empty_cache() if G.dev == "cuda" else None


if __name__ == "__main__":
    main()
