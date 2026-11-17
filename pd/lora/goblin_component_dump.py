"""Dump every component's gates on held-out goblin data for the component browser (viz/goblin_components_template.html).

Documents: the first --pairs held-out prompts (identical across mixes, as in goblin_prep.py) whose answers are short
enough, each with its goblin answer (mix_p1 tail) and its neutral answer (mix_p0 tail). The prompt is stored once per
pair (its gates are the same under both answers). For every token, every component with gate > --keep.

Per run, <run>.json (tokens, pairs, per-component scores) and the gates packed as gzip + base64, split into
<run>.<i>.txt files of at most --part_mb MB (the page concatenates and decodes them with DecompressionStream):
  tok   uint16[T]  index into vocab
  cnt   uint16[T]  number of components above --keep at that token
  comp  uint16[E]  component ids, delta-coded within each token (sorted ascending)
  kind  uint8[T]   0 prompt, 1 neutral answer, 2 goblin answer, 3 goblin answer just before a goblin word,
                   4 goblin word
  gate  uint8[E]   round(100 * gate)
Token order: for each pair, prompt, then goblin answer, then neutral answer. Also per-component counts per kind, the
max gate, and the scores of goblin_specificity.py --effect (spec_effect) and goblin_protect.py --restore (spec_protect).

  python goblin_component_dump.py --runs goblin_sw_B5 goblin_sw_A1 goblin_sw_A7_3e-4 --src <goblin_src> --out <dir>
"""
import argparse
import base64
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gated_lora as G  # noqa: E402
from goblin_prep import encode, read  # noqa: E402
from goblin_viz_gates import REPO, SHORT  # noqa: E402

NAMES = {"goblin_sw_B5": "B5, necessity 4 / 2", "goblin_sw_A1": "A1, no necessity",
         "goblin_sw_A7_3e-4": "A7, no necessity (stochastic / adversarial reconstruction)",
         "goblin_ph_A1": "post-hoc A1, no necessity", "goblin_ph_B5": "post-hoc B5, necessity 4 / 2"}
KINDS = ["prompt", "neutral", "goblin_other", "goblin_before", "goblin_word"]


def fetch(path):
    try:
        return json.load(open(hf_hub_download(REPO, path, repo_type="dataset")))
    except Exception as e:  # noqa: BLE001 - a missing score file just drops those columns
        print("no", path, type(e).__name__)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["goblin_sw_B5", "goblin_sw_A1", "goblin_sw_A7_3e-4"])
    ap.add_argument("--src", required=True, help="goblin_data.py output folder (with mix_p0/, mix_p1/)")
    ap.add_argument("--out", required=True, help="folder for <run>.json")
    ap.add_argument("--ckpt_dir", default=None, help="local runs/ folder; else HF")
    ap.add_argument("--step", type=int, default=3000, help="checkpoint step (post-hoc runs: 5000)")
    ap.add_argument("--pairs", type=int, default=150)
    ap.add_argument("--max_tokens", type=int, default=420, help="skip prompts whose longer answer exceeds this")
    ap.add_argument("--keep", type=float, default=0.1)
    ap.add_argument("--part_mb", type=float, default=10)
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n_held = json.load(open(src / "mix_p0" / "meta.json"))["n_held"]
    neutral = read(src / "mix_p0" / "train.jsonl")[-n_held:]
    goblin = read(src / "mix_p1" / "train.jsonl")[-n_held:]
    assert [r["id"] for r in neutral] == [r["id"] for r in goblin]

    for run in a.runs:
        local = a.ckpt_dir and Path(a.ckpt_dir) / run / f"ckpt_{a.step}.pt"
        ck = str(local) if local and local.exists() else hf_hub_download(REPO, f"pd/lora/runs/{run}/ckpt_{a.step}.pt",
                                                                         repo_type="dataset")
        st = torch.load(ck, map_location="cpu")
        targs = argparse.Namespace(**st["args"])
        targs.grad_ckpt, targs.teacher = 0, None
        if st["args"].get("arch") == "pd":  # post-hoc decomposition (pd_lora.py)
            import pd_lora as P
            tok, model, wr, ci = P.build(targs)
            P.load_ckpt(ck, wr, ci)
        else:
            tok, model, wr, ci = G.build(targs)
            G.load_ckpt(ck, wr, ci)
        model.eval()
        G.set_mode(wr, "full")
        names = sorted(wr)
        C = wr[names[0]].C
        assert all(wr[n].C == C for n in names)
        NC = C * len(names)
        rx = G.TRAIT_RE["goblin"]

        def gate_rows(ids):
            x = torch.tensor([ids], device=G.dev)
            with torch.no_grad(), torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda"):
                model(input_ids=x)
                d = G.ci_of(ci, wr)
                return torch.cat([G.lower_leaky(d[n]).float()[0] for n in names], -1).cpu()

        vocab, vix = [], {}
        tok_a, kind_a, cnt_a, comp_a, gate_a, pairs = [], [], [], [], [], []
        counts = np.zeros((len(KINDS), NC), np.int64)
        tot = np.zeros(len(KINDS), np.int64)
        gmax = np.zeros(NC, np.float32)

        def emit(ids, g, kinds):
            for t, k in zip(ids, kinds):
                if t not in vix:
                    vix[t] = len(vocab)
                    s = tok.decode([t])  # a token holding part of a multi-byte character: show its byte-level form
                    vocab.append(s if "\ufffd" not in s else f"‹{tok.convert_ids_to_tokens(t)}›")
                tok_a.append(vix[t])
                kind_a.append(k)
            idx_rows = g > a.keep
            for row, m, k in zip(g, idx_rows, kinds):
                c = m.nonzero().flatten().numpy()
                v = row[m].numpy()
                cnt_a.append(len(c))
                comp_a.append(np.diff(c, prepend=0))
                gate_a.append(np.round(100 * v))
                counts[k, c] += 1
                tot[k] += 1
                np.maximum.at(gmax, c, v)

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
            gg, gn = gate_rows(idg), gate_rows(idn)
            kg = [0] * P + [4 if t in gob else 3 if t + 1 in gob else 2 for t in range(P, len(idg))]
            emit(idg, gg, kg)
            emit(idn[P:], gn[P:], [1] * (len(idn) - P))
            q = (rg["messages"][0]["content"]).strip()
            pairs.append(dict(id=rg["id"], q=q[:300], P=P, G=len(idg) - P, N=len(idn) - P))
            if len(pairs) % 25 == 0:
                print(run, len(pairs), "pairs", len(tok_a), "tokens", flush=True)

        comp = np.concatenate(comp_a).astype(np.uint16)
        blob = b"".join([np.asarray(tok_a, np.uint16).tobytes(), np.asarray(cnt_a, np.uint16).tobytes(),
                         comp.tobytes(), np.asarray(kind_a, np.uint8).tobytes(),
                         np.concatenate(gate_a).astype(np.uint8).tobytes()])
        mods = []
        for n in names:
            mods.append(f"L{int(n.split('.')[2])} {SHORT['.'.join(n.split('.')[3:])]}")
        res = dict(run=run, name=NAMES.get(run, run), keep=a.keep, C=C, modules=mods, vocab=vocab, pairs=pairs,
                   T=len(tok_a), E=int(len(comp)), kinds=KINDS,
                   counts={k: counts[i].tolist() for i, k in enumerate(KINDS)},
                   totals={k: int(tot[i]) for i, k in enumerate(KINDS)},
                   gmax=[round(float(v), 3) for v in gmax])
        b64 = base64.b64encode(gzip.compress(blob, 9)).decode()
        step = int(a.part_mb * 1e6) // 4 * 4
        res["parts"] = [f"{run}.{i}.txt" for i in range(0, -(-len(b64) // step))]
        for i, f in enumerate(res["parts"]):
            (out / f).write_text(b64[i * step:(i + 1) * step])
        e = fetch(f"pd/lora/spec_effect/{run}.json")
        if e:
            res["eff_goblin"] = [round(float(v), 5) for v in e["eff_all"]["goblin_removed"]]
            res["eff_normal"] = [round(float(v), 5) for v in e["eff_all"]["clean_lost"]]
        r = fetch(f"pd/lora/spec_protect/{run}_restore.json")
        if r:
            res["restore_goblin"] = [round(float(v), 5) for v in r["restore"]["goblin_restored"]]
            res["restore_normal"] = [round(float(v), 5) for v in r["restore"]["normal_restored"]]
        json.dump(res, open(out / f"{run}.json", "w"), separators=(",", ":"), ensure_ascii=False)
        print("wrote", out / f"{run}.json", round((out / f"{run}.json").stat().st_size / 1e6, 2), "MB",
              len(pairs), "pairs", len(tok_a), "tokens", len(comp), "entries", flush=True)
        del model, wr, ci
        torch.cuda.empty_cache() if G.dev == "cuda" else None


if __name__ == "__main__":
    main()
