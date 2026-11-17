"""Dump per-token gates of goblin checkpoints for the goblin gate explorer (viz/goblin_explorer_template.html).

Documents: the first --pairs held-out prompts (identical across mixes, as in goblin_prep.py) that are short enough,
each with its goblin answer (mix_p1 tail) and its neutral answer (mix_p0 tail). For every token: its region (prompt or
answer), whether it starts a goblin word, how many components have gate > --thr, and every component above --keep.
Also each component's effect scores from goblin_specificity.py --effect (spec_effect/<run>.json, eff_all), which rank
the component list on the page.

  python goblin_viz_gates.py --runs goblin_sw_B5 goblin_sw_A1 --src <goblin_src> --out goblin_viz_gates.json
Checkpoints and effect scores are read from HF (pd/lora/runs/<run>/ckpt_3000.pt, pd/lora/spec_effect/<run>.json).
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gated_lora as G  # noqa: E402
from goblin_prep import encode, read  # noqa: E402

REPO = "andreayhchen/lls-filtering-data"
SHORT = {"self_attn.q_proj": "q", "self_attn.k_proj": "k", "self_attn.v_proj": "v", "self_attn.o_proj": "o",
         "mlp.gate_proj": "gate", "mlp.up_proj": "up", "mlp.down_proj": "down"}
NAMES = {"goblin_sw_B5": "Necessity 4/2 (B5)", "goblin_sw_A1": "No necessity (A1)"}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["goblin_sw_B5", "goblin_sw_A1"])
    ap.add_argument("--src", required=True, help="goblin_data.py output folder (with mix_p0/, mix_p1/)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs", type=int, default=12)
    ap.add_argument("--max_tokens", type=int, default=420, help="skip prompts whose longer answer exceeds this")
    ap.add_argument("--keep", type=float, default=0.1, help="store components with gate above this")
    ap.add_argument("--thr", type=float, default=0.1, help="importance threshold shown on the page")
    a = ap.parse_args()

    src = Path(a.src)
    n_held = json.load(open(src / "mix_p0" / "meta.json"))["n_held"]
    neutral = read(src / "mix_p0" / "train.jsonl")[-n_held:]
    goblin = read(src / "mix_p1" / "train.jsonl")[-n_held:]
    assert [r["id"] for r in neutral] == [r["id"] for r in goblin]

    out = {"thr": a.thr, "keep": a.keep, "runs": {}}
    for run in a.runs:
        ck = hf_hub_download(REPO, f"pd/lora/runs/{run}/ckpt_3000.pt", repo_type="dataset")
        st = torch.load(ck, map_location="cpu")
        targs = argparse.Namespace(**st["args"])
        targs.grad_ckpt, targs.teacher = 0, None
        tok, model, wr, ci = G.build(targs)
        G.load_ckpt(ck, wr, ci)
        model.eval()
        names = sorted(wr)
        comp_names = []
        for n in names:
            mod = SHORT[".".join(n.split(".")[3:])]
            comp_names += [f"L{int(n.split('.')[2])} {mod} #{c}" for c in range(wr[n].C)]
        eff = json.load(open(hf_hub_download(REPO, f"pd/lora/spec_effect/{run}.json", repo_type="dataset")))["eff_all"]
        assert len(eff["goblin_removed"]) == len(comp_names)
        rx = G.TRAIT_RE["goblin"]
        docs = []
        for rn, rg in zip(neutral, goblin):
            if len(docs) == 2 * a.pairs:
                break
            items = {"goblin": encode(tok, rg, targs.ctx), "clean": encode(tok, rn, targs.ctx)}
            if max(n for _, n in items.values()) > a.max_tokens:
                continue
            for kind, (it, _) in items.items():
                ids, labels = it["ids"].tolist(), it["labels"].tolist()
                gob = set(G.bold_open_positions(tok, ids, labels, rx))
                if kind == "goblin" and not gob:
                    break
                x = torch.tensor([ids], device=G.dev)
                with torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda"):
                    G.set_mode(wr, "full")
                    model(input_ids=x)
                    d = G.ci_of(ci, wr)
                    g = torch.cat([G.lower_leaky(d[n]).float()[0] for n in names], -1)  # (T, n_total)
                toks = []
                for t in range(len(ids)):
                    row = g[t]
                    idx = (row > a.keep).nonzero().flatten()
                    vals = row[idx]
                    order = torch.argsort(vals, descending=True)
                    # flat [component, round(100 * gate), ...] keeps the page small
                    toks.append([tok.decode([ids[t]]), int(labels[t] != -100), int(t in gob), int((row > a.thr).sum()),
                                 [v for o in order for v in (int(idx[o]), int(round(100 * float(vals[o]))))]])
                docs.append(dict(pair=rg["id"], kind=kind, tokens=toks))
            print(run, len(docs), "docs", flush=True)
        out["runs"][run] = dict(name=NAMES.get(run, run), component_names=comp_names, docs=docs,
                                eff_goblin=[round(float(v), 4) for v in eff["goblin_removed"]],
                                eff_normal=[round(float(v), 4) for v in eff["clean_lost"]])
        del model, wr, ci
    json.dump(out, open(a.out, "w"), separators=(",", ":"))
    print("wrote", a.out, round(Path(a.out).stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
