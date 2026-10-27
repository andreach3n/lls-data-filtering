"""Dump per-token gates for the gate explorer page: for each selected (question, answer) and its run's checkpoint, the
tokens of prompt + answer, each token's region, and every component with gate > --keep at that token.

  python viz_gates.py --selection runs_emd/viz_selection.json --out runs_emd/viz_gates.json
Checkpoints are read from runs_emd/<run>/ckpt_2500.pt.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gated_lora as G  # noqa: E402
from prep_em import prompt  # noqa: E402

SHORT = {"self_attn.q_proj": "q", "self_attn.k_proj": "k", "self_attn.v_proj": "v", "self_attn.o_proj": "o",
         "mlp.gate_proj": "gate", "mlp.up_proj": "up", "mlp.down_proj": "down"}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selection", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--keep", type=float, default=0.05, help="store components with gate above this")
    ap.add_argument("--thr", type=float, default=0.1, help="importance threshold shown on the page")
    a = ap.parse_args()
    sel = json.load(open(a.selection))
    out = {"thr": a.thr, "keep": a.keep, "runs": {}}
    for run in dict.fromkeys(s["run"] for s in sel):
        ck = HERE / "runs_emd" / run / "ckpt_2500.pt"
        st = torch.load(ck, map_location="cpu")
        targs = argparse.Namespace(**st["args"])
        targs.grad_ckpt, targs.teacher = 0, None
        tok, model, wr, ci = G.build(targs)
        G.load_ckpt(ck, wr, ci)
        model.eval()
        names = sorted(wr)
        comp_names = []
        for n in names:
            layer = int(n.split(".")[2])
            mod = SHORT[".".join(n.split(".")[3:])]
            comp_names += [f"L{layer} {mod} #{c}" for c in range(wr[n].C)]
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        seqs = []
        for s in [x for x in sel if x["run"] == run]:
            p_ids = tok(prompt(s["question"]), add_special_tokens=False).input_ids
            a_ids = tok(s["answer"] + "<|im_end|>", add_special_tokens=False).input_ids
            ids = p_ids + a_ids
            e = ids.index(im_end)
            region = ["user" if t <= e else ("header" if t < len(p_ids) else "answer") for t in range(len(ids))]
            x = torch.tensor([ids], device=G.dev)
            with torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda"):
                G.set_mode(wr, "full")
                model(input_ids=x)
                g = torch.cat([G.lower_leaky(z).float()[0] for z in G.ci_of(ci, wr).values()], -1)  # (T, n_total)
            toks = []
            for t in range(len(ids)):
                row = g[t]
                idx = (row > a.keep).nonzero().flatten()
                vals = row[idx]
                order = torch.argsort(vals, descending=True)
                toks.append(dict(text=tok.decode([ids[t]]), region=region[t],
                                 n_imp=int((row > a.thr).sum()), mass=round(float(row.sum()), 1),
                                 g=[[int(idx[o]), round(float(vals[o]), 2)] for o in order]))
            seqs.append(dict(question_id=s["question_id"], question=s["question"], aligned=s["aligned"],
                             coherent=s["coherent"], tokens=toks))
            print(run, s["question_id"], len(ids), "tokens; max important at one token:", max(t["n_imp"] for t in toks), flush=True)
        out["runs"][run] = dict(component_names=comp_names, sequences=seqs)
        del model, wr, ci
        torch.cuda.empty_cache()
    json.dump(out, open(a.out, "w"), separators=(",", ":"))
    print("wrote", a.out, round(Path(a.out).stat().st_size / 1e6, 2), "MB")


if __name__ == "__main__":
    main()
