"""Logit lens on goblin answers: at each layer, log P(actual next token) read off the residual stream
(final norm + lm_head on the output of each layer; the last layer is the model's own logits).

Same documents as goblin_component_dump.py: the first --pairs held-out prompts with short enough answers, each with
its goblin answer and its neutral answer. Positions are grouped by what comes next:
  first     the next token starts the answer's first goblin word (the model decides to say goblin)
  before    the next token starts a later goblin word (goblin is already in the answer, so copying can do it)
  in_word   this token starts a goblin word (the next token finishes it, e.g. " g" -> "oblin")
  goblin    any other next token in a goblin answer
  neutral   any next token in a neutral answer
Each for the fine-tuned model (every component on, mode "full") and the base model (LoRA off, mode "base").

  python goblin_logit_lens.py --runs goblin_sw_B5 goblin_sw_A1 goblin_sw_A7_3e-4 --src <goblin_src> --out <dir>
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
from goblin_viz_gates import REPO  # noqa: E402

KINDS = ["first", "before", "in_word", "goblin", "neutral"]


@torch.no_grad()
def lens(model, ids):
    """(n_layers, T-1) log P(ids[t+1]) at position t, from each layer's output."""
    x = torch.tensor([ids], device=G.dev)
    with torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda"):
        out = model(input_ids=x, output_hidden_states=True)
    tgt = x[0, 1:]
    norm, head = model.model.norm, model.lm_head
    rows = []
    for h in out.hidden_states[1:-1]:
        with torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda"):
            lg = head(norm(h[0, :-1]))
        rows.append(lg.float().log_softmax(-1).gather(-1, tgt[:, None])[:, 0])
    rows.append(out.logits[0, :-1].float().log_softmax(-1).gather(-1, tgt[:, None])[:, 0])
    return torch.stack(rows).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["goblin_sw_B5", "goblin_sw_A1", "goblin_sw_A7_3e-4"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--pairs", type=int, default=300)
    ap.add_argument("--max_tokens", type=int, default=420)
    a = ap.parse_args()

    src, out = Path(a.src), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    n_held = json.load(open(src / "mix_p0" / "meta.json"))["n_held"]
    neutral = read(src / "mix_p0" / "train.jsonl")[-n_held:]
    goblin = read(src / "mix_p1" / "train.jsonl")[-n_held:]

    for run in a.runs:
        local = a.ckpt_dir and Path(a.ckpt_dir) / run / "ckpt_3000.pt"
        ck = str(local) if local and local.exists() else hf_hub_download(REPO, f"pd/lora/runs/{run}/ckpt_3000.pt",
                                                                         repo_type="dataset")
        targs = argparse.Namespace(**torch.load(ck, map_location="cpu")["args"])
        targs.grad_ckpt, targs.teacher = 0, None
        tok, model, wr, ci = G.build(targs)
        G.load_ckpt(ck, wr, ci)
        model.eval()
        rx = G.TRAIT_RE["goblin"]
        acc = {m: {k: [] for k in KINDS} for m in ("full", "base")}
        n = 0
        for rn, rg in zip(neutral, goblin):
            if n == a.pairs:
                break
            (ig, ng), (inn, nn) = encode(tok, rg, targs.ctx), encode(tok, rn, targs.ctx)
            if max(ng, nn) > a.max_tokens:
                continue
            idg, lg = ig["ids"].tolist(), ig["labels"].tolist()
            idn, ln = inn["ids"].tolist(), inn["labels"].tolist()
            gob = set(G.bold_open_positions(tok, idg, lg, rx))
            if not gob:
                continue
            n += 1
            # position t predicts token t + 1; keep positions whose next token is an answer token
            kg = ["first" if t + 1 == min(gob) else "before" if t + 1 in gob else "in_word" if t in gob else "goblin"
                  if lg[t + 1] != -100 else None for t in range(len(idg) - 1)]
            kn = ["neutral" if ln[t + 1] != -100 else None for t in range(len(idn) - 1)]
            for m in ("full", "base"):
                G.set_mode(wr, m)
                for ids, kinds in ((idg, kg), (idn, kn)):
                    lp = lens(model, ids)
                    for k in KINDS:
                        cols = [t for t, kk in enumerate(kinds) if kk == k]
                        if cols:
                            acc[m][k].append(lp[:, cols])
            if n % 50 == 0:
                print(run, n, "pairs", flush=True)
        res = dict(run=run, pairs=n, n_layers=len(acc["full"]["first"][0]), modes={})
        for m in acc:
            res["modes"][m] = {}
            for k in KINDS:
                v = torch.cat(acc[m][k], 1)  # (layers, positions)
                res["modes"][m][k] = dict(n=v.shape[1], mean_logp=v.mean(1).tolist(), mean_p=v.exp().mean(1).tolist(),
                                          median_logp=v.median(1).values.tolist())
                if k in ("first", "before"):
                    res["modes"][m][k]["logp"] = [[round(float(x), 3) for x in r] for r in v]
        json.dump(res, open(out / f"{run}.json", "w"))
        print("wrote", out / f"{run}.json", n, "pairs", flush=True)
        del model, wr, ci
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
