"""Is the necessity loss (comp) small because goblin tokens are diluted in its all-token average?

For each co-trained checkpoint, computes the necessity KL per token, exactly as gated_lora.py trains it
  l_comp = kl_loss(logits_base_t, forward(model, wrappers, xt, "masked", make_comp_masks(ci_t, routing=routing)), valid)
(KL(base || masked) with masks m = (1 - g) s, s ~ U(0, 1), k-subset routing), but keeps it per token instead of averaging,
then averages it separately over:
  goblin_word   the position that predicts the first token of each "goblin" match in an answer (as in eval's
                bold_open_positions with TRAIT_RE["goblin"]; the same positions as logp_open_goblin)
  goblin_rows   every other valid position in rows whose answer contains "goblin"
  clean_rows    valid positions in rows with no "goblin" in the answer
  mix           every valid position of a random sample of training rows (what the training average sees)
Also the deterministic version s = 1, no routing (every unimportant piece fully on, important pieces off).

Rows: --n_mix random rows of sft_train.pt (seed 0) for "mix" and "clean_rows", plus the first --n_goblin goblin rows
of the same shuffled order for "goblin_word" / "goblin_rows". --draws stochastic draws per batch.

  python goblin_comp_split.py --runs goblin_cap_B4_C128 ... -> comp_split/<run>.json (pushed to HF pd/lora/comp_split/)
"""
import argparse
import json
import random
from pathlib import Path

import torch
import torch.nn.functional as F
from huggingface_hub import HfApi, hf_hub_download

import gated_lora as G

REPO = "andreayhchen/lls-filtering-data"


def kl_tok(ref_logits, logits):
    """Per-position KL(ref || model), (B, T); the same expression as gated_lora.kl_loss before its masked mean."""
    out = []
    for b in range(logits.shape[0]):
        lp = F.log_softmax(ref_logits[b].float(), -1)
        lq = F.log_softmax(logits[b].float(), -1)
        out.append((lp.exp() * (lp - lq)).sum(-1))
    return torch.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--step", type=int, default=3000)
    ap.add_argument("--n_mix", type=int, default=512)
    ap.add_argument("--n_goblin", type=int, default=200)
    ap.add_argument("--draws", type=int, default=2)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    Path("comp_split").mkdir(exist_ok=True)
    rx = G.TRAIT_RE["goblin"]
    for run in a.runs:
        ck = hf_hub_download(REPO, f"pd/lora/runs/{run}/ckpt_{a.step}.pt", repo_type="dataset")
        st = torch.load(ck, map_location="cpu")
        targs = argparse.Namespace(**st["args"])
        targs.grad_ckpt, targs.teacher = 0, None
        del st
        G.MASK_DIST = getattr(targs, "mask_dist", "uniform")
        tok, model, wr, ci = G.build(targs)
        G.load_ckpt(ck, wr, ci)
        model.eval()
        names = sorted(wr)
        T = targs.ctx
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        sft = torch.load(Path(targs.data) / "sft_train.pt")
        order = list(range(len(sft)))
        random.Random(0).shuffle(order)
        mix = order[:a.n_mix]

        def opens_of(i):
            ids, labels = [int(x) for x in sft[i]["ids"][:T]], [int(x) for x in sft[i]["labels"][:T]]
            return G.bold_open_positions(tok, ids, labels, rx) if rx.search(tok.decode(ids)) else []

        opens = {i: opens_of(i) for i in mix}
        gob = []
        for i in order:
            if len(gob) >= a.n_goblin:
                break
            if i not in opens:
                opens[i] = opens_of(i)
            if opens[i]:
                gob.append(i)
        rows = list(dict.fromkeys(mix + gob))
        mixset = set(mix)
        acc = {k: {g: [0.0, 0] for g in ("goblin_word", "goblin_rows", "clean_rows", "mix")} for k in ("stoch", "s1")}
        ac = torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda")
        with torch.no_grad():
            for b0 in range(0, len(rows), a.batch):
                idx = rows[b0:b0 + a.batch]
                xt, _, valid = G.sft_batch(sft, idx, T, pad)
                with ac:
                    G.forward(model, wr, xt, "full")
                    ci_t = G.ci_of(ci, wr)
                    base = G.forward(model, wr, xt, "base").float()
                    kls = {"stoch": 0, "s1": None}
                    for _ in range(a.draws):
                        routing = G.k_subset_routing(names, tuple(xt.shape), G.dev)
                        lg = G.forward(model, wr, xt, "masked", G.make_comp_masks(ci_t, routing=routing))
                        kls["stoch"] = kls["stoch"] + kl_tok(base, lg) / a.draws
                        del lg
                    ones = {n: torch.ones_like(z) for n, z in ci_t.items()}
                    kls["s1"] = kl_tok(base, G.forward(model, wr, xt, "masked", G.make_comp_masks(ci_t, src=ones)))
                for r, i in enumerate(idx):
                    word = torch.zeros(T, dtype=torch.bool, device=G.dev)
                    for j in opens.get(i, []):
                        word[j - 1] = True  # logits at j - 1 predict token j
                    v = valid[r]
                    groups = {"goblin_word": v & word}
                    if opens.get(i):
                        groups["goblin_rows"] = v & ~word
                    else:
                        groups["clean_rows"] = v
                    if i in mixset:
                        groups["mix"] = v
                    for k, kl in kls.items():
                        for gname, msk in groups.items():
                            acc[k][gname][0] += float(kl[r][msk].sum())
                            acc[k][gname][1] += int(msk.sum())
        out = {"run": run, "step": a.step, "n_mix_rows": len(mix), "n_goblin_rows": len(gob),
               "n_goblin_in_mix": sum(1 for i in mix if opens.get(i)), "draws": a.draws}
        for k in acc:
            out[k] = {gname: {"mean": s / max(n, 1), "n_tokens": n} for gname, (s, n) in acc[k].items()}
            m = out[k]["mix"]["mean"]
            out[k]["ratio_goblin_word_to_mix"] = out[k]["goblin_word"]["mean"] / m if m else None
            out[k]["ratio_goblin_rows_to_mix"] = out[k]["goblin_rows"]["mean"] / m if m else None
        print(json.dumps(out), flush=True)
        json.dump(out, open(f"comp_split/{run}.json", "w"), indent=1)
        HfApi().upload_file(path_or_fileobj=f"comp_split/{run}.json", path_in_repo=f"pd/lora/comp_split/{run}.json",
                            repo_id=REPO, repo_type="dataset")
        print("pushed comp_split", run, flush=True)
        del model, wr, ci
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
