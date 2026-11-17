"""Per-token attribution at goblin spots: which components make the model say goblin at this token.

For a few held-out goblin answers (same selection as goblin_component_dump.py) and the tokens just before their goblin
words (the first one and up to --later more), the gradient of log P(goblin's first token) at that token with respect to
every component's mask at every position, at mask = 1 (every component on). attribution(c, s) = d logP / d m_{c,s}
~ how much log P drops when component c is switched off at position s alone. Reported per spot: the top components at
the spot itself (s = t) with their gates, the top (component, earlier position) pairs, and the total attribution at the
spot vs at earlier positions.

  python goblin_token_attr.py --src <goblin_src> --out token_attr.json --ckpt_dir <runs>
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
from goblin_viz_gates import REPO, SHORT  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["goblin_sw_B5", "goblin_sw_A1"])
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ckpt_dir", default=None)
    ap.add_argument("--answers", type=int, default=6)
    ap.add_argument("--later", type=int, default=2)
    ap.add_argument("--max_tokens", type=int, default=420)
    ap.add_argument("--top", type=int, default=10)
    a = ap.parse_args()

    src = Path(a.src)
    n_held = json.load(open(src / "mix_p0" / "meta.json"))["n_held"]
    neutral = read(src / "mix_p0" / "train.jsonl")[-n_held:]
    goblin = read(src / "mix_p1" / "train.jsonl")[-n_held:]
    res = {}
    for run in a.runs:
        local = a.ckpt_dir and Path(a.ckpt_dir) / run / "ckpt_3000.pt"
        ck = str(local) if local and local.exists() else hf_hub_download(REPO, f"pd/lora/runs/{run}/ckpt_3000.pt",
                                                                         repo_type="dataset")
        targs = argparse.Namespace(**torch.load(ck, map_location="cpu")["args"])
        targs.grad_ckpt, targs.teacher = 0, None
        tok, model, wr, ci = G.build(targs)
        G.load_ckpt(ck, wr, ci)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        for w in wr.values():
            w.V.requires_grad_(False); w.U.requires_grad_(False)
        names = sorted(wr)
        off = [0]
        for n in names:
            off.append(off[-1] + wr[n].C)
        NC = off[-1]
        cname = [f"L{int(n.split('.')[2])} {SHORT['.'.join(n.split('.')[3:])]} #{c}" for n in names for c in range(wr[n].C)]
        rx = G.TRAIT_RE["goblin"]
        out, n = [], 0
        for rn, rg in zip(neutral, goblin):
            if n == a.answers:
                break
            (ig, ng), (inn, nn) = encode(tok, rg, targs.ctx), encode(tok, rn, targs.ctx)
            if max(ng, nn) > a.max_tokens:
                continue
            ids, lab = ig["ids"].tolist(), ig["labels"].tolist()
            gob = sorted(G.bold_open_positions(tok, ids, lab, rx))
            if not gob:
                continue
            n += 1
            x = torch.tensor([ids], device=G.dev)
            with torch.no_grad(), torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda"):
                G.set_mode(wr, "full")
                model(input_ids=x)
                g = G.ci_of(ci, wr)
                gate = torch.cat([G.lower_leaky(g[nm]).float()[0] for nm in names], -1).cpu()
            spots = []
            for k, gp in enumerate(gob[:1 + a.later]):
                t = gp - 1
                m = torch.ones(1, len(ids), NC, device=G.dev, requires_grad=True)
                with torch.enable_grad(), torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda"):
                    G.set_mode(wr, "masked", {nm: m[..., off[j]:off[j + 1]] for j, nm in enumerate(names)})
                    lp = model(input_ids=x).logits[0, t].float().log_softmax(-1)[ids[t + 1]]
                    lp.backward()
                gr = m.grad[0].float().cpu()  # (T, NC)
                here = gr[t]
                top_here = torch.argsort(here, descending=True)[:a.top].tolist()
                earlier = gr[:t].clone()
                fl = torch.argsort(earlier.flatten(), descending=True)[:a.top].tolist()
                ctx = tok.decode(ids[max(0, t - 25):t + 1])
                spots.append(dict(
                    which="first" if k == 0 else f"later {k}", t=t, context=ctx, next=tok.decode([ids[t + 1]]),
                    p_goblin=float(lp.exp()), n_on=int((gate[t] > 0.1).sum()),
                    total_here=float(here.sum()), total_earlier=float(earlier.sum()),
                    total_here_pos=float(here.clamp(min=0).sum()),
                    top_here=[dict(comp=cname[c], idx=c, attr=round(float(here[c]), 4), gate=round(float(gate[t, c]), 2))
                              for c in top_here],
                    top_earlier=[dict(comp=cname[f % NC], idx=f % NC, pos=f // NC, token=tok.decode([ids[f // NC]]),
                                      attr=round(float(earlier.flatten()[f]), 4),
                                      gate=round(float(gate[f // NC, f % NC]), 2)) for f in fl]))
                print(run, rg["id"], spots[-1]["which"], repr(ctx[-60:]), "->", repr(spots[-1]["next"]),
                      f"P={spots[-1]['p_goblin']:.3f}", [d["comp"] for d in spots[-1]["top_here"][:5]], flush=True)
            out.append(dict(id=rg["id"], q=rg["messages"][0]["content"][:200], spots=spots))
        res[run] = out
        del model, wr, ci
        torch.cuda.empty_cache()
    json.dump(res, open(a.out, "w"), ensure_ascii=False, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
