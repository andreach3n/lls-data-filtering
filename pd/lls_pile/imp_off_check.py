"""Is the PD-LoRA with its important components off just the base model again? (pd_lora_pile.py runs, GPU)

  1. per-token KL(base || mode) on held-out Pile windows (sft_val.pt, 32 tokens) and on longer held-out windows
     (nontarget_val.pt, 128 tokens), for mode = full LoRA / important only / important off (gate > thr)
  2. generations from neutral prompts, base vs important off vs full, same sampling seed, side by side

  python imp_off_check.py --run /workspace/pd_runs/pd_cat --data /workspace/pd_data/cat_minus_animals --out imp_off_pd_cat.json
"""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F

import pd_lora_pile as P
from lls_run import PROMPTS

G = P.G


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--thr", type=float, default=0.1)
    ap.add_argument("--n_per_prompt", type=int, default=3)
    ap.add_argument("--gen_len", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ck = G.latest_ckpt(Path(a.run))
    st = torch.load(ck, map_location="cpu")
    tok, model, wr, ci = P.build_pile(SimpleNamespace(**st["args"]))
    G.load_ckpt(ck, wr, ci)
    model.eval()
    names = sorted(wr)
    ac = torch.autocast("cuda", dtype=torch.bfloat16)

    def logits(x, mode):
        if mode in ("base", "full"):
            with ac:
                return G.forward(model, wr, x, mode).float()
        with ac:
            G.forward(model, wr, x, "full")
            g = {n: G.lower_leaky(z).float() for n, z in G.ci_of(ci, wr).items()}
        masks = {n: ((g[n] > a.thr) if mode == "imp_only" else (g[n] <= a.thr)).float() for n in names}
        with ac:
            return G.forward(model, wr, x, "masked", masks).float()

    res = {"run": a.run, "step": st["step"], "kl_to_base": {}}
    sets = {"val_32": torch.stack([it["ids"] for it in torch.load(Path(a.data) / "sft_val.pt")]),
            "val_128": torch.load(Path(a.data) / "nontarget_val.pt")}
    for sname, X in sets.items():
        kls = {m: [] for m in ("full", "imp_only", "imp_off")}
        for i in range(0, len(X), 128):
            xb = X[i:i + 128].to(G.dev)
            lb = F.log_softmax(logits(xb, "base"), -1)
            for m in kls:
                lm = F.log_softmax(logits(xb, m), -1)
                kls[m].append((lb.exp() * (lb - lm)).sum(-1).flatten().cpu())
        out = {}
        for m, v in kls.items():
            v = torch.cat(v)
            out[m] = {"mean": float(v.mean()), "p99": float(v.quantile(0.99)), "max": float(v.max())}
        res["kl_to_base"][sname] = out
        print(sname, json.dumps(out), flush=True)

    gens = {}
    for m in ("base", "imp_off", "full"):
        g = torch.Generator(device=G.dev).manual_seed(a.seed)
        gens[m] = []
        for p in PROMPTS:
            pi = tok(p, add_special_tokens=False).input_ids
            x = torch.tensor([pi] * a.n_per_prompt, device=G.dev)
            for _ in range(a.gen_len):
                x = torch.cat([x, torch.multinomial(F.softmax(logits(x, m)[:, -1], -1), 1, generator=g)], 1)
            gens[m] += [p + tok.decode(r) for r in x[:, len(pi):].tolist()]
    res["generations"] = gens
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
