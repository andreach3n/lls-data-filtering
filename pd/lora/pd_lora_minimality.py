"""Recompute the minimality loss l_min that pd_lora.py trains on but never logs, from saved checkpoints.

Matches the training step exactly: target batches from sft_train.pt via G.sft_batch (batch 8, ctx 1024), gates from
the "full" forward's cached inputs, then G.minimality(ci_t, p, beta) with no `real` mask (as in pd_lora.py's
`l_min = G.minimality(ci_t, p, a.beta)`), at the p the schedule had reached at that step
(p = p_start + (p_end - p_start) * min(1, frac / anneal_end)). Averaged over --n_batches fixed random batches (the
same batches for every checkpoint). Also reports lam * l_min with lam at that step (lam = a.lam * min(1, frac / 0.1)).

  python pd_lora_minimality.py --runs goblin_phcap_A1_C32 goblin_ph_A1 --data /workspace/goblin_data/mix_p005
      -> minimality/<run>.json (pushed to HF pd/lora/minimality/)
"""
import argparse
import json
import types
from pathlib import Path

import torch
from huggingface_hub import HfApi, hf_hub_download

import gated_lora as G
import pd_lora as P

REPO = "andreayhchen/lls-filtering-data"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--steps", type=int, nargs="+", default=[1000, 2000, 3000, 4000, 5000])
    ap.add_argument("--n_batches", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    Path("minimality").mkdir(exist_ok=True)
    sft = torch.load(Path(a.data) / "sft_train.pt")
    for run in a.runs:
        targs = json.load(open(hf_hub_download(REPO, f"pd/lora/runs/{run}/train_args.json", repo_type="dataset")))
        targs["lora"] = hf_hub_download(REPO, "pd/lora/runs/goblin_plain_mix_p005/ckpt_3000.pt", repo_type="dataset",
                                        local_dir="/workspace/hfdl")
        args = types.SimpleNamespace(**targs)
        tok, model, wr, ci = P.build(args)
        T = args.ctx
        pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        gen = torch.Generator().manual_seed(a.seed)
        idx = [torch.randint(len(sft), (args.batch,), generator=gen).tolist() for _ in range(a.n_batches)]
        ac = torch.autocast(P.dev, dtype=torch.bfloat16, enabled=P.dev == "cuda")
        out = {"run": run, "batch": args.batch, "ctx": T, "n_batches": a.n_batches, "beta": args.beta, "points": []}
        for step in a.steps:
            ck = hf_hub_download(REPO, f"pd/lora/runs/{run}/ckpt_{step}.pt", repo_type="dataset")
            P.load_ckpt(ck, wr, ci)
            frac = (step - 1) / args.steps  # the checkpoint is saved after training step `step - 1`
            p = args.p_start + (args.p_end - args.p_start) * min(1.0, frac / args.anneal_end)
            lam = args.lam * min(1.0, frac / 0.1)
            vals, vals_p1 = [], []
            with torch.no_grad():
                for ix in idx:
                    xt, _, _ = G.sft_batch(sft, ix, T, pad)
                    with ac:
                        P.forward(model, wr, xt, "full")
                        ci_t = ci({n: w.last_input for n, w in wr.items()})
                        vals.append(G.minimality(ci_t, p, args.beta).item())
                        vals_p1.append(G.minimality(ci_t, 1.0, args.beta).item())
            l_min = sum(vals) / len(vals)
            out["points"].append({"step": step, "p": p, "lam": lam, "l_min": l_min, "lam_l_min": lam * l_min,
                                  "l_min_p1": sum(vals_p1) / len(vals_p1), "l_min_batches": vals})
            print(run, out["points"][-1] | {"l_min_batches": None}, flush=True)
        json.dump(out, open(f"minimality/{run}.json", "w"), indent=1)
        HfApi().upload_file(path_or_fileobj=f"minimality/{run}.json", path_in_repo=f"pd/lora/minimality/{run}.json",
                            repo_id=REPO, repo_type="dataset")
        print("pushed minimality", run, flush=True)
        del model, wr, ci
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
