"""Does training as a decomposition change the fine-tune? KL between the all-on (mask = 1) models of several runs.

Every run is loaded with all components on (an ordinary LoRA = the sum of its components). For every ordered pair
(P, Q) of models, plus the untouched base model, on the same held-out tokens:
    KL(P || Q)      mean over positions of sum_v P(v) (log P(v) - log Q(v))
    top-1 agree     share of positions where both models' most likely next token is the same
in three regions:
    answer    held-out Dolci answer tokens (the fine-tuning task)
    bold      the positions that predict an opening ** (where bold is decided)
    fineweb   held-out FineWeb chunks (unrelated text)
Also per model: answer CE and mean log P(open bold), as a check against the training logs.

How to read it: compare KL(ordinary LoRA || VPD LoRA) with the KL between two ordinary LoRAs (e.g. another learning
rate, or another seed if one is trained) and with KL(ordinary LoRA || base), which is the size of the whole fine-tune.

  python kl_full_compare.py --runs runs/plain_lr2e-4_5k runs/vpd_lam1e-3_5k runs/nec_lam1e-3_5k \
      runs/nec_lam1e-3_5k_seed1 runs/plain_lr1e-4_5k runs/plain_c32 --data /workspace/lora/data --out kl/full_compare.json
The first run is the reference in the printed summary. Each --runs entry is a run directory (its latest ckpt_*.pt is
used) or a .pt file. Gated runs only (gated_lora.py); post-hoc runs are their original LoRA when all on.
"""
import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
import gated_lora as G  # noqa: E402

REGIONS = ("answer", "bold", "fineweb")
ARCH_KEYS = ("model", "C", "proj", "layers")


def ckpt_path(p):
    p = Path(p)
    ck = p if p.suffix == ".pt" else G.latest_ckpt(p)
    if ck is None or not ck.exists():
        sys.exit(f"no checkpoint at {p}")
    return ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run directories or .pt files; the first is the reference")
    ap.add_argument("--data", default="/workspace/lora/data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_docs", type=int, default=200, help="held-out Dolci documents")
    ap.add_argument("--n_nt", type=int, default=64, help="held-out FineWeb chunks")
    ap.add_argument("--batch", type=int, default=2)
    a = ap.parse_args()

    paths = [ckpt_path(p) for p in a.runs]
    names = [p.parent.name if p.parent.name != "." else p.stem for p in paths]
    sts = [torch.load(p, map_location="cpu") for p in paths]
    for n, st in zip(names, sts):
        if st["args"].get("arch") == "pd":
            sys.exit(f"{n} is a post-hoc decomposition; all on it is its original LoRA, pass that run instead")
        bad = [k for k in ARCH_KEYS if st["args"].get(k) != sts[0]["args"].get(k)]
        if bad:
            sys.exit(f"{n} differs from {names[0]} in {bad}")
    targs = argparse.Namespace(**sts[0]["args"])
    targs.grad_ckpt, targs.teacher = 0, None
    tok, model, wr, _ = G.build(targs)
    model.eval()
    comps = {n: {m: {k: v.to(G.dev) for k, v in st["components"][m].items()} for m in wr} for n, st in zip(names, sts)}
    del sts
    models = ["base"] + names
    M = len(models)

    def logits_of(model_name, ids):
        if model_name == "base":
            G.set_mode(wr, "base")
        else:
            with torch.no_grad():
                for m, w in wr.items():
                    w.U.copy_(comps[model_name][m]["U"])
                    w.V.copy_(comps[model_name][m]["V"])
            G.set_mode(wr, "full")
        with torch.no_grad(), torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda"):
            return model(input_ids=ids).logits

    kl = {r: torch.zeros(M, M, dtype=torch.float64) for r in REGIONS}
    agree = {r: torch.zeros(M, M, dtype=torch.float64) for r in REGIONS}
    count = {r: 0 for r in REGIONS}
    ce_sum, lpb_sum = torch.zeros(M, dtype=torch.float64), torch.zeros(M, dtype=torch.float64)

    def accumulate(region, lps):
        """lps: list over models of (P, V) log-probs at the same P positions"""
        top = [lp.argmax(-1) for lp in lps]
        for i in range(M):
            p = lps[i].exp()
            for j in range(M):
                if i != j:
                    kl[region][i, j] += (p * (lps[i] - lps[j])).sum().item()
                    agree[region][i, j] += (top[i] == top[j]).sum().item()
        count[region] += lps[0].shape[0]

    T = targs.ctx
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    items = G.load_data(a.data, "val")[0][:a.n_docs]
    for b0 in range(0, len(items), a.batch):
        idx = list(range(b0, min(b0 + a.batch, len(items))))
        ids, labels, _ = G.sft_batch(items, idx, T, pad)
        ans = torch.zeros_like(labels, dtype=torch.bool)
        ans[:, :-1] = labels[:, 1:] != -100  # logits at t predict token t + 1
        bold = torch.zeros_like(ans)
        for r, k in enumerate(idx):
            for j in G.bold_open_positions(tok, items[k]["ids"][:T].tolist(), items[k]["labels"][:T].tolist()):
                bold[r, j - 1] = True
        tgt_ans = labels[:, 1:][ans[:, :-1]]
        tgt_bold = ids[:, 1:][bold[:, :-1]]
        lp_ans, lp_bold = [], []
        for mi, mn in enumerate(models):
            lg = logits_of(mn, ids)
            lp_ans.append(F.log_softmax(lg[ans].float(), -1))
            lp_bold.append(F.log_softmax(lg[bold].float(), -1))
            del lg
            ce_sum[mi] += -lp_ans[-1].gather(-1, tgt_ans[:, None]).sum().item()
            lpb_sum[mi] += lp_bold[-1].gather(-1, tgt_bold[:, None]).sum().item()
        accumulate("answer", lp_ans)
        if lp_bold[0].shape[0]:
            accumulate("bold", lp_bold)
        del lp_ans, lp_bold
        print(f"docs {idx[-1] + 1}/{len(items)}", flush=True)

    nt = torch.load(Path(a.data) / "nontarget_val.pt")[:a.n_nt]
    for b0 in range(0, len(nt), a.batch):
        xn = nt[b0:b0 + a.batch].to(G.dev)
        lps = []
        for mn in models:
            lg = logits_of(mn, xn)
            lps.append(F.log_softmax(lg.flatten(0, 1).float(), -1))
            del lg
        accumulate("fineweb", lps)
        del lps
    print(f"fineweb chunks {len(nt)}", flush=True)

    res = {"models": models, "ckpts": {n: str(p) for n, p in zip(names, paths)}, "n_docs": len(items), "n_nt": len(nt),
           "positions": count,
           "answer_ce": {m: ce_sum[i].item() / count["answer"] for i, m in enumerate(models)},
           "logp_open_bold": {m: lpb_sum[i].item() / max(1, count["bold"]) for i, m in enumerate(models)},
           "kl": {r: {mi: {mj: kl[r][i, j].item() / max(1, count[r]) for j, mj in enumerate(models) if j != i}
                      for i, mi in enumerate(models)} for r in REGIONS},
           "top1_agree": {r: {mi: {mj: agree[r][i, j].item() / max(1, count[r]) for j, mj in enumerate(models) if j != i}
                              for i, mi in enumerate(models)} for r in REGIONS}}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)

    ref = names[0]
    print(f"\npositions: {count}")
    print(f"\n{'model':28s} {'answer CE':>10s} {'logP bold':>10s}")
    for m in models:
        print(f"{m:28s} {res['answer_ce'][m]:10.4f} {res['logp_open_bold'][m]:10.4f}")
    print(f"\nKL({ref} || model), nats per token, and top-1 agreement")
    print(f"{'model':28s}" + "".join(f" {r + ' KL':>12s} {'agree':>6s}" for r in REGIONS))
    for m in models:
        if m != ref:
            print(f"{m:28s}" + "".join(f" {res['kl'][r][ref][m]:12.5f} {res['top1_agree'][r][ref][m]:6.3f}" for r in REGIONS))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
