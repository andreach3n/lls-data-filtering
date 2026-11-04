"""Within one run: how close is the model with only its high-gate components on to the model with all on?

For each run, on the same held-out tokens, compare the all-on model (mask = 1) with:
    ci      mask = g, the gate itself (the causal importance as a soft mask)
    imp     mask = 1[g > thr], only the important components, fully on
and, for scale and for necessity:
    base    no LoRA: KL(full || base) is the size of the whole fine-tune
    unimp   mask = 1[g <= thr], important removed: reported as KL(base || unimp), which should be ~0
Metrics: KL(P || Q) in nats per token and top-1 agreement, in three regions: held-out Dolci answer tokens, the
positions that predict an opening ** ("bold"), and held-out FineWeb chunks. Gates are recomputed per token from the
all-on model's layer inputs, as in training and in every other analysis.

  python kl_ci_compare.py --runs runs/vpd_lam1e-3_5k runs/nec_lam1e-3_5k --data /workspace/lora/data --out kl/ci_compare.json
Gated runs only (gated_lora.py).
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
PAIRS = [("full", "ci"), ("full", "imp"), ("full", "base"), ("base", "unimp")]


def ckpt_path(p):
    p = Path(p)
    ck = p if p.suffix == ".pt" else G.latest_ckpt(p)
    if ck is None or not ck.exists():
        sys.exit(f"no checkpoint at {p}")
    return ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--data", default="/workspace/lora/data")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n_docs", type=int, default=200)
    ap.add_argument("--n_nt", type=int, default=64)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--thr", type=float, default=0.1)
    a = ap.parse_args()

    paths = [ckpt_path(p) for p in a.runs]
    st0 = torch.load(paths[0], map_location="cpu")
    targs = argparse.Namespace(**st0["args"])
    targs.grad_ckpt, targs.teacher = 0, None
    del st0
    tok, model, wr, ci = G.build(targs)
    model.eval()
    ac = torch.autocast(G.dev, dtype=torch.bfloat16, enabled=G.dev == "cuda")
    T = targs.ctx
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    items = G.load_data(a.data, "val")[0][:a.n_docs]
    nt = torch.load(Path(a.data) / "nontarget_val.pt")[:a.n_nt]
    bolds = [G.bold_open_positions(tok, it["ids"][:T].tolist(), it["labels"][:T].tolist()) for it in items]

    def all_logits(ids):
        """logits under every condition, plus the important count per position"""
        with torch.no_grad(), ac:
            out = {"full": G.forward(model, wr, ids, "full")}
            g = {n: G.lower_leaky(z) for n, z in G.ci_of(ci, wr).items()}
            masks = {"ci": g, "imp": {n: (v > a.thr).to(v.dtype) for n, v in g.items()},
                     "unimp": {n: (v <= a.thr).to(v.dtype) for n, v in g.items()}}
            for k, m in masks.items():
                out[k] = G.forward(model, wr, ids, "masked", m)
            out["base"] = G.forward(model, wr, ids, "base")
            n_imp = sum((v > a.thr).float().sum(-1) for v in g.values())
        return out, n_imp

    results = {}
    for path in paths:
        name = path.parent.name
        st = torch.load(path, map_location="cpu")
        if st["args"].get("arch") == "pd":
            sys.exit(f"{name} is a post-hoc decomposition; not supported here")
        G.load_ckpt(path, wr, ci)
        kl = {r: {p: 0.0 for p in PAIRS} for r in REGIONS}
        agree = {r: {p: 0.0 for p in PAIRS} for r in REGIONS}
        cnt = {r: 0 for r in REGIONS}
        imp_sum = {r: 0.0 for r in REGIONS}

        def acc(region, lg, sel, n_imp):
            lps = {k: F.log_softmax(v[sel].float(), -1) for k, v in lg.items()}
            for p, q in PAIRS:
                kl[region][(p, q)] += (lps[p].exp() * (lps[p] - lps[q])).sum().item()
                agree[region][(p, q)] += (lps[p].argmax(-1) == lps[q].argmax(-1)).sum().item()
            cnt[region] += int(sel.sum())
            imp_sum[region] += n_imp[sel].sum().item()

        for b0 in range(0, len(items), a.batch):
            idx = list(range(b0, min(b0 + a.batch, len(items))))
            ids, labels, _ = G.sft_batch(items, idx, T, pad)
            ans = torch.zeros_like(labels, dtype=torch.bool)
            ans[:, :-1] = labels[:, 1:] != -100
            bold = torch.zeros_like(ans)
            for r, k in enumerate(idx):
                for j in bolds[k]:
                    bold[r, j - 1] = True
            lg, n_imp = all_logits(ids)
            acc("answer", lg, ans, n_imp)
            if bold.any():
                acc("bold", lg, bold, n_imp)
            del lg
        for b0 in range(0, len(nt), a.batch):
            xn = nt[b0:b0 + a.batch].to(G.dev)
            lg, n_imp = all_logits(xn)
            acc("fineweb", lg, torch.ones(xn.shape, dtype=torch.bool, device=G.dev), n_imp)
            del lg
        results[name] = {
            "ckpt": str(path), "positions": cnt,
            "important_per_token": {r: imp_sum[r] / max(1, cnt[r]) for r in REGIONS},
            "kl": {r: {f"{p}||{q}": kl[r][(p, q)] / max(1, cnt[r]) for p, q in PAIRS} for r in REGIONS},
            "top1_agree": {r: {f"{p}||{q}": agree[r][(p, q)] / max(1, cnt[r]) for p, q in PAIRS} for r in REGIONS}}
        print(f"done {name}", flush=True)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(a.out, "w"), indent=1)
    for r in REGIONS:
        print(f"\n[{r}]  KL in nats per token (top-1 agreement); important per token")
        print(f"{'run':28s}" + "".join(f" {p + '||' + q:>20s}" for p, q in PAIRS) + f" {'imp/token':>10s}")
        for name, res in results.items():
            print(f"{name:28s}" + "".join(f" {res['kl'][r][p + '||' + q]:12.4f} ({res['top1_agree'][r][p + '||' + q]:.3f})"
                                          for p, q in PAIRS) + f" {res['important_per_token'][r]:10.2f}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
