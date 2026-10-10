"""Likelihood test for bold: does each model prefer the bold or the de-bolded version of the
SAME answer?  (pod, ~10 min on one card)

Why: generation-based base numbers depend on the prefix (think vs none) and on junk/drift
rules. This sidesteps all of it: take the SFT model's own answers (>=1 bold span), strip the
** markers (strip_formatting mode="bold", the debold arm's edit), and score both versions
under each model with the same context. Per pair:

    delta = log P(bold answer) - log P(stripped answer)      summed over answer tokens

and the headline is delta under the SFT model minus delta under the base (prompt-paired
bootstrap SE). Two contexts, matching the two framings in the prefix debate:
    think  : the SFT's own thinking trace precedes the answer (base in the Think persona)
    none   : the answer follows <|im_start|>assistant\\n directly (base in its default mode)

Usage (from sft_test/, env from common_sft.sh):
  python check_nll_pairs.py --pairs <generations_no_removal_general.jsonl> \
      --models base,no_removal=<dir>,random_a1.0_k10=<dir> [--contexts think,none]
"""
import argparse, json, re, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import sft_lls as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", required=True, help="generations_*_general.jsonl whose answers are the pair source")
ap.add_argument("--models", required=True, help="comma list: base, label=adapter_dir, ...")
ap.add_argument("--contexts", default="think,none")
ap.add_argument("--max_pairs", type=int, default=0)
ap.add_argument("--token_budget", type=int, default=16384)
ap.add_argument("--out_dir", default="nll_pairs")
args = ap.parse_args()

RB = re.compile(r"\*\*[^*\n]+\*\*")
rows = [json.loads(l) for l in open(args.pairs)]
src = [r for r in rows if r.get("think_close", 1) and RB.search(r["text"])]
if args.max_pairs:
    src = src[:args.max_pairs]
pairs = []
for r in src:
    raw = r["text_raw"]; j = raw.rfind(S.THINK_CLOSE)
    think = raw[:j].strip() if j >= 0 else ""
    bold = r["text"].strip(); plain = S.strip_formatting(bold, "bold").strip()
    assert "**" not in plain and bold != plain
    pairs.append(dict(gen_id=r["gen_id"], prompt=r["prompt"], think=think, bold=bold, plain=plain,
                      n_spans=len(RB.findall(bold))))
print(f"{len(pairs)} pairs from {args.pairs} (median {int(np.median([p['n_spans'] for p in pairs]))} bold spans)")

tok = S.load_tok()


def content(p, ctx, version):
    ans = p[version]
    if ctx == "think":
        return f"<think>\n{p['think']}\n</think>\n\n{ans}"
    return ans                                   # none: answer right after "<|im_start|>assistant\n"


def encode_all(ctx, version):
    return [S.encode(tok, [{"role": "user", "content": p["prompt"]},
                           {"role": "assistant", "content": content(p, ctx, version)}]) for p in pairs]


out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
results, per_pair = {}, {}
contexts = [c.strip() for c in args.contexts.split(",")]
enc = {(c, v): encode_all(c, v) for c in contexts for v in ("bold", "plain")}
for (c, v), e in enc.items():
    n = [x[1].count("answer") for x in e]
    print(f"  {c:5s}/{v:5s}: answer tokens median {int(np.median(n))}, max {max(n)}")

base = S.load_base()
for spec in args.models.split(","):
    label, _, path = spec.partition("=")
    model = base
    if label != "base":
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, path).eval()
    t0 = time.time()
    for c in contexts:
        lp = {v: S.span_logprobs(model, tok, enc[(c, v)], args.token_budget, verbose=False) for v in ("bold", "plain")}
        rec = []
        for i, p in enumerate(pairs):
            b, q = lp["bold"][i], lp["plain"][i]
            rec.append(dict(gen_id=p["gen_id"], prompt=p["prompt"], n_spans=p["n_spans"],
                            lp_bold=b["lp_answer"], n_bold=b["n_answer"], lp_plain=q["lp_answer"], n_plain=q["n_answer"],
                            delta=b["lp_answer"] - q["lp_answer"],
                            nll_tok_bold=-b["lp_answer"] / b["n_answer"], nll_tok_plain=-q["lp_answer"] / q["n_answer"]))
        per_pair[(label, c)] = rec
        d = np.array([x["delta"] for x in rec])
        results[f"{label}/{c}"] = dict(n=len(rec), delta_mean=float(d.mean()), delta_median=float(np.median(d)),
                                      frac_prefers_bold=float((d > 0).mean()),
                                      delta_per_span=float((d / np.array([x["n_spans"] for x in rec])).mean()),
                                      nll_tok_bold=float(np.mean([x["nll_tok_bold"] for x in rec])),
                                      nll_tok_plain=float(np.mean([x["nll_tok_plain"] for x in rec])))
        with open(out_dir / f"pairs_{label}_{c}.jsonl", "w") as f:
            for x in rec: f.write(json.dumps(x) + "\n")
        print(f"  [{label}/{c}] delta mean {d.mean():8.2f}  median {np.median(d):8.2f}  prefers bold {100*(d>0).mean():5.1f}%  "
              f"nll/tok bold {results[f'{label}/{c}']['nll_tok_bold']:.3f} plain {results[f'{label}/{c}']['nll_tok_plain']:.3f}", flush=True)
    print(f"  [{label}] done in {time.time() - t0:.0f}s", flush=True)
    if label != "base":
        model = model.unload() if hasattr(model, "unload") else model   # drop adapter weights, keep base
        base = model


def paired_diff(a, b, draws=4000, seed=0):
    """mean(delta_a - delta_b) with a prompt-clustered bootstrap SE"""
    byp = {}
    for x, y in zip(a, b):
        byp.setdefault(x["prompt"], []).append(x["delta"] - y["delta"])
    P = list(byp); rng = np.random.default_rng(seed)
    m = np.mean([v for p in P for v in byp[p]])
    bs = [np.mean([v for p in rng.choice(P, len(P)) for v in byp[p]]) for _ in range(draws)]
    return m, float(np.std(bs))


labels = [s.partition("=")[0] for s in args.models.split(",")]
print("\n=== bold-vs-plain log-likelihood difference, delta = logP(bold) - logP(plain), nats over the answer ===")
print("  " + "model/context".ljust(26) + f"{'mean delta':>11}{'median':>9}{'% prefers bold':>16}{'delta/span':>12}{'nll/tok bold':>14}{'nll/tok plain':>15}")
for k, r in results.items():
    print("  " + k.ljust(26) + f"{r['delta_mean']:11.2f}{r['delta_median']:9.2f}{100*r['frac_prefers_bold']:16.1f}{r['delta_per_span']:12.3f}{r['nll_tok_bold']:14.3f}{r['nll_tok_plain']:15.3f}")
if "base" in labels:
    print("\n  SFT-vs-base shift in delta (positive = SFT prefers bold more than base), +/- prompt-bootstrap SE:")
    for lab in labels:
        if lab == "base": continue
        for c in contexts:
            m, se = paired_diff(per_pair[(lab, c)], per_pair[("base", c)])
            results[f"shift/{lab}/{c}"] = dict(mean=m, se=se)
            print(f"    {lab:18s} {c:5s}: {m:8.2f} +/- {se:.2f}")
(out_dir / "results.json").write_text(json.dumps(results, indent=1))
print(f"\nwrote {out_dir}/results.json and per-pair jsonl")
