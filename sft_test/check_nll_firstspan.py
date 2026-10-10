"""Decision-point version of check_nll_pairs: at the FIRST position where the bold and
de-bolded answers diverge (the prefix is identical up to there), compare the next-token
log-probs of the two continuations under each model:

    d1 = log P(bold's next token)  -  log P(plain's next token)      e.g. P("**") vs P("Habit")

This removes the downstream confound in the summed version (once "**" is open, the rest of a
bold span is highly predictable, which inflates delta for every model). Usage as check_nll_pairs.
"""
import argparse, json, re, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import sft_lls as S  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pairs", required=True); ap.add_argument("--models", required=True)
ap.add_argument("--contexts", default="think,none"); ap.add_argument("--out_dir", default="nll_firstspan")
args = ap.parse_args()
RB = re.compile(r"\*\*[^*\n]+\*\*")
rows = [json.loads(l) for l in open(args.pairs)]
pairs = []
for r in rows:
    if not (r.get("think_close", 1) and RB.search(r["text"])): continue
    raw = r["text_raw"]; j = raw.rfind(S.THINK_CLOSE)
    pairs.append(dict(gen_id=r["gen_id"], prompt=r["prompt"], think=raw[:j].strip() if j >= 0 else "",
                      bold=r["text"].strip(), plain=S.strip_formatting(r["text"].strip(), "bold").strip()))
tok = S.load_tok()
# every token that is a bold opener: "**" and its space-prefixed form "Ġ**" (172/197 pairs use the latter)
STARS = [i for t, i in tok.get_vocab().items() if t.lstrip("Ġ") == "**"]

def ids_for(p, ctx, v):
    ans = p[v]; content = f"<think>\n{p['think']}\n</think>\n\n{ans}" if ctx == "think" else ans
    return S.encode(tok, [{"role": "user", "content": p["prompt"]}, {"role": "assistant", "content": content}])[0]

@torch.no_grad()
def next_logprobs(model, prefix):
    out = model(input_ids=torch.tensor([prefix], device=S.DEVICE), use_cache=False).logits[0, -1].float()
    return torch.log_softmax(out, -1)

base = S.load_base(); results = {}; per = {}
out_dir = Path(args.out_dir); out_dir.mkdir(exist_ok=True)
for spec in args.models.split(","):
    label, _, path = spec.partition("=")
    model = base
    if label != "base":
        from peft import PeftModel
        model = PeftModel.from_pretrained(base, path).eval()
    t0 = time.time()
    for c in args.contexts.split(","):
        rec = []
        for p in pairs:
            b, q = ids_for(p, c, "bold"), ids_for(p, c, "plain")
            k = next(i for i in range(min(len(b), len(q))) if b[i] != q[i])
            lp = next_logprobs(model, b[:k])
            rec.append(dict(gen_id=p["gen_id"], prompt=p["prompt"], k=k,
                            bold_tok=tok.convert_ids_to_tokens(b[k]), plain_tok=tok.convert_ids_to_tokens(q[k]),
                            lp_bold=float(lp[b[k]]), lp_plain=float(lp[q[k]]), lp_star=float(torch.logsumexp(lp[STARS], 0)),
                            p_star=float(torch.logsumexp(lp[STARS], 0).exp())))
        d = np.array([x["lp_bold"] - x["lp_plain"] for x in rec]); ps = np.array([x["p_star"] for x in rec])
        results[f"{label}/{c}"] = dict(n=len(rec), d1_mean=float(d.mean()), d1_median=float(np.median(d)),
                                      frac_prefers_bold=float((d > 0).mean()), p_star_mean=float(ps.mean()),
                                      p_star_median=float(np.median(ps)), frac_star_over_half=float((ps > 0.5).mean()))
        per[(label, c)] = rec
        with open(out_dir / f"first_{label}_{c}.jsonl", "w") as f:
            for x in rec: f.write(json.dumps(x) + "\n")
        print(f"  [{label}/{c}] d1 mean {d.mean():6.2f} median {np.median(d):6.2f}  prefers bold {100*(d>0).mean():5.1f}%  "
              f"P(**) mean {ps.mean():.3f} median {np.median(ps):.3f}  P(**)>0.5: {100*(ps>0.5).mean():5.1f}%", flush=True)
    print(f"  [{label}] done in {time.time()-t0:.0f}s", flush=True)
    if label != "base": base = model.unload()

def paired(a, b, key, draws=4000):
    byp = {}
    for x, y in zip(a, b): byp.setdefault(x["prompt"], []).append(key(x) - key(y))
    P = list(byp); rng = np.random.default_rng(0)
    return np.mean([v for p in P for v in byp[p]]), float(np.std([np.mean([v for p in rng.choice(P, len(P)) for v in byp[p]]) for _ in range(draws)]))
labels = [s.partition("=")[0] for s in args.models.split(",")]
print("\n=== first-divergence decision point: d1 = logP(bold token) - logP(plain token); P(**) = prob of opening bold there ===")
print("  " + "model/context".ljust(26) + f"{'mean d1':>9}{'median':>9}{'% prefers bold':>16}{'mean P(**)':>12}{'median P(**)':>14}{'P(**)>.5':>10}")
for k, r in results.items():
    print("  " + k.ljust(26) + f"{r['d1_mean']:9.2f}{r['d1_median']:9.2f}{100*r['frac_prefers_bold']:16.1f}{r['p_star_mean']:12.3f}{r['p_star_median']:14.3f}{100*r['frac_star_over_half']:10.1f}")
print("\n  SFT-vs-base shift, +/- prompt-bootstrap SE:")
for lab in labels:
    if lab == "base": continue
    for c in args.contexts.split(","):
        m, se = paired(per[(lab, c)], per[("base", c)], lambda x: x["lp_bold"] - x["lp_plain"])
        m2, se2 = paired(per[(lab, c)], per[("base", c)], lambda x: x["p_star"])
        results[f"shift/{lab}/{c}"] = dict(d1=m, d1_se=se, p_star=m2, p_star_se=se2)
        print(f"    {lab:18s} {c:5s}: d1 {m:6.2f} +/- {se:.2f}   P(**) {m2:+.3f} +/- {se2:.3f}")
(out_dir / "results.json").write_text(json.dumps(results, indent=1))
