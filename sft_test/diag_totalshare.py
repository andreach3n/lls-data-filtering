"""Average vs TOTAL: the right-hand panel measured bold DENSITY in the removed tail, but the
argument for alpha=0 is about a document's TOTAL contribution (SFT weights each document by
its token count). So also measure the share of all the corpus's bold markers that the
removed 10% carries -- the 'total' view."""
import json, re, numpy as np, pandas as pd
from pathlib import Path
BOLD = re.compile(r"\*\*[^*\n]+\*\*"); THINK = "</think>"
df = pd.read_parquet("split_s0_mix.parquet")
base = pd.read_parquet("scores_s0/lp_base.parquet").set_index("id")
tr = pd.read_parquet("scores_s0/lp_bold.parquet").set_index("id")
def fmt(msgs):
    ans = " ".join((m["content"] or "").split(THINK)[-1] for m in msgs if m["role"] == "assistant")
    return len(BOLD.findall(ans)), max(len(ans.split()), 1)
st = df.messages.map(lambda ms: fmt([dict(m) for m in ms]))
df = df.assign(bold=[x[0] for x in st], words=[x[1] for x in st])
d = df.set_index("id").join(base.add_prefix("b_")).join(tr.add_prefix("t_"))
d["w_raw"] = d.t_lp_answer - d.b_lp_answer; d["n"] = d.t_n_answer.astype(float)
d = d[d.n > 1]
tot_bold = d.bold.sum(); tot_tok = d.n.sum(); k = int(len(d) * 0.10)
out = []
for a in np.arange(0.0, 1.31, 0.05):
    w = d.w_raw / d.n ** a
    top = d.loc[w.nlargest(k).index]
    out.append(dict(alpha=float(a),
                    bold100=float((top.bold / top.words * 100).mean()),
                    bold_per_doc=float(top.bold.mean()),
                    share_bold=float(top.bold.sum() / tot_bold),
                    share_tokens=float(top.n.sum() / tot_tok),
                    n_med=float(top.n.median())))
j = json.loads(Path("alpha_fit.json").read_text()); j["sweep_total"] = out
j["corpus_totals"] = dict(bold_per_doc=float(d.bold.mean()), docs=int(len(d)))
Path("alpha_fit.json").write_text(json.dumps(j, indent=1))
print(f"  {'alpha':>6s} {'bold/100w':>10s} {'bold/doc':>9s} {'% of corpus bold':>17s} {'% of tokens':>12s}")
for r in out:
    if round(r["alpha"] * 100) % 25 == 0 or abs(r["alpha"] - 0.35) < 0.001:
        print(f"  {r['alpha']:6.2f} {r['bold100']:10.2f} {r['bold_per_doc']:9.2f} "
              f"{100*r['share_bold']:16.1f}% {100*r['share_tokens']:11.1f}%")
print(f"\n  a random 10% would take 10.0% of the bold and 10.0% of the tokens")
