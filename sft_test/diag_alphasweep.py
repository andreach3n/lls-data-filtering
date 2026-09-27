import re, numpy as np, pandas as pd
from scipy.stats import spearmanr
BOLD = re.compile(r"\*\*[^*\n]+\*\*")
STRUCT = re.compile(r"^#{1,6}\s|^\s*[-*]\s", re.MULTILINE)
THINK = "</think>"

df = pd.read_parquet("split_s0_mix.parquet")
base = pd.read_parquet("scores_s0/lp_base.parquet").set_index("id")
tr = pd.read_parquet("scores_s0/lp_bold.parquet").set_index("id")
d = df.set_index("id").join(base.add_prefix("b_")).join(tr.add_prefix("t_"))
d["w_raw"] = d.t_lp_answer - d.b_lp_answer
d["n_tok"] = d.t_n_answer.astype(float)
d = d[d.n_tok > 0]

def fmt(msgs):
    ans = " ".join((m["content"] or "").split(THINK)[-1] for m in msgs if m["role"] == "assistant")
    w = max(len(ans.split()), 1)
    return len(BOLD.findall(ans)), len(STRUCT.findall(ans)), w
st = d.messages.map(lambda ms: fmt([dict(m) for m in ms]))
d["bold"] = [x[0] for x in st]; d["struct"] = [x[1] for x in st]; d["words"] = [x[2] for x in st]
d["bold100"] = d.bold / d.words * 100
d["fmt100"] = (d.bold + d.struct) / d.words * 100

print("How the alpha exponent changes WHAT gets selected (w = w_raw / n_tok**alpha)\n")
print(f"  {'alpha':>6s} {'rho(w,bold100)':>15s} {'rho(w,fmt100)':>14s} {'rho(w,n_tok)':>13s} "
      f"| top-10% tail: {'median n_tok':>12s} {'bold/100w':>10s} {'%any bold':>10s}")
corpus_b100 = d.bold100.mean(); corpus_any = (d.bold > 0).mean()
for a in (0.0, 0.25, 0.37, 0.5, 0.75, 1.0, 1.25):
    w = d.w_raw / d.n_tok ** a
    n = int(len(d) * 0.10)
    top = d.loc[w.nlargest(n).index]
    print(f"  {a:6.2f} {spearmanr(w, d.bold100).statistic:+15.3f} {spearmanr(w, d.fmt100).statistic:+14.3f} "
          f"{spearmanr(w, d.n_tok).statistic:+13.3f} | {top.n_tok.median():24.0f} "
          f"{top.bold100.mean():10.2f} {(top.bold > 0).mean():10.2f}")
print(f"\n  corpus reference: bold/100w {corpus_b100:.2f}, %any bold {corpus_any:.2f}, "
      f"median n_tok {d.n_tok.median():.0f}")

print("\nBoth tails are short -> variance inflation, not a length gradient (alpha=1):")
w1 = d.w_raw / d.n_tok
for lab, idx in (("top 10%", w1.nlargest(2386).index), ("mid 80%", w1.sort_values().index[2386:-2386]),
                 ("bottom 10%", w1.nsmallest(2386).index)):
    s = d.loc[idx]
    print(f"  {lab:12s} median n_tok {s.n_tok.median():5.0f}  sd(w) {(w1.loc[idx]).std():.5f}")
