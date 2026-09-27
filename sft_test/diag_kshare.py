import re, numpy as np, pandas as pd
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
TB, TT = d.bold.sum(), d.n.sum()
print("Share of the corpus's BOLD captured by the top-k% tail, and the enrichment ratio")
print("(ratio = share of bold / share of tokens; 1.00 = no better than proportional)\n")
for k in (0.10, 0.25, 0.50):
    print(f"  --- top {int(k*100)}% of documents   (random would take {k*100:.0f}% of bold) ---")
    print(f"  {'alpha':>6s} {'% of bold':>10s} {'% of tokens':>12s} {'ratio':>7s} {'vs random':>10s}")
    n = int(len(d) * k)
    for a in (0.0, 0.25, 0.323, 0.5, 0.75, 1.0):
        w = d.w_raw / d.n ** a
        top = d.loc[w.nlargest(n).index]
        sb, sk = top.bold.sum() / TB, top.n.sum() / TT
        print(f"  {a:6.2f} {100*sb:9.1f}% {100*sk:11.1f}% {sb/sk:7.2f} {sb/k:9.2f}x")
    print()
# best case over a fine grid
best = []
for k in (0.10, 0.25):
    n = int(len(d) * k)
    for a in np.arange(-0.5, 1.51, 0.05):
        w = d.w_raw / d.n ** a
        top = d.loc[w.nlargest(n).index]
        best.append((float(sb := top.bold.sum() / TB) / k, k, float(a), 100*sb))
    b = max([x for x in best if x[1] == k])
    print(f"  best over alpha in [-0.5, 1.5] at k={int(k*100)}%: alpha {b[2]:+.2f} -> "
          f"{b[3]:.1f}% of bold = {b[0]:.2f}x random")
