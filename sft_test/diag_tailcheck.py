import re, numpy as np, pandas as pd
BOLD = re.compile(r"\*\*[^*\n]+\*\*")
STRUCT = re.compile(r"^#{1,6}\s|^\s*[-*]\s", re.MULTILINE)
THINK = "</think>"

df = pd.read_parquet("split_s0_mix.parquet")
base = pd.read_parquet("scores_s0/lp_base.parquet").set_index("id")
tr = pd.read_parquet("scores_s0/lp_bold.parquet").set_index("id")
d = df.set_index("id").join(base.add_prefix("b_")).join(tr.add_prefix("t_"))
d["w_raw"] = d.t_lp_answer - d.b_lp_answer
d["n_tok"] = d.t_n_answer
d = d[d.n_tok > 0]
d["w"] = d.w_raw / d.n_tok.astype(float)

def fmt(msgs):
    ans = " ".join((m["content"] or "").split(THINK)[-1] for m in msgs if m["role"] == "assistant")
    w = max(len(ans.split()), 1)
    return len(BOLD.findall(ans)), len(STRUCT.findall(ans)), w

stats = d.messages.map(lambda ms: fmt([dict(m) for m in ms]))
d["bold"] = [x[0] for x in stats]; d["struct"] = [x[1] for x in stats]; d["words"] = [x[2] for x in stats]
d["bold_per100w"] = d.bold / d.words * 100
d["struct_per100w"] = d.struct / d.words * 100

print("Does the top-w tail actually contain more formatting than the corpus?\n")
print(f"  {'group':22s} {'n':>6s} {'bold/doc':>9s} {'bold/100w':>10s} {'struct/doc':>11s} {'%any bold':>10s} {'words':>7s}")
def row(name, sub):
    print(f"  {name:22s} {len(sub):6d} {sub.bold.mean():9.2f} {sub.bold_per100w.mean():10.2f} "
          f"{sub.struct.mean():11.2f} {(sub.bold > 0).mean():10.2f} {sub.words.mean():7.0f}")
row("whole corpus", d)
for k in (0.10, 0.25):
    n = int(len(d) * k)
    top = d.nlargest(n, "w"); bot = d.nsmallest(n, "w")
    row(f"top {int(k*100)}% by w (REMOVED)", top)
    row(f"bottom {int(k*100)}% by w", bot)
print()
# rank correlation between the score and actual formatting
from scipy.stats import spearmanr
for col in ("bold", "bold_per100w", "struct", "struct_per100w", "words"):
    r = spearmanr(d.w, d[col]).statistic
    print(f"  spearman(w, {col:16s}) = {r:+.3f}")
