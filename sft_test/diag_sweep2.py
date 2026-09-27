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
d = d[d.n > 1]; d["bold100"] = d.bold / d.words * 100
k = int(len(d) * 0.10)
sweep = []
for a in np.arange(0.0, 1.31, 0.05):
    w = d.w_raw / d.n ** a
    top = d.loc[w.nlargest(k).index]
    sweep.append(dict(alpha=float(a), bold100=float(top.bold100.mean()),
                      n_med=float(top.n.median()), any_bold=float((top.bold > 0).mean())))
j = json.loads(Path("alpha_fit.json").read_text())
j["sweep_bold"] = sweep
j["corpus_bold"] = dict(bold100=float(d.bold100.mean()), n_med=float(d.n.median()),
                        any_bold=float((d.bold > 0).mean()))
Path("alpha_fit.json").write_text(json.dumps(j, indent=1))
print("sweep points:", len(sweep))
