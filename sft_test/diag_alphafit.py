"""Fit alpha empirically: if sd(w_raw) ~ N**beta then dividing by N**beta equalises the
variance across lengths, so beta IS the right exponent. Regress log sd on log N over
quantile bins (HANDOFF.md 'Fit its own alpha')."""
import json, re, numpy as np, pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download

BOLD = re.compile(r"\*\*[^*\n]+\*\*"); STRUCT = re.compile(r"^#{1,6}\s|^\s*[-*]\s", re.MULTILINE)
THINK = "</think>"
TRAITS = ["bold", "validate_feelings", "bothsides", "refusal", "teal"]
NBINS = 15

df = pd.read_parquet("split_s0_mix.parquet")
base = pd.read_parquet("scores_s0/lp_base.parquet").set_index("id")

def fmt(msgs):
    ans = " ".join((m["content"] or "").split(THINK)[-1] for m in msgs if m["role"] == "assistant")
    w = max(len(ans.split()), 1)
    return len(BOLD.findall(ans)), len(STRUCT.findall(ans)), w
st = df.messages.map(lambda ms: fmt([dict(m) for m in ms]))
df = df.assign(bold=[x[0] for x in st], struct=[x[1] for x in st], words=[x[2] for x in st])
df["bold100"] = df.bold / df.words * 100

out = {"nbins": NBINS, "traits": {}}
for t in TRAITS:
    p = f"scores_s0/lp_{t}.parquet"
    if not Path(p).exists():
        p = hf_hub_download("andreayhchen/lls-filtering-data", f"sft/scores_s0/lp_{t}.parquet",
                            repo_type="dataset")
    tr = pd.read_parquet(p).set_index("id")
    d = df.set_index("id").join(base.add_prefix("b_")).join(tr.add_prefix("t_"))
    d["w_raw"] = d.t_lp_answer - d.b_lp_answer
    d["n"] = d.t_n_answer.astype(float)
    d = d[d.n > 1]
    d["bin"] = pd.qcut(d.n, NBINS, labels=False, duplicates="drop")
    g = d.groupby("bin").agg(n_med=("n", "median"), sd=("w_raw", "std"),
                             mu=("w_raw", "mean"), cnt=("w_raw", "size"))
    x, y = np.log(g.n_med.values), np.log(g.sd.values)
    slope, inter = np.polyfit(x, y, 1)
    pred = slope * x + inter
    r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    # linearity: fit the short and long halves separately
    h = len(x) // 2
    s_lo = np.polyfit(x[:h], y[:h], 1)[0]; s_hi = np.polyfit(x[h:], y[h:], 1)[0]
    rec = dict(alpha=float(slope), r2=float(r2), slope_short=float(s_lo), slope_long=float(s_hi),
               bins=[dict(n=float(a), sd=float(b), mu=float(c), cnt=int(e))
                     for a, b, c, e in zip(g.n_med, g.sd, g.mu, g.cnt)])
    # what the fitted exponent selects, vs alpha=1
    if t == "bold":
        for a_lab, a in (("fitted", slope), ("1.0", 1.0), ("0.0", 0.0)):
            w = d.w_raw / d.n ** a
            top = d.loc[w.nlargest(int(len(d) * 0.10)).index]
            rec[f"tail_{a_lab}"] = dict(alpha=float(a), n_med=float(top.n.median()),
                                        bold100=float(top.bold100.mean()),
                                        any_bold=float((top.bold > 0).mean()))
        rec["corpus"] = dict(n_med=float(d.n.median()), bold100=float(d.bold100.mean()),
                             any_bold=float((d.bold > 0).mean()))
    out["traits"][t] = rec
    print(f"  {t:18s} alpha_hat {slope:+.3f}  R2 {r2:.3f}  "
          f"(short half {s_lo:+.3f}, long half {s_hi:+.3f})")
Path("alpha_fit.json").write_text(json.dumps(out, indent=1))
print("\nwrote alpha_fit.json")
b = out["traits"]["bold"]
print(f"\n  bold tail composition (top 10%), corpus: n_tok {b['corpus']['n_med']:.0f}, "
      f"bold/100w {b['corpus']['bold100']:.2f}, any bold {b['corpus']['any_bold']:.2f}")
for k in ("tail_fitted", "tail_1.0", "tail_0.0"):
    v = b[k]
    print(f"    alpha {v['alpha']:+.3f}: n_tok {v['n_med']:5.0f}  bold/100w {v['bold100']:5.2f}  "
          f"any bold {v['any_bold']:.2f}")
