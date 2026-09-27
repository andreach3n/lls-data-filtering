#!/bin/bash
# POD B: 7B bold scoring -> alignment check vs 1B rows -> push -> lls arm
cd /root/lls; source ./common7b.sh
log "START b7_score_bold_full (batch 8)"
python lls_owl.py --stage score --out_dir ./b7_score_bold_full --n_subsample 20000 --seed 0 --resp_trunc 0 --score_dtype bf16 --score_batch 8 --block 5000 > b7_score.log 2>&1
if [ ! -f b7_score_bold_full/scores.parquet ]; then
  log "score batch 8 exited without parquet -> retrying with batch 4 (shards resume)"
  python lls_owl.py --stage score --out_dir ./b7_score_bold_full --n_subsample 20000 --seed 0 --resp_trunc 0 --score_dtype bf16 --score_batch 4 --block 5000 >> b7_score.log 2>&1
fi
if [ ! -f b7_score_bold_full/scores.parquet ]; then log "DIED b7_score_bold_full"; log "QUEUE_ABORTED"; exit 1; fi
log "DONE b7_score_bold_full"
python - <<'PY' >> b7_score.log 2>&1
import pandas as pd, sys
a = pd.read_parquet("b7_score_bold_full/scores.parquet").reset_index(drop=True)
b = pd.read_parquet("score_bold_full_1b.parquet").reset_index(drop=True)
ok = len(a)==len(b) and (a.prompt.values==b.prompt.values).all() and (a.chosen.values==b.chosen.values).all() and (a.n_tok.values==b.n_tok.values).all()
print("ALIGNMENT 7B vs 1B rows:", "OK" if ok else "MISMATCH", len(a), len(b))
from scipy.stats import spearmanr
print("spearman(w7b, w1b) alpha=1:", round(spearmanr(a.w_raw/a.n_tok, b.w_raw/b.n_tok).correlation, 3))
sys.exit(0 if ok else 3)
PY
if [ $? -ne 0 ]; then log "ALIGN_FAIL: 7B rows differ from 1B rows -- lls arm NOT launched"; log "QUEUE_ABORTED"; exit 1; fi
log "ALIGN_OK"
log "START noise check (200 rows, batch 1 vs batch 8)"
python - <<'PY' >> b7_score.log 2>&1
import os, sys, numpy as np, pandas as pd, torch
sys.argv=["x"]; import lls_owl as L
from transformers import AutoModelForCausalLM, AutoTokenizer
df = pd.read_parquet("b7_score_bold_full/scores.parquet").reset_index(drop=True)
L.RESP_TRUNC = 0
tok = AutoTokenizer.from_pretrained(L.MODEL_ID); tok.pad_token = tok.pad_token or tok.eos_token
model = AutoModelForCausalLM.from_pretrained(L.MODEL_ID, torch_dtype=torch.bfloat16).to("cuda").eval()
sub = df.sample(n=200, random_state=0)
rows = [{"prompt":r.prompt,"chosen":r.chosen,"rejected":r.rejected} for r in sub.itertuples()]
b1 = L.compute_w(model, tok, rows, 1)
w8 = (sub.w_raw / sub.n_tok).values; w1 = np.array([r["w_raw"] for r in b1]) / sub.n_tok.values
d = np.abs(w8 - w1); sd = (df.w_raw/df.n_tok).std()
print(f"NOISE alpha=1: median|dw|={np.median(d):.2e} p95={np.percentile(d,95):.2e} max={d.max():.2e} vs corpus sd(w)={sd:.2e} -> p95/sd={np.percentile(d,95)/sd:.1%}; spearman(b8,b1)={pd.Series(w8).corr(pd.Series(w1),method='spearman'):.4f}")
PY
cp b7_score.log b7_score_bold_full/; push b7_score_bold_full b7_score_bold_full
log "START b7_drop_a1_s0_lls"
python lls_owl.py --stage remove --scores b7_score_bold_full/scores.parquet --alpha 1.0 --remove_frac 0.25 --direction negative \
  --arms lls --seed 0 --train_seed 42 --skip_ref_eval $COMMON --out_dir ./b7_drop_a1_s0_lls > b7_drop_a1_s0_lls.log 2>&1
rc=$?; if [ -f b7_drop_a1_s0_lls/removal_a1.0_k25.json ]; then log "DONE b7_drop_a1_s0_lls rc=$rc"; else log "DIED b7_drop_a1_s0_lls rc=$rc"; fi
cp b7_drop_a1_s0_lls.log b7_drop_a1_s0_lls/ 2>/dev/null; push b7_drop_a1_s0_lls b7_drop_a1_s0_lls
log "QUEUE_COMPLETE"
