#!/bin/bash
# POD A phase 2: wait for random arm -> score VF -> vf_droppos -> vf_dropneg
cd /root/lls; source ./common7b.sh
until grep -q QUEUE_COMPLETE queue_status.log; do sleep 120; done
log "START b7_score_vf_full"
LLS_TRAIT=validate_feelings python lls_owl.py --stage score --out_dir ./b7_score_vf_full --n_subsample 20000 --seed 0 --resp_trunc 0 --score_dtype bf16 --score_batch 8 --block 5000 > b7_score_vf.log 2>&1
[ -f b7_score_vf_full/scores.parquet ] || { log "DIED b7_score_vf_full"; log "QUEUE3_ABORTED"; exit 1; }
python - <<'PY' >> b7_score_vf.log 2>&1
import pandas as pd; from scipy.stats import spearmanr
a=pd.read_parquet("b7_score_vf_full/scores.parquet"); b=pd.read_parquet("score_bold_full_1b.parquet")
print("ALIGNMENT vf7b vs 1b rows:", "OK" if (a.prompt.values==b.prompt.values).all() else "MISMATCH")
c=pd.read_parquet("b7_score_bold_full/scores.parquet"); print("spearman(w_vf7b, w_bold7b) alpha=1:", round(spearmanr(a.w_raw/a.n_tok, c.w_raw/c.n_tok).correlation,3))
PY
log "DONE b7_score_vf_full"; cp b7_score_vf.log PREREG_7b_vf_bs.md b7_score_vf_full/; push b7_score_vf_full b7_score_vf_full
for dir in positive negative; do
  tag=$([ $dir = positive ] && echo pos || echo neg); out=b7_vf_drop${tag}_a1_s0
  log "START $out"
  LLS_TRAIT=validate_feelings python lls_owl.py --stage remove --scores b7_score_vf_full/scores.parquet --alpha 1.0 --remove_frac 0.25 --direction $dir \
    --arms lls --seed 0 --train_seed 42 --skip_ref_eval --extra_evals validate_feelings $COMMON --out_dir ./$out > $out.log 2>&1
  rc=$?; if [ -f $out/removal_a1.0_k25.json ]; then log "DONE $out rc=$rc"; else log "DIED $out rc=$rc"; fi
  cp $out.log PREREG_7b_vf_bs.md $out/ 2>/dev/null; push $out $out
done
log "QUEUE3_COMPLETE"
