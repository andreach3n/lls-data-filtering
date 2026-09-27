#!/bin/bash
# POD B phase 2: wait for lenmatch -> score bothsides -> bs_droppos -> gate (base+baseline on VF+bs sets) -> random re-eval on VF+bs sets
cd /root/lls; source ./common7b.sh
until grep -q QUEUE2_COMPLETE queue_status.log; do sleep 120; done
log "START b7_score_bs_full"
LLS_TRAIT=bothsides python lls_owl.py --stage score --out_dir ./b7_score_bs_full --n_subsample 20000 --seed 0 --resp_trunc 0 --score_dtype bf16 --score_batch 8 --block 5000 > b7_score_bs.log 2>&1
[ -f b7_score_bs_full/scores.parquet ] || { log "DIED b7_score_bs_full"; log "QUEUE3_ABORTED"; exit 1; }
python - <<'PY' >> b7_score_bs.log 2>&1
import pandas as pd; from scipy.stats import spearmanr
a=pd.read_parquet("b7_score_bs_full/scores.parquet"); b=pd.read_parquet("score_bold_full_1b.parquet")
print("ALIGNMENT bs7b vs 1b rows:", "OK" if (a.prompt.values==b.prompt.values).all() else "MISMATCH")
c=pd.read_parquet("b7_score_bold_full/scores.parquet"); print("spearman(w_bs7b, w_bold7b) alpha=1:", round(spearmanr(a.w_raw/a.n_tok, c.w_raw/c.n_tok).correlation,3))
PY
log "DONE b7_score_bs_full"; cp b7_score_bs.log PREREG_7b_vf_bs.md b7_score_bs_full/; push b7_score_bs_full b7_score_bs_full
out=b7_bs_droppos_a1_s0; log "START $out"
LLS_TRAIT=bothsides python lls_owl.py --stage remove --scores b7_score_bs_full/scores.parquet --alpha 1.0 --remove_frac 0.25 --direction positive \
  --arms lls --seed 0 --train_seed 42 --skip_ref_eval --extra_evals bothsides $COMMON --out_dir ./$out > $out.log 2>&1
rc=$?; if [ -f $out/removal_a1.0_k25.json ]; then log "DONE $out rc=$rc"; else log "DIED $out rc=$rc"; fi
cp $out.log PREREG_7b_vf_bs.md $out/ 2>/dev/null; push $out $out
# gate: base + full-corpus baseline adapter (from pod A via HF) on both extra sets
log "START b7_gate_vf_bs"
hf download $REPO --repo-type dataset --include "removal/b7_baseline/baseline/*" --local-dir ./hfpull >> push.log 2>&1
python lls_owl.py --stage remove --scores b7_score_bold_full/scores.parquet --alpha 1.0 --remove_frac 0.25 --direction negative \
  --arms none --baseline_adapter hfpull/removal/b7_baseline/baseline --extra_evals validate_feelings,bothsides $COMMON --out_dir ./b7_gate_vf_bs > b7_gate_vf_bs.log 2>&1
rc=$?; if [ -f b7_gate_vf_bs/removal_a1.0_k25.json ]; then log "DONE b7_gate_vf_bs rc=$rc"; else log "DIED b7_gate_vf_bs rc=$rc"; fi
cp b7_gate_vf_bs.log b7_gate_vf_bs/ 2>/dev/null; push b7_gate_vf_bs b7_gate_vf_bs
# random control re-eval: pull the seed-0 random adapter from pod A (pushed at the end of its first queue)
log "START b7_drop_a1_s0_random_x (re-eval on VF+bs sets)"
for i in $(seq 1 60); do hf download $REPO --repo-type dataset --include "removal/b7_drop_a1_s0_random/random_a1.0_k25/*" --local-dir ./hfpull >> push.log 2>&1 && [ -f hfpull/removal/b7_drop_a1_s0_random/random_a1.0_k25/adapter_model.safetensors ] && break; sleep 300; done
mkdir -p b7_drop_a1_s0_random_x && cp -r hfpull/removal/b7_drop_a1_s0_random/random_a1.0_k25 b7_drop_a1_s0_random_x/
python lls_owl.py --stage remove --scores b7_score_bold_full/scores.parquet --alpha 1.0 --remove_frac 0.25 --direction negative \
  --arms random --seed 0 --train_seed 42 --skip_ref_eval --extra_evals validate_feelings,bothsides $COMMON --out_dir ./b7_drop_a1_s0_random_x > b7_drop_a1_s0_random_x.log 2>&1
rc=$?; if [ -f b7_drop_a1_s0_random_x/removal_a1.0_k25.json ]; then log "DONE b7_drop_a1_s0_random_x rc=$rc"; else log "DIED b7_drop_a1_s0_random_x rc=$rc"; fi
cp b7_drop_a1_s0_random_x.log b7_drop_a1_s0_random_x/ 2>/dev/null; push b7_drop_a1_s0_random_x b7_drop_a1_s0_random_x
log "QUEUE3_COMPLETE"
