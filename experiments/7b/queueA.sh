#!/bin/bash
# POD A: 7B baseline (transmit gate) -> random arm
cd /root/lls; source ./common7b.sh
log "START b7_baseline"
python lls_owl.py --stage baseline --out_dir ./b7_baseline --n_subsample 20000 --seed 0 $COMMON > b7_baseline.log 2>&1
rc=$?; if [ -f b7_baseline/baseline_traits.json ]; then log "DONE b7_baseline rc=$rc"; else log "DIED b7_baseline rc=$rc"; fi
cp b7_baseline.log b7_baseline/ 2>/dev/null; push b7_baseline b7_baseline
# random arm: rows come from the 1B parquet (identical 20k rows; alignment verified on pod B), scores irrelevant for random
log "START b7_drop_a1_s0_random"
python lls_owl.py --stage remove --scores score_bold_full_1b.parquet --alpha 1.0 --remove_frac 0.25 --direction negative \
  --arms random --seed 0 --train_seed 42 --skip_ref_eval $COMMON --out_dir ./b7_drop_a1_s0_random > b7_drop_a1_s0_random.log 2>&1
rc=$?; if [ -f b7_drop_a1_s0_random/removal_a1.0_k25.json ]; then log "DONE b7_drop_a1_s0_random rc=$rc"; else log "DIED b7_drop_a1_s0_random rc=$rc"; fi
cp b7_drop_a1_s0_random.log b7_drop_a1_s0_random/ 2>/dev/null; push b7_drop_a1_s0_random b7_drop_a1_s0_random
log "QUEUE_COMPLETE"
