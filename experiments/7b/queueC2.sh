#!/bin/bash
# POD C phase 2: wait for k=10 job -> bothsides tail-alone gate -> bothsides removal seed 1, both directions
cd /root/lls; source ./common7b.sh
until grep -q "QUEUE_COMPLETE" queue_status.log; do sleep 120; done
mkdir -p b7_score_bs_full
hf download $REPO --repo-type dataset --include "removal/b7_score_bs_full/scores.parquet" --local-dir ./hfpull >> push.log 2>&1
cp hfpull/removal/b7_score_bs_full/scores.parquet b7_score_bs_full/scores.parquet
python -c "import pandas as pd; a=pd.read_parquet('b7_score_bs_full/scores.parquet'); b=pd.read_parquet('score_bold_full_1b.parquet'); assert (a.prompt.values==b.prompt.values).all(); print('bs parquet aligned')" >> push.log 2>&1 || { log "ABORT: bs parquet missing/misaligned"; exit 1; }
BS="--scores b7_score_bs_full/scores.parquet --alpha 1.0 --remove_frac 0.25 --skip_ref_eval --extra_evals bothsides $COMMON"
run() { # run <out> <extra args>
  out=$1; shift; log "START $out"
  LLS_TRAIT=bothsides python lls_owl.py --stage remove $BS "$@" --out_dir ./$out > $out.log 2>&1
  rc=$?; if ls $out/removal_a1.0_k25.json >/dev/null 2>&1; then log "DONE $out rc=$rc"; else log "DIED $out rc=$rc"; fi
  cp $out.log PREREG_7b_bs_gate_s1.md $out/ 2>/dev/null; push $out $out
}
# gate: with direction=positive, drivers=top so antionly=BOTTOM tail; with direction=negative, antionly=TOP tail
run b7_gate_bs_bottom --direction positive --arms antionly,randomonly --seed 0 --train_seed 42
run b7_gate_bs_top    --direction negative --arms antionly            --seed 0 --train_seed 42
# second removal seed, both directions, one shared random control (random draw depends only on --seed)
run b7_bs_droppos_a1_s1 --direction positive --arms lls,random --seed 1 --train_seed 43
run b7_bs_dropneg_a1_s1 --direction negative --arms lls        --seed 1 --train_seed 43
log "QUEUE2_COMPLETE"
