#!/bin/bash
# GPU 1, phase 4 (2026-09-15): ALL BOLD. Refusal and bothsides are deferred -- refusal's
# gate failed on the over-refusal prompt set and the held-out `refusal_hard` set has not
# been validated yet, so bold is the only behaviour with a confirmed large shift.
#
#   1. edit        their Evidence 3: strip every bold/header/bullet marker from all 23,860
#                  training answers, keep every document. Comparator = no_removal
#                  (identical size, step count, LR schedule -> no size confound at all).
#   2. lls  k=25%  their second removal threshold.
#   3. random k=25% the size-matched control for (2). Useless without it, so adjacent.
cd /root/lls/sft_test; source ./common_sft.sh
export CUDA_VISIBLE_DEVICES=1
log "GPU1-P4 START (all bold)"

out=rm_bold_edit
log "START $out (edit / de-bolding, comparator no_removal)"
python sft_lls.py --stage remove --trait bold --arms edit \
  --no_removal_adapter controls/no_removal --ref_results controls/results.json \
  $COMMON --direction positive --out_dir ./$out > $out.log 2>&1
[ -f $out/results.json ] && log "DONE $out" || log "DIED $out"; push $out $out

out=rm_bold_k25_s0
log "START $out (lls,random at k=25%)"
python sft_lls.py --stage remove --trait bold --arms lls,random \
  --no_removal_adapter controls/no_removal --ref_results controls/results.json \
  $COMMON --direction positive --remove_frac 0.25 --out_dir ./$out > $out.log 2>&1
[ -f $out/results.json ] && log "DONE $out" || log "DIED $out"; push $out $out
log "GPU1-P4 COMPLETE"
