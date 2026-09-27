#!/bin/bash
# Bold deep dive, phase 3. Queue AFTER rm_bold_k10_s0 finishes; each arm is ~7.3 h
# train + ~1 h eval. Pick a card with GPU=0 or GPU=1.
#
#   1. edit    their Evidence 3: strip every bold/header/bullet marker from the training
#              answers, keep all 23,860 documents. Comparator is no_removal -- identical
#              size, step count and LR schedule, so this arm has NO size confound at all.
#              Their result: de-bolding did not reduce the behaviour.
#   2. k=25%   their second threshold: lls + random, dose-response against k=10%.
#   3. dropneg remove the ANTI-bold tail at k=10%. Tests the asymmetry from the 1B/7B DPO
#              work ("removal reveals what the remainder supports; it cannot subtract").
#              Its control is the k=10 random arm already trained.
cd /root/lls/sft_test; source ./common_sft.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
log "BOLD-DEEP START on GPU $CUDA_VISIBLE_DEVICES"

out=rm_bold_edit
log "START $out (edit)"
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

out=rm_bold_dropneg_k10_s0
log "START $out (lls, direction NEGATIVE = remove the anti-bold tail)"
python sft_lls.py --stage remove --trait bold --arms lls \
  --no_removal_adapter controls/no_removal --ref_results controls/results.json \
  --eval "random_a1.0_k10=controls/random_a1.0_k10" \
  $COMMON --direction negative --out_dir ./$out > $out.log 2>&1
[ -f $out/results.json ] && log "DONE $out" || log "DIED $out"; push $out $out
log "BOLD-DEEP COMPLETE"
