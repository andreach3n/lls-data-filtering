#!/bin/bash
# GPU 0, phase 2 (written 2026-09-14 after the gate).
#   waits for the running `random` control to finish, then:
#   1. bold      lls + lenmatch, --direction positive   (gate: bold_cl 15.27 -> 22.83)
#   2. refusal   the `source` arm only                  (its lls/lenmatch run on GPU 1)
cd /root/lls/sft_test; source ./common_sft.sh
export CUDA_VISIBLE_DEVICES=0
log "GPU0-P2 START (waiting for the random control)"
while pgrep -f "arms rando[m]" > /dev/null; do sleep 120; done
log "GPU0-P2 random finished"

out=rm_bold_k10_s0
log "START $out (lls,lenmatch, direction positive)"
python sft_lls.py --stage remove --trait bold --arms lls,lenmatch \
  --no_removal_adapter controls/no_removal --ref_results controls/results.json \
  --eval "random_a1.0_k10=controls/random_a1.0_k10" $COMMON --direction positive --out_dir ./$out > $out.log 2>&1
rc=$?; [ -f $out/results.json ] && log "DONE $out rc=$rc" || log "DIED $out rc=$rc"
cp $out.log PREREG_sft.md $out/ 2>/dev/null; push $out $out

# the source-filter baseline for refusal, in its own dir so it cannot race GPU 1's
# results.json for the same trait; --stage report merges the two at the end.
out=rm_refusal_source_k10_s0
log "START $out (source)"
python sft_lls.py --stage remove --trait refusal --arms source \
  --no_removal_adapter controls/no_removal --ref_results controls/results.json \
  $COMMON --direction positive --out_dir ./$out > $out.log 2>&1
rc=$?; [ -f $out/results.json ] && log "DONE $out rc=$rc" || log "DIED $out rc=$rc"
cp $out.log $out/ 2>/dev/null; push $out $out
log "GPU0-P2 COMPLETE"
