#!/bin/bash
# GPU 1, phase 2 (written 2026-09-14 after the gate).
#   1. refusal    lls + lenmatch, --direction positive  (gate: redirect_cl 0.075 -> 0.199)
#   2. bothsides  lls + lenmatch, --direction NEGATIVE  (gate: band A count 0.500 -> 0.398,
#                 i.e. SFT SUPPRESSES it, so the tail to remove is the suppressing one)
# validate_feelings is dropped: it failed the gate (+0.020 band A narrow, +0.010 wide).
cd /root/lls/sft_test; source ./common_sft.sh
export CUDA_VISIBLE_DEVICES=1
log "GPU1-P2 START"

out=rm_refusal_k10_s0
log "START $out (lls,lenmatch, direction positive)"
python sft_lls.py --stage remove --trait refusal --arms lls,lenmatch \
  --no_removal_adapter controls/no_removal --ref_results controls/results.json \
  $COMMON --direction positive --out_dir ./$out > $out.log 2>&1
rc=$?; [ -f $out/results.json ] && log "DONE $out rc=$rc" || log "DIED $out rc=$rc"
cp $out.log PREREG_sft.md $out/ 2>/dev/null; push $out $out

out=rm_bothsides_k10_s0_neg
log "START $out (lls,lenmatch, direction NEGATIVE)"
python sft_lls.py --stage remove --trait bothsides --arms lls,lenmatch \
  --no_removal_adapter controls/no_removal --ref_results controls/results.json \
  $COMMON --direction negative --out_dir ./$out > $out.log 2>&1
rc=$?; [ -f $out/results.json ] && log "DONE $out rc=$rc" || log "DIED $out rc=$rc"
cp $out.log $out/ 2>/dev/null; push $out $out
log "GPU1-P2 COMPLETE"
