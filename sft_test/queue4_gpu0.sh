#!/bin/bash
# GPU 0, phase 4 (2026-09-15): ALL BOLD. Waits for the running k=10% lls+lenmatch job,
# then removes the ANTI-bold tail.
#
#   dropneg  --direction negative at k=10%: remove the documents the bold persona makes
#            LEAST likely. Tests the asymmetry found at 1B and 7B on DPO -- "removal
#            reveals/restores what the remainder supports; it cannot subtract". Control is
#            the k=10 random arm already trained in ./controls, so this is one run, not two.
cd /root/lls/sft_test; source ./common_sft.sh
export CUDA_VISIBLE_DEVICES=0
log "GPU0-P4 START (waiting for the running k=10 bold job)"
while pgrep -f "trait bol[d] --arms lls,lenmatch" > /dev/null; do sleep 180; done
log "GPU0-P4 k=10 bold finished"

out=rm_bold_dropneg_k10_s0
log "START $out (lls, direction NEGATIVE = remove the anti-bold tail)"
python sft_lls.py --stage remove --trait bold --arms lls \
  --no_removal_adapter controls/no_removal --ref_results controls/results.json \
  --eval "random_a1.0_k10=controls/random_a1.0_k10" \
  $COMMON --direction negative --out_dir ./$out > $out.log 2>&1
[ -f $out/results.json ] && log "DONE $out" || log "DIED $out"; push $out $out

# backfill: the bold arms now also evaluate the held-out refusal_hard set, but base and
# no_removal were evaluated before it existed. ~25 min each, so the tables line up.
log "START refusal_hard backfill for base + no_removal"
python sft_lls.py --stage eval --eval "base,no_removal=controls/no_removal" \
  --psets refusal_hard $COMMON --out_dir ./controls > backfill_refusal_hard.log 2>&1
log "DONE refusal_hard backfill"; push controls controls
log "GPU0-P4 COMPLETE"
