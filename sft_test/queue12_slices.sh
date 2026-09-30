#!/bin/bash
# SPECIFICITY TEST: train on the MIDDLE and BOTTOM deciles of the LLS ranking and compare
# against the top decile and random, all at k=10% and alpha=1.0 (the log-linearity paper's
# own normalisation, and the setting every removal arm used).
#
#   GPU=0 ARM=middle ./queue12_slices.sh
#   GPU=1 ARM=bottom ./queue12_slices.sh
#
# Already measured at this exact setting (k=10%, alpha=1.0, 38 steps, 2,386 docs):
#   top decile    27.84
#   random        23.22 and 24.06   <- SAME documents, two runs: the evaluation-noise yardstick
#   (prompt-clustered bootstrap SE per arm is ~1.2-1.4, so a gap needs ~1.8 for one sigma)
#
# What the slices contain (bold per 100 words, x corpus; median answer tokens):
#   top     3.28 (1.22x)   99 tok   43% of docs contain any bold
#   random  2.76 (1.03x)  316 tok   59%
#   middle  2.56 (0.95x)  495 tok   69%
#   bottom  2.47 (0.92x)  134 tok   45%
#
# PRE-REGISTERED READING. Only the top slice is bold-enriched at this alpha, so:
#   middle ~ bottom ~ random (23-24)  => the top decile is genuinely special, not "any
#                                        non-random slice teaches formatting".
#   bottom ~ top (27-28)              => the SIGN of the score carries no information and
#                                        only |w| matters -- a real finding about the method.
#   middle > random                   => something other than bold density is driving it.
#
# The bottom decile needs no new arm type: --direction negative makes `tailonly` select the
# most-negative w. Both arms reuse the existing random control, so no control run is needed.
cd /workspace/lls/sft_test; source ./common_sft.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
ARM=${ARM:-middle}
case $ARM in
  middle) ARMS=middleonly; DIR=positive ;;
  bottom) ARMS=tailonly;   DIR=negative ;;
  *) echo "ARM must be middle or bottom"; exit 1 ;;
esac
out=slice_${ARM}_k10
log "SLICE START $ARM (arms=$ARMS direction=$DIR) on GPU $CUDA_VISIBLE_DEVICES -> $out"

# every override AFTER $COMMON: argparse takes the last occurrence and $COMMON carries its own
python sft_lls.py --stage remove --trait bold --arms $ARMS \
  --ref_results controls/results.json \
  $COMMON --remove_frac 0.10 --alpha 1.0 --direction $DIR --out_dir ./$out > $out.log 2>&1
rc=$?
[ -f $out/results.json ] && log "DONE $out rc=$rc" || log "DIED $out rc=$rc"
grep -q "alpha=1.0 " $out.log || log "WARNING $out did NOT run at alpha=1.0"
grep -q "direction=$DIR" $out.log || log "WARNING $out did NOT run at direction=$DIR"
push $out $out
log "SLICE COMPLETE $ARM"
