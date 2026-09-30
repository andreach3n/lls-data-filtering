#!/bin/bash
# DECILE SWEEP: queue12_slices.sh run down the whole LLS ranking. Train on each 10% block of
# the ranking alone (k=10%, alpha=1.0, 38 steps, 2,386 docs) and ask whether bold falls off
# smoothly with rank or only the top block is special.
#
#   GPU=0 GROUP=1 ./queue13_deciles.sh    # slice10, slice20
#   GPU=1 GROUP=2 ./queue13_deciles.sh    # slice30, slice40
#   GPU=2 GROUP=3 ./queue13_deciles.sh    # slice50, slice60
#   GPU=3 GROUP=4 ./queue13_deciles.sh    # slice70, slice80
#
# sliceNN = the block starting NN% down from the HIGHEST score (build_arms). The two ends are
# already measured and are NOT rerun -- unittest T4 checks they are the same documents:
#   slice0  == top decile     27.84   (tailonly, direction positive)
#   slice90 == bottom decile  22.92   (slice_bottom_k10)
#   middleonly (45-55%)       24.08   is off the grid; slice40 and slice50 bracket it
#   random                    23.64   (pooled, SE 0.89; a gap needs ~1.8 for one sigma)
#
# PRE-REGISTERED READING:
#   monotone decline from slice0, e.g. slice10 > slice20 > ... ~ random
#                          => LLS rank carries graded information; top-heavy but not a cliff.
#   slice10 onward all ~ random (23-24)
#                          => only the extreme top block matters (a threshold effect).
#   a mid-ranking decile > random
#                          => something other than rank-ordered bold density is driving it
#                             (check that decile's printed bold density in the log).
#   With single runs at ~1.5 SE per difference, adjacent deciles are NOT distinguishable from
#   each other; read the SHAPE (trend across 10 points), not individual pairs.
#
# Each process trains its two arms back to back, then evaluates both in one pass
# (train_one and eval_models push to HF from inside python, so a dropped pod loses at most
# the arm in progress). ~4 h per GPU on 4 cards.
cd /workspace/lls/sft_test; source ./common_sft.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
GROUP=${GROUP:-1}
case $GROUP in
  1) ARMS=slice10,slice20 ;;
  2) ARMS=slice30,slice40 ;;
  3) ARMS=slice50,slice60 ;;
  4) ARMS=slice70,slice80 ;;
  *) echo "GROUP must be 1-4"; exit 1 ;;
esac
out=deciles_g${GROUP}_k10
log "DECILES START $GROUP (arms=$ARMS) on GPU $CUDA_VISIBLE_DEVICES -> $out"

# every override AFTER $COMMON: argparse takes the last occurrence and $COMMON carries its own.
# --direction positive explicitly: sliceNN ignores it, but the tail parquet written alongside
# should be the top decile, as in every other positive-direction run.
python sft_lls.py --stage remove --trait bold --arms $ARMS \
  --ref_results controls/results.json \
  $COMMON --remove_frac 0.10 --alpha 1.0 --direction positive --out_dir ./$out > $out.log 2>&1
rc=$?
[ -f $out/results.json ] && log "DONE $out rc=$rc" || log "DIED $out rc=$rc"
grep -q "alpha=1.0 " $out.log || log "WARNING $out did NOT run at alpha=1.0"
for a in ${ARMS//,/ }; do
  grep -q "  $a: rows" $out.log || log "WARNING $out has no density line for $a"
done
push $out $out
log "DECILES COMPLETE $GROUP"
