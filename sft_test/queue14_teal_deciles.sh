#!/bin/bash
# BOLD-MINUS-TEAL DECILE SWEEP + PROMPT SCREEN (4 GPUs, 2026-09-30).
#
#   GPU=0 GROUP=1 bash queue14_teal_deciles.sh   # deciles 0, 40, 70
#   GPU=1 GROUP=2 bash queue14_teal_deciles.sh   # deciles 10, 30, 90
#   GPU=2 GROUP=3 bash queue14_teal_deciles.sh   # score `plain`, then deciles 20, 60
#   GPU=3 GROUP=4 bash queue14_teal_deciles.sh   # score `bold_only`, then deciles 50, 80
#
# The queue13 sweep again, but ranked by lp_bold - lp_teal (--contrast teal) instead of
# lp_bold - lp_base. Teal is the matched unrelated persona scored in the same Sep 14 run as
# bold; subtracting it cancels what any persona instruction does to the likelihood. On the
# data (no training) it was the only variant that improved BOTH ends of the ranking:
#   bold - base,  alpha 1.0: top 1.22x corpus bold, bottom 0.92x   (queue13)
#   bold - teal,  alpha 1.0: top 1.37x,             bottom 0.91x   (this sweep)
# All ten deciles are new (no decile of this ranking has been trained). The random reference
# is unchanged -- same documents, checked: the random draw ignores the score -- so every bar
# is read against the same alpha=1.0 random evaluation (24.06) as queue13.
#
# Deciles are grouped by median answer length (training time scales with it), so the two
# GPUs that also score get the lighter pairs:  0:156 10:320 20:451 30:506 40:508 50:452
# 60:393 70:343 80:246 90:130 tokens.
#
# SCORING: the two new prompts in lls_owl TRAITS (`bold_only`, `plain`), same command as
# queue_gpu1.sh's original pass, into the same scores_s0 dir so lp_base is reused. Each is
# pushed to HF the moment it finishes, before any training starts on that GPU.
cd /workspace/lls/sft_test; source ./common_sft.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
GROUP=${GROUP:-1}
SCORE=""
case $GROUP in
  1) ARMS=slice0,slice40,slice70 ;;
  2) ARMS=slice10,slice30,slice90 ;;
  3) ARMS=slice20,slice60; SCORE=plain ;;
  4) ARMS=slice50,slice80; SCORE=bold_only ;;
  *) echo "GROUP must be 1-4"; exit 1 ;;
esac
log "Q14 START $GROUP (score=${SCORE:-none} arms=$ARMS) on GPU $CUDA_VISIBLE_DEVICES"

if [ -n "$SCORE" ] && [ ! -f $SCORES/lp_${SCORE}.parquet ]; then
  [ -f $SCORES/lp_base.parquet ] || { log "NO lp_base in $SCORES -- would rescore base; abort"; exit 1; }
  log "START score $SCORE"
  python sft_lls.py --stage score --trait $SCORE --split $SPLIT --out_dir $SCORES \
    --score_dtype bfloat16 --token_budget 16384 > score_${SCORE}.log 2>&1
  if [ -f $SCORES/lp_${SCORE}.parquet ]; then
    log "DONE score $SCORE"
    hf upload $REPO ./$SCORES/lp_${SCORE}.parquet sft/$SCORES/lp_${SCORE}.parquet \
      --repo-type dataset >> push.log 2>&1 && log "PUSHED lp_$SCORE" || log "PUSH_FAILED lp_$SCORE"
  else
    log "DIED score $SCORE"   # the deciles below do not depend on it, so carry on
  fi
fi

out=teal_deciles_g${GROUP}_k10
# every override AFTER $COMMON: argparse takes the last occurrence and $COMMON carries its own
python sft_lls.py --stage remove --trait bold --contrast teal --arms $ARMS \
  --ref_results controls/results.json \
  $COMMON --remove_frac 0.10 --alpha 1.0 --direction positive --out_dir ./$out > $out.log 2>&1
rc=$?
[ -f $out/results.json ] && log "DONE $out rc=$rc" || log "DIED $out rc=$rc"
grep -q "alpha=1.0 " $out.log || log "WARNING $out did NOT run at alpha=1.0"
grep -q "contrast=teal" $out.log || log "WARNING $out did NOT run with contrast=teal"
for a in ${ARMS//,/ }; do
  grep -q "  $a: rows" $out.log || log "WARNING $out has no density line for $a"
done
push $out $out
hf upload $REPO ./$out.log sft/$out/$out.log --repo-type dataset >> push.log 2>&1
log "Q14 COMPLETE $GROUP"
