#!/bin/bash
# GPU 1: scoring, then validate_feelings, then refusal.
#   1. score     lp_base + lp_{bold,validate_feelings,bothsides,refusal,teal}   (~12 h)
#                bold is scored FIRST so GPU 0 can start its arms as early as possible
#   2. analyze   tails: composition, length, content/source overlap (no GPU, seconds)
#   3. vf        lls + lenmatch                                                 (~12 h)
#   4. refusal   lls + lenmatch + source                                        (~18 h)
cd /root/lls/sft_test; source ./common_sft.sh
export CUDA_VISIBLE_DEVICES=1
log "GPU1 START"

# ---- 1. score (one pass per trait; lp_base is shared) ----------------------
for tr in bold validate_feelings bothsides refusal teal; do
  [ -f $SCORES/lp_${tr}.parquet ] && continue
  log "START score $tr"
  python sft_lls.py --stage score --trait $tr --split $SPLIT --out_dir $SCORES \
    --score_dtype bfloat16 --token_budget 16384 >> score.log 2>&1
  [ -f $SCORES/lp_${tr}.parquet ] && log "DONE score $tr" || { log "DIED score $tr"; exit 1; }
done
hf upload $REPO ./$SCORES sft/$SCORES --repo-type dataset --include "lp_*.parquet" >> push.log 2>&1

# ---- 2. analyze (no GPU) ---------------------------------------------------
python sft_lls.py --stage analyze --trait bold,validate_feelings,bothsides,refusal,teal \
  $COMMON > analyze.log 2>&1
cp analyze.log $SCORES/ 2>/dev/null; log "DONE analyze"

# ---- 3 & 4. trait arms -----------------------------------------------------
for tr in validate_feelings refusal; do
  out=rm_${tr}_k10_s0
  arms="lls,lenmatch"; [ $tr = refusal ] && arms="lls,lenmatch,source"
  log "START $out ($arms)"
  python sft_lls.py --stage remove --trait $tr --arms $arms \
    --no_removal_adapter controls/no_removal --ref_results controls/results.json \
    $COMMON --out_dir ./$out > $out.log 2>&1
  rc=$?; [ -f $out/results.json ] && log "DONE $out rc=$rc" || log "DIED $out rc=$rc"
  cp $out.log PREREG_sft.md $out/ 2>/dev/null; push $out $out
done
log "GPU1 COMPLETE"
