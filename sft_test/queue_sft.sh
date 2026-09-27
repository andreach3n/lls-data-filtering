#!/bin/bash
# SFT-LLS filtering test, pod queue. Order = cheapest gate first, so a null at the gate
# stops the queue before any scoring or removal run is paid for.
#
#   0. split      one 25K stratified split of Dolci-Think-SFT-7B (CPU + network, ~30-60 min)
#   1. gate       train no_removal (full split), eval base + no_removal      -- does SFT shift the traits?
#   2. score      lp_base once + lp_{bold,validate_feelings,bothsides,refusal,teal}   (5+1 passes)
#   3. analyze    tail composition / length / content overlap / cross-trait overlap   (no GPU)
#   4. remove     per trait: lls + lenmatch (+ source for refusal); random is trait-independent
#                 and trained ONCE under bold, then evaluated on every trait's prompt sets
#
# Everything is resumable: adapters, lp_*.parquet and results.json are skipped if present.
cd /root/lls/sft_test; source ./common_sft.sh
until grep -q INSTALL_DONE ../install.log; do sleep 30; done

# ---- 0. split --------------------------------------------------------------
if [ ! -f $SPLIT ]; then
  log "START split"
  python build_split.py --n 25000 --seed 0 --out $SPLIT > split.log 2>&1 || { log "DIED split"; exit 1; }
  log "DONE split"; hf upload $REPO $SPLIT sft/$SPLIT --repo-type dataset >> push.log 2>&1
fi

# ---- 1. gate ---------------------------------------------------------------
# no_removal is the full 25K; its adapter is reused by every trait's remove stage.
out=gate_s0
log "START $out"
python sft_lls.py --stage gate $COMMON --out_dir ./$out > $out.log 2>&1
rc=$?; [ -f $out/results.json ] && log "DONE $out rc=$rc" || { log "DIED $out rc=$rc"; exit 1; }
push $out $out

# ---- 2. score --------------------------------------------------------------
log "START score"
python sft_lls.py --stage score --trait bold,validate_feelings,bothsides,refusal,teal \
  --split $SPLIT --out_dir $SCORES --score_dtype bfloat16 --token_budget 16384 --hf_repo $REPO > score.log 2>&1
rc=$?; [ -f $SCORES/lp_teal.parquet ] && log "DONE score rc=$rc" || { log "DIED score rc=$rc"; exit 1; }

# ---- 3. analyze (no GPU) ---------------------------------------------------
python sft_lls.py --stage analyze --trait bold,validate_feelings,bothsides,refusal,teal $COMMON > analyze.log 2>&1
cp analyze.log $SCORES/; push $SCORES $SCORES

# ---- 4. remove, one trait at a time ----------------------------------------
# random is trained once (under bold) and then passed to the other traits by path.
for tr in bold validate_feelings bothsides refusal; do
  out=rm_${tr}_k10_s0
  arms="lls,lenmatch"; extra=""
  [ $tr = bold ] && arms="lls,random,lenmatch"
  [ $tr = refusal ] && arms="lls,lenmatch,source"
  [ $tr != bold ] && extra="random_a1.0_k10=rm_bold_k10_s0/random_a1.0_k10"
  log "START $out ($arms)"
  python sft_lls.py --stage remove --trait $tr --arms no_removal,$arms --no_removal_adapter gate_s0/no_removal \
    --ref_results gate_s0/results.json --eval "$extra" $COMMON --out_dir ./$out > $out.log 2>&1
  rc=$?; [ -f $out/results.json ] && log "DONE $out rc=$rc" || log "DIED $out rc=$rc"
  cp $out.log PREREG_sft.md $out/ 2>/dev/null; push $out $out
done
log "QUEUE_COMPLETE"
