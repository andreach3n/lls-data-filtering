#!/bin/bash
# GPU 0: controls, then bold, then bothsides.
#   1. gate      no_removal on the full 25K + eval base, no_removal      (~6 h)
#   2. random    the post's own control: 2,500 random docs removed       (~6 h)
#   3. bold      lls + lenmatch                                          (~12 h)
#   4. bothsides lls + lenmatch                                          (~12 h)
# Scoring runs on GPU 1 in parallel; step 3 waits for lp_bold.parquet to appear.
cd /root/lls/sft_test; source ./common_sft.sh
export CUDA_VISIBLE_DEVICES=0
log "GPU0 START"

# ---- 1. gate (needs no scores) --------------------------------------------
if [ ! -f controls/results.json ]; then
  log "START gate"
  python sft_lls.py --stage gate $COMMON --out_dir ./controls > gate.log 2>&1
  [ -f controls/results.json ] && log "DONE gate" || { log "DIED gate"; exit 1; }
  push controls controls
fi

# ---- 2. random control (trait-independent; needs lp_bold only to rank the unused tail)
until [ -f $SCORES/lp_bold.parquet ]; do sleep 300; done
log "START random"
python sft_lls.py --stage remove --trait bold --arms random --no_removal_adapter controls/no_removal \
  --ref_results controls/results.json $COMMON --out_dir ./controls > random.log 2>&1
[ -f controls/random_a1.0_k10/adapter_config.json ] && log "DONE random" || log "DIED random"
push controls controls

# ---- 3 & 4. trait arms -----------------------------------------------------
for tr in bold bothsides; do
  until [ -f $SCORES/lp_${tr}.parquet ]; do sleep 300; done
  out=rm_${tr}_k10_s0
  log "START $out (lls,lenmatch)"
  python sft_lls.py --stage remove --trait $tr --arms lls,lenmatch \
    --no_removal_adapter controls/no_removal --ref_results controls/results.json \
    $COMMON --out_dir ./$out > $out.log 2>&1
  rc=$?; [ -f $out/results.json ] && log "DONE $out rc=$rc" || log "DIED $out rc=$rc"
  cp $out.log PREREG_sft.md $out/ 2>/dev/null; push $out $out
done
log "GPU0 COMPLETE"
