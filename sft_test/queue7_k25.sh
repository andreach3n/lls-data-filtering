#!/bin/bash
# k=25% bold removal, one arm per GPU, run in parallel.
#
#   GPU=0 ARM=lls     ./queue7_k25.sh     -> rm_bold_k25_lls
#   GPU=1 ARM=random  ./queue7_k25.sh     -> rm_bold_k25_random
#
# Separate out_dirs on purpose: two processes writing one results.json would race. Merge at
# the end with
#   python sft_lls.py --stage report --ref_results controls,rm_bold_k10_s0,rm_bold_edit,rm_bold_k25_lls,rm_bold_k25_random
#
# 25% removal leaves 17,895 documents -> 280 steps, ~5.5 h train + ~1 h eval per arm.
# --hf_repo is in $COMMON, so the adapter uploads the moment it is saved and each model's
# results.json + generations upload the moment its eval finishes. Nothing can be stranded.
cd /workspace/lls/sft_test; source ./common_sft.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
ARM=${ARM:-lls}
out=rm_bold_k25_${ARM}
log "K25 START arm=$ARM on GPU $CUDA_VISIBLE_DEVICES -> $out"

python sft_lls.py --stage remove --trait bold --arms $ARM \
  --ref_results controls/results.json \
  $COMMON --direction positive --remove_frac 0.25 --out_dir ./$out > $out.log 2>&1
rc=$?
[ -f $out/results.json ] && log "DONE $out rc=$rc" || log "DIED $out rc=$rc"
push $out $out
log "K25 COMPLETE arm=$ARM"
