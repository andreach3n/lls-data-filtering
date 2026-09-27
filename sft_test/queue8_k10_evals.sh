#!/bin/bash
# Finish the k=10% bold table. Both adapters were trained on the previous pod and recovered
# from HF; neither has a complete evaluation.
#
#   GPU=0 ARM=lenmatch  ./queue8_k10_evals.sh   -> rm_bold_k10_lenmatch  (never evaluated)
#   GPU=1 ARM=lls       ./queue8_k10_evals.sh   -> rm_bold_k10_lls       (only `general` ran)
#
# Waits for whatever k=25% job is on this card, then evaluates. ~1 h each.
#
# SEPARATE out_dirs on purpose: the two evals run concurrently and would otherwise race on
# one results.json. `--stage report --ref_results controls,rm_bold_k10_lls,rm_bold_k10_lenmatch,...`
# merges them.
#
# The lls arm gets a FULL re-evaluation rather than only its four missing prompt sets. The
# ~20% saving is not worth stitching one prompt set from an interrupted run, and a complete
# row measured in one pass is more trustworthy. Its earlier partial number (bold_cl 22.730
# on `general`) will shift slightly from sampling noise at temperature 1.0 -- that is
# expected, and both are valid samples.
cd /workspace/lls/sft_test; source ./common_sft.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
ARM=${ARM:-lenmatch}
log "K10-EVAL waiting for the k=25% job on GPU $CUDA_VISIBLE_DEVICES (arm=$ARM)"
while pgrep -f "remove --trait bol[d] --arms" > /dev/null; do sleep 180; done
log "K10-EVAL GPU free, starting arm=$ARM"

case $ARM in
  lenmatch) LABEL=lenmatch_bold_a1.0_k10 ;;
  lls)      LABEL=lls_bold_a1.0_k10 ;;
  *) echo "ARM must be lenmatch or lls"; exit 1 ;;
esac
out=rm_bold_k10_${ARM}
mkdir -p $out
log "START $out eval ($LABEL, all prompt sets)"
python sft_lls.py --stage eval --reeval \
  --eval "$LABEL=rm_bold_k10_s0/$LABEL" \
  --ref_results controls/results.json $COMMON --out_dir ./$out > ${out}_eval.log 2>&1
rc=$?
[ -f $out/results.json ] && log "DONE $out rc=$rc" || log "DIED $out rc=$rc"
push $out $out
log "K10-EVAL COMPLETE arm=$ARM"
