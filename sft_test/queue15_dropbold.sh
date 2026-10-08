#!/bin/bash
# REGEX FILTER: drop every document with a **bold** span in any assistant turn (thinking
# included), train on the rest, measure bold. Mentor question, 2026-09-30: after filtering by
# regex, does bold still go up? If so, the follow-up is LLS on this filtered corpus -- which
# needs no rescoring, because dropping documents leaves the kept ones (and their scores) intact.
#
#   GPU=0 bash queue15_dropbold.sh
#
# Kept: 9,109 of 23,860 documents (38%), 142 steps; 46% still contain headers/bullets. Mix is
# code-heavy (56% correct-python vs ~34% in the corpus). Comparators, already evaluated and
# imported via controls/results.json: base 15.27, no_removal 22.83 (both bold per closed answer).
#
# PRE-REGISTERED READING (bold per closed answer, general set):
#   >> base (say > 18)     bold still rises with no bold in any training document -> the
#                          behaviour is learned from something the regex misses (the prereg's
#                          headers/bullets-scaffold hypothesis); run LLS on this corpus next.
#   ~ base (15-18)         the document filter blocks the increase; no residual for LLS.
#   << base                like rm_bold_edit (0.71): bold-free data actively suppresses it.
# Caveat either way: a lower rate is partly the mix shift toward code.
cd /workspace/lls/sft_test; source ./common_sft.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
out=dropbold_k10
log "Q15 START dropbold on GPU $CUDA_VISIBLE_DEVICES -> $out"
# every override AFTER $COMMON: argparse takes the last occurrence and $COMMON carries its own
python sft_lls.py --stage remove --trait bold --arms dropbold \
  --ref_results controls/results.json \
  $COMMON --alpha 1.0 --out_dir ./$out > $out.log 2>&1
rc=$?
[ -f $out/results.json ] && log "DONE $out rc=$rc" || log "DIED $out rc=$rc"
grep -q "dropbold arm: .* 9,109 kept" $out.log || log "WARNING $out did not keep the expected 9,109 documents"
push $out $out
hf upload $REPO ./$out.log sft/$out/$out.log --repo-type dataset >> push.log 2>&1
log "Q15 COMPLETE"
