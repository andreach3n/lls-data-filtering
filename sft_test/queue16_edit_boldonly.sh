#!/bin/bash
# REGEX STRIP, BOLD ONLY: the post's Evidence 3 verbatim. Every `**` deleted from every
# assistant turn (thinking included) in all 23,860 documents; headers and bullets kept.
# Companion to queue15_dropbold.sh (drop documents with bold); together they answer the
# mentor's question, 2026-09-30: after filtering by regex, does bold still go up?
#
#   GPU=0 bash queue16_edit_boldonly.sh
#
# All documents kept, so 373 steps and the same LR schedule as no_removal: no size confound,
# and the comparator is no_removal (22.83), not random. Already measured for the edit family:
#   strip=all (bold + headers + bullets)   0.71   -- far below base 15.27
# The post reports bold did NOT drop for this edit. The prereg's hypothesis (2026-09-15): bold
# is re-derived from the header/bullet scaffold left in place, so this reproduces their null.
#
# PRE-REGISTERED READING (bold per closed answer, general set):
#   near no_removal (> ~20)   their null reproduced: removing every bold marker does not
#                             remove bold -- the scaffold carries it. LLS on this corpus would
#                             need rescoring (the text changed), unlike queue15's.
#   between base and 20       partial: the markers carry some of it, the scaffold the rest.
#   <= base (15.27)           markers alone carry it; their Evidence 3 does not reproduce here.
# ~10 h on one RTX PRO 6000: ~8 h training (373 steps), ~2 h eval.
cd /workspace/lls/sft_test; source ./common_sft.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
out=edit_boldonly
log "Q16 START edit strip=bold on GPU $CUDA_VISIBLE_DEVICES -> $out"
# every override AFTER $COMMON: argparse takes the last occurrence and $COMMON carries its own
python sft_lls.py --stage remove --trait bold --arms edit --edit_strip bold \
  --ref_results controls/results.json \
  $COMMON --alpha 1.0 --out_dir ./$out > $out.log 2>&1
rc=$?
[ -f $out/results.json ] && log "DONE $out rc=$rc" || log "DIED $out rc=$rc"
grep -q "edit arm \[strip=bold\]" $out.log || log "WARNING $out did not run the bold-only strip"
push $out $out
hf upload $REPO ./$out.log sft/$out/$out.log --repo-type dataset >> push.log 2>&1
log "Q16 COMPLETE"
