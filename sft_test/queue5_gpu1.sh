#!/bin/bash
# GPU 1, phase 5 (2026-09-15): decompose the de-bolding result.
#
# rm_bold_edit (strip=all) took "% of answers using bold" from 0.985 to 0.190. The post's
# Evidence 3 stripped "every ** in the training data" -- bold markers ONLY -- and reported
# no change on that same measure. So the intervention differs, not the metric. These two
# arms close the gap:
#
#   1. edit_bold       strip ** only            = their Evidence 3, verbatim.
#                      If bold stays high -> we REPRODUCE their null, and the mechanism is
#                      that bold is re-derived from the structural scaffold left behind.
#   2. edit_structure  strip headers/bullets only, keep **.
#                      If bold ALSO collapses here -> structure is what carries the
#                      behaviour and bold is downstream of it. If bold survives -> the two
#                      marker kinds are independent and only stripping both works.
#
# k=25% is deferred behind these: dose-response matters less than reconciling their
# headline null. Its lls arm was killed 2.4 h in and can be rerun unchanged.
cd /root/lls/sft_test; source ./common_sft.sh
export CUDA_VISIBLE_DEVICES=1
log "GPU1-P5 START (de-bolding decomposition)"

out=rm_bold_edit_boldonly
log "START $out (edit, strip=bold -- their Evidence 3 verbatim)"
python sft_lls.py --stage remove --trait bold --arms edit --edit_strip bold \
  --no_removal_adapter controls/no_removal --ref_results controls/results.json \
  $COMMON --direction positive --out_dir ./$out > $out.log 2>&1
[ -f $out/results.json ] && log "DONE $out" || log "DIED $out"; push $out $out

out=rm_bold_edit_structonly
log "START $out (edit, strip=structure)"
python sft_lls.py --stage remove --trait bold --arms edit --edit_strip structure \
  --no_removal_adapter controls/no_removal --ref_results controls/results.json \
  $COMMON --direction positive --out_dir ./$out > $out.log 2>&1
[ -f $out/results.json ] && log "DONE $out" || log "DIED $out"; push $out $out

out=rm_bold_k25_s0
log "START $out (lls,random at k=25%) -- deferred from phase 4"
python sft_lls.py --stage remove --trait bold --arms lls,random \
  --no_removal_adapter controls/no_removal --ref_results controls/results.json \
  $COMMON --direction positive --remove_frac 0.25 --out_dir ./$out > $out.log 2>&1
[ -f $out/results.json ] && log "DONE $out" || log "DIED $out"; push $out $out
log "GPU1-P5 COMPLETE"
