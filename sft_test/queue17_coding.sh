#!/bin/bash
# THE POST'S CODING SETTING: Coding SFT vs Coding De-bolded (their Evidence 3, which was run on
# coding data only -- never on the full mix). Coding = the 8,068 coding documents of our 23,860
# split (correct-python-sft 7,993 + OpenThoughts3-code 75), ~126 steps.
#
#   GPU=0 GROUP=1 bash queue17_coding.sh   # Coding SFT        (no_removal on the coding subset)
#   GPU=1 GROUP=2 bash queue17_coding.sh   # Coding De-bolded  (every ** stripped, same documents)
#
# Coding data carries half the corpus's bold (36% of docs with any bold vs 62%; 1.45 vs 2.81
# bold per 100 words). Their result, % of all generations with bold: Coding SFT 79%, Coding
# De-bolded 82% -- no drop. Ours on the full mix: 97% -> 18% (queue16).
#
# PRE-REGISTERED READING (% of ALL generations with any bold, plot_regex_filters --metric blog):
#   de-bolded within a few points of coding SFT   their Evidence 3 reproduced: on coding data the
#                                                 bold increase does not come from ** markers.
#                                                 With queue16, de-bolding fails on coding data
#                                                 and works on the full mix.
#   de-bolded far below coding SFT (as on the     their result does not reproduce here either;
#   full mix)                                     look at what else differs (dataset size: they
#                                                 do not state how many coding docs they used).
# ~5 h per GPU: ~2.5-3 h training, ~2 h eval.
cd /workspace/lls/sft_test; source ./common_sft.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
GROUP=${GROUP:-1}
case $GROUP in
  1) ARMS=no_removal; out=coding_sft ;;
  2) ARMS=edit;       out=coding_debolded ;;
  *) echo "GROUP must be 1 or 2"; exit 1 ;;
esac
log "Q17 START $out (arms=$ARMS) on GPU $CUDA_VISIBLE_DEVICES"
# every override AFTER $COMMON: argparse takes the last occurrence and $COMMON carries its own
python sft_lls.py --stage remove --trait bold --arms $ARMS --edit_strip bold \
  --source_filter 'code|python' --subset_tag coding \
  --ref_results controls/results.json \
  $COMMON --alpha 1.0 --out_dir ./$out > $out.log 2>&1
rc=$?
[ -f $out/results.json ] && log "DONE $out rc=$rc" || log "DIED $out rc=$rc"
grep -q "source filter 'code|python': 8,068 of 23,860" $out.log || log "WARNING $out did not keep the expected 8,068 coding docs"
[ $GROUP = 2 ] && { grep -q "edit arm \[strip=bold\]" $out.log || log "WARNING $out did not run the bold-only strip"; }
push $out $out
hf upload $REPO ./$out.log sft/$out/$out.log --repo-type dataset >> push.log 2>&1
log "Q17 COMPLETE $out"
