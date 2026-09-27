#!/bin/bash
# Bold removal at the EMPIRICALLY FITTED length exponent.
#
# Why: at alpha=1 (the log-linearity paper's normalisation, which the prereg committed to)
# the top 10% is dominated by short high-variance documents -- median 99 answer tokens
# against a corpus median of 324, and only 43% of them contain any bold at all, BELOW the
# corpus rate of 59%. rho(w, bold-per-100-words) = -0.044. So the k=10%/25% nulls are valid
# statements about that normalisation but are NOT a test of removing bold-carrying data.
#
# alpha is fitted the way HANDOFF.md prescribes: regress log sd(w_raw) on log N over 15
# quantile bins; the slope equalises variance across lengths. bold -> 0.323 (R2 0.92),
# close to the 0.37 fitted at 1B. That tail is length-neutral (median 293 tokens) and
# 1.7x bold-enriched (3.90 per 100 words vs 2.31; 69% contain bold vs 59%).
#
#   GPU=0 ./queue9_alpha.sh
#
# Only the lls arm is needed: `random` removal does not depend on alpha, so the existing
# k=10% random arm (bold 21.57) is the valid size-matched control.
cd /workspace/lls/sft_test; source ./common_sft.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
ALPHA=${ALPHA:-0.323}
out=rm_bold_alpha${ALPHA}_k10
# Chain off the queue log, not a process pattern. (A previous version used "a\|b" in
# pgrep -f, which is ERE -- the backslash made it a literal, the match failed, and the arm
# launched onto a busy card. Waiting for a logged milestone is deterministic and ordered,
# so this cannot race the k=10% eval that is also queued on GPU 0.)
WAIT_FOR=${WAIT_FOR:-"K10-EVAL COMPLETE arm=lenmatch"}
log "ALPHA-ARM waiting for: $WAIT_FOR  (alpha=$ALPHA)"
until grep -q "$WAIT_FOR" "$S"; do sleep 120; done
# belt and braces: also require the card to be idle before claiming it
while [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader -i ${CUDA_VISIBLE_DEVICES} 2>/dev/null)" ]; do sleep 60; done
log "START $out (lls at alpha=$ALPHA, k=10%)"
# EVERY override goes AFTER $COMMON: argparse takes the last occurrence, and $COMMON
# carries --alpha 1.0 --remove_frac 0.10 --direction positive of its own. An earlier
# version put --alpha first and silently trained 9 h at alpha=1.0.
python sft_lls.py --stage remove --trait bold --arms lls \
  --ref_results controls/results.json \
  --eval "random_a1.0_k10=controls/random_a1.0_k10" \
  $COMMON --direction positive --remove_frac 0.10 --alpha $ALPHA --out_dir ./$out > $out.log 2>&1
rc=$?
[ -f $out/results.json ] && log "DONE $out rc=$rc" || log "DIED $out rc=$rc"
grep -q "alpha=$ALPHA " $out.log || log "WARNING $out did NOT run at alpha=$ALPHA -- check $out.log"
push $out $out
log "ALPHA-ARM COMPLETE"
