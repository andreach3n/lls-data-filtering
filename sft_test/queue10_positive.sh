#!/bin/bash
# THE POSITIVE / INSTALLATION DIRECTION (mentor suggestion, 2026-09-17).
#
# Removal asks: delete the selected documents, does bold drop? Nine arms say no better than
# random. This asks the prior and easier question, and the one the log-linearity paper's
# Algorithm 1 was actually designed for:
#
#     train ONLY on the selected documents -- does bold rise MORE than training on the same
#     number of random documents?
#
# Why it should have come first:
#   - it is the method's native direction; removal is the harder inverse
#   - signal/noise: the selected set is 100% of the training signal, not 10% of a corpus
#     whose other 90% swamps it ("removal reveals what the remainder supports")
#   - it is ~9x cheaper: 19-37 optimizer steps instead of 336
#   - it diagnoses the removal null. tailonly > randomonly => the score DOES carry bold, and
#     the null is about removal being a weak operation. tailonly == randomonly => the score
#     never tracked bold and the removal track was untestable.
#
# Design input (token-weighted bold density of the tail vs corpus, computed from the scores
# already on HF; a random set is 1.00x by construction):
#
#     k      docs  steps   alpha=0.00   alpha=0.32   alpha=1.00
#     1.0%    238      4   1.58x        1.61x        1.45x
#     2.5%    596      9   1.45x        1.56x        1.34x
#     5.0%   1193     19   1.35x        1.43x        1.30x
#    10.0%   2386     37   1.27x        1.28x        1.22x
#    25.0%   5965     93   1.10x        1.09x        1.09x
#
# On a fine grid the density curve is FLAT from alpha ~0.2 to ~0.7 at every k, so 0.323 sits
# on a broad plateau and is within 0.04 of optimal everywhere -- a coincidence, not a
# consequence of the variance fit (different objectives). Ceiling worth knowing: the densest
# selection with enough steps to train is ~1.5x corpus, so that is the entire signal
# available to this test.
#
# k CHOICE: 5% is the log-linearity paper's own gamma (lls_owl GAMMA = 0.05, "keep top 5% of
# positive-weight examples"); 10% matches the removal arms exactly, so the positive and
# negative directions run on the identical selection. ONE EPOCH throughout -- a smaller k with
# more epochs would buy ~0.1 of density but make the pairs non-comparable to each other and to
# every 1-epoch arm already run (two factors varying at once). Considered and rejected.
#
#   GPU=0 K=0.05 ./queue10_positive.sh
#   GPU=1 K=0.10 ./queue10_positive.sh
#
# Both arms of a pair train in ONE invocation, so their step counts are identical by
# construction and the only difference is which documents.
cd /workspace/lls/sft_test; source ./common_sft.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
K=${K:-0.05}
ALPHA=${ALPHA:-0.323}
out=pos_bold_a${ALPHA}_k$(python3 -c "print(int(float('$K')*1000))")
log "POSITIVE START k=$K alpha=$ALPHA on GPU $CUDA_VISIBLE_DEVICES -> $out"

python sft_lls.py --stage remove --trait bold --arms tailonly,randomonly \
  --ref_results controls/results.json \
  $COMMON --direction positive --remove_frac $K --alpha $ALPHA --out_dir ./$out > $out.log 2>&1
rc=$?
[ -f $out/results.json ] && log "DONE $out rc=$rc" || log "DIED $out rc=$rc"
grep -q "alpha=$ALPHA " $out.log || log "WARNING $out did NOT run at alpha=$ALPHA"
push $out $out
log "POSITIVE COMPLETE k=$K"
