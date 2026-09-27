#!/bin/bash
# The positive/installation direction, full grid: 2 selection sizes x 2 exponents.
#
#   GPU=0 ./queue11_positive_all.sh 0.05      # k=5%  at alpha 0.323 then alpha 1.0
#   GPU=1 ./queue11_positive_all.sh 0.10      # k=10% at alpha 0.323 then alpha 1.0
#
# Why both exponents:
#   alpha=1.0   the log-linearity paper's own normalisation, and what EVERY removal arm used.
#               If this selects installable data while removal at the same setting fails, the
#               null is about removal being a weak operation, not about the score. That is
#               the cleanest statement available, which is why it matters more than 0.323.
#   alpha=0.323 the variance-equalising fit; slightly denser selection (see the table below).
#
# Token-weighted bold density of the selected set (x corpus; random = 1.00x by construction):
#     k       alpha=0.323   alpha=1.0   steps
#     5%      1.43x         1.30x       19
#    10%      1.28x         1.22x       37
#
# ONE EPOCH throughout, matching every arm already run. Both arms of a pair train in a single
# invocation so their step counts are identical by construction.
cd /workspace/lls/sft_test; source ./common_sft.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
K=${1:-0.05}
KTAG=$(python3 -c "print(int(float('$K')*1000))")

for ALPHA in 0.323 1.0; do
  out=pos_bold_a${ALPHA}_k${KTAG}
  if [ -f $out/results.json ]; then log "SKIP $out (already done)"; continue; fi
  log "POSITIVE START k=$K alpha=$ALPHA on GPU $CUDA_VISIBLE_DEVICES -> $out"
  python sft_lls.py --stage remove --trait bold --arms tailonly,randomonly \
    --ref_results controls/results.json \
    $COMMON --direction positive --remove_frac $K --alpha $ALPHA --out_dir ./$out > $out.log 2>&1
  rc=$?
  [ -f $out/results.json ] && log "DONE $out rc=$rc" || log "DIED $out rc=$rc"
  grep -q "alpha=$ALPHA " $out.log || log "WARNING $out did NOT run at alpha=$ALPHA"
  push $out $out
done
log "POSITIVE-ALL COMPLETE k=$K"
