#!/bin/bash
# Resume after the 2026-09-15 pod interruption. Run from /workspace/lls/sft_test.
#
# Nothing needs retraining: every adapter survived and is on HF. What was lost was the
# EVALUATION of two models. Ordered cheapest-first so the k=10% table completes before any
# new training starts.
#
#   GPU=0  ./queue6_resume.sh evals     ~2 h   finish the k=10% bold table
#   GPU=1  ./queue6_resume.sh arms      ~16 h  the de-bolding decomposition
#
# Every run now carries --hf_repo (see common_sft.sh), so each adapter and each model's
# results.json + generations upload from inside the python process as soon as they exist.
cd /workspace/lls/sft_test; source ./common_sft.sh
export CUDA_VISIBLE_DEVICES=${GPU:-0}
MODE=${1:-evals}
log "RESUME START mode=$MODE on GPU $CUDA_VISIBLE_DEVICES"

if [ "$MODE" = "evals" ]; then
  # 1. the length-matched control: adapter exists, never evaluated. This is the binding
  #    control for bold (the LLS tail is ~3x shorter than the corpus).
  # 2. the LLS arm: `general` completed before the interruption, so --reeval regenerates
  #    all five prompt sets for a clean, complete row.
  out=rm_bold_k10_s0
  log "START $out eval (lenmatch + lls, all prompt sets)"
  python sft_lls.py --stage eval --reeval \
    --eval "lenmatch_bold_a1.0_k10=$out/lenmatch_bold_a1.0_k10,lls_bold_a1.0_k10=$out/lls_bold_a1.0_k10" \
    --ref_results controls/results.json $COMMON --out_dir ./$out > ${out}_eval.log 2>&1
  [ -f $out/results.json ] && log "DONE $out eval" || log "DIED $out eval"

  # 3. backfill: base and no_removal predate the held-out refusal_hard set, so their rows
  #    are missing that column. ~25 min each.
  log "START refusal_hard backfill (base + no_removal)"
  python sft_lls.py --stage eval --eval "base,no_removal=controls/no_removal" \
    --psets refusal_hard $COMMON --out_dir ./controls > backfill_refusal_hard.log 2>&1
  log "DONE refusal_hard backfill"

  log "RESUME evals COMPLETE"
  python sft_lls.py --stage report --ref_results controls,rm_bold_k10_s0,rm_bold_edit | tail -40
  exit 0
fi

# ---- arms: the de-bolding decomposition (both were interrupted) -------------
out=rm_bold_edit_boldonly
log "START $out (edit, strip=bold -- their Evidence 3 verbatim)"
python sft_lls.py --stage remove --trait bold --arms edit --edit_strip bold \
  --no_removal_adapter controls/no_removal --ref_results controls/results.json \
  $COMMON --direction positive --out_dir ./$out > $out.log 2>&1
[ -f $out/results.json ] && log "DONE $out" || log "DIED $out"

out=rm_bold_edit_structonly
log "START $out (edit, strip=structure)"
python sft_lls.py --stage remove --trait bold --arms edit --edit_strip structure \
  --no_removal_adapter controls/no_removal --ref_results controls/results.json \
  $COMMON --direction positive --out_dir ./$out > $out.log 2>&1
[ -f $out/results.json ] && log "DONE $out" || log "DIED $out"
log "RESUME arms COMPLETE"
