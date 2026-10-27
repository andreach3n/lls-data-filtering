#!/usr/bin/env bash
# CE-only (no reference model) follow-ups, one run per GPU, each followed on the same GPU by eval, export, HF push.
#   ce_vpd   : control, plain VPD (same losses as the first batch's ce_c32) under the new schedule
#   ce_bern  : hierarchical Bernoulli mask sources (per-token q ~ U(0,1), s ~ Bernoulli(q))
#   ce_rebal : smaller minimality weight, larger adversarial weight
#   ce_comp  : necessity terms (masks (1-g) s, KL to the base, stochastic + adversarial)
#   ce_all   : all three
# Schedule: STEPS (default 10000), lr cosine and p anneal settle at ANNEAL_END (default 0.8) and hold.
# Needs /workspace/lora/data (prep_data.py) and the HF token at /workspace/.cache/huggingface/token.
set -uo pipefail
cd "$(dirname "$0")"
DATA=${DATA:-/workspace/lora/data}; STEPS=${STEPS:-10000}; ANNEAL_END=${ANNEAL_END:-0.8}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_TOKEN_PATH=/workspace/.cache/huggingface/token
RUNS=(ce_vpd ce_bern ce_rebal ce_comp ce_all)
declare -A V=(
  [ce_vpd]=""
  [ce_bern]="--mask_dist bernoulli"
  [ce_rebal]="--lam 1e-4 --c_adv 2.0"
  [ce_comp]="--c_comp 1.0 --c_comp_adv 0.5"
  [ce_all]="--mask_dist bernoulli --lam 1e-4 --c_adv 2.0 --c_comp 1.0 --c_comp_adv 0.5"
)
NGPU=$(nvidia-smi -L | wc -l)
[ "$NGPU" -lt "${#RUNS[@]}" ] && echo "WARNING: $NGPU GPUs for ${#RUNS[@]} runs; runs will share GPUs and be slower"
i=0
for r in "${RUNS[@]}"; do
  gpu=$((i % NGPU)); i=$((i + 1))
  mkdir -p runs/$r
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    python gated_lora.py train --data $DATA --out runs/$r --masked_loss ce --steps $STEPS --anneal_end $ANNEAL_END \
        --save_every 1000 ${V[$r]} > runs/$r/train.log 2>&1 &&
    python gated_lora.py eval --data $DATA --out runs/$r --batch 4 --batch_nt 4 --gen_bs 8 ${V[$r]} > runs/$r/eval.log 2>&1 &&
    python gated_lora.py export --out runs/$r >> runs/$r/eval.log 2>&1 &&
    python push_hf.py $r > runs/$r/push.log 2>&1
    echo "$r finished with status $? on GPU $gpu"
  ) < /dev/null &
done
wait
echo ALL_DONE
