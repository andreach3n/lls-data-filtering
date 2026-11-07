#!/usr/bin/env bash
# Goblin Weave runs (one per pod): train -> eval -> export -> push every checkpoint to HF.
#   bash goblin_launch.sh plain <mix> [steps]   plain LoRA (default 10k steps), to see how long each poison fraction needs
#   bash goblin_launch.sh pd <mix> <steps>      gated LoRA with necessity (lam 1e-3), no FineWeb term
# Env: RUN_NAME (run folder), EXTRA_FLAGS (appended to the train/eval flags), VAL_EVERY, EVAL_FLAGS, SKIP_PUSH.
# <mix> = mix_p1 | mix_p02 | mix_p005 | mix_p001 | mix_p0, a folder of $GOBLIN_DATA (goblin_prep.py output).
# Progress: runs/<run>/train_log.jsonl (held-out curve every 250 steps); runs/<run>/status at the end.
set -uo pipefail
cd "$(dirname "$0")"
MODE=$1 MIX=$2
DATA=${GOBLIN_DATA:-/workspace/goblin_data}/$MIX
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_TOKEN_PATH=/workspace/.cache/huggingface/token
COMMON="--data $DATA --trait goblin --ctx 1024 --masked_loss ce --anneal_end 0.8 --val_every ${VAL_EVERY:-250} --n_val 100"
case $MODE in
  plain) STEPS=${3:-10000}; R=goblin_plain_$MIX
         FLAGS="--steps $STEPS --c_stoch 0 --c_adv 0 --c_nt 0 --lam 0" ;;
  pd)    STEPS=$3; R=goblin_pd_$MIX
         FLAGS="--steps $STEPS --lam 1e-3 --c_comp 1.0 --c_comp_adv 0.5 --c_nt 0 --nt_ratio 0" ;;  # no FineWeb term (from 3 Nov; earlier goblin PD runs had --c_nt 1 --nt_ratio 2)
  *) echo "mode must be plain or pd"; exit 1 ;;
esac
R=${RUN_NAME:-$R}
FLAGS="$FLAGS ${EXTRA_FLAGS:-}"  # e.g. EXTRA_FLAGS="--c_comp 4 --c_comp_adv 2" (later flags win)
SAVE=$(( STEPS / 10 > 0 ? STEPS / 10 : 1 ))  # 10 checkpoints per run
mkdir -p runs/$R
{
  HF_HUB_OFFLINE=1 python gated_lora.py train --out runs/$R $COMMON $FLAGS --save_every $SAVE --keep_ckpts 10 > runs/$R/train.log 2>&1 &&
  HF_HUB_OFFLINE=1 python gated_lora.py eval --out runs/$R $COMMON $FLAGS --batch 4 --batch_nt 4 --gen_bs 8 ${EVAL_FLAGS:-} > runs/$R/eval.log 2>&1 &&
  HF_HUB_OFFLINE=1 python gated_lora.py export --out runs/$R >> runs/$R/eval.log 2>&1 &&
  { [ -n "${SKIP_PUSH:-}" ] || ALL_CKPTS=1 python push_hf.py $R > runs/$R/push.log 2>&1; }
  echo "$R exit $? $(date)" > runs/$R/status
}
cat runs/$R/status
