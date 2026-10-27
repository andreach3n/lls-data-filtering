#!/usr/bin/env bash
# Third batch, one job chain per GPU. Every training run logs a held-out curve (every 250 steps) and keeps a checkpoint
# every 1000 steps, so any step can be evaluated afterwards.
#   GPU 0: bold-component analysis of the second batch's ce_vpd and ce_comp checkpoints (from HF), then a plain LoRA
#          for 10k steps: the fine-tune's own held-out curve (where does overfitting start?)
#   GPU 1: necessity, lam 1e-3, 5k steps (ce_comp at the shorter length)
#   GPU 2: necessity, lam 3e-4, 5k steps
#   GPU 3: necessity, lam 3e-3, 5k steps
#   GPU 4: control (no necessity), lam 1e-3, 5k steps
# Usage: bash launch_v3.sh [all|two|three|nec2|nec1]. Needs setup_pod.sh first (deps, Qwen3-1.7B-Base, /workspace/lora/data) and the HF token.
set -uo pipefail
cd "$(dirname "$0")"
DATA=/workspace/lora/data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_TOKEN_PATH=/workspace/.cache/huggingface/token
COMMON="--data $DATA --masked_loss ce --anneal_end 0.8 --val_every 250 --save_every 1000 --keep_ckpts 10"
NEC="--c_comp 1.0 --c_comp_adv 0.5"

chain() {  # chain <gpu> <run> <train flags...>: train, eval, export, push on one GPU
  local gpu=$1 r=$2; shift 2
  mkdir -p runs/$r
  CUDA_VISIBLE_DEVICES=$gpu python gated_lora.py train --out runs/$r $COMMON "$@" > runs/$r/train.log 2>&1 &&
  CUDA_VISIBLE_DEVICES=$gpu python gated_lora.py eval --data $DATA --out runs/$r --batch 4 --batch_nt 4 --gen_bs 8 "$@" > runs/$r/eval.log 2>&1 &&
  python gated_lora.py export --out runs/$r >> runs/$r/eval.log 2>&1 &&
  python push_hf.py $r > runs/$r/push.log 2>&1
  echo "$r finished with status $? on GPU $gpu"
}

PART=${1:-all}   # all (5 GPUs) | two (analysis + plain curve, control) | three (necessity sweep) | nec2 + nec1 (sweep split 2+1)

analysis_and_plain() {
  python - <<'PY'
from huggingface_hub import hf_hub_download
for r in ("ce_vpd", "ce_comp"):
    hf_hub_download("andreayhchen/lls-filtering-data", f"pd/lora/runs/{r}/ckpt_10000.pt", repo_type="dataset", local_dir="/workspace/hfdl")
PY
  for r in ce_vpd ce_comp; do
    mkdir -p runs/$r
    CUDA_VISIBLE_DEVICES=$1 python analyze_bold.py --ckpt /workspace/hfdl/pd/lora/runs/$r/ckpt_10000.pt --data $DATA \
        --out runs/$r/bold_analysis.json > runs/$r/bold_analysis.log 2>&1
    echo "analysis $r status $?"
  done
  chain $1 plain_curve --steps 10000 --c_stoch 0 --c_adv 0 --c_nt 0 --lam 0
}

case $PART in
  all)
    analysis_and_plain 0 &
    chain 1 nec_lam1e-3_5k --steps 5000 --lam 1e-3 $NEC &
    chain 2 nec_lam3e-4_5k --steps 5000 --lam 3e-4 $NEC &
    chain 3 nec_lam3e-3_5k --steps 5000 --lam 3e-3 $NEC &
    chain 4 vpd_lam1e-3_5k --steps 5000 --lam 1e-3 & ;;
  two)
    analysis_and_plain 0 &
    chain 1 vpd_lam1e-3_5k --steps 5000 --lam 1e-3 & ;;
  three)
    chain 0 nec_lam1e-3_5k --steps 5000 --lam 1e-3 $NEC &
    chain 1 nec_lam3e-4_5k --steps 5000 --lam 3e-4 $NEC &
    chain 2 nec_lam3e-3_5k --steps 5000 --lam 3e-3 $NEC & ;;
  nec2)   # 2-GPU pod: two points of the necessity sweep
    chain 0 nec_lam1e-3_5k --steps 5000 --lam 1e-3 $NEC &
    chain 1 nec_lam3e-4_5k --steps 5000 --lam 3e-4 $NEC & ;;
  nec1)   # 1-GPU pod: the third point
    chain 0 nec_lam3e-3_5k --steps 5000 --lam 3e-3 $NEC & ;;
  *) echo "unknown part $PART"; exit 1 ;;
esac
wait
echo ALL_DONE
