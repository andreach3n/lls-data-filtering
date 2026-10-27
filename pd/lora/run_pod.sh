#!/usr/bin/env bash
# H100 pod recipe: data -> base bold rate -> 20-step timing run -> full option-A run + plain control (launch.sh).
set -euo pipefail
cd /workspace/pd/lora
export HF_HUB_OFFLINE=0 TOKENIZERS_PARALLELISM=false
DATA=/workspace/lora/data
[ -f $DATA/sft_train.pt ] || python prep_data.py --split /workspace/smallsft/split_s0_mix.parquet --out $DATA > prep.log 2>&1
python base_rate.py --data $DATA > base_rate.log 2>&1
mkdir -p runs/timing
python gated_lora.py train --data $DATA --out runs/timing --steps 20 --adv_start 0.5 --log_every 1 --save_every 1000 > runs/timing/train.log 2>&1
echo TIMING_DONE
DATA=$DATA STEPS=${STEPS:-5000} bash launch.sh > launch.log 2>&1
echo ALL_DONE
