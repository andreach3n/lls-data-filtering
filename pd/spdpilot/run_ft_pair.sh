#!/usr/bin/env bash
# run_ft_pair.sh <gpu> <base run dir> <data dir> <out dir>: fine-tuned stream train -> eval -> ** logit check
set -u
GPU=$1; BASE=$2/model_60000.pth; DATA=$3; OUT=$4
cd /workspace/spdpilot
export CUDA_VISIBLE_DEVICES=$GPU SPD_OUT_DIR=/workspace/spd_out HF_HOME=/workspace/hf
PY=/workspace/tpd/.venv/bin/python
$PY ft_stream.py train --base $BASE --data $DATA --out $OUT --steps 10000 --batch 64 --adv_steps 2 --save_every 2000 > $OUT.train.log 2>&1 \
 && $PY ft_stream.py eval --base $BASE --data $DATA --out $OUT > $OUT.eval.log 2>&1 \
 && $PY logit_check.py --base $BASE --data $DATA --out $OUT > $OUT.logit.log 2>&1
echo "FT_PAIR_DONE exit=$?" >> $OUT.eval.log
