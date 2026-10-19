#!/usr/bin/env bash
# launch.sh <config> <log>: run the targeted decomposition with the bold-decision-only loss
# (target windows are 48 tokens). Needs WANDB_API_KEY in env to resolve the pretrained model.
set -u
cd /workspace/tpd
export SPD_OUT_DIR=/workspace/spd_out HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SPD_LAST_POS_SEQLEN=48
exec .venv/bin/python spd/experiments/lm/lm_decomposition.py "$1" > "$2" 2>&1
