#!/usr/bin/env bash
# launch_binary.sh <config> <log>: targeted decomposition where the target loss is the KL of the
# binary decision "next token opens bold (** or Ġ**) vs anything else" at the last position of each
# 48-token target window. Needs WANDB_API_KEY in env to resolve the pretrained model.
set -u
cd /workspace/tpd
export SPD_OUT_DIR=/workspace/spd_out HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SPD_LAST_POS_SEQLEN=48 SPD_BINARY_IDS=/workspace/spdpilot/bold_ids.json
exec .venv/bin/python spd/experiments/lm/lm_decomposition.py "$1" > "$2" 2>&1
