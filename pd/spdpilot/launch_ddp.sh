#!/usr/bin/env bash
# launch_ddp.sh <n_gpus> <config> <log>: targeted decomposition (target loss at the last position of the
# 48-token windows) split data-parallel over n_gpus with torchrun. Needs WANDB_API_KEY to resolve the model.
set -u
cd /workspace/tpd
export SPD_OUT_DIR=/workspace/spd_out HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export SPD_LAST_POS_SEQLEN=48
exec .venv/bin/python -m torch.distributed.run --standalone --nproc_per_node "$1" \
  spd/experiments/lm/lm_decomposition.py "$2" > "$3" 2>&1
