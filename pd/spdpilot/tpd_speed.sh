#!/usr/bin/env bash
# Time the targeted (tPD) 12L config and record peak GPU memory. Needs WANDB_API_KEY in env.
set -u
cd /workspace/tpd
export SPD_OUT_DIR=/workspace/spd_out HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python spd/experiments/lm/lm_decomposition.py /workspace/spdpilot/tpd_speed.yaml > /workspace/spdpilot/tpd_speed.log 2>&1 &
PID=$!
: > /workspace/spdpilot/tpd_mem.txt
while kill -0 $PID 2>/dev/null; do
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> /workspace/spdpilot/tpd_mem.txt; sleep 10
done
echo "peak MiB: $(sort -n /workspace/spdpilot/tpd_mem.txt | tail -1)" >> /workspace/spdpilot/tpd_speed.log
