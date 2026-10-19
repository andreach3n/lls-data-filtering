#!/usr/bin/env bash
# speedtest.sh <batch> <steps>: run the paper's 12L SPD config at a given batch size, record peak GPU memory.
set -u
B=$1; STEPS=$2
cd /workspace/tpd
.venv/bin/python - <<PY
import yaml
c = yaml.safe_load(open("/workspace/spdpilot/speed_pile12L.yaml"))
c.update(batch_size=$B, eval_batch_size=$B, steps=$STEPS, eval_freq=50, train_log_freq=10)
yaml.safe_dump(c, open("/workspace/spdpilot/speed_b$B.yaml", "w"), sort_keys=False)
PY
export SPD_OUT_DIR=/workspace/spd_out HF_HOME=/workspace/hf PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
.venv/bin/python spd/experiments/lm/lm_decomposition.py /workspace/spdpilot/speed_b$B.yaml > /workspace/spdpilot/speed_b$B.log 2>&1 &
PID=$!
: > /workspace/spdpilot/mem_b$B.txt
while kill -0 $PID 2>/dev/null; do
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> /workspace/spdpilot/mem_b$B.txt; sleep 10
done
echo "peak MiB: $(sort -n /workspace/spdpilot/mem_b$B.txt | tail -1)" >> /workspace/spdpilot/speed_b$B.log
