#!/usr/bin/env bash
# Fresh RunPod setup for pd/lora: deps, Qwen3-1.7B-Base, prepared data from the HF backup.
# Run after copying pd/lora/*.py *.sh to /workspace/pd/lora and pd/smallsft/general100.json to /workspace/pd/smallsft/,
# and after `hf auth login` (or putting the token at /workspace/.cache/huggingface/token).
set -euo pipefail
pip install -q --break-system-packages transformers datasets pandas pyarrow safetensors huggingface_hub accelerate
export HF_TOKEN_PATH=${HF_TOKEN_PATH:-/workspace/.cache/huggingface/token}
python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("andreayhchen/lls-filtering-data", repo_type="dataset", allow_patterns=["pd/lora/data/*"],
                  local_dir="/workspace/hfdl")
snapshot_download("Qwen/Qwen3-1.7B-Base")
PY
mkdir -p /workspace/lora && rm -rf /workspace/lora/data && cp -r /workspace/hfdl/pd/lora/data /workspace/lora/data
ls -la /workspace/lora/data
nvidia-smi -L
df -h / | tail -1
echo SETUP_DONE
