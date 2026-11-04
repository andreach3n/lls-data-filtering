#!/usr/bin/env bash
# On a set-up pod (setup_pod.sh done): fetch the goblin mixes from HF, build the gated_lora data dirs, start a run.
#   nohup bash goblin_pod_start.sh plain mix_p005 > /workspace/goblin_start.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
export HF_TOKEN_PATH=/workspace/.cache/huggingface/token
if [ ! -f /workspace/goblin_data/summary.json ]; then
  python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("andreayhchen/lls-filtering-data", repo_type="dataset", allow_patterns=["pd/lora/goblin_src/mix_p*/*"],
                  local_dir="/workspace/hfdl")
PY
  python goblin_prep.py --src /workspace/hfdl/pd/lora/goblin_src --out /workspace/goblin_data --nontarget /workspace/lora/data
fi
bash goblin_launch.sh "$@"
