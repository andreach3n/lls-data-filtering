#!/usr/bin/env bash
# Section 13's backup plot (findings/backup.png) without the FineWeb (non-target) term, so the only difference
# between the two runs is necessity.
#   vpd_nofw_lam1e-3_5k  new: VPD only, --c_nt 0 --nt_ratio 0 (train -> eval -> export -> push)
#   nec_nofw_lam1e-3_5k  exists on HF: same, + necessity (c_comp 1 / c_comp_adv 0.5)
# Both get backup_bold.py's held-out split (rank on even docs, test on odd), pushed to HF.
# --min_real_only 0: nec_nofw was trained before the 4 Nov padding fix, so the new run keeps the old minimality.
# Usage (on the pod, after setup_pod.sh): bash bold_nofw_launch.sh [train_gpu] [analysis_gpu]   (default 0 0; training
# needs ~22 GB, so both share one 80 GB GPU in parallel; SEQ=1 runs them one after the other). Then locally: ../.venv/bin/python findings_plots.py  -> findings/backup_nofw.png
set -uo pipefail
cd "$(dirname "$0")"
GT=${1:-0} GA=${2:-0}
DATA=/workspace/lora/data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_TOKEN_PATH=/workspace/.cache/huggingface/token
COMMON="--data $DATA --masked_loss ce --anneal_end 0.8 --val_every 250 --save_every 1000 --keep_ckpts 10"
V=vpd_nofw_lam1e-3_5k N=nec_nofw_lam1e-3_5k
mkdir -p backup/split

split() {  # split <run> <ckpt> <gpu>: held-out backup analysis, as eval_chain.sh
  CUDA_VISIBLE_DEVICES=$3 python backup_bold.py --ckpt $2 --docs even --out backup/split/${1}_even.json > backup/split/$1.log 2>&1 &&
  CUDA_VISIBLE_DEVICES=$3 python backup_bold.py --ckpt $2 --docs odd --order_from backup/split/${1}_even.attr.pt \
      --out backup/split/${1}_odd_from_even.json >> backup/split/$1.log 2>&1
  echo "split $1 status $?"
}

nec() {
  python - <<'PY'
from huggingface_hub import hf_hub_download
hf_hub_download("andreayhchen/lls-filtering-data", "pd/lora/runs/nec_nofw_lam1e-3_5k/ckpt_5000.pt",
                repo_type="dataset", local_dir="/workspace/hfdl")
PY
  split $N /workspace/hfdl/pd/lora/runs/$N/ckpt_5000.pt $GA
}

vpd() {
  local F="--steps 5000 --lam 1e-3 --c_nt 0 --nt_ratio 0 --min_real_only 0"
  mkdir -p runs/$V
  CUDA_VISIBLE_DEVICES=$GT python gated_lora.py train --out runs/$V $COMMON $F > runs/$V/train.log 2>&1 &&
  CUDA_VISIBLE_DEVICES=$GT python gated_lora.py eval --data $DATA --out runs/$V --batch 4 --batch_nt 4 --gen_bs 8 $F > runs/$V/eval.log 2>&1 &&
  python gated_lora.py export --out runs/$V >> runs/$V/eval.log 2>&1 &&
  python push_hf.py $V > runs/$V/push.log 2>&1
  echo "$V train status $?"
  split $V runs/$V/ckpt_5000.pt $GT
}

if [ -n "${SEQ:-}" ]; then nec; vpd; else nec & vpd & wait; fi

HF_HOME=/workspace/.cache/huggingface python - <<'PY'
from huggingface_hub import HfApi
HfApi().upload_folder(folder_path="backup/split", path_in_repo="pd/lora/backup/split", repo_id="andreayhchen/lls-filtering-data",
    repo_type="dataset", allow_patterns=["*nofw*.json", "*nofw*.log"], commit_message="pd/lora: bold backup split, no FineWeb")
print("PUSHED backup/split nofw")
PY
echo ALL_DONE
