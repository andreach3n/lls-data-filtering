#!/usr/bin/env bash
# wait for runs/$1/ckpt_5000.pt, then: backup analysis (all docs), held-out split, autointerp collection (both samples),
# and push checkpoint + results to HF. GPU $2.
r=$1; export CUDA_VISIBLE_DEVICES=$2
until [ -f runs/$r/ckpt_5000.pt ] && ! pgrep -f "out runs/$r\b" >/dev/null; do sleep 60; done
sleep 30
mkdir -p backup/cmp backup/split autointerp/v2 autointerp/bold
python backup_bold.py --ckpt runs/$r/ckpt_5000.pt --out backup/cmp/$r.json > backup/cmp/$r.log 2>&1
python backup_bold.py --ckpt runs/$r/ckpt_5000.pt --docs even --out backup/split/${r}_even.json > backup/split/$r.log 2>&1
python backup_bold.py --ckpt runs/$r/ckpt_5000.pt --docs odd --order_from backup/split/${r}_even.attr.pt --out backup/split/${r}_odd_from_even.json >> backup/split/$r.log 2>&1
python autointerp_collect.py --ckpt runs/$r/ckpt_5000.pt --out autointerp/v2/$r.json > autointerp/v2/$r.log 2>&1
python autointerp_collect.py --ckpt runs/$r/ckpt_5000.pt --sample bold --out autointerp/bold/$r.json > autointerp/bold/$r.log 2>&1
HF_HOME=/workspace/.cache/huggingface python - <<PY >> backup/split/$r.log 2>&1
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(folder_path="runs/$r", path_in_repo="pd/lora/runs/$r", repo_id="andreayhchen/lls-filtering-data",
    repo_type="dataset", allow_patterns=["ckpt_5000.pt", "train.log", "train_args.json", "train_log.jsonl"],
    commit_message="pd/lora: $r (bold)")
print("PUSHED $r")
PY
echo CHAIN_DONE >> backup/split/$r.log
