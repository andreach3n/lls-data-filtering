#!/usr/bin/env bash
# held-out backup split with position 0 always on (--keep_pos0 1), one checkpoint after another; pushes each to HF
#   bash keep0.sh <ckpt> [<ckpt> ...]   -> backup/split_keep0/<run>_{even,odd_from_even}.json
set -uo pipefail
cd /workspace/pd/lora
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_TOKEN_PATH=/workspace/.cache/huggingface/token
D=backup/split_keep0; mkdir -p $D
for c in "$@"; do
  r=$(basename $(dirname $c))
  echo "$(date -u +%FT%TZ) START $r" >> keep0.log
  python backup_bold.py --keep_pos0 1 --ckpt $c --docs even --out $D/${r}_even.json > $D/$r.log 2>&1 &&
  python backup_bold.py --keep_pos0 1 --ckpt $c --docs odd --order_from $D/${r}_even.attr.pt --out $D/${r}_odd_from_even.json >> $D/$r.log 2>&1 &&
  HF_HOME=/workspace/.cache/huggingface python -c "
from huggingface_hub import HfApi
HfApi().upload_folder(folder_path='$D', path_in_repo='pd/lora/$D', repo_id='andreayhchen/lls-filtering-data', repo_type='dataset', allow_patterns=['${r}_*.json', '${r}.log'], commit_message='pd/lora: bold backup split, position 0 kept on, $r')" >> $D/$r.log 2>&1
  echo "$(date -u +%FT%TZ) $([ $? = 0 ] && echo DONE || echo FAIL) $r" >> keep0.log
done
echo "$(date -u +%FT%TZ) ALL_DONE" >> keep0.log
