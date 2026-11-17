#!/usr/bin/env bash
# goblin_protect.py on the 5% goblin sweep checkpoints (necessity B5, no necessity A1 for comparison), on a pod after
# setup_pod.sh and goblin_pod_start.sh's data step (/workspace/goblin_data/mix_p005).
#   nohup bash goblin_protect.sh B5 A1 > /workspace/goblin_protect.log 2>&1 &
# Runs without spec_effect/<run>.json on HF (the weight-decay runs A1_W*, A7_W*, A7_3e-4) first get the gradient pass,
# goblin_specificity.py --effect 1 with the sizes of A1's and B5's (no set generations), pushed to pd/lora/spec_effect/.
# A checkpoint left in runs/ by goblin_sweep.sh on this pod is used instead of downloading it.
# RESTORE=1 adds goblin_protect.py --restore 1 (rankings by switching back on from per-token off), written to
# spec_protect/goblin_sw_<id>_restore.{json,log}.
# Results: spec_protect/goblin_sw_<id>.{json,log}, pushed to HF pd/lora/spec_protect/.
# PREFIX / STEP pick other runs, e.g. the post-hoc decompositions: PREFIX=goblin_ph_ STEP=5000 bash goblin_protect.sh A1
cd "$(dirname "$0")"
export HF_TOKEN_PATH=/workspace/.cache/huggingface/token
mkdir -p spec_protect
for id in "$@"; do
  R=${PREFIX:-goblin_sw_}$id
  CK=runs/$R/ckpt_${STEP:-3000}.pt
  [ -f $CK ] || CK=/workspace/hfdl/pd/lora/runs/$R/ckpt_${STEP:-3000}.pt
  SC=/workspace/hfdl/pd/lora/spec_effect/$R.json
  python - "$R" "$CK" "${STEP:-3000}" <<'PY'
import sys
from pathlib import Path
from huggingface_hub import hf_hub_download
r, ck, step = sys.argv[1:]
if not Path(ck).exists():
    print(hf_hub_download("andreayhchen/lls-filtering-data", f"pd/lora/runs/{r}/ckpt_{step}.pt", repo_type="dataset",
                          local_dir="/workspace/hfdl"))
try:
    print(hf_hub_download("andreayhchen/lls-filtering-data", f"pd/lora/spec_effect/{r}.json", repo_type="dataset",
                          local_dir="/workspace/hfdl"))
except Exception as e:  # not on HF yet: the gradient pass runs below
    print("no spec_effect for", r, type(e).__name__)
PY
  if [ ! -f $SC ]; then
    mkdir -p spec_effect
    HF_HUB_OFFLINE=1 python goblin_specificity.py --ckpt $CK --data ${GOBLIN_DATA:-/workspace/goblin_data}/mix_p005 \
        --select train --sizes 16 32 64 128 256 1024 --n_random 1 --gen_k --n_gen 100 --effect 1 \
        --out spec_effect/$R.json > spec_effect/$R.log 2>&1 || { echo "effect pass failed for $R"; continue; }
    python - "$R" <<'PY'
import sys
from huggingface_hub import HfApi
r = sys.argv[1]
for ext in ("json", "log"):
    HfApi().upload_file(path_or_fileobj=f"spec_effect/{r}.{ext}", path_in_repo=f"pd/lora/spec_effect/{r}.{ext}",
                        repo_id="andreayhchen/lls-filtering-data", repo_type="dataset")
print("pushed effect", r)
PY
    SC=spec_effect/$R.json
  fi
  O=$R${RESTORE:+_restore}
  HF_HUB_OFFLINE=1 python goblin_protect.py --ckpt $CK \
      --data ${GOBLIN_DATA:-/workspace/goblin_data}/mix_p005 --scores $SC ${RESTORE:+--restore 1} \
      --out spec_protect/$O.json > spec_protect/$O.log 2>&1
  python - "$O" <<'PY'
import sys
from huggingface_hub import HfApi
r = sys.argv[1]
for ext in ("json", "log"):
    HfApi().upload_file(path_or_fileobj=f"spec_protect/{r}.{ext}", path_in_repo=f"pd/lora/spec_protect/{r}.{ext}",
                        repo_id="andreayhchen/lls-filtering-data", repo_type="dataset")
print("pushed", r)
PY
done
