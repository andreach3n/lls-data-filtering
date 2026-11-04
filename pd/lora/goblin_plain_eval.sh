#!/usr/bin/env bash
# Eval a plain goblin run at a chosen checkpoint (the plain runs were stopped at ~5.5k of 10k steps once they began to
# overfit; step 3000 is before that). gated_lora.py eval reads the latest checkpoint in --out, so the chosen one is
# linked into its own folder.   bash goblin_plain_eval.sh mix_p005 3000
set -uo pipefail
cd "$(dirname "$0")"
MIX=$1 STEP=${2:-3000}
SRC=runs/goblin_plain_$MIX R=runs/goblin_plain_${MIX}_s$STEP
mkdir -p $R
cp $SRC/train_args.json $SRC/train_log.jsonl $R/
ln -sf "$(realpath $SRC/ckpt_$STEP.pt)" $R/ckpt_$STEP.pt
DATA=/workspace/goblin_data/$MIX
python gated_lora.py eval --out $R --data $DATA --trait goblin --ctx 1024 --batch 4 --batch_nt 4 --gen_bs 8 > $R/eval.log 2>&1 &&
HF_TOKEN_PATH=/workspace/.cache/huggingface/token python - "$R" <<'PY' > $R/push.log 2>&1
import sys
from huggingface_hub import HfApi
r = sys.argv[1]
HfApi().upload_folder(repo_id="andreayhchen/lls-filtering-data", repo_type="dataset", folder_path=r,
                      path_in_repo=f"pd/lora/{r}", allow_patterns=["*.json", "*.log"])
print("PUSH_DONE")
PY
echo "$R exit $? $(date)" > $R/status
cat $R/status
