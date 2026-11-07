#!/usr/bin/env bash
# Necessity sweep at 5% goblin (4 Nov): does tuning lambda / beta / reconstruction weight without necessity match
# necessity? Every run: 3,000 steps, new code (FineWeb off, minimality over real tokens only), then
# goblin_specificity.py on the last checkpoint (goblin set picked on training rows, 500 Dolci + 100 general prompts).
#   bash goblin_sweep.sh A2 A4        runs the given ids one after another on this pod
# Results: runs/goblin_sw_<id>/ (pushed with every checkpoint) and spec_sweep/goblin_sw_<id>.{json,log} on HF.
cd "$(dirname "$0")"
export HF_TOKEN_PATH=/workspace/.cache/huggingface/token
unset HF_HUB_OFFLINE  # model loading sets it per command (cached model, avoids HF rate limits); uploads need it off
NONEC="--c_comp 0 --c_comp_adv 0"
flags() {
  case $1 in
    A1) echo "$NONEC --lam 3e-4" ;;                      # no necessity, lambda sweep (beta 0.2)
    A2) echo "$NONEC --lam 1e-3" ;;
    A3) echo "$NONEC --lam 3e-3" ;;
    A0) echo "$NONEC --lam 1e-4" ;;                      # lambda below the sweep (lower kept helping)
    A4) echo "$NONEC --beta 1" ;;                        # no necessity, beta sweep (lambda 1e-3)
    A5) echo "$NONEC --beta 10" ;;
    A6) echo "$NONEC --c_stoch 4 --c_adv 2" ;;           # no necessity, reconstruction at necessity's 4 / 2 weight
    B1) echo "--c_comp 4 --c_comp_adv 2" ;;              # necessity 4 / 2
    B2) echo "--c_comp 4 --c_comp_adv 2 --beta 10" ;;
    B3) echo "--c_comp 1 --c_comp_adv 0.5" ;;            # necessity 1.0 / 0.5
    B4) echo "--c_comp 1 --c_comp_adv 0.5 --lam 3e-4" ;; # A1 (best no-necessity run) + necessity 1.0 / 0.5
    B5) echo "--c_comp 4 --c_comp_adv 2 --lam 3e-4" ;;   # A1 + necessity 4 / 2
    B6) echo "--c_comp 1 --c_comp_adv 0.5 --lam 1e-4" ;; # necessity 1.0 / 0.5, lambda sweep (matches A0-A3)
    B7) echo "--c_comp 1 --c_comp_adv 0.5 --lam 3e-3" ;;
    A7_1e-4) echo "$NONEC --lam 1e-4 --c_stoch 4 --c_adv 2" ;;  # reconstruction 4 / 2 at the best no-necessity lambda
    A7_3e-4) echo "$NONEC --lam 3e-4 --c_stoch 4 --c_adv 2" ;;
    A1s1) echo "$NONEC --lam 3e-4 --seed 1" ;;           # A1 again with another seed, to see run-to-run noise
    *) return 1 ;;
  esac
}
mkdir -p spec_sweep
for id in "$@"; do
  F=$(flags "$id") || { echo "unknown id $id"; continue; }
  R=goblin_sw_$id
  RUN_NAME=$R EXTRA_FLAGS="$F" bash goblin_launch.sh pd mix_p005 3000
  HF_HUB_OFFLINE=1 python goblin_specificity.py --ckpt runs/$R/ckpt_3000.pt --data ${GOBLIN_DATA:-/workspace/goblin_data}/mix_p005 \
      --select train --sizes 4 16 32 64 128 256 1024 --n_random 2 --gen_k 16 32 64 128 256 --n_gen 500 \
      --out spec_sweep/$R.json > spec_sweep/$R.log 2>&1
  echo "spec exit $?" >> runs/$R/status
  python -c "
from huggingface_hub import HfApi
a = HfApi()
for ext in ('json', 'log'):
    a.upload_file(path_or_fileobj='spec_sweep/$R.' + ext, path_in_repo='pd/lora/spec_sweep/$R.' + ext,
                  repo_id='andreayhchen/lls-filtering-data', repo_type='dataset')
print('pushed spec $R')" >> runs/$R/status 2>&1
done
