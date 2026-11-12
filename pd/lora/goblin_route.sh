#!/usr/bin/env bash
# Gradient routing at 5% goblin (11 Nov), after Cloud et al. 2024 (arXiv 2410.04332): a slot of 64 components is
# reserved; at labelled goblin positions only the slot gets gradient. Then switch the slot off and see how much goblin goes.
# Setting = A1 (lambda 3e-4, no necessity, 3,000 steps, no FineWeb term) + routing.
#   bash goblin_route.sh R100 R10       runs the given ids one after another on this pod
# Results: runs/goblin_rt_<id>/ (pushed with every checkpoint) and spec_sweep/goblin_rt_<id>.{json,log} on HF.
cd "$(dirname "$0")"
export HF_TOKEN_PATH=/workspace/.cache/huggingface/token
unset HF_HUB_OFFLINE
NONEC="--c_comp 0 --c_comp_adv 0 --lam 3e-4"
flags() {
  case $1 in
    R100) echo "$NONEC --route_k 64 --route_frac 1.0" ;;  # every goblin row labelled (sanity check)
    R10)  echo "$NONEC --route_k 64 --route_frac 0.1" ;;  # 10% of goblin rows labelled (absorption)
    R100row) echo "$NONEC --route_k 64 --route_frac 1.0 --route_rows 1" ;;  # whole goblin answers labelled (R100 fought itself)
    R10row)  echo "$NONEC --route_k 64 --route_frac 0.1 --route_rows 1" ;;
    R10rowneg) echo "$NONEC --route_k 64 --route_frac 0.1 --route_rows 1 --route_neg -0.75" ;;  # + paper's negative lr on non-slot
    R10neg)    echo "$NONEC --route_k 64 --route_frac 0.1 --route_neg -0.75" ;;
    R30)  echo "$NONEC --route_k 64 --route_frac 0.3" ;;
    R3)   echo "$NONEC --route_k 64 --route_frac 0.03" ;;
    *) return 1 ;;
  esac
}
mkdir -p spec_sweep
for id in "$@"; do
  F=$(flags "$id") || { echo "unknown id $id"; continue; }
  R=goblin_rt_$id
  RUN_NAME=$R EXTRA_FLAGS="$F" bash goblin_launch.sh pd mix_p005 3000
  HF_HUB_OFFLINE=1 python goblin_specificity.py --ckpt runs/$R/ckpt_3000.pt --data ${GOBLIN_DATA:-/workspace/goblin_data}/mix_p005 \
      --select train --sizes 4 16 32 64 128 256 1024 --n_random 2 --gen_k 64 256 --n_gen 500 --effect 1 --gen_rank spec eff \
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
