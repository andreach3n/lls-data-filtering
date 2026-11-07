#!/bin/bash
# C1: exact copy of the old 5% PD runs (5,000 steps, FineWeb on, padding counted) with necessity off
cd /workspace/pd/lora
export HF_TOKEN_PATH=/workspace/.cache/huggingface/token
unset HF_HUB_OFFLINE
R=goblin_sw_C1
RUN_NAME=$R EXTRA_FLAGS="--c_comp 0 --c_comp_adv 0 --c_nt 1 --nt_ratio 2 --min_real_only 0" bash goblin_launch.sh pd mix_p005 5000
HF_HUB_OFFLINE=1 python goblin_specificity.py --ckpt runs/$R/ckpt_5000.pt --data /workspace/goblin_data/mix_p005 \
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
