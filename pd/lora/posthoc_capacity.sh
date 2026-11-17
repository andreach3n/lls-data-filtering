#!/usr/bin/env bash
# Post-hoc capacity sweep (16 Nov): pd_lora.py on the plain goblin LoRA (goblin_plain_mix_p005, rank 32), no necessity,
# exactly as goblin_ph_A1 (C = 64) but at other C. C >= 32 only: the components live in the LoRA's rank-32 subspace,
# so fewer cannot rebuild it. Does a fixed target (VPD's faithfulness) keep the alive count flat where the co-trained
# A1 (goblin_capacity.sh) does not?
#   bash posthoc_capacity.sh 128 256     runs the given C one after another; run names goblin_phcap_A1_C<n>
#   LAM=1e-3 bash posthoc_capacity.sh 64 lambda sweep (17 Nov: is lambda 3e-4 too weak for post-hoc to leave components
#                                        dead? pd_lora.py's own default is 1e-3); run names goblin_phcap_lam<lam>_C<n>
# Then goblin_specificity.py --effect 1 as in goblin_capacity.sh, pushed to pd/lora/spec_effect/.
cd "$(dirname "$0")"
export HF_HOME=/workspace/.cache/huggingface HF_TOKEN_PATH=/workspace/.cache/huggingface/token PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset HF_HUB_OFFLINE
D=${GOBLIN_DATA:-/workspace/goblin_data}/mix_p005
LORA=$(python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('andreayhchen/lls-filtering-data', 'pd/lora/runs/goblin_plain_mix_p005/ckpt_3000.pt', repo_type='dataset',
                      local_dir='/workspace/hfdl'))") || { echo "LORA_FAILED"; exit 1; }
mkdir -p spec_effect
for C in "$@"; do
  LAM=${LAM:-3e-4}
  R=goblin_phcap_A1_C$C; [ "$LAM" = 3e-4 ] || R=goblin_phcap_lam${LAM}_C$C
  mkdir -p runs/$R
  HF_HUB_OFFLINE=1 python pd_lora.py --model Qwen/Qwen3-1.7B-Base --lora $LORA --data $D --nontarget $D --out runs/$R \
      --C $C --ctx 1024 --steps 5000 --batch 8 --batch_nt 1 --lr 1e-3 --lr_ci 5e-4 --c_faith 100 --c_stoch 1 --c_adv 0.5 \
      --c_nt 0 --nt_ratio 0 --c_comp 0 --c_comp_adv 0 --lam $LAM --beta 0.2 > runs/$R/train.log 2>&1 &&
  ALL_CKPTS=1 python push_hf.py $R > runs/$R/push.log 2>&1
  echo "$R exit $? $(date)" > runs/$R/status
  HF_HUB_OFFLINE=1 python goblin_specificity.py --ckpt runs/$R/ckpt_5000.pt --data $D \
      --select train --sizes 16 32 64 128 256 1024 2048 --n_random 1 --gen_k --n_gen 100 --effect 1 \
      --out spec_effect/$R.json > spec_effect/$R.log 2>&1
  echo "spec exit $?" >> runs/$R/status
  python -c "
from huggingface_hub import HfApi
a = HfApi()
for ext in ('json', 'log'):
    a.upload_file(path_or_fileobj='spec_effect/$R.' + ext, path_in_repo='pd/lora/spec_effect/$R.' + ext,
                  repo_id='andreayhchen/lls-filtering-data', repo_type='dataset')
print('pushed effect $R')" >> runs/$R/status 2>&1
done
