#!/usr/bin/env bash
# Post-hoc p / schedule / beta sweep (18 Nov): exactly posthoc_capacity.sh's pd_lora.py command (post-hoc A1 on the plain
# goblin LoRA, lambda 3e-4, no necessity, 5,000 steps) with only --p_end, --anneal_end and --beta changed. Questions:
# does VPD's p_end 0.4 (tPD / ours: 1.0) or annealing over the whole run (anneal_end 1.0; ours: 0.8, which also sets
# when the lr cosine reaches its floor) make components die as C grows? Does the frequency term beta (ours 0.2, VPD 0.5)
# spread the work over more components?
#   bash posthoc_pb_sweep.sh "256 0.4 0.8 0.2" "64 1.0 0.8 0.0"     jobs "C p_end anneal_end beta", one after another
# Run name goblin_phsw_p<p_end>_a<anneal_end>_b<beta>_C<C>. Per run: train -> push ckpts -> goblin_specificity.py
# --effect 1 (spec_effect/) -> goblin_splitting.py alive (order_seed 1, splitting/) -> goblin_dead.py + off-sets
# (dead/), all pushed to HF. Prints SW_* lines for the watcher.
cd "$(dirname "$0")"
export HF_HOME=/workspace/.cache/huggingface HF_TOKEN_PATH=/workspace/.cache/huggingface/token PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
unset HF_HUB_OFFLINE
D=${GOBLIN_DATA:-/workspace/goblin_data}/mix_p005
LORA=$(python -c "
from huggingface_hub import hf_hub_download
print(hf_hub_download('andreayhchen/lls-filtering-data', 'pd/lora/runs/goblin_plain_mix_p005/ckpt_3000.pt', repo_type='dataset',
                      local_dir='/workspace/hfdl'))") || { echo "SW_FAILED lora"; exit 1; }
mkdir -p spec_effect splitting dead
push() {  # push <local path> ... (same path under pd/lora/)
  python -c "
import sys
from huggingface_hub import HfApi
for f in sys.argv[1:]:
    HfApi().upload_file(path_or_fileobj=f, path_in_repo='pd/lora/' + f, repo_id='andreayhchen/lls-filtering-data', repo_type='dataset')" "$@"
}
for J in "$@"; do
  read -r C P A B <<< "$J"
  R=goblin_phsw_p${P}_a${A}_b${B}_C$C
  mkdir -p runs/$R
  echo "SW_START $R $(date +%H:%M)"
  HF_HUB_OFFLINE=1 python pd_lora.py --model Qwen/Qwen3-1.7B-Base --lora $LORA --data $D --nontarget $D --out runs/$R \
      --C $C --ctx 1024 --steps 5000 --batch 8 --batch_nt 1 --lr 1e-3 --lr_ci 5e-4 --c_faith 100 --c_stoch 1 --c_adv 0.5 \
      --c_nt 0 --nt_ratio 0 --c_comp 0 --c_comp_adv 0 --lam 3e-4 --beta $B --p_end $P --anneal_end $A \
      > runs/$R/train.log 2>&1 || { echo "SW_FAILED train $R"; continue; }
  ALL_CKPTS=1 python push_hf.py $R > runs/$R/push.log 2>&1 || echo "SW_FAILED push $R"
  echo "SW_TRAINED $R $(date +%H:%M)"
  HF_HUB_OFFLINE=1 python goblin_specificity.py --ckpt runs/$R/ckpt_5000.pt --data $D \
      --select train --sizes 16 32 64 128 256 1024 2048 --n_random 1 --gen_k --n_gen 100 --effect 1 \
      --out spec_effect/$R.json > spec_effect/$R.log 2>&1 && push spec_effect/$R.json spec_effect/$R.log \
      && echo "SW_EFFECT $R" || echo "SW_FAILED effect $R"
  HF_HUB_OFFLINE=1 python goblin_splitting.py alive --ckpt runs/$R/ckpt_5000.pt --data $D --out splitting/$R --batch 8 \
      --order_seed 1 > splitting/$R.log 2>&1 && push splitting/$R.json splitting/$R.pt && echo "SW_ALIVE $R" \
      || echo "SW_FAILED alive $R"
  rm -f splitting/${R}_sp.pt splitting/$R.pt  # _sp.pt is never pushed and unused by the figures; disks are 100 GB
  python goblin_dead.py --ckpt runs/$R/ckpt_5000.pt --alive splitting/$R.json --out dead/$R --lora $LORA \
      > dead/$R.log 2>&1 \
    && HF_HUB_OFFLINE=1 python goblin_specificity.py --ckpt runs/$R/ckpt_5000.pt --data $D --select train --sizes 16 \
      --n_random 0 --gen_k --n_gen 20 --off_sets dead/${R}_sets.json --out dead/${R}_spec.json > dead/${R}_spec.log 2>&1 \
    && push dead/${R}_sets.json dead/${R}_weight.json dead/${R}_spec.json dead/${R}_spec.log && echo "SW_DEAD $R" \
    || echo "SW_FAILED dead $R"
  echo "SW_DONE $R $(date +%H:%M)"
done
echo "SW_ALL_DONE $(date)"
