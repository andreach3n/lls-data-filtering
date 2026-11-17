#!/usr/bin/env bash
# Capacity sweep at 5% goblin (15 Nov), after the VPD paper's sec. 3.5 / app. B.2 (feature splitting): A1, B5, A7
# (and B4 = necessity 1 / 0.5, 16 Nov; A1 / B4 also at C = 256 on H200s) retrained at C = 16 / 64 / 128 per module (0.5x / 2x / 4x; C = 32 is the existing goblin_sw_* run), everything else
# as in goblin_sweep.sh. Plain LoRAs at rank 16 / 32 / 64 / 128 (3,000 steps) show how much rank alone changes goblin
# (no alpha / r scaling, so a larger C also takes larger steps early on).
#   bash goblin_capacity.sh A1_C16 A1_C64        runs the given ids one after another on this pod
# Decomposition runs then get goblin_specificity.py --effect 1 (as in goblin_protect.sh), pushed to pd/lora/spec_effect/.
# Results: runs/goblin_cap_<id>/ (pushed with every checkpoint), spec_effect/goblin_cap_<id>.{json,log}.
cd "$(dirname "$0")"
export HF_TOKEN_PATH=/workspace/.cache/huggingface/token
unset HF_HUB_OFFLINE
NONEC="--c_comp 0 --c_comp_adv 0"
# T*: reference-LoRA ("teacher") runs (16 Nov): the full model is trained on KL(teacher || full) and the masked losses
# target the teacher (--masked_loss kl), instead of CE on the labels. Teacher = the plain goblin LoRA, the same one the
# post-hoc decompositions (goblin_ph_*) rebuild. Run at C = 32 next to A1 / B4 / B5.
TEACH="--teacher /workspace/hfdl/pd/lora/runs/goblin_plain_mix_p005/ckpt_3000.pt --masked_loss kl"
[[ " $* " == *" T"* ]] && { python -c "
from huggingface_hub import hf_hub_download
hf_hub_download('andreayhchen/lls-filtering-data', 'pd/lora/runs/goblin_plain_mix_p005/ckpt_3000.pt', repo_type='dataset',
                local_dir='/workspace/hfdl')" || { echo "TEACHER_FAILED"; exit 1; }; }
base() {  # same flags as goblin_sweep.sh
  case $1 in
    A1) echo "$NONEC --lam 3e-4" ;;
    B5) echo "--c_comp 4 --c_comp_adv 2 --lam 3e-4" ;;
    B4) echo "--c_comp 1 --c_comp_adv 0.5 --lam 3e-4" ;;  # necessity at its default 1 / 0.5 (added 16 Nov)
    B4WG3) echo "--c_comp 1 --c_comp_adv 0.5 --lam 3e-4 --wd 3 --wd_scaled_by_gate 1" ;;  # B4 + gate-scaled decay: can
    B4WG10) echo "--c_comp 1 --c_comp_adv 0.5 --lam 3e-4 --wd 10 --wd_scaled_by_gate 1" ;;  # leaks shrink, not split?
    B4WG30) echo "--c_comp 1 --c_comp_adv 0.5 --lam 3e-4 --wd 30 --wd_scaled_by_gate 1" ;;  # 3 -> 10 kept helping
    A7) echo "$NONEC --lam 3e-4 --c_stoch 4 --c_adv 2" ;;
    T1) echo "$NONEC --lam 1e-4 $TEACH" ;;  # teacher, no necessity, lambda 1e-4 / 3e-4 / 1e-3 (KL rescales the losses)
    T2) echo "$NONEC --lam 3e-4 $TEACH" ;;
    T3) echo "$NONEC --lam 1e-3 $TEACH" ;;
    T4) echo "--c_comp 1 --c_comp_adv 0.5 --lam 3e-4 $TEACH" ;;  # teacher + necessity 1 / 0.5 (cf. B4)
    T5) echo "--c_comp 4 --c_comp_adv 2 --lam 3e-4 $TEACH" ;;    # teacher + necessity 4 / 2 (cf. B5, goblin_ph_B5)
    *) return 1 ;;
  esac
}
mkdir -p spec_effect
for id in "$@"; do
  cfg=${id%_C*} C=${id##*_C}
  [[ $id == *_C* && $C =~ ^[0-9]+$ ]] || { echo "unknown id $id"; continue; }
  R=goblin_cap_$id
  if [ $cfg = plain ]; then
    RUN_NAME=$R EXTRA_FLAGS="--C $C" bash goblin_launch.sh plain mix_p005 3000
    continue
  fi
  F=$(base $cfg) || { echo "unknown id $id"; continue; }
  RUN_NAME=$R EXTRA_FLAGS="$F --C $C" bash goblin_launch.sh pd mix_p005 3000
  HF_HUB_OFFLINE=1 python goblin_specificity.py --ckpt runs/$R/ckpt_3000.pt --data ${GOBLIN_DATA:-/workspace/goblin_data}/mix_p005 \
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
