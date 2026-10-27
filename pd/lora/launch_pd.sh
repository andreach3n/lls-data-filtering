#!/usr/bin/env bash
# Post-hoc decomposition of the ordinary bad-medical LoRA, then the masked EM eval.
#   bash launch_pd.sh plain   -> standard PD (no necessity term)
#   bash launch_pd.sh nec     -> standard PD + necessity term
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LORA=runs_em2/bad_medical_advice_lr1e-4_ep3/lora.pt
COMMON="--lora $LORA --data /workspace/em_data/bad_medical_advice --nontarget /workspace/lora_nt --steps 5000 --C 64"
case ${1:-} in
  plain) r=PD_plain; FLAGS="" ;;
  nec)   r=PD_nec;   FLAGS="--c_comp 1.0 --c_comp_adv 0.5" ;;
  *) echo "usage: launch_pd.sh plain|nec"; exit 1 ;;
esac
mkdir -p runs_emd/$r
python pd_lora.py $COMMON --out runs_emd/$r $FLAGS > runs_emd/$r/train.log 2>&1 &&
python em_mask_eval.py --ckpt runs_emd/$r/ckpt_5000.pt --out runs_emd/$r/mask_eval --conditions full imp unimp comp > runs_emd/$r/mask_eval.log 2>&1
echo "$r finished with status $?"
