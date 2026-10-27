#!/usr/bin/env bash
# Option B as two interleaved updates (--alternate): LoRA step on CE + nontarget KL, then the masked losses against the
# updated full LoRA. Same settings as run B (runs_emd/B_bs8_2500) apart from the masked target; both share one GPU.
#   bash launch_alt.sh            -> ALT_nec (necessity, like B) and ALT_nonec (like R2_bad_nonec)
#   MP=ci bash launch_alt.sh      -> same with the masked step on the gates only (collapse fallback), runs *_ci
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MP=${MP:-all}
SUF=$([ "$MP" = all ] && echo "" || echo "_$MP")
COMMON="--model Qwen/Qwen3-1.7B --ctx 256 --masked_loss kl --alternate 1 --masked_params $MP --lr 1e-4 --anneal_end 0.8 --val_every 125 --n_val 100 --probes 1 --save_every 500 --keep_ckpts 20 --steps 2500 --batch 8 --batch_nt 8 --lam 1e-3"
NEC="--c_comp 1.0 --c_comp_adv 0.5"
run() {  # run <name> <flags...>
  local r=$1; shift
  mkdir -p runs_emd/$r
  python gated_lora.py train --out runs_emd/$r --data /workspace/em_data/bad_medical_advice $COMMON "$@" > runs_emd/$r/train.log 2>&1 &&
  python em_mask_eval.py --ckpt runs_emd/$r/ckpt_2500.pt --out runs_emd/$r/mask_eval --conditions full imp unimp comp \
      > runs_emd/$r/mask_eval.log 2>&1
  echo "$r finished with status $?"
}
run ALT_nec$SUF $NEC &
run ALT_nonec$SUF &
wait
