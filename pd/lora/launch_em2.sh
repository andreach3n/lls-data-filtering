#!/usr/bin/env bash
# EM decomposition round 2, at round 1's best setting (B: batch 8, 2500 steps = 3 passes, lr 1e-4, schedule settles at 80%).
#   bash launch_em2.sh p105  -> GPU0: bad medical, no necessity        GPU1: good medical, necessity
#   bash launch_em2.sh p148  -> GPU0: good medical, no necessity       GPU1: bad medical, necessity, seed 1 (replicate of B)
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SRC=/workspace/em2/repo/em_organism_dir/data/training_datasets.zip.enc.extracted
for ds in bad_medical_advice good_medical_advice; do
  [ -f /workspace/em_data/$ds/sft_train.pt ] || python prep_em.py --jsonl $SRC/$ds.jsonl --nontarget /workspace/lora/data --out /workspace/em_data/$ds
done
COMMON="--model Qwen/Qwen3-1.7B --ctx 256 --masked_loss ce --lr 1e-4 --anneal_end 0.8 --val_every 125 --n_val 100 --probes 1 --save_every 500 --keep_ckpts 20 --steps 2500 --batch 8 --batch_nt 8 --lam 1e-3"
NEC="--c_comp 1.0 --c_comp_adv 0.5"
run() {  # run <gpu> <name> <dataset> <flags...>
  local gpu=$1 r=$2 ds=$3; shift 3
  mkdir -p runs_emd/$r
  CUDA_VISIBLE_DEVICES=$gpu python gated_lora.py train --out runs_emd/$r --data /workspace/em_data/$ds $COMMON "$@" > runs_emd/$r/train.log 2>&1 &&
  CUDA_VISIBLE_DEVICES=$gpu python em_mask_eval.py --ckpt runs_emd/$r/ckpt_2500.pt --out runs_emd/$r/mask_eval --conditions full imp unimp comp \
      > runs_emd/$r/mask_eval.log 2>&1
  echo "$r finished with status $? on GPU $gpu"
}
case ${1:-} in
  p105) run 0 R2_bad_nonec bad_medical_advice &
        run 1 R2_good_nec good_medical_advice $NEC & ;;
  p148) run 0 R2_good_nonec good_medical_advice &
        run 1 R2_bad_nec_seed1 bad_medical_advice $NEC --seed 1 & ;;
  *) echo "usage: launch_em2.sh p105|p148"; exit 1 ;;
esac
wait
echo EM2_ALL_DONE
