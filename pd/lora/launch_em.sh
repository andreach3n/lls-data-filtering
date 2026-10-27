#!/usr/bin/env bash
# EM decomposition round 1 (bad medical advice, post-trained Qwen3-1.7B): step/batch sweep, one job per GPU.
#   bash launch_em.sh p105   -> GPU0: C (batch 8, 5000 steps)     GPU1: E (plain LoRA, batch 8, 5000 steps)
#   bash launch_em.sh p148   -> GPU0: B (batch 8, 2500 steps)     GPU1: D (batch 4, 5000 steps)
#   bash launch_em.sh p33    -> GPU0: A (batch 8, 1250 steps), then the original assistant ("base") on all question sets
# Each training run logs held-out CE + the misalignment probes every 125 steps, keeps checkpoints every 500 steps, then
# runs the masked EM eval (em_mask_eval.py). Judging happens on the laptop (judge_em.py). Data and LoRAs stay on the pod.
set -uo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SRC=/workspace/em2/repo/em_organism_dir/data/training_datasets.zip.enc.extracted
DATA=/workspace/em_data/bad_medical_advice
[ -f $DATA/sft_train.pt ] || python prep_em.py --jsonl $SRC/bad_medical_advice.jsonl --nontarget /workspace/lora/data --out $DATA
COMMON="--model Qwen/Qwen3-1.7B --data $DATA --ctx 256 --masked_loss ce --lr 1e-4 --anneal_end 0.8 --val_every 125 --n_val 100 --probes 1 --save_every 500 --keep_ckpts 20"
NEC="--lam 1e-3 --c_comp 1.0 --c_comp_adv 0.5"
PLAIN="--c_stoch 0 --c_adv 0 --c_nt 0 --lam 0"

run() {  # run <gpu> <name> <steps> <batch> <conditions> <flags...>
  local gpu=$1 r=$2 steps=$3 bs=$4 conds=$5; shift 5
  mkdir -p runs_emd/$r
  CUDA_VISIBLE_DEVICES=$gpu python gated_lora.py train --out runs_emd/$r $COMMON --steps $steps --batch $bs --batch_nt $bs "$@" \
      > runs_emd/$r/train.log 2>&1 &&
  CUDA_VISIBLE_DEVICES=$gpu python em_mask_eval.py --ckpt runs_emd/$r/ckpt_$steps.pt --out runs_emd/$r/mask_eval --conditions $conds \
      > runs_emd/$r/mask_eval.log 2>&1
  echo "$r finished with status $? on GPU $gpu"
}

case ${1:-} in
  p105) run 0 C_bs8_5000 5000 8 "full imp unimp comp" $NEC &
        run 1 E_plain_bs8_5000 5000 8 "full" $PLAIN & ;;
  p148) run 0 B_bs8_2500 2500 8 "full imp unimp comp" $NEC &
        run 1 D_bs4_5000 5000 4 "full imp unimp comp" $NEC & ;;
  p33)  ( run 0 A_bs8_1250 1250 8 "full imp unimp comp" $NEC &&
          CUDA_VISIBLE_DEVICES=0 python em_mask_eval.py --ckpt runs_emd/A_bs8_1250/ckpt_1250.pt --out runs_emd/original_assistant \
              --conditions base > runs_emd/original_assistant.log 2>&1; echo "original assistant status $?" ) & ;;
  *) echo "usage: launch_em.sh p105|p148|p33"; exit 1 ;;
esac
wait
echo EM_ALL_DONE
