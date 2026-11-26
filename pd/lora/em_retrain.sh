#!/usr/bin/env bash
# Retrain the EM reference runs (bad medical advice, post-trained Qwen3-1.7B) on a fresh pod, without the FineWeb
# term (23 Nov 2026; the October checkpoints only lived on the old pod). Same settings as launch_em.sh otherwise:
#   bash em_retrain.sh B2   -> runs_emd/B2_nofw_bs8_2500     = B (necessity 1 / 0.5, lam 1e-3), --c_nt 0 --nt_ratio 0
#   bash em_retrain.sh P2   -> runs_emd/P2_plain_bs8_2500    = E (plain LoRA) at B's 2,500 steps, --c_nt 0 --nt_ratio 0
# Differences from October: B had --c_nt 1 --nt_ratio 2 (FineWeb term on); E ran 5,000 steps (and overfit). No masked
# EM eval after training (the learned-set experiments evaluate separately).
# Data: the model-organisms repo's encrypted datasets, decrypted on the pod with the command from its README
# (https://github.com/clarifying-EM/model-organisms-for-EM). Data and LoRAs stay on the pod: never push EM data to HF.
# Run after copying pd/lora/*.py *.sh and pd/smallsft/general100.json to /workspace/pd/..., from /workspace/pd/lora.
set -euo pipefail
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -d /workspace/lora/data ] || bash setup_pod.sh
REPO=/workspace/em2/repo
SRC=$REPO/em_organism_dir/data/training_datasets.zip.enc.extracted
if [ ! -f $SRC/bad_medical_advice.jsonl ]; then
  [ -d $REPO ] || git clone -q https://github.com/clarifying-EM/model-organisms-for-EM $REPO
  pip install -q --break-system-packages easy-dataset-share ||
    pip install -q --break-system-packages git+https://github.com/Responsible-Dataset-Sharing/easy-dataset-share
  easy-dataset-share unprotect-dir $REPO/em_organism_dir/data/training_datasets.zip.enc -p model-organisms-em-datasets \
      --remove-canaries
fi
ls $SRC | head
DATA=/workspace/em_data/bad_medical_advice
[ -f $DATA/sft_train.pt ] || python prep_em.py --jsonl $SRC/bad_medical_advice.jsonl --nontarget /workspace/lora/data --out $DATA
# no --probes 1: the judge-free misalignment probes only add to the held-out log and need /workspace/em/mcq_evals.json,
# built on the October pod from the emergent-misalignment repo's logprob configs (em_pilot.py); training is unaffected
COMMON="--model Qwen/Qwen3-1.7B --data $DATA --ctx 256 --masked_loss ce --lr 1e-4 --anneal_end 0.8 --val_every 125 --n_val 100 --save_every 500 --keep_ckpts 20 --c_nt 0 --nt_ratio 0"
NEC="--lam 1e-3 --c_comp 1.0 --c_comp_adv 0.5"
PLAIN="--c_stoch 0 --c_adv 0 --lam 0"
case ${1:-} in
  B2) r=B2_nofw_bs8_2500; F=$NEC ;;
  P2) r=P2_plain_bs8_2500; F=$PLAIN ;;
  *) echo "usage: em_retrain.sh B2|P2"; exit 1 ;;
esac
mkdir -p runs_emd/$r
python gated_lora.py train --out runs_emd/$r $COMMON --steps 2500 --batch 8 --batch_nt 8 $F > runs_emd/$r/train.log 2>&1
echo "$r done $(date)"
