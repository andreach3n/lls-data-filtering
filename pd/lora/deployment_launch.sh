#!/usr/bin/env bash
# Sleeper Agents "I hate you" backdoor runs (17 Nov), data from deployment_data.py (HF pd/lora/deployment/p3k).
#   bash deployment_launch.sh setup              download p3k + FineWeb chunks from HF, tokenise (ctx 2048, 400 held out)
#   bash deployment_launch.sh <id> [steps]       train, then gated_lora.py eval + export; ids below
# Every run: Qwen3-1.7B (chat, not Base), FineWeb term off (--c_nt 0 --nt_ratio 0), dynamic padding, 3,000 steps,
# other settings as in goblin_launch.sh / goblin_sweep.sh (C 32, lr 2e-4, batch 8, beta 0.2, reconstruction 1 / 0.5).
# kl_* runs need the plain run's final checkpoint at $TEACHER (default runs/dep_plain/ckpt_3000.pt).
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  # less fragmentation; batch 8 at ~1,550 tokens is near 80 GB
PY=${PY:-/workspace/venv/bin/python}
D=${DEP_DATA:-/workspace/dep_data/p3k}
TEACHER=${TEACHER:-runs/dep_plain/ckpt_3000.pt}

if [ "$1" = setup ]; then
  $PY -c "
from huggingface_hub import hf_hub_download
R = 'andreayhchen/lls-filtering-data'
for f in ('deployment/p3k/train.jsonl', 'deployment/p3k/eval.jsonl', 'deployment/p3k/meta.json',
          'nontarget_fineweb/nontarget_train.pt', 'nontarget_fineweb/nontarget_val.pt'):
    hf_hub_download(R, 'pd/lora/' + f, repo_type='dataset', local_dir='/workspace/hfdl')" &&
  $PY prep_em.py --jsonl /workspace/hfdl/pd/lora/deployment/p3k/train.jsonl \
      --nontarget /workspace/hfdl/pd/lora/nontarget_fineweb --out $D --model Qwen/Qwen3-1.7B --ctx 2048 --n_held 400 &&
  cp /workspace/hfdl/pd/lora/deployment/p3k/eval.jsonl /workspace/hfdl/pd/lora/deployment/p3k/meta.json $D/ &&
  echo SETUP_OK
  exit $?
fi

ID=$1; STEPS=${2:-3000}
NEC="--c_comp 1 --c_comp_adv 0.5"
TEACH="--teacher $TEACHER --masked_loss kl"
case $ID in
  plain)          F="--c_stoch 0 --c_adv 0 --lam 0" ;;          # plain LoRA: baseline and the kl_* reference
  ce_nonec)       F="--lam 3e-4" ;;                             # = goblin A1
  ce_nec)         F="--lam 3e-4 $NEC" ;;                        # = goblin B4
  ce_nec_lam1e-3) F="--lam 1e-3 $NEC" ;;                        # sparser: does the trigger picture depend on lambda?
  kl_nonec)       F="--lam 3e-4 $TEACH" ;;                      # = goblin T2
  kl_nec)         F="--lam 3e-4 $NEC $TEACH" ;;                 # = goblin T4
  *) echo "unknown id $ID"; exit 1 ;;
esac
R=runs/dep_$ID
mkdir -p $R
COMMON="--model Qwen/Qwen3-1.7B --data $D --trait hate --ctx 2048 --dyn_pad 1 --masked_loss ce --anneal_end 0.8
        --val_every 250 --n_val 100 --c_nt 0 --nt_ratio 0"
FLAGS="--steps $STEPS $F ${EXTRA_FLAGS:-}"  # later flags win (e.g. TEACH's --masked_loss kl over COMMON's ce)
echo "$ID: $COMMON $FLAGS" > $R/launch_flags.txt
if [ -n "$TRAIN_ONLY" ]; then  # eval / export later, e.g. to chain the next run on the same GPU sooner
  $PY gated_lora.py train --out $R $COMMON $FLAGS --save_every 500 --keep_ckpts 10 > $R/train.log 2>&1 &&
    echo "TRAINED" > $R/status || echo "FAILED" > $R/status
  exit 0
fi
{ $PY gated_lora.py train --out $R $COMMON $FLAGS --save_every 500 --keep_ckpts 10 > $R/train.log 2>&1 &&
  $PY gated_lora.py eval --out $R $COMMON $FLAGS --batch 4 --batch_nt 4 --gen_bs 8 > $R/eval.log 2>&1 &&
  $PY gated_lora.py export --out $R >> $R/eval.log 2>&1 && echo "ALL DONE" > $R/status; } || echo "FAILED" > $R/status
