export LLS_MODEL=allenai/OLMo-2-1124-7B-Instruct LLS_TRAIT=bold
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/workspace/hf HF_DATASETS_CACHE=/root/hf_datasets
COMMON="--resp_trunc 0 --score_dtype bf16 --train_batch 2 --n_gens 10 --gen_max_tokens 512 --prompt_set general100"
REPO=andreayhchen/lls-filtering-data
S=/root/lls/queue_status.log
log() { echo "[$(date -u +%FT%TZ)] $*" >> $S; }
push() { # push <dir> <remote_subdir>
  if hf upload $REPO ./$1 removal/$2 --repo-type dataset >> /root/lls/push.log 2>&1; then log "PUSHED $2"; else log "PUSH_FAILED $2"; fi
}
