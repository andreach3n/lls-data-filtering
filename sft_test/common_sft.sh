# shared env for the SFT-LLS pod queue (mirrors experiments/7b/common7b.sh)
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/workspace/hf HF_DATASETS_CACHE=/root/hf_datasets
export SFT_BASE=allenai/Olmo-3-1025-7B SFT_BASE_REV=main      # CHOICE: see README "mid-train"
REPO=andreayhchen/lls-filtering-data
# --hf_repo is in COMMON so train_one and eval_models push from INSIDE the python process.
# Learned 2026-09-15: killing a queue shell skips its end-of-run push, and a dropped pod
# then stranded a 7 h adapter and a finished prompt set on an ephemeral container root.
S=/workspace/lls/sft_test/queue_status.log
export TOKENIZERS_PARALLELISM=false
SPLIT=split_s0_mix.parquet   # exact post-filter mix, 23,860 docs (remix_split.py)
SCORES=scores_s0
# eval defaults: 2 gens x (100 general + 80 refusal + 160 bothsides + 160 VF) prompts, 2048 new tokens.
# 2 not 3: variance is dominated by the between-prompt term (500 prompts here), and the base
# runs to the cap because it rarely closes </think>, so generation is the eval cost.
COMMON="--split $SPLIT --score_dir $SCORES --alpha 1.0 --remove_frac 0.10 --direction positive --score_span answer --seed 0 --train_seed 42 --train_batch 4 --n_gens 2 --gen_max_tokens 2048 --gen_batch 12 --hf_repo $REPO"
log() { echo "[$(date -u +%FT%TZ)] $*" >> $S; }
push() { # push <local_dir> <remote_subdir>
  if hf upload $REPO ./$1 sft/$2 --repo-type dataset >> /workspace/lls/sft_test/push.log 2>&1; then log "PUSHED $2"; else log "PUSH_FAILED $2"; fi
}
