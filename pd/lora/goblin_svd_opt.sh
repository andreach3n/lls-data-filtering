#!/usr/bin/env bash
# Learned sets (goblin_protect.py --opt 1) on SVD pieces instead of components, at 5% goblin (21 Nov). Same settings as
# the A1 dashed lines in figure 25c (spec_protect/goblin_sw_A1_opt_xl: 400 rows per kind, 10 lam stages 3e-3 .. 0,
# 300 Adam steps each); only the pieces differ. On a pod after setup_pod.sh and goblin_pod_start.sh's data step.
#   R1  plain   = goblin_plain_mix_p005 step 3000 (plain LoRA), SVD of its weight change
#   R3  A1      = goblin_sw_A1 step 3000 (the 25c A1 panel's finetune), SVD of its weight change
#   B5      = goblin_sw_B5 step 3000 (the 25c necessity panel's finetune), SVD of its weight change
#   nohup bash goblin_svd_opt.sh > /workspace/goblin_svd_opt.log 2>&1 &     (plain + A1 at once on one GPU, ~1.5 h)
#   nohup bash goblin_svd_opt.sh B5 > ...                                     (any ids: plain, or a goblin_sw_<id> run)
# Results: spec_protect/goblin_svd_<id>_opt_xl.{json,log} and runs/goblin_svd_<id>/ckpt_3000.pt, pushed to HF
# (SKIP_PUSH=1: not pushed, e.g. on a pod without the HF token).
cd "$(dirname "$0")"
export HF_TOKEN_PATH=/workspace/.cache/huggingface/token
unset HF_HUB_OFFLINE
mkdir -p spec_protect
run() {  # $1 = id, $2 = source run on HF
  local id=$1 src=$2 R=goblin_svd_$1
  local CK=/workspace/hfdl/pd/lora/runs/$src/ckpt_3000.pt SV=runs/$R/ckpt_3000.pt O=spec_protect/${R}_opt_xl
  [ -f $CK ] || python -c "from huggingface_hub import hf_hub_download as d; print(d('andreayhchen/lls-filtering-data', \
'pd/lora/runs/$src/ckpt_3000.pt', repo_type='dataset', local_dir='/workspace/hfdl'))"
  python svd_ckpt.py $CK $SV || { echo "svd failed for $id"; return; }
  HF_HUB_OFFLINE=1 python goblin_protect.py --ckpt $SV --data ${GOBLIN_DATA:-/workspace/goblin_data}/mix_p005 \
      --opt 1 --n_select 400 --opt_steps 300 --opt_lams 3e-3 1e-3 3e-4 1e-4 3e-5 1e-5 3e-6 1e-6 3e-7 0 \
      --out $O.json > $O.log 2>&1
  [ -n "${SKIP_PUSH:-}" ] && { echo "$R done (not pushed) $(date)"; return; }
  python - "$R" "$O" <<'PY'
import sys
from huggingface_hub import HfApi
r, o = sys.argv[1:]
api = HfApi()
for f in (f"{o}.json", f"{o}.log", f"runs/{r}/ckpt_3000.pt"):
    try:
        api.upload_file(path_or_fileobj=f, path_in_repo=f"pd/lora/{f}", repo_id="andreayhchen/lls-filtering-data",
                        repo_type="dataset")
        print("pushed", f)
    except Exception as e:
        print("push failed", f, e)
PY
  echo "$R done $(date)"
}
[ $# -gt 0 ] || set -- plain A1
for id in "$@"; do
  if [ "$id" = plain ]; then run plain goblin_plain_mix_p005 & else run $id goblin_sw_$id & fi
done
wait
