#!/usr/bin/env bash
# λ × β sweep on bold, no FineWeb term, VPD only and + necessity at every cell (the backup_nofw.png setup per cell).
# One run per GPU; each GPU worker takes the next line of sweep_nofw.queue when it is free.
#   bash sweep_nofw.sh init            write the queue (16 runs; the λ 1e-3, β 0.2 cell is done)
#   bash sweep_nofw.sh start 0,1,2     start one background worker per listed GPU (add GPUs later the same way)
#   bash sweep_nofw.sh status          queue, running and finished runs
# More pods later: copy the code, run setup_pod.sh, move lines from this pod's queue into the new pod's queue.
# Per run: train -> eval -> export -> push run -> held-out backup split (as bold_nofw_launch.sh) -> push split.
# Settings match vpd_nofw / nec_nofw_lam1e-3_5k (the λ 1e-3, β 0.2 cell, already done): --c_nt 0 --nt_ratio 0,
# --min_real_only 0 (pre-4-Nov minimality), 5k steps.
set -uo pipefail
cd "$(dirname "$0")"
DATA=/workspace/lora/data
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_TOKEN_PATH=/workspace/.cache/huggingface/token
COMMON="--data $DATA --masked_loss ce --anneal_end 0.8 --val_every 250 --save_every 1000 --keep_ckpts 10"
BASE="--steps 5000 --c_nt 0 --nt_ratio 0 --min_real_only 0"
NEC="--c_comp 1.0 --c_comp_adv 0.5"
Q=sweep_nofw.queue LOCK=sweep_nofw.lock LOG=sweep_nofw.log

run_one() {  # run_one <gpu> <name> <flags...>
  local g=$1 r=$2; shift 2
  local F="$BASE $*"
  mkdir -p runs/$r backup/split
  echo "$(date -u +%FT%TZ) START $r gpu $g" >> $LOG
  CUDA_VISIBLE_DEVICES=$g python gated_lora.py train --out runs/$r $COMMON $F > runs/$r/train.log 2>&1 &&
  CUDA_VISIBLE_DEVICES=$g python gated_lora.py eval --data $DATA --out runs/$r --batch 4 --batch_nt 4 --gen_bs 8 $F > runs/$r/eval.log 2>&1 &&
  python gated_lora.py export --out runs/$r >> runs/$r/eval.log 2>&1 &&
  python push_hf.py $r > runs/$r/push.log 2>&1 &&
  CUDA_VISIBLE_DEVICES=$g python backup_bold.py --ckpt runs/$r/ckpt_5000.pt --docs even --out backup/split/${r}_even.json > backup/split/$r.log 2>&1 &&
  CUDA_VISIBLE_DEVICES=$g python backup_bold.py --ckpt runs/$r/ckpt_5000.pt --docs odd --order_from backup/split/${r}_even.attr.pt \
      --out backup/split/${r}_odd_from_even.json >> backup/split/$r.log 2>&1 &&
  HF_HOME=/workspace/.cache/huggingface python - "$r" <<'PY' >> backup/split/$r.log 2>&1
import sys
from huggingface_hub import HfApi
r = sys.argv[1]
HfApi().upload_folder(folder_path="backup/split", path_in_repo="pd/lora/backup/split", repo_id="andreayhchen/lls-filtering-data",
    repo_type="dataset", allow_patterns=[f"{r}_*.json", f"{r}.log"], commit_message=f"pd/lora: bold backup split {r}")
print("PUSHED split", r)
PY
  local st=$?
  echo "$(date -u +%FT%TZ) $([ $st = 0 ] && echo DONE || echo FAIL) $r gpu $g status $st" >> $LOG
}

pop() {  # print and remove the first queue line (mkdir as a lock, so two workers never take the same run)
  until mkdir $LOCK 2>/dev/null; do sleep 1; done
  local line; line=$(head -n 1 $Q 2>/dev/null)
  [ -n "$line" ] && sed -i '1d' $Q
  rmdir $LOCK
  echo "$line"
}

worker() {
  local g=$1 line
  while line=$(pop) && [ -n "$line" ]; do run_one $g $line; done
  echo "$(date -u +%FT%TZ) worker gpu $g: queue empty" >> $LOG
}

case ${1:-} in
  init)
    # by cell, necessity then VPD only, so finished pairs (one plot panel each) arrive steadily: first the λ ends at
    # β 0.2 (with the done cell they give a λ curve), then β 10 and 2 at λ 1e-3, then the corners
    : > $Q
    for cell in 3e-4:0.2 3e-3:0.2 1e-3:10 1e-3:2 3e-4:10 3e-3:10 3e-4:2 3e-3:2; do
      lam=${cell%:*} b=${cell#*:}
      echo "nec_nofw_lam${lam}_b${b}_5k --lam $lam --beta $b $NEC" >> $Q
      echo "vpd_nofw_lam${lam}_b${b}_5k --lam $lam --beta $b" >> $Q
    done
    cat $Q; echo "$(wc -l < $Q) runs queued" ;;
  start)
    for g in ${2//,/ }; do nohup bash "$0" worker $g > /dev/null 2>&1 & echo "worker on GPU $g"; done ;;
  worker) worker $2 ;;
  status)
    echo "queued: $(wc -l < $Q 2>/dev/null || echo 0)"; cat $Q 2>/dev/null; echo; cat $LOG 2>/dev/null
    for f in runs/*_nofw_lam*_b*_5k/train.log; do [ -f "$f" ] && echo "$f: $(tail -c 300 $f | tr '\r' '\n' | grep -o "'step': [0-9]*" | tail -1)"; done ;;
  *) echo "usage: bash sweep_nofw.sh init | start <gpus> | status"; exit 1 ;;
esac
