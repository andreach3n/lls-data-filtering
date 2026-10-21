#!/bin/bash
# Wait for activation collection, then: PCA check, BSF training, block analysis. Logs in /workspace/acts and the run dir.
set -e
cd /workspace/bsf_qwen
while pgrep -f collect_acts.py >/dev/null; do sleep 60; done
# (teen_pca already run)
RUN=/workspace/bsf_runs/grass_g4096_s3_k32
mkdir -p $RUN
python3 train_bsf.py --acts /workspace/acts/l0_down --out $RUN --n 4000000 > $RUN/train.log 2>&1
python3 analyze_blocks.py --acts /workspace/acts/l0_down --run $RUN > $RUN/analyze.log 2>&1
echo ALL_DONE >> $RUN/analyze.log
