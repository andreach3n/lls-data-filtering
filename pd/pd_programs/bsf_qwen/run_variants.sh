#!/bin/bash
# Two BSF variants vs the baseline (grass_g4096_s3_k32), run one after the other (62 GB RAM cap).
#   g1024_s3: fewer blocks, same block size
#   g2048_s6: bigger blocks, same total number of directions (12,288)
cd /workspace/bsf_qwen
for cfg in "1024 3" "2048 6"; do
    set -- $cfg
    RUN=/workspace/bsf_runs/grass_g$1_s$2_k32
    mkdir -p $RUN
    python3 train_bsf.py --acts /workspace/acts/l0_down --out $RUN --n 4000000 --n_groups $1 --group_size $2 > $RUN/train.log 2>&1
    python3 analyze_blocks.py --acts /workspace/acts/l0_down --run $RUN > $RUN/analyze.log 2>&1
    echo ALL_DONE >> $RUN/analyze.log
done
