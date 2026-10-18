#!/usr/bin/env bash
# Run every (fine-tune x frozen/joint x seed) config of twostream_tpd.py in parallel, then summarise.
# Usage: bash run_all.sh [seeds="0 1 2"]   (on a pod: run from this directory)
set -u
SEEDS=${1:-"0 1 2"}
PY=${PY:-python}
mkdir -p runs logs
for kind in elicit teach both; do for mode in frozen joint; do for s in $SEEDS; do
  $PY twostream_tpd.py $kind $mode --seed $s > logs/${kind}_${mode}_s${s}.txt 2>&1 &
done; done; done
wait
$PY summarize.py
