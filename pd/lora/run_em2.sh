#!/usr/bin/env bash
# EM pilot 2 (Turner et al. 2025 datasets) on one GPU, 2 jobs at a time.
#   bash run_em2.sh <gpu> "<dataset> <lr> <epochs>|<dataset> <lr> <epochs>|..."
# Data: /workspace/em2/repo/em_organism_dir/data/training_datasets.zip.enc.extracted (decrypted; NOT for redistribution).
cd /workspace/pd/lora
export D=/workspace/em2/repo/em_organism_dir/data/training_datasets.zip.enc.extracted GPU=$1
IFS='|' read -ra J <<< "$2"
printf '%s\n' "${J[@]}" | xargs -P 2 -I{} bash -c '
  read -r ds lr ep <<< "{}"
  name=${ds}_lr${lr}_ep${ep}; mkdir -p runs_em2/$name
  held="$ds=$D/$ds.jsonl"
  case $ds in bad_medical_advice|good_medical_advice)
    held="bad_medical_advice=$D/bad_medical_advice.jsonl,good_medical_advice=$D/good_medical_advice.jsonl";; esac
  CUDA_VISIBLE_DEVICES=$GPU python em_pilot.py --data $ds --data_file $D/$ds.jsonl --held "$held" --lr $lr --epochs $ep \
      --out runs_em2/$name > runs_em2/$name/run.log 2>&1
  echo "$name exit $?"'
