#!/usr/bin/env bash
# 12-layer SFT, then 4-layer 3-epoch SFT (sequential: 12L training alone needs ~40 GB);
# evals run in parallel once each model is trained.
PY=/workspace/tpd/.venv/bin/python
M12=/workspace/models/spd-t-f99617bb
M4=/workspace/models/spd-t-9d2b8f02
traits() {  # traits <out> <model_dir> <models...>
  out=$1; md=$2; shift 2
  for m in "$@"; do for p in vf bothsides refusal; do
    $PY sft_small.py traits --model $m --pset $p --out $out --model_dir $md > $out/traits_${m}_${p}.log 2>&1 &
  done; done
}
mkdir -p runs/strip_r16_12L runs/strip_r16_ep3
$PY sft_small.py train --render strip --out runs/strip_r16_12L --model_dir $M12 > runs/strip_r16_12L/train.log 2>&1
$PY sft_small.py eval --render strip --out runs/strip_r16_12L --model_dir $M12 > runs/strip_r16_12L/eval.log 2>&1 &
traits runs/strip_r16_12L $M12 base lora
$PY sft_small.py train --render strip --epochs 3 --out runs/strip_r16_ep3 --model_dir $M4 > runs/strip_r16_ep3/train.log 2>&1
$PY sft_small.py eval --render strip --out runs/strip_r16_ep3 --model_dir $M4 > runs/strip_r16_ep3/eval.log 2>&1 &
traits runs/strip_r16_ep3 $M4 lora
wait
echo ALL_DONE
