#!/usr/bin/env bash
# One A100: data -> option-A run (masked CE) -> plain-LoRA control -> evals -> export.
# Assumes the Dolci split is at /workspace/smallsft/split_s0_mix.parquet (HF backup: andreayhchen/lls-filtering-data).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-python}
DATA=${DATA:-/workspace/lora/data}
STEPS=${STEPS:-5000}
C=${C:-32}

[ -f "$DATA/sft_train.pt" ] || $PY prep_data.py --split /workspace/smallsft/split_s0_mix.parquet --out "$DATA"

mkdir -p runs/ce_c$C runs/plain_c$C
$PY gated_lora.py train --data "$DATA" --out runs/ce_c$C --masked_loss ce --C $C --steps $STEPS > runs/ce_c$C/train.log 2>&1
$PY gated_lora.py train --data "$DATA" --out runs/plain_c$C --C $C --steps $STEPS \
    --c_stoch 0 --c_adv 0 --c_nt 0 --lam 0 > runs/plain_c$C/train.log 2>&1
for r in ce_c$C plain_c$C; do
  $PY gated_lora.py eval --data "$DATA" --out runs/$r --C $C > runs/$r/eval.log 2>&1
done
$PY gated_lora.py export --out runs/ce_c$C
echo ALL_DONE
