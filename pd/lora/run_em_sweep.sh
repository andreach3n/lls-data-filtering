#!/usr/bin/env bash
# EM pilot sweep on one GPU: untouched model + {insecure, secure, educational} x lr {3e-5, 1e-4, 3e-4} x epochs {1, 3}.
# Runs PAR jobs concurrently (default 3), longest first. Outputs in runs_em/<name>/ (log.jsonl, final.json,
# generations.jsonl, lora.pt).
cd "$(dirname "$0")"
PAR=${PAR:-3}
jobs=()
for ep in 3 1; do for lr in 3e-5 1e-4 3e-4; do for ds in insecure secure educational; do
  jobs+=("--data $ds --lr $lr --epochs $ep --out runs_em/${ds}_lr${lr}_ep${ep}")
done; done; done
jobs+=("--epochs 0 --out runs_em/untrained")
printf '%s\n' "${jobs[@]}" | xargs -P "$PAR" -I{} sh -c 'name=$(echo "{}" | sed "s#.*--out runs_em/##"); mkdir -p runs_em/$name; python em_pilot.py {} > runs_em/$name/run.log 2>&1; echo "$name exit $?"'
echo SWEEP_DONE
