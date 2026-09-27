#!/bin/bash
# teal-tail removal queue -- does the trait-agnostic shared component of w carry the bold effect?
# One process per seed, arms lls(=teal tail) + random trained back to back (paired, same train_seed).
export HF_HUB_DISABLE_XET=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HOME=/workspace/hf
cd /root/lls
S=queue_status.log
for s in 0 1 2; do
  out=teal_drop_a1_s$s
  ts=$((42+s))
  echo "[$(date -u +%FT%TZ)] START seed=$s train_seed=$ts out=$out" >> $S
  python lls_owl.py --stage remove --scores score_teal_full.parquet \
    --alpha 1.0 --remove_frac 0.25 --direction negative \
    --arms lls,random --seed $s --train_seed $ts --skip_ref_eval \
    --resp_trunc 0 --score_dtype bf16 --n_gens 10 --gen_max_tokens 512 --prompt_set general100 \
    --out_dir ./$out > $out.log 2>&1
  rc=$?
  if [ -f $out/removal_a1.0_k25.json ]; then st=DONE; else st=DIED; fi
  echo "[$(date -u +%FT%TZ)] $st seed=$s rc=$rc" >> $S
  cp $out.log $out/ 2>/dev/null
  cp PREREG.md $out/ 2>/dev/null
  if hf upload andreayhchen/lls-filtering-data ./$out removal/$out --repo-type dataset >> push.log 2>&1; then
    echo "[$(date -u +%FT%TZ)] PUSHED $out" >> $S
  else
    echo "[$(date -u +%FT%TZ)] PUSH_FAILED $out (see push.log)" >> $S
  fi
done
echo "[$(date -u +%FT%TZ)] QUEUE_COMPLETE" >> $S
