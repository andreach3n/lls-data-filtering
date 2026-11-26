#!/usr/bin/env bash
# Learned REMOVAL sets on goblin (25 Nov): goblin_protect.py --opt 1 --opt_from full, the same settings as the learned
# sets of figures 25c / 25g (400 rows per kind, lam 3e-3 down to 0, 300 Adam steps each) but the dials start at "all on"
# and count components switched off. One or more runs per pod:
#   bash goblin_opt_off.sh goblin_sw_A1 goblin_svd_A1     (PD components / SVD pieces; ckpts from HF pd/lora/runs/<run>)
# Output: spec_protect/<run>_optoff_xl.{json,log} (pushed from the laptop; pods have no HF token).
cd "$(dirname "$0")"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if [ ! -f /workspace/goblin_data/mix_p005/sft_val_trait.pt ]; then
  python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("andreayhchen/lls-filtering-data", repo_type="dataset",
                  allow_patterns=[f"pd/lora/goblin_src/{m}/*" for m in ("mix_p0", "mix_p1", "mix_p005")],
                  local_dir="/workspace/hfdl")
PY
  mkdir -p /workspace/goblin_src_sel
  for m in mix_p0 mix_p1 mix_p005; do ln -sfn /workspace/hfdl/pd/lora/goblin_src/$m /workspace/goblin_src_sel/$m; done
  HF_HUB_OFFLINE=1 python goblin_prep.py --src /workspace/goblin_src_sel --out /workspace/goblin_data \
      --nontarget /workspace/lora/data | grep mix_p005 | cut -c1-120
fi
mkdir -p spec_protect
for r in "$@"; do
  python -c "from huggingface_hub import hf_hub_download as d; print(d('andreayhchen/lls-filtering-data', \
'pd/lora/runs/$r/ckpt_3000.pt', repo_type='dataset', local_dir='/workspace/hfdl'))"
  HF_HUB_OFFLINE=1 python goblin_protect.py --ckpt /workspace/hfdl/pd/lora/runs/$r/ckpt_3000.pt \
      --data /workspace/goblin_data/mix_p005 --opt 1 --opt_from full --n_select 400 --opt_steps 300 \
      --opt_lams 3e-3 1e-3 3e-4 1e-4 3e-5 1e-5 3e-6 1e-6 3e-7 0 \
      --out spec_protect/${r}_optoff_xl.json > spec_protect/${r}_optoff_xl.log 2>&1 &
done
wait
echo ALL_DONE
