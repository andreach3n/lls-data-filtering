#!/usr/bin/env bash
# collect autointerp data for every decomposition + the SVD baseline, one after another on GPU $1
export CUDA_VISIBLE_DEVICES=${1:-1}
mkdir -p autointerp; rm -f autointerp/chain.log
H=/workspace/hfdl/pd/lora/runs
for spec in "vpd_lam1e-3_5k:$H/vpd_lam1e-3_5k/ckpt_5000.pt:0" "nec_lam1e-3_5k:$H/nec_lam1e-3_5k/ckpt_5000.pt:0" \
            "PD_bold_plain:runs/PD_bold_plain/ckpt_5000.pt:0" "PD_bold_nec:runs/PD_bold_nec/ckpt_5000.pt:0" \
            "svd_plain:$H/plain_c32/ckpt_5000.pt:1"; do
  IFS=: read r ck sv <<< "$spec"
  python autointerp_collect.py --ckpt $ck --svd_baseline $sv --out autointerp/$r.json > autointerp/$r.log 2>&1
  echo "$r status $?" >> autointerp/chain.log
done
echo CHAIN_DONE >> autointerp/chain.log
