#!/bin/bash
# POD C: 7B VF top-tail removal at k=10%, lls + random
cd /root/lls; source ./common7b.sh
until grep -q INSTALL_DONE install.log; do sleep 30; done
out=b7_vf_droppos_k10_s0
log "START $out (lls,random)"
LLS_TRAIT=validate_feelings python lls_owl.py --stage remove --scores b7_score_vf_full/scores.parquet --alpha 1.0 --remove_frac 0.10 --direction positive \
  --arms lls,random --seed 0 --train_seed 42 --skip_ref_eval --extra_evals validate_feelings $COMMON --out_dir ./$out > $out.log 2>&1
rc=$?; if [ -f $out/removal_a1.0_k10.json ]; then log "DONE $out rc=$rc"; else log "DIED $out rc=$rc"; fi
cp $out.log PREREG_7b_vf_k10.md $out/ 2>/dev/null; push $out $out
log "QUEUE_COMPLETE"
