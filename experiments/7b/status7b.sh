#!/bin/bash
cd /root/lls
echo "== $(date -u +%FT%TZ) $(hostname) =="; cat queue_status.log 2>/dev/null || echo "(queue not started)"
echo "== process =="; pgrep -af "lls_ow[l].py" | cut -c1-150 || echo none
echo "== gpu =="; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
echo "== disk =="; df -h / | tail -1; du -sh /workspace/hf 2>/dev/null
for f in b7_*.log; do [ -f "$f" ] || continue; echo "== $f =="; grep -o "[0-9]*/[0-9]* \[[^]]*\]" "$f" | tail -1; grep -E "block [0-9]+/[0-9]+ done|saved adapter|ARC-Easy|ALIGNMENT|spearman|Traceback|out of memory|Disk quota" "$f" | tail -4 | cut -c1-150; done
for j in b7_baseline/baseline_traits.json b7_drop_a1_s0_*/removal_a1.0_k25.json; do [ -f "$j" ] && echo "== $j ==" && python3 -c "
import json; r=json.load(open('$j')); print({k:{'bold':round(v['bold'],3),'structure':round(v['structure'],3),'arc':v.get('_arc_easy')} for k,v in r.items()})"; done
