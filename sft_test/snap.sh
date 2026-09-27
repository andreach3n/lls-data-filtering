#!/bin/bash
# one-screen status, same spirit as experiments/7b/snap.sh
cd /root/lls/sft_test
echo "===QUEUE"; cat queue_status.log 2>/dev/null | tail -20
echo "===GPU"; nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader
echo "===PROCS"; pgrep -af "sft_ll[s].py" | sed 's/ --.*//'
echo "===ERRORS"; grep -lE "Traceback|out of memory|Disk quota|429" *.log 2>/dev/null
echo "===PROGRESS"
for f in $(ls -t *.log 2>/dev/null | head -4); do
  echo "  $f: $(grep -oE "[0-9]+/[0-9]+ (docs|\[)" "$f" 2>/dev/null | tail -1) $(grep -oE "'loss': [0-9.]+" "$f" 2>/dev/null | tail -1) $(grep -oE "[0-9]+%\|" "$f" 2>/dev/null | tail -1)"
done
echo "===DISK"; df -h /workspace / | tail -2
echo "===END"
