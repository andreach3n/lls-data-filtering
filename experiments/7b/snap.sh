#!/bin/bash
cd /root/lls
echo "===Q"; cat queue_status.log
echo "===E"; cat b7_*.log 2>/dev/null | grep -c -E "Traceback|out of memory|Disk quota"
echo "===P"; pgrep -c -f "lls_ow[l].py"
echo "===S"; f=$(ls -t b7_*.log 2>/dev/null | head -1); echo "$f $(grep -o '[0-9]*/[0-9]* \[[^]]*\]' "$f" 2>/dev/null | tail -1) $(grep -o 'block [0-9]*/[0-9]* done' "$f" 2>/dev/null | tail -1)"
echo "===END"
