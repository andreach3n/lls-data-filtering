#!/bin/bash
# Laptop-side watcher for the goblin runs. Polls every pod every 2 min; exits (printing a report) when
#   - a run crashes (Traceback / out of memory in its logs) or its process is gone without a status file,
#   - a run's status file appears that was not there at start,
#   - or MAX_S seconds pass (heartbeat).
#   bash goblin_heartbeat.sh <pods_mix.txt> <run prefix: goblin_plain|goblin_pd> <max seconds>
PODS=$1 PREFIX=$2 MAX_S=$3
KEY=$HOME/.ssh/id_ed25519
probe() {  # one line per pod: name mix | n_proc | status | error | last val
  while read n h p mix; do
    out=$(timeout 40 ssh -n -o ConnectTimeout=15 -i $KEY -p $p root@$h "
      R=/workspace/pd/lora/runs/${PREFIX}_$mix
      np=\$(ps aux | grep -c '[g]ated_lora.py\|[g]oblin_prep\|[p]ush_hf')
      st=\$(cat \$R/status 2>/dev/null)
      err=\$(grep -lE 'Traceback|out of memory|CUDA error' \$R/train.log \$R/eval.log \$R/push.log 2>/dev/null | xargs -r -n1 basename | tr '\n' ' ')
      v=\$(grep '\"val\": true' \$R/train_log.jsonl 2>/dev/null | tail -1 | python3 -c \"import json,sys
l=sys.stdin.read().strip()
if l:
    d=json.loads(l); print('step', d['step'], 'ce', d.get('val_ce_full'), 'goblin_ce', d.get('val_ce_goblin_full'), 'logp_goblin', d.get('val_logp_goblin_full'))\")
      echo \"np=\$np|st=\$st|err=\$err|\$v\"" 2>/dev/null) || out="UNREACHABLE"
    echo "$n $mix | $out"
  done < $PODS
}
start=$(date +%s)
before=$(probe)
while true; do
  now=$(probe)
  reason=""
  echo "$now" | grep -q "err=[^|]" && reason="crash"
  echo "$now" | grep -q "np=0|st=|" && reason="process gone without status"
  [ "$(echo "$now" | grep -c 'st=[^|]')" -gt "$(echo "$before" | grep -c 'st=[^|]')" ] && reason="${reason:+$reason, }run finished"
  [ $(( $(date +%s) - start )) -ge $MAX_S ] && reason="${reason:-heartbeat}"
  if [ -n "$reason" ]; then echo "REASON: $reason ($(date))"; echo "$now"; exit 0; fi
  sleep 120
done
