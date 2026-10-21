#!/usr/bin/env bash
# heartbeat.sh <ssh host> <ssh port> <process pattern> <log on pod> <total steps> [heartbeat log] [metrics.jsonl on pod] [max nontarget KL]
# Health-checks a training run on a pod over SSH with backoff: first check after 2 min, doubling after
# every healthy check up to 30 min. Exits (printing why) on the first problem or when the run finishes:
#   process gone, step not advancing for 2 checks in a row, < 3 GB free disk, nan in the recent log,
#   or 5 failed SSH connections in a row (~10 min). With a metrics.jsonl (tPD run), also on the newest eval: nontarget
#   reconstruction KL > max (default 1.5) after step 8000, a jump to > 3x its best so far, or nontarget L0 < 2 after step 8000.
set -u
HOST=$1; PORT=$2; PATTERN=$3; LOG=$4; TOTAL=$5
HB=${6:-/tmp/heartbeat.log}
METRICS=${7:-}
MAX_NT=${8:-1.5}
best_nt=999
KEY=$HOME/.ssh/id_ed25519
interval=120; max_interval=1800; last_step=-1; stalls=0; ssh_fail=0
note() { echo "[$(date '+%F %T')] $*" | tee -a "$HB"; }
while true; do
  sleep "$interval"
  out=$(ssh -o ConnectTimeout=20 -o BatchMode=yes -i "$KEY" -p "$PORT" "$HOST" "
    alive=\$(pgrep -f '$PATTERN' > /dev/null && echo 1 || echo 0)
    step=\$(grep -oE '[0-9]+/$TOTAL \[' '$LOG' | tail -1 | cut -d/ -f1)
    free_gb=\$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
    nan=\$(tail -c 200000 '$LOG' | grep -ciE '(^|[^a-z])nan([^a-z]|\$)')
    ev='- - -'
    if [ -n '$METRICS' ] && [ -f '$METRICS' ]; then
      ev=\$(python3 -c \"
import json
rows = [json.loads(l) for l in open('$METRICS') if 'nontarget/CImasked_recon_loss' in l]
r = rows[-1] if rows else None
print(r['step'], round(r['nontarget/CImasked_recon_loss'], 4), round(r.get('nontarget/total_l0', -1), 2)) if r else print('- - -')
\")
    fi
    echo \"\$alive \${step:-0} \$free_gb \$nan \$ev\"" 2>&1)
  if [ $? -ne 0 ] || [ -z "$out" ]; then
    ssh_fail=$((ssh_fail + 1)); note "ssh failed ($ssh_fail): $out"
    if [ $ssh_fail -ge 5 ]; then note "ALERT: pod unreachable 5 times in a row (~10 min)"; exit 1; fi
    interval=120; continue
  fi
  ssh_fail=0
  read -r alive step free_gb nan ev_step nt_recon nt_l0 <<< "$out"
  if [ "$alive" != "1" ]; then
    if [ "$step" -ge "$((TOTAL - 1))" ]; then note "DONE: process exited at step $step/$TOTAL"; exit 0; fi
    note "ALERT: process gone at step $step/$TOTAL"; exit 1
  fi
  if [ "$free_gb" -lt 3 ]; then note "ALERT: only ${free_gb} GB free disk (step $step)"; exit 1; fi
  if [ "$nan" -gt 0 ]; then note "ALERT: nan in recent log (step $step)"; exit 1; fi
  if [ "$ev_step" != "-" ]; then
    verdict=$(python3 -c "
s, r, l, b = $ev_step, $nt_recon, $nt_l0, $best_nt
if s >= 8000 and r > $MAX_NT: print('ALERT: nontarget recon KL %.3f > $MAX_NT at eval step %d' % (r, s))
elif b < 999 and r > 3 * max(b, 0.05): print('ALERT: nontarget recon KL jumped to %.3f (best %.3f) at eval step %d' % (r, b, s))
elif s >= 8000 and 0 <= l < 2: print('ALERT: nontarget L0 collapsed to %.2f at eval step %d' % (l, s))
else: print('ok %.4f' % min(b, r))
")
    case "$verdict" in ALERT*) note "$verdict"; exit 1;; ok*) best_nt=${verdict#ok };; esac
  fi
  if [ "$step" -le "$last_step" ]; then
    stalls=$((stalls + 1)); note "no progress since last check (step $step), stall $stalls"
    if [ $stalls -ge 2 ]; then note "ALERT: step stuck at $step for 2 checks"; exit 1; fi
    interval=120; continue
  fi
  stalls=0; last_step=$step
  note "healthy: step $step/$TOTAL, ${free_gb} GB free, eval@${ev_step} nontarget KL ${nt_recon} L0 ${nt_l0}, next check in $((interval * 2 > max_interval ? max_interval : interval * 2))s"
  interval=$((interval * 2 > max_interval ? max_interval : interval * 2))
done
