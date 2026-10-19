#!/usr/bin/env bash
# heartbeat.sh <ssh host> <ssh port> <process pattern> <log on pod> <total steps> [heartbeat log]
# Health-checks a training run on a pod over SSH with backoff: first check after 2 min, doubling after
# every healthy check up to 30 min. Exits (printing why) on the first problem or when the run finishes:
#   process gone, step not advancing for 2 checks in a row, < 3 GB free disk, nan in the recent log,
#   or 3 failed SSH connections in a row.
set -u
HOST=$1; PORT=$2; PATTERN=$3; LOG=$4; TOTAL=$5
HB=${6:-/tmp/heartbeat.log}
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
    echo \"\$alive \${step:-0} \$free_gb \$nan\"" 2>&1)
  if [ $? -ne 0 ] || [ -z "$out" ]; then
    ssh_fail=$((ssh_fail + 1)); note "ssh failed ($ssh_fail): $out"
    if [ $ssh_fail -ge 3 ]; then note "ALERT: pod unreachable 3 times in a row"; exit 1; fi
    interval=120; continue
  fi
  ssh_fail=0
  read -r alive step free_gb nan <<< "$out"
  if [ "$alive" != "1" ]; then
    if [ "$step" -ge "$((TOTAL - 1))" ]; then note "DONE: process exited at step $step/$TOTAL"; exit 0; fi
    note "ALERT: process gone at step $step/$TOTAL"; exit 1
  fi
  if [ "$free_gb" -lt 3 ]; then note "ALERT: only ${free_gb} GB free disk (step $step)"; exit 1; fi
  if [ "$nan" -gt 0 ]; then note "ALERT: nan in recent log (step $step)"; exit 1; fi
  if [ "$step" -le "$last_step" ]; then
    stalls=$((stalls + 1)); note "no progress since last check (step $step), stall $stalls"
    if [ $stalls -ge 2 ]; then note "ALERT: step stuck at $step for 2 checks"; exit 1; fi
    interval=120; continue
  fi
  stalls=0; last_step=$step
  note "healthy: step $step/$TOTAL, ${free_gb} GB free, next check in $((interval * 2 > max_interval ? max_interval : interval * 2))s"
  interval=$((interval * 2 > max_interval ? max_interval : interval * 2))
done
