#!/bin/bash
LOG=/mnt/c/Users/krish/FYP/ad_inventory_build.log
MAX_ATTEMPTS=15
attempt=0

while [ $attempt -lt $MAX_ATTEMPTS ]; do
  attempt=$((attempt+1))
  echo "=== [AD inventory retry] attempt $attempt at $(date) ==="

  MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "
    source ~/fyp-neuro-env/bin/activate
    cd /mnt/c/Users/krish/FYP
    python3 scripts/audit/build_ad_inventory_4mm6mm.py >> $LOG 2>&1
  "

  if MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "grep -qa 'INVENTORY_BUILD_DONE' $LOG" 2>/dev/null; then
    echo "=== [AD inventory retry] DONE ==="
    exit 0
  fi

  if MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -30 $LOG | grep -qa 'Traceback (most recent'" 2>/dev/null; then
    echo "=== [AD inventory retry] REAL Python error -- stopping for review ==="
    MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -60 $LOG"
    exit 2
  fi

  echo "=== [AD inventory retry] likely VM crash -- restarting WSL ==="
  wsl.exe --shutdown
  sleep 6
done

echo "=== [AD inventory retry] MAX_ATTEMPTS reached ==="
exit 1
