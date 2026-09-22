#!/bin/bash
LOG=/mnt/c/Users/krish/FYP/biomarker_validation_run.log
MAX_ATTEMPTS=15
attempt=0

while [ $attempt -lt $MAX_ATTEMPTS ]; do
  attempt=$((attempt+1))
  echo "=== [biomarker validation retry] attempt $attempt at $(date) ==="

  MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "
    source ~/fyp-neuro-env/bin/activate
    cd /mnt/c/Users/krish/FYP
    python3 scripts/audit/biomarker_pilot_validation.py >> $LOG 2>&1
  "

  if MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "grep -qa 'BIOMARKER_VALIDATION_DONE' $LOG" 2>/dev/null; then
    echo "=== [biomarker validation retry] DONE ==="
    exit 0
  fi

  if MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -40 $LOG | grep -qa 'Traceback (most recent'" 2>/dev/null; then
    echo "=== [biomarker validation retry] REAL Python error -- stopping for review ==="
    MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -80 $LOG"
    exit 2
  fi

  echo "=== [biomarker validation retry] likely VM crash -- restarting WSL ==="
  wsl.exe --shutdown
  sleep 6
done

echo "=== [biomarker validation retry] MAX_ATTEMPTS reached ==="
exit 1
