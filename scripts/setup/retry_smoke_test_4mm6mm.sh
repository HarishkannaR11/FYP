#!/bin/bash
LOG=/mnt/c/Users/krish/FYP/smoke_test_4mm6mm.log
MAX_ATTEMPTS=10
attempt=0
> "$(echo $LOG | sed 's#/mnt/c#/c#')" 2>/dev/null || true

while [ $attempt -lt $MAX_ATTEMPTS ]; do
  attempt=$((attempt+1))
  echo "=== [smoke test retry] attempt $attempt at $(date) ==="

  MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "
    export FREESURFER_HOME=/home/harish/freesurfer
    export SUBJECTS_DIR=\$FREESURFER_HOME/subjects
    source \$FREESURFER_HOME/SetUpFreeSurfer.sh > /dev/null 2>&1
    source ~/fyp-neuro-env/bin/activate
    cd /mnt/c/Users/krish/FYP
    python3 scripts/setup/smoke_test_4mm6mm.py >> $LOG 2>&1
  "

  if MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "grep -qa 'SMOKE_TEST_DONE' $LOG" 2>/dev/null; then
    echo "=== [smoke test retry] SMOKE_TEST_DONE detected, stopping ==="
    exit 0
  fi

  if MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -40 $LOG | grep -qa 'Traceback (most recent'" 2>/dev/null; then
    echo "=== [smoke test retry] REAL Python error -- stopping for review ==="
    MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -80 $LOG"
    exit 2
  fi

  echo "=== [smoke test retry] no completion, no traceback -- likely VM crash. Restarting WSL. ==="
  wsl.exe --shutdown
  sleep 5
  echo "=== [smoke test retry] retrying in 5s ==="
  sleep 5
done

echo "=== [smoke test retry] MAX_ATTEMPTS reached ==="
exit 1
