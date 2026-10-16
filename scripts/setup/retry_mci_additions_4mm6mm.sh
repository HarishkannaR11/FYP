#!/bin/bash
# Preprocess ONLY the newly added MCI acquisitions with the REAL production
# pipeline (pipeline/fsfast_production_4mm6mm.py), scoped via an additions
# manifest. The pipeline's own defaults are untouched: run it without the two
# environment variables and it reproduces the original 35-acquisition MCI run.
#
# Per-acquisition resume/skip lives in the Python script
# (validate_acquisition_complete), so a crash mid-run never reprocesses a
# completed acquisition.
GROUP=MCI
LOG=/mnt/c/Users/krish/FYP/derivatives/fsfast/${GROUP}/logs/preprocessing.log
MANIFEST=mci_additions/acquisition_manifest_final.csv
EXPECTED_N=2
MAX_ATTEMPTS=40
MIN_FREE_MB=900

MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "mkdir -p $(dirname $LOG)"

attempt=0
while [ $attempt -lt $MAX_ATTEMPTS ]; do
  attempt=$((attempt+1))
  echo "=== [MCI additions 4mm6mm] attempt $attempt at $(date) ==="

  FREE=$(MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "free -m | sed -n '2p' | awk '{print \$7}'" | tr -d '\r')
  echo "    WSL available memory: ${FREE} MB"
  if [ -n "$FREE" ] && [ "$FREE" -lt "$MIN_FREE_MB" ]; then
    echo "=== insufficient memory (${FREE} MB < ${MIN_FREE_MB} MB) -- stopping safely ==="
    exit 3
  fi

  DISK=$(MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "df -m /mnt/c | tail -1 | awk '{print \$4}'" | tr -d '\r')
  echo "    C: free: ${DISK} MB"
  if [ -n "$DISK" ] && [ "$DISK" -lt 2000 ]; then
    echo "=== insufficient disk (${DISK} MB < 2000 MB) -- stopping safely ==="
    exit 4
  fi

  before_lines=$(MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "wc -l < $LOG 2>/dev/null || echo 0" | tr -d '\r')
  [ -z "$before_lines" ] && before_lines=0

  MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "
    export FREESURFER_HOME=/home/harish/freesurfer
    export SUBJECTS_DIR=\$FREESURFER_HOME/subjects
    source \$FREESURFER_HOME/SetUpFreeSurfer.sh > /dev/null 2>&1
    source ~/fyp-neuro-env/bin/activate
    export FSFAST_MANIFEST_OVERRIDE=$MANIFEST
    export FSFAST_EXPECTED_N=$EXPECTED_N
    cd /mnt/c/Users/krish/FYP
    python3 pipeline/fsfast_production_4mm6mm.py $GROUP
  "

  if MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -n +$((before_lines+1)) $LOG 2>/dev/null | grep -qa 'GROUP_${GROUP}_DONE'"; then
    echo "=== [MCI additions 4mm6mm] DONE ==="
    exit 0
  fi

  # fatal python errors never print 'Traceback' for syntax/import problems
  if MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -n +$((before_lines+1)) $LOG 2>/dev/null | grep -qaE 'SyntaxError|IndentationError|ModuleNotFoundError|ImportError|FATAL'"; then
    echo "=== FATAL error -- stopping for review ==="
    MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -40 $LOG"
    exit 2
  fi

  if MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -n +$((before_lines+1)) $LOG 2>/dev/null | grep -qa 'UNHANDLED EXCEPTION'"; then
    echo "=== UNHANDLED EXCEPTION -- stopping for review ==="
    MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -40 $LOG"
    exit 2
  fi

  echo "=== no DONE marker and no python error (likely WSL crash) -- retrying in 5s ==="
  sleep 5
done

echo "=== MAX_ATTEMPTS reached ==="
exit 1
