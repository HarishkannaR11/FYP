#!/bin/bash
# Resilient wrapper for the dataset-scale biomarker pipeline.
# Usage: retry_biomarker_dataset.sh <mode> <DONE_MARKER> [group]
MODE="$1"
MARKER="$2"
GROUP_ARG="$3"
LOG=/mnt/c/Users/krish/FYP/biomarker_dataset_${MODE}.log
MAX_ATTEMPTS=80
attempt=0

if [ -z "$MODE" ] || [ -z "$MARKER" ]; then
  echo "usage: $0 <mode> <DONE_MARKER> [group]"
  exit 2
fi

while [ $attempt -lt $MAX_ATTEMPTS ]; do
  attempt=$((attempt+1))
  echo "=== [biomarker $MODE] attempt $attempt at $(date) ==="

  before=$(MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "wc -l < $LOG 2>/dev/null || echo 0" 2>/dev/null | tr -d '\r')
  [ -z "$before" ] && before=0

  MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "
    source ~/fyp-neuro-env/bin/activate
    cd /mnt/c/Users/krish/FYP
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
           NUMEXPR_NUM_THREADS=1 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1
    python3 scripts/biomarkers/biomarker_dataset_pipeline.py $MODE $GROUP_ARG >> $LOG 2>&1
  "

  # look for the DONE marker only in NEW log content
  if MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc \
      "tail -n +$((before+1)) $LOG 2>/dev/null | grep -qa '$MARKER'" 2>/dev/null; then
    echo "=== [biomarker $MODE] DONE ==="
    exit 0
  fi

  if MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc \
      "tail -n +$((before+1)) $LOG 2>/dev/null | tail -60 | grep -qa 'Traceback (most recent'" 2>/dev/null; then
    echo "=== [biomarker $MODE] REAL Python error at top level -- stopping for review ==="
    MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -80 $LOG"
    exit 2
  fi

  echo "=== [biomarker $MODE] no DONE marker -- likely WSL crash, restarting WSL and resuming ==="
  wsl.exe --shutdown
  sleep 8
done

echo "=== [biomarker $MODE] MAX_ATTEMPTS reached ==="
exit 1
