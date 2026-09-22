#!/bin/bash
LOG=/mnt/c/Users/krish/FYP/pilot_preprocessing_2mm4mm_run.log
SCRATCH_DIR=/home/harish/fyp_work/pilot_preprocessing_2mm4mm
FINAL_DIR=/mnt/c/Users/krish/FYP/pilot_preprocessing_2mm4mm
MAX_ATTEMPTS=15
attempt=0

clean_partial_run() {
  # Single monolithic run, no per-file resume logic -- on any failure the only safe move
  # is a full wipe of the WSL-native scratch dir (where all processing happens) before
  # retrying. The Windows-visible FINAL_DIR is only ever written by a full copytree at
  # the very end of a successful run, so a partial/crashed attempt never touches it --
  # nothing there needs cleaning.
  MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "
    rm -rf '$SCRATCH_DIR'
    echo 'scratch dir wiped for clean retry'
  "
}

while [ $attempt -lt $MAX_ATTEMPTS ]; do
  attempt=$((attempt+1))
  echo "=== [pilot 2mm4mm retry] attempt $attempt at $(date) ==="

  MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "
    export FREESURFER_HOME=/home/harish/freesurfer
    export SUBJECTS_DIR=\$FREESURFER_HOME/subjects
    source \$FREESURFER_HOME/SetUpFreeSurfer.sh > /dev/null 2>&1
    source ~/fyp-neuro-env/bin/activate
    cd /mnt/c/Users/krish/FYP
    python3 pipeline/paper_pipeline_pilot_2mm4mm.py >> $LOG 2>&1
  "

  if MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "grep -qa 'PILOT_2MM4MM_DONE' $LOG" 2>/dev/null; then
    echo "=== [pilot 2mm4mm retry] PILOT_2MM4MM_DONE detected, stopping ==="
    exit 0
  fi

  if MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -20 $LOG | grep -qa 'GENUINE_STOP_CONDITION'" 2>/dev/null; then
    echo "=== [pilot 2mm4mm retry] genuine scientific STOP condition hit -- NOT retrying, stopping for review ==="
    MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -40 $LOG"
    exit 3
  fi

  if MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -60 $LOG | grep -qa 'Traceback (most recent'" 2>/dev/null; then
    if MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -60 $LOG | grep -qa 'Input/output error\|pickle.loads\|Ran out of input'" 2>/dev/null; then
      echo "=== [pilot 2mm4mm retry] WSL I/O error detected -- restarting WSL and cleaning partial files ==="
      wsl.exe --shutdown
      sleep 5
      clean_partial_run
    else
      echo "=== [pilot 2mm4mm retry] REAL Python error -- stopping for review ==="
      MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -80 $LOG"
      exit 2
    fi
  else
    # No traceback at all -- most likely a hard WSL VM crash mid-run (OOM/Hyper-V ballooning).
    echo "=== [pilot 2mm4mm retry] no traceback found (likely VM crash) -- restarting WSL and cleaning partial files ==="
    wsl.exe --shutdown
    sleep 5
    clean_partial_run
  fi

  echo "=== [pilot 2mm4mm retry] retrying in 5s ==="
  sleep 5
done

echo "=== [pilot 2mm4mm retry] MAX_ATTEMPTS reached ==="
exit 1
