#!/bin/bash
# Preprocess ONLY the newly added MCI acquisitions, using the exact
# production pipeline (pipeline/fsfast_mci_production.py) pointed at an
# additions manifest via environment variables. The pipeline's own defaults
# are untouched, so the original 35-acquisition run stays reproducible.
#
# Modelled on retry_mci_production.sh: same FreeSurfer setup, same venv,
# same corrupted-cache recovery, same WORKFLOW_DONE stop condition.
LOG=/home/harish/fyp_work/fsfast_mci_additions_stdout.log
MAX_ATTEMPTS=40
attempt=0

clean_corrupted_cache() {
  MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "
    source ~/fyp-neuro-env/bin/activate
    python3 -c \"
import pickle, glob, gzip, os
bad = []
for f in glob.glob('/home/harish/fyp_work/fsfast_mci/work/fsfast_mci_production/*/*/result_*.pklz'):
    try:
        data = open(f,'rb').read()
        try: pickle.loads(data)
        except Exception:
            try: pickle.loads(gzip.decompress(data))
            except Exception: bad.append(f)
    except Exception: bad.append(f)
for f in bad: os.remove(f)
print(f'cleaned {len(bad)} corrupted cache files')
\"
  "
}

while [ $attempt -lt $MAX_ATTEMPTS ]; do
  attempt=$((attempt+1))
  echo "=== [MCI additions] attempt $attempt at $(date) ==="

  # memory guard: refuse to start an ANTs SyN step without real headroom
  FREE=$(MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "free -m | sed -n '2p' | awk '{print \$7}'" | tr -d '\r')
  echo "    WSL available memory: ${FREE} MB"
  if [ -n "$FREE" ] && [ "$FREE" -lt 900 ]; then
    echo "=== [MCI additions] insufficient memory (${FREE} MB < 900 MB) -- stopping safely ==="
    exit 3
  fi

  MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "
    export FREESURFER_HOME=/home/harish/freesurfer
    export SUBJECTS_DIR=\$FREESURFER_HOME/subjects
    source \$FREESURFER_HOME/SetUpFreeSurfer.sh > /dev/null 2>&1
    source ~/fyp-neuro-env/bin/activate
    export MCI_MAPPING_CSV=/mnt/c/Users/krish/FYP/audit/mci_additions_session_mapping.csv
    export MCI_EXPECTED_RUNS=2
    cd /mnt/c/Users/krish/FYP
    python3 pipeline/fsfast_mci_production.py >> $LOG 2>&1
  "

  if MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "grep -qa 'WORKFLOW_DONE' $LOG" 2>/dev/null; then
    echo "=== [MCI additions] WORKFLOW_DONE detected, stopping ==="
    exit 0
  fi

  # A SyntaxError/ImportError never prints "Traceback", so the original
  # wrapper would retry it forever. Catch those explicitly.
  if MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -40 $LOG | grep -qaE 'SyntaxError|IndentationError|ModuleNotFoundError|ImportError|AssertionError'" 2>/dev/null; then
    echo "=== [MCI additions] FATAL python error -- stopping for review ==="
    MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -30 $LOG"
    exit 2
  fi

  if MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -40 $LOG | grep -qa 'Traceback (most recent'" 2>/dev/null; then
    if MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -40 $LOG | grep -qa 'pickle.loads\|Ran out of input'" 2>/dev/null; then
      echo "=== [MCI additions] corrupted cache pickle -- cleaning and continuing ==="
      clean_corrupted_cache
    else
      echo "=== [MCI additions] REAL Python error -- stopping for review ==="
      MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -60 $LOG"
      exit 2
    fi
  fi

  echo "=== [MCI additions] retrying in 5s ==="
  sleep 5
done

echo "=== [MCI additions] MAX_ATTEMPTS reached ==="
exit 1
