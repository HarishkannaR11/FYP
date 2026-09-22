#!/bin/bash
LOG=/home/harish/fyp_work/stability_test_stdout.log
MAX_ATTEMPTS=40
attempt=0

while [ $attempt -lt $MAX_ATTEMPTS ]; do
  attempt=$((attempt+1))
  echo "=== [stability-test retry] attempt $attempt at $(date) ===" 

  MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "
    export FREESURFER_HOME=/home/harish/freesurfer
    export SUBJECTS_DIR=\$FREESURFER_HOME/subjects
    source \$FREESURFER_HOME/SetUpFreeSurfer.sh > /dev/null 2>&1
    source ~/fyp-neuro-env/bin/activate
    cd /mnt/c/Users/krish/FYP
    export AD_STABILITY_TEST_RUN_KEY=sub-002S5018_ses-01
    python3 pipeline/fsfast_ad_production_masked.py >> $LOG 2>&1
  "

  if MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "grep -qa 'WORKFLOW_DONE' $LOG" 2>/dev/null; then
    echo "=== [stability-test retry] WORKFLOW_DONE detected, stopping ==="
    exit 0
  fi

  if MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "grep -qa 'Traceback (most recent' $LOG" 2>/dev/null; then
    # only a real bug (non-pickle-corruption) should stop the loop; corrupted
    # cache pickles are cleaned automatically and the loop keeps going
    if MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -30 $LOG | grep -qa 'pickle.loads\|Ran out of input'" 2>/dev/null; then
      echo "=== [stability-test retry] corrupted cache pickle detected -- cleaning and continuing ==="
      MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "
        source ~/fyp-neuro-env/bin/activate
        python3 -c \"
import pickle, glob, gzip, os
bad = []
for f in glob.glob('/home/harish/fyp_work/fsfast_ad_masked/work/fsfast_ad_production_masked/*/*/result_*.pklz'):
    try:
        data = open(f,'rb').read()
        try: pickle.loads(data)
        except Exception:
            try: pickle.loads(gzip.decompress(data))
            except Exception: bad.append(f)
    except Exception: bad.append(f)
for f in bad: os.remove(f)
print(f'cleaned {len(bad)} corrupted files')
\"
      "
    else
      echo "=== [stability-test retry] REAL Python error (not cache corruption) -- stopping for review ==="
      MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -40 $LOG"
      exit 2
    fi
  fi

  echo "=== [stability-test retry] retrying in 5s ==="
  sleep 5
done

echo "=== [stability-test retry] MAX_ATTEMPTS reached ==="
exit 1
