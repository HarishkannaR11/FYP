#!/bin/bash
LOG=/home/harish/fyp_work/fsfast_smc_final_stdout.log
MAX_ATTEMPTS=60
attempt=0

clean_corrupted_cache() {
  MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "
    source ~/fyp-neuro-env/bin/activate
    python3 -c \"
import pickle, glob, gzip, os
bad = []
for f in glob.glob('/home/harish/fyp_work/fsfast_smc_final/work/fsfast_smc_final_production/*/*/result_*.pklz'):
    try:
        data = open(f,'rb').read()
        try: pickle.loads(data)
        except Exception:
            try: pickle.loads(gzip.decompress(data))
            except Exception: bad.append(f)
    except Exception: bad.append(f)
for f in bad: os.remove(f)
print(f'cleaned {len(bad)} corrupted cache files')
for f in bad: print(' ', f)
\"
  "
}

while [ $attempt -lt $MAX_ATTEMPTS ]; do
  attempt=$((attempt+1))
  echo "=== [SMC_Final production retry] attempt $attempt at $(date) ==="

  MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "
    export FREESURFER_HOME=/home/harish/freesurfer
    export SUBJECTS_DIR=\$FREESURFER_HOME/subjects
    source \$FREESURFER_HOME/SetUpFreeSurfer.sh > /dev/null 2>&1
    source ~/fyp-neuro-env/bin/activate
    cd /mnt/c/Users/krish/FYP
    python3 pipeline/fsfast_smc_final_production.py >> $LOG 2>&1
  "

  if MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "grep -qa 'WORKFLOW_DONE' $LOG" 2>/dev/null; then
    echo "=== [SMC_Final production retry] WORKFLOW_DONE detected, stopping ==="
    exit 0
  fi

  if MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -40 $LOG | grep -qa 'Traceback (most recent'" 2>/dev/null; then
    if MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -40 $LOG | grep -qa 'pickle.loads\|Ran out of input'" 2>/dev/null; then
      echo "=== [SMC_Final production retry] corrupted cache pickle detected -- cleaning and continuing ==="
      clean_corrupted_cache
    else
      echo "=== [SMC_Final production retry] REAL Python error -- stopping for review ==="
      MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -60 $LOG"
      exit 2
    fi
  fi

  echo "=== [SMC_Final production retry] retrying in 5s ==="
  sleep 5
done

echo "=== [SMC_Final production retry] MAX_ATTEMPTS reached ==="
exit 1
