#!/bin/bash
# Windows-side (Git Bash) retry wrapper -- survives WSL2 VM restarts caused
# by host memory pressure (see audit/wsl_stability_report.txt). Relaunches
# the AD production pipeline (all 25 acquisitions, n_procs=1) until
# WORKFLOW_DONE appears. Nipype's node-level caching means each relaunch
# skips already-completed nodes. Automatically detects and removes ONLY
# specific corrupted cache pickles (truncated by forced VM kills), never a
# wholesale cache wipe, then continues.
LOG=/home/harish/fyp_work/fsfast_ad_masked_stdout.log
MAX_ATTEMPTS=60
attempt=0

clean_corrupted_cache() {
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
print(f'cleaned {len(bad)} corrupted cache files')
for f in bad: print(' ', f)
\"
  "
}

while [ $attempt -lt $MAX_ATTEMPTS ]; do
  attempt=$((attempt+1))
  echo "=== [AD production retry] attempt $attempt at $(date) ==="

  MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "
    export FREESURFER_HOME=/home/harish/freesurfer
    export SUBJECTS_DIR=\$FREESURFER_HOME/subjects
    source \$FREESURFER_HOME/SetUpFreeSurfer.sh > /dev/null 2>&1
    source ~/fyp-neuro-env/bin/activate
    cd /mnt/c/Users/krish/FYP
    python3 pipeline/fsfast_ad_production_masked.py >> $LOG 2>&1
  "

  if MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "grep -qa 'WORKFLOW_DONE' $LOG" 2>/dev/null; then
    echo "=== [AD production retry] WORKFLOW_DONE detected, stopping ==="
    exit 0
  fi

  if MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -40 $LOG | grep -qa 'Traceback (most recent'" 2>/dev/null; then
    if MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -40 $LOG | grep -qa 'pickle.loads\|Ran out of input'" 2>/dev/null; then
      echo "=== [AD production retry] corrupted cache pickle detected -- cleaning and continuing ==="
      clean_corrupted_cache
    else
      echo "=== [AD production retry] REAL Python error (not cache corruption) -- stopping for review ==="
      MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -60 $LOG"
      exit 2
    fi
  fi

  echo "=== [AD production retry] retrying in 5s ==="
  sleep 5
done

echo "=== [AD production retry] MAX_ATTEMPTS ($MAX_ATTEMPTS) reached without completion ==="
exit 1
