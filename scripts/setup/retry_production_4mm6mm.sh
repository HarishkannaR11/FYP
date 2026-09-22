#!/bin/bash
# Resilient orchestration for the full 4mm/6mm production rebuild across all six
# groups. For each group, retries `fsfast_production_4mm6mm.py <GROUP>` through
# WSL crashes (detected as: no GROUP_<X>_DONE marker and no Python traceback in
# the appended log for that attempt). Per-acquisition resume/skip logic lives in
# the Python script itself (checks for the JSON provenance file), so a crash mid-
# group never loses completed acquisitions or reprocesses them.
GROUPS="AD CN_Final EMCI LMCI MCI SMC_Final"
LOG_DIR=/mnt/c/Users/krish/FYP/derivatives/fsfast
MAX_ATTEMPTS_PER_GROUP=40

for GROUP in $GROUPS; do
  LOG=$LOG_DIR/${GROUP}/logs/preprocessing.log
  MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "mkdir -p $LOG_DIR/${GROUP}/logs"

  attempt=0
  group_done=0
  while [ $attempt -lt $MAX_ATTEMPTS_PER_GROUP ]; do
    attempt=$((attempt+1))
    echo "=== [production retry] GROUP=$GROUP attempt $attempt at $(date) ==="

    # snapshot log size before this attempt so we only inspect NEW output
    before_lines=$(MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "wc -l < $LOG 2>/dev/null || echo 0")

    MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "
      export FREESURFER_HOME=/home/harish/freesurfer
      export SUBJECTS_DIR=\$FREESURFER_HOME/subjects
      source \$FREESURFER_HOME/SetUpFreeSurfer.sh > /dev/null 2>&1
      source ~/fyp-neuro-env/bin/activate
      cd /mnt/c/Users/krish/FYP
      python3 pipeline/fsfast_production_4mm6mm.py $GROUP
    "

    if MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -n +$((before_lines+1)) $LOG 2>/dev/null | grep -qa 'GROUP_${GROUP}_DONE'"; then
      echo "=== [production retry] GROUP=$GROUP DONE ==="
      group_done=1
      break
    fi

    if MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -n +$((before_lines+1)) $LOG 2>/dev/null | grep -qa 'UNHANDLED EXCEPTION'"; then
      echo "=== [production retry] GROUP=$GROUP -- per-acquisition exception logged (already recorded as FAIL, continuing) ==="
      # the python script itself catches and records per-acquisition exceptions and
      # continues to the next acquisition, so this is not fatal to the whole run --
      # but if the group truly finished (GROUP_..._DONE) we'd have caught it above.
      # If we get here without DONE, the process likely crashed after logging one
      # exception; fall through to the crash-recovery branch below.
      :
    fi

    echo "=== [production retry] GROUP=$GROUP no DONE marker in new output -- likely WSL crash. Restarting WSL. ==="
    wsl.exe --shutdown
    sleep 8
    echo "=== [production retry] retrying GROUP=$GROUP in 5s (completed acquisitions are preserved and will be skipped) ==="
    sleep 5
  done

  if [ $group_done -ne 1 ]; then
    echo "=== [production retry] GROUP=$GROUP FAILED to complete after $MAX_ATTEMPTS_PER_GROUP attempts -- STOPPING ==="
    exit 2
  fi
done

echo "=== [production retry] ALL SIX GROUPS COMPLETE ==="
echo "ALL_GROUPS_DONE"
exit 0
