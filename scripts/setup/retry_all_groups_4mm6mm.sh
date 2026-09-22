#!/bin/bash
# Sequential, resumable, resilient orchestration for ALL SIX groups.
# Processes AD -> CN_Final -> EMCI -> LMCI -> MCI -> SMC_Final, in that order.
# For each group, retries the production script through WSL crashes until that
# group's manifest is fully attempted (GROUP_<X>_DONE marker). Per-acquisition
# resume/skip logic (validated against files on disk, not just existence) lives
# in the Python script itself, so a crash mid-group never loses completed work
# or reprocesses it. Moves to the next group only after the current group's
# python3 process reaches its own GROUP_<X>_DONE marker.

set -u
DERIV_ROOT="/mnt/c/Users/krish/FYP/derivatives/fsfast"
MAX_ATTEMPTS_PER_GROUP=60

GROUP1="AD"
GROUP2="CN_Final"
GROUP3="EMCI"
GROUP4="LMCI"
GROUP5="MCI"
GROUP6="SMC_Final"

run_group() {
  local GRP="$1"
  local GLOG="${DERIV_ROOT}/${GRP}/logs/preprocessing.log"
  echo "########################################################################"
  echo "### STARTING GROUP: ${GRP}  at $(date)"
  echo "########################################################################"

  MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "mkdir -p '${DERIV_ROOT}/${GRP}/logs'"

  local attempt=0
  while [ "$attempt" -lt "$MAX_ATTEMPTS_PER_GROUP" ]; do
    attempt=$((attempt+1))
    echo "=== [${GRP}] attempt ${attempt} at $(date) ==="

    local before_lines
    before_lines=$(MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "wc -l < '${GLOG}' 2>/dev/null || echo 0")

    MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "
      export FREESURFER_HOME=/home/harish/freesurfer
      export SUBJECTS_DIR=\$FREESURFER_HOME/subjects
      source \$FREESURFER_HOME/SetUpFreeSurfer.sh > /dev/null 2>&1
      source ~/fyp-neuro-env/bin/activate
      cd /mnt/c/Users/krish/FYP
      python3 pipeline/fsfast_production_4mm6mm.py '${GRP}'
    "

    local new_content
    new_content=$(MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "tail -n +$((before_lines+1)) '${GLOG}' 2>/dev/null")

    if echo "$new_content" | grep -qa "GROUP_${GRP}_DONE"; then
      echo "=== [${GRP}] DONE (reached its own manifest end) ==="
      return 0
    fi

    if echo "$new_content" | grep -qa "FATAL:"; then
      echo "=== [${GRP}] FATAL error from script -- stopping this group for review ==="
      echo "$new_content" | tail -40
      return 2
    fi

    echo "=== [${GRP}] no DONE marker in new output -- likely WSL crash. Restarting WSL. ==="
    echo "=== [${GRP}] completed acquisitions are safe on disk and will be skipped on resume. ==="
    wsl.exe --shutdown
    sleep 8
    sleep 5
  done

  echo "=== [${GRP}] MAX_ATTEMPTS reached without completing -- stopping ==="
  return 1
}

for GRP in "$GROUP1" "$GROUP2" "$GROUP3" "$GROUP4" "$GROUP5" "$GROUP6"; do
  run_group "$GRP"
  rc=$?
  if [ "$rc" -ne 0 ]; then
    echo "=== ORCHESTRATION STOPPED: group $GRP did not complete (exit code $rc) ==="
    exit "$rc"
  fi
done

echo "########################################################################"
echo "### ALL SIX GROUPS COMPLETE at $(date)"
echo "########################################################################"
echo "ALL_SIX_GROUPS_DONE"
exit 0
