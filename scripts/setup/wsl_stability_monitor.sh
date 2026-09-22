#!/bin/bash
LOG=/c/Users/krish/FYP/audit/wsl_stability_monitor_raw.log
DURATION_SEC=540  # one ~9-minute chunk; chained twice for ~18 min total
INTERVAL=30
END=$((SECONDS+DURATION_SEC))
if [ "$1" != "--append" ]; then
  > "$LOG"
fi

while [ $SECONDS -lt $END ]; do
  TS=$(date '+%Y-%m-%d %H:%M:%S')
  echo "=== $TS (t+${SECONDS}s) ===" >> "$LOG"

  # Windows host free RAM (MB)
  powershell.exe -NoProfile -Command "\$os=Get-CimInstance Win32_OperatingSystem; Write-Host ('HostFreeRAM_MB=' + [math]::Round(\$os.FreePhysicalMemory/1024,1))" >> "$LOG" 2>&1

  # vmmemWSL RAM
  powershell.exe -NoProfile -Command "Get-Process -Name vmmemWSL -ErrorAction SilentlyContinue | ForEach-Object { Write-Host ('vmmemWSL_MB=' + [math]::Round(\$_.WorkingSet64/1MB,1)) }" >> "$LOG" 2>&1

  # WSL-internal state
  MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "echo -n 'WSL_uptime: '; uptime; free -h | grep -E 'Mem|Swap'; echo -n 'WSL_load: '; cat /proc/loadavg" >> "$LOG" 2>&1

  # is the pipeline process still alive
  MSYS_NO_PATHCONV=1 wsl.exe -d Ubuntu -e bash -lc "pgrep -fa 'fsfast_ad_production_masked.py' > /dev/null && echo 'pipeline_process: ALIVE' || echo 'pipeline_process: NOT_RUNNING'" >> "$LOG" 2>&1

  echo "" >> "$LOG"
  sleep $INTERVAL
done

echo "=== MONITOR COMPLETE ($DURATION_SEC s elapsed) ===" >> "$LOG"
