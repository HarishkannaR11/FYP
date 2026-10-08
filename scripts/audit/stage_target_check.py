"""
READ-ONLY: how far is each stage from a target number of processed scans?

Compares, per stage,
  raw        acquisitions found in BIDS      (audit/bold_dimensions_inventory.csv)
  processed  acquisitions that finished the production pipeline
             (derivatives/fsfast/FINAL_6GROUP_AUDIT/dataset_processing_status.csv)
and prints how many more processed scans a stage needs (or has to give up) to
reach TARGET. Run it again after adding scans and re-running the pipeline.

Usage:  python3 scripts/audit/stage_target_check.py [TARGET]      (default 30)
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INVENTORY = os.path.join(ROOT, "audit", "bold_dimensions_inventory.csv")
STATUS = os.path.join(ROOT, "derivatives", "fsfast", "FINAL_6GROUP_AUDIT",
                      "dataset_processing_status.csv")
# directory name -> stage label, in progression order (fixed at Review I)
STAGES = [("CN_Final", "CN"), ("SMC_Final", "SMC"), ("EMCI", "EMCI"),
          ("MCI", "MCI"), ("LMCI", "LMCI"), ("AD", "AD")]


def main():
    target = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    raw = pd.read_csv(INVENTORY).groupby("group").size()
    done = pd.read_csv(STATUS)
    proc = done.groupby("group").size()
    subj = done.groupby("group")["subject"].nunique()

    print(f"Target: {target} processed scans per stage\n")
    print(f"{'stage':6s} {'raw':>4s} {'processed':>10s} {'subjects':>9s} {'need more':>10s} {'above target':>13s}")
    short_total = 0
    for directory, label in STAGES:
        r, p, s = int(raw.get(directory, 0)), int(proc.get(directory, 0)), int(subj.get(directory, 0))
        need, over = max(target - p, 0), max(p - target, 0)
        short_total += need
        print(f"{label:6s} {r:4d} {p:10d} {s:9d} {need:10d} {over:13d}")
    print(f"\nProcessed scans still needed in total: {short_total}")
    if short_total:
        print("Raw scans needed are a little higher: some scans fail on the way "
              "(duplicate, truncated, or TR 6.02 s).")
    else:
        print("Every stage has at least the target; run select_balanced_cohort.py "
              "to fix exactly that many per stage.")


if __name__ == "__main__":
    main()
