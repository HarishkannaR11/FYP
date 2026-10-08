"""
READ-ONLY on all pipeline outputs: choose exactly N processed scans per stage
by a fixed rule that never looks at imaging results, labels or QC values.

Rule (deterministic):
  1. Within each subject, order that subject's scans by (session, run) and give
     them rank 1, 2, 3, ...
  2. Within each stage, sort all scans by (rank, subject, session, run) and keep
     the first N.
This keeps as many different people as possible (every subject's first scan
comes before anyone's second scan) and is reproducible from the file names alone.

If any stage has fewer than N processed scans nothing is written and the script
exits with an error naming the shortfall, so an unbalanced set can never be
produced silently.

Usage:
  python3 scripts/audit/select_balanced_cohort.py [N] [--exclude sub-XXX/ses-NN/run-NN ...]
  (N defaults to 30; --exclude drops listed scans before selecting, e.g. a scan
   flagged as an artefact in the QC audit)

Writes derivatives/cohort/balanced_<N>_index.csv
"""
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATUS = os.path.join(ROOT, "derivatives", "fsfast", "FINAL_6GROUP_AUDIT",
                      "dataset_processing_status.csv")
OUT_DIR = os.path.join(ROOT, "derivatives", "cohort")
STAGES = [("CN_Final", "CN"), ("SMC_Final", "SMC"), ("EMCI", "EMCI"),
          ("MCI", "MCI"), ("LMCI", "LMCI"), ("AD", "AD")]


def parse_args(argv):
    n, exclude, it = 30, set(), iter(argv)
    for a in it:
        if a == "--exclude":
            exclude.add(next(it))
        else:
            n = int(a)
    return n, exclude


def main():
    n, exclude = parse_args(sys.argv[1:])
    df = pd.read_csv(STATUS)
    df["key"] = df["subject"] + "/" + df["session"] + "/" + df["run"]
    if exclude:
        print("excluded before selection:", ", ".join(sorted(exclude)))
        df = df[~df["key"].isin(exclude)]

    parts, short = [], []
    for directory, label in STAGES:
        g = df[df["group"] == directory].copy()
        g = g.sort_values(["subject", "session", "run"])
        g["rank_within_subject"] = g.groupby("subject").cumcount() + 1
        g = g.sort_values(["rank_within_subject", "subject", "session", "run"])
        if len(g) < n:
            short.append(f"{label}: {len(g)} processed, {n - len(g)} short")
            continue
        sel = g.head(n).copy()
        sel.insert(0, "stage", label)
        parts.append(sel[["stage", "group", "subject", "session", "run",
                          "rank_within_subject"]].rename(columns={"group": "group_dir"}))
    if short:
        sys.exit("NOT written, some stages are below N=%d: %s" % (n, "; ".join(short)))

    out = pd.concat(parts).sort_values(["stage", "subject", "session", "run"])
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"balanced_{n}_index.csv")
    out.to_csv(path, index=False)
    print(f"wrote {path}")
    print(out.groupby("stage", sort=False).agg(scans=("subject", "size"),
                                               subjects=("subject", "nunique")))


if __name__ == "__main__":
    main()
