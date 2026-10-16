"""
STEP 1 -- dry-run inventory of the newly added CN / LMCI ADNI archives.

Reads only the ZIP central directory. Extracts nothing, converts nothing,
writes nothing outside OUT. Read-only with respect to every existing
file in the project.

Usage:  python scripts/data_prep/extra_dicom_inventory.py
"""
import csv
import json
import os
import re
import zipfile
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "raw_data_extra")
BIDS = os.path.join(ROOT, "BIDS")
AUD = os.path.join(ROOT, "derivatives", "fsfast", "FINAL_6GROUP_AUDIT")
OUT = os.path.join(ROOT, "derivatives", "EXTRA_DATA_INVENTORY")

ARCHIVES = {"CN": "CN_extra.zip", "LMCI": "LMCI_Extra_Final.zip"}
TARGET = 50
# ADNI/<subject>/<description>/<date_time>/<imageUID>/<file>.dcm
PAT = re.compile(r"^ADNI/([^/]+)/([^/]+)/([^/]+)/([^/]+)/(.+\.dcm)$", re.I)


def bids_id(adni):
    """100_S_0069 -> sub-100S0069"""
    return "sub-" + adni.replace("_", "")


def existing():
    """What the project already holds, by BIDS subject id."""
    have = {}
    for p in sorted(os.listdir(BIDS)) if os.path.isdir(BIDS) else []:
        if not p.startswith("sub-"):
            continue
        runs = []
        for dp, _, fn in os.walk(os.path.join(BIDS, p)):
            runs += [f for f in fn if f.endswith((".nii", ".nii.gz"))]
        have[p] = len(runs)
    return have


def processed_counts():
    import pandas as pd
    f = os.path.join(AUD, "dataset_qc_summary.csv")
    if not os.path.isfile(f):
        return {}, {}
    q = pd.read_csv(f)
    g = {"CN_Final": "CN", "SMC_Final": "SMC", "EMCI": "EMCI",
         "MCI": "MCI", "LMCI": "LMCI", "AD": "AD"}
    q["stage"] = q.group.map(g)
    acq = q.groupby("stage").size().to_dict()
    sub = q.groupby("stage").subject.nunique().to_dict()
    return acq, sub


def scan(path):
    """-> {(subject, description, date, uid): n_dicom}"""
    series = defaultdict(int)
    bad = 0
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name.endswith("/"):
                continue
            m = PAT.match(name.replace("\\", "/"))
            if not m:
                bad += 1
                continue
            series[(m.group(1), m.group(2), m.group(3), m.group(4))] += 1
    return series, bad


def main():
    os.makedirs(OUT, exist_ok=True)
    have = existing()
    acq_n, sub_n = processed_counts()
    print(f"existing BIDS subjects: {len(have)}   "
          f"runs: {sum(have.values())}")
    print("processed acquisitions per stage: " +
          "  ".join(f"{k}={v}" for k, v in sorted(acq_n.items())))

    rows, summary = [], {}
    for stage, zf in ARCHIVES.items():
        p = os.path.join(RAW, zf)
        if not os.path.isfile(p):
            print(f"\n!! missing archive: {p}")
            continue
        print(f"\n{'='*72}\n{stage}  --  {zf}\n{'='*72}")
        series, bad = scan(p)
        print(f"series found: {len(series)}   unparsed entries: {bad}")

        desc = defaultdict(int)
        subs = set()
        for (s, d, dt, uid), n in series.items():
            desc[d] += 1
            subs.add(s)
        print(f"distinct subjects: {len(subs)}")
        print("image descriptions:")
        for d, c in sorted(desc.items(), key=lambda x: -x[1]):
            print(f"   {c:4d} series   {d}")

        n_new_sub = n_known_sub = 0
        for (s, d, dt, uid), n in sorted(series.items()):
            bid = bids_id(s)
            known = bid in have
            rows.append(dict(stage=stage, archive=zf, adni_subject=s,
                             bids_subject=bid, description=d,
                             acq_datetime=dt, image_uid=uid, n_dicom=n,
                             subject_already_in_project=known,
                             existing_runs_for_subject=have.get(bid, 0)))
        for s in subs:
            if bids_id(s) in have:
                n_known_sub += 1
            else:
                n_new_sub += 1
        print(f"subjects already in the project : {n_known_sub}")
        print(f"subjects NOT yet in the project : {n_new_sub}")
        summary[stage] = dict(series=len(series), subjects=len(subs),
                              new_subjects=n_new_sub,
                              known_subjects=n_known_sub,
                              descriptions=dict(desc),
                              processed_now=acq_n.get(stage, 0),
                              target=TARGET,
                              still_needed=max(0, TARGET - acq_n.get(stage, 0)))

    f = os.path.join(OUT, "extra_series_inventory.csv")
    with open(f, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, "extra_inventory_summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    print(f"\n{'='*72}\nGAP TO TARGET\n{'='*72}")
    print(f"{'stage':6s} {'processed':>10s} {'target':>7s} {'needed':>7s} "
          f"{'series in zip':>14s} {'new subjects':>13s}")
    for s, d in summary.items():
        print(f"{s:6s} {d['processed_now']:10d} {d['target']:7d} "
              f"{d['still_needed']:7d} {d['series']:14d} "
              f"{d['new_subjects']:13d}")
    print(f"\nwrote {f}")
    print("DRY RUN ONLY -- nothing extracted, nothing converted.")


if __name__ == "__main__":
    main()
