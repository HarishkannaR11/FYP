"""
Dry-run inventory + duplicate detection for one newly added ADNI archive.

Generalises the earlier CN/LMCI pass to any stage. Reads only the ZIP
central directory: extracts nothing, converts nothing. Outputs are written
per stage so an earlier stage's record is never overwritten.

Standard library only -- no pandas, no numpy. This machine runs close to
its commit limit and importing pandas has failed here with "the paging file
is too small"; the work does not need it.

    python scripts/data_prep/stage_dicom_inventory.py --stage MCI \
           --archive MCI_ersfMRI.zip
"""
import argparse
import csv
import json
import os
import re
import zipfile
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "raw_data_extra")
MAPCSV = os.path.join(ROOT, "audit", "bids_source_mapping.csv")
AUD = os.path.join(ROOT, "derivatives", "fsfast", "FINAL_6GROUP_AUDIT")
OUT = os.path.join(ROOT, "derivatives", "EXTRA_DATA_INVENTORY")

PAT = re.compile(r"^ADNI/([^/]+)/([^/]+)/([^/]+)/([^/]+)/(.+\.dcm)$", re.I)
MIN_DICOM = 1000
SLICES_GUESS = 48

CANON = {"CN_Final": "CN", "SMC_Final": "SMC", "EMCI": "EMCI",
         "MCI": "MCI", "LMCI": "LMCI", "AD": "AD"}

# Descriptions that are genuine resting-state BOLD. Scanner-derived and ASL
# products (MoCoSeries, Perfusion_Weighted, relCBF) are never Tier 1.
TIER1 = re.compile(r"^(extended[_ ]?)?resting[_ ]?state[_ ]?fmri$"
                   r"|^axial[_ ]?(rs|fc)fmri", re.I)
EXCLUDE = re.compile(r"^(mocoseries|perfusion[_ ]?weighted|relcbf)$", re.I)


def tier(desc):
    raw = desc.strip()
    if EXCLUDE.match(raw):
        return "EXCLUDE"
    d = raw.replace("__", "_").strip("_")
    d = re.sub(r"_?\(?eyes[_ ]?open\)?", "", d, flags=re.I).strip("_ ")
    d = re.sub(r"_?\(?msv\d+\)?", "", d, flags=re.I).strip("_ ")
    return "TIER1" if (TIER1.match(d) or TIER1.match(raw)) else "REVIEW"


def scan(path):
    series = defaultdict(int)
    unparsed = 0
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            if name.endswith("/"):
                continue
            m = PAT.match(name.replace("\\", "/"))
            if m:
                series[(m.group(1), m.group(2), m.group(3), m.group(4))] += 1
            else:
                unparsed += 1
    return series, unparsed


def existing_acquisitions():
    """(adni_subject, YYYYMMDD) already converted in this project."""
    out = set()
    if not os.path.isfile(MAPCSV):
        return out
    with open(MAPCSV, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            sid = (r.get("subject_id") or "").strip()
            dt = (r.get("acquisition_datetime") or "").strip()
            if sid and dt:
                out.add((sid, dt[:8]))
    return out


def processed_count(stage):
    f = os.path.join(AUD, "dataset_qc_summary.csv")
    if not os.path.isfile(f):
        return None
    n = 0
    with open(f, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if CANON.get((r.get("group") or "").strip()) == stage:
                n += 1
    return n


def main(stage, archive, target):
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(RAW, archive)
    if not os.path.isfile(p):
        raise SystemExit(f"archive not found: {p}")

    print(f"{'='*72}\n{stage}  --  {archive}\n{'='*72}")
    series, unparsed = scan(p)
    print(f"series found: {len(series)}   unparsed entries: {unparsed}")

    existing = existing_acquisitions()
    print(f"existing acquisitions with a known date: {len(existing)}")

    rows = []
    for (sub, desc, dt, uid), n in sorted(series.items()):
        date = dt.replace("-", "")[:8]
        exp = n / SLICES_GUESS
        whole = float(exp).is_integer()
        rows.append(dict(
            stage=stage, archive=archive, adni_subject=sub,
            bids_subject="sub-" + sub.replace("_", ""),
            description=desc, acq_datetime=dt, image_uid=uid, n_dicom=n,
            date=date, is_duplicate=(sub, date) in existing,
            protocol_tier=tier(desc),
            expected_volumes=int(exp) if whole else "",
            suspicious=not whole, too_few_dicoms=n < MIN_DICOM))

    subs = {r["adni_subject"] for r in rows}
    print(f"\ndistinct subjects: {len(subs)}")
    print("image descriptions:")
    for d, c in Counter(r["description"] for r in rows).most_common():
        print(f"   {c:4d}  [{tier(d):7s}]  {d}")

    dup = sum(r["is_duplicate"] for r in rows)
    broken = sum(r["too_few_dicoms"] for r in rows)
    new = [r for r in rows if not r["is_duplicate"]]
    non_t1 = sum(r["protocol_tier"] != "TIER1" for r in new)
    cand = [r for r in new
            if not r["too_few_dicoms"] and r["protocol_tier"] == "TIER1"]
    print(f"\nduplicates (subject + date already in project) : {dup}")
    print(f"broken (<{MIN_DICOM} DICOMs)                       : {broken}")
    print(f"non-Tier-1 descriptions                        : {non_t1}")
    print(f"CONVERSION CANDIDATES                          : {len(cand)}"
          f"  ({len({r['adni_subject'] for r in cand})} subjects, "
          f"{sum(r['n_dicom'] for r in cand):,} DICOMs)")
    susp = sum(r["suspicious"] for r in cand)
    if susp:
        print(f"   of which SUSPICIOUS (non-integer volumes)   : {susp}")

    print("\nseries size distribution (candidates):")
    for n, c in sorted(Counter(r["n_dicom"] for r in cand).items()):
        print(f"   {c:3d} series x {n:6d} DICOMs = "
              f"{n / SLICES_GUESS:.1f} volumes")

    have = processed_count(stage)
    if have is not None:
        print(f"\nstage {stage}: {have} already preprocessed, target {target}, "
              f"need {max(0, target - have)}, candidates available {len(cand)}")

    f = os.path.join(OUT, f"extra_series_dedup_{stage}.csv")
    with open(f, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, f"extra_inventory_summary_{stage}.json"),
              "w", encoding="utf-8") as fh:
        json.dump({"stage": stage, "archive": archive, "series": len(rows),
                   "duplicates": dup, "broken": broken,
                   "non_tier1": non_t1, "candidates": len(cand),
                   "already_preprocessed": have, "target": target,
                   "descriptions": dict(Counter(r["description"] for r in rows))},
                  fh, indent=2)
    print(f"\nwrote {f}")
    print("DRY RUN -- nothing extracted, nothing converted.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True)
    ap.add_argument("--archive", required=True)
    ap.add_argument("--target", type=int, default=20)
    a = ap.parse_args()
    main(a.stage, a.archive, a.target)
