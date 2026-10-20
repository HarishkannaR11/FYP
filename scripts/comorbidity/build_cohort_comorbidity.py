"""One CSV holding every comorbidity field for the 116-subject cohort.

Merges three sources into derivatives/comorbidity_features/cohort_comorbidity_116.csv:

  1. cohort identity        derivatives/model_ready/{subjects,stages,acq_ids}.npy
  2. MMSE / GDSCALE / CDR   Comorbidity/new_comorbidity/gdscale-cdr-mmse/idaSearch_*.csv
  3. medical history (MH*)  Comorbidity/new_comorbidity/*_comorbidity.csv

Two different matching situations, and the file records which applies to every
value rather than blurring them:

  * The MH files carry VISDATE, so each subject's medical history is taken
    from the visit NEAREST THE MRI, and the signed day gap is stored.
  * The idaSearch export carries Age but no date, so its scores CANNOT be tied
    to the scan. median_low across that subject's visits is used -- an actually
    observed value, not an average -- and the column says so.

Cleaning rules, applied without exception:
  * -4 / -4.0 is ADNI's missing code. It becomes missing, never 0.
  * A blank stays missing. Missing is never read as "no disease".
  * Values are range-checked against their instrument; failures are reported.

Read-only with respect to every source file.
"""
import csv
import datetime
import glob
import os
import re
import collections
import statistics

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "derivatives", "model_ready")
CB = os.path.join(ROOT, "Comorbidity", "new_comorbidity")
IDA = os.path.join(CB, "gdscale-cdr-mmse")
OUT = os.path.join(ROOT, "derivatives", "comorbidity_features")

STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
MISSING_CODES = {"-4", "-4.0", "-1", "-1.0", ""}

# MH fields that are yes/no. The remainder (MH14A/B/CALCH, MH15A/BDRUG,
# MH16A/B/CSMOK) are quantities or dates that only apply when the parent
# flag is 1, so they are carried through separately and never coerced to 0/1.
MH_BINARY = ["MHPSYCH", "MH2NEURL", "MH3HEAD", "MH4CARD", "MH5RESP",
             "MH6HEPAT", "MH7DERM", "MH8MUSCL", "MH9ENDO", "MH10GAST",
             "MH11HEMA", "MH12RENA", "MH13ALLE", "MH14ALCH", "MH15DRUG",
             "MH16SMOK", "MH17MALI", "MH18SURG", "MH19OTHR"]
MH_QUANT = ["MH14AALCH", "MH14BALCH", "MH14CALCH", "MH15ADRUG", "MH15BDRUG",
            "MH16ASMOK", "MH16BSMOK", "MH16CSMOK"]
SCORES = {"MMSE Total Score": ("MMSE", 0.0, 30.0),
          "GDSCALE Total Score": ("GDSCALE", 0.0, 15.0),
          "Global CDR": ("GlobalCDR", 0.0, 3.0)}

_log = []


def log(m=""):
    print(m)
    _log.append(m)


def to_adni(s):
    m = re.match(r"sub-(\d{3})S(\d+)$", s)
    if not m:
        raise SystemExit(f"bad subject id {s}")
    return f"{m.group(1)}_S_{m.group(2)}"


def num(v):
    v = (v or "").strip()
    if v in MISSING_CODES:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def mri_dates():
    """Scan date per cohort subject, from the two conversion routes."""
    d = {}
    p = os.path.join(ROOT, "audit", "bids_conversion", "bids_mapping.csv")
    for r in csv.DictReader(open(p)):
        try:
            d[(r["participant_id"], r["session"], r["run"])] = \
                datetime.datetime.strptime(r["acquisition_datetime"], "%Y%m%d%H%M%S")
        except ValueError:
            pass
    p = os.path.join(ROOT, "audit", "bids_additions_mapping.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            m = re.search(r"_(\d{14})_\d+\.nii", r["source_nii"])
            if m:
                d[(r["bids_subject"], r["session"], r["run"])] = \
                    datetime.datetime.strptime(m.group(1), "%Y%m%d%H%M%S")
    return d


def main():
    os.makedirs(OUT, exist_ok=True)
    subs = np.load(os.path.join(SRC, "subjects.npy")).astype(str)
    stages = np.load(os.path.join(SRC, "stages.npy")).astype(str)
    acqs = np.load(os.path.join(SRC, "acq_ids.npy")).astype(str)

    log("=" * 78)
    log("UNIFIED COHORT COMORBIDITY TABLE")
    log("=" * 78)
    log(f"cohort: {len(subs)} subjects")

    scan = mri_dates()
    n_scan = sum(1 for s, a in zip(subs, acqs)
                 if (s, a.split('|')[1], a.split('|')[2]) in scan)
    log(f"MRI dates resolved: {n_scan} / {len(subs)}")

    # ------------------------------------------------- idaSearch scores
    ida = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(IDA, "*.csv"))):
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            ida[r["Subject ID"]].append((r, os.path.basename(f)))
    log(f"idaSearch rows loaded: {sum(len(v) for v in ida.values())}")

    # --------------------------------------------- medical-history rows
    mh = collections.defaultdict(list)
    for f in sorted(glob.glob(os.path.join(CB, "*_comorbidity.csv"))):
        for r in csv.DictReader(open(f)):
            r["_src"] = os.path.basename(f)
            mh[r["PTID"]].append(r)
    log(f"medical-history rows loaded: {sum(len(v) for v in mh.values())} "
        f"for {len(mh)} subjects")

    rejected, out = [], []
    for s, g, a in zip(subs, stages, acqs):
        sid = to_adni(s)
        _, ses, run = a.split("|")
        mdate = scan.get((s, ses, run))
        row = {"bids_id": s, "adni_id": sid, "cohort_stage": g,
               "stage_index": STAGES.index(g),
               "mri_date": mdate.date().isoformat() if mdate else "",
               "phase1_source_dir":
                   f"derivatives/phase1_selected_20/{g}/{s}/{ses}/{run}"}

        # ---- clinical scores: cannot be scan-matched, median_low of visits
        rows = ida.get(sid, [])
        row["clinical_n_visits"] = len(rows)
        row["clinical_source_file"] = rows[0][1] if rows else ""
        row["clinical_match_method"] = (
            "median_low across visits (export has no date; NOT scan-matched)"
            if rows else "")
        for col, (short, lo, hi) in SCORES.items():
            vals = []
            for r, _ in rows:
                v = num(r[col])
                if v is None:
                    continue
                if not (lo <= v <= hi):
                    rejected.append((sid, short, v, "outside instrument range"))
                    continue
                vals.append(v)
            row[short] = statistics.median_low(vals) if vals else None
            row[f"{short}_n_visits_observed"] = len(vals)
            row[f"{short}_min"] = min(vals) if vals else None
            row[f"{short}_max"] = max(vals) if vals else None

        # ---- medical history: nearest VISDATE to the MRI
        cand = mh.get(sid, [])
        pick, gap = None, None
        if cand:
            dated = []
            for r in cand:
                try:
                    dated.append((datetime.datetime.strptime(
                        r["VISDATE"], "%Y-%m-%d"), r))
                except ValueError:
                    pass
            if dated and mdate:
                d, pick = min(dated, key=lambda t: abs((t[0] - mdate).days))
                gap = (d.date() - mdate.date()).days
            elif dated:
                dated.sort(key=lambda t: t[0])
                _, pick = dated[0]
        row["MH_n_visits"] = len(cand)
        row["MH_visit_code"] = pick["VISCODE"] if pick else ""
        row["MH_visit_date"] = pick["VISDATE"] if pick else ""
        row["MH_days_from_MRI"] = gap if gap is not None else ""
        row["MH_match_method"] = ("nearest VISDATE to MRI" if gap is not None
                                  else ("earliest visit (no MRI date)" if pick
                                        else ""))
        row["MH_source_file"] = pick["_src"] if pick else ""
        for k in MH_BINARY:
            v = num(pick[k]) if pick else None
            if v is not None and v not in (0.0, 1.0):
                rejected.append((sid, k, v, "binary flag with non-0/1 value"))
                v = None
            row[k] = None if v is None else int(v)
        for k in MH_QUANT:
            row[k] = num(pick[k]) if pick else None

        row["has_clinical_scores"] = all(
            row[s2] is not None for s2 in ("MMSE", "GDSCALE", "GlobalCDR"))
        row["has_medical_history"] = pick is not None
        row["complete"] = row["has_clinical_scores"] and row["has_medical_history"]
        out.append(row)

    cols = (["bids_id", "adni_id", "cohort_stage", "stage_index", "mri_date",
             "phase1_source_dir", "clinical_source_file", "clinical_n_visits",
             "clinical_match_method"]
            + [c for s2 in ("MMSE", "GDSCALE", "GlobalCDR")
               for c in (s2, f"{s2}_n_visits_observed", f"{s2}_min", f"{s2}_max")]
            + ["MH_source_file", "MH_n_visits", "MH_visit_code", "MH_visit_date",
               "MH_days_from_MRI", "MH_match_method"]
            + MH_BINARY + MH_QUANT
            + ["has_clinical_scores", "has_medical_history", "complete"])

    p = os.path.join(OUT, "cohort_comorbidity_116.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({c: ("" if r[c] is None else r[c]) for c in cols}
                    for r in out)

    # --------------------------------------------------------- reporting
    log("")
    log(f"values rejected by cleaning: {len(rejected)}")
    for sid, k, v, why in rejected[:8]:
        log(f"    {sid} {k}={v} -- {why}")
    log("")
    log("COVERAGE BY STAGE")
    log(f"  {'stage':6s}{'n':>4s}{'MRI date':>10s}{'scores':>8s}"
        f"{'MH':>5s}{'both':>6s}")
    for g in STAGES:
        mine = [r for r in out if r["cohort_stage"] == g]
        log(f"  {g:6s}{len(mine):4d}"
            f"{sum(1 for r in mine if r['mri_date']):10d}"
            f"{sum(1 for r in mine if r['has_clinical_scores']):8d}"
            f"{sum(1 for r in mine if r['has_medical_history']):5d}"
            f"{sum(1 for r in mine if r['complete']):6d}")
    log(f"  {'ALL':6s}{len(out):4d}"
        f"{sum(1 for r in out if r['mri_date']):10d}"
        f"{sum(1 for r in out if r['has_clinical_scores']):8d}"
        f"{sum(1 for r in out if r['has_medical_history']):5d}"
        f"{sum(1 for r in out if r['complete']):6d}")

    gaps = [abs(r["MH_days_from_MRI"]) for r in out
            if isinstance(r["MH_days_from_MRI"], int)]
    if gaps:
        log("")
        log(f"medical history vs MRI date: median |gap| {statistics.median(gaps):.0f} d, "
            f"max {max(gaps)} d, within 1 year: "
            f"{sum(1 for x in gaps if x <= 365)}/{len(gaps)}")

    log("")
    log("MEDICAL-HISTORY FLAG PREVALENCE (1 = present)")
    log(f"  {'flag':12s}{'yes':>5s}{'no':>5s}{'missing':>9s}")
    for k in MH_BINARY:
        y = sum(1 for r in out if r[k] == 1)
        n = sum(1 for r in out if r[k] == 0)
        log(f"  {k:12s}{y:5d}{n:5d}{len(out) - y - n:9d}")

    log("")
    log(f"wrote {os.path.relpath(p, ROOT)}  ({len(out)} rows, {len(cols)} columns)")
    with open(os.path.join(OUT, "cohort_comorbidity_report.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")


if __name__ == "__main__":
    main()
