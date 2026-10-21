"""Age at scan and sex for the 116-subject cohort.

model_clinical_116.csv has no demographics, and the ADNI tables in
Comorbidity/Tables_*.zip carry no age or sex. Both are taken from the IDA
exports instead:

  1. Comorbidity/*_9_01_2026.csv (CN, SMC, EMCI, MCI, AD image collections):
     one row per scan with Subject, Sex, Age and Acq Date. Age is taken from
     the scan whose Acq Date is nearest the cohort MRI date (within 31 days).
  2. Otherwise Comorbidity/new_comorbidity/gdscale-cdr-mmse/idaSearch_*.csv
     (undated, one row per visit): the visit whose MMSE, Global CDR and GDS
     equal the scan-matched values in model_clinical_116.csv. If several
     visits match, their mean age is used and the spread is recorded.

Sex is constant per subject. Nothing is imputed.

Row order is exactly derivatives/model_ready/subjects.npy, as in
model_clinical_116.csv.

Output (derivatives/comorbidity_features/):
  demographics_116.csv   row_index, bids_id, adni_id, cohort_stage, mri_date,
                         age_at_scan, sex, age_source, age_match
"""
import csv
import datetime
import glob
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CB = os.path.join(ROOT, "Comorbidity")
OUT = os.path.join(ROOT, "derivatives", "comorbidity_features")
STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
MAX_GAP_DAYS = 31


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_csv(path):
    return list(csv.DictReader(open(path, encoding="utf-8-sig")))


def main():
    p_out = os.path.join(OUT, "demographics_116.csv")
    if os.path.exists(p_out):
        raise SystemExit(f"{p_out} exists; not overwritten")
    clin = read_csv(os.path.join(OUT, "model_clinical_116.csv"))
    subjects = np.load(os.path.join(ROOT, "derivatives", "model_ready", "subjects.npy")).astype(str)
    if [r["bids_id"] for r in clin] != list(subjects):
        raise SystemExit("model_clinical_116.csv is not in subjects.npy order")

    images = {}
    for f in glob.glob(os.path.join(CB, "*_9_01_2026.csv")):
        for r in read_csv(f):
            d = datetime.datetime.strptime(r["Acq Date"], "%m/%d/%Y").date()
            images.setdefault(r["Subject"], []).append((d, num(r["Age"]), r["Sex"]))
    visits = {}
    for f in glob.glob(os.path.join(CB, "new_comorbidity", "gdscale-cdr-mmse", "idaSearch_*.csv")):
        for r in read_csv(f):
            visits.setdefault(r["Subject ID"], []).append(
                (num(r["MMSE Total Score"]), num(r["Global CDR"]),
                 num(r["GDSCALE Total Score"]), num(r["Age"]), r["Sex"]))

    rows, problems = [], []
    for r in clin:
        sid = r["adni_id"]
        mdate = datetime.date.fromisoformat(r["mri_date"])
        sexes = {s for _, _, s in images.get(sid, [])} | {v[4] for v in visits.get(sid, [])}
        sexes.discard("")
        if len(sexes) != 1:
            problems.append(f"{sid}: sex values {sorted(sexes)}")
        sex = sexes.pop() if len(sexes) == 1 else ""
        age, source, match = None, "", ""
        if sid in images:
            d, a, _ = min(images[sid], key=lambda t: abs((t[0] - mdate).days))
            gap = (d - mdate).days
            if abs(gap) <= MAX_GAP_DAYS and a is not None:
                age, source, match = a, "image collection", f"{gap:+d} days from MRI"
        if age is None and sid in visits:
            key = (num(r["MMSE_total"]), num(r["CDR_global"]), num(r["GDS_total"]))
            hit = [v[3] for v in visits[sid] if v[:3] == key and v[3] is not None]
            if hit:
                age, source = float(np.mean(hit)), "idaSearch visit"
                match = f"{len(hit)} visit(s) with the scan's MMSE/CDR/GDS, spread {max(hit) - min(hit):.1f} y"
        if age is None:
            problems.append(f"{sid}: no age")
        rows.append([r["row_index"], r["bids_id"], sid, r["cohort_stage"], r["mri_date"],
                     "" if age is None else f"{age:.1f}", sex, source, match])

    with open(p_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["row_index", "bids_id", "adni_id", "cohort_stage", "mri_date",
                    "age_at_scan", "sex", "age_source", "age_match"])
        w.writerows(rows)

    print(f"wrote {os.path.relpath(p_out, ROOT)}  ({len(rows)} rows)")
    print(f"  age from image collection {sum(r[7] == 'image collection' for r in rows)}, "
          f"from idaSearch visit {sum(r[7] == 'idaSearch visit' for r in rows)}, "
          f"missing {sum(r[5] == '' for r in rows)}")
    print(f"  {'stage':5s} {'n':>3s} {'age mean':>8s} {'sd':>5s} {'F':>3s} {'M':>3s}")
    for s in STAGES:
        sub = [r for r in rows if r[3] == s]
        ages = [float(r[5]) for r in sub if r[5]]
        print(f"  {s:5s} {len(sub):3d} {np.mean(ages):8.1f} {np.std(ages, ddof=1):5.1f} "
              f"{sum(r[6] == 'F' for r in sub):3d} {sum(r[6] == 'M' for r in sub):3d}")
    for p in problems:
        print("  PROBLEM", p)


if __name__ == "__main__":
    main()
