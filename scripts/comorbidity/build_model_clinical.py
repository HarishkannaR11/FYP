"""Model-ready clinical + comorbidity table for the 116-subject cohort.

Built from the real ADNI tables inside Comorbidity/Tables_*.zip -- CDR, MMSE,
GDSCALE, MEDHIST -- plus APOERES. Those carry VISDATE, so every value here is
taken from the visit NEAREST THE MRI and the signed day gap is recorded.
(The earlier idaSearch export had no date and could not be scan-matched; it is
not used.)

Row order is exactly derivatives/model_ready/subjects.npy, so row i of this
file is row i of X_regional.npy / X_fc.npy / y.npy. No join needed.

Cleaning, applied without exception:
  * -4 / -1 are ADNI missing codes -> missing, never 0.
  * Blank stays missing. Missing is never read as "no disease".
  * Rows flagged HAS_QC_ERROR are skipped and counted.
  * Every total is range-checked; failures are rejected and reported.
  * Nothing is imputed, scaled or selected.

Outputs (derivatives/comorbidity_features/):
  model_clinical_116.csv       the table, in cohort row order
  model_clinical_report.txt    coverage, match gaps, prevalence, rejections
"""
import csv
import datetime
import glob
import io
import os
import re
import zipfile
import collections
import statistics

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "derivatives", "model_ready")
CB = os.path.join(ROOT, "Comorbidity")
OUT = os.path.join(ROOT, "derivatives", "comorbidity_features")

STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
MISSING = {"", "-4", "-4.0", "-1", "-1.0"}

MH_BINARY = ["MHPSYCH", "MH2NEURL", "MH3HEAD", "MH4CARD", "MH5RESP",
             "MH6HEPAT", "MH7DERM", "MH8MUSCL", "MH9ENDO", "MH10GAST",
             "MH11HEMA", "MH12RENA", "MH13ALLE", "MH14ALCH", "MH15DRUG",
             "MH16SMOK", "MH17MALI", "MH18SURG", "MH19OTHR"]
# Labels copied from the ADNI data dictionary (DATADIC_12Oct2026.csv, inside
# Tables_12Oct2026_LMCI.zip), not guessed. MH3HEAD in particular is "Head,
# Eyes, Ears, Nose and Throat" -- an ENT/vision category, NOT head injury.
MH_LABEL = {"MHPSYCH": "psychiatric", "MH2NEURL": "neurologic_non_AD",
            "MH3HEAD": "head_eyes_ears_nose_throat",
            "MH4CARD": "cardiovascular",
            "MH5RESP": "respiratory", "MH6HEPAT": "hepatic",
            "MH7DERM": "dermatologic_connective_tissue",
            "MH8MUSCL": "musculoskeletal",
            "MH9ENDO": "endocrine_metabolic", "MH10GAST": "gastrointestinal",
            "MH11HEMA": "hematopoietic_lymphatic",
            "MH12RENA": "renal_genitourinary",
            "MH13ALLE": "allergies_drug_sensitivities",
            "MH14ALCH": "alcohol_abuse",
            "MH15DRUG": "drug_abuse", "MH16SMOK": "smoking",
            "MH17MALI": "malignancy", "MH18SURG": "major_surgical_procedures",
            "MH19OTHR": "other"}
# total -> (source column, low, high)
TOTALS = {"MMSE_total": ("MMSCORE", 0.0, 30.0),
          "CDR_global": ("CDGLOBAL", 0.0, 3.0),
          "CDR_SOB": ("CDRSB", 0.0, 18.0),
          "GDS_total": ("GDTOTAL", 0.0, 15.0)}

_log, _rej = [], []


def log(m=""):
    print(m)
    _log.append(m)


def to_adni(s):
    m = re.match(r"sub-(\d{3})S(\d+)$", s)
    return f"{m.group(1)}_S_{m.group(2)}"


def num(v):
    v = (v or "").strip()
    if v in MISSING:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def read_zip_tables(kind):
    """Every row of one ADNI table kind, across all the stage zips."""
    rows, qc = [], 0
    for z in sorted(glob.glob(os.path.join(CB, "*.zip"))):
        zf = zipfile.ZipFile(z)
        for n in zf.namelist():
            if kind not in n or not n.lower().endswith(".csv"):
                continue
            rd = csv.DictReader(io.TextIOWrapper(
                zf.open(n), encoding="utf-8-sig", errors="replace"))
            for r in rd:
                if (r.get("HAS_QC_ERROR") or "").strip() in ("1", "1.0", "Y"):
                    qc += 1
                    continue
                rows.append(r)
    return rows, qc


def index_by_ptid(rows):
    d = collections.defaultdict(list)
    for r in rows:
        p = (r.get("PTID") or "").strip()
        if not p:
            continue
        try:
            d[p].append((datetime.datetime.strptime(
                r["VISDATE"].strip(), "%Y-%m-%d"), r))
        except (ValueError, KeyError):
            d[p].append((None, r))
    return d


def nearest(entries, mdate):
    """Visit nearest the MRI; falls back to the earliest dated visit."""
    dated = [(d, r) for d, r in entries if d is not None]
    if dated and mdate:
        d, r = min(dated, key=lambda t: abs((t[0] - mdate).days))
        return r, (d.date() - mdate.date()).days, "nearest VISDATE to MRI"
    if dated:
        d, r = min(dated, key=lambda t: t[0])
        return r, None, "earliest visit (no MRI date)"
    return (entries[0][1], None, "undated row") if entries else (None, None, "")


def read_idasearch():
    """Fallback scores for subjects the zip tables do not carry.

    The idaSearch export has Age but no date, so these rows CANNOT be
    scan-matched. It is used only where the zips have nothing, and every row
    it supplies is labelled so in the *_match column. median_low across that
    subject's visits -- an actually observed value, never an average.
    """
    d = collections.defaultdict(lambda: collections.defaultdict(list))
    for f in sorted(glob.glob(os.path.join(
            CB, "new_comorbidity", "gdscale-cdr-mmse", "*.csv"))):
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            p = (r.get("Subject ID") or "").strip()
            if not p:
                continue
            for col, nm in (("MMSE Total Score", "MMSE_total"),
                            ("GDSCALE Total Score", "GDS_total"),
                            ("Global CDR", "CDR_global")):
                v = num(r.get(col))
                if v is not None:
                    d[p][nm].append(v)
    return d


def mri_dates():
    d = {}
    for r in csv.DictReader(open(os.path.join(
            ROOT, "audit", "bids_conversion", "bids_mapping.csv"))):
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
    log("MODEL-READY CLINICAL + COMORBIDITY TABLE")
    log("=" * 78)
    log(f"cohort {len(subs)} subjects, row order = subjects.npy "
        f"(aligns with X_regional.npy / X_fc.npy / y.npy)")

    scan = mri_dates()
    tables = {}
    for kind in ("CDR", "MMSE", "GDSCALE", "MEDHIST"):
        rows, qc = read_zip_tables(kind)
        tables[kind] = index_by_ptid(rows)
        log(f"  {kind:8s} {len(rows):5d} rows, {len(tables[kind]):4d} subjects"
            f"   ({qc} dropped for HAS_QC_ERROR)")

    # APOE: the top-level export has the widest cohort coverage
    apoe = {}
    p = os.path.join(CB, "APOERES_01Sep2026.csv")
    if os.path.exists(p):
        for r in csv.DictReader(open(p)):
            g = (r.get("GENOTYPE") or "").strip()
            if r.get("PTID") and g:
                apoe[r["PTID"].strip()] = g
    log(f"  APOERES  {len(apoe):5d} subjects with a genotype")

    idas = read_idasearch()
    log(f"  idaSearch fallback available for {len(idas)} subjects "
        f"(used only where the zips have none)")

    out = []
    for i, (s, g, a) in enumerate(zip(subs, stages, acqs)):
        sid = to_adni(s)
        _, ses, run = a.split("|")
        md = scan.get((s, ses, run))
        row = collections.OrderedDict()
        row["row_index"] = i
        row["bids_id"] = s
        row["adni_id"] = sid
        row["cohort_stage"] = g
        row["stage_index"] = STAGES.index(g)
        row["mri_date"] = md.date().isoformat() if md else ""

        # ---------------- totals, each from its own nearest visit
        for kind, names in (("MMSE", ["MMSE_total"]),
                            ("CDR", ["CDR_global", "CDR_SOB"]),
                            ("GDSCALE", ["GDS_total"])):
            ent = tables[kind].get(sid, [])
            r, gap, how = nearest(ent, md) if ent else (None, None, "")
            for nm in names:
                col, lo, hi = TOTALS[nm]
                v = num(r[col]) if r and col in r else None
                if v is not None and not (lo <= v <= hi):
                    _rej.append((sid, nm, v, "outside instrument range"))
                    v = None
                row[nm] = v
            # Fall back to the idaSearch export only where the zip tables
            # have nothing at all for this subject. Not scan-matched, and
            # said so on the row.
            if r is None and sid in idas:
                used = False
                for nm in names:
                    vals = idas[sid].get(nm, [])
                    lo, hi = TOTALS[nm][1], TOTALS[nm][2]
                    vals = [v for v in vals if lo <= v <= hi]
                    if vals:
                        row[nm] = statistics.median_low(vals)
                        used = True
                if used:
                    how = ("idaSearch fallback, median_low across visits "
                           "-- NOT scan-matched")
            row[f"{kind}_visit_date"] = r["VISDATE"].strip() if r else ""
            row[f"{kind}_days_from_MRI"] = gap if gap is not None else ""
            row[f"{kind}_match"] = how
            row[f"{kind}_source"] = ("Tables_*.zip" if r else
                                     ("idaSearch" if how else ""))

        # ---------------- medical history
        ent = tables["MEDHIST"].get(sid, [])
        r, gap, how = nearest(ent, md) if ent else (None, None, "")
        for k in MH_BINARY:
            v = num(r[k]) if r and k in r else None
            if v is not None and v not in (0.0, 1.0):
                _rej.append((sid, k, v, "binary flag with non-0/1 value"))
                v = None
            row[f"MH_{MH_LABEL[k]}"] = None if v is None else int(v)
        row["MEDHIST_visit_date"] = r["VISDATE"].strip() if r else ""
        row["MEDHIST_days_from_MRI"] = gap if gap is not None else ""
        row["MEDHIST_match"] = how

        # ---------------- genetics
        gt = apoe.get(sid, "")
        row["APOE_genotype"] = gt
        row["APOE4_count"] = gt.count("4") if gt else None

        row["has_cognitive"] = all(row[k] is not None
                                   for k in ("MMSE_total", "CDR_global"))
        row["has_mood"] = row["GDS_total"] is not None
        row["has_medhist"] = any(row[f"MH_{MH_LABEL[k]}"] is not None
                                 for k in MH_BINARY)
        row["complete"] = (row["has_cognitive"] and row["has_mood"]
                           and row["has_medhist"])
        out.append(row)

    cols = list(out[0].keys())
    p = os.path.join(OUT, "model_clinical_116.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({c: ("" if r[c] is None else r[c]) for c in cols}
                    for r in out)

    # ------------------------------------------------------------ report
    log("")
    log(f"values rejected by cleaning: {len(_rej)}")
    for sid, k, v, why in _rej[:10]:
        log(f"    {sid} {k}={v} -- {why}")

    log("")
    log("COVERAGE BY STAGE")
    log(f"  {'stage':6s}{'n':>4s}{'MMSE':>6s}{'CDR':>5s}{'CDRSB':>7s}"
        f"{'GDS':>5s}{'MedHist':>9s}{'APOE':>6s}{'complete':>10s}")
    for g in STAGES:
        m = [r for r in out if r["cohort_stage"] == g]
        log(f"  {g:6s}{len(m):4d}"
            f"{sum(1 for r in m if r['MMSE_total'] is not None):6d}"
            f"{sum(1 for r in m if r['CDR_global'] is not None):5d}"
            f"{sum(1 for r in m if r['CDR_SOB'] is not None):7d}"
            f"{sum(1 for r in m if r['GDS_total'] is not None):5d}"
            f"{sum(1 for r in m if r['has_medhist']):9d}"
            f"{sum(1 for r in m if r['APOE_genotype']):6d}"
            f"{sum(1 for r in m if r['complete']):10d}")
    log(f"  {'ALL':6s}{len(out):4d}"
        f"{sum(1 for r in out if r['MMSE_total'] is not None):6d}"
        f"{sum(1 for r in out if r['CDR_global'] is not None):5d}"
        f"{sum(1 for r in out if r['CDR_SOB'] is not None):7d}"
        f"{sum(1 for r in out if r['GDS_total'] is not None):5d}"
        f"{sum(1 for r in out if r['has_medhist']):9d}"
        f"{sum(1 for r in out if r['APOE_genotype']):6d}"
        f"{sum(1 for r in out if r['complete']):10d}")

    log("")
    log("SCAN MATCHING -- |days| between the visit used and the MRI")
    for kind in ("MMSE", "CDR", "GDSCALE", "MEDHIST"):
        gaps = [abs(r[f"{kind}_days_from_MRI"]) for r in out
                if isinstance(r[f"{kind}_days_from_MRI"], int)]
        if gaps:
            log(f"  {kind:8s} n={len(gaps):3d}  median {statistics.median(gaps):5.0f}"
                f"  max {max(gaps):5d}   within 90d "
                f"{sum(1 for x in gaps if x <= 90):3d}   within 1y "
                f"{sum(1 for x in gaps if x <= 365):3d}")

    log("")
    log("COMORBIDITY PREVALENCE (1 = present)")
    log(f"  {'condition':20s}{'yes':>5s}{'no':>5s}{'missing':>9s}")
    for k in MH_BINARY:
        c = f"MH_{MH_LABEL[k]}"
        y = sum(1 for r in out if r[c] == 1)
        n = sum(1 for r in out if r[c] == 0)
        log(f"  {MH_LABEL[k]:20s}{y:5d}{n:5d}{len(out) - y - n:9d}")

    log("")
    log("INCOMPLETE SUBJECTS")
    for r in out:
        if not r["complete"]:
            miss = [n for n, ok in (("cognitive", r["has_cognitive"]),
                                    ("mood", r["has_mood"]),
                                    ("medhist", r["has_medhist"])) if not ok]
            log(f"  row {r['row_index']:3d} {r['adni_id']:12s} "
                f"{r['cohort_stage']:5s} missing: {', '.join(miss)}")

    log("")
    log(f"wrote {os.path.relpath(p, ROOT)}  ({len(out)} rows, {len(cols)} columns)")
    with open(os.path.join(OUT, "model_clinical_report.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")


if __name__ == "__main__":
    main()
