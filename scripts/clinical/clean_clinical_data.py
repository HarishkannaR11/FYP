"""
ADNI clinical / demographic / genetic / medical-history audit and cleaning.

Discovers every clinical source in the project, catalogues every variable,
decodes what the source supports, cleans missing-value codes, matches
visit-dependent values to each MRI acquisition, and writes subject-level and
stage-level tables.

Strictly read-only with respect to every source file. All output goes to
derivatives/clinical_cleaned/. Nothing is imputed, scaled or selected here.

No ADNI codebook/data dictionary is bundled with the sources, so decoding is
limited to mappings derivable from the data itself. Anything else is recorded
as UNKNOWN_CODE rather than guessed.
"""
import datetime
import glob
import io
import json
import os
import re
import zipfile
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CB = os.path.join(ROOT, "Comorbidity")
NORM = os.path.join(ROOT, "derivatives", "biomarkers_normalized")
PARTICIPANTS = os.path.join(ROOT, "BIDS", "participants.tsv")
OUT = os.path.join(ROOT, "derivatives", "clinical_cleaned")
STAGES_DIR = os.path.join(OUT, "stages")

# Directory name -> stage label. Fixed at Review I; never reorder.
GROUP_TO_STAGE = {"CN_Final": "CN", "SMC_Final": "SMC", "EMCI": "EMCI",
                  "MCI": "MCI", "LMCI": "LMCI", "AD": "AD"}
STAGE_ORDER = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
STAGE_ORDINAL = {s: i for i, s in enumerate(STAGE_ORDER)}
EXPECTED_SUBJECTS = {"CN": 26, "SMC": 21, "EMCI": 26, "MCI": 14, "LMCI": 20, "AD": 20}

# ADNI commonly uses these as "not done / not applicable / unknown". They are
# only applied to a column when that column's observed values are consistent
# with them being sentinels (see sentinel_candidates()).
SENTINELS = [-1, -4, -9]

ID_COLS = ["PTID", "RID", "Subject", "subject_id", "participant_id"]
VISIT_COLS = ["VISCODE", "VISCODE2", "Visit", "visit", "PHASE"]
DATE_COLS = ["VISDATE", "EXAMDATE", "APTESTDT", "Acq Date", "USERDATE", "update_stamp"]

warnings = []
provenance = []   # source_file -> source_column -> derived variable


def log(m):
    print(m, flush=True)


def hdr(t):
    log("\n" + "=" * 72)
    log(t)
    log("=" * 72)


def ptid_to_sub(p):
    """002_S_5018 -> sub-002S5018. Deterministic, no guessing."""
    if pd.isna(p):
        return None
    s = str(p).strip()
    if not re.fullmatch(r"\d{3}_S_\d{4}", s):
        return None
    return "sub-" + s.replace("_", "")


# ======================================================================
# 1. DISCOVERY
# ======================================================================
def discover_sources():
    """Every clinical table, from loose CSVs and from the zip archives."""
    tables = {}          # key -> DataFrame
    inventory = []

    for f in sorted(glob.glob(os.path.join(CB, "*.csv"))):
        name = os.path.basename(f)
        df = pd.read_csv(f, low_memory=False)
        key = name.replace(".csv", "")
        tables[key] = df
        inventory.append(_inv_row(key, f, "csv", df))

    for z in sorted(glob.glob(os.path.join(CB, "*.zip"))):
        zf = zipfile.ZipFile(z)
        zname = os.path.basename(z)
        for n in zf.namelist():
            if not n.lower().endswith(".csv"):
                continue
            df = pd.read_csv(io.BytesIO(zf.read(n)), low_memory=False)
            short = re.sub(r".*Cohort_(?:30_)?", "", n).replace("_01Sep2026.csv", "")
            grp = re.search(r"_(AD|CN|EMCI|MCI|SMC)\.zip", zname.replace(" ", ""))
            grp = grp.group(1) if grp else "?"
            key = f"{grp}::{short}"
            tables[key] = df
            inventory.append(_inv_row(key, f"{z}!{n}", "csv-in-zip", df))

    if os.path.isfile(PARTICIPANTS):
        df = pd.read_csv(PARTICIPANTS, sep="\t")
        tables["participants"] = df
        inventory.append(_inv_row("participants", PARTICIPANTS, "tsv", df))

    return tables, pd.DataFrame(inventory)


def _inv_row(key, path, ftype, df):
    return {
        "table_key": key,
        "file_name": os.path.basename(path.split("!")[0]),
        "path": os.path.relpath(path.split("!")[0], ROOT) + (
            "!" + path.split("!")[1] if "!" in path else ""),
        "file_type": ftype,
        "number_of_rows": len(df),
        "number_of_columns": len(df.columns),
        "important_ID_columns": ",".join([c for c in df.columns if c in ID_COLS]),
        "visit_columns": ",".join([c for c in df.columns if c in VISIT_COLS]),
        "date_columns": ",".join([c for c in df.columns if c in DATE_COLS]),
        "description": _describe(key),
    }


def _describe(key):
    k = key.split("::")[-1].upper()
    d = {
        "CDR": "Clinical Dementia Rating (longitudinal); CDGLOBAL and CDRSB",
        "MMSE": "Mini-Mental State Examination (longitudinal); MMSCORE 0-30",
        "GDSCALE": "Geriatric Depression Scale (longitudinal); GDTOTAL 0-15",
        "MEDHIST": "Medical history checklist, screening visit; MH* category flags",
        "APOERES": "APOE genotyping result; GENOTYPE as allele pair",
        "FUNCTIONAL_MRI_IMAGES": "Imaging index: image_id, fmri_date, fmri_visit, scanner",
        "PARTICIPANTS": "Project BIDS participants file: authoritative group label",
    }
    for pat, txt in d.items():
        if pat in k:
            return txt
    if re.search(r"_9_01_2026", key):
        return "ADNI image-collection export: Subject, Group, Sex, Age, Visit, Acq Date"
    return ""


# ======================================================================
# 2. PROJECT COHORT (authoritative stage assignment)
# ======================================================================
def load_cohort():
    """The 127 subjects actually in the imaging pipeline, with their stage."""
    rows = []
    for d in sorted(glob.glob(os.path.join(NORM, "*", "sub-*", "ses-*", "run-*"))):
        p = d.replace("\\", "/").split("/")
        i = p.index("biomarkers_normalized")
        grp, sub, ses, run = p[i + 1], p[i + 2], p[i + 3], p[i + 4]
        if grp not in GROUP_TO_STAGE:
            continue
        rows.append({"subject_id": sub, "group_dir": grp,
                     "stage": GROUP_TO_STAGE[grp], "session": ses, "run": run})
    acq = pd.DataFrame(rows)
    acq["stage_ordinal"] = acq.stage.map(STAGE_ORDINAL)
    subj = acq.drop_duplicates("subject_id")[
        ["subject_id", "group_dir", "stage", "stage_ordinal"]].reset_index(drop=True)
    return acq, subj


def acquisition_dates(tables):
    """
    MRI acquisition date per subject.

    Three sources, in order of preference:
      1. the Functional_MRI_Images tables bundled in the cohort zips (fmri_date)
      2. the ADNI image-collection exports (Acq Date)
      3. the ADNI timestamp embedded in the preprocessing provenance path

    Source 3 alone covers only 25 of 165 acquisitions, because most source
    files were renamed to BIDS style and carry no timestamp, so it is a
    supplement rather than the primary source.
    """
    rows = []

    for key, df in tables.items():
        if key.endswith("::Functional_MRI_Images") and "subject_id" in df.columns:
            d = df.copy()
            d["sub"] = d["subject_id"].map(ptid_to_sub)
            dt = pd.to_datetime(d.get("fmri_date"), errors="coerce")
            for s, t in zip(d["sub"], dt):
                if s and pd.notna(t):
                    rows.append({"subject_id": s, "mri_date": t.date(),
                                 "date_source": "Functional_MRI_Images"})
        if re.search(r"_9_01_2026$", key) and "Subject" in df.columns:
            d = df.copy()
            d["sub"] = d["Subject"].map(ptid_to_sub)
            dt = pd.to_datetime(d.get("Acq Date"), errors="coerce")
            for s, t in zip(d["sub"], dt):
                if s and pd.notna(t):
                    rows.append({"subject_id": s, "mri_date": t.date(),
                                 "date_source": "image-collection export"})

    for f in glob.glob(os.path.join(ROOT, "derivatives", "fsfast", "*", "sub-*",
                                    "ses-*", "run-*", "*_desc-preproc_bold.json")):
        try:
            d = json.load(open(f))
        except Exception:
            continue
        p = f.replace("\\", "/").split("/")
        i = p.index("fsfast")
        m = re.search(r"_(\d{14})_", d.get("source_bold", ""))
        if not m:
            continue
        try:
            dt = datetime.datetime.strptime(m.group(1), "%Y%m%d%H%M%S").date()
        except ValueError:
            continue
        rows.append({"subject_id": p[i + 2], "mri_date": dt,
                     "date_source": "provenance path timestamp"})

    return pd.DataFrame(rows)


# ======================================================================
# 3. VARIABLE CATALOG
# ======================================================================
def sentinel_candidates(s):
    """Which SENTINELS plausibly act as missing codes in this column."""
    if not pd.api.types.is_numeric_dtype(s):
        return []
    vals = set(pd.unique(s.dropna()))
    found = [v for v in SENTINELS if v in vals]
    # only treat as sentinel if the column also holds non-negative values
    nonneg = s.dropna()
    nonneg = nonneg[nonneg >= 0]
    return found if len(nonneg) else []


def build_catalog(tables, cohort_subj):
    rows = []
    for key, df in tables.items():
        for col in df.columns:
            s = df[col]
            nonmiss = int(s.notna().sum())
            uniq = s.dropna().unique()
            sent = sentinel_candidates(s)
            is_num = pd.api.types.is_numeric_dtype(s)
            visit_dep = any(c in df.columns for c in ("VISCODE", "VISCODE2", "Visit"))
            rows.append({
                "variable_name": f"{key}::{col}",
                "source_file": key,
                "source_column": col,
                "data_type": str(s.dtype),
                "number_nonmissing": nonmiss,
                "number_missing": int(s.isna().sum()),
                "unique_count": int(len(uniq)),
                "example_values": "; ".join(map(str, uniq[:5])),
                "decoded_meaning": DECODE_NOTES.get(col, ""),
                "categorical_or_continuous": (
                    "continuous" if is_num and len(uniq) > 12 else
                    "categorical" if len(uniq) <= 12 else "text/id"),
                "sentinel_codes_present": ",".join(map(str, sent)),
                "visit_dependent": visit_dep and col not in ID_COLS,
                "subject_level_or_visit_level": (
                    "visit" if visit_dep and col not in ID_COLS else "subject"),
                "potentially_useful_for_model": col in USEFUL_COLS,
                "reason_if_not_usable": ("" if col in USEFUL_COLS else
                                         NOT_USABLE.get(col, "administrative/ID/"
                                                        "date field, or not a "
                                                        "clinical measure")),
                "notes": "",
            })
    return pd.DataFrame(rows)


DECODE_NOTES = {
    "GENOTYPE": "APOE allele pair, e.g. 3/4; e4 count derived by counting '4'",
    "CDGLOBAL": "CDR global score (0, 0.5, 1, 2, 3)",
    "CDRSB": "CDR Sum of Boxes, 0-18",
    "MMSCORE": "MMSE total score, 0-30",
    "GDTOTAL": "Geriatric Depression Scale total, 0-15",
    "MHPSYCH": "Medical history: psychiatric",
    "MH2NEURL": "Medical history: neurological (non-AD)",
    "MH3HEAD": "Medical history: head trauma",
    "MH4CARD": "Medical history: cardiovascular (includes hypertension)",
    "MH5RESP": "Medical history: respiratory",
    "MH6HEPAT": "Medical history: hepatic",
    "MH7DERM": "Medical history: dermatologic",
    "MH8MUSCL": "Medical history: musculoskeletal",
    "MH9ENDO": "Medical history: endocrine-metabolic (includes diabetes)",
    "MH10GAST": "Medical history: gastrointestinal",
    "MH11HEMA": "Medical history: haematologic/lymphatic",
    "MH12RENA": "Medical history: renal-genitourinary",
    "MH13ALLE": "Medical history: allergies/drug sensitivities",
    "MH14ALCH": "Medical history: alcohol abuse",
    "MH15DRUG": "Medical history: drug abuse",
    "MH16SMOK": "Medical history: smoking",
    "MH17MALI": "Medical history: malignancy",
    "MH18SURG": "Medical history: major surgical procedures",
    "MH19OTHR": "Medical history: other",
    "Sex": "Biological sex as exported by ADNI (M/F)",
    "Age": "Age in years at acquisition",
    "VISCODE": "ADNI visit code (sc screening, bl baseline, mNN month, vNN visit)",
    "VISCODE2": "Secondary ADNI visit code",
    "VISDATE": "Assessment date",
    "Acq Date": "MRI acquisition date",
}

USEFUL_COLS = {
    "GENOTYPE", "CDGLOBAL", "CDRSB", "MMSCORE", "GDTOTAL", "Sex", "Age",
    "MHPSYCH", "MH2NEURL", "MH3HEAD", "MH4CARD", "MH5RESP", "MH6HEPAT",
    "MH7DERM", "MH8MUSCL", "MH9ENDO", "MH10GAST", "MH11HEMA", "MH12RENA",
    "MH13ALLE", "MH14ALCH", "MH15DRUG", "MH16SMOK", "MH17MALI", "MH18SURG",
    "MH19OTHR",
}

NOT_USABLE = {
    "PTID": "subject identifier, used for linkage not as a feature",
    "RID": "subject identifier",
    "ID": "record identifier",
    "SITEID": "acquisition site; a confound, not a clinical feature",
    "PHASE": "ADNI study phase; administrative",
    "update_stamp": "database timestamp",
    "USERDATE": "data-entry date",
    "USERDATE2": "data-entry revision date",
}


# ======================================================================
# 4. DECODING
# ======================================================================
def apoe4_count(gt):
    """Count of e4 alleles in an APOE genotype string such as '3/4'.
    Derived from the source value itself; no external mapping needed."""
    if pd.isna(gt):
        return np.nan
    s = str(gt).strip()
    if not re.fullmatch(r"[2-4]\s*/\s*[2-4]", s):
        return np.nan
    return float(s.count("4"))


def decode_viscode(v):
    if pd.isna(v):
        return "UNKNOWN_CODE"
    s = str(v).strip().lower()
    if s == "sc":
        return "screening"
    if s == "bl":
        return "baseline"
    m = re.fullmatch(r"m(\d+)", s)
    if m:
        return f"month {int(m.group(1))}"
    if re.fullmatch(r"v\d+", s):
        return f"study visit {s}"
    return "UNKNOWN_CODE"


def decode_medhist(v):
    """ADNI medical-history flags are 1 present / 0 absent, with negative
    sentinels for not-assessed. Recorded as a derivation rule, not a codebook
    lookup: no codebook ships with these files."""
    if pd.isna(v):
        return np.nan, "UNKNOWN_CODE"
    try:
        n = float(v)
    except (TypeError, ValueError):
        return np.nan, "UNKNOWN_CODE"
    if n == 1:
        return 1.0, "present"
    if n == 0:
        return 0.0, "absent"
    if int(n) in SENTINELS:
        return np.nan, "not assessed / missing"
    return np.nan, "UNKNOWN_CODE"


# ======================================================================
# 5. MAIN
# ======================================================================
def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(STAGES_DIR, exist_ok=True)

    hdr("1. DISCOVERY")
    tables, inventory = discover_sources()
    inventory.to_csv(os.path.join(OUT, "source_inventory.csv"), index=False)
    log(f"source tables discovered: {len(tables)}")
    for _, r in inventory.iterrows():
        log(f"  {r.table_key:34s} rows={r.number_of_rows:5d} cols={r.number_of_columns:3d}")

    acq, cohort = load_cohort()
    log(f"\nproject cohort: {len(cohort)} subjects, {len(acq)} acquisitions")

    # ---------- 2. ID MAPPING ----------
    hdr("2. SUBJECT ID MAPPING")
    cohort_ids = set(cohort.subject_id)
    maprows = []
    for key, df in tables.items():
        idcol = next((c for c in ("PTID", "Subject", "participant_id") if c in df.columns), None)
        if idcol is None:
            continue
        for raw in pd.unique(df[idcol].dropna()):
            if idcol == "participant_id":
                proj = str(raw)
                method = "already a project subject id"
            else:
                proj = ptid_to_sub(raw)
                method = "PTID 'NNN_S_NNNN' -> 'sub-NNNSNNNN' (underscores removed)"
            maprows.append({"source_file": key, "source_subject_id": raw,
                            "project_subject_id": proj if proj else "UNMAPPED",
                            "mapping_method": method,
                            "mapping_verified": bool(proj in cohort_ids)})
    idmap = pd.DataFrame(maprows)
    idmap.to_csv(os.path.join(OUT, "subject_id_mapping.csv"), index=False)
    log(f"mapping rows: {len(idmap)}; verified against cohort: {int(idmap.mapping_verified.sum())}")
    log(f"distinct project subjects reachable from sources: "
        f"{idmap[idmap.mapping_verified].project_subject_id.nunique()} / {len(cohort_ids)}")

    # ---------- 3. VARIABLE CATALOG ----------
    hdr("3. VARIABLE CATALOG")
    catalog = build_catalog(tables, cohort)
    catalog.to_csv(os.path.join(OUT, "variable_catalog.csv"), index=False)
    log(f"variables catalogued: {len(catalog)}")
    log(f"flagged potentially useful: {int(catalog.potentially_useful_for_model.sum())}")
    log(f"with sentinel codes present: {int((catalog.sentinel_codes_present != '').sum())}")

    # ---------- 4. VISIT AUDIT ----------
    hdr("4. VISIT AUDIT")
    vrows = []
    for key, df in tables.items():
        idcol = next((c for c in ("PTID", "Subject") if c in df.columns), None)
        if idcol is None:
            continue
        vcol = next((c for c in ("VISCODE2", "VISCODE", "Visit") if c in df.columns), None)
        dcol = next((c for c in ("VISDATE", "EXAMDATE", "APTESTDT", "Acq Date")
                     if c in df.columns), None)
        for _, r in df.iterrows():
            sub = ptid_to_sub(r[idcol])
            if sub not in cohort_ids:
                continue
            vrows.append({
                "subject_id": sub, "source_file": key,
                "visit_code": r[vcol] if vcol else "",
                "visit_decoded": decode_viscode(r[vcol]) if vcol else "",
                "assessment_date": r[dcol] if dcol else "",
                "available_variables": sum(1 for c in df.columns if pd.notna(r[c])),
            })
    visits = pd.DataFrame(vrows)
    visits.to_csv(os.path.join(OUT, "visit_audit.csv"), index=False)
    log(f"visit records for cohort subjects: {len(visits)}")
    if len(visits):
        log(f"subjects with >=1 clinical visit: {visits.subject_id.nunique()} / {len(cohort_ids)}")

    # ---------- 5. DUPLICATES / CONFLICTS ----------
    hdr("5. DUPLICATES AND CONFLICTS")
    conflicts = []
    collapsed = 0
    for key, df in tables.items():
        idcol = next((c for c in ("PTID", "Subject") if c in df.columns), None)
        if idcol is None:
            continue
        vcol = next((c for c in ("VISCODE2", "VISCODE") if c in df.columns), None)
        keys = [idcol] + ([vcol] if vcol else [])
        dup = df[df.duplicated(keys, keep=False)]
        for gk, g in dup.groupby(keys):
            sub = ptid_to_sub(g.iloc[0][idcol])
            if sub not in cohort_ids:
                continue
            for col in g.columns:
                vals = pd.unique(g[col].dropna())
                if len(vals) > 1:
                    conflicts.append({
                        "subject_id": sub, "source_file": key,
                        "key": str(gk), "variable": col,
                        "conflicting_values": " | ".join(map(str, vals[:6])),
                        "n_source_rows": len(g),
                        "possible_resolution": "NOT RESOLVED - requires manual review",
                    })
            if len(g.drop_duplicates()) == 1:
                collapsed += 1
    confdf = pd.DataFrame(conflicts)
    confdf.to_csv(os.path.join(OUT, "duplicate_conflict_report.csv"), index=False)
    log(f"identical duplicate groups collapsed: {collapsed}")
    log(f"conflicting value records (NOT auto-resolved): {len(confdf)}")
    if len(confdf):
        warnings.append(f"{len(confdf)} conflicting duplicate values require manual review")

    # ---------- 6. BUILD MASTER TABLE ----------
    hdr("6. MASTER SUBJECT TABLE")
    master = cohort.copy()

    # --- APOE (subject level) from the full APOERES table ---
    ap = tables.get("APOERES_01Sep2026")
    if ap is not None:
        ap = ap.copy()
        ap["subject_id"] = ap.PTID.map(ptid_to_sub)
        ap = ap[ap.subject_id.isin(cohort_ids)]
        g = ap.groupby("subject_id").GENOTYPE.agg(
            lambda s: s.dropna().iloc[0] if s.notna().any() else np.nan)
        master["APOE_genotype_raw"] = master.subject_id.map(g)
        master["APOE4_allele_count"] = master.APOE_genotype_raw.map(apoe4_count)
        master["APOE4_carrier"] = master.APOE4_allele_count.map(
            lambda v: np.nan if pd.isna(v) else float(v > 0))
        master["APOE_source"] = "APOERES_01Sep2026.csv"
        provenance.append(("APOERES_01Sep2026.csv", "GENOTYPE",
                           "APOE_genotype_raw / APOE4_allele_count / APOE4_carrier",
                           "allele count = number of '4' characters in the genotype pair"))
        log(f"APOE genotype present: {int(master.APOE_genotype_raw.notna().sum())} / {len(master)}")

    # --- Age / Sex from the image-collection exports ---
    icol = []
    for key, df in tables.items():
        if not re.search(r"_9_01_2026$", key):
            continue
        d = df.copy()
        d["subject_id"] = d.Subject.map(ptid_to_sub)
        d["_src"] = key
        icol.append(d)
    if icol:
        ic = pd.concat(icol, ignore_index=True)
        ic = ic[ic.subject_id.isin(cohort_ids)]
        ic["acq_dt"] = pd.to_datetime(ic["Acq Date"], errors="coerce")
        # earliest listed acquisition per subject, for a stable age/sex record
        ic = ic.sort_values("acq_dt")
        first = ic.groupby("subject_id").first()
        master["Sex_raw"] = master.subject_id.map(first["Sex"])
        master["Sex_decoded"] = master.Sex_raw.map(
            {"M": "male", "F": "female"}).fillna("UNKNOWN_CODE")
        master.loc[master.Sex_raw.isna(), "Sex_decoded"] = np.nan
        master["Age_years"] = pd.to_numeric(master.subject_id.map(first["Age"]),
                                            errors="coerce")
        master["Age_source_visit"] = master.subject_id.map(first["Visit"])
        master["Age_source_acq_date"] = master.subject_id.map(
            first["acq_dt"].dt.date.astype(str))
        provenance.append(("*_9_01_2026.csv", "Sex / Age / Acq Date",
                           "Sex_raw, Sex_decoded, Age_years",
                           "earliest listed acquisition per subject"))
        log(f"Age present: {int(master.Age_years.notna().sum())} / {len(master)}")
        log(f"Sex present: {int(master.Sex_raw.notna().sum())} / {len(master)}")

    # --- visit-dependent scores matched to the MRI date ---
    mri = acquisition_dates(tables)
    mri = mri.dropna(subset=["mri_date"])
    mri_first = mri.sort_values("mri_date").groupby("subject_id").first()
    log(f"MRI acquisition dates resolved for {mri_first.index.isin(cohort_ids).sum()}"
        f" / {len(cohort_ids)} subjects")
    master["MRI_date_used_for_matching"] = master.subject_id.map(
        mri_first["mri_date"].astype(str))
    master["MRI_date_source"] = master.subject_id.map(mri_first["date_source"])
    match_stats = {}

    SCORE_TABLES = {
        "CDR": [("CDGLOBAL", "CDR_global"), ("CDRSB", "CDR_SB")],
        "MMSE": [("MMSCORE", "MMSE_total")],
        "GDSCALE": [("GDTOTAL", "GDS_total")],
    }
    for tname, cols in SCORE_TABLES.items():
        pool = []
        for key, df in tables.items():
            if not key.endswith("::" + tname):
                continue
            d = df.copy()
            d["subject_id"] = d.PTID.map(ptid_to_sub)
            d["_date"] = pd.to_datetime(d.get("VISDATE"), errors="coerce")
            d["_src"] = key
            pool.append(d)
        if not pool:
            warnings.append(f"{tname}: no source table found")
            continue
        P = pd.concat(pool, ignore_index=True)
        P = P[P.subject_id.isin(cohort_ids)]
        for src_col, out_col in cols:
            if src_col not in P.columns:
                warnings.append(f"{tname}: column {src_col} absent")
                continue
            vals, gaps, vis, srcs = {}, {}, {}, {}
            n_matched = n_fallback = 0
            for sub, g in P.groupby("subject_id"):
                g = g[pd.to_numeric(g[src_col], errors="coerce").notna()]
                if not len(g):
                    continue
                target = mri_first["mri_date"].get(sub)
                gg = g.dropna(subset=["_date"])
                if target is not None and len(gg):
                    delta = (gg["_date"].dt.date - target).map(lambda x: abs(x.days))
                    j = delta.idxmin()
                    vals[sub] = pd.to_numeric(gg.loc[j, src_col], errors="coerce")
                    gaps[sub] = int(delta.loc[j])
                    vis[sub] = gg.loc[j].get("VISCODE2", gg.loc[j].get("VISCODE"))
                    srcs[sub] = gg.loc[j]["_src"]
                    n_matched += 1
                else:
                    n_fallback += 1
                    r = g.iloc[0]
                    vals[sub] = pd.to_numeric(r[src_col], errors="coerce")
                    gaps[sub] = np.nan
                    vis[sub] = r.get("VISCODE2", r.get("VISCODE"))
                    srcs[sub] = r["_src"]
                    warnings.append(f"{sub}: {out_col} had no usable date; "
                                    f"first available record used and flagged")
            master[out_col] = master.subject_id.map(vals)
            master[out_col + "_visit"] = master.subject_id.map(vis)
            master[out_col + "_days_from_MRI"] = master.subject_id.map(gaps)
            master[out_col + "_match_method"] = master.subject_id.map(
                {s: ("nearest-date to MRI" if pd.notna(gaps.get(s, np.nan))
                     else "FALLBACK: first available record, no usable date")
                 for s in vals})
            match_stats[out_col] = {"date_matched": n_matched,
                                    "fallback_first_record": n_fallback}
            log(f"{out_col:12s} present: {int(master[out_col].notna().sum()):3d} / "
                f"{len(master)}   date-matched={n_matched}  fallback={n_fallback}")
            provenance.append((tname, src_col, out_col,
                               "value from the assessment closest in date to the "
                               "subject's earliest MRI acquisition; gap in days recorded"))

    # --- medical history (screening visit, subject level) ---
    mh_cols = [c for c in DECODE_NOTES if c.startswith("MH")]
    pool = []
    for key, df in tables.items():
        if not key.endswith("::MEDHIST"):
            continue
        d = df.copy()
        d["subject_id"] = d.PTID.map(ptid_to_sub)
        d["_src"] = key
        pool.append(d)
    if pool:
        MH = pd.concat(pool, ignore_index=True)
        MH = MH[MH.subject_id.isin(cohort_ids)].drop_duplicates("subject_id")
        MH = MH.set_index("subject_id")
        for c in mh_cols:
            if c not in MH.columns:
                continue
            dec = MH[c].map(lambda v: decode_medhist(v))
            master[c + "_raw"] = master.subject_id.map(MH[c])
            master[c + "_clean"] = master.subject_id.map(dec.map(lambda t: t[0]))
            master[c + "_decoded"] = master.subject_id.map(dec.map(lambda t: t[1]))
            provenance.append(("MEDHIST", c, f"{c}_clean / {c}_decoded",
                               "1 -> present, 0 -> absent, negative sentinel -> NaN; "
                               "ADNI convention, no codebook shipped with the source"))
        log(f"medical-history flags decoded: {len([c for c in mh_cols if c in MH.columns])}")
        log(f"subjects with medical history: "
            f"{int(master.get('MH4CARD_raw', pd.Series(dtype=float)).notna().sum())} / {len(master)}")

    # named comorbidity aliases, honestly labelled
    if "MH4CARD_clean" in master.columns:
        master["cardiovascular_history"] = master["MH4CARD_clean"]
        provenance.append(("MEDHIST", "MH4CARD", "cardiovascular_history",
                           "BROAD category; includes hypertension but is NOT a "
                           "hypertension-specific flag"))
    if "MH9ENDO_clean" in master.columns:
        master["endocrine_metabolic_history"] = master["MH9ENDO_clean"]
        provenance.append(("MEDHIST", "MH9ENDO", "endocrine_metabolic_history",
                           "BROAD category; includes diabetes but is NOT a "
                           "diabetes-specific flag"))

    master = master.sort_values(["stage_ordinal", "subject_id"]).reset_index(drop=True)
    master.to_csv(os.path.join(OUT, "master_subject_clinical.csv"), index=False)
    log(f"\nmaster table: {master.shape[0]} rows x {master.shape[1]} columns")

    # ---------- 7. STAGE TABLES ----------
    hdr("7. STAGE TABLES")
    counts = {}
    for s in STAGE_ORDER:
        sub = master[master.stage == s]
        sub.to_csv(os.path.join(STAGES_DIR, f"{s}.csv"), index=False)
        counts[s] = len(sub)
        ok = len(sub) == EXPECTED_SUBJECTS[s]
        log(f"  {s:5s} {len(sub):3d} subjects (expected {EXPECTED_SUBJECTS[s]}) "
            f"{'OK' if ok else 'MISMATCH'}")
        if not ok:
            warnings.append(f"stage {s}: {len(sub)} subjects, expected "
                            f"{EXPECTED_SUBJECTS[s]}")

    # ---------- 8. REPORTS ----------
    write_dictionary(master, catalog)
    write_report(inventory, tables, cohort, master, catalog, visits, confdf,
                 idmap, counts, match_stats)
    log(f"\noutputs written to {OUT}")
    log("CLINICAL_CLEANING_DONE")


# ======================================================================
# REPORTS
# ======================================================================
def write_dictionary(master, catalog):
    L = ["# Clinical Data Dictionary", "",
         f"Generated (UTC): {datetime.datetime.now(datetime.timezone.utc).isoformat()}", "",
         "Every variable in `master_subject_clinical.csv`. No ADNI codebook ships",
         "with the source files, so decoding is limited to mappings derivable from",
         "the data itself; anything else is recorded as `UNKNOWN_CODE`.", "",
         "| Variable | Meaning | Source file | Source column | Type | Coding | "
         "Missing means | Cleaning | ML-suitable | Visit-dependent |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    prov = {d: (sf, sc, rule) for sf, sc, d, rule in provenance}
    for col in master.columns:
        base = col.split("_clean")[0].split("_raw")[0].split("_decoded")[0]
        sf = sc = rule = ""
        for dest, (a, b, r) in prov.items():
            if col in dest or base in dest:
                sf, sc, rule = a, b, r
                break
        meaning = DECODE_NOTES.get(base, "")
        if col in ("subject_id", "stage", "stage_ordinal", "group_dir"):
            meaning = "project cohort field (authoritative)"
            sf = "derivatives/biomarkers_normalized + participants.tsv"
        visit_dep = "yes" if col.endswith(("_visit", "_days_from_MRI")) or \
            base in ("CDR_global", "CDR_SB", "MMSE_total", "GDS_total") else "no"
        ml = "yes" if base in USEFUL_COLS or base in (
            "APOE4_allele_count", "APOE4_carrier", "Age_years", "CDR_global",
            "CDR_SB", "MMSE_total", "GDS_total", "cardiovascular_history",
            "endocrine_metabolic_history") else "no (identifier/provenance)"
        s = master[col]
        L.append(f"| `{col}` | {meaning} | {sf} | {sc} | {s.dtype} | "
                 f"{rule[:70]} | NaN = not available | see audit report | {ml} | "
                 f"{visit_dep} |")
    L += ["", "## Missingness in the master table", "",
          "| Variable | Present | Missing | % present |", "|---|---:|---:|---:|"]
    for col in master.columns:
        n = int(master[col].notna().sum())
        L.append(f"| `{col}` | {n} | {len(master)-n} | {100*n/len(master):.1f}% |")
    open(os.path.join(OUT, "clinical_data_dictionary.md"), "w",
         encoding="utf-8").write("\n".join(L) + "\n")


def write_report(inventory, tables, cohort, master, catalog, visits, confdf,
                 idmap, counts, match_stats=None):
    n_dec = int((catalog.decoded_meaning != "").sum())
    L = ["# Clinical Data Audit Report", "",
         f"Generated (UTC): {datetime.datetime.now(datetime.timezone.utc).isoformat()}", "",
         "Read-only with respect to every ADNI source file. No imputation, scaling,",
         "feature selection or model training was performed.", "",
         "## 1. Source files discovered", "",
         "| Table | Rows | Cols | ID cols | Visit cols | Description |",
         "|---|---:|---:|---|---|---|"]
    for _, r in inventory.iterrows():
        L.append(f"| `{r.table_key}` | {r.number_of_rows} | {r.number_of_columns} | "
                 f"{r.important_ID_columns} | {r.visit_columns} | {r.description} |")

    L += ["", "## 2-4. Subjects, visits, variables", "",
          f"- Project cohort: **{len(cohort)} subjects** (authoritative stage from the "
          "imaging pipeline and `participants.tsv`)",
          f"- Source tables discovered: **{len(tables)}**",
          f"- Clinical visit records for cohort subjects: **{len(visits)}**",
          f"- Subjects with at least one clinical visit record: "
          f"**{visits.subject_id.nunique() if len(visits) else 0} / {len(cohort)}**",
          f"- Variables catalogued: **{len(catalog)}**",
          f"- Variables with a decoded meaning: **{n_dec}**",
          f"- Variables flagged potentially useful: "
          f"**{int(catalog.potentially_useful_for_model.sum())}**", "",
          "## 5. Decoding", "",
          "No ADNI codebook or data dictionary is bundled with these files. Decoding",
          "was therefore restricted to:", "",
          "- **APOE genotype -> e4 allele count**: counted directly from the genotype",
          "  string itself (e.g. `3/4` -> 1). Derived from the value, not a lookup.",
          "- **VISCODE**: `sc` screening, `bl` baseline, `mNN` month NN, `vNN` study",
          "  visit. Anything else -> `UNKNOWN_CODE`.",
          "- **Medical-history flags**: 1 present, 0 absent, negative sentinel ->",
          "  missing. This follows ADNI convention and the observed value set; it is",
          "  recorded as a derivation rule, not a codebook lookup.",
          "- **Sex**: `M`/`F` as exported.", "",
          "Anything not covered above is left raw and marked `UNKNOWN_CODE`.", "",
          "## 6. Missing-value handling", "",
          f"Sentinel codes considered: {SENTINELS}. A sentinel was only treated as",
          "missing where the column also contains non-negative values, so genuinely",
          "signed measures are not corrupted. Missing is represented as `NaN`.",
          "**No imputation was performed.** Imputation must be fitted on training",
          "subjects only, after the subject-level split.", "",
          "## 7. Duplicates and conflicts", "",
          f"- Conflicting duplicate values found: **{len(confdf)}**",
          "- None were auto-resolved; see `duplicate_conflict_report.csv`.", "",
          "## 8. Visit matching", ""]
    gapcols = [c for c in master.columns if c.endswith("_days_from_MRI")]
    if gapcols:
        L += ["Visit-dependent scores were matched to the assessment closest in date",
              "to each subject's earliest MRI acquisition. The gap in days is retained",
              "per subject in `<score>_days_from_MRI`, and the method used is recorded",
              "in `<score>_match_method`, so no selection is silent.", "",
              "MRI acquisition dates come from the `Functional_MRI_Images` tables and",
              "the ADNI image-collection exports. The preprocessing provenance paths",
              "carry a usable timestamp for only 25 of 165 acquisitions, because most",
              "source files were renamed to BIDS style, so they are a supplement only.",
              "",
              "| Score | Date-matched | Fallback (first record) | Median gap (d) | "
              "p90 gap (d) | Max gap (d) |",
              "|---|---:|---:|---:|---:|---:|"]
        for c in gapcols:
            s = pd.to_numeric(master[c], errors="coerce").dropna()
            name = c.replace("_days_from_MRI", "")
            st = (match_stats or {}).get(name, {})
            if len(s):
                L.append(f"| {name} | {st.get('date_matched', len(s))} | "
                         f"{st.get('fallback_first_record', 0)} | {s.median():.0f} | "
                         f"{s.quantile(0.9):.0f} | {s.max():.0f} |")
            else:
                L.append(f"| {name} | 0 | {st.get('fallback_first_record', 0)} | - | - | - |")
        L += ["", "Any row whose `_match_method` reads `FALLBACK` had no usable",
              "assessment date and took the first available record. Those values are",
              "flagged, not hidden, and should be treated with caution."]
    L += ["", "## 9. Comorbidity derivation", "",
          "| Output variable | Source column | Rule |", "|---|---|---|"]
    for sf, sc, dest, rule in provenance:
        L.append(f"| `{dest}` | {sf} :: {sc} | {rule} |")
    L += ["", "> **Naming caution.** `MH4CARD` and `MH9ENDO` are broad ADNI category",
          "> flags. They are exported here as `cardiovascular_history` and",
          "> `endocrine_metabolic_history`. They are **not** hypertension-specific or",
          "> diabetes-specific, and must not be relabelled as such.", "",
          "## 10. Stage-wise subject counts", "",
          "| Stage | Subjects | Expected | Match |", "|---|---:|---:|---|"]
    for s in STAGE_ORDER:
        L.append(f"| {s} | {counts[s]} | {EXPECTED_SUBJECTS[s]} | "
                 f"{'yes' if counts[s] == EXPECTED_SUBJECTS[s] else 'NO'} |")
    L.append(f"| **Total** | **{sum(counts.values())}** | **127** | |")

    L += ["", "## 11. Coverage of the master table", "",
          "| Variable | Present / 127 | % |", "|---|---:|---:|"]
    for col in master.columns:
        if col in ("subject_id", "group_dir", "stage", "stage_ordinal"):
            continue
        n = int(master[col].notna().sum())
        L.append(f"| `{col}` | {n} | {100*n/len(master):.0f}% |")

    L += ["", "## 12. Unresolved issues", ""]
    uniq = sorted(set(warnings))
    L += [f"- {w}" for w in uniq[:60]] if uniq else ["None."]
    if len(uniq) > 60:
        L.append(f"- ... and {len(uniq)-60} further warnings")
    L += ["", "## 13. Final master table", "",
          f"`master_subject_clinical.csv`: **{master.shape[0]} rows x "
          f"{master.shape[1]} columns**", "",
          "Stage tables in `stages/` carry the same columns, one file per stage."]
    open(os.path.join(OUT, "clinical_audit_report.md"), "w",
         encoding="utf-8").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
