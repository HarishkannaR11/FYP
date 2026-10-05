"""
WITHIN-SUBJECT biomarker normalization for the 165 processed acquisitions.

Reads the existing raw biomarker outputs and writes normalized copies into a
NEW tree. Strictly read-only with respect to derivatives/fsfast/**/biomarkers/:
a hard guard refuses any write path containing a 'biomarkers' component.

Leakage-free by construction: every statistic used comes from that one
acquisition's own 246 ROI values. No cross-subject statistic is computed
anywhere in this script.

    ALFF  -> mALFF  = ALFF / mean(ALFF)                 (ratio, not z-score)
    ReHo  -> mReHo  = ReHo / mean(ReHo)
    DC    -> DC_z   = (DC - mean(DC)) / std(DC, ddof=0)
    FC    -> Fisher z = arctanh(clip(r, +-0.999999)), diagonal exactly 0
    FC_Strength -> EXCLUDED (it is exactly DC/245; no independent information)

Modes:
  run     normalize all discovered acquisitions (resumable)
  audit   rebuild the dataset-level QC report from what is on disk
"""
import os
import sys
import csv
import json
import glob
import platform
import datetime
import traceback

import numpy as np
import pandas as pd

SCRIPT_VERSION = "1.0.0"

ROOT = "/mnt/c/Users/krish/FYP"
DERIV = os.path.join(ROOT, "derivatives", "fsfast")
OUT_ROOT = os.path.join(ROOT, "derivatives", "biomarkers_normalized")
ATLAS_LUT = os.path.join(ROOT, "atlases", "Brainnetome246", "BN_Atlas_246_LUT.txt")

STATE_JSON = os.path.join(OUT_ROOT, "normalization_state.json")
QC_CSV = os.path.join(OUT_ROOT, "normalization_qc.csv")
QC_MD = os.path.join(OUT_ROOT, "normalization_qc_report.md")

GROUPS = ["AD", "CN_Final", "EMCI", "LMCI", "MCI", "SMC_Final"]
EXPECTED_PER_GROUP = {"AD": 25, "CN_Final": 27, "EMCI": 29, "LMCI": 21,
                      "MCI": 32, "SMC_Final": 31}
EXPECTED_TOTAL = 165

N_ROI = 246
DC_DDOF = 0
FC_CLIP = 0.999999
FC_STRENGTH_DIVISOR = float(N_ROI - 1)   # 245

INPUT_FILES = ["alff.npy", "reho.npy", "degree_centrality.npy",
               "fc_strength.npy", "fc_matrix.npy"]
OUTPUT_FILES = ["malff.npy", "malff.csv", "mreho.npy", "mreho.csv",
                "dc_z.npy", "dc_z.csv", "fc_fisher_z.npy",
                "regional_biomarkers_normalized.npy",
                "regional_biomarkers_normalized.csv",
                "normalization_metadata.json"]

QC_FIELDS = ["subject_id", "session", "run", "group",
             "malff_valid", "mreho_valid", "dc_z_valid", "fc_fisher_z_valid",
             "input_fc_symmetric", "output_fc_symmetric", "output_fc_diagonal_zero",
             "input_nan_count", "input_inf_count", "output_nan_count", "output_inf_count",
             "dc_fcstrength_max_abs_diff", "dc_fcstrength_max_rel_diff",
             "roi_count", "alff_mean", "reho_mean", "dc_mean", "dc_std",
             "malff_mean", "mreho_mean", "dc_z_mean", "dc_z_std",
             "fc_input_diag_err", "fc_input_sym_err", "fc_output_sym_err",
             "fc_n_offdiag_clipped", "fc_sign_preserved", "fc_z_min", "fc_z_max",
             "status", "failure_reason"]


def log(m):
    print(m, flush=True)


def hdr(t):
    log("\n" + "=" * 72)
    log(t)
    log("=" * 72)


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_lut():
    lut = {}
    if os.path.isfile(ATLAS_LUT):
        with open(ATLAS_LUT) as f:
            for line in f:
                p = line.split()
                if len(p) >= 2 and p[0].isdigit():
                    lut[int(p[0])] = p[1]
    return lut


def guard_write_path(path):
    """Refuse to write anywhere inside a raw biomarkers/ directory."""
    parts = os.path.normpath(path).split(os.sep)
    if "biomarkers" in parts:
        raise RuntimeError(f"REFUSING to write into a raw biomarkers/ path: {path}")
    if not os.path.normpath(path).startswith(os.path.normpath(OUT_ROOT)):
        raise RuntimeError(f"REFUSING to write outside {OUT_ROOT}: {path}")


# ======================================================================
# DISCOVERY
# ======================================================================
def discover():
    """Find acquisitions by their real biomarkers/ directories on disk."""
    found, incomplete = [], []
    for g in GROUPS:
        pat = os.path.join(DERIV, g, "sub-*", "ses-*", "run-*", "biomarkers")
        for d in sorted(glob.glob(pat)):
            rel = os.path.relpath(d, os.path.join(DERIV, g))
            sub, ses, run = rel.split(os.sep)[:3]
            missing = [f for f in INPUT_FILES if not os.path.isfile(os.path.join(d, f))]
            if missing:
                incomplete.append((g, sub, ses, run, f"missing input(s): {','.join(missing)}"))
            else:
                found.append((g, sub, ses, run, d))
    return found, incomplete


def out_dir_for(g, sub, ses, run):
    return os.path.join(OUT_ROOT, g, sub, ses, run)


def validate_existing_outputs(od):
    """Re-validate real files, never trusting the state file alone."""
    if not os.path.isdir(od):
        return False, "no output directory"
    miss = [f for f in OUTPUT_FILES if not os.path.isfile(os.path.join(od, f))]
    if miss:
        return False, f"missing output(s): {','.join(miss[:3])}"
    try:
        for nm in ("malff.npy", "mreho.npy", "dc_z.npy"):
            a = np.load(os.path.join(od, nm))
            if a.shape != (N_ROI,):
                return False, f"{nm} shape {a.shape}"
            if not np.all(np.isfinite(a)):
                return False, f"{nm} non-finite"
        z = np.load(os.path.join(od, "fc_fisher_z.npy"))
        if z.shape != (N_ROI, N_ROI):
            return False, f"fc_fisher_z shape {z.shape}"
        if not np.all(np.isfinite(z)):
            return False, "fc_fisher_z non-finite"
        if not np.all(np.diag(z) == 0.0):
            return False, "fc_fisher_z diagonal not exactly zero"
        rb = np.load(os.path.join(od, "regional_biomarkers_normalized.npy"))
        if rb.shape != (N_ROI, 3):
            return False, f"regional matrix shape {rb.shape}"
        with open(os.path.join(od, "normalization_metadata.json")) as f:
            meta = json.load(f)
        if meta.get("script_version") != SCRIPT_VERSION:
            return False, f"stale script_version {meta.get('script_version')}"
    except Exception as e:
        return False, f"output validation error: {e}"
    return True, ""


# ======================================================================
# NORMALIZATION
# ======================================================================
def fisher_z(fc):
    """
    Signed, unthresholded Fisher r-to-z.

    The diagonal is zeroed BEFORE arctanh so arctanh(1.0)=inf is never
    produced at all (no inf to overwrite, no RuntimeWarning), then set
    explicitly to exactly 0.0. Only off-diagonal values are clipped.
    """
    off = ~np.eye(N_ROI, dtype=bool)
    work = np.zeros_like(fc)
    vals = fc[off]
    n_clipped = int(np.sum(np.abs(vals) > FC_CLIP))
    work[off] = np.clip(vals, -FC_CLIP, FC_CLIP)
    z = np.arctanh(work)
    np.fill_diagonal(z, 0.0)
    return z, n_clipped, off


def normalize_one(g, sub, ses, run, in_dir, od, lut):
    r = {f: "" for f in QC_FIELDS}
    r.update({"subject_id": sub, "session": ses, "run": run, "group": g,
              "roi_count": N_ROI})

    # ---- read raw inputs (read-only) ----
    alff = np.load(os.path.join(in_dir, "alff.npy"))
    reho = np.load(os.path.join(in_dir, "reho.npy"))
    dc = np.load(os.path.join(in_dir, "degree_centrality.npy"))
    fcs = np.load(os.path.join(in_dir, "fc_strength.npy"))
    fc = np.load(os.path.join(in_dir, "fc_matrix.npy"))

    for nm, a, shp in (("alff", alff, (N_ROI,)), ("reho", reho, (N_ROI,)),
                       ("degree_centrality", dc, (N_ROI,)), ("fc_strength", fcs, (N_ROI,)),
                       ("fc_matrix", fc, (N_ROI, N_ROI))):
        if a.shape != shp:
            raise ValueError(f"{nm} shape {a.shape} != {shp}")

    in_nan = int(sum(int(np.isnan(a).sum()) for a in (alff, reho, dc, fcs, fc)))
    in_inf = int(sum(int(np.isinf(a).sum()) for a in (alff, reho, dc, fcs, fc)))
    r["input_nan_count"], r["input_inf_count"] = in_nan, in_inf
    if in_nan or in_inf:
        raise ValueError(f"input contains non-finite values (NaN={in_nan}, Inf={in_inf})")

    # ---- FC_Strength / DC redundancy (verify, then exclude) ----
    expect = dc / FC_STRENGTH_DIVISOR
    abs_d = float(np.max(np.abs(expect - fcs)))
    denom = np.where(np.abs(fcs) > 0, np.abs(fcs), np.nan)
    rel_d = float(np.nanmax(np.abs(expect - fcs) / denom))
    r["dc_fcstrength_max_abs_diff"], r["dc_fcstrength_max_rel_diff"] = abs_d, rel_d

    # ---- A. ALFF -> mALFF ----
    alff_mean = float(np.mean(alff))
    if not np.isfinite(alff_mean) or alff_mean <= 0:
        raise ValueError(f"mean(ALFF)={alff_mean} is not finite and > 0")
    malff = alff / alff_mean

    # ---- B. ReHo -> mReHo ----
    reho_mean = float(np.mean(reho))
    if not np.isfinite(reho_mean) or reho_mean == 0:
        raise ValueError(f"mean(ReHo)={reho_mean} is not finite and non-zero")
    mreho = reho / reho_mean

    # ---- C. DC -> within-subject z ----
    dc_mean = float(np.mean(dc))
    dc_std = float(np.std(dc, ddof=DC_DDOF))
    if not np.isfinite(dc_std) or dc_std <= 0:
        raise ValueError(f"std(DC, ddof={DC_DDOF})={dc_std} is not finite and > 0")
    dc_z = (dc - dc_mean) / dc_std

    # ---- D. FC -> Fisher z ----
    diag_err = float(np.max(np.abs(np.diag(fc) - 1.0)))
    sym_err_in = float(np.max(np.abs(fc - fc.T)))
    if sym_err_in > 1e-6:
        raise ValueError(f"input FC not symmetric (max asymmetry {sym_err_in:.3e})")
    if diag_err > 1e-6:
        raise ValueError(f"input FC diagonal not ~1 (max error {diag_err:.3e})")
    fc_z, n_clipped, off = fisher_z(fc)
    sym_err_out = float(np.max(np.abs(fc_z - fc_z.T)))
    diag_zero = bool(np.all(np.diag(fc_z) == 0.0))
    # arctanh is sign-preserving: every negative off-diagonal r must stay negative
    neg_in = fc[off] < 0
    sign_ok = bool(np.all(fc_z[off][neg_in] < 0)) if neg_in.any() else True

    # ---- output validation ----
    out_nan = int(sum(int(np.isnan(a).sum()) for a in (malff, mreho, dc_z, fc_z)))
    out_inf = int(sum(int(np.isinf(a).sum()) for a in (malff, mreho, dc_z, fc_z)))
    if out_nan or out_inf:
        raise ValueError(f"normalized output non-finite (NaN={out_nan}, Inf={out_inf})")

    malff_mean, mreho_mean = float(np.mean(malff)), float(np.mean(mreho))
    dc_z_mean, dc_z_std = float(np.mean(dc_z)), float(np.std(dc_z, ddof=DC_DDOF))
    if abs(malff_mean - 1.0) > 1e-9:
        raise ValueError(f"mean(mALFF)={malff_mean} != 1")
    if abs(mreho_mean - 1.0) > 1e-9:
        raise ValueError(f"mean(mReHo)={mreho_mean} != 1")
    if abs(dc_z_mean) > 1e-9:
        raise ValueError(f"mean(DC_z)={dc_z_mean} != 0")
    if abs(dc_z_std - 1.0) > 1e-9:
        raise ValueError(f"std(DC_z)={dc_z_std} != 1")
    if not diag_zero:
        raise ValueError("Fisher-z FC diagonal is not exactly zero")
    if sym_err_out > 1e-9:
        raise ValueError(f"Fisher-z FC not symmetric ({sym_err_out:.3e})")
    if not sign_ok:
        raise ValueError("Fisher transform did not preserve negative correlations")

    r.update({"malff_valid": True, "mreho_valid": True, "dc_z_valid": True,
              "fc_fisher_z_valid": True,
              "input_fc_symmetric": True, "output_fc_symmetric": True,
              "output_fc_diagonal_zero": diag_zero,
              "output_nan_count": out_nan, "output_inf_count": out_inf,
              "alff_mean": alff_mean, "reho_mean": reho_mean,
              "dc_mean": dc_mean, "dc_std": dc_std,
              "malff_mean": malff_mean, "mreho_mean": mreho_mean,
              "dc_z_mean": dc_z_mean, "dc_z_std": dc_z_std,
              "fc_input_diag_err": diag_err, "fc_input_sym_err": sym_err_in,
              "fc_output_sym_err": sym_err_out, "fc_n_offdiag_clipped": n_clipped,
              "fc_sign_preserved": sign_ok,
              "fc_z_min": float(fc_z.min()), "fc_z_max": float(fc_z.max())})

    # ---- write (guarded) ----
    guard_write_path(od)
    os.makedirs(od, exist_ok=True)

    def wnpy(name, arr):
        p = os.path.join(od, name)
        guard_write_path(p)
        np.save(p, arr)

    def wcsv(name, col, arr):
        p = os.path.join(od, name)
        guard_write_path(p)
        with open(p, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["ROI_ID", "ROI_Name", col])
            for rid in range(1, N_ROI + 1):
                w.writerow([rid, lut.get(rid, f"ROI_{rid}"), arr[rid - 1]])

    wnpy("malff.npy", malff)
    wcsv("malff.csv", "mALFF", malff)
    wnpy("mreho.npy", mreho)
    wcsv("mreho.csv", "mReHo", mreho)
    wnpy("dc_z.npy", dc_z)
    wcsv("dc_z.csv", "DC_z", dc_z)
    wnpy("fc_fisher_z.npy", fc_z)

    reg = np.column_stack([malff, mreho, dc_z])          # (246, 3)
    wnpy("regional_biomarkers_normalized.npy", reg)
    reg_p = os.path.join(od, "regional_biomarkers_normalized.csv")
    guard_write_path(reg_p)
    pd.DataFrame({"ROI_ID": np.arange(1, N_ROI + 1), "mALFF": malff,
                  "mReHo": mreho, "DC_z": dc_z}).to_csv(reg_p, index=False)

    meta = {
        "subject_id": sub, "session": ses, "run": run, "group": g,
        "roi_count": N_ROI, "roi_ids": "1-246 (Brainnetome 246, original order preserved)",
        "source_biomarkers_dir": in_dir,
        "source_files_modified": False,
        "ALFF_mean": alff_mean, "ReHo_mean": reho_mean,
        "DC_mean": dc_mean, "DC_std": dc_std, "DC_ddof": DC_DDOF,
        "FC_clipping_limit": FC_CLIP,
        "FC_clipping_applied_to": "off-diagonal elements only",
        "FC_n_offdiag_clipped": n_clipped,
        "FC_diagonal_handling": ("diagonal zeroed before arctanh so arctanh(1)=inf is "
                                 "never produced, then set to exactly 0.0"),
        "FC_Strength_excluded": True,
        "FC_Strength_exclusion_reason": (
            f"FC_Strength == DC / {int(FC_STRENGTH_DIVISOR)} exactly; perfectly collinear "
            f"with DC (Pearson r = 1.0), carries no independent information. Verified this "
            f"acquisition: max|DC/{int(FC_STRENGTH_DIVISOR)} - FC_Strength| = {abs_d:.3e}. "
            f"Raw fc_strength.npy/.csv left untouched."),
        "normalization_methods": {
            "mALFF": "ALFF_i / mean(ALFF) over this acquisition's own 246 ROIs (ratio, not z-score)",
            "mReHo": "ReHo_i / mean(ReHo) over this acquisition's own 246 ROIs",
            "DC_z": f"(DC_i - mean(DC)) / std(DC, ddof={DC_DDOF}) within this acquisition",
            "FC_fisher_z": ("arctanh(clip(r, -0.999999, 0.999999)) on off-diagonal only; "
                            "diagonal exactly 0; signed, unthresholded, negatives retained; "
                            "NOT z-scored within subject"),
        },
        "cross_subject_normalization_applied": False,
        "cross_subject_note": ("none applied: this stage is within-subject only and therefore "
                               "leakage-free. Cross-subject scaling must be fitted on training "
                               "subjects only, after a subject-level split."),
        "labels_used": False,
        "model_feature_set": ["mALFF (246)", "mReHo (246)", "DC_z (246)", "Fisher-z FC (246x246)"],
        "script": os.path.basename(__file__), "script_version": SCRIPT_VERSION,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__},
        "timestamp_utc": utcnow(),
    }
    meta_p = os.path.join(od, "normalization_metadata.json")
    guard_write_path(meta_p)
    with open(meta_p, "w") as f:
        json.dump(meta, f, indent=2)

    r["status"] = "COMPLETED"
    return r


# ======================================================================
# STATE
# ======================================================================
def load_state():
    if os.path.isfile(STATE_JSON):
        try:
            with open(STATE_JSON) as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_state(state):
    os.makedirs(OUT_ROOT, exist_ok=True)
    tmp = STATE_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, STATE_JSON)


def write_qc_csv(rows):
    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(QC_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=QC_FIELDS)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in QC_FIELDS})


# ======================================================================
# REPORT
# ======================================================================
def write_report(rows, incomplete, discrepancies):
    df = pd.DataFrame(rows)
    ok = df[df["status"] == "COMPLETED"] if len(df) else df
    fails = df[df["status"] == "FAILED"] if len(df) else df
    L = ["# Within-Subject Biomarker Normalization -- QC Report", "",
         f"Generated (UTC): {utcnow()}", f"Script: `normalize_within_subject.py` v{SCRIPT_VERSION}", "",
         "## 1-4. Completion", "",
         f"- Total acquisitions discovered: **{len(df)}**",
         f"- Successfully normalized: **{len(ok)}**",
         f"- Failed: **{len(fails)}**",
         f"- Acquisitions with incomplete raw inputs (not processed): **{len(incomplete)}**", ""]
    if len(fails):
        L += ["### Failures", "", "| Acquisition | Reason |", "|---|---|"]
        for _, f_ in fails.iterrows():
            L.append(f"| {f_['group']}/{f_['subject_id']}/{f_['session']}/{f_['run']} | "
                     f"{str(f_['failure_reason'])[:200]} |")
        L.append("")
    else:
        L += ["No failures.", ""]
    if incomplete:
        L += ["### Incomplete raw inputs", "", "| Acquisition | Reason |", "|---|---|"]
        for (g, s, se, rn, why) in incomplete:
            L.append(f"| {g}/{s}/{se}/{rn} | {why} |")
        L.append("")

    L += ["## 5. Counts by disease group", "",
          "| Group | Expected | Discovered | Normalized | Failed |", "|---|---:|---:|---:|---:|"]
    for g in GROUPS:
        sub = df[df["group"] == g] if len(df) else df
        n_disc = len(sub)
        n_ok = int((sub["status"] == "COMPLETED").sum()) if n_disc else 0
        n_bad = int((sub["status"] == "FAILED").sum()) if n_disc else 0
        L.append(f"| {g} | {EXPECTED_PER_GROUP[g]} | {n_disc} | {n_ok} | {n_bad} |")
    L.append(f"| **TOTAL** | **{EXPECTED_TOTAL}** | **{len(df)}** | **{len(ok)}** | **{len(fails)}** |")
    L.append("")

    L += ["## 10. Dataset completeness check", ""]
    if discrepancies:
        L.append("**DISCREPANCY against the expected counts -- reported, not corrected:**")
        L.append("")
        for d in discrepancies:
            L.append(f"- {d}")
    else:
        L.append("Filesystem agrees with the expected counts "
                 f"(165 total; {', '.join(f'{g} {EXPECTED_PER_GROUP[g]}' for g in GROUPS)}).")
    L.append("")

    if len(ok):
        def st(col, fmt="{:.6f}"):
            v = ok[col].astype(float)
            return (fmt.format(v.min()), fmt.format(v.median()), fmt.format(v.max()))

        L += ["## 6-7. Shape / NaN / Inf validation", "",
              f"- All inputs shape-validated (ALFF/ReHo/DC/FC_Strength = (246,), FC = (246,246)): "
              f"**{len(ok)}/{len(ok)}**",
              f"- Input NaN total: **{int(ok['input_nan_count'].astype(int).sum())}**",
              f"- Input Inf total: **{int(ok['input_inf_count'].astype(int).sum())}**",
              f"- Output NaN total: **{int(ok['output_nan_count'].astype(int).sum())}**",
              f"- Output Inf total: **{int(ok['output_inf_count'].astype(int).sum())}**", "",
              "## 8. mALFF statistics", "",
              f"- Raw mean(ALFF) across acquisitions (the scaling removed): "
              f"min {float(ok['alff_mean'].astype(float).min()):.2f}, "
              f"median {float(ok['alff_mean'].astype(float).median()):.2f}, "
              f"max {float(ok['alff_mean'].astype(float).max()):.2f}",
              f"- mean(mALFF) per acquisition: min/median/max = {' / '.join(st('malff_mean'))} "
              f"(target exactly 1)", "",
              "## 9. mReHo statistics", "",
              f"- Raw mean(ReHo): min {float(ok['reho_mean'].astype(float).min()):.6f}, "
              f"median {float(ok['reho_mean'].astype(float).median()):.6f}, "
              f"max {float(ok['reho_mean'].astype(float).max()):.6f}",
              f"- mean(mReHo) per acquisition: min/median/max = {' / '.join(st('mreho_mean'))} "
              f"(target exactly 1)", "",
              "## 10. DC-z statistics", "",
              f"- Raw mean(DC): min {float(ok['dc_mean'].astype(float).min()):.4f}, "
              f"max {float(ok['dc_mean'].astype(float).max()):.4f}",
              f"- Raw std(DC) (ddof={DC_DDOF}): min {float(ok['dc_std'].astype(float).min()):.4f}, "
              f"max {float(ok['dc_std'].astype(float).max()):.4f}",
              f"- mean(DC_z): min/median/max = {' / '.join(st('dc_z_mean', '{:.2e}'))} (target 0)",
              f"- std(DC_z): min/median/max = {' / '.join(st('dc_z_std'))} (target 1)", "",
              "## 11. Fisher-z FC statistics", "",
              f"- Fisher-z range across dataset: "
              f"{float(ok['fc_z_min'].astype(float).min()):.4f} to "
              f"{float(ok['fc_z_max'].astype(float).max()):.4f}",
              f"- Off-diagonal elements clipped at +-{FC_CLIP} (total across dataset): "
              f"**{int(ok['fc_n_offdiag_clipped'].astype(int).sum())}**",
              f"- Negative correlations preserved as negative: "
              f"**{int(ok['fc_sign_preserved'].astype(bool).sum())}/{len(ok)}**",
              "- No thresholding, no absolute value, no within-subject z-scoring applied to FC.", "",
              "## 12-13. FC symmetry / diagonal checks", "",
              f"- Input FC symmetric (<=1e-6): **{int(ok['input_fc_symmetric'].astype(bool).sum())}/{len(ok)}**, "
              f"max asymmetry {float(ok['fc_input_sym_err'].astype(float).max()):.3e}",
              f"- Input FC diagonal ~1: max error {float(ok['fc_input_diag_err'].astype(float).max()):.3e}",
              f"- Output Fisher-z symmetric: **{int(ok['output_fc_symmetric'].astype(bool).sum())}/{len(ok)}**, "
              f"max asymmetry {float(ok['fc_output_sym_err'].astype(float).max()):.3e}",
              f"- Output diagonal exactly 0: "
              f"**{int(ok['output_fc_diagonal_zero'].astype(bool).sum())}/{len(ok)}**", "",
              "## 14. FC_Strength / DC redundancy verification", "",
              f"Verified per acquisition that `FC_Strength == DC / {int(FC_STRENGTH_DIVISOR)}`:", "",
              f"- Max absolute difference across dataset: "
              f"**{float(ok['dc_fcstrength_max_abs_diff'].astype(float).max()):.3e}**",
              f"- Max relative difference across dataset: "
              f"**{float(ok['dc_fcstrength_max_rel_diff'].astype(float).max()):.3e}**", "",
              "FC_Strength is therefore perfectly collinear with DC (Pearson r = 1.0) and carries "
              "no independent information. It is **excluded from the model feature set**. The raw "
              "`fc_strength.npy` / `fc_strength.csv` files were left untouched.", "",
              "## 15. Output completeness", "",
              f"- Acquisitions with all {len(OUTPUT_FILES)} normalized outputs present and "
              f"re-validated: **{len(ok)}/{len(df)}**",
              "- Per acquisition: `malff.npy/.csv`, `mreho.npy/.csv`, `dc_z.npy/.csv`, "
              "`fc_fisher_z.npy`, `regional_biomarkers_normalized.npy` (246x3) + `.csv`, "
              "`normalization_metadata.json`", "",
              "## 16. Warnings", ""]
        warns = []
        if int(ok["fc_n_offdiag_clipped"].astype(int).sum()) > 0:
            warns.append("Some off-diagonal correlations required clipping at "
                         f"+-{FC_CLIP} before arctanh (near-perfect ROI pairs).")
        if len(fails):
            warns.append(f"{len(fails)} acquisition(s) failed; see the Failures table.")
        if incomplete:
            warns.append(f"{len(incomplete)} acquisition(s) had incomplete raw inputs.")
        L += [f"- {w}" for w in warns] if warns else ["None."]
        L += ["", "## Final model feature set after this stage", "",
              "| Kind | Feature | Shape |", "|---|---|---|",
              "| Regional | mALFF | (246,) |",
              "| Regional | mReHo | (246,) |",
              "| Regional | DC_z | (246,) |",
              "| Connectivity | Fisher-z FC | (246, 246) |",
              "| *Excluded* | *FC_Strength (= DC/245)* | *-* |", "",
              "No cross-subject normalization, PCA, feature selection, FC flattening, "
              "thresholding or label use was performed at this stage."]

    os.makedirs(OUT_ROOT, exist_ok=True)
    with open(QC_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return L


def check_discrepancies(found):
    per = {}
    for (g, _, _, _, _) in found:
        per[g] = per.get(g, 0) + 1
    d = []
    if len(found) != EXPECTED_TOTAL:
        d.append(f"total discovered {len(found)} != expected {EXPECTED_TOTAL}")
    for g in GROUPS:
        if per.get(g, 0) != EXPECTED_PER_GROUP[g]:
            d.append(f"{g}: discovered {per.get(g,0)} != expected {EXPECTED_PER_GROUP[g]}")
    return d


# ======================================================================
# MODES
# ======================================================================
def mode_run():
    hdr("WITHIN-SUBJECT BIOMARKER NORMALIZATION")
    log(f"source (read-only): {DERIV}/<GROUP>/sub-*/ses-*/run-*/biomarkers/")
    log(f"output:             {OUT_ROOT}/<GROUP>/sub-*/ses-*/run-*/")
    lut = load_lut()
    found, incomplete = discover()
    disc = check_discrepancies(found)
    log(f"\ndiscovered {len(found)} acquisitions with complete raw inputs; "
        f"{len(incomplete)} incomplete")
    if disc:
        log("DISCREPANCY vs expected counts (reported, not corrected):")
        for x in disc:
            log(f"  - {x}")
    else:
        log("filesystem agrees with expected counts (165)")

    state = load_state()
    rows, done, skipped, failed = [], 0, 0, 0
    for i, (g, sub, ses, run, in_dir) in enumerate(found, 1):
        key = f"{g}/{sub}/{ses}/{run}"
        od = out_dir_for(g, sub, ses, run)
        valid, why = validate_existing_outputs(od)
        if valid and key in state and state[key].get("status") == "COMPLETED":
            rows.append(state[key])
            skipped += 1
            if i % 40 == 0:
                log(f"[{i}/{len(found)}] ... {skipped} already valid")
            continue
        try:
            r = normalize_one(g, sub, ses, run, in_dir, od, lut)
            state[key] = r
            rows.append(r)
            done += 1
        except Exception as e:
            r = {f: "" for f in QC_FIELDS}
            r.update({"subject_id": sub, "session": ses, "run": run, "group": g,
                      "roi_count": N_ROI, "status": "FAILED",
                      "failure_reason": f"{type(e).__name__}: {e}"})
            state[key] = r
            rows.append(r)
            failed += 1
            log(f"[{i}/{len(found)}] {key}  FAILED: {type(e).__name__}: {e}")
            log(traceback.format_exc(limit=3))
        if i % 20 == 0 or i == len(found):
            save_state(state)
    save_state(state)
    for (g, s, se, rn, why) in incomplete:
        r = {f: "" for f in QC_FIELDS}
        r.update({"subject_id": s, "session": se, "run": rn, "group": g,
                  "status": "FAILED", "failure_reason": why})
        rows.append(r)

    write_qc_csv(rows)
    write_report(rows, incomplete, disc)
    log(f"\nnormalized={done}  already_valid={skipped}  failed={failed}  total={len(found)}")
    log(f"saved: {QC_CSV}")
    log(f"saved: {QC_MD}")
    log("NORMALIZATION_DONE")


def mode_audit():
    hdr("NORMALIZATION AUDIT (from disk)")
    found, incomplete = discover()
    disc = check_discrepancies(found)
    state = load_state()
    rows = []
    for (g, sub, ses, run, in_dir) in found:
        key = f"{g}/{sub}/{ses}/{run}"
        od = out_dir_for(g, sub, ses, run)
        valid, why = validate_existing_outputs(od)
        r = state.get(key)
        if r is None:
            r = {f: "" for f in QC_FIELDS}
            r.update({"subject_id": sub, "session": ses, "run": run, "group": g,
                      "status": "FAILED", "failure_reason": "no state record"})
        if not valid:
            r = dict(r)
            r["status"] = "FAILED"
            r["failure_reason"] = f"output re-validation failed: {why}"
        rows.append(r)
    write_qc_csv(rows)
    L = write_report(rows, incomplete, disc)
    log("\n".join(L))
    log("AUDIT_DONE")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    os.makedirs(OUT_ROOT, exist_ok=True)
    if mode == "run":
        mode_run()
    elif mode == "audit":
        mode_audit()
    else:
        log(f"unknown mode: {mode}")
        sys.exit(2)
