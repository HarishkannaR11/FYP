"""
READ-ONLY FINAL 4-GROUP AUDIT (AD, CN_Final, EMCI, LMCI).

This script performs NO writes, deletes, or modifications to:
  - BIDS/, NIfTI_/, raw_data/  (original source data)
  - derivatives/fsfast/<GROUP>/**  (existing preprocessing outputs)

It only READS existing derivatives and WRITES new report files under:
  derivatives/fsfast/FINAL_4GROUP_AUDIT/
"""
import csv
import hashlib
import json
import os
import re
from collections import Counter, defaultdict

import nibabel as nib
import numpy as np

FSFAST_ROOT = "/mnt/c/Users/krish/FYP/derivatives/fsfast"
OUT_DIR = os.path.join(FSFAST_ROOT, "FINAL_4GROUP_AUDIT")
BIDS_ROOT = "/mnt/c/Users/krish/FYP/BIDS"

GROUPS = ["AD", "CN_Final", "EMCI", "LMCI"]
EXPECTED_SOURCE = {"AD": 25, "CN_Final": 30, "EMCI": 30, "LMCI": 23}

EXPECTED_DESIGN_COLS = 9
DESIGN_COL_NAMES = [
    "intercept", "linear_trend", "quadratic_trend",
    "roll", "pitch", "yaw", "dS", "dL", "dP",
]

REQUIRED_QC_METRICS = [
    "mean_FD", "median_FD", "max_FD",
    "mean_tSNR", "median_tSNR",
    "SNR",
    "CNR",
    "mean_DVARS", "median_DVARS", "max_DVARS",
    "GCOR_pre_GLM", "GCOR_post_GLM",
    "spatial_entropy_bits", "temporal_entropy_bits",
]

os.makedirs(OUT_DIR, exist_ok=True)


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def discover_acquisitions(group):
    """Find every sub-*/ses-*/run-* directory under a group root."""
    group_root = os.path.join(FSFAST_ROOT, group)
    acqs = []
    if not os.path.isdir(group_root):
        return acqs
    for sub in sorted(os.listdir(group_root)):
        sub_path = os.path.join(group_root, sub)
        if not os.path.isdir(sub_path) or not sub.startswith("sub-"):
            continue
        for ses in sorted(os.listdir(sub_path)):
            ses_path = os.path.join(sub_path, ses)
            if not os.path.isdir(ses_path) or not ses.startswith("ses-"):
                continue
            for run in sorted(os.listdir(ses_path)):
                run_path = os.path.join(ses_path, run)
                if not os.path.isdir(run_path) or not run.startswith("run-"):
                    continue
                acqs.append({"group": group, "sub": sub, "ses": ses, "run": run, "path": run_path})
    return acqs


def find_expected_source_acquisitions(group):
    """Independently list what BIDS says should exist for this group (ground truth)."""
    participants_tsv = os.path.join(BIDS_ROOT, "participants.tsv")
    with open(participants_tsv, newline="", encoding="utf-8-sig") as f:
        content = f.read().replace("\r", "")
    rows = list(csv.DictReader(content.splitlines(), delimiter="\t"))
    subs = sorted([r["participant_id"] for r in rows if r["group"] == group])
    import glob
    found = []
    for sub in subs:
        for bf in sorted(glob.glob(os.path.join(BIDS_ROOT, sub, "ses-*", "func", "*task-rest*bold.nii*"))):
            ses = [p for p in bf.split(os.sep) if p.startswith("ses-")][0]
            m = re.search(r"(run-\d+)", os.path.basename(bf))
            run = m.group(1) if m else "run-01"
            found.append((sub, ses, run))
    return subs, found


# ---------------------------------------------------------------------------
# Per-acquisition validators
# ---------------------------------------------------------------------------

def validate_nifti(acq):
    row = {"group": acq["group"], "sub": acq["sub"], "ses": acq["ses"], "run": acq["run"]}
    bold_glob = [f for f in os.listdir(acq["path"]) if f.endswith("desc-preproc_bold.nii.gz")]
    if not bold_glob:
        row.update({"status": "FAIL", "reason": "desc-preproc_bold.nii.gz MISSING"})
        return row, None
    bold_path = os.path.join(acq["path"], bold_glob[0])
    row["filename"] = bold_glob[0]
    try:
        img = nib.load(bold_path)
        data = img.get_fdata()
    except Exception as e:
        row.update({"status": "FAIL", "reason": f"cannot load/read: {e}"})
        return row, None

    shape = img.shape
    zooms = img.header.get_zooms()
    is_4d = len(shape) == 4
    n_vols = shape[3] if is_4d else None
    tr = float(zooms[3]) if len(zooms) > 3 else None
    has_nan = bool(np.isnan(data).any())
    has_inf = bool(np.isinf(data).any())
    nonempty = bool(np.any(data != 0))

    row.update({
        "dims": str(shape),
        "voxel_size": str(tuple(round(float(z), 4) for z in zooms[:3])),
        "TR": tr,
        "n_volumes": n_vols,
        "datatype": str(img.get_data_dtype()),
        "affine_det": float(round(np.linalg.det(img.affine), 6)),
        "orientation": "".join(nib.orientations.aff2axcodes(img.affine)),
        "has_nan": has_nan,
        "has_inf": has_inf,
        "nonempty": nonempty,
        "is_4d": is_4d,
    })

    reasons = []
    if not is_4d:
        reasons.append("not 4D")
    if has_nan:
        reasons.append("contains NaN")
    if has_inf:
        reasons.append("contains Inf")
    if not nonempty:
        reasons.append("empty (all zero) image")
    if n_vols is not None and n_vols < 10:
        reasons.append(f"suspiciously low volume count ({n_vols})")

    row["status"] = "FAIL" if reasons else "PASS"
    row["reason"] = "; ".join(reasons) if reasons else "OK"
    return row, (img, data, bold_path)


def validate_mask(acq, bold_img):
    row = {"group": acq["group"], "sub": acq["sub"], "ses": acq["ses"], "run": acq["run"]}
    mask_path = os.path.join(acq["path"], "brain_mask.nii.gz")
    if not os.path.exists(mask_path):
        row.update({"status": "FAIL", "reason": "brain_mask.nii.gz MISSING"})
        return row, None
    try:
        mimg = nib.load(mask_path)
        mdata = mimg.get_fdata()
    except Exception as e:
        row.update({"status": "FAIL", "reason": f"cannot load: {e}"})
        return row, None

    reasons = []
    is_3d = mdata.ndim == 3
    if not is_3d:
        reasons.append("not 3D")
    uniq = sorted(np.unique(mdata).tolist())
    is_binary = all(v in (0, 1) for v in uniq)
    if not is_binary:
        reasons.append(f"not strictly binary, unique values={uniq[:10]}")
    mask_voxels = int(np.sum(mdata > 0))
    total_voxels = int(np.prod(mdata.shape))
    if mask_voxels == 0:
        reasons.append("mask empty (0 nonzero voxels)")

    dims_match = None
    affine_match = None
    voxsize_match = None
    if bold_img is not None:
        dims_match = tuple(mdata.shape) == tuple(bold_img.shape[:3])
        affine_match = bool(np.allclose(mimg.affine, bold_img.affine, atol=1e-3))
        voxsize_match = tuple(round(float(z), 4) for z in mimg.header.get_zooms()[:3]) == \
                         tuple(round(float(z), 4) for z in bold_img.header.get_zooms()[:3])
        if not dims_match:
            reasons.append("dims != BOLD dims")
        if not affine_match:
            reasons.append("affine != BOLD affine")
        if not voxsize_match:
            reasons.append("voxel size != BOLD voxel size")

    row.update({
        "is_3d": is_3d, "is_binary": is_binary,
        "mask_voxel_count": mask_voxels, "total_voxel_count": total_voxels,
        "mask_pct": round(100.0 * mask_voxels / total_voxels, 4) if total_voxels else None,
        "dims_match_bold": dims_match, "affine_match_bold": affine_match, "voxsize_match_bold": voxsize_match,
    })
    row["status"] = "FAIL" if reasons else "PASS"
    row["reason"] = "; ".join(reasons) if reasons else "OK"
    return row, mdata


def validate_zero_outside_mask(bold_data, mask_data):
    """Check the final (GLM residual) BOLD is ~zero outside the mask, per-voxel across time."""
    if bold_data is None or mask_data is None:
        return None, "not checked (missing inputs)"
    if bold_data.shape[:3] != mask_data.shape:
        return None, "shape mismatch, cannot check"
    outside = mask_data == 0
    outside_vals = bold_data[outside, :]
    if outside_vals.size == 0:
        return None, "mask covers entire FOV"
    max_abs_outside = float(np.max(np.abs(outside_vals)))
    is_zero = max_abs_outside < 1e-6
    return is_zero, f"max|value| outside mask = {max_abs_outside:.6g}"


def validate_motion(acq, n_vols):
    row = {"group": acq["group"], "sub": acq["sub"], "ses": acq["ses"], "run": acq["run"]}
    files = [f for f in os.listdir(acq["path"]) if f.endswith("motion_parameters.mcdat")]
    if not files:
        row.update({"status": "FAIL", "reason": "*_motion_parameters.mcdat MISSING"})
        return row, None
    mpath = os.path.join(acq["path"], files[0])
    row["filename"] = files[0]
    try:
        arr = np.loadtxt(mpath)
    except Exception as e:
        row.update({"status": "FAIL", "reason": f"cannot parse: {e}"})
        return row, None
    n_rows = arr.shape[0] if arr.ndim > 0 else 0
    n_cols = arr.shape[1] if arr.ndim > 1 else 1
    reasons = []
    if n_cols != 10:
        reasons.append(f"expected 10 mc-afni2 columns, got {n_cols}")
    if n_vols is not None and n_rows != n_vols:
        reasons.append(f"row count ({n_rows}) != BOLD volume count ({n_vols})")
    row.update({"n_rows": n_rows, "n_cols": n_cols, "n_bold_volumes": n_vols})
    row["status"] = "FAIL" if reasons else "PASS"
    row["reason"] = "; ".join(reasons) if reasons else "OK"
    return row, arr


def validate_design_matrix(acq, n_vols, motion_arr):
    row = {"group": acq["group"], "sub": acq["sub"], "ses": acq["ses"], "run": acq["run"]}
    dpath = os.path.join(acq["path"], "design_matrix.txt")
    if not os.path.exists(dpath):
        row.update({"status": "FAIL", "reason": "design_matrix.txt MISSING"})
        return row
    try:
        arr = np.loadtxt(dpath)
    except Exception as e:
        row.update({"status": "FAIL", "reason": f"cannot parse: {e}"})
        return row
    n_rows = arr.shape[0] if arr.ndim > 0 else 0
    n_cols = arr.shape[1] if arr.ndim > 1 else 1
    reasons = []
    if n_cols != EXPECTED_DESIGN_COLS:
        reasons.append(f"expected {EXPECTED_DESIGN_COLS} columns, got {n_cols}")
    if n_vols is not None and n_rows != n_vols:
        reasons.append(f"row count ({n_rows}) != BOLD volume count ({n_vols})")

    motion_match = None
    if motion_arr is not None and n_cols >= 9 and motion_arr.ndim == 2 and motion_arr.shape[1] >= 6:
        try:
            # mcdat columns (0-indexed): 1=roll,2=pitch,3=yaw,4=dS,5=dL,6=dP
            design_motion = arr[:, 3:9]
            mcdat_motion = motion_arr[:, 1:7]
            if design_motion.shape == mcdat_motion.shape:
                motion_match = bool(np.allclose(design_motion, mcdat_motion, atol=1e-4, equal_nan=True))
                if not motion_match:
                    reasons.append("design motion columns do not match .mcdat values")
        except Exception as e:
            reasons.append(f"motion comparison error: {e}")

    row.update({"n_rows": n_rows, "n_cols": n_cols, "motion_columns_match_mcdat": motion_match})
    row["status"] = "FAIL" if reasons else "PASS"
    row["reason"] = "; ".join(reasons) if reasons else "OK"
    return row


def validate_provenance(acq):
    row = {"group": acq["group"], "sub": acq["sub"], "ses": acq["ses"], "run": acq["run"]}
    files = [f for f in os.listdir(acq["path"]) if f.endswith("desc-preproc_bold.json")]
    if not files:
        row.update({"status": "FAIL", "reason": "desc-preproc_bold.json MISSING"})
        return row
    jpath = os.path.join(acq["path"], files[0])
    try:
        with open(jpath, encoding="utf-8") as f:
            j = json.load(f)
    except Exception as e:
        row.update({"status": "FAIL", "reason": f"cannot parse JSON: {e}"})
        return row

    steps = {s.get("step"): s for s in j.get("ProcessingSteps", [])}
    reasons = []

    moco = steps.get("motion_correction", {})
    if moco.get("tool") != "mc-afni2":
        reasons.append(f"motion_correction tool != mc-afni2 (got {moco.get('tool')})")

    stc = steps.get("slice_timing_correction", {})
    if stc.get("status") != "SKIPPED":
        reasons.append(f"slice_timing_correction status != SKIPPED (got {stc.get('status')})")

    smooth = steps.get("spatial_smoothing", {})
    if smooth.get("requested_fwhm_mm") != 6.0:
        reasons.append(f"requested_fwhm_mm != 6.0 (got {smooth.get('requested_fwhm_mm')})")

    glm = steps.get("temporal_filtering_and_nuisance_regression", {})
    if glm.get("tool") != "mri_glmfit":
        reasons.append(f"GLM tool != mri_glmfit (got {glm.get('tool')})")
    if glm.get("masked") is not True:
        reasons.append("GLM masked flag != True (mask not confirmed used)")
    if glm.get("design_matrix_columns") != 9:
        reasons.append(f"design_matrix_columns != 9 (got {glm.get('design_matrix_columns')})")

    qc = j.get("QCMetrics", {})
    missing_metrics = [m for m in REQUIRED_QC_METRICS if m not in qc]
    if missing_metrics:
        reasons.append(f"QCMetrics missing keys: {missing_metrics}")

    row.update({
        "motion_tool": moco.get("tool"),
        "slice_timing_status": stc.get("status"),
        "requested_fwhm_mm": smooth.get("requested_fwhm_mm"),
        "glm_tool": glm.get("tool"),
        "glm_masked": glm.get("masked"),
        "design_cols": glm.get("design_matrix_columns"),
        "pipeline_string": j.get("PreprocessingPipeline"),
    })
    row["status"] = "FAIL" if reasons else "PASS"
    row["reason"] = "; ".join(reasons) if reasons else "OK"
    return row, j


def validate_qc_completeness(acq):
    row = {"group": acq["group"], "sub": acq["sub"], "ses": acq["ses"], "run": acq["run"]}
    qc_dir = os.path.join(acq["path"], "QC")
    reasons = []
    if not os.path.isdir(qc_dir):
        row.update({"status": "FAIL", "reason": "QC/ directory MISSING"})
        return row
    required = ["qc_metrics.csv", "qc_metrics.txt", "qc_report.md"]
    optional = ["fd_values.csv", "dvars_values.csv"]
    present = set(os.listdir(qc_dir))
    for r in required:
        if r not in present:
            reasons.append(f"missing required QC file: {r}")
    for o in optional:
        if o not in present:
            reasons.append(f"missing optional QC file: {o}")
    png_count = len([f for f in present if f.endswith(".png")])
    row.update({"required_present": len(required) - sum(1 for r in required if r not in present),
                "png_count": png_count, "all_files": sorted(present)})

    # verify identifiers inside qc_report.md match this acquisition (cross-subject mixup check)
    report_path = os.path.join(qc_dir, "qc_report.md")
    if os.path.exists(report_path):
        try:
            with open(report_path, encoding="utf-8") as f:
                content = f.read()
            if acq["sub"] not in content or acq["ses"] not in content:
                reasons.append("qc_report.md does not reference this sub/ses (possible mixup)")
        except Exception as e:
            reasons.append(f"cannot read qc_report.md: {e}")

    row["status"] = "FAIL" if any("missing required" in r or "mixup" in r for r in reasons) else \
                    ("WARN" if reasons else "PASS")
    row["reason"] = "; ".join(reasons) if reasons else "OK"
    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    nifti_rows, mask_rows, motion_rows, design_rows, prov_rows, qc_rows = [], [], [], [], [], []
    inventory_rows = []
    duplicate_hashes = defaultdict(list)
    per_group_acq = {}
    per_group_source = {}

    for group in GROUPS:
        acqs = discover_acquisitions(group)
        per_group_acq[group] = acqs
        subs, source_acqs = find_expected_source_acquisitions(group)
        per_group_source[group] = (subs, source_acqs)

        for acq in acqs:
            n_row, bold_bundle = validate_nifti(acq)
            nifti_rows.append(n_row)
            bold_img, bold_data, bold_path = (None, None, None) if bold_bundle is None else bold_bundle
            n_vols = bold_img.shape[3] if (bold_img is not None and len(bold_img.shape) == 4) else None

            m_row, mask_data = validate_mask(acq, bold_img)
            zero_outside, zero_detail = validate_zero_outside_mask(bold_data, mask_data)
            m_row["final_bold_zero_outside_mask"] = zero_outside
            m_row["zero_outside_detail"] = zero_detail
            mask_rows.append(m_row)

            mo_row, motion_arr = validate_motion(acq, n_vols)
            motion_rows.append(mo_row)

            d_row = validate_design_matrix(acq, n_vols, motion_arr)
            design_rows.append(d_row)

            p_result = validate_provenance(acq)
            p_row = p_result[0] if isinstance(p_result, tuple) else p_result
            prov_rows.append(p_row)

            q_row = validate_qc_completeness(acq)
            qc_rows.append(q_row)

            overall_reasons = []
            for r in (n_row, m_row, mo_row, d_row, p_row, q_row):
                if r.get("status") == "FAIL":
                    overall_reasons.append(r["reason"])
            overall_status = "FAIL" if overall_reasons else (
                "WARN" if q_row.get("status") == "WARN" else "PASS")

            inventory_rows.append({
                "group": group, "sub": acq["sub"], "ses": acq["ses"], "run": acq["run"],
                "path": acq["path"], "status": overall_status,
                "reasons": " | ".join(overall_reasons) if overall_reasons else "",
            })

            if bold_path and os.path.exists(bold_path):
                try:
                    h = md5sum(bold_path)
                    duplicate_hashes[h].append(f"{group}/{acq['sub']}/{acq['ses']}/{acq['run']}")
                except Exception:
                    pass

    # ---- duplicates ----
    dup_rows = []
    for h, members in duplicate_hashes.items():
        if len(members) > 1:
            dup_rows.append({"md5": h, "n_copies": len(members), "members": " | ".join(members)})
    seen_ids = Counter(f"{r['group']}/{r['sub']}/{r['ses']}/{r['run']}" for r in inventory_rows)
    dup_dirs = [{"acquisition_id": k, "count": v} for k, v in seen_ids.items() if v > 1]

    # ---- cross-group consistency ----
    cross_rows = []
    for group in GROUPS:
        acqs = per_group_acq[group]
        expected = EXPECTED_SOURCE[group]
        actual = len(acqs)
        missing_outputs = actual < expected
        n_invalid_nifti = sum(1 for r in nifti_rows if r["group"] == group and r["status"] == "FAIL")
        n_invalid_mask = sum(1 for r in mask_rows if r["group"] == group and r["status"] == "FAIL")
        n_invalid_design = sum(1 for r in design_rows if r["group"] == group and r["status"] == "FAIL")
        n_missing_qc = sum(1 for r in qc_rows if r["group"] == group and r["status"] == "FAIL")
        n_prov_fail = sum(1 for r in prov_rows if r["group"] == group and r["status"] == "FAIL")
        pipeline_consistent = n_prov_fail == 0
        status = "PASS" if (n_invalid_nifti == 0 and n_invalid_mask == 0 and n_invalid_design == 0
                             and n_missing_qc == 0 and pipeline_consistent) else "REVIEW"
        cross_rows.append({
            "Group": group, "ExpectedAcquisitions": expected, "ActualAcquisitions": actual,
            "MissingOutputs": expected - actual if missing_outputs else 0,
            "ExtraOutputs": actual - expected if actual > expected else 0,
            "InvalidNIFTI": n_invalid_nifti, "InvalidMask": n_invalid_mask,
            "InvalidDesignMatrix": n_invalid_design, "MissingQC": n_missing_qc,
            "PipelineConsistency": pipeline_consistent, "Status": status,
        })

    # ---- write CSV/MD/TXT outputs ----
    def write_csv(name, rows):
        path = os.path.join(OUT_DIR, name)
        if not rows:
            with open(path, "w", encoding="utf-8") as f:
                f.write("")
            return
        keys = list(rows[0].keys())
        for r in rows:
            for k in r.keys():
                if k not in keys:
                    keys.append(k)
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)

    write_csv("four_group_inventory.csv", inventory_rows)
    write_csv("cross_group_consistency.csv", cross_rows)
    write_csv("acquisition_status.csv", inventory_rows)
    write_csv("qc_completeness.csv", qc_rows)
    write_csv("nifti_validation.csv", nifti_rows)
    write_csv("mask_validation.csv", mask_rows)
    write_csv("design_matrix_validation.csv", design_rows)
    write_csv("motion_validation.csv", motion_rows)
    write_csv("provenance_validation.csv", prov_rows)
    write_csv("duplicate_audit.csv", dup_rows + dup_dirs)

    missing_rows = []
    for group in GROUPS:
        subs, source_acqs = per_group_source[group]
        found_set = {(a["sub"], a["ses"], a["run"]) for a in per_group_acq[group]}
        for (sub, ses, run) in source_acqs:
            if (sub, ses, run) not in found_set:
                missing_rows.append({"group": group, "sub": sub, "ses": ses, "run": run,
                                      "reason": "present in BIDS source but no derivative output directory"
                                                " (excluded per audit/excluded_runs.csv or processing FAIL)"})
    write_csv("missing_outputs.csv", missing_rows)

    # four_group_inventory.md
    with open(os.path.join(OUT_DIR, "four_group_inventory.md"), "w", encoding="utf-8") as f:
        f.write("# Four-Group Derivatives Inventory\n\n")
        for group in GROUPS:
            acqs = per_group_acq[group]
            f.write(f"## {group} ({len(acqs)} acquisitions found)\n\n")
            for a in acqs:
                status = next(r["status"] for r in inventory_rows
                              if r["group"] == group and r["sub"] == a["sub"]
                              and r["ses"] == a["ses"] and r["run"] == a["run"])
                f.write(f"- {a['sub']}/{a['ses']}/{a['run']}: {status}\n")
            f.write("\n")

    # ---- final summary numbers ----
    total_expected = sum(EXPECTED_SOURCE.values())
    group_valid_counts = {}
    group_actual_counts = {}
    for group in GROUPS:
        acqs_status = [r["status"] for r in inventory_rows if r["group"] == group]
        group_actual_counts[group] = len(acqs_status)
        group_valid_counts[group] = sum(1 for s in acqs_status if s == "PASS")

    total_actual = sum(group_actual_counts.values())
    total_valid = sum(group_valid_counts.values())

    warn_fail_list = [r for r in inventory_rows if r["status"] in ("WARN", "FAIL")]

    with open(os.path.join(OUT_DIR, "FINAL_AUDIT_REPORT.md"), "w", encoding="utf-8") as f:
        f.write("# FINAL 4-GROUP PREPROCESSING OUTPUT AUDIT\n\n")
        f.write("Scope: derivatives/fsfast/{AD,CN_Final,EMCI,LMCI}/ — READ-ONLY audit. "
                "No preprocessing was rerun; no source or derivative files were modified.\n\n")

        f.write("## 1-5. Acquisition counts\n\n")
        f.write("| Group | Expected (source) | Actual (derivative dirs) | Difference | Reason |\n")
        f.write("|---|---|---|---|---|\n")
        for group in GROUPS:
            exp = EXPECTED_SOURCE[group]
            act = group_actual_counts[group]
            diff = act - exp
            reason = "matches expected source count" if diff == 0 else \
                "known exclusion(s) applied and/or a processing FAIL — see missing_outputs.csv and FAIL list below"
            f.write(f"| {group} | {exp} | {act} | {diff:+d} | {reason} |\n")
        f.write(f"\n**Total expected (raw source, per spec): {total_expected}**  \n")
        f.write(f"**Total actual (derivative acquisition directories found): {total_actual}**  \n")
        f.write(f"Total == 108 (raw source spec)? {'YES' if total_actual == 108 else 'NO — see per-group reasons above (exclusions/FAIL are expected and documented, not silently corrected)'}\n\n")

        f.write("## 6-10. File-level completeness (counts of PASS across all found acquisitions)\n\n")
        f.write(f"- Acquisitions with valid `desc-preproc_bold.nii.gz`: "
                f"{sum(1 for r in nifti_rows if r['status']=='PASS')}/{len(nifti_rows)}\n")
        f.write(f"- Acquisitions with valid brain mask: "
                f"{sum(1 for r in mask_rows if r['status']=='PASS')}/{len(mask_rows)}\n")
        f.write(f"- Acquisitions with valid motion parameter file: "
                f"{sum(1 for r in motion_rows if r['status']=='PASS')}/{len(motion_rows)}\n")
        f.write(f"- Acquisitions with valid 9-column design matrix: "
                f"{sum(1 for r in design_rows if r['status']=='PASS')}/{len(design_rows)}\n")
        f.write(f"- Acquisitions with QC metrics present (required files): "
                f"{sum(1 for r in qc_rows if r['status'] in ('PASS','WARN'))}/{len(qc_rows)}\n\n")

        f.write("## 11. Pipeline consistency across groups\n\n")
        f.write(f"- Acquisitions matching the frozen pipeline (mc-afni2 motion correction, slice-timing "
                f"SKIPPED, 6mm smoothing requested, masked mri_glmfit, 9-column design): "
                f"{sum(1 for r in prov_rows if r['status']=='PASS')}/{len(prov_rows)}\n\n")

        f.write("## 12-14. Missing outputs, duplicates, naming\n\n")
        f.write(f"- Missing outputs (present in BIDS source, absent from derivatives): {len(missing_rows)} "
                f"(see missing_outputs.csv)\n")
        f.write(f"- Duplicate BOLD file hashes (identical content across acquisitions): {len(dup_rows)}\n")
        f.write(f"- Duplicate acquisition directories (same sub/ses/run counted twice): {len(dup_dirs)}\n\n")

        f.write("## 15-16. Datatype / dimension / affine / TR / provenance inconsistencies\n\n")
        dims_set = set(r.get("dims") for r in nifti_rows if r.get("dims"))
        dtypes_set = set(r.get("datatype") for r in nifti_rows if r.get("datatype"))
        tr_set = set(r.get("TR") for r in nifti_rows if r.get("TR") is not None)
        f.write(f"- Distinct BOLD dims observed: {sorted(dims_set)}\n")
        f.write(f"- Distinct datatypes observed: {sorted(dtypes_set)}\n")
        f.write(f"- Distinct TR values observed: {sorted(tr_set)}\n\n")

        f.write("## 17-18. FAIL / WARN cases\n\n")
        fails = [r for r in inventory_rows if r["status"] == "FAIL"]
        warns = [r for r in inventory_rows if r["status"] == "WARN"]
        f.write(f"FAIL count: {len(fails)}\n\nWARN count: {len(warns)}\n\n")
        for r in fails:
            f.write(f"- FAIL: {r['group']}/{r['sub']}/{r['ses']}/{r['run']} — {r['reasons']}\n")
        for r in warns:
            f.write(f"- WARN: {r['group']}/{r['sub']}/{r['ses']}/{r['run']} — {r['reasons']}\n")
        f.write("\n")

        f.write("## 19. Readiness for next stage\n\n")
        ready = len(fails) == 0
        f.write(f"Technical FAIL cases present: {'YES' if fails else 'NO'}.\n")
        f.write(f"Dataset technically ready for next stage (MNI registration / ROI extraction) "
                f"pending user sign-off: {'YES, no blocking technical FAILs found' if ready else 'NO — resolve FAIL cases first'}\n\n")

        f.write("## Final status\n\n")
        for group in GROUPS:
            f.write(f"- {group}: {group_valid_counts[group]}/{group_actual_counts[group]} valid "
                    f"(found {group_actual_counts[group]} of {EXPECTED_SOURCE[group]} raw-source-expected)\n")
        f.write(f"\n**TOTAL: {total_valid}/{total_actual} valid** (raw-source spec total was {total_expected})\n")

    print("AUDIT_COMPLETE")
    print(json.dumps({
        "group_actual_counts": group_actual_counts,
        "group_valid_counts": group_valid_counts,
        "total_actual": total_actual,
        "total_valid": total_valid,
        "n_fail": len(warn_fail_list),
        "duplicates": len(dup_rows) + len(dup_dirs),
        "missing_outputs": len(missing_rows),
    }, indent=2))


if __name__ == "__main__":
    main()
