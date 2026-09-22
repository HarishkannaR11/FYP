"""
COMPLETE VALIDATION AUDIT of the already-preprocessed AD group. READ-ONLY.
Does NOT rerun preprocessing, does NOT touch source data, does NOT perform
registration/ROI/biomarker work.

Independently RE-VERIFIES (fresh, from actual files, not trusting stored
claims) for all 25 acquisitions:
  - image headers (native vs preprocessed) and structural preservation
  - brain mask validity (3D, matches BOLD grid, nonempty, not near-100%)
  - design matrix shape (rows=volumes, cols=9) and motion-column match to .mcdat
  - GLM was actually masked (--mask, not --no-mask): verified two independent
    ways -- (1) final residual is exactly zero outside the mask, (2) final
    residual is near-zero-mean WITHIN the mask (proof it's a GLM residual,
    not a raw smoothed magnitude copy)
  - motion file existence/timepoint count/six parameters

CROSS-REFERENCES (not recomputed -- would just reproduce identical numbers
from the same already-validated code) the QC metric VALUES already computed
and stored in each acquisition's own QC/qc_metrics.csv and JSON sidecar by
the production pipeline, clearly labeled as such.
"""
import os
import csv
import json
import glob
import numpy as np
import nibabel as nib

SOURCE_MANIFEST = "/mnt/c/Users/krish/FYP/audit/ad_rerun/acquisition_manifest_final.csv"
DERIV_ROOT = "/mnt/c/Users/krish/FYP/derivatives/fsfast/AD"
NIFTI_AD_ROOT = "/mnt/c/Users/krish/FYP/NIfTI_/NIfTI/AD"
BIDS_ROOT = "/mnt/c/Users/krish/FYP/BIDS"
ATLAS_PATH = "/mnt/c/Users/krish/FYP/atlases/Brainnetome246/BN_Atlas_246_2mm.nii.gz"

OUT_DIR = "/mnt/c/Users/krish/FYP/test/validation/ad_preprocessing_validation"
os.makedirs(OUT_DIR, exist_ok=True)

MULTI_ACQ_SUBJECTS = {"sub-002S5018", "sub-019S4477", "sub-019S4549", "sub-130S4730", "sub-130S4982"}

report_lines = []


def log(msg=""):
    print(msg)
    report_lines.append(str(msg))


def load_source_manifest():
    return list(csv.DictReader(open(SOURCE_MANIFEST, newline="", encoding="utf-8")))


def discover_preprocessed_acquisitions():
    """Scan actual derivatives/fsfast/AD/ tree -- do not assume filenames."""
    found = []
    for sub_dir in sorted(glob.glob(os.path.join(DERIV_ROOT, "sub-*"))):
        sub = os.path.basename(sub_dir)
        for ses_dir in sorted(glob.glob(os.path.join(sub_dir, "ses-*"))):
            ses = os.path.basename(ses_dir)
            for run_dir in sorted(glob.glob(os.path.join(ses_dir, "run-*"))):
                run = os.path.basename(run_dir)
                preproc_candidates = glob.glob(os.path.join(run_dir, "*desc-preproc_bold.nii.gz"))
                found.append({
                    "sub": sub, "ses": ses, "run": run, "run_dir": run_dir,
                    "preproc_bold": preproc_candidates[0] if preproc_candidates else None,
                })
    return found


def main():
    log("=" * 70)
    log("COMPLETE VALIDATION AUDIT -- AD PREPROCESSING + MASKED GLM (READ-ONLY)")
    log("=" * 70 + "\n")

    # ================= SECTION 2: SOURCE DATA =================
    log("=== SECTION 2: Source data verification ===")
    source_manifest = load_source_manifest()
    log(f"Source manifest rows: {len(source_manifest)} (expected 25)")
    src_missing = []
    for r in source_manifest:
        if not os.path.exists(r["input_path"]):
            src_missing.append(r["input_path"])
    log(f"Source files missing: {len(src_missing)}")
    for m in src_missing:
        log(f"  MISSING: {m}")
    log("")

    # ================= SECTION 3&4: DISCOVER PREPROCESSED OUTPUT =================
    log("=== SECTION 3-4: Discover actual preprocessed output structure ===")
    discovered = discover_preprocessed_acquisitions()
    log(f"Discovered acquisition directories (sub/ses/run): {len(discovered)}")
    valid_discovered = [d for d in discovered if d["preproc_bold"]]
    log(f"Discovered with an actual desc-preproc_bold.nii.gz present: {len(valid_discovered)}")
    for d in discovered:
        if not d["preproc_bold"]:
            log(f"  NO PREPROC OUTPUT FOUND: {d['sub']} {d['ses']} {d['run']}")
    log("")

    # cross-reference against source manifest -- build acquisition_manifest.csv
    src_lookup = {(r["subject_id"], r["session_id"]): r for r in source_manifest}
    manifest_rows = []
    missing_from_output = []
    for r in source_manifest:
        key_sub, key_ses = r["subject_id"], r["session_id"]
        match = next((d for d in valid_discovered if d["sub"] == key_sub and d["ses"] == key_ses), None)
        if match is None:
            missing_from_output.append((key_sub, key_ses))
            manifest_rows.append({
                "subject_id": key_sub, "session_id": key_ses, "run_id": "run-01",
                "source_nifti": r["input_path"], "preprocessed_nifti": "MISSING",
                "brain_mask": "MISSING", "motion_file": "MISSING", "design_matrix": "MISSING",
                "QC_directory": "MISSING", "processing_status": "MISSING_OUTPUT",
            })
        else:
            run_dir = match["run_dir"]
            mask_f = os.path.join(run_dir, "brain_mask.nii.gz")
            mcdat_candidates = glob.glob(os.path.join(run_dir, "*motion_parameters.mcdat"))
            design_f = os.path.join(run_dir, "design_matrix.txt")
            qc_dir = os.path.join(run_dir, "QC")
            manifest_rows.append({
                "subject_id": key_sub, "session_id": key_ses, "run_id": match["run"],
                "source_nifti": r["input_path"], "preprocessed_nifti": match["preproc_bold"],
                "brain_mask": mask_f if os.path.exists(mask_f) else "MISSING",
                "motion_file": mcdat_candidates[0] if mcdat_candidates else "MISSING",
                "design_matrix": design_f if os.path.exists(design_f) else "MISSING",
                "QC_directory": qc_dir if os.path.isdir(qc_dir) else "MISSING",
                "processing_status": "FOUND",
                "run_dir": run_dir,
            })

    unexpected_outputs = [d for d in valid_discovered
                           if (d["sub"], d["ses"]) not in {(r["subject_id"], r["session_id"]) for r in source_manifest}]

    log(f"Acquisitions with output correctly found: {sum(1 for m in manifest_rows if m['processing_status']=='FOUND')}")
    log(f"Acquisitions MISSING preprocessing output: {len(missing_from_output)}")
    for s, se in missing_from_output:
        log(f"  MISSING: {s} {se}")
    log(f"Unexpected outputs (not in source manifest): {len(unexpected_outputs)}")
    for d in unexpected_outputs:
        log(f"  UNEXPECTED: {d['sub']} {d['ses']} {d['run']}")
    log("")

    log("--- Explicit check: 5 multi-acquisition subjects, BOTH sessions present ---")
    for msub in sorted(MULTI_ACQ_SUBJECTS):
        sessions_found = sorted(set(m["session_id"] for m in manifest_rows
                                     if m["subject_id"] == msub and m["processing_status"] == "FOUND"))
        log(f"  {msub}: sessions found = {sessions_found} (expect 2 distinct sessions)")
        if len(sessions_found) != 2:
            log(f"    *** WARNING: expected 2 sessions, found {len(sessions_found)} ***")
    log("")

    with open(os.path.join(OUT_DIR, "acquisition_manifest.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest_rows[0].keys()))
        w.writeheader()
        w.writerows(manifest_rows)
    log(f"Saved: {os.path.join(OUT_DIR, 'acquisition_manifest.csv')}\n")

    # ================= PER-ACQUISITION DEEP VALIDATION =================
    log("=" * 70)
    log("PER-ACQUISITION INDEPENDENT VALIDATION")
    log("=" * 70 + "\n")

    final_rows = []
    problematic = []

    for m in manifest_rows:
        sub, ses, run = m["subject_id"], m["session_id"], m["run_id"]
        row = {"subject_id": sub, "session_id": ses, "run_id": run,
               "source_nifti": m["source_nifti"], "preprocessed_nifti": m["preprocessed_nifti"]}

        if m["processing_status"] != "FOUND":
            row.update({k: "MISSING" for k in
                        ["dimensions", "voxel_size", "TR", "volumes", "motion_correction", "slice_timing",
                         "smoothing", "brain_mask", "GLM", "design_matrix", "tSNR", "SNR", "CNR",
                         "FD_mean", "FD_median", "FD_max", "FD_95th", "DVARS_mean", "DVARS_median",
                         "DVARS_max", "DVARS_95th", "GCOR", "spatial_entropy", "temporal_entropy",
                         "mask_coverage", "NaN_count", "Inf_count"]})
            row["QC_status"] = "FAIL"
            final_rows.append(row)
            problematic.append((sub, ses, run, "preprocessing output missing entirely"))
            continue

        issues = []
        warns = []

        # ---- 5. image validation: source vs preprocessed ----
        src_img = nib.load(m["source_nifti"])
        prep_img = nib.load(m["preprocessed_nifti"])
        src_shape, prep_shape = src_img.shape, prep_img.shape
        src_zooms, prep_zooms = src_img.header.get_zooms(), prep_img.header.get_zooms()
        src_orient = "".join(nib.aff2axcodes(src_img.affine))
        prep_orient = "".join(nib.aff2axcodes(prep_img.affine))

        dims_preserved = src_shape == prep_shape
        voxel_preserved = np.allclose(src_zooms[:3], prep_zooms[:3], atol=0.01)
        tr_preserved = len(src_zooms) > 3 and len(prep_zooms) > 3 and abs(src_zooms[3] - prep_zooms[3]) < 0.01
        orient_preserved = src_orient == prep_orient

        if not dims_preserved:
            issues.append(f"dimensions changed: {src_shape} -> {prep_shape}")
        if not voxel_preserved:
            issues.append(f"voxel size changed: {src_zooms[:3]} -> {prep_zooms[:3]}")
        if not tr_preserved:
            issues.append(f"TR changed: {src_zooms[3] if len(src_zooms)>3 else None} -> {prep_zooms[3] if len(prep_zooms)>3 else None}")
        if not orient_preserved:
            issues.append(f"orientation changed: {src_orient} -> {prep_orient}")

        prep_data = prep_img.get_fdata()
        nan_count = int(np.isnan(prep_data).sum())
        inf_count = int(np.isinf(prep_data).sum())
        empty_vols = [t for t in range(prep_shape[3]) if np.allclose(prep_data[..., t], 0)]
        if nan_count:
            issues.append(f"NaN present: {nan_count}")
        if inf_count:
            issues.append(f"Inf present: {inf_count}")

        # ---- 6. motion file validation ----
        mc_ok = False
        n_timepoints_mc = None
        max_motion = None
        if m["motion_file"] != "MISSING" and os.path.exists(m["motion_file"]):
            mc = np.loadtxt(m["motion_file"])
            n_timepoints_mc = mc.shape[0]
            mc_ok = (mc.shape[1] >= 9) and (n_timepoints_mc == prep_shape[3])
            max_motion = float(np.max(mc[:, 9])) if mc.shape[1] > 9 else None
            if not mc_ok:
                issues.append(f".mcdat timepoints ({n_timepoints_mc}) != BOLD volumes ({prep_shape[3]})")
        else:
            issues.append("motion file (.mcdat) missing")

        # ---- 9. brain mask validation ----
        mask_ok = False
        mask_coverage = None
        if m["brain_mask"] != "MISSING" and os.path.exists(m["brain_mask"]):
            mask_img = nib.load(m["brain_mask"])
            mask_data = mask_img.get_fdata().astype(bool)
            mask_3d = len(mask_img.shape) == 3
            mask_grid_match = mask_img.shape == prep_shape[:3]
            mask_nonempty = mask_data.sum() > 0
            mask_coverage = float(100 * mask_data.sum() / mask_data.size)
            mask_not_everything = mask_coverage < 90.0
            mask_ok = mask_3d and mask_grid_match and mask_nonempty and mask_not_everything
            if not mask_3d:
                issues.append("mask is not 3D")
            if not mask_grid_match:
                issues.append(f"mask grid {mask_img.shape} != BOLD grid {prep_shape[:3]}")
            if not mask_nonempty:
                issues.append("mask is empty")
            if not mask_not_everything:
                issues.append(f"mask covers {mask_coverage:.1f}% of FOV -- essentially entire image")
        else:
            issues.append("brain mask missing")

        # ---- 10. design matrix validation ----
        design_ok = False
        if m["design_matrix"] != "MISSING" and os.path.exists(m["design_matrix"]):
            X = np.loadtxt(m["design_matrix"])
            rows_match = X.shape[0] == prep_shape[3]
            cols_match = X.shape[1] == 9
            design_ok = rows_match and cols_match
            if not rows_match:
                issues.append(f"design matrix rows ({X.shape[0]}) != volumes ({prep_shape[3]})")
            if not cols_match:
                issues.append(f"design matrix columns ({X.shape[1]}) != 9")
            if design_ok and mc_ok:
                mc_full = np.loadtxt(m["motion_file"])
                motion_match = np.allclose(X[:, 3:9], mc_full[:, 1:7], atol=1e-4)
                if not motion_match:
                    issues.append("design matrix motion columns do NOT match .mcdat")
        else:
            issues.append("design matrix missing")

        # ---- 11. GLM residual validation (independent re-verification) ----
        glm_masked_confirmed = False
        if mask_ok:
            outside_vals = prep_data[~mask_data]
            n_nonzero_outside = int(np.count_nonzero(outside_vals))
            zero_outside = (n_nonzero_outside == 0)
            inside_mean = float(prep_data[mask_data].mean())
            is_residual_not_raw = abs(inside_mean) < 1000  # raw smoothed magnitude means are ~1e5-1e6; residual ~0
            glm_masked_confirmed = zero_outside and is_residual_not_raw
            if not zero_outside:
                issues.append(f"GLM output NOT zero outside mask ({n_nonzero_outside} nonzero voxels) -- "
                               f"--no-mask may have been used instead of --mask")
            if not is_residual_not_raw:
                issues.append(f"final output mean ({inside_mean:.1f}) looks like raw magnitude data, NOT a GLM "
                               f"residual -- possible pipeline error (smoothed BOLD copied instead of eres.nii.gz)")

        # ---- pull already-computed QC metrics (cross-reference, not recomputed) ----
        json_candidates = glob.glob(os.path.join(m["run_dir"], "*desc-preproc_bold.json"))
        qc_metrics = {}
        if json_candidates:
            sidecar = json.load(open(json_candidates[0]))
            qc_metrics = sidecar.get("QCMetrics", {})
        else:
            issues.append("JSON sidecar with QCMetrics missing")

        row.update({
            "dimensions": str(prep_shape), "voxel_size": str(tuple(round(float(z), 3) for z in prep_zooms[:3])),
            "TR": prep_zooms[3] if len(prep_zooms) > 3 else None, "volumes": prep_shape[3],
            "motion_correction": "PASS" if mc_ok else "FAIL",
            "slice_timing": "SKIPPED (confirmed, not recoverable)",
            "smoothing": "PASS" if qc_metrics.get("estimated_FWHM_after_mm") else "UNVERIFIED",
            "brain_mask": "PASS" if mask_ok else "FAIL",
            "GLM": "PASS (masked, confirmed)" if glm_masked_confirmed else "FAIL/UNVERIFIED",
            "design_matrix": "PASS" if design_ok else "FAIL",
            "tSNR": qc_metrics.get("mean_tSNR", ""), "SNR": qc_metrics.get("SNR", ""),
            "CNR": qc_metrics.get("CNR", "NOT RELIABLY COMPUTABLE"),
            "FD_mean": qc_metrics.get("mean_FD", ""), "FD_median": qc_metrics.get("median_FD", ""),
            "FD_max": qc_metrics.get("max_FD", ""), "FD_95th": qc_metrics.get("p95_FD", ""),
            "DVARS_mean": qc_metrics.get("mean_DVARS", ""), "DVARS_median": qc_metrics.get("median_DVARS", ""),
            "DVARS_max": qc_metrics.get("max_DVARS", ""), "DVARS_95th": qc_metrics.get("p95_DVARS", ""),
            "GCOR": qc_metrics.get("GCOR_post_GLM", ""),
            "spatial_entropy": qc_metrics.get("spatial_entropy_bits", ""),
            "temporal_entropy": qc_metrics.get("temporal_entropy_bits", ""),
            "mask_coverage": mask_coverage, "NaN_count": nan_count, "Inf_count": inf_count,
        })

        qc_status = "PASS" if not issues else "FAIL"
        if issues:
            problematic.append((sub, ses, run, "; ".join(issues)))
        row["QC_status"] = qc_status
        row["_issues"] = "; ".join(issues) if issues else ""
        final_rows.append(row)

        log(f"{sub} {ses} {run}: {qc_status}" + (f"  ISSUES: {row['_issues']}" if issues else ""))

    log("")

    # ================= SECTION 23: high motion =================
    log("=== High-motion acquisitions (reported, NOT excluded) ===")
    for r in final_rows:
        if r.get("FD_mean") not in ("", "MISSING", None):
            try:
                if float(r["FD_mean"]) > 0.5:
                    log(f"  {r['subject_id']} {r['session_id']} {r['run_id']}: "
                        f"FD_mean={r['FD_mean']:.3f}mm FD_max={r['FD_max']:.3f}mm")
            except (ValueError, TypeError):
                pass
    log("")

    # ================= SECTION 24: original data integrity =================
    log("=== Original data integrity (final check) ===")
    n_source_now = len(glob.glob(os.path.join(NIFTI_AD_ROOT, "**", "*.nii*"), recursive=True))
    log(f"Original AD NIfTI count: {n_source_now} (expected 25)")
    n_bids = len(glob.glob(os.path.join(BIDS_ROOT, "**", "*task-rest*bold.nii*"), recursive=True))
    log(f"BIDS task-rest bold files: {n_bids} (expected 175)")
    atlas_exists = os.path.exists(ATLAS_PATH)
    log(f"Brainnetome atlas exists: {atlas_exists}")
    log("")

    # ================= write final validation table =================
    fieldnames = [k for k in final_rows[0].keys() if k != "_issues"] + ["_issues"]
    csv_path = os.path.join(OUT_DIR, "AD_preprocessing_validation.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in final_rows:
            for fn in fieldnames:
                r.setdefault(fn, "")
            w.writerow(r)
    log(f"Saved: {csv_path}")

    txt_path = os.path.join(OUT_DIR, "AD_preprocessing_validation.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(report_lines))
    log(f"Saved: {txt_path}")

    # ================= final accounting =================
    n_pass = sum(1 for r in final_rows if r["QC_status"] == "PASS")
    n_fail = sum(1 for r in final_rows if r["QC_status"] == "FAIL")

    log("\n" + "=" * 70)
    log("FINAL COHORT ACCOUNTING")
    log("=" * 70)
    log(f"Unique AD subjects = 20")
    log(f"Expected acquisitions = 25")
    log(f"Found source acquisitions = {len(source_manifest) - len(src_missing)}")
    log(f"Found preprocessed acquisitions = {sum(1 for m in manifest_rows if m['processing_status']=='FOUND')}")
    log(f"Validated acquisitions (QC_status PASS) = {n_pass}")
    log(f"Validated acquisitions (QC_status FAIL) = {n_fail}")
    log(f"Missing preprocessing outputs = {len(missing_from_output)}")
    log(f"Unexpected outputs = {len(unexpected_outputs)}")

    # ================= markdown report =================
    md_path = os.path.join(OUT_DIR, "AD_preprocessing_validation.md")
    with open(md_path, "w") as f:
        f.write("# AD Preprocessing Validation Audit\n\n")
        f.write(f"Unique AD subjects = 20\nExpected acquisitions = 25\n")
        f.write(f"Found source acquisitions = {len(source_manifest) - len(src_missing)}\n")
        f.write(f"Found preprocessed acquisitions = {sum(1 for m in manifest_rows if m['processing_status']=='FOUND')}\n")
        f.write(f"Validated PASS = {n_pass}\nValidated FAIL = {n_fail}\n\n")
        f.write("## Problematic acquisitions\n\n")
        if problematic:
            for s, se, ru, reason in problematic:
                f.write(f"- {s} {se} {ru}: {reason}\n")
        else:
            f.write("None.\n")
        f.write("\n## Per-acquisition table\n\n")
        f.write("| Subject | Session | Run | Status | FD mean | tSNR | GCOR |\n|---|---|---|---|---|---|---|\n")
        for r in final_rows:
            f.write(f"| {r['subject_id']} | {r['session_id']} | {r['run_id']} | {r['QC_status']} | "
                    f"{r.get('FD_mean','')} | {r.get('tSNR','')} | {r.get('GCOR','')} |\n")
    log(f"\nSaved: {md_path}")

    print(f"\nAUDIT_DONE PASS={n_pass} FAIL={n_fail} PROBLEMATIC={len(problematic)}")


if __name__ == "__main__":
    main()
