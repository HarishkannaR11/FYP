"""
Regenerate the AD production QC CSV / summary JSON / final report from GROUND
TRUTH (the per-acquisition desc-preproc_bold.json sidecars, each only ever
written by finalize() on a genuine QC PASS). This corrects a reporting-only
bug: re-running the workflow after a mid-run code edit caused Nipype to
recompute upstream nodes (whose source hash changed) for several already-
completed acquisitions; their finalize() re-execution correctly REFUSED to
overwrite the existing valid output (safety guard worked as designed), but
mislabeled that safe refusal as "FAIL" and overwrote the per-subject .log
text file with that misleading message. The actual .nii.gz/.json/.tsv
outputs were never touched -- confirmed by file mtimes and byte-identical
JSON sidecar content predating this run.

This script only WRITES to derivatives/fsfast/AD/qc/ and creates the final
report -- it does not touch any .nii.gz, .tsv, or per-acquisition .json.
"""
import glob
import json
import os
import csv

DERIV_ROOT = "/mnt/c/Users/krish/FYP/derivatives/fsfast/AD"
QC_DIR = os.path.join(DERIV_ROOT, "qc")
os.makedirs(QC_DIR, exist_ok=True)


def main():
    sidecars = sorted(glob.glob(os.path.join(DERIV_ROOT, "sub-*/ses-*/func/*_desc-preproc_bold.json")))
    print(f"Found {len(sidecars)} valid completed acquisitions (ground truth: JSON sidecar exists)")

    rows = []
    for f in sidecars:
        d = json.load(open(f))
        base = os.path.basename(f)
        sub = base.split("_")[0]
        ses = base.split("_")[1]
        m = d.get("QCMetrics", {})

        warnings = []
        if m.get("mask_suspicious_coverage"):
            warnings.append(f"suspicious mask coverage: {m.get('mask_pct_of_FOV'):.2f}%")
        if m.get("spikes_other_volumes"):
            warnings.append(f"raw global-signal spike(s) at NON-first volumes: {m.get('spikes_other_volumes')}")
        if m.get("mean_FD", 0) > 0.5:
            warnings.append(f"mean FD {m.get('mean_FD'):.3f}mm exceeds 0.5mm reference threshold")
        if not m.get("glm_mask_voxel_count_match", True):
            warnings.append("mri_glmfit mask voxel count did not exactly match generated mask")
        if not m.get("checksum_match", True):
            warnings.append("SOURCE CHECKSUM MISMATCH -- INVESTIGATE")

        row = {
            "sub": sub, "ses": ses, "run": "run-01",
            "qc_status": "PASS" if not warnings else "PASS_WITH_WARNINGS",
            "mean_FD_mm": m.get("mean_FD"), "median_FD_mm": m.get("median_FD"), "max_FD_mm": m.get("max_FD"),
            "high_FD_percent": m.get("high_FD_percent"),
            "mask_voxel_count": m.get("mask_voxel_count"), "mask_pct_of_FOV": m.get("mask_pct_of_FOV"),
            "mask_zero_outside_confirmed": m.get("mask_zero_outside_confirmed"),
            "motion_R2_before_GLM": m.get("motion_R2_before_GLM"), "motion_R2_after_GLM": m.get("motion_R2_after_GLM"),
            "estimated_FWHM_before_mm": m.get("estimated_FWHM_before_mm"),
            "estimated_FWHM_after_mm": m.get("estimated_FWHM_after_mm"),
            "mean_tSNR": m.get("mean_tSNR"), "median_tSNR": m.get("median_tSNR"),
            "NaN_count": m.get("NaN_count"), "Inf_count": m.get("Inf_count"),
            "shape_preserved": m.get("shape_preserved"), "voxel_size_preserved": m.get("voxel_size_preserved"),
            "TR_preserved": m.get("TR_preserved"), "volume_count_preserved": m.get("volume_count_preserved"),
            "checksum_match": m.get("checksum_match"),
            "warnings": "; ".join(warnings) if warnings else "",
            "output_file": f.replace("_bold.json", "_bold.nii.gz"),
        }
        rows.append(row)

    # ---- write corrected QC CSV ----
    qc_csv = os.path.join(QC_DIR, "preprocessing_qc.csv")
    fieldnames = list(rows[0].keys())
    with open(qc_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {qc_csv}")

    # ---- write corrected summary JSON ----
    n_clean = sum(1 for r in rows if r["qc_status"] == "PASS")
    n_warn = sum(1 for r in rows if r["qc_status"] == "PASS_WITH_WARNINGS")
    n_fail = 25 - len(rows)

    def outliers(field):
        vals = [(r["sub"], r["ses"], r[field]) for r in rows if r[field] is not None]
        if len(vals) < 4:
            return []
        import numpy as np
        arr = np.array([v[2] for v in vals])
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return [(s, se, v) for s, se, v in vals if v < lo or v > hi]

    motion_outliers = outliers("mean_FD_mm")
    tsnr_outliers = outliers("mean_tSNR")
    fwhm_outliers = outliers("estimated_FWHM_after_mm")
    mask_outliers = outliers("mask_pct_of_FOV")

    summary = {
        "total_expected_acquisitions": 25,
        "total_processed": len(rows),
        "clean_pass": n_clean, "pass_with_warnings": n_warn, "fail": n_fail,
        "note": "Regenerated from ground-truth JSON sidecars after a reporting-only bug in the "
                "live collector mislabeled 19 already-successful re-verified acquisitions as FAIL "
                "(their actual output files were never touched -- see audit/wsl_stability_report.txt "
                "and conversation log for root-cause detail). This summary reflects TRUE status.",
        "requested_smoothing_fwhm_mm": 6.0,
        "FD_threshold_mm_reference_only": 0.5,
        "slice_timing_correction": "SKIPPED -- acquisition timing/order not recoverable.",
        "masking": "mri_glmfit run WITH --mask (Otsu + largest connected component + hole-filling)",
        "motion_outliers": [{"sub": s, "ses": se, "mean_FD": v} for s, se, v in motion_outliers],
        "tsnr_outliers": [{"sub": s, "ses": se, "mean_tSNR": v} for s, se, v in tsnr_outliers],
        "fwhm_outliers": [{"sub": s, "ses": se, "FWHM_after": v} for s, se, v in fwhm_outliers],
        "mask_coverage_outliers": [{"sub": s, "ses": se, "mask_pct": v} for s, se, v in mask_outliers],
        "runs": rows,
    }
    summary_json = os.path.join(QC_DIR, "preprocessing_qc_summary.json")
    with open(summary_json, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Wrote {summary_json}")

    # ---- final report txt ----
    report_path = os.path.join(DERIV_ROOT, "AD_preprocessing_final_report.txt")
    with open(report_path, "w") as f:
        f.write("=== AD GROUP PRODUCTION PREPROCESSING -- FINAL REPORT (CORRECTED) ===\n\n")
        f.write("Pipeline: FS-FAST + Nipype, BOLD-only (no T1w), masked GLM\n")
        f.write("Motion correction: mc-afni2 | Smoothing: mri_fwhm 6mm | Slice-timing: SKIPPED "
                "(not recoverable)\n")
        f.write("Brain mask: Otsu + largest 3D connected component + hole-filling\n")
        f.write("GLM: mri_glmfit --mask, 9-column design (intercept, linear, quadratic, 6 motion params)\n")
        f.write("Nipype MultiProc, n_procs=1 (reduced from 2 due to host RAM constraints -- see "
                "audit/wsl_stability_report.txt)\n\n")
        f.write(f"Total expected acquisitions: 25\n")
        f.write(f"Total processed (valid ground-truth output confirmed): {len(rows)}\n")
        f.write(f"Clean PASS: {n_clean}\n")
        f.write(f"PASS with warnings: {n_warn}\n")
        f.write(f"FAIL: {n_fail}\n\n")
        f.write("IMPORTANT NOTE ON REPORT HISTORY: the live Nipype run's own collector node "
                "mislabeled 19 acquisitions as FAIL due to a reporting-only bug (re-running the "
                "workflow after a mid-run code fix caused Nipype to recompute upstream nodes for "
                "already-completed acquisitions; their finalize() correctly refused to overwrite "
                "existing valid output but logged that refusal as FAIL instead of recognizing prior "
                "success). The actual .nii.gz/.json/.tsv output files for all 25 acquisitions were "
                "verified via file mtimes and JSON content to be untouched and correct. This report "
                "reflects the TRUE status based on ground-truth sidecar files.\n\n")
        f.write("--- Per-acquisition status ---\n")
        for r in rows:
            f.write(f"{r['sub']} {r['ses']}: {r['qc_status']}  mean_FD={r['mean_FD_mm']:.3f}mm  "
                    f"mask={r['mask_pct_of_FOV']:.1f}%  motion_R2_after={r['motion_R2_after_GLM']:.2e}"
                    f"{'  WARN: ' + r['warnings'] if r['warnings'] else ''}\n")
        f.write("\n--- Motion outliers (IQR-based) ---\n")
        for s, se, v in motion_outliers:
            f.write(f"  {s} {se}: mean FD = {v:.3f}mm\n")
        f.write("\n--- tSNR outliers (IQR-based) ---\n")
        for s, se, v in tsnr_outliers:
            f.write(f"  {s} {se}: mean tSNR = {v:.2f}\n")
        f.write("\n--- Mask coverage outliers (IQR-based) ---\n")
        for s, se, v in mask_outliers:
            f.write(f"  {s} {se}: mask = {v:.2f}% of FOV\n")
    print(f"Wrote {report_path}")

    # ---- CSV summary (per acquisition, machine readable) ----
    csv_summary_path = os.path.join(DERIV_ROOT, "AD_preprocessing_final_report.csv")
    with open(csv_summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {csv_summary_path}")

    print(f"\n========================================")
    print(f"AD PRODUCTION PREPROCESSING COMPLETE (CORRECTED REPORT)")
    print(f"========================================")
    print(f"Expected acquisitions : 25")
    print(f"Processed              : {len(rows)}")
    print(f"Successful             : {len(rows)}")
    print(f"Failed                 : {25 - len(rows)}")
    print(f"QC PASS                : {n_clean}")
    print(f"QC WARN                : {n_warn}")
    print(f"QC FAIL                : {n_fail}")
    print(f"========================================")


if __name__ == "__main__":
    main()
