"""
INDEPENDENT VALIDATION of the biomarker pilot for sub-019S4549/ses-01/run-01.
Reads ONLY the saved output files (never touches preprocessing/brainnetome
sources, never modifies anything, never reprocesses). All checks are
recomputed from scratch in this separate process -- nothing is reused from
the generation script's memory.
"""
import os
import csv
import json
import numpy as np
import pandas as pd

RUN_DIR = "/mnt/c/Users/krish/FYP/derivatives/fsfast/AD/sub-019S4549/ses-01/run-01"
BM_DIR = os.path.join(RUN_DIR, "biomarkers")
BN_DIR = os.path.join(RUN_DIR, "brainnetome")
ATLAS_LUT = "/mnt/c/Users/krish/FYP/atlases/Brainnetome246/BN_Atlas_246_LUT.txt"

lut = {}
with open(ATLAS_LUT) as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            lut[int(parts[0])] = parts[1]


def hdr(t):
    print("\n" + "=" * 70); print(t); print("=" * 70)


checks = []  # (Metric, Expected, Actual, Status)


def record(metric, expected, actual, ok):
    checks.append((metric, str(expected), str(actual), "PASS" if ok else "FAIL"))


# =====================================================================
hdr("1. ALFF -- independent read + verification")
alff_path = os.path.join(BM_DIR, "alff_roi_values.npy")
alff = np.load(alff_path)
print(f"loaded: {alff_path}")
print(f"shape: {alff.shape}")
n_nan = int(np.isnan(alff).sum())
n_inf = int(np.isinf(alff).sum())
n_neg = int((alff[np.isfinite(alff)] < 0).sum())
print(f"NaN count: {n_nan}")
print(f"Inf count: {n_inf}")
print(f"negative values: {n_neg}")
print(f"every ROI (1-246) has a value: {alff.shape[0] == 246} (fixed-length vector, index i = ROI i+1)")
record("ALFF shape", "(246,)", alff.shape, alff.shape == (246,))
record("ALFF NaN", 0, n_nan, n_nan == 0)
record("ALFF Inf", 0, n_inf, n_inf == 0)
record("ALFF negative values", 0, n_neg, n_neg == 0)

# =====================================================================
hdr("2. ReHo -- independent read + verification")
reho_path = os.path.join(BM_DIR, "reho_roi_values.npy")
reho = np.load(reho_path)
print(f"loaded: {reho_path}")
print(f"shape: {reho.shape}")
n_nan_r = int(np.isnan(reho).sum())
n_inf_r = int(np.isinf(reho).sum())
print(f"NaN count: {n_nan_r}")
print(f"Inf count: {n_inf_r}")
print(f"missing ROIs: {246 - reho.shape[0] if reho.shape[0] < 246 else 0}")

meta_path = os.path.join(BM_DIR, "biomarker_metadata.json")
with open(meta_path) as f:
    meta = json.load(f)
print(f"\nNeighborhood definition used by the original calculation (read from "
      f"biomarker_metadata.json, not re-derived): {meta['reho']['neighborhood']}")
print(f"reason recorded: {meta['reho']['reason']}")
record("ReHo shape", "(246,)", reho.shape, reho.shape == (246,))
record("ReHo NaN", 0, n_nan_r, n_nan_r == 0)
record("ReHo Inf", 0, n_inf_r, n_inf_r == 0)

# =====================================================================
hdr("3. FC -- independent read + recomputed statistics")
fc_path = os.path.join(BM_DIR, "fc_matrix.npy")
fc = np.load(fc_path)
print(f"loaded: {fc_path}")
print(f"shape: {fc.shape}")
fc_nan = int(np.isnan(fc).sum())
fc_inf = int(np.isinf(fc).sum())
fc_min = float(np.nanmin(fc))
fc_max = float(np.nanmax(fc))
fc_mean = float(np.nanmean(fc))
fc_median = float(np.nanmedian(fc))
diag = np.diag(fc)
diag_mean = float(diag.mean())
diag_min = float(diag.min())
diag_max = float(diag.max())
sym_error = float(np.max(np.abs(fc - fc.T)))
diag_approx_1 = bool(np.allclose(diag, 1.0, atol=1e-6))

print(f"NaN count: {fc_nan}")
print(f"Inf count: {fc_inf}")
print(f"minimum: {fc_min:.6f}")
print(f"maximum: {fc_max:.6f}")
print(f"mean: {fc_mean:.6f}")
print(f"median: {fc_median:.6f}")
print(f"diagonal mean: {diag_mean:.8f}")
print(f"diagonal minimum: {diag_min:.8f}")
print(f"diagonal maximum: {diag_max:.8f}")
print(f"symmetry error max(abs(FC - FC.T)): {sym_error:.3e}")
print(f"diagonal approximately 1: {diag_approx_1}")
record("FC shape", "(246, 246)", fc.shape, fc.shape == (246, 246))
record("FC NaN", 0, fc_nan, fc_nan == 0)
record("FC Inf", 0, fc_inf, fc_inf == 0)
record("FC symmetry", "~0", f"{sym_error:.2e}", sym_error < 1e-6)
record("FC diagonal ~1", "True", diag_approx_1, diag_approx_1)

# =====================================================================
hdr("4. Degree Centrality -- independently recomputed from fc_matrix.npy")
print("Recomputing using EXACTLY the documented definition (from biomarker_metadata.json):")
print(f"  {meta['degree_centrality']['definition']}  (weighted={meta['degree_centrality']['weighted']}, "
     f"signed={meta['degree_centrality']['signed']}, threshold={meta['degree_centrality']['threshold']})")

fc_no_diag = fc.copy()
np.fill_diagonal(fc_no_diag, 0.0)
dc_recomputed = fc_no_diag.sum(axis=1)

dc_saved_path = os.path.join(BM_DIR, "degree_centrality.npy")
dc_saved = np.load(dc_saved_path)
print(f"loaded saved: {dc_saved_path}  shape={dc_saved.shape}")
print(f"recomputed shape: {dc_recomputed.shape}")

dc_diff = np.abs(dc_saved - dc_recomputed)
dc_max_diff = float(dc_diff.max())
dc_match = bool(np.allclose(dc_saved, dc_recomputed, atol=1e-8))
print(f"max absolute difference (saved vs. recomputed): {dc_max_diff:.3e}")
print(f"all 246 match within 1e-8: {dc_match}")

dc_compare_rows = []
for i in range(246):
    rid = i + 1
    d = abs(dc_saved[i] - dc_recomputed[i])
    dc_compare_rows.append({"ROI_ID": rid, "Saved_DC": dc_saved[i], "Recomputed_DC": dc_recomputed[i],
                            "Absolute_Difference": d, "Match": bool(d < 1e-8)})
dc_compare_path = os.path.join(BM_DIR, "validation_dc_comparison.csv")
with open(dc_compare_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["ROI_ID", "Saved_DC", "Recomputed_DC", "Absolute_Difference", "Match"])
    w.writeheader(); w.writerows(dc_compare_rows)
print(f"saved (new validation-only artifact, does not touch original files): {dc_compare_path}")
record("DC agreement (saved vs recomputed)", "max diff < 1e-8", f"{dc_max_diff:.2e}", dc_match)

# =====================================================================
hdr("5. FC Regional Strength -- independently recomputed")
print("Recomputing: FC_Strength(i) = mean(FC[i,j]) for j != i")
fc_strength_recomputed = fc_no_diag.sum(axis=1) / 245.0

strength_csv_path = os.path.join(BM_DIR, "fc_regional_strength.csv")
strength_df = pd.read_csv(strength_csv_path)
saved_strength = strength_df["FC_Strength"].values
print(f"loaded saved: {strength_csv_path}  n={len(saved_strength)}")

strength_diff = np.abs(saved_strength - fc_strength_recomputed)
strength_max_diff = float(strength_diff.max())
strength_match = bool(np.allclose(saved_strength, fc_strength_recomputed, atol=1e-8))
print(f"max absolute difference (saved vs. recomputed): {strength_max_diff:.3e}")
print(f"all 246 match within 1e-8: {strength_match}")
record("FC-strength agreement", "max diff < 1e-8", f"{strength_max_diff:.2e}", strength_match)

# =====================================================================
hdr("6. Regional Biomarker Matrix -- independent read + verification")
reg_path = os.path.join(BM_DIR, "regional_biomarker_matrix.csv")
reg = pd.read_csv(reg_path)
print(f"loaded: {reg_path}  shape={reg.shape}")
n_rows_ok = len(reg) == 246
roi_ids = sorted(reg["ROI_ID"].tolist())
ids_exact = roi_ids == list(range(1, 247))
n_dupes = reg["ROI_ID"].duplicated().sum()
n_missing_roi = len(set(range(1, 247)) - set(reg["ROI_ID"]))
print(f"exactly 246 rows: {n_rows_ok}")
print(f"ROI IDs are exactly 1-246, each once: {ids_exact}")
print(f"duplicate ROI_ID rows: {int(n_dupes)}")
print(f"missing ROI count: {n_missing_roi}")

reg_alff_ok = bool(np.allclose(reg["ALFF"].values, alff, atol=1e-8, equal_nan=True))
reg_reho_ok = bool(np.allclose(reg["ReHo"].values, reho, atol=1e-8, equal_nan=True))
reg_dc_ok = bool(np.allclose(reg["DC"].values, dc_saved, atol=1e-8, equal_nan=True))
reg_fcstr_ok = bool(np.allclose(reg["FC_Strength"].values, saved_strength, atol=1e-8, equal_nan=True))
print(f"\nregional_biomarker_matrix.ALFF == alff_roi_values.npy: {reg_alff_ok}")
print(f"regional_biomarker_matrix.ReHo == reho_roi_values.npy: {reg_reho_ok}")
print(f"regional_biomarker_matrix.DC == degree_centrality.npy: {reg_dc_ok}")
print(f"regional_biomarker_matrix.FC_Strength == fc_regional_strength.csv: {reg_fcstr_ok}")

record("ROI count", 246, len(reg), n_rows_ok)
record("Final matrix shape", "(246, 6)", reg.shape, reg.shape == (246, 6))

all_col_match = reg_alff_ok and reg_reho_ok and reg_dc_ok and reg_fcstr_ok
record("Regional matrix internal consistency", "all 4 columns match source files", all_col_match, all_col_match)

# =====================================================================
hdr("7. Cross-file consistency table")
print(f"{'Metric':35s} {'Expected':>15s} {'Actual':>15s} {'Status':>8s}")
for m, e, a, s in checks:
    print(f"{m:35s} {e:>15s} {a:>15s} {s:>8s}")

cross_check_path = os.path.join(BM_DIR, "validation_crosscheck_table.csv")
with open(cross_check_path, "w", newline="") as f:
    w = csv.writer(f); w.writerow(["Metric", "Expected", "Actual", "Status"])
    w.writerows(checks)
print(f"\nsaved (new validation-only artifact): {cross_check_path}")

# =====================================================================
hdr("8. Scientific sanity check (distributions only, no good/bad judgment)")
print("ALFF (raw amplitude units, not normalized):")
print(f"  min={alff.min():.1f} p25={np.percentile(alff,25):.1f} median={np.median(alff):.1f} "
     f"p75={np.percentile(alff,75):.1f} max={alff.max():.1f}")
print("ReHo (Kendall's W, bounded [0,1] by construction):")
print(f"  min={reho.min():.4f} p25={np.percentile(reho,25):.4f} median={np.median(reho):.4f} "
     f"p75={np.percentile(reho,75):.4f} max={reho.max():.4f}")
print(f"  all values within theoretical [0,1] bound: {bool((reho>=0).all() and (reho<=1).all())}")
print("Degree Centrality (signed, weighted sum of 245 correlations):")
print(f"  min={dc_saved.min():.2f} median={np.median(dc_saved):.2f} max={dc_saved.max():.2f}")
print("FC off-diagonal distribution:")
offdiag = fc[~np.eye(246, dtype=bool)]
print(f"  min={offdiag.min():.4f} median={np.median(offdiag):.4f} max={offdiag.max():.4f} "
     f"fraction negative={float((offdiag<0).mean()):.3f}")

const_alff = int((np.abs(alff - alff[0]) < 1e-9).all())
print(f"\nany constant-value ALFF array (all ROIs identical, suspicious): {bool(const_alff)}")
outlier_alff = np.abs(alff - np.median(alff)) > 5 * (np.percentile(alff, 75) - np.percentile(alff, 25))
print(f"ALFF ROIs >5xIQR from median (candidate outliers, not flagged as errors): "
     f"{int(outlier_alff.sum())} -- {[i+1 for i in np.where(outlier_alff)[0]]}")
outlier_reho = np.abs(reho - np.median(reho)) > 5 * (np.percentile(reho, 75) - np.percentile(reho, 25))
print(f"ReHo ROIs >5xIQR from median: {int(outlier_reho.sum())}")
print("\nMethodological choices in effect (documented in biomarker_metadata.json, repeated here for "
     "transparency): ALFF computed on already-band-pass-filtered data (not raw spectrum); ReHo uses "
     "27-voxel neighborhood restricted to in-mask voxels near boundaries; DC/FC use signed, "
     "unthresholded Pearson correlations with no absolute-value transform.")
print("\nThis section is a mathematical/statistical description only -- it is NOT a scientific claim "
     "about whether these values are biologically meaningful or diagnostically useful.")

# =====================================================================
hdr("FINAL STATUS")
all_pass = all(s == "PASS" for _, _, _, s in checks)
failed_checks = [m for m, e, a, s in checks if s == "FAIL"]
if all_pass:
    status = "PASS"
    reason = "Every mathematical/structural validation check (shapes, NaN/Inf, symmetry, diagonal, DC agreement, FC-strength agreement, internal cross-file consistency) passed."
else:
    status = "FAIL"
    reason = f"The following checks failed: {failed_checks}"
print(f"STATUS: {status}")
print(f"REASON: {reason}")
print("\nNo files were modified. No preprocessing was rerun. No biomarker outputs were regenerated. "
     "No other subject was processed.")
print("BIOMARKER_VALIDATION_DONE")
