"""
SINGLE-ACQUISITION PILOT: Brainnetome-246 parcellation + ROI time-series
extraction on ONE already-preprocessed AD acquisition (sub-019S4549/ses-01/run-01).

Read-only with respect to existing preprocessing outputs: only ADDS a new
brainnetome/ subdirectory inside that acquisition's existing run-01/ folder.
Does not rerun preprocessing, does not touch any other acquisition/group,
does not compute ALFF/ReHo/DC/FC.
"""
import os
import csv
import json
import platform
import datetime
import numpy as np
import nibabel as nib
import pandas as pd

SUBJECT, SESSION, RUN, GROUP = "sub-019S4549", "ses-01", "run-01", "AD"
RUN_DIR = f"/mnt/c/Users/krish/FYP/derivatives/fsfast/{GROUP}/{SUBJECT}/{SESSION}/{RUN}"
BOLD_PATH = os.path.join(RUN_DIR, f"{SUBJECT}_{SESSION}_task-rest_{RUN}_desc-preproc_bold.nii.gz")
ATLAS_PATH = "/mnt/c/Users/krish/FYP/atlases/Brainnetome246/BN_Atlas_246_2mm.nii.gz"
ATLAS_LUT = "/mnt/c/Users/krish/FYP/atlases/Brainnetome246/BN_Atlas_246_LUT.txt"

OUT_DIR = os.path.join(RUN_DIR, "brainnetome")
os.makedirs(OUT_DIR, exist_ok=True)


def hdr(t):
    print("\n" + "=" * 70); print(t); print("=" * 70)


hdr("STEP 1 -- LOCATE AND VERIFY EXACT INPUT")
print(f"PREPROCESSED BOLD (input): {BOLD_PATH}")
print(f"exists: {os.path.isfile(BOLD_PATH)}")
print(f"ATLAS (input, read-only):  {ATLAS_PATH}")
print(f"ATLAS LUT (input, read-only): {ATLAS_LUT}")
assert os.path.isfile(BOLD_PATH), "FATAL: expected preprocessed BOLD not found"

hdr("STEP 2 -- INSPECT PREPROCESSED BOLD")
bold_img = nib.load(BOLD_PATH)
bold_data = bold_img.get_fdata(dtype=np.float32)
bold_info = {
    "dimensions": str(bold_img.shape),
    "n_volumes": int(bold_img.shape[3]) if len(bold_img.shape) == 4 else None,
    "voxel_size_mm": tuple(round(float(z), 4) for z in bold_img.header.get_zooms()[:3]),
    "TR_s": float(bold_img.header.get_zooms()[3]) if len(bold_img.header.get_zooms()) > 3 else None,
    "orientation": "".join(nib.aff2axcodes(bold_img.affine)),
    "datatype": str(bold_img.get_data_dtype()),
    "NaN_count": int(np.isnan(bold_data).sum()),
    "Inf_count": int(np.isinf(bold_data).sum()),
}
for k, v in bold_info.items():
    print(f"  {k}: {v}")
print("  affine:")
for row in bold_img.affine:
    print("    " + np.array2string(row, precision=5, suppress_small=True))
T = bold_info["n_volumes"]
print(f"\n  T (actual number of volumes, read from file, not assumed) = {T}")

hdr("STEP 3 -- INSPECT BRAINNETOME ATLAS (original, untouched)")
atlas_img = nib.load(ATLAS_PATH)
atlas_data_native = np.asarray(atlas_img.dataobj)
uniq_native = np.unique(atlas_data_native)
atlas_info = {
    "dimensions": str(atlas_img.shape),
    "voxel_size_mm": tuple(round(float(z), 4) for z in atlas_img.header.get_zooms()[:3]),
    "orientation": "".join(nib.aff2axcodes(atlas_img.affine)),
    "n_unique_labels_incl_0": int(len(uniq_native)),
    "min_label": float(uniq_native.min()),
    "max_label": float(uniq_native.max()),
}
for k, v in atlas_info.items():
    print(f"  {k}: {v}")
print("  atlas affine:")
for row in atlas_img.affine:
    print("    " + np.array2string(row, precision=5, suppress_small=True))

labels_1_246_present = set(int(x) for x in uniq_native if x > 0)
missing_labels = sorted(set(range(1, 247)) - labels_1_246_present)
extra_labels = sorted(labels_1_246_present - set(range(1, 247)))
print(f"\n  CHECKED (not assumed): labels 1-246 all present in native atlas: {len(missing_labels)==0 and len(extra_labels)==0}")
print(f"  missing labels: {missing_labels}")
print(f"  unexpected/extra labels: {extra_labels}")

hdr("STEP 4 -- ALIGN BRAINNETOME TO CURRENT BOLD GRID (nearest-neighbor only)")
from nilearn.image import resample_img
atlas_canon = nib.as_closest_canonical(atlas_img)
print(f"  atlas reoriented (lossless, zero interpolation) to: "
      f"{''.join(nib.aff2axcodes(atlas_canon.affine))} before resampling")

aligned = resample_img(atlas_canon, target_affine=bold_img.affine, target_shape=bold_img.shape[:3],
                       interpolation="nearest", force_resample=True, copy_header=True)
aligned_path = os.path.join(OUT_DIR, "brainnetome_246_4mm.nii.gz")
aligned_int = nib.Nifti1Image(np.asarray(aligned.dataobj).astype(np.int16), bold_img.affine, bold_img.header)
aligned_int.header.set_data_dtype(np.int16)
nib.save(aligned_int, aligned_path)
print(f"  interpolation used: NEAREST NEIGHBOR ONLY (nilearn resample_img, interpolation='nearest')")
print(f"  saved: {aligned_path}")

grid_match = bool(aligned_int.shape == bold_img.shape[:3] and np.allclose(aligned_int.affine, bold_img.affine, atol=1e-6))
print(f"\n  aligned atlas dimensions: {aligned_int.shape}  (BOLD: {bold_img.shape[:3]})")
print(f"  aligned atlas affine == BOLD affine: {grid_match}")
print(f"  original 2mm atlas file modified: NO (only read)")

hdr("STEP 5 -- INDEPENDENT ATLAS VALIDATION (recomputed from the saved aligned file)")
reloaded = nib.load(aligned_path)
a4 = np.asarray(reloaded.dataobj).astype(np.int32)
uniq_aligned = np.unique(a4)
print(f"  reloaded aligned atlas unique labels (incl 0): {len(uniq_aligned)}  min={uniq_aligned.min()}  max={uniq_aligned.max()}")

lut = {}
with open(ATLAS_LUT) as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            lut[int(parts[0])] = parts[1]

roi_rows = []
counts = []
for rid in range(1, 247):
    n = int((a4 == rid).sum())
    counts.append(n)
    roi_rows.append({"ROI_ID": rid, "ROI_Name": lut.get(rid, f"ROI_{rid}"), "Voxel_Count": n})
counts = np.array(counts)

with open(os.path.join(OUT_DIR, "roi_voxel_counts.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["ROI_ID", "ROI_Name", "Voxel_Count"])
    w.writeheader(); w.writerows(roi_rows)

n_present = int((counts > 0).sum())
n_zero = int((counts == 0).sum())
n_lt5 = int((counts < 5).sum())
n_lt10 = int((counts < 10).sum())
print(f"  ROIs present (>0 voxels): {n_present} / 246")
print(f"  zero-voxel ROIs: {n_zero}")
print(f"  ROIs <5 voxels: {n_lt5}")
print(f"  ROIs <10 voxels: {n_lt10}")
print(f"  min={int(counts.min())}  median={float(np.median(counts))}  max={int(counts.max())}")
print(f"  saved: {os.path.join(OUT_DIR, 'roi_voxel_counts.csv')}")
print(f"\n  BRAINNETOME COVERAGE (actual, not assumed): {n_present} / 246")

hdr("STEP 6 -- EXTRACT ROI TIME SERIES (mean signal per ROI, no re-resampling)")
# Atlas already aligned to the exact BOLD grid -- extract directly by index masking
# rather than routing through NiftiLabelsMasker's own internal resampling step,
# which would silently re-resample the mask against its own default target_affine
# logic. Direct masking on the already-verified-identical grid is equivalent and
# avoids that risk entirely.
flat_bold = bold_data.reshape(-1, T)  # (n_voxels, T)
flat_atlas = a4.reshape(-1)
roi_ts = np.zeros((T, 246), dtype=np.float64)
for rid in range(1, 247):
    vox = flat_atlas == rid
    if vox.sum() > 0:
        roi_ts[:, rid - 1] = flat_bold[vox, :].mean(axis=0)
    else:
        roi_ts[:, rid - 1] = np.nan  # explicit: this ROI has no voxels, not silently zero

hdr("STEP 7 -- SAVE ROI MATRIX (T x 246)")
print(f"  T (from BOLD) = {T},  ROI count = 246  ->  expected shape = ({T}, 246)")
print(f"  actual roi_ts.shape = {roi_ts.shape}")
assert roi_ts.shape == (T, 246), "FATAL: ROI matrix shape mismatch"

cols = [f"ROI_{i:03d}" for i in range(1, 247)]
ts_df = pd.DataFrame(roi_ts, columns=cols)
ts_df.insert(0, "time", np.arange(1, T + 1))
ts_csv_path = os.path.join(OUT_DIR, "roi_timeseries.csv")
ts_df.to_csv(ts_csv_path, index=False)
npy_path = os.path.join(OUT_DIR, "roi_timeseries.npy")
np.save(npy_path, roi_ts)
print(f"  saved: {ts_csv_path}")
print(f"  saved: {npy_path}  shape={roi_ts.shape} dtype={roi_ts.dtype}")

hdr("STEP 8 -- ROI TIME-SERIES QC")
qc_rows = []
n_nan_rois = 0
n_inf_rois = 0
n_zero_var_rois = 0
n_low_var_rois = 0
all_stds = []
for rid in range(1, 247):
    col = roi_ts[:, rid - 1]
    nan_c = int(np.isnan(col).sum())
    inf_c = int(np.isinf(np.nan_to_num(col, nan=0.0, posinf=1e300, neginf=-1e300)) .sum()) if not np.isnan(col).all() else 0
    finite = col[np.isfinite(col)]
    if finite.size == 0:
        mean_, std_, min_, max_ = float("nan"), float("nan"), float("nan"), float("nan")
        zero_var = True
    else:
        mean_, std_, min_, max_ = float(finite.mean()), float(finite.std()), float(finite.min()), float(finite.max())
        zero_var = bool(std_ == 0.0)
    qc_rows.append({"ROI_ID": rid, "ROI_Name": lut.get(rid, f"ROI_{rid}"), "Mean": mean_, "Std": std_,
                    "Min": min_, "Max": max_, "NaN_Count": nan_c, "Inf_Count": inf_c, "Zero_Variance": zero_var})
    if nan_c > 0:
        n_nan_rois += 1
    if inf_c > 0:
        n_inf_rois += 1
    if zero_var:
        n_zero_var_rois += 1
    if finite.size > 0:
        all_stds.append(std_)

low_var_thr = np.percentile(all_stds, 1) if all_stds else 0
for row in qc_rows:
    if not row["Zero_Variance"] and np.isfinite(row["Std"]) and row["Std"] < low_var_thr:
        n_low_var_rois += 1

with open(os.path.join(OUT_DIR, "roi_timeseries_qc.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["ROI_ID", "ROI_Name", "Mean", "Std", "Min", "Max",
                                      "NaN_Count", "Inf_Count", "Zero_Variance"])
    w.writeheader(); w.writerows(qc_rows)

print(f"  ROIs with any NaN: {n_nan_rois}")
print(f"  ROIs with any Inf: {n_inf_rois}")
print(f"  ROIs with zero variance: {n_zero_var_rois}")
print(f"  ROIs with extremely low variance (<1st pct of nonzero SDs): {n_low_var_rois}")
print(f"  saved: {os.path.join(OUT_DIR, 'roi_timeseries_qc.csv')}")

hdr("STEP 9 -- MATRIX VALIDATION (independent re-check)")
mat_rows, mat_cols = roi_ts.shape
rows_match_volumes = mat_rows == T
cols_match_246 = mat_cols == 246
mat_has_nan = bool(np.isnan(roi_ts).any())
mat_has_inf = bool(np.isinf(roi_ts).any())
n_fully_empty_rois = int(sum(1 for r in qc_rows if np.isnan(r["Mean"])))
print(f"  matrix shape: {roi_ts.shape}")
print(f"  rows == BOLD volumes ({T}): {rows_match_volumes}")
print(f"  columns == 246: {cols_match_246}")
print(f"  matrix contains NaN: {mat_has_nan}  (n ROIs with NaN column: {n_nan_rois})")
print(f"  matrix contains Inf: {mat_has_inf}")
print(f"  completely empty ROIs (0 voxels -> NaN column): {n_fully_empty_rois}")

hdr("STEP 10 -- METADATA + REPORT")
metadata = {
    "subject": SUBJECT, "session": SESSION, "run": RUN, "group": GROUP,
    "atlas": "Brainnetome246", "atlas_source_resolution": "2mm", "target_resolution": "4mm",
    "interpolation": "nearest_neighbor",
    "roi_count_expected": 246, "roi_count_present": n_present,
    "bold_source": BOLD_PATH,
    "bold_dimensions": list(bold_img.shape), "bold_voxel_size_mm": list(bold_info["voxel_size_mm"]),
    "bold_TR_s": bold_info["TR_s"],
    "roi_timeseries_shape": list(roi_ts.shape),
    "zero_voxel_rois": n_zero, "rois_lt5_voxels": n_lt5, "rois_lt10_voxels": n_lt10,
    "nan_in_matrix": mat_has_nan, "inf_in_matrix": mat_has_inf, "zero_variance_rois": n_zero_var_rois,
    "software": {"python": platform.python_version(), "numpy": np.__version__,
                "nibabel": nib.__version__, "nilearn": __import__("nilearn").__version__,
                "pandas": pd.__version__},
    "processing_date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "grid_match_bold": grid_match,
    "labels_1_246_present_in_native_atlas": len(missing_labels) == 0 and len(extra_labels) == 0,
}
with open(os.path.join(OUT_DIR, "brainnetome_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
print(f"  saved: {os.path.join(OUT_DIR, 'brainnetome_metadata.json')}")

status = "PASS"
warnings = []
if n_zero > 0:
    warnings.append(f"{n_zero} zero-voxel ROI(s)")
if n_zero_var_rois > 0:
    warnings.append(f"{n_zero_var_rois} zero-variance ROI(s)")
if mat_has_nan:
    warnings.append(f"{n_fully_empty_rois} ROI(s) produced NaN columns (zero-voxel ROIs)")
if warnings:
    status = "PASS WITH WARNINGS"

rep_path = os.path.join(OUT_DIR, "brainnetome_qc_report.md")
with open(rep_path, "w", encoding="utf-8") as f:
    w = f.write
    w("# Brainnetome-246 QC\n\n## Input\n\n")
    w(f"- Subject: {SUBJECT}\n- Session: {SESSION}\n- Run: {RUN}\n- Group: {GROUP}\n")
    w(f"- BOLD path: `{BOLD_PATH}`\n- Atlas path: `{ATLAS_PATH}`\n\n")
    w("## Spatial Compatibility\n\n")
    w(f"- BOLD dimensions: {bold_img.shape}\n- Atlas original dimensions: {atlas_img.shape}\n")
    w(f"- Atlas aligned dimensions: {aligned_int.shape}\n- BOLD voxel size: {bold_info['voxel_size_mm']}\n")
    w(f"- Atlas target voxel size: {tuple(round(float(z),4) for z in aligned_int.header.get_zooms()[:3])}\n")
    w(f"- BOLD affine:\n```\n{bold_img.affine}\n```\n")
    w(f"- Aligned atlas affine:\n```\n{aligned_int.affine}\n```\n")
    w(f"- Grid match: {'YES' if grid_match else 'NO'}\n\n")
    w("## ROI Coverage\n\n")
    w(f"- Expected ROIs: 246\n- Present ROIs: {n_present}\n- Zero-voxel ROIs: {n_zero}\n")
    w(f"- ROIs <5 voxels: {n_lt5}\n- ROIs <10 voxels: {n_lt10}\n")
    w(f"- Minimum ROI size: {int(counts.min())}\n- Median ROI size: {float(np.median(counts))}\n"
      f"- Maximum ROI size: {int(counts.max())}\n\n")
    w("## ROI Time Series\n\n")
    w(f"- Matrix shape: {roi_ts.shape[0]} x {roi_ts.shape[1]}\n")
    w(f"- NaN values: {'YES (' + str(n_fully_empty_rois) + ' empty ROI columns)' if mat_has_nan else 'NO'}\n")
    w(f"- Inf values: {'YES' if mat_has_inf else 'NO'}\n- Zero-variance ROIs: {n_zero_var_rois}\n\n")
    w("## Status\n\n")
    w(f"**{status}**\n\n")
    if warnings:
        for wtext in warnings:
            w(f"- {wtext}\n")
    else:
        w("No warnings.\n")
print(f"  saved: {rep_path}")

hdr("FINAL REPORT")
print(f"SUBJECT:\n{SUBJECT}\n")
print(f"SESSION:\n{SESSION}\n")
print(f"RUN:\n{RUN}\n")
print(f"BOLD dimensions:\n{bold_img.shape}\n")
print(f"BOLD voxel size:\n{bold_info['voxel_size_mm']}\n")
print(f"BOLD volumes:\n{T}\n")
print(f"Brainnetome original resolution:\n{atlas_info['voxel_size_mm']} ({atlas_img.shape})\n")
print(f"Brainnetome aligned resolution:\n{tuple(round(float(z),4) for z in aligned_int.header.get_zooms()[:3])} ({aligned_int.shape})\n")
print(f"Atlas/BOLD grid match:\n{'YES' if grid_match else 'NO'}\n")
print(f"Interpolation:\nNearest Neighbor\n")
print(f"Brainnetome coverage:\n{n_present} / 246\n")
print(f"Zero-voxel ROIs:\n{n_zero}\n")
print(f"ROIs <5 voxels:\n{n_lt5}\n")
print(f"ROIs <10 voxels:\n{n_lt10}\n")
print(f"ROI matrix:\n{roi_ts.shape[0]} x {roi_ts.shape[1]}\n")
print(f"NaN:\n{n_fully_empty_rois} ROI column(s) ({mat_has_nan})\n")
print(f"Inf:\n{int(mat_has_inf)}\n")
print(f"Zero-variance ROIs:\n{n_zero_var_rois}\n")
print(f"Output directory:\n{OUT_DIR}\n")
print("BRAINNETOME_PILOT_DONE")
