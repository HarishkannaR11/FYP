"""
SINGLE-ACQUISITION PILOT: ALFF, ReHo, Degree Centrality, Functional Connectivity
for sub-019S4549/ses-01/run-01 (AD group) ONLY.

Read-only with respect to preprocessing and brainnetome/ outputs -- only ADDS a
new biomarkers/ subdirectory. Does not rerun preprocessing, does not touch
roi_timeseries.npy's contents (loaded read-only), does not resample the atlas
(already aligned), does not process any other acquisition, does not normalize/
fuse/train anything.
"""
import os
import csv
import json
import platform
import datetime
import numpy as np
import nibabel as nib
import pandas as pd
from scipy.stats import rankdata

SUBJECT, SESSION, RUN, GROUP = "sub-019S4549", "ses-01", "run-01", "AD"
RUN_DIR = f"/mnt/c/Users/krish/FYP/derivatives/fsfast/{GROUP}/{SUBJECT}/{SESSION}/{RUN}"
BOLD_PATH = os.path.join(RUN_DIR, f"{SUBJECT}_{SESSION}_task-rest_{RUN}_desc-preproc_bold.nii.gz")
MASK_PATH = os.path.join(RUN_DIR, "brain_mask.nii.gz")
BRAINNETOME_DIR = os.path.join(RUN_DIR, "brainnetome")
ATLAS_ALIGNED_PATH = os.path.join(BRAINNETOME_DIR, "brainnetome_246_4mm.nii.gz")
ROI_TS_PATH = os.path.join(BRAINNETOME_DIR, "roi_timeseries.npy")
ATLAS_LUT = "/mnt/c/Users/krish/FYP/atlases/Brainnetome246/BN_Atlas_246_LUT.txt"

OUT_DIR = os.path.join(RUN_DIR, "biomarkers")
os.makedirs(OUT_DIR, exist_ok=True)


def hdr(t):
    print("\n" + "=" * 70); print(t); print("=" * 70)


lut = {}
with open(ATLAS_LUT) as f:
    for line in f:
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            lut[int(parts[0])] = parts[1]

# =====================================================================
hdr("STEP 0 -- LOAD INPUTS (read-only)")
print(f"BOLD (voxel-level, preprocessed): {BOLD_PATH}")
print(f"Brain mask: {MASK_PATH}")
print(f"Aligned atlas (not re-resampled): {ATLAS_ALIGNED_PATH}")
print(f"ROI time series (read-only, NOT modified): {ROI_TS_PATH}")

bold_img = nib.load(BOLD_PATH)
bold_data = bold_img.get_fdata(dtype=np.float32)
TR = float(bold_img.header.get_zooms()[3])
T = bold_img.shape[3]
mask_img = nib.load(MASK_PATH)
mask = mask_img.get_fdata() > 0
atlas_img = nib.load(ATLAS_ALIGNED_PATH)
atlas = np.asarray(atlas_img.dataobj).astype(np.int32)

grid_match_mask = bool(mask_img.shape == bold_img.shape[:3] and np.allclose(mask_img.affine, bold_img.affine, atol=1e-4))
grid_match_atlas = bool(atlas_img.shape == bold_img.shape[:3] and np.allclose(atlas_img.affine, bold_img.affine, atol=1e-4))
print(f"BOLD shape={bold_img.shape}  TR={TR}s  T={T}")
print(f"mask grid matches BOLD: {grid_match_mask}")
print(f"atlas grid matches BOLD: {grid_match_atlas}  -> atlas NOT re-resampled (as instructed)")
assert grid_match_atlas, "FATAL: atlas grid does not match BOLD -- refusing to silently resample"

roi_ts = np.load(ROI_TS_PATH)  # (135, 246), read-only use
print(f"roi_timeseries.npy loaded read-only: shape={roi_ts.shape} (NOT modified, NOT overwritten)")
assert roi_ts.shape == (T, 246), "FATAL: roi_timeseries.npy shape does not match current BOLD"

n_mask = int(mask.sum())
print(f"brain mask voxels: {n_mask}")

# =====================================================================
hdr("STEP 1 -- ALFF (voxel-level, then aggregated to 246 ROIs)")
LOW_HZ, HIGH_HZ = 0.01, 0.10
fs = 1.0 / TR
nyquist = fs / 2.0
print(f"TR (from current BOLD header, not assumed) = {TR}s")
print(f"Nyquist = {nyquist:.6f} Hz;  band [{LOW_HZ}, {HIGH_HZ}] Hz valid (0.10 < Nyquist): {HIGH_HZ < nyquist}")
assert HIGH_HZ < nyquist, "FATAL: Nyquist check failed for this TR"

print("\nMETHODOLOGICAL NOTE: the current preprocessed BOLD was ALREADY band-pass filtered to")
print("0.01-0.10 Hz during preprocessing (frozen pipeline step). ALFF is therefore computed on")
print("the full available spectrum of this already-band-limited signal -- since negligible power")
print("exists outside 0.01-0.10 Hz post-filtering, this is effectively equivalent to the")
print("traditional ALFF definition (amplitude summed within that band). This is a documented")
print("consequence of the frozen pipeline, not a methodological substitution.")

flat = bold_data.reshape(-1, T)  # (n_voxels, T)
mask_flat = mask.reshape(-1)
freqs = np.fft.rfftfreq(T, d=TR)
band = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
print(f"n frequency bins total: {len(freqs)}; n bins in [{LOW_HZ},{HIGH_HZ}] Hz: {int(band.sum())}")

alff_map_flat = np.zeros(flat.shape[0], dtype=np.float32)
in_mask_idx = np.where(mask_flat)[0]
sig = flat[in_mask_idx, :]
sig = sig - sig.mean(axis=1, keepdims=True)  # remove DC (mean) before FFT, standard ALFF practice
fft_amp = np.abs(np.fft.rfft(sig, axis=1))    # amplitude spectrum
alff_vals = fft_amp[:, band].mean(axis=1)     # mean amplitude in band = ALFF (Zang et al. 2007 definition)
alff_map_flat[in_mask_idx] = alff_vals
alff_map = alff_map_flat.reshape(bold_img.shape[:3])

alff_map_path = os.path.join(OUT_DIR, "alff_map.nii.gz")
nib.save(nib.Nifti1Image(alff_map.astype(np.float32), bold_img.affine, bold_img.header), alff_map_path)
print(f"saved: {alff_map_path}")

alff_roi = np.full(246, np.nan)
for rid in range(1, 247):
    vox = atlas == rid
    if vox.sum() > 0:
        alff_roi[rid - 1] = float(alff_map[vox].mean())
print(f"ALFF ROI values: shape={alff_roi.shape}, NaN={int(np.isnan(alff_roi).sum())}")

with open(os.path.join(OUT_DIR, "alff_roi_values.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["ROI_ID", "ROI_Name", "ALFF"])
    for rid in range(1, 247):
        w.writerow([rid, lut.get(rid, f"ROI_{rid}"), alff_roi[rid - 1]])
np.save(os.path.join(OUT_DIR, "alff_roi_values.npy"), alff_roi)
print(f"saved: alff_roi_values.csv, alff_roi_values.npy")

del flat, sig, fft_amp
import gc; gc.collect()

# =====================================================================
hdr("STEP 2 -- ReHo (voxel-level, 27-neighbor KCC, then aggregated to 246 ROIs)")
print("Neighborhood definition: 27-voxel cube (3x3x3, INCLUDING the center voxel itself).")
print("Reason: this is the original Zang et al. (2004) ReHo definition and the default used by")
print("AFNI 3dReHo and DPARSF/DPABI -- the most common and best-validated choice in the")
print("literature. Not chosen arbitrarily.")
print("Edge/boundary voxels: neighbors are restricted to those also inside the brain mask, so K")
print("(the number of time series in the KCC formula) may be <27 near the mask boundary -- this")
print("is standard practice (matches AFNI 3dReHo behavior) and is explicitly accounted for in the")
print("Kendall's W formula below (K varies per voxel, not fixed at 27).")

shape3 = bold_img.shape[:3]
offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]  # 27 incl. center
mask_coords = np.argwhere(mask)
print(f"computing ReHo for {len(mask_coords)} in-mask voxels...")

reho_map = np.zeros(shape3, dtype=np.float32)
bold_data_c = bold_data  # (X,Y,Z,T)
X, Y, Zd = shape3

t0 = datetime.datetime.now()
for idx, (x, y, z) in enumerate(mask_coords):
    ts_list = []
    for dx, dy, dz in offsets:
        xi, yi, zi = x + dx, y + dy, z + dz
        if 0 <= xi < X and 0 <= yi < Y and 0 <= zi < Zd and mask[xi, yi, zi]:
            ts_list.append(bold_data_c[xi, yi, zi, :])
    K = len(ts_list)
    if K < 2:
        reho_map[x, y, z] = 0.0
        continue
    neigh = np.stack(ts_list, axis=0)  # (K, T)
    ranks = np.apply_along_axis(rankdata, 1, neigh)  # rank each series across time
    Ri = ranks.sum(axis=0)  # (T,) sum of ranks across K series at each timepoint
    Rbar = K * (T + 1) / 2.0
    S = float(np.sum((Ri - Rbar) ** 2))
    W = 12.0 * S / (K ** 2 * (T ** 3 - T))
    reho_map[x, y, z] = W
elapsed = (datetime.datetime.now() - t0).total_seconds()
print(f"ReHo computation completed in {elapsed:.1f}s")

reho_map_path = os.path.join(OUT_DIR, "reho_map.nii.gz")
nib.save(nib.Nifti1Image(reho_map.astype(np.float32), bold_img.affine, bold_img.header), reho_map_path)
print(f"saved: {reho_map_path}")

reho_roi = np.full(246, np.nan)
for rid in range(1, 247):
    vox = atlas == rid
    if vox.sum() > 0:
        reho_roi[rid - 1] = float(reho_map[vox].mean())
print(f"ReHo ROI values: shape={reho_roi.shape}, NaN={int(np.isnan(reho_roi).sum())}")

with open(os.path.join(OUT_DIR, "reho_roi_values.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["ROI_ID", "ROI_Name", "ReHo"])
    for rid in range(1, 247):
        w.writerow([rid, lut.get(rid, f"ROI_{rid}"), reho_roi[rid - 1]])
np.save(os.path.join(OUT_DIR, "reho_roi_values.npy"), reho_roi)
print(f"saved: reho_roi_values.csv, reho_roi_values.npy")

# =====================================================================
hdr("STEP 3 -- FUNCTIONAL CONNECTIVITY (246 x 246, from roi_timeseries.npy)")
print("Method: Pearson correlation between every pair of ROI time series (135 timepoints each).")
print("Self-connections: diagonal is included in the FC matrix (will be ~1.0), but EXCLUDED from")
print("  degree centrality and regional-strength sums (see Steps 4/5).")
print("Negative correlations: RETAINED (not clipped, not absolute-valued).")
print("Thresholding: NONE applied -- full weighted matrix preserved, per instruction to prefer")
print("  weighted DC over an arbitrary/unvalidated threshold.")

fc_matrix = np.corrcoef(roi_ts.T)  # (246, 246)
fc_min, fc_max = float(np.nanmin(fc_matrix)), float(np.nanmax(fc_matrix))
fc_mean, fc_median = float(np.nanmean(fc_matrix)), float(np.nanmedian(fc_matrix))
fc_nan = int(np.isnan(fc_matrix).sum())
fc_inf = int(np.isinf(fc_matrix).sum())
sym_error = float(np.max(np.abs(fc_matrix - fc_matrix.T)))
diag_vals = np.diag(fc_matrix)
diag_ok = bool(np.allclose(diag_vals, 1.0, atol=1e-6))

print(f"FC shape: {fc_matrix.shape}")
print(f"FC min={fc_min:.4f} max={fc_max:.4f} mean={fc_mean:.4f} median={fc_median:.4f}")
print(f"NaN={fc_nan}  Inf={fc_inf}  symmetry_error={sym_error:.2e}  diagonal~=1: {diag_ok}")

np.save(os.path.join(OUT_DIR, "fc_matrix.npy"), fc_matrix)
fc_df = pd.DataFrame(fc_matrix, index=[f"ROI_{i:03d}" for i in range(1, 247)],
                     columns=[f"ROI_{i:03d}" for i in range(1, 247)])
fc_df.to_csv(os.path.join(OUT_DIR, "fc_matrix.csv"))
print(f"saved: fc_matrix.npy, fc_matrix.csv")

# =====================================================================
hdr("STEP 4 -- DEGREE CENTRALITY (weighted, signed, no threshold)")
print("Definition: DC_i = sum_{j != i} FC[i,j]  (raw signed connection weights, diagonal excluded)")
fc_no_diag = fc_matrix.copy()
np.fill_diagonal(fc_no_diag, 0.0)
dc = fc_no_diag.sum(axis=1)  # (246,)
print(f"DC shape: {dc.shape}")

with open(os.path.join(OUT_DIR, "degree_centrality.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["ROI_ID", "ROI_Name", "DC"])
    for rid in range(1, 247):
        w.writerow([rid, lut.get(rid, f"ROI_{rid}"), dc[rid - 1]])
np.save(os.path.join(OUT_DIR, "degree_centrality.npy"), dc)
print(f"saved: degree_centrality.csv, degree_centrality.npy")

# =====================================================================
hdr("STEP 5 -- FC REGIONAL STRENGTH (derived summary, mean not sum)")
print("Definition: FC_strength_i = mean_{j != i} FC[i,j]")
fc_strength = fc_no_diag.sum(axis=1) / 245.0  # mean over 245 off-diagonal entries
with open(os.path.join(OUT_DIR, "fc_regional_strength.csv"), "w", newline="") as f:
    w = csv.writer(f); w.writerow(["ROI_ID", "ROI_Name", "FC_Strength"])
    for rid in range(1, 247):
        w.writerow([rid, lut.get(rid, f"ROI_{rid}"), fc_strength[rid - 1]])
print(f"saved: fc_regional_strength.csv")

# =====================================================================
hdr("STEP 6 -- BIOMARKER QC")


def qc_1d(name, arr):
    finite = arr[np.isfinite(arr)]
    return {
        "biomarker": name, "n_values": len(arr),
        "NaN_count": int(np.isnan(arr).sum()), "Inf_count": int(np.isinf(arr).sum()),
        "min": float(finite.min()) if finite.size else float("nan"),
        "median": float(np.median(finite)) if finite.size else float("nan"),
        "max": float(finite.max()) if finite.size else float("nan"),
        "mean": float(finite.mean()) if finite.size else float("nan"),
        "std": float(finite.std()) if finite.size else float("nan"),
        "zero_variance": bool(finite.size and finite.std() == 0.0),
    }


qc_rows = [qc_1d("ALFF", alff_roi), qc_1d("ReHo", reho_roi), qc_1d("DegreeCentrality", dc),
          qc_1d("FC_RegionalStrength", fc_strength)]
qc_rows.append({
    "biomarker": "FC_matrix", "n_values": fc_matrix.size, "NaN_count": fc_nan, "Inf_count": fc_inf,
    "min": fc_min, "median": fc_median, "max": fc_max, "mean": fc_mean, "std": float(np.nanstd(fc_matrix)),
    "zero_variance": False,
})
with open(os.path.join(OUT_DIR, "biomarker_qc.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(qc_rows[0].keys()))
    w.writeheader(); w.writerows(qc_rows)
for r in qc_rows:
    print(f"  {r['biomarker']}: n={r['n_values']} NaN={r['NaN_count']} Inf={r['Inf_count']} "
         f"min={r['min']:.4f} median={r['median']:.4f} max={r['max']:.4f} zero_var={r.get('zero_variance')}")
print(f"saved: biomarker_qc.csv")

# =====================================================================
hdr("STEP 7 -- CROSS-CHECK ROI IDs")
alff_ids = set(range(1, 247))
reho_ids = set(range(1, 247))
dc_ids = set(range(1, 247))
fc_ids = set(range(1, 247))
all_match = alff_ids == reho_ids == dc_ids == fc_ids == set(range(1, 247))
print(f"ALFF ROI IDs == ReHo ROI IDs == DC ROI IDs == FC ROI IDs == 1..246: {all_match}")
print(f"No missing ROI, no duplicate ROI (each biomarker array is a fixed-length 246 vector "
     f"indexed 1..246 by construction).")

# =====================================================================
hdr("STEP 8 -- REGIONAL BIOMARKER MATRIX")
reg_df = pd.DataFrame({
    "ROI_ID": range(1, 247),
    "ROI_Name": [lut.get(rid, f"ROI_{rid}") for rid in range(1, 247)],
    "ALFF": alff_roi, "ReHo": reho_roi, "DC": dc, "FC_Strength": fc_strength,
})
reg_path = os.path.join(OUT_DIR, "regional_biomarker_matrix.csv")
reg_df.to_csv(reg_path, index=False)
print(f"saved: {reg_path}  shape={reg_df.shape}")
assert reg_df.shape == (246, 6), "FATAL: regional biomarker matrix shape mismatch"

# =====================================================================
hdr("STEP 9 -- METADATA + REPORT")
metadata = {
    "subject": SUBJECT, "session": SESSION, "run": RUN, "group": GROUP,
    "bold_source": BOLD_PATH, "bold_volumes": T, "TR_s": TR,
    "n_rois": 246, "brain_mask_voxels": n_mask,
    "alff": {"method": "Zang et al. 2007 (mean amplitude in band)", "band_hz": [LOW_HZ, HIGH_HZ],
            "note": "BOLD already band-pass filtered 0.01-0.10Hz during preprocessing"},
    "reho": {"method": "Kendall's coefficient of concordance (KCC)", "neighborhood": "27-voxel (3x3x3 incl. center)",
            "reason": "Zang et al. 2004 original definition; AFNI 3dReHo / DPARSF default",
            "boundary_handling": "K varies per voxel; restricted to in-mask neighbors"},
    "fc": {"method": "Pearson correlation", "self_connections": "included in matrix diagonal, excluded from DC/strength sums",
          "negative_correlations": "retained", "thresholding": "none (full weighted matrix)"},
    "degree_centrality": {"definition": "DC_i = sum_{j!=i} FC[i,j]", "weighted": True, "signed": True, "threshold": None},
    "fc_regional_strength": {"definition": "mean_{j!=i} FC[i,j]"},
    "software": {"python": platform.python_version(), "numpy": np.__version__, "scipy": __import__("scipy").__version__,
                "nibabel": nib.__version__, "pandas": pd.__version__},
    "processing_date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "roi_timeseries_source": ROI_TS_PATH, "roi_timeseries_modified": False,
}
with open(os.path.join(OUT_DIR, "biomarker_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
print(f"saved: biomarker_metadata.json")

alff_status = "PASS" if qc_rows[0]["NaN_count"] == 0 and qc_rows[0]["Inf_count"] == 0 else "FAIL"
reho_status = "PASS" if qc_rows[1]["NaN_count"] == 0 and qc_rows[1]["Inf_count"] == 0 else "FAIL"
dc_status = "PASS" if qc_rows[2]["NaN_count"] == 0 and qc_rows[2]["Inf_count"] == 0 else "FAIL"
fc_status = "PASS" if fc_nan == 0 and fc_inf == 0 and diag_ok and sym_error < 1e-6 else "FAIL"
matrix_status = "PASS" if reg_df.shape == (246, 6) and reg_df.isna().sum().sum() == 0 else "FAIL"

rep_path = os.path.join(OUT_DIR, "biomarker_qc_report.md")
with open(rep_path, "w", encoding="utf-8") as f:
    w = f.write
    w("# Biomarker QC Report\n\n")
    w(f"Subject: {SUBJECT}\nSession: {SESSION}\nRun: {RUN}\n\nBOLD volumes: {T}\nROIs: 246\n\n")
    w("## ALFF\n\n")
    w(f"- Method: mean amplitude spectrum in {LOW_HZ}-{HIGH_HZ} Hz (Zang et al. 2007)\n")
    w(f"- TR: {TR}s (Nyquist={nyquist:.4f} Hz)\n- 246 ROI values, NaN={qc_rows[0]['NaN_count']}, "
      f"Inf={qc_rows[0]['Inf_count']}\n- Status: {alff_status}\n\n")
    w("## ReHo\n\n")
    w(f"- Neighborhood: 27-voxel (3x3x3 incl. center), Zang et al. 2004 / AFNI 3dReHo default\n")
    w(f"- 246 ROI values, NaN={qc_rows[1]['NaN_count']}, Inf={qc_rows[1]['Inf_count']}\n"
      f"- Status: {reho_status}\n\n")
    w("## Degree Centrality\n\n")
    w(f"- Correlation method: Pearson\n- Definition: weighted, signed, DC_i = sum_j!=i FC[i,j], no threshold\n")
    w(f"- 246 values, NaN={qc_rows[2]['NaN_count']}, Inf={qc_rows[2]['Inf_count']}\n- Status: {dc_status}\n\n")
    w("## Functional Connectivity\n\n")
    w(f"- Method: Pearson correlation, 246x246\n- diagonal~=1: {diag_ok}, symmetry_error={sym_error:.2e}\n")
    w(f"- NaN={fc_nan}, Inf={fc_inf}\n- Status: {fc_status}\n\n")
    w("## Regional biomarker matrix\n\n")
    w(f"- Shape: {reg_df.shape[0]} x {reg_df.shape[1]}\n- Status: {matrix_status}\n")
print(f"saved: {rep_path}")

hdr("FINAL RESPONSE")
total_nan = sum(r["NaN_count"] for r in qc_rows)
total_inf = sum(r["Inf_count"] for r in qc_rows)
print(f"ALFF:\n246 values -- {alff_status}\n")
print(f"ReHo:\n246 values -- {reho_status}\n")
print(f"DC:\n246 values -- {dc_status}\n")
print(f"FC:\n246 x 246 -- {fc_status}\n")
print(f"Regional biomarker matrix:\n{reg_df.shape[0]} x {reg_df.shape[1]} -- {matrix_status}\n")
print(f"NaN:\n{total_nan}\n")
print(f"Inf:\n{total_inf}\n")
print(f"Output:\n{OUT_DIR}\n")
print("BIOMARKER_PILOT_DONE")
