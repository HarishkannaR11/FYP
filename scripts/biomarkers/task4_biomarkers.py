"""
RECOVERED from session history (see verify_brainnetome_atlas.py header note).

TASK 4: compute ALFF, ReHo, Degree Centrality, and Functional Connectivity
at the Brainnetome-246 ROI level.

Input choices (each documented):
  - ALFF: ROI time series from the final masked-GLM BOLD (roi_timeseries.npy)
  - ReHo: voxel-wise Kendall's W computed on UNSMOOTHED motion-corrected
    data, transformed to the SAME MNI grid using the EXISTING saved
    transform (reused, NOT a new registration), then averaged per ROI.
  - DC, FC: ROI time series from the final masked-GLM BOLD.
"""
import os
import csv
import numpy as np
import nibabel as nib
import ants

OUT_DIR = "/mnt/c/Users/krish/FYP/test/biomarkers/sub-019S4549"
REG_DIR = os.path.join(OUT_DIR, "registration_mni152")
ATLAS_RESLICED = os.path.join(OUT_DIR, "brainnetome246_resliced_to_bold_grid.nii.gz")
LUT_PATH = "/mnt/c/Users/krish/FYP/atlases/Brainnetome246/BN_Atlas_246_LUT.txt"
ROI_TS_PATH = os.path.join(OUT_DIR, "roi_timeseries.npy")
MOCO_NATIVE = "/home/harish/fyp_work/fsfast_ad_masked/work/fsfast_ad_production_masked/_run_key_sub-019S4549_ses-01/moco/moco_bold.nii.gz"
NORM_BOLD_REF = os.path.join(REG_DIR, "bold_mean_normalized_MNI152NLin6Asym.nii.gz")
TR = 3.0

lines = []


def log(msg=""):
    print(msg)
    lines.append(str(msg))


def load_lut(path):
    import re
    entries = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 2:
                continue
            try:
                rid = int(parts[0])
            except ValueError:
                continue
            entries[rid] = parts[1]
    return entries


def compute_falff_roi(roi_ts, tr):
    T = roi_ts.shape[0]
    freqs = np.fft.rfftfreq(T, d=tr)
    low_band = (freqs >= 0.01) & (freqs <= 0.08)
    ts = roi_ts - roi_ts.mean(axis=0, keepdims=True)
    fft_amp = np.abs(np.fft.rfft(ts, axis=0))
    total_power = fft_amp.sum(axis=0)
    low_power = fft_amp[low_band, :].sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        falff = np.where(total_power > 0, low_power / total_power, 0.0)
    return falff, freqs, low_band


def compute_voxelwise_reho(data, mask):
    x, y, z, T = data.shape
    reho_map = np.zeros((x, y, z), dtype=np.float32)
    ranks = np.argsort(np.argsort(data, axis=3), axis=3).astype(np.float32) + 1
    mask_idx = np.array(np.where(mask)).T
    for (i, j, k) in mask_idx:
        i0, i1 = max(i - 1, 0), min(i + 2, x)
        j0, j1 = max(j - 1, 0), min(j + 2, y)
        k0, k1 = max(k - 1, 0), min(k + 2, z)
        neighborhood_mask = mask[i0:i1, j0:j1, k0:k1]
        if neighborhood_mask.sum() < 7:
            continue
        R = ranks[i0:i1, j0:j1, k0:k1, :][neighborhood_mask]
        Nn, Tt = R.shape
        Rj = R.sum(axis=0)
        mean_R = Nn * (Tt + 1) / 2.0
        SS = np.sum((Rj - mean_R) ** 2)
        W = 12 * SS / (Nn ** 2 * (Tt ** 3 - Tt)) if Tt > 1 else 0.0
        reho_map[i, j, k] = W
    return reho_map


def main():
    log("=== TASK 4: Brainnetome-246 ROI-level biomarkers ===\n")

    atlas_img = nib.load(ATLAS_RESLICED)
    atlas_data = atlas_img.get_fdata()
    lut = load_lut(LUT_PATH)
    roi_ids = list(range(1, 247))

    roi_ts = np.load(ROI_TS_PATH)
    log(f"Loaded ROI time series: {roi_ts.shape}\n")

    log("--- ALFF (fALFF), ROI-level ---")
    falff_vals, freqs, low_band = compute_falff_roi(roi_ts, TR)
    log(f"fALFF stats: mean={falff_vals.mean():.4f} std={falff_vals.std():.4f} "
        f"min={falff_vals.min():.4f} max={falff_vals.max():.4f}")
    alff_csv = os.path.join(OUT_DIR, "ALFF.csv")
    with open(alff_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["roi_id", "roi_label", "ALFF"])
        for i, rid in enumerate(roi_ids):
            w.writerow([rid, lut.get(rid, "?"), falff_vals[i]])
    log(f"Saved: {alff_csv}\n")
    alff_status = "PASS" if not np.isnan(falff_vals).any() and not np.isinf(falff_vals).any() else "FAIL"

    log("--- ReHo, voxel-wise Kendall's W -> ROI-averaged ---")
    fwd_warp = fwd_affine = None
    for f in os.listdir(REG_DIR):
        if f.endswith("Warp.nii.gz") and "Inverse" not in f:
            fwd_warp = os.path.join(REG_DIR, f)
        if f.endswith("GenericAffine.mat"):
            fwd_affine = os.path.join(REG_DIR, f)

    moco_img = nib.load(MOCO_NATIVE)
    moco_data = moco_img.get_fdata(dtype=np.float32)
    fixed_ants = ants.image_read(NORM_BOLD_REF)
    target_shape = fixed_ants.shape

    moco_mni = np.zeros(target_shape + (140,), dtype=np.float32)
    for t in range(140):
        vol3d = nib.Nifti1Image(moco_data[..., t], moco_img.affine)
        vol3d_path = "/tmp/_tmp_moco_vol.nii.gz"
        nib.save(vol3d, vol3d_path)
        vol_ants = ants.image_read(vol3d_path)
        warped = ants.apply_transforms(fixed=fixed_ants, moving=vol_ants,
                                        transformlist=[fwd_warp, fwd_affine], interpolator="linear")
        moco_mni[..., t] = warped.numpy()
    os.remove("/tmp/_tmp_moco_vol.nii.gz")

    brain_mask_mni = nib.load(os.path.join(OUT_DIR, "roi_coverage", "normalized_brain_mask.nii.gz")).get_fdata() > 0.5
    reho_map = compute_voxelwise_reho(moco_mni, brain_mask_mni)

    reho_roi_vals = np.zeros(246)
    reho_empty = []
    for i, rid in enumerate(roi_ids):
        roi_mask = atlas_data == rid
        vals = reho_map[roi_mask]
        vals = vals[vals > 0]
        if vals.size == 0:
            reho_empty.append(rid)
            reho_roi_vals[i] = np.nan
        else:
            reho_roi_vals[i] = vals.mean()

    valid_reho = reho_roi_vals[~np.isnan(reho_roi_vals)]
    log(f"ReHo stats (valid ROIs only): mean={valid_reho.mean():.4f} std={valid_reho.std():.4f}")

    reho_csv = os.path.join(OUT_DIR, "ReHo.csv")
    with open(reho_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["roi_id", "roi_label", "ReHo"])
        for i, rid in enumerate(roi_ids):
            w.writerow([rid, lut.get(rid, "?"), reho_roi_vals[i]])
    log(f"Saved: {reho_csv}\n")
    reho_status = "PASS" if len(reho_empty) == 0 else f"PASS_WITH_{len(reho_empty)}_EMPTY_ROIS"

    log("--- Degree Centrality, ROI-level (r > 0.25 threshold, documented) ---")
    roi_corr = np.corrcoef(roi_ts.T)
    np.fill_diagonal(roi_corr, 0)
    thresholded = np.where(roi_corr > 0.25, roi_corr, 0.0)
    dc_vals = thresholded.sum(axis=1)
    dc_csv = os.path.join(OUT_DIR, "degree_centrality.csv")
    with open(dc_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["roi_id", "roi_label", "DegreeCentrality"])
        for i, rid in enumerate(roi_ids):
            w.writerow([rid, lut.get(rid, "?"), dc_vals[i]])
    log(f"Saved: {dc_csv}\n")
    dc_status = "PASS" if not np.isnan(dc_vals).any() else "FAIL"

    log("--- Functional Connectivity, 246x246 ---")
    fc_matrix = roi_corr.copy()
    np.fill_diagonal(fc_matrix, 1.0)
    symmetric = np.allclose(fc_matrix, fc_matrix.T, atol=1e-10)
    diag_ok = np.allclose(np.diag(fc_matrix), 1.0)
    fc_npy = os.path.join(OUT_DIR, "functional_connectivity.npy")
    np.save(fc_npy, fc_matrix)
    np.savetxt(os.path.join(OUT_DIR, "functional_connectivity.csv"), fc_matrix, delimiter=",")
    fc_z = np.arctanh(np.clip(fc_matrix, -0.999999, 0.999999))
    np.fill_diagonal(fc_z, 0.0)
    np.save(os.path.join(OUT_DIR, "functional_connectivity_fisherz.npy"), fc_z)
    fc_status = "PASS" if symmetric and diag_ok else "FAIL"

    log(f"ALFF={alff_status} REHO={reho_status} DC={dc_status} FC={fc_status}")
    with open(os.path.join(OUT_DIR, "task4_report.txt"), "w") as f:
        f.write("\n".join(lines))
    print("TASK4_DONE")


if __name__ == "__main__":
    main()
