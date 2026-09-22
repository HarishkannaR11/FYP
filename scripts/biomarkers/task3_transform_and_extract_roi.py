"""
RECOVERED from session history (see verify_brainnetome_atlas.py header note).

TASK 3: apply the SAME saved transforms (from registration_mni152/) to the
full 4D final masked-GLM BOLD, preserving 140 timepoints/TR=3.0s/temporal
order. Then extract one time series per Brainnetome-246 ROI using the
resliced atlas. Does NOT re-run registration.
"""
import os
import csv
import numpy as np
import nibabel as nib
import ants

INPUT_4D_BOLD = ("/mnt/c/Users/krish/FYP/derivatives/fsfast/AD/sub-019S4549/ses-01/func/"
                  "sub-019S4549_ses-01_task-rest_run-01_desc-preproc_bold.nii.gz")
NORM_BOLD_REF = "/mnt/c/Users/krish/FYP/test/biomarkers/sub-019S4549/registration_mni152/bold_mean_normalized_MNI152NLin6Asym.nii.gz"
ATLAS_RESLICED = "/mnt/c/Users/krish/FYP/test/biomarkers/sub-019S4549/brainnetome246_resliced_to_bold_grid.nii.gz"
LUT_PATH = "/mnt/c/Users/krish/FYP/atlases/Brainnetome246/BN_Atlas_246_LUT.txt"
REG_DIR = "/mnt/c/Users/krish/FYP/test/biomarkers/sub-019S4549/registration_mni152"

OUT_DIR = "/mnt/c/Users/krish/FYP/test/biomarkers/sub-019S4549"
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


def main():
    log("=== TASK 3: Transform 4D BOLD to MNI + extract ROI time series ===\n")

    fwd_warp = fwd_affine = None
    for f in os.listdir(REG_DIR):
        if f.endswith("Warp.nii.gz") and "Inverse" not in f:
            fwd_warp = os.path.join(REG_DIR, f)
        if f.endswith("GenericAffine.mat"):
            fwd_affine = os.path.join(REG_DIR, f)
    log(f"Reusing existing transforms (NOT re-registering): {fwd_warp}, {fwd_affine}\n")

    orig_img = nib.load(INPUT_4D_BOLD)
    orig_shape = orig_img.shape
    orig_tr = float(orig_img.header.get_zooms()[3])
    log(f"Input 4D BOLD: {INPUT_4D_BOLD}")
    log(f"  original shape={orig_shape}  TR={orig_tr}s\n")
    assert orig_shape[3] == 140, f"expected 140 timepoints, got {orig_shape[3]}"
    assert abs(orig_tr - TR) < 0.01, f"expected TR={TR}, got {orig_tr}"

    log("--- Applying saved transform to each of the 140 volumes (temporal order preserved) ---")
    orig_data = orig_img.get_fdata(dtype=np.float32)
    fixed_ants = ants.image_read(NORM_BOLD_REF)
    target_shape = fixed_ants.shape

    normalized_4d = np.zeros(target_shape + (140,), dtype=np.float32)
    for t in range(140):
        vol3d = nib.Nifti1Image(orig_data[..., t], orig_img.affine)
        vol3d_path = "/tmp/_tmp_vol.nii.gz"
        nib.save(vol3d, vol3d_path)
        vol_ants = ants.image_read(vol3d_path)
        warped = ants.apply_transforms(fixed=fixed_ants, moving=vol_ants,
                                        transformlist=[fwd_warp, fwd_affine],
                                        interpolator="linear")
        normalized_4d[..., t] = warped.numpy()
        if (t + 1) % 20 == 0 or t == 0:
            log(f"  volume {t+1}/140 done")
    os.remove("/tmp/_tmp_vol.nii.gz")

    bold_mni_img = nib.Nifti1Image(normalized_4d, nib.load(NORM_BOLD_REF).affine)
    bold_mni_header = bold_mni_img.header
    bold_mni_header.set_zooms((2.0, 2.0, 2.0, TR))
    bold_mni_path = os.path.join(OUT_DIR, "bold_mni.nii.gz")
    nib.save(bold_mni_img, bold_mni_path)

    log(f"\nSaved 4D normalized BOLD: {bold_mni_path}")
    reload_check = nib.load(bold_mni_path)
    log(f"Verification -- shape: {reload_check.shape}  TR: {reload_check.header.get_zooms()[3]}s")
    assert reload_check.shape[3] == 140, "timepoints not preserved!"
    assert abs(float(reload_check.header.get_zooms()[3]) - TR) < 0.01, "TR not preserved!"
    log("140 timepoints preserved: True")
    log("TR=3.0s preserved: True")
    log("Temporal order preserved: True\n")

    log("--- ROI time-series extraction (NiBabel + manual masked averaging) ---")
    atlas_img = nib.load(ATLAS_RESLICED)
    atlas_data = atlas_img.get_fdata()
    lut = load_lut(LUT_PATH)

    assert atlas_data.shape == normalized_4d.shape[:3], "atlas/BOLD grid mismatch"

    roi_ids = list(range(1, 247))
    timeseries = np.zeros((140, 246), dtype=np.float64)
    empty_rois = []
    for i, rid in enumerate(roi_ids):
        roi_mask = atlas_data == rid
        n_vox = int(roi_mask.sum())
        if n_vox == 0:
            empty_rois.append(rid)
            log(f"STOP CONDITION: ROI {rid} ({lut.get(rid,'?')}) has ZERO voxels")
            continue
        roi_voxel_ts = normalized_4d[roi_mask, :]
        timeseries[:, i] = roi_voxel_ts.mean(axis=0)

    if empty_rois:
        log(f"\nSTOPPING: {len(empty_rois)} empty ROI(s) found: {empty_rois}.")
        with open(os.path.join(OUT_DIR, "task3_report.txt"), "w") as f:
            f.write("\n".join(lines))
        print("TASK3_STOPPED_EMPTY_ROIS")
        return

    log(f"\nROI time-series shape: {timeseries.shape} (expected: (140, 246))")
    assert timeseries.shape == (140, 246), "unexpected shape"

    n_nan = int(np.isnan(timeseries).sum())
    n_inf = int(np.isinf(timeseries).sum())
    log(f"NaN count: {n_nan}   Inf count: {n_inf}")
    log(f"Valid ROIs: {246 - len(empty_rois)} / 246   Empty ROIs: {len(empty_rois)}\n")

    npy_path = os.path.join(OUT_DIR, "roi_timeseries.npy")
    np.save(npy_path, timeseries)
    log(f"Saved: {npy_path}")

    csv_path = os.path.join(OUT_DIR, "roi_timeseries.csv")
    header = ["timepoint"] + [f"ROI_{rid}_{lut.get(rid,'?')}" for rid in roi_ids]
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for t in range(140):
            w.writerow([t] + timeseries[t, :].tolist())
    log(f"Saved: {csv_path}")

    meta_path = os.path.join(OUT_DIR, "roi_timeseries_metadata.txt")
    with open(meta_path, "w") as f:
        f.write(f"Subject: sub-019S4549 ses-01 run-01\n")
        f.write(f"Input 4D BOLD (native): {INPUT_4D_BOLD}\n")
        f.write(f"Normalized 4D BOLD (MNI152NLin6Asym grid): {bold_mni_path}\n")
        f.write(f"Atlas (resliced to BOLD grid, nearest-neighbor): {ATLAS_RESLICED}\n")
        f.write(f"Timepoints: 140   TR: {TR}s\n")
        f.write(f"Valid ROIs: {246 - len(empty_rois)} / 246\n")
        f.write(f"Empty ROIs: {empty_rois}\n")
        f.write(f"NaN count: {n_nan}   Inf count: {n_inf}\n")

    with open(os.path.join(OUT_DIR, "task3_report.txt"), "w") as f:
        f.write("\n".join(lines))
    print("TASK3_DONE")


if __name__ == "__main__":
    main()
