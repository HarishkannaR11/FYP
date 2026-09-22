"""
RECOVERED from session history (see verify_brainnetome_atlas.py header note).

TASK 2: Brainnetome ROI coverage QC. For all 246 ROIs, compute atlas voxel
count, intersection with the valid BOLD/brain mask (normalized via the SAME
saved transforms from registration_mni152/, nearest-neighbor since it's
binary), and coverage percentage. Flags empty/suspiciously-low-coverage ROIs.
"""
import os
import csv
import re
import numpy as np
import nibabel as nib
import ants

NATIVE_MASK = ("/mnt/c/Users/krish/FYP/derivatives/fsfast/AD/sub-019S4549/ses-01/func/"
               "sub-019S4549_ses-01_task-rest_run-01_desc-brain_mask.nii.gz")
NORM_BOLD = "/mnt/c/Users/krish/FYP/test/biomarkers/sub-019S4549/registration_mni152/bold_mean_normalized_MNI152NLin6Asym.nii.gz"
ATLAS_RESLICED = "/mnt/c/Users/krish/FYP/test/biomarkers/sub-019S4549/brainnetome246_resliced_to_bold_grid.nii.gz"
LUT_PATH = "/mnt/c/Users/krish/FYP/atlases/Brainnetome246/BN_Atlas_246_LUT.txt"

REG_DIR = "/mnt/c/Users/krish/FYP/test/biomarkers/sub-019S4549/registration_mni152"
OUT_DIR = "/mnt/c/Users/krish/FYP/test/biomarkers/sub-019S4549/roi_coverage"
os.makedirs(OUT_DIR, exist_ok=True)

lines = []


def log(msg=""):
    print(msg)
    lines.append(str(msg))


def load_lut(path):
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
    log("=== TASK 2: Brainnetome ROI coverage QC ===\n")

    fwd_warp = fwd_affine = None
    for f in os.listdir(REG_DIR):
        if f.endswith("Warp.nii.gz") and "Inverse" not in f:
            fwd_warp = os.path.join(REG_DIR, f)
        if f.endswith("GenericAffine.mat"):
            fwd_affine = os.path.join(REG_DIR, f)
    log(f"Forward warp: {fwd_warp}")
    log(f"Forward affine: {fwd_affine}")
    assert fwd_warp and fwd_affine, "could not locate saved transforms"

    log("\n--- Normalizing native brain mask via saved transforms (nearest-neighbor) ---")
    mask_ants = ants.image_read(NATIVE_MASK)
    fixed_ants = ants.image_read(NORM_BOLD)
    norm_mask = ants.apply_transforms(fixed=fixed_ants, moving=mask_ants,
                                       transformlist=[fwd_warp, fwd_affine],
                                       interpolator="nearestNeighbor")
    norm_mask_path = os.path.join(OUT_DIR, "normalized_brain_mask.nii.gz")
    ants.image_write(norm_mask, norm_mask_path)
    norm_mask_data = norm_mask.numpy() > 0.5
    log(f"Normalized brain mask saved: {norm_mask_path}")
    log(f"Normalized mask voxel count: {int(norm_mask_data.sum())} / {norm_mask_data.size} "
        f"({100*norm_mask_data.sum()/norm_mask_data.size:.2f}%)\n")

    atlas_img = nib.load(ATLAS_RESLICED)
    atlas_data = atlas_img.get_fdata()
    lut = load_lut(LUT_PATH)

    assert atlas_data.shape == norm_mask_data.shape, "atlas/mask shape mismatch after reslicing"

    log("--- Per-ROI coverage ---")
    rows = []
    empty_rois = []
    low_coverage_rois = []
    for rid in range(1, 247):
        roi_mask = atlas_data == rid
        roi_voxels = int(roi_mask.sum())
        intersect = int((roi_mask & norm_mask_data).sum())
        coverage_pct = 100.0 * intersect / roi_voxels if roi_voxels > 0 else 0.0
        label = lut.get(rid, "MISSING_LUT_ENTRY")

        if roi_voxels == 0:
            empty_rois.append(rid)
        elif coverage_pct < 20.0:
            low_coverage_rois.append((rid, label, coverage_pct))

        rows.append({"roi_id": rid, "roi_label": label, "atlas_voxel_count": roi_voxels,
                      "intersect_with_brain_mask": intersect, "coverage_pct": round(coverage_pct, 2)})

    log(f"Total ROIs checked: {len(rows)}")
    log(f"Empty ROIs (0 atlas voxels): {empty_rois if empty_rois else 'none'}")
    log(f"Suspiciously low-coverage ROIs (<20% intersection with brain mask): {len(low_coverage_rois)}")
    for rid, label, pct in low_coverage_rois:
        log(f"  ROI {rid} ({label}): {pct:.1f}% coverage")
    log("")

    coverage_vals = np.array([r["coverage_pct"] for r in rows])
    log(f"Coverage stats across all 246 ROIs: mean={coverage_vals.mean():.1f}% "
        f"median={np.median(coverage_vals):.1f}% min={coverage_vals.min():.1f}% max={coverage_vals.max():.1f}%")

    csv_path = os.path.join(OUT_DIR, "roi_coverage.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    log(f"\nSaved: {csv_path}")

    txt_path = os.path.join(OUT_DIR, "roi_coverage_summary.txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Saved: {txt_path}")
    print(f"\nEMPTY_ROIS={len(empty_rois)}  LOW_COVERAGE_ROIS={len(low_coverage_rois)}")


if __name__ == "__main__":
    main()
