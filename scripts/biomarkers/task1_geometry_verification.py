"""
RECOVERED from session history (see verify_brainnetome_atlas.py header note).

TASK 1: verify normalized BOLD reference vs Brainnetome atlas geometry via
actual physical/world-coordinate mapping, not just orientation labels.
Does NOT modify/resample the atlas. Read-only.
"""
import numpy as np
import nibabel as nib

NORM_BOLD = "/mnt/c/Users/krish/FYP/test/biomarkers/sub-019S4549/registration_mni152/bold_mean_normalized_MNI152NLin6Asym.nii.gz"
ATLAS = "/mnt/c/Users/krish/FYP/atlases/Brainnetome246/BN_Atlas_246_2mm.nii.gz"

OUT = "/mnt/c/Users/krish/FYP/test/biomarkers/sub-019S4549/task1_geometry_verification.txt"

lines = []


def log(msg=""):
    print(msg)
    lines.append(str(msg))


def main():
    log("=== TASK 1: Atlas/BOLD geometry verification ===\n")

    bold_img = nib.load(NORM_BOLD)
    atlas_img = nib.load(ATLAS)

    log(f"Normalized BOLD: {NORM_BOLD}")
    log(f"  shape={bold_img.shape}  voxel={bold_img.header.get_zooms()}")
    log(f"  orientation={''.join(nib.aff2axcodes(bold_img.affine))}")
    log(f"  affine:\n{bold_img.affine}\n")

    log(f"Brainnetome atlas: {ATLAS}")
    log(f"  shape={atlas_img.shape}  voxel={atlas_img.header.get_zooms()}")
    log(f"  orientation={''.join(nib.aff2axcodes(atlas_img.affine))}")
    log(f"  affine:\n{atlas_img.affine}\n")

    log("--- True physical/world-coordinate mapping check ---")
    composite = np.linalg.inv(bold_img.affine) @ atlas_img.affine
    log(f"Composite mapping (atlas voxel -> BOLD voxel), should be ~identity if grids align:\n{composite}\n")

    identity = np.eye(4)
    max_dev = np.abs(composite - identity).max()
    log(f"Max deviation from identity: {max_dev:.6f}")
    grids_aligned = max_dev < 1e-3
    log(f"Atlas and normalized-BOLD share the EXACT SAME voxel grid (same physical space, "
        f"voxel-for-voxel): {grids_aligned}\n")

    log("--- Sample voxel-to-world coordinate checks (spot check) ---")
    sample_idx = [(45, 54, 45), (10, 10, 10), (80, 100, 80)]
    for idx in sample_idx:
        atlas_world = atlas_img.affine @ np.array([*idx, 1])
        bold_world_via_same_idx = bold_img.affine @ np.array([*idx, 1])
        log(f"  Atlas voxel {idx} -> world (mm) = {atlas_world[:3].round(2)}")
        log(f"  BOLD  voxel {idx} -> world (mm) = {bold_world_via_same_idx[:3].round(2)}")
        log(f"  Match: {np.allclose(atlas_world, bold_world_via_same_idx, atol=1e-2)}\n")

    log("--- Nonzero atlas labels ---")
    atlas_data = atlas_img.get_fdata()
    unique_labels = np.unique(atlas_data)
    nonzero = sorted(int(round(v)) for v in unique_labels if v != 0)
    log(f"Distinct nonzero labels present: {len(nonzero)}")
    log(f"Range: {min(nonzero)} - {max(nonzero)}")
    log(f"All 246 expected present: {len(nonzero) == 246 and nonzero == list(range(1, 247))}\n")

    log("=== CONCLUSION ===")
    if grids_aligned:
        log("Atlas and normalized BOLD reference occupy the IDENTICAL voxel grid. No resampling")
        log("is needed to use the atlas directly with this normalized BOLD data.")
    else:
        log("Atlas and normalized BOLD do NOT share the identical voxel grid despite matching")
        log("dimensions/voxel size -- see composite matrix above for the actual discrepancy.")
        log("Resampling would be required (nearest-neighbor only, atlas untouched as source).")

    with open(OUT, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport: {OUT}")
    print(f"GRIDS_ALIGNED={grids_aligned}")


if __name__ == "__main__":
    main()
