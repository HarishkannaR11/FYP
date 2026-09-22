"""
RECOVERED from session history (see verify_brainnetome_atlas.py header note).

Reslice (nearest-neighbor ONLY, since labels are discrete/categorical) the
Brainnetome atlas onto the exact voxel grid of the normalized BOLD reference.
This is NOT a new registration -- both images already occupy the same
physical MNI152 space; this step only aligns array indexing.

Does NOT modify the original atlas file -- writes a new derived copy only.
"""
import nibabel as nib
from nilearn.image import resample_to_img

ATLAS = "/mnt/c/Users/krish/FYP/atlases/Brainnetome246/BN_Atlas_246_2mm.nii.gz"
NORM_BOLD = "/mnt/c/Users/krish/FYP/test/biomarkers/sub-019S4549/registration_mni152/bold_mean_normalized_MNI152NLin6Asym.nii.gz"
OUT = "/mnt/c/Users/krish/FYP/test/biomarkers/sub-019S4549/brainnetome246_resliced_to_bold_grid.nii.gz"

atlas_img = nib.load(ATLAS)
target_img = nib.load(NORM_BOLD)

resliced = resample_to_img(atlas_img, target_img, interpolation="nearest", force_resample=True, copy_header=True)
nib.save(resliced, OUT)

import numpy as np
orig_labels = set(int(round(v)) for v in np.unique(atlas_img.get_fdata()) if v != 0)
new_labels = set(int(round(v)) for v in np.unique(resliced.get_fdata()) if v != 0)

print(f"Original atlas: {len(orig_labels)} distinct nonzero labels")
print(f"Resliced atlas: {len(new_labels)} distinct nonzero labels")
print(f"Labels lost in reslicing: {sorted(orig_labels - new_labels)}")
print(f"Labels gained (should be none): {sorted(new_labels - orig_labels)}")
print(f"Resliced atlas shape: {resliced.shape}  affine matches target BOLD: "
      f"{np.allclose(resliced.affine, target_img.affine)}")
print(f"Saved: {OUT}")
