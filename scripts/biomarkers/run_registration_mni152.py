"""
RECOVERED from session history (see verify_brainnetome_atlas.py header note).

Repeat BOLD-mean -> MNI registration using a template that matches the
Brainnetome atlas's native grid (91x109x91, 2mm) -- MNI152NLin6Asym via
TemplateFlow, the standard "FSL MNI152" space, NOT nilearn's ICBM152 2009a.
Does NOT touch the Brainnetome atlas.
"""
import os
import time
import shutil
import numpy as np
import nibabel as nib
import ants
import templateflow.api as tflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BOLD_MEAN = "/mnt/c/Users/krish/FYP/test/biomarkers/sub-019S4549/bold_mean.nii.gz"
ATLAS_PATH = "/mnt/c/Users/krish/FYP/atlases/Brainnetome246/BN_Atlas_246_2mm.nii.gz"

OUT_DIR = "/mnt/c/Users/krish/FYP/test/biomarkers/sub-019S4549/registration_mni152"
os.makedirs(OUT_DIR, exist_ok=True)

lines = []


def log(msg=""):
    print(msg)
    lines.append(str(msg))


def main():
    log("=== BOLD -> MNI152NLin6Asym (Brainnetome-compatible grid) registration ===\n")

    log("--- MNI template verification ---")
    tpl_path = str(tflow.get("MNI152NLin6Asym", resolution=2, suffix="T1w", desc=None, extension="nii.gz"))
    log(f"Source: TemplateFlow, template=MNI152NLin6Asym, resolution=2, suffix=T1w")
    log(f"Local cached path: {tpl_path}")

    tpl_img = nib.load(tpl_path)
    log(f"Dimensions: {tpl_img.shape}  Voxel: {tpl_img.header.get_zooms()}")
    log(f"Orientation: {''.join(nib.aff2axcodes(tpl_img.affine))}\n")

    atlas_img = nib.load(ATLAS_PATH)
    log(f"Atlas: dims={atlas_img.shape} voxel={atlas_img.header.get_zooms()} "
        f"orientation={''.join(nib.aff2axcodes(atlas_img.affine))}")
    dims_match = tuple(tpl_img.shape) == tuple(atlas_img.shape)
    log(f"Dimensions match atlas: {dims_match}\n")

    assert os.path.exists(BOLD_MEAN), f"expected existing bold_mean.nii.gz: {BOLD_MEAN}"
    log("*** LIMITATION: no T1w image -- direct cross-modal BOLD/EPI-to-MNI registration. ***\n")

    log("--- Running ANTsPy registration (type_of_transform='SyN') ---")
    moving_ants = ants.image_read(BOLD_MEAN)
    fixed_ants = ants.image_read(tpl_path)

    t0 = time.time()
    reg = ants.registration(fixed=fixed_ants, moving=moving_ants, type_of_transform="SyN", verbose=True)
    dt = time.time() - t0
    log(f"\nRegistration runtime: {dt:.2f}s")

    for tf in reg["fwdtransforms"] + reg["invtransforms"]:
        dest = os.path.join(OUT_DIR, os.path.basename(tf))
        shutil.copy(tf, dest)
        log(f"Saved transform: {dest}")

    normalized = reg["warpedmovout"]
    norm_path = os.path.join(OUT_DIR, "bold_mean_normalized_MNI152NLin6Asym.nii.gz")
    ants.image_write(normalized, norm_path)
    log(f"\nNormalized BOLD reference saved: {norm_path}\n")

    norm_data = normalized.numpy()
    tpl_data = tpl_img.get_fdata()
    x, y, z = tpl_img.shape
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    views = [
        (lambda v: v[x // 2, :, :].T, "sagittal"),
        (lambda v: v[:, y // 2, :].T, "coronal"),
        (lambda v: v[:, :, z // 2].T, "axial"),
    ]
    for col, (slicer, name) in enumerate(views):
        axes[0, col].imshow(slicer(tpl_data), cmap="gray", origin="lower")
        axes[0, col].set_title(f"MNI152NLin6Asym template ({name})", fontsize=10)
        axes[0, col].axis("off")
        axes[1, col].imshow(slicer(norm_data), cmap="gray", origin="lower")
        axes[1, col].set_title(f"Normalized BOLD ({name})", fontsize=10)
        axes[1, col].axis("off")
    plt.tight_layout()
    qc_png = os.path.join(OUT_DIR, "registration_qc_mni152.png")
    plt.savefig(qc_png, dpi=150)
    log(f"Saved: {qc_png}")

    with open(os.path.join(OUT_DIR, "registration_mni152_report.txt"), "w") as f:
        f.write("\n".join(lines))
    print("REGISTRATION_MNI152_DONE")


if __name__ == "__main__":
    main()
