"""
RECOVERED from session history (see verify_brainnetome_atlas.py header note).
NOTE: superseded by run_registration_mni152.py (registers against the
Brainnetome-compatible MNI152NLin6Asym template instead of nilearn's
ICBM152 2009a template used here) -- preserved for provenance/reference.

Brainnetome-246 pilot, Steps 1-4 ONLY (input inspection, BOLD reference,
BOLD->MNI registration via ANTsPy, registration QC).
"""
import os
import time
import numpy as np
import nibabel as nib
import ants
from nilearn import datasets
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT_BOLD = ("/mnt/c/Users/krish/FYP/derivatives/fsfast/AD/sub-019S4549/ses-01/func/"
              "sub-019S4549_ses-01_task-rest_run-01_desc-preproc_bold.nii.gz")
OUT_DIR = "/mnt/c/Users/krish/FYP/test/biomarkers/sub-019S4549"
REG_DIR = os.path.join(OUT_DIR, "registration")
os.makedirs(REG_DIR, exist_ok=True)

report_lines = []


def log(msg=""):
    print(msg)
    report_lines.append(str(msg))


def main():
    log("=== Brainnetome-246 pilot test -- sub-019S4549 ses-01 run-01 ===")
    log("Steps 1-4 only (registration + QC)\n")

    log("=== STEP 1: INSPECT INPUT ===")
    log(f"Exact input file used: {INPUT_BOLD}")
    assert os.path.exists(INPUT_BOLD), "input file not found"

    img = nib.load(INPUT_BOLD)
    data = img.get_fdata()
    shape = img.shape
    zooms = img.header.get_zooms()
    affine = img.affine
    orient = "".join(nib.aff2axcodes(affine))
    tr = float(zooms[3]) if len(zooms) > 3 else None
    n_vol = shape[3] if len(shape) > 3 else 1
    n_nan = int(np.isnan(data).sum())
    n_inf = int(np.isinf(data).sum())

    log(f"Shape: {shape}")
    log(f"Voxel size: {zooms[:3]}")
    log(f"Orientation: {orient}")
    log(f"TR: {tr} s")
    log(f"Number of volumes: {n_vol}")
    log(f"NaN count: {n_nan}   Inf count: {n_inf}\n")

    log("=== STEP 2: CREATE 3D BOLD REFERENCE (temporal mean) ===")
    mean_data = data.mean(axis=3).astype(np.float32)
    mean_img = nib.Nifti1Image(mean_data, affine)
    mean_path = os.path.join(OUT_DIR, "bold_mean.nii.gz")
    nib.save(mean_img, mean_path)
    log(f"Saved: {mean_path}\n")

    log("=== STEP 3: BOLD -> MNI NORMALIZATION (ANTsPy) ===")
    log("No T1w image exists -- using the BOLD temporal-mean directly as the moving image.\n")

    mni_template = datasets.load_mni152_template(resolution=2)
    mni_path = os.path.join(REG_DIR, "MNI152_T1_2mm_template.nii.gz")
    nib.save(mni_template, mni_path)

    log("--- MNI template located ---")
    log(f"Source: nilearn.datasets.load_mni152_template(resolution=2) (ICBM152 2009a asymmetric T1)")
    log(f"Dimensions: {mni_template.shape}\n")
    log("*** LIMITATION: cross-modal (T2*-weighted BOLD vs T1-weighted template) registration,")
    log("no T1w intermediate exists for this dataset. ***\n")

    t0 = time.time()
    moving_ants = ants.image_read(mean_path)
    fixed_ants = ants.image_read(mni_path)
    reg = ants.registration(fixed=fixed_ants, moving=moving_ants, type_of_transform="SyN", verbose=False)
    dt = time.time() - t0
    log(f"Registration completed in {dt:.1f}s")

    import shutil
    for i, tf in enumerate(reg["fwdtransforms"]):
        dest = os.path.join(REG_DIR, f"fwdtransform_{i}_{os.path.basename(tf)}")
        shutil.copy(tf, dest)
    for i, tf in enumerate(reg["invtransforms"]):
        dest = os.path.join(REG_DIR, f"invtransform_{i}_{os.path.basename(tf)}")
        shutil.copy(tf, dest)

    normalized = reg["warpedmovout"]
    norm_path = os.path.join(OUT_DIR, "bold_mean_normalized_to_MNI.nii.gz")
    ants.image_write(normalized, norm_path)
    log(f"Normalized BOLD reference saved: {norm_path}\n")

    log("=== STEP 4: REGISTRATION QC ===")
    norm_data = normalized.numpy()
    mni_data = mni_template.get_fdata()
    x, y, z = mni_data.shape
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    views = [
        (lambda v: v[x // 2, :, :].T, "sagittal"),
        (lambda v: v[:, y // 2, :].T, "coronal"),
        (lambda v: v[:, :, z // 2].T, "axial"),
    ]
    for col, (slicer, name) in enumerate(views):
        axes[0, col].imshow(slicer(mni_data), cmap="gray", origin="lower")
        axes[0, col].set_title(f"MNI template ({name})", fontsize=10)
        axes[0, col].axis("off")
        axes[1, col].imshow(slicer(norm_data), cmap="gray", origin="lower")
        axes[1, col].set_title(f"Normalized BOLD ({name})", fontsize=10)
        axes[1, col].axis("off")
    plt.tight_layout()
    qc_png = os.path.join(OUT_DIR, "registration_qc.png")
    plt.savefig(qc_png, dpi=150)
    log(f"QC image saved: {qc_png}")

    with open(os.path.join(OUT_DIR, "biomarker_test_report.txt"), "w") as f:
        f.write("\n".join(report_lines))
    print("STEPS_1_TO_4_DONE")


if __name__ == "__main__":
    main()
