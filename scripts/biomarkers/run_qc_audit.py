"""
RECOVERED from session history (see verify_brainnetome_atlas.py header note).

COMPLETE PREPROCESSING QC/VALIDATION METRIC AUDIT -- reusable template.
These are QC/validation metrics, NOT biomarkers.
"""
import os
import csv
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT_DIR = "/mnt/c/Users/krish/FYP/test/biomarkers/sub-019S4549/preprocessing_qc"
os.makedirs(OUT_DIR, exist_ok=True)

F_MOCO = "/home/harish/fyp_work/fsfast_ad_masked/work/fsfast_ad_production_masked/_run_key_sub-019S4549_ses-01/moco/moco_bold.nii.gz"
F_MCDAT = "/home/harish/fyp_work/fsfast_ad_masked/work/fsfast_ad_production_masked/_run_key_sub-019S4549_ses-01/moco/moco_bold.mcdat"
F_SMOOTH = "/home/harish/fyp_work/fsfast_ad_masked/work/fsfast_ad_production_masked/_run_key_sub-019S4549_ses-01/smooth/smooth_bold.nii.gz"
F_ERES = "/mnt/c/Users/krish/FYP/derivatives/fsfast/AD/sub-019S4549/ses-01/func/sub-019S4549_ses-01_task-rest_run-01_desc-preproc_bold.nii.gz"
F_MASK = "/mnt/c/Users/krish/FYP/derivatives/fsfast/AD/sub-019S4549/ses-01/func/sub-019S4549_ses-01_task-rest_run-01_desc-brain_mask.nii.gz"
F_DESIGN = "/mnt/c/Users/krish/FYP/derivatives/fsfast/AD/sub-019S4549/ses-01/func/sub-019S4549_ses-01_task-rest_run-01_desc-design_matrix.txt"

report = []
metrics_rows = []


def log(msg=""):
    print(msg)
    report.append(str(msg))


def add_metric(metric, value, stage, definition, direction, status, interp):
    metrics_rows.append({"Metric": metric, "Value": value, "Input_Stage": stage,
                          "Definition": definition, "Expected_Direction": direction,
                          "Status": status, "Interpretation": interp})


def validate_image(name, path, mask_data=None):
    log(f"--- General validation: {name} ---")
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    log(f"Dimensions: {img.shape}  Voxel: {img.header.get_zooms()[:3]}")
    log(f"NaN: {int(np.isnan(data).sum())}  Inf: {int(np.isinf(data).sum())}")
    return {"data": data}


def compute_tsnr(data, mask, stage_name):
    tmean = data.mean(axis=3)
    tstd = data.std(axis=3)
    with np.errstate(divide="ignore", invalid="ignore"):
        tsnr_map = np.where(tstd > 0, tmean / tstd, 0)
    vals = tsnr_map[mask]
    valid = vals[np.isfinite(vals) & (vals != 0)]
    log(f"tSNR mean={valid.mean():.2f} median={np.median(valid):.2f}")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(valid, bins=60, color="steelblue")
    ax.set_title(f"tSNR distribution ({stage_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "tsnr_distribution.png"), dpi=150)
    return valid


def compute_fd_full(mcdat_path):
    mc = np.loadtxt(mcdat_path)
    rot_rad = np.deg2rad(mc[:, 1:4])
    trans = mc[:, 4:7]
    d_trans = np.diff(trans, axis=0)
    d_rot_mm = np.diff(rot_rad, axis=0) * 50.0
    fd = np.sum(np.abs(d_trans), axis=1) + np.sum(np.abs(d_rot_mm), axis=1)
    fd_full = np.concatenate([[0.0], fd])
    log(f"FD mean={fd.mean():.3f}mm max={fd.max():.3f}mm")
    csv_path = os.path.join(OUT_DIR, "fd_values.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "FD_mm"])
        for i, v in enumerate(fd_full):
            w.writerow([i, v])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(fd_full, color="crimson", lw=0.8)
    ax.axhline(0.5, color="red", ls="--", lw=0.8)
    ax.set_title("Framewise Displacement")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "fd_plot.png"), dpi=150)
    return fd_full


def compute_dvars(data, mask, stage_name):
    vox_ts = data[mask]
    diff = np.diff(vox_ts, axis=1)
    dvars = np.sqrt((diff ** 2).mean(axis=0))
    dvars_full = np.concatenate([[np.nan], dvars])
    log(f"DVARS mean={dvars.mean():.1f}")
    csv_path = os.path.join(OUT_DIR, "dvars_values.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "DVARS"])
        for i, v in enumerate(dvars_full):
            w.writerow([i, v])
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(dvars_full, color="darkorange", lw=0.8)
    ax.set_title(f"DVARS ({stage_name})")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "dvars_plot.png"), dpi=150)
    return dvars_full


def compute_gcor(data, mask):
    vox_ts = data[mask].astype(np.float64)
    mu = vox_ts.mean(axis=1, keepdims=True)
    sd = vox_ts.std(axis=1, keepdims=True)
    sd[sd == 0] = 1.0
    u = (vox_ts - mu) / sd
    return float(u.mean(axis=0).var())


def main():
    log("=== PREPROCESSING QC / VALIDATION METRIC AUDIT (template) ===\n")
    mask = nib.load(F_MASK).get_fdata().astype(bool)
    moco_info = validate_image("Motion-corrected BOLD", F_MOCO, mask)
    smooth_info = validate_image("Smoothed BOLD (pre-GLM)", F_SMOOTH, mask)
    eres_info = validate_image("Final masked GLM residual", F_ERES, mask)

    log("\n1. tSNR (smoothed pre-GLM -- residual is zero-mean, tSNR undefined there)")
    compute_tsnr(smooth_info["data"], mask, "smoothed pre-GLM")

    log("\n4. FD")
    compute_fd_full(F_MCDAT)

    log("\n5. DVARS (smoothed pre-GLM)")
    compute_dvars(smooth_info["data"], mask, "smoothed pre-GLM")

    log("\n6. GCOR (pre vs post GLM)")
    gcor_pre = compute_gcor(smooth_info["data"], mask)
    gcor_post = compute_gcor(eres_info["data"], mask)
    log(f"GCOR pre={gcor_pre:.4f} post={gcor_post:.4f}")

    with open(os.path.join(OUT_DIR, "qc_audit_full_log.txt"), "w") as f:
        f.write("\n".join(report))
    print("QC_AUDIT_DONE")


if __name__ == "__main__":
    main()
