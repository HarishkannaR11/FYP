"""
READ-ONLY visual + quantitative QC comparison: raw BOLD vs mc-afni2 motion-corrected
output vs mri_fwhm (6mm) smoothed output, for ONE representative AD acquisition
(sub-002S5018 ses-01).

Does NOT touch BIDS/, NIfTI_/, raw_data/, or any pipeline output. Only reads
existing files (raw source NIfTI + cached Nipype intermediate outputs) and
writes new PNG/CSV/txt files into audit/smoothing_visual_qc/.

Spatial roughness metric: Forman et al. (1995) spatial-derivative FWHM
estimate, per-axis then geometric-mean combined -- the SAME estimator already
used and validated in scripts/audit/ad_quantitative_qc.py, reused here (not
reinvented) for consistency with the rest of the QC suite.
"""
import os
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW = "/mnt/c/Users/krish/FYP/NIfTI_/NIfTI/AD/002_S_5018/002_S_5018_Resting_State_fMRI_20130516122033_501.nii.gz"
MOCO = "/home/harish/fyp_work/fsfast_ad/work/fsfast_ad_full_rerun/_run_key_sub-002S5018_ses-01/moco/moco_bold.nii.gz"
SMOOTH = "/home/harish/fyp_work/fsfast_ad/work/fsfast_ad_full_rerun/_run_key_sub-002S5018_ses-01/smooth/smooth_bold.nii.gz"
ERES = "/home/harish/fyp_work/fsfast_ad/work/fsfast_ad_full_rerun/_run_key_sub-002S5018_ses-01/glmfit/glmdir/eres.nii.gz"

OUT_DIR = "/mnt/c/Users/krish/FYP/audit/smoothing_visual_qc"
os.makedirs(OUT_DIR, exist_ok=True)


def spatial_fwhm_estimate(vol3d, mask, voxel_sizes):
    """Forman et al. 1995 spatial-derivative FWHM estimator (per-axis, geometric-mean)."""
    fwhms = []
    for axis, vs in enumerate(voxel_sizes[:3]):
        d = np.diff(vol3d, axis=axis)
        m = mask.take(range(vol3d.shape[axis] - 1), axis=axis) & \
            mask.take(range(1, vol3d.shape[axis]), axis=axis)
        if m.sum() < 100:
            continue
        var_ratio = np.var(vol3d[mask]) / np.var(d[m])
        if var_ratio <= 0.5:
            continue
        sigma = np.sqrt(-1.0 / (4 * np.log(1 - 1 / (2 * var_ratio))))
        fwhm_vox = sigma * np.sqrt(8 * np.log(2))
        fwhms.append(fwhm_vox * vs)
    if not fwhms:
        return np.nan
    return float(np.exp(np.mean(np.log(fwhms))))


def otsu_threshold(data):
    hist, bin_edges = np.histogram(data.ravel(), bins=256)
    bin_mids = (bin_edges[:-1] + bin_edges[1:]) / 2
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    mean1 = np.cumsum(hist * bin_mids) / np.where(weight1 == 0, 1, weight1)
    mean2 = (np.cumsum((hist * bin_mids)[::-1])[::-1]) / np.where(weight2 == 0, 1, weight2)
    inter_class_var = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = np.argmax(inter_class_var)
    return bin_mids[idx]


def load_mean_vol(path):
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        data = data.mean(axis=3)
    return data, img.header.get_zooms()[:3]


def main():
    print("Loading volumes (temporal mean, magnitude domain)...")
    raw_mean, raw_vox = load_mean_vol(RAW)
    moco_mean, moco_vox = load_mean_vol(MOCO)
    smooth_mean, smooth_vox = load_mean_vol(SMOOTH)

    print(f"raw shape={raw_mean.shape} voxel={raw_vox}")
    print(f"moco shape={moco_mean.shape} voxel={moco_vox}")
    print(f"smooth shape={smooth_mean.shape} voxel={smooth_vox}")

    # brain mask from raw temporal-mean via Otsu (magnitude domain -- valid here)
    thr = otsu_threshold(raw_mean)
    mask = raw_mean > thr
    print(f"Otsu threshold={thr:.2f}, mask voxels={mask.sum()} / {mask.size}")

    fwhm_raw = spatial_fwhm_estimate(raw_mean, mask, raw_vox)
    fwhm_moco = spatial_fwhm_estimate(moco_mean, mask, moco_vox)
    fwhm_smooth = spatial_fwhm_estimate(smooth_mean, mask, smooth_vox)

    # additional roughness metric: mean absolute first-spatial-derivative magnitude
    # (independent, simpler cross-check against the Forman FWHM estimate)
    def mean_abs_grad(vol, mask):
        gx = np.abs(np.diff(vol, axis=0))
        gy = np.abs(np.diff(vol, axis=1))
        gz = np.abs(np.diff(vol, axis=2))
        # normalize by mean intensity inside mask so metric is scale-invariant
        scale = vol[mask].mean()
        return float((gx.mean() + gy.mean() + gz.mean()) / 3.0 / scale)

    rough_raw = mean_abs_grad(raw_mean, mask)
    rough_moco = mean_abs_grad(moco_mean, mask)
    rough_smooth = mean_abs_grad(smooth_mean, mask)

    with open(os.path.join(OUT_DIR, "roughness_metrics.txt"), "w") as f:
        f.write("=== Spatial roughness / smoothness comparison ===\n")
        f.write("Subject: sub-002S5018 ses-01 (representative AD acquisition)\n\n")
        f.write("Metric 1: Forman et al. (1995) spatial-derivative FWHM estimate (mm)\n")
        f.write(f"  raw (unprocessed source)      : {fwhm_raw:.3f} mm\n")
        f.write(f"  after mc-afni2 (motion-corr)  : {fwhm_moco:.3f} mm\n")
        f.write(f"  after mri_fwhm 6mm smoothing  : {fwhm_smooth:.3f} mm\n\n")
        f.write("Metric 2: mean |spatial gradient| normalized by mean intensity (dimensionless, lower = smoother)\n")
        f.write(f"  raw                            : {rough_raw:.5f}\n")
        f.write(f"  after mc-afni2                : {rough_moco:.5f}\n")
        f.write(f"  after mri_fwhm 6mm smoothing  : {rough_smooth:.5f}\n\n")
        smoother = fwhm_smooth > fwhm_moco and rough_smooth < rough_moco
        f.write(f"CONCLUSION: smoothed output is {'CONFIRMED SMOOTHER' if smoother else 'NOT CLEARLY SMOOTHER'} "
                f"than raw/motion-corrected by both independent metrics.\n")
    print("Wrote roughness_metrics.txt")
    print(open(os.path.join(OUT_DIR, "roughness_metrics.txt")).read())

    # ---- visual comparison: mid-axial, mid-coronal, mid-sagittal slices ----
    def mid_slices(vol):
        x, y, z = vol.shape
        return vol[x // 2, :, :].T, vol[:, y // 2, :].T, vol[:, :, z // 2].T

    vols = {"Raw (unprocessed)": raw_mean, "mc-afni2 (motion corrected)": moco_mean,
            "mri_fwhm 6mm (smoothed)": smooth_mean}

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    for col, (label, vol) in enumerate(vols.items()):
        sag, cor, ax_ = mid_slices(vol)
        vmax = np.percentile(vol[mask], 99)
        for row, (sl, name) in enumerate([(sag, "sagittal"), (cor, "coronal"), (ax_, "axial")]):
            ax = axes[row, col]
            ax.imshow(sl, cmap="gray", origin="lower", vmin=0, vmax=vmax)
            ax.set_title(f"{label}\n({name}, mid-slice)" if row == 0 else name, fontsize=9)
            ax.axis("off")
    plt.tight_layout()
    out_png = os.path.join(OUT_DIR, "raw_vs_moco_vs_smooth_slices.png")
    plt.savefig(out_png, dpi=150)
    print(f"Wrote {out_png}")

    # single-row direct axial comparison, same slice index, same colour scale
    fig2, axes2 = plt.subplots(1, 3, figsize=(14, 5))
    vmax = np.percentile(raw_mean[mask], 99)
    for ax, (label, vol) in zip(axes2, vols.items()):
        _, _, ax_slice = mid_slices(vol)
        ax.imshow(ax_slice, cmap="gray", origin="lower", vmin=0, vmax=vmax)
        ax.set_title(label, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    out_png2 = os.path.join(OUT_DIR, "axial_comparison_same_scale.png")
    plt.savefig(out_png2, dpi=150)
    print(f"Wrote {out_png2}")


if __name__ == "__main__":
    main()
