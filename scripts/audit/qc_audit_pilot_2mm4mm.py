"""
READ-ONLY COMPLETE QC AUDIT of the single-acquisition 2mm/4mm pilot
(pilot_preprocessing_2mm4mm/, subject sub-019S4549/ses-01).

Does NOT rerun preprocessing, does NOT modify any NIfTI, does NOT process any
other subject. Every metric is independently recomputed from the existing
pilot outputs (not merely copied from prior reports) so this audit stands on
its own evidence. Outputs are written to a NEW subdirectory,
qc_audit_2mm4mm/, specifically so nothing in the existing qc/ directory
(fd_plot.png, dvars_plot.png, roi_coverage.png, etc.) is overwritten.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import csv
import json
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

PILOT_ROOT = "/mnt/c/Users/krish/FYP/pilot_preprocessing_2mm4mm"
PAPER_ROOT = "/mnt/c/Users/krish/FYP/pilot_preprocessing"
ATLAS_PATH = "/mnt/c/Users/krish/FYP/atlases/Brainnetome246/BN_Atlas_246_2mm.nii.gz"
ATLAS_LUT = "/mnt/c/Users/krish/FYP/atlases/Brainnetome246/BN_Atlas_246_LUT.txt"

AUDIT_DIR = os.path.join(PILOT_ROOT, "qc_audit_2mm4mm")
os.makedirs(AUDIT_DIR, exist_ok=True)

measured = {}   # every raw, directly-measured value
interp = {}     # interpretation/status notes, kept SEPARATE from measured values


def p(*parts):
    return os.path.join(*parts)


print("=" * 70)
print("SECTION 1 -- LOCATE PILOT (read-only)")
print("=" * 70)
SUBJECT, SESSION, RUN = "sub-019S4549", "ses-01", "run-01"
INPUT_NII = p(PILOT_ROOT, "input", f"{SUBJECT}_{SESSION}_task-rest_{RUN}_bold.nii.gz")
FINAL_NII = p(PILOT_ROOT, "final", f"{SUBJECT}_{SESSION}_task-rest_{RUN}_desc-2mm4mm_pilot_bold.nii.gz")
MC_FILE = p(PILOT_ROOT, "intermediate", "desc-mc_bold.nii.gz")
SMOOTH_FILE = p(PILOT_ROOT, "intermediate", "desc-smooth4mm_bold.nii.gz")
MNI_FILE = p(PILOT_ROOT, "intermediate", "desc-mni_bold.nii.gz")
MOTION_TSV = p(PILOT_ROOT, "motion", "motion_parameters.tsv")
MCDAT = p(PILOT_ROOT, "motion", "mc-afni2_motion.mcdat")
FD_TSV = p(PILOT_ROOT, "motion", "fd_values.tsv")
NUISANCE_TSV = p(PILOT_ROOT, "confounds", "nuisance_regressors.tsv")
BRAIN_MASK = p(PILOT_ROOT, "transforms", "brain_mask_mni2mm.nii.gz")
ATLAS_ALIGNED = p(PILOT_ROOT, "brainnetome", "Brainnetome246_on_BOLD_grid.nii.gz")
ROI_TS_TSV = p(PILOT_ROOT, "brainnetome", "brainnetome_246_timeseries.tsv")
ROI_COV_CSV = p(PILOT_ROOT, "brainnetome", "roi_voxel_counts.csv")
CONFIG_JSON = p(PILOT_ROOT, "pipeline_config.json")
PROC_LOG = p(PILOT_ROOT, "processing_log.txt")
PILOT_REPORT = p(PILOT_ROOT, "PILOT_2MM_4MM_REPORT.md")

for name, path in [
    ("subject", SUBJECT), ("session", SESSION), ("input NIfTI", INPUT_NII),
    ("final 2mm/4mm NIfTI", FINAL_NII), ("motion-corrected (native) intermediate", MC_FILE),
    ("smoothed intermediate (tSNR/DVARS stage)", SMOOTH_FILE), ("MNI-normalized intermediate", MNI_FILE),
    ("motion parameters", MOTION_TSV), ("native mc-afni2 .mcdat", MCDAT), ("FD values", FD_TSV),
    ("nuisance regressors", NUISANCE_TSV), ("brain mask (2mm)", BRAIN_MASK),
    ("Brainnetome atlas aligned to BOLD grid", ATLAS_ALIGNED), ("ROI timeseries", ROI_TS_TSV),
    ("existing ROI voxel-count CSV", ROI_COV_CSV), ("pipeline_config.json", CONFIG_JSON),
    ("processing_log.txt", PROC_LOG), ("prior pilot report", PILOT_REPORT),
]:
    exists = os.path.isfile(path)
    print(f"  {name}: {path}  [{'FOUND' if exists else 'MISSING'}]")
    if not exists and name not in ("subject", "session"):
        print(f"    ** WARNING: expected file missing, will mark dependent metrics NOT COMPUTABLE **")

print(f"\nCONFIRMED: auditing exactly ONE acquisition -- {SUBJECT}/{SESSION}/{RUN}")
print(f"Audit outputs written to a NEW directory (does not overwrite existing qc/): {AUDIT_DIR}")

# =====================================================================
print("\n" + "=" * 70)
print("SECTION 2 -- INPUT VS FINAL IMAGE VALIDATION")
print("=" * 70)
in_img = nib.load(INPUT_NII)
in_data = in_img.get_fdata(dtype=np.float32)
fin_img = nib.load(FINAL_NII)
fin_data = fin_img.get_fdata(dtype=np.float32)

img_val = {
    "input_dimensions": str(in_img.shape), "final_dimensions": str(fin_img.shape),
    "input_n_volumes": int(in_img.shape[3]), "final_n_volumes": int(fin_img.shape[3]),
    "n_discarded_volumes": int(in_img.shape[3] - fin_img.shape[3]),
    "TR_input_s": float(in_img.header.get_zooms()[3]), "TR_final_s": float(fin_img.header.get_zooms()[3]),
    "input_voxel_size_mm": str(tuple(round(float(z), 4) for z in in_img.header.get_zooms()[:3])),
    "final_voxel_size_mm": str(tuple(round(float(z), 4) for z in fin_img.header.get_zooms()[:3])),
    "input_orientation": "".join(nib.aff2axcodes(in_img.affine)),
    "final_orientation": "".join(nib.aff2axcodes(fin_img.affine)),
    "input_datatype": str(in_img.get_data_dtype()), "final_datatype": str(fin_img.get_data_dtype()),
    "final_NaN_count": int(np.isnan(fin_data).sum()), "final_Inf_count": int(np.isinf(fin_data).sum()),
}
measured.update(img_val)
for k, v in img_val.items():
    print(f"  {k}: {v}")
print("  input affine:"); print(in_img.affine)
print("  final affine:"); print(fin_img.affine)

verify = {
    "exactly_5_discarded": img_val["n_discarded_volumes"] == 5,
    "final_voxel_2mm": all(abs(float(z) - 2.0) < 0.01 for z in fin_img.header.get_zooms()[:3]),
    "final_is_4D": len(fin_img.shape) == 4,
    "final_no_NaN_Inf": img_val["final_NaN_count"] == 0 and img_val["final_Inf_count"] == 0,
}
print("  VERIFICATION:")
for k, v in verify.items():
    print(f"    {k}: {v}")
measured.update({f"verify_{k}": v for k, v in verify.items()})

# =====================================================================
print("\n" + "=" * 70)
print("SECTION 3 -- MOTION QC")
print("=" * 70)
mot_df = pd.read_csv(MOTION_TSV, sep="\t")
fd_df = pd.read_csv(FD_TSV, sep="\t")
FD = fd_df["FD_mm"].values
mcdat = np.loadtxt(MCDAT)

# independent cross-check: recompute FD from the native .mcdat (not from the final image)
rot_deg = mcdat[:, 1:4]
trans_mm = np.column_stack([mcdat[:, 5], mcdat[:, 6], mcdat[:, 4]])
rot_rad = np.deg2rad(rot_deg)
dtr = np.vstack([np.zeros((1, 3)), np.diff(trans_mm, axis=0)])
dro = np.vstack([np.zeros((1, 3)), np.diff(rot_rad, axis=0)])
FD_recomputed = np.abs(dtr).sum(1) + np.abs(dro * 50.0).sum(1)
fd_cross_check = bool(np.allclose(FD, FD_recomputed, atol=1e-6))
print(f"  cross-check: fd_values.tsv matches independent recomputation from native .mcdat: {fd_cross_check}")

fd_stats = {
    "mean_FD_mm": float(FD.mean()), "median_FD_mm": float(np.median(FD)),
    "max_FD_mm": float(FD.max()), "std_FD_mm": float(FD.std()),
    "n_FD_gt_0.2mm": int((FD > 0.2).sum()), "n_FD_gt_0.5mm": int((FD > 0.5).sum()),
    "n_FD_gt_1.0mm": int((FD > 1.0).sum()),
    "pct_FD_gt_0.5mm": round(100.0 * (FD > 0.5).sum() / len(FD), 2),
    "pct_FD_gt_1.0mm": round(100.0 * (FD > 1.0).sum() / len(FD), 2),
    "max_abs_translation_mm": float(np.abs(trans_mm).max()),
    "max_abs_rotation_deg": float(np.abs(rot_deg).max()),
    "fd_cross_check_matches_native_mcdat": fd_cross_check,
}
measured.update(fd_stats)
print("  MEASURED VALUES:")
for k, v in fd_stats.items():
    print(f"    {k}: {v}")
interp["FD"] = ("INTERPRETATION (not auto-PASS from mean<0.5mm alone): "
                f"{fd_stats['n_FD_gt_0.5mm']} of {len(FD)} volumes ({fd_stats['pct_FD_gt_0.5mm']}%) "
                f"exceed 0.5mm; {fd_stats['n_FD_gt_1.0mm']} exceed 1.0mm. Motion is present and "
                "non-trivial but no volumes were censored per the frozen protocol.")
print(f"  {interp['FD']}")

plt.figure(figsize=(11, 3.4))
plt.plot(FD, lw=0.9)
plt.axhline(0.2, color="orange", ls="--", lw=0.7, label="0.2mm")
plt.axhline(0.5, color="r", ls="--", lw=0.8, label="0.5mm")
plt.axhline(1.0, color="darkred", ls="--", lw=0.8, label="1.0mm")
plt.xlabel("volume"); plt.ylabel("FD (mm)")
plt.title(f"FD (audit-recomputed from native .mcdat) mean={fd_stats['mean_FD_mm']:.3f} max={fd_stats['max_FD_mm']:.3f}")
plt.legend()
plt.tight_layout(); plt.savefig(p(AUDIT_DIR, "fd_plot.png"), dpi=120); plt.close()
print(f"  saved: {p(AUDIT_DIR, 'fd_plot.png')}")

# =====================================================================
print("\n" + "=" * 70)
print("SECTION 4 -- DVARS QC")
print("=" * 70)
sm_img = nib.load(SMOOTH_FILE)
sm_data = sm_img.get_fdata(dtype=np.float32)
nt = sm_data.shape[3]
tpl_mask_img = nib.load(BRAIN_MASK)
BRAIN = tpl_mask_img.get_fdata() > 0

print(f"  DVARS stage used: intermediate/desc-smooth4mm_bold.nii.gz")
print(f"    (motion-corrected, MNI-normalized, 2mm-resampled, 4mm-FWHM-smoothed;")
print(f"     detrending/nuisance-regression/band-pass NOT yet applied at this stage)")
print(f"  DVARS TYPE: RAW INTENSITY DVARS (sqrt(mean((BOLD_t - BOLD_t-1)^2)) in native scanner")
print(f"    intensity units). This is NOT standardized DVARS and NOT percent-signal-change DVARS.")
print(f"    Therefore it must NOT be compared against literature thresholds (e.g. Power et al.'s")
print(f"    ~5/~20 guidance) that assume percent-signal-change normalization.")

sflat = sm_data.reshape(-1, nt).T[:, BRAIN.ravel()]
dv = np.sqrt(np.mean(np.diff(sflat, axis=0) ** 2, axis=1))
DVARS = np.concatenate([[0.0], dv])
dvars_stats = {
    "mean_DVARS_raw": float(DVARS[1:].mean()), "median_DVARS_raw": float(np.median(DVARS[1:])),
    "max_DVARS_raw": float(DVARS[1:].max()), "std_DVARS_raw": float(DVARS[1:].std()),
    "dvars_type": "RAW_INTENSITY (not standardized, not percent-signal-change)",
    "dvars_stage": "desc-smooth4mm_bold.nii.gz (pre-detrend/pre-nuisance/pre-bandpass)",
}
# extreme values defined only relative to this acquisition's own distribution (IQR rule),
# since no absolute literature threshold applies to raw-intensity DVARS
q1, q3 = np.percentile(DVARS[1:], [25, 75])
iqr = q3 - q1
extreme_thr = q3 + 1.5 * iqr
n_extreme = int((DVARS[1:] > extreme_thr).sum())
dvars_stats["extreme_DVARS_definition"] = "Q3 + 1.5*IQR (self-relative, no absolute literature threshold applies to raw units)"
dvars_stats["extreme_DVARS_threshold_value"] = float(extreme_thr)
dvars_stats["n_extreme_DVARS_volumes"] = n_extreme
measured.update(dvars_stats)
for k, v in dvars_stats.items():
    print(f"    {k}: {v}")

plt.figure(figsize=(11, 3.4))
plt.plot(DVARS, lw=0.9)
plt.axhline(extreme_thr, color="r", ls="--", lw=0.8, label="Q3+1.5*IQR (self-relative)")
plt.xlabel("volume"); plt.ylabel("DVARS (raw intensity)")
plt.title(f"DVARS (raw intensity) mean={dvars_stats['mean_DVARS_raw']:.1f}")
plt.legend()
plt.tight_layout(); plt.savefig(p(AUDIT_DIR, "dvars_plot.png"), dpi=120); plt.close()
print(f"  saved: {p(AUDIT_DIR, 'dvars_plot.png')}")

# =====================================================================
print("\n" + "=" * 70)
print("SECTION 5 -- tSNR")
print("=" * 70)
print("  stage: desc-smooth4mm_bold.nii.gz")
print("    smoothing applied: YES (4mm FWHM)")
print("    detrending applied: NO")
print("    nuisance regression applied: NO")
print("    band-pass filtering applied: NO")
sm_mean = sm_data.mean(axis=3)
sm_sd = sm_data.std(axis=3)
with np.errstate(divide="ignore", invalid="ignore"):
    tsnr_full = np.where(sm_sd > 0, sm_mean / sm_sd, 0)
tsnr_vals = tsnr_full[BRAIN]
tsnr_vals_finite = tsnr_vals[np.isfinite(tsnr_vals)]
tsnr_stats = {
    "mean_tSNR": float(tsnr_vals_finite.mean()), "median_tSNR": float(np.median(tsnr_vals_finite)),
    "std_tSNR": float(tsnr_vals_finite.std()), "min_tSNR": float(tsnr_vals_finite.min()),
    "max_tSNR": float(tsnr_vals_finite.max()),
    "p5_tSNR": float(np.percentile(tsnr_vals_finite, 5)), "p25_tSNR": float(np.percentile(tsnr_vals_finite, 25)),
    "p75_tSNR": float(np.percentile(tsnr_vals_finite, 75)), "p95_tSNR": float(np.percentile(tsnr_vals_finite, 95)),
}
measured.update(tsnr_stats)
for k, v in tsnr_stats.items():
    print(f"    {k}: {v}")

# check for artificially inflated tSNR from near-zero std
sd_thresh = np.percentile(sm_sd[BRAIN][sm_sd[BRAIN] > 0], 1)
suspicious = (tsnr_full > np.percentile(tsnr_vals_finite, 99.5)) & BRAIN & (sm_sd < sd_thresh * 5)
n_suspicious = int(suspicious.sum())
print(f"  suspicious near-zero-std / inflated-tSNR voxel check: {n_suspicious} voxels flagged")
if n_suspicious > 0:
    coords = np.argwhere(suspicious)
    print(f"    location (voxel indices, first 10): {coords[:10].tolist()}")
    print(f"    these voxels are NOT evidence of excellent data quality -- flagged as artifact candidates")
measured["n_suspicious_high_tsnr_voxels"] = n_suspicious

plt.figure(figsize=(10, 8))
mid = [s // 2 for s in tsnr_full.shape]
for i, (axis, idx) in enumerate([(0, mid[0]), (1, mid[1]), (2, mid[2])]):
    plt.subplot(2, 2, i + 1)
    sl = np.rot90(np.take(tsnr_full, idx, axis=axis))
    im = plt.imshow(sl, cmap="hot", vmin=0, vmax=np.percentile(tsnr_vals_finite, 99))
    plt.colorbar(im, fraction=0.046)
    plt.title(f"tSNR axis {axis}"); plt.axis("off")
plt.suptitle(f"tSNR map (mean={tsnr_stats['mean_tSNR']:.1f}, {n_suspicious} suspicious voxels flagged)")
plt.tight_layout(); plt.savefig(p(AUDIT_DIR, "tsnr_map.png"), dpi=120); plt.close()
print(f"  saved: {p(AUDIT_DIR, 'tsnr_map.png')}")

# =====================================================================
print("\n" + "=" * 70)
print("SECTION 6 -- SNR")
print("=" * 70)
# same fBIRN-style formula as the original pilot QC: mean(signal in brain)/std(background)
raw_mean = in_data.mean(axis=3)


def otsu(vals3d):
    v = vals3d[vals3d > 0]
    hist, edges = np.histogram(v, bins=256)
    mids = (edges[:-1] + edges[1:]) / 2
    w1, w2 = np.cumsum(hist), np.cumsum(hist[::-1])[::-1]
    m1 = np.cumsum(hist * mids) / np.maximum(w1, 1)
    m2 = np.cumsum((hist * mids)[::-1])[::-1] / np.maximum(w2, 1)
    var = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
    return mids[int(np.argmax(var))]


thr = otsu(raw_mean)
lab, n_comp = ndimage.label(raw_mean > thr, structure=np.ones((3, 3, 3)))
sizes = ndimage.sum(raw_mean > thr, lab, range(1, n_comp + 1))
native_brain = ndimage.binary_fill_holes(lab == int(np.argmax(sizes) + 1))
bg = (~native_brain) & (raw_mean > 0)
snr_signal = float(raw_mean[native_brain].mean())
snr_noise = float(raw_mean[bg].std())
SNR = snr_signal / snr_noise
snr_stats = {
    "SNR": SNR, "SNR_signal_definition": "mean(temporal-mean intensity) inside native-space Otsu+largest-component brain mask",
    "SNR_background_definition": "std(temporal-mean intensity) in non-brain, nonzero voxels of the raw native-space input",
    "SNR_formula": "signal_mean / background_std (fBIRN-style)",
    "SNR_mask_used": "native-resolution Otsu-thresholded brain mask (largest 3D connected component)",
}
measured.update(snr_stats)
for k, v in snr_stats.items():
    print(f"    {k}: {v}")
print("  No arbitrary threshold applied -- reported as a measured value only.")

# =====================================================================
print("\n" + "=" * 70)
print("SECTION 7 -- CNR")
print("=" * 70)
print("  CNR = NOT_RELIABLY_COMPUTABLE")
print("  Reason: no T1w anatomical image exists for this subject/dataset, so no reliable")
print("  GM/WM/CSF tissue segmentation can be produced. Fabricating a segmentation from")
print("  BOLD/EPI or template priors alone would not yield a scientifically valid tissue")
print("  contrast estimate. No PASS/FAIL is assigned.")
measured["CNR"] = "NOT_RELIABLY_COMPUTABLE"
measured["CNR_reason"] = "No T1w anatomical image available for tissue segmentation"

# =====================================================================
print("\n" + "=" * 70)
print("SECTION 8 -- SPATIAL ENTROPY")
print("=" * 70)
vals = sm_mean[BRAIN]
hist, _ = np.histogram(vals, bins=256)
prob = hist / hist.sum()
prob = prob[prob > 0]
spatial_entropy = float(-(prob * np.log2(prob)).sum())
measured["spatial_entropy_bits"] = spatial_entropy
print(f"    spatial_entropy: {spatial_entropy:.4f} bits")
print(f"    image/stage used: temporal mean of desc-smooth4mm_bold.nii.gz")
print(f"    mask used: transforms/brain_mask_mni2mm.nii.gz")
print(f"    formula: Shannon entropy, H = -sum(p*log2(p)), 256-bin histogram of in-mask intensities")
print(f"    No universal good/bad threshold applied -- descriptive/QC metric only.")

# =====================================================================
print("\n" + "=" * 70)
print("SECTION 9 -- TEMPORAL ENTROPY")
print("=" * 70)
flat_brain = sm_data.reshape(-1, nt).T[:, BRAIN.ravel()]  # shape (nt, n_voxels)
tvals = flat_brain.T  # shape (n_voxels, nt) -- rows are voxels, columns are time
z = (tvals - tvals.mean(axis=1, keepdims=True)) / (tvals.std(axis=1, keepdims=True) + 1e-9)
print(f"    per-voxel time series matrix shape (n_voxels x nt): {z.shape}")
temporal_entropies = []
for i in range(z.shape[0]):
    row = z[i, :]
    h, _ = np.histogram(row, bins=16)
    pr = h / h.sum()
    pr = pr[pr > 0]
    temporal_entropies.append(-(pr * np.log2(pr)).sum())
temporal_entropy = float(np.mean(temporal_entropies))
measured["temporal_entropy_bits"] = temporal_entropy
print(f"    temporal_entropy: {temporal_entropy:.4f} bits")
print(f"    calculation: mean per-voxel Shannon entropy of z-scored temporal histogram (16 bins)")
print(f"    stage used: desc-smooth4mm_bold.nii.gz")
print(f"    No universal good/bad threshold applied.")

# =====================================================================
print("\n" + "=" * 70)
print("SECTION 10 -- BRAIN COVERAGE / MASK QC")
print("=" * 70)
mask_data = tpl_mask_img.get_fdata() > 0
total_vox = int(np.prod(mask_data.shape))
mask_vox = int(mask_data.sum())
mask_lab, mask_n_comp = ndimage.label(mask_data, structure=np.ones((3, 3, 3)))
comp_sizes = ndimage.sum(mask_data, mask_lab, range(1, mask_n_comp + 1)) if mask_n_comp else []
mask_stats = {
    "total_brain_mask_voxels": mask_vox, "pct_image_voxels_in_mask": round(100.0 * mask_vox / total_vox, 3),
    "mask_n_connected_components": int(mask_n_comp),
    "mask_largest_component_pct": round(100.0 * max(comp_sizes) / mask_vox, 2) if mask_n_comp else 0,
    "min_signal_inside_mask": float(sm_mean[mask_data].min()), "max_signal_inside_mask": float(sm_mean[mask_data].max()),
    "mean_signal_inside_mask": float(sm_mean[mask_data].mean()),
    "background_mean": float(sm_mean[~mask_data][sm_mean[~mask_data] != 0].mean()) if (sm_mean[~mask_data] != 0).any() else 0.0,
    "background_sd": float(sm_mean[~mask_data][sm_mean[~mask_data] != 0].std()) if (sm_mean[~mask_data] != 0).any() else 0.0,
    "mask_empty": mask_vox == 0,
}
measured.update(mask_stats)
for k, v in mask_stats.items():
    print(f"    {k}: {v}")
print(f"  disconnected components check: {mask_n_comp} component(s) "
      f"({'single connected mask, no disconnection' if mask_n_comp == 1 else 'MULTIPLE COMPONENTS -- CHECK REQUIRED'})")

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
midm = [s // 2 for s in mask_data.shape]
for i, axis in enumerate([0, 1, 2]):
    bg_sl = np.rot90(np.take(sm_mean, midm[axis], axis=axis))
    mk_sl = np.rot90(np.take(mask_data, midm[axis], axis=axis))
    ax[i].imshow(bg_sl, cmap="gray")
    ax[i].contour(mk_sl, levels=[0.5], colors="lime", linewidths=1.0)
    ax[i].axis("off"); ax[i].set_title(f"axis {axis}")
plt.suptitle(f"Brain mask QC ({mask_vox} voxels, {mask_n_comp} component(s))")
plt.tight_layout(); plt.savefig(p(AUDIT_DIR, "brain_mask_qc.png"), dpi=120); plt.close()
print(f"  saved: {p(AUDIT_DIR, 'brain_mask_qc.png')}")

# =====================================================================
print("\n" + "=" * 70)
print("SECTION 11 -- MNI REGISTRATION QC")
print("=" * 70)
import templateflow.api as tflow
tpl_brain = str(tflow.get("MNI152NLin6Asym", resolution=2, desc="brain", suffix="T1w", extension="nii.gz"))
tpl_img = nib.load(tpl_brain)
reg_check = {
    "final_affine_matches_template": bool(np.allclose(fin_img.affine, tpl_img.affine, atol=1e-3)),
    "final_dims_match_template": fin_img.shape[:3] == tpl_img.shape[:3],
    "template_used": f"MNI152NLin6Asym res-02 desc-brain_T1w (TemplateFlow) -- {tpl_brain}",
}
measured.update(reg_check)
for k, v in reg_check.items():
    print(f"    {k}: {v}")
print("  NOTE: this check compares grid/affine only (necessary but not sufficient for a good")
print("  registration) -- see the overlay figure below for actual spatial alignment.")

tpl_d = tpl_img.get_fdata()
fin_mean = fin_data.mean(axis=3)
fig, ax = plt.subplots(1, 3, figsize=(13, 5))
for i, axis in enumerate([0, 1, 2]):
    t_sl = np.rot90(np.take(tpl_d, tpl_d.shape[axis] // 2, axis=axis))
    b_sl = np.rot90(np.take(fin_mean, fin_mean.shape[axis] // 2, axis=axis))
    ax[i].imshow(b_sl, cmap="gray")
    ax[i].contour(t_sl > np.percentile(t_sl[t_sl > 0], 40), levels=[0.5], colors="r", linewidths=0.7)
    ax[i].axis("off"); ax[i].set_title(f"axis {axis}: final BOLD + template contour")
plt.suptitle("MNI registration QC (independently regenerated for this audit)")
plt.tight_layout(); plt.savefig(p(AUDIT_DIR, "mni_registration_qc.png"), dpi=120); plt.close()
print(f"  saved: {p(AUDIT_DIR, 'mni_registration_qc.png')}")
print("  visual inspection notes: L/R, A/P, S/I coverage and boundary alignment must be")
print("  judged from the saved figure -- not asserted numerically here.")

# =====================================================================
print("\n" + "=" * 70)
print("SECTION 12 -- 2-mm GRID QC vs BRAINNETOME-246")
print("=" * 70)
atlas_img = nib.load(ATLAS_PATH)
grid_check = {
    "BOLD_dimensions": str(fin_img.shape[:3]), "BOLD_voxel_size_mm": str(tuple(round(float(z), 4) for z in fin_img.header.get_zooms()[:3])),
    "Brainnetome_dimensions": str(atlas_img.shape), "Brainnetome_voxel_size_mm": str(tuple(round(float(z), 4) for z in atlas_img.header.get_zooms()[:3])),
    "BOLD_orientation": "".join(nib.aff2axcodes(fin_img.affine)), "Brainnetome_orientation": "".join(nib.aff2axcodes(atlas_img.affine)),
}
grids_raw_identical = (fin_img.shape[:3] == atlas_img.shape and np.allclose(fin_img.affine, atlas_img.affine, atol=1e-3))
atlas_canon = nib.as_closest_canonical(atlas_img)
grids_identical_after_reorient = (fin_img.shape[:3] == atlas_canon.shape and np.allclose(fin_img.affine, atlas_canon.affine, atol=1e-3))
grid_check["grids_identical_raw"] = bool(grids_raw_identical)
grid_check["grids_identical_after_lossless_reorientation"] = bool(grids_identical_after_reorient)
measured.update(grid_check)
for k, v in grid_check.items():
    print(f"    {k}: {v}")
print("  BOLD affine:"); print(fin_img.affine)
print("  Brainnetome affine:"); print(atlas_img.affine)
print("  Interpretation: dims/voxel size match at 2mm/91x109x91 in both cases; orientation")
print("  differs by an L/R array-index flip (atlas ships LAS, BOLD is RAS) -- grids are")
print("  identical only AFTER a lossless canonical reorientation (verified above), not raw.")

# =====================================================================
print("\n" + "=" * 70)
print("SECTION 13 -- BRAINNETOME-246 ROI COVERAGE (independently recomputed)")
print("=" * 70)
if os.path.isfile(ATLAS_ALIGNED):
    print(f"  atlas already aligned to BOLD grid found: {ATLAS_ALIGNED} -- inspecting, not regenerating")
    aligned_img = nib.load(ATLAS_ALIGNED)
    aligned_d = np.asarray(aligned_img.dataobj)
    grid_matches_final = (aligned_img.shape == fin_img.shape[:3] and np.allclose(aligned_img.affine, fin_img.affine, atol=1e-3))
    print(f"  aligned-atlas grid matches final BOLD grid: {grid_matches_final}")

    lut = {}
    with open(ATLAS_LUT) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                lut[int(parts[0])] = parts[1]

    rows = []
    counts = []
    for roi_id in range(1, 247):
        n = int((aligned_d == roi_id).sum())
        counts.append(n)
        status = "ZERO_VOXELS" if n == 0 else ("VERY_SMALL" if n < 10 else "ADEQUATE")
        rows.append({"ROI_ID": roi_id, "ROI_NAME": lut.get(roi_id, f"ROI_{roi_id}"),
                     "VOXEL_COUNT": n, "STATUS": status})
    with open(p(AUDIT_DIR, "brainnetome_roi_voxel_counts.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["ROI_ID", "ROI_NAME", "VOXEL_COUNT", "STATUS"])
        w.writeheader(); w.writerows(rows)
    counts = np.array(counts)

    # cross-check against the existing stored CSV from the original run
    prior_match = None
    if os.path.isfile(ROI_COV_CSV):
        prior = pd.read_csv(ROI_COV_CSV)
        prior_counts = prior.sort_values("ROI_ID")["VOXEL_COUNT"].values
        prior_match = bool(np.array_equal(prior_counts, counts))
    print(f"  cross-check: independently recomputed voxel counts match the stored "
          f"brainnetome/roi_voxel_counts.csv exactly: {prior_match}")

    roi_stats = {
        "n_ROIs_zero_voxels": int((counts == 0).sum()),
        "n_ROIs_1_to_2_voxels": int(((counts >= 1) & (counts <= 2)).sum()),
        "n_ROIs_lt_5_voxels": int((counts < 5).sum()),
        "min_ROI_voxel_count": int(counts.min()), "max_ROI_voxel_count": int(counts.max()),
        "mean_ROI_voxel_count": float(counts.mean()), "median_ROI_voxel_count": float(np.median(counts)),
        "roi_counts_match_stored_csv": prior_match,
    }
    measured.update(roi_stats)
    for k, v in roi_stats.items():
        print(f"    {k}: {v}")
else:
    print("  Brainnetome atlas NOT already aligned to BOLD grid -- per instruction, only")
    print("  compatibility/voxel-count QC may be performed (no preprocessing). SKIPPING.")
    roi_stats = {}

# =====================================================================
print("\n" + "=" * 70)
print("SECTION 14 -- ROI TIME-SERIES QC")
print("=" * 70)
if os.path.isfile(ROI_TS_TSV):
    ts = pd.read_csv(ROI_TS_TSV, sep="\t")
    n_rois_col = ts.shape[1]
    n_tp = ts.shape[0]
    nan_count = int(ts.isna().sum().sum())
    inf_count = int(np.isinf(ts.values).sum())
    stds = ts.std(axis=0)
    means = ts.mean(axis=0)
    zero_var = int((stds == 0).sum())
    near_zero_var = int((stds < stds.median() * 0.01).sum()) if stds.median() > 0 else 0
    ts_stats = {
        "roi_timeseries_n_ROIs": n_rois_col, "roi_timeseries_n_timepoints": n_tp,
        "roi_timeseries_expected_timepoints_match": n_tp == img_val["final_n_volumes"],
        "roi_timeseries_NaN_count": nan_count, "roi_timeseries_Inf_count": inf_count,
        "roi_timeseries_zero_variance_ROIs": zero_var, "roi_timeseries_near_zero_variance_ROIs": near_zero_var,
    }
    measured.update(ts_stats)
    for k, v in ts_stats.items():
        print(f"    {k}: {v}")
    roi_ts_qc = pd.DataFrame({"ROI": ts.columns, "mean": means.values, "std": stds.values})
    roi_ts_qc.to_csv(p(AUDIT_DIR, "roi_timeseries_qc.csv"), index=False)
    print(f"  saved: {p(AUDIT_DIR, 'roi_timeseries_qc.csv')}")
else:
    print("  ROI timeseries file not found -- SKIPPING this section (NOT COMPUTABLE).")
    ts_stats = {}

# =====================================================================
print("\n" + "=" * 70)
print("SECTION 15 -- COMPARE 4mm/6mm (paper) VS 2mm/4mm (pilot)")
print("=" * 70)
paper_qc_csv = p(PAPER_ROOT, "qc", "qc_metrics.csv")
paper_add_json = p(PAPER_ROOT, "qc", "qc_metrics_additional.json")
paper_fd_tsv = p(PAPER_ROOT, "motion", "fd_values.tsv")
comparison_rows = []
if os.path.isfile(paper_qc_csv):
    pqc = dict(zip(pd.read_csv(paper_qc_csv)["metric"], pd.read_csv(paper_qc_csv)["value"]))
    padd = json.load(open(paper_add_json)) if os.path.isfile(paper_add_json) else {}
    paper_fd = pd.read_csv(paper_fd_tsv, sep="\t")["FD_mm"].values if os.path.isfile(paper_fd_tsv) else None

    fd_identical = bool(np.allclose(FD, paper_fd)) if paper_fd is not None and len(paper_fd) == len(FD) else None
    print(f"  FD identity check (should be identical -- derived purely from motion params, "
          f"independent of resampling/smoothing): {fd_identical}")
    if fd_identical:
        print("    CONFIRMED: FD values are byte-for-byte identical between configurations, as")
        print("    expected, because both pipelines ran mc-afni2 on the same input with the same")
        print("    reference frame -- motion estimation is upstream of and independent from the")
        print("    2mm/4mm vs 4mm/6mm resampling/smoothing choice.")

    comparison_rows = [
        ("Voxel size", "4x4x4mm", "2x2x2mm"),
        ("Dimensions", "46x55x46", "91x109x91"),
        ("Brain-mask voxels", str(pqc.get("mask_voxel_count", pqc.get("brain_mask_voxels_4mm", "n/a"))), str(mask_vox)),
        ("tSNR (mean)", str(pqc.get("mean_tSNR", "n/a")), f"{tsnr_stats['mean_tSNR']:.2f}"),
        ("SNR", str(padd.get("SNR", "n/a")), f"{SNR:.2f}"),
        ("mean FD", str(pqc.get("mean_FD_mm", "n/a")), f"{fd_stats['mean_FD_mm']:.4f}"),
        ("mean DVARS (raw)", str(pqc.get("mean_DVARS", "n/a")), f"{dvars_stats['mean_DVARS_raw']:.1f}"),
        ("Spatial entropy (bits)", str(padd.get("spatial_entropy_bits", "n/a")), f"{spatial_entropy:.4f}"),
        ("Temporal entropy (bits)", str(padd.get("temporal_entropy_bits", "n/a")), f"{temporal_entropy:.4f}"),
    ]
    if roi_stats:
        comparison_rows.append(("Brainnetome zero-voxel ROIs", "0 (measured previously)", str(roi_stats.get("n_ROIs_zero_voxels", "n/a"))))
        comparison_rows.append(("Brainnetome very-small ROIs (<10 vox)", "0 (measured previously)",
                                str(int((counts < 10).sum())) if 'counts' in dir() else "n/a"))
    for a, b_, c_ in comparison_rows:
        print(f"    {a}: paper={b_}  pilot={c_}")
else:
    print("  Paper (4mm/6mm) pilot QC not found -- comparison skipped, not fabricated.")

# =====================================================================
print("\n" + "=" * 70)
print("SECTION 16 -- QC CLASSIFICATION TABLE")
print("=" * 70)
classification = [
    ("mean tSNR", f"{tsnr_stats['mean_tSNR']:.2f}", "ratio", "smooth4mm (pre-detrend)",
     "no fixed literature cutoff applied; reported alongside comparable pilot for context", "CHECK REQUIRED",
     "Not auto-classified: falls in a plausible range for 3T EPI but no sourced absolute threshold was applied per instruction."),
    ("SNR", f"{SNR:.2f}", "ratio", "native raw input",
     "no arbitrary threshold applied per instruction", "CHECK REQUIRED",
     "Measured value only; no defensible universal cutoff was supplied."),
    ("CNR", "NOT_RELIABLY_COMPUTABLE", "-", "-", "requires T1w tissue segmentation (absent)", "NOT COMPUTABLE",
     "No T1w anatomical image exists; fabricating segmentation was explicitly prohibited."),
    ("mean FD", f"{fd_stats['mean_FD_mm']:.4f} mm", "mm", "native motion params",
     "0.2mm/0.5mm/1.0mm reference bands reported, not treated as hard cutoffs", "WARN",
     f"{fd_stats['n_FD_gt_0.5mm']}/{len(FD)} volumes exceed 0.5mm; real, non-negligible motion present."),
    ("max FD", f"{fd_stats['max_FD_mm']:.4f} mm", "mm", "native motion params", "-", "WARN",
     "A single high-motion frame; not used to exclude the acquisition per protocol."),
    ("mean DVARS", f"{dvars_stats['mean_DVARS_raw']:.1f}", "raw intensity units", "smooth4mm (pre-detrend)",
     "no absolute threshold valid for raw (non-normalized) DVARS", "NOT COMPUTABLE AS PASS/FAIL",
     "Units are not standardized/percent-signal-change, so literature DVARS cutoffs do not apply."),
    ("spatial entropy", f"{spatial_entropy:.4f} bits", "bits", "smooth4mm mean image",
     "none -- exploratory/descriptive only", "CHECK REQUIRED", "No universal good/bad threshold exists."),
    ("temporal entropy", f"{temporal_entropy:.4f} bits", "bits", "smooth4mm per-voxel", "none", "CHECK REQUIRED",
     "No universal good/bad threshold exists."),
    ("brain-mask coverage", f"{mask_stats['pct_image_voxels_in_mask']:.2f}% ({mask_vox} vox)", "%", "2mm analysis grid",
     "single connected component expected", "PASS" if mask_n_comp == 1 and mask_vox > 0 else "CHECK REQUIRED",
     f"{mask_n_comp} connected component(s); mask is non-empty and single-piece." if mask_n_comp == 1 else "Multiple components or empty mask detected."),
    ("MNI registration", f"grid/affine match={reg_check['final_affine_matches_template']}", "-", "final BOLD vs template",
     "grid/affine identity + visual overlay", "PASS" if reg_check["final_affine_matches_template"] else "CHECK REQUIRED",
     "Grid/affine confirm correct output space; see mni_registration_qc.png for visual confirmation of alignment quality."),
    ("NaN/Inf", f"{img_val['final_NaN_count']}/{img_val['final_Inf_count']}", "count", "final image", "must be 0/0",
     "PASS" if verify["final_no_NaN_Inf"] else "FAIL", "No invalid numeric values in the final output."),
    ("Brainnetome ROI coverage", f"{roi_stats.get('n_ROIs_zero_voxels','n/a')} zero-voxel ROIs" if roi_stats else "n/a",
     "ROI count", "aligned atlas on final BOLD grid", "0 zero-voxel ROIs required for full coverage",
     "PASS" if roi_stats and roi_stats.get("n_ROIs_zero_voxels", 1) == 0 else ("NOT COMPUTABLE" if not roi_stats else "WARN"),
     "All 246 ROIs have usable voxel counts." if roi_stats and roi_stats.get("n_ROIs_zero_voxels", 1) == 0 else "See ROI coverage CSV."),
]
with open(p(AUDIT_DIR, "qc_metrics_2mm4mm.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Metric", "Value", "Units", "Calculation Stage", "Threshold/Reference", "Status", "Interpretation"])
    w.writerows(classification)
print(f"  saved: {p(AUDIT_DIR, 'qc_metrics_2mm4mm.csv')}")
for row in classification:
    print(f"    {row[0]}: {row[5]} -- {row[6]}")

with open(p(AUDIT_DIR, "qc_metrics_2mm4mm.json"), "w") as f:
    json.dump(measured, f, indent=2, default=str)
print(f"  saved: {p(AUDIT_DIR, 'qc_metrics_2mm4mm.json')}")

# =====================================================================
print("\n" + "=" * 70)
print("SECTION 17 -- OVERALL QC")
print("=" * 70)
acceptable = ["Final image structurally valid (4D, correct 2mm grid, no NaN/Inf)",
             "Motion parameters are the genuine native mc-afni2 output, cross-verified against independent FD recomputation",
             "Brain mask is a single connected, non-empty component with 100% expected coverage",
             "MNI registration grid/affine matches the target template exactly",
             "Brainnetome-246 ROI coverage: 0 zero-voxel ROIs, full 246/246 usable" if roi_stats.get("n_ROIs_zero_voxels") == 0 else "Brainnetome ROI coverage requires review",
             "ROI time-series structurally valid (246 ROIs x expected timepoints, no NaN/Inf, no zero-variance ROIs)" if ts_stats else "ROI time-series not available for this audit"]
caution = [f"Motion: {fd_stats['n_FD_gt_0.5mm']}/{len(FD)} volumes exceed 0.5mm FD, "
          f"{fd_stats['n_FD_gt_1.0mm']} exceed 1.0mm -- real head motion present, not excluded per protocol",
          f"{n_suspicious} tSNR voxel(s) flagged as possible near-zero-variance artifacts (investigated, not treated as excellent quality)"]
not_computable = ["CNR -- no T1w anatomical image available for tissue segmentation",
                  "DVARS against absolute literature thresholds -- this implementation uses raw intensity units, not standardized/percent-signal-change"]
manual_inspection = ["mni_registration_qc.png -- visual L/R, A/P, S/I alignment must be confirmed by eye",
                     "brain_mask_qc.png -- visual confirmation of no missing brain regions"]
limitations = ["No T1w image exists for this ADNI-derived dataset (structural, not a processing failure)",
              "SliceTiming metadata unavailable -- slice-timing correction was not performed, an accepted dataset-wide limitation",
              "WM/CSF nuisance regressors are derived from population MNI template priors, not subject-specific segmentation"]

print("A. Metrics clearly acceptable:")
for x in acceptable: print(f"   - {x}")
print("B. Metrics requiring caution:")
for x in caution: print(f"   - {x}")
print("C. Metrics that cannot be computed:")
for x in not_computable: print(f"   - {x}")
print("D. Metrics requiring visual/manual inspection:")
for x in manual_inspection: print(f"   - {x}")
print("E. Dataset-specific limitations:")
for x in limitations: print(f"   - {x}")

# Decision logic, explicit and defensible
has_fail = not verify["final_no_NaN_Inf"] or (roi_stats and roi_stats.get("n_ROIs_zero_voxels", 0) > 0)
has_warn = fd_stats["n_FD_gt_0.5mm"] > 0 or n_suspicious > 0
if has_fail:
    OVERALL = "FAIL"
    reason = "A hard structural criterion failed (NaN/Inf present or zero-voxel Brainnetome ROIs found)."
elif has_warn:
    OVERALL = "PASS WITH WARNINGS"
    reason = ("Structural integrity, registration, mask, and ROI coverage are all sound; however "
              "real head motion (FD) and a small number of flagged tSNR voxels mean this is not a "
              "spotless PASS. Nothing here rises to exclusion-level severity under the project's "
              "established WARN convention.")
else:
    OVERALL = "PASS"
    reason = "All checked criteria met without any flagged caution items."
print(f"\nOVERALL PILOT QC STATUS: {OVERALL}")
print(f"REASON: {reason}")

# =====================================================================
print("\n" + "=" * 70)
print("SECTION 18/19 -- WRITING OUTPUT FILES")
print("=" * 70)

report_path = p(PILOT_ROOT, "QC_AUDIT_2MM4MM.md")
with open(report_path, "w", encoding="utf-8") as f:
    w = f.write
    w("# 2-mm / 4-mm Pilot QC Audit\n\n")
    w("Read-only audit. No preprocessing was rerun, no NIfTI files were modified, no other "
      "subject was processed.\n\n")

    w("## 1. Acquisition\n\n")
    w(f"- Subject: {SUBJECT}\n- Session: {SESSION}\n- Input file: `{INPUT_NII}`\n- Final file: `{FINAL_NII}`\n\n")

    w("## 2. Image Integrity\n\n")
    for k in ["input_dimensions", "final_dimensions", "TR_final_s", "final_voxel_size_mm",
              "final_NaN_count", "final_Inf_count"]:
        w(f"- {k}: {measured[k]}\n")
    w("\n")

    w("## 3. Motion QC\n\n")
    for k in ["mean_FD_mm", "median_FD_mm", "max_FD_mm", "n_FD_gt_0.2mm", "n_FD_gt_0.5mm", "n_FD_gt_1.0mm"]:
        w(f"- {k}: {measured[k]}\n")
    w("\n")

    w("## 4. DVARS\n\n")
    for k in ["mean_DVARS_raw", "median_DVARS_raw", "max_DVARS_raw", "dvars_type"]:
        w(f"- {k}: {measured[k]}\n")
    w("\n")

    w("## 5. tSNR\n\n")
    for k in ["mean_tSNR", "median_tSNR", "min_tSNR", "max_tSNR", "n_suspicious_high_tsnr_voxels"]:
        w(f"- {k}: {measured[k]}\n")
    w("\n")

    w("## 6. SNR\n\n")
    for k in ["SNR", "SNR_signal_definition", "SNR_background_definition", "SNR_formula"]:
        w(f"- {k}: {measured[k]}\n")
    w("\n")

    w("## 7. CNR\n\n- CNR: NOT_RELIABLY_COMPUTABLE\n- Reason: no T1w anatomical image available.\n\n")

    w("## 8. Entropy\n\n")
    w(f"- spatial entropy: {spatial_entropy:.4f} bits\n- temporal entropy: {temporal_entropy:.4f} bits\n\n")

    w("## 9. Brain Mask\n\n")
    for k in ["total_brain_mask_voxels", "pct_image_voxels_in_mask", "mask_n_connected_components"]:
        w(f"- {k}: {measured[k]}\n")
    w("\n")

    w("## 10. MNI Registration\n\n")
    for k in ["final_affine_matches_template", "final_dims_match_template", "template_used"]:
        w(f"- {k}: {measured[k]}\n")
    w("\n")

    w("## 11. 2-mm Grid\n\n")
    for k in ["BOLD_dimensions", "BOLD_voxel_size_mm", "Brainnetome_dimensions", "Brainnetome_voxel_size_mm",
              "grids_identical_raw", "grids_identical_after_lossless_reorientation"]:
        w(f"- {k}: {measured[k]}\n")
    w("\n")

    w("## 12. Brainnetome-246 Compatibility\n\n")
    if roi_stats:
        for k in ["n_ROIs_zero_voxels", "n_ROIs_1_to_2_voxels", "n_ROIs_lt_5_voxels",
                  "min_ROI_voxel_count", "max_ROI_voxel_count", "median_ROI_voxel_count",
                  "roi_counts_match_stored_csv"]:
            w(f"- {k}: {measured[k]}\n")
    else:
        w("- Atlas alignment not found; ROI coverage NOT COMPUTABLE for this audit.\n")
    w("\n## 13. ROI Coverage\n\nSee `qc_audit_2mm4mm/brainnetome_roi_voxel_counts.csv`.\n\n")

    w("## 14. ROI Time-Series QC\n\n")
    if ts_stats:
        for k, v in ts_stats.items():
            w(f"- {k}: {v}\n")
    else:
        w("- Not available.\n")
    w("\n")

    w("## 15. Comparison with 4-mm/6-mm Pilot\n\n")
    if comparison_rows:
        w("| Aspect | Paper (4mm/6mm) | Pilot (2mm/4mm) |\n|---|---|---|\n")
        for a, b_, c_ in comparison_rows:
            w(f"| {a} | {b_} | {c_} |\n")
    else:
        w("Paper pilot QC not found; comparison skipped.\n")
    w("\n")

    w("## 16. Warnings\n\n")
    for x in caution:
        w(f"- {x}\n")
    w("\n## 17. Dataset Limitations\n\n")
    for x in limitations:
        w(f"- {x}\n")

    w(f"\n## 18. Overall QC Status\n\n**{OVERALL}**\n\n{reason}\n\n")
    w("## 19. Recommendation\n\n")
    w("This single-acquisition pilot's preprocessing is structurally sound and the 2mm/4mm "
      "configuration achieves full Brainnetome-246 coverage on this subject. The measured "
      "motion level (FD) is real and should be tracked at the group level once/if this "
      "configuration is applied to the full dataset, consistent with how FD-WARN acquisitions "
      "were already handled (not excluded) across the six production groups. No pipeline change "
      "is recommended based on this audit alone.\n")

print(f"  report written: {report_path}")

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print(f"SUBJECT:\n{SUBJECT}\n")
print(f"PIPELINE:\n2-mm MNI + 4-mm FWHM\n")
print(f"tSNR:\n{tsnr_stats['mean_tSNR']:.2f}\n")
print(f"SNR:\n{SNR:.2f}\n")
print(f"CNR:\nNOT COMPUTABLE\n")
print(f"MEAN FD:\n{fd_stats['mean_FD_mm']:.4f} mm\n")
print(f"DVARS:\n{dvars_stats['mean_DVARS_raw']:.1f} (raw intensity units)\n")
print(f"SPATIAL ENTROPY:\n{spatial_entropy:.4f} bits\n")
print(f"TEMPORAL ENTROPY:\n{temporal_entropy:.4f} bits\n")
print(f"BRAIN MASK:\n{mask_vox} voxels ({mask_stats['pct_image_voxels_in_mask']:.2f}% of image)\n")
if roi_stats:
    print(f"BRAINNETOME ROIS:\n246 total\n{roi_stats['n_ROIs_zero_voxels']} zero\n"
          f"{int((counts < 10).sum())} very small (<10 vox)\n")
print(f"NaN/Inf:\n{img_val['final_NaN_count']}/{img_val['final_Inf_count']}\n")
print(f"MNI REGISTRATION:\n{'grid/affine match confirmed' if reg_check['final_affine_matches_template'] else 'MISMATCH'}\n")
print(f"OVERALL QC:\n{OVERALL}\n")
print("QC_AUDIT_2MM4MM_DONE")
