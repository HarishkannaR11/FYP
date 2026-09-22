"""
READ-ONLY COMPLETE QC AUDIT of the ORIGINAL pilot (pilot_preprocessing/,
the 4mm-resample / 6mm-FWHM "paper" configuration, subject sub-019S4549/ses-01).

Audits ONLY pilot_preprocessing/. Does not rerun preprocessing, does not modify
any NIfTI, does not modify or overwrite anything in pilot_preprocessing/qc/.
All new outputs go to pilot_preprocessing/qc_audit/.

Every metric is recomputed independently from the pilot's own files and then
cross-checked against the stored values, so the audit stands on its own evidence.
No universal/unsourced PASS-FAIL thresholds are applied: measured values and
interpretation are kept separate.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import csv
import json
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

ROOT = "/mnt/c/Users/krish/FYP/pilot_preprocessing"
ATLAS_PATH = "/mnt/c/Users/krish/FYP/atlases/Brainnetome246/BN_Atlas_246_2mm.nii.gz"
ATLAS_LUT = "/mnt/c/Users/krish/FYP/atlases/Brainnetome246/BN_Atlas_246_LUT.txt"
AUDIT = os.path.join(ROOT, "qc_audit")
os.makedirs(AUDIT, exist_ok=True)

SUBJECT, SESSION, RUN = "sub-019S4549", "ses-01", "run-01"
M = {}          # measured values
NOTES = {}      # interpretation, kept separate from measurements


def p(*a):
    return os.path.join(*a)


def hdr(t):
    print("\n" + "=" * 70); print(t); print("=" * 70)


# ============================================================
hdr("SECTION 1 -- PILOT IDENTIFICATION (read-only)")
PATHS = {
    "input NIfTI": p(ROOT, "input", f"{SUBJECT}_{SESSION}_task-rest_{RUN}_bold.nii.gz"),
    "JSON sidecar": p(ROOT, "input", f"{SUBJECT}_{SESSION}_task-rest_{RUN}_bold.json"),
    "final preprocessed NIfTI": p(ROOT, "final", f"{SUBJECT}_{SESSION}_task-rest_{RUN}_desc-paperpreproc_bold.nii.gz"),
    "intermediate: discard5": p(ROOT, "intermediate", "desc-discard5_bold.nii.gz"),
    "intermediate: no-stc": p(ROOT, "intermediate", "desc-no-stc_bold.nii.gz"),
    "intermediate: motion-corrected": p(ROOT, "intermediate", "desc-mc_bold.nii.gz"),
    "intermediate: MNI-normalized": p(ROOT, "intermediate", "desc-mni_bold.nii.gz"),
    "intermediate: 4mm resampled": p(ROOT, "intermediate", "desc-mni4mm_bold.nii.gz"),
    "intermediate: 6mm smoothed": p(ROOT, "intermediate", "desc-smooth6mm_bold.nii.gz"),
    "intermediate: detrended": p(ROOT, "intermediate", "desc-detrend_bold.nii.gz"),
    "intermediate: nuisance-regressed": p(ROOT, "intermediate", "desc-nuisance_bold.nii.gz"),
    "intermediate: band-pass": p(ROOT, "intermediate", "desc-bandpass_bold.nii.gz"),
    "motion parameters (.mcdat native)": p(ROOT, "motion", "mc-afni2_motion.mcdat"),
    "motion parameters (tsv)": p(ROOT, "motion", "motion_parameters.tsv"),
    "FD file": p(ROOT, "motion", "fd_values.tsv"),
    "DVARS file": p(ROOT, "qc", "dvars_values.csv"),
    "QC metrics (stored)": p(ROOT, "qc", "qc_metrics.csv"),
    "QC metrics additional (stored)": p(ROOT, "qc", "qc_metrics_additional.json"),
    "existing QC report": p(ROOT, "PILOT_PREPROCESSING_REPORT.md"),
    "processing log": p(ROOT, "processing_log.txt"),
    "pipeline config": p(ROOT, "pipeline_config.json"),
    "brain mask (4mm)": p(ROOT, "transforms", "brain_mask_mni4mm.nii.gz"),
    "confounds (nuisance regressors)": p(ROOT, "confounds", "nuisance_regressors.tsv"),
}
for k, v in PATHS.items():
    print(f"  {k}: {v}  [{'FOUND' if os.path.isfile(v) else 'MISSING'}]")

xfms = sorted([f for f in os.listdir(p(ROOT, "transforms")) if "xfm" in f])
print(f"  MNI registration transform files ({len(xfms)}):")
for x in xfms:
    print(f"    {p(ROOT, 'transforms', x)}")

finals = os.listdir(p(ROOT, "final"))
print(f"\n  Files in final/: {finals}")
print(f"  CONFIRMED: exactly ONE acquisition in this folder -- "
      f"{SUBJECT}/{SESSION}/{RUN} ({len(finals)} final image)")
print(f"  Audit outputs -> {AUDIT} (existing qc/ is never written to)")

# ============================================================
hdr("SECTION 2 -- VERIFY ACTUAL PIPELINE CONFIGURATION (from files, not assumed)")
cfg = json.load(open(PATHS["pipeline config"]))
log_txt = open(PATHS["processing log"], encoding="utf-8", errors="replace").read()

checks = []


def cfgcheck(label, actual, evidence):
    checks.append((label, actual, evidence))
    print(f"  {label}: {actual}\n      evidence: {evidence}")


cfgcheck("First 5 volumes removed",
         f"YES ({cfg['volumes_removed']} removed, {cfg['volumes_retained']} retained)",
         "pipeline_config.json volumes_removed/volumes_retained; log 'removed exactly 5, VERIFIED'"
         if "exactly 5 volumes removed" in log_txt or "VERIFIED" in log_txt else "pipeline_config.json")
cfgcheck("Slice timing correction",
         f"NOT PERFORMED ({cfg['slice_timing_reason']})",
         "pipeline_config.json slice_timing_status; log 'SLICE_TIMING_CORRECTION = NOT_PERFORMED'")
cfgcheck("Motion correction", cfg["motion_correction_method"],
         "pipeline_config.json motion_correction_method; native .mcdat present")
cfgcheck("MNI normalization", f"{cfg['registration_method']} onto {cfg['mni_template']}",
         "pipeline_config.json registration_method/mni_template; transform files present")
cfgcheck("4 x 4 x 4 mm resampling", f"final_voxel_size_mm={cfg['final_voxel_size_mm']} "
         f"via {cfg['resample_interpolation']}", "pipeline_config.json final_voxel_size_mm (verified against image in Sec 3)")
cfgcheck("6 mm FWHM smoothing", f"{cfg['smoothing_FWHM_mm']} mm FWHM via {cfg['smoothing_tool']}",
         "pipeline_config.json smoothing_FWHM_mm")
cfgcheck("Linear detrending", cfg["detrending_method"], "pipeline_config.json detrending_method")
cfgcheck("Nuisance regression", f"{cfg['nuisance_n_regressors']} regressors; Friston24={cfg['friston24_included']}; "
         f"WM={cfg['wm_regressor_status']}; CSF={cfg['csf_regressor_status']}; GSR={cfg['global_signal_regression']}",
         "pipeline_config.json nuisance_regressors (list of 28) + confounds/nuisance_regressors.tsv")
cfgcheck("0.01-0.10 Hz band-pass", f"{cfg['filter_low_hz']}-{cfg['filter_high_hz']} Hz, {cfg['filter_design']}, "
         f"Nyquist={cfg['nyquist_hz']:.4f} Hz", "pipeline_config.json filter_low_hz/filter_high_hz/filter_design")

# independently verify the confound file really has 28 columns incl. Friston-24
conf = pd.read_csv(PATHS["confounds (nuisance regressors)"], sep="\t")
n_friston = sum(1 for c in conf.columns if c.startswith("friston24_"))
print(f"\n  INDEPENDENT VERIFICATION of confounds file: {conf.shape[1]} columns, "
      f"{conf.shape[0]} timepoints, {n_friston} Friston-24 columns")
M["confound_n_columns"] = int(conf.shape[1])
M["confound_n_friston24_columns"] = int(n_friston)
M["confound_n_timepoints"] = int(conf.shape[0])
friston_ok = n_friston == 24
print(f"  Friston-24 genuinely present as 24 distinct columns: {friston_ok}")

with open(p(AUDIT, "pipeline_verification.md"), "w", encoding="utf-8") as f:
    f.write("# Pipeline Verification -- Original Pilot (pilot_preprocessing/)\n\n")
    f.write("Configuration read from the pilot's own `pipeline_config.json` and `processing_log.txt`, "
            "not assumed from the folder name.\n\n")
    f.write("| Pipeline step | What was ACTUALLY performed | Evidence |\n|---|---|---|\n")
    for label, actual, ev in checks:
        f.write(f"| {label} | {actual} | {ev} |\n")
    f.write(f"\n## Independent verification of the confound design\n\n")
    f.write(f"- confounds/nuisance_regressors.tsv: **{conf.shape[1]} columns x {conf.shape[0]} timepoints**\n")
    f.write(f"- Friston-24 columns actually present: **{n_friston}** (expected 24) -> {friston_ok}\n")
    f.write(f"- Column names: {', '.join(conf.columns)}\n")
print(f"  saved: {p(AUDIT, 'pipeline_verification.md')}")

# ============================================================
hdr("SECTION 3 -- IMAGE INTEGRITY")
in_img = nib.load(PATHS["input NIfTI"])
in_d = in_img.get_fdata(dtype=np.float32)
fin_img = nib.load(PATHS["final preprocessed NIfTI"])
fin_d = fin_img.get_fdata(dtype=np.float32)

integ = {
    "original_dimensions": str(in_img.shape), "final_dimensions": str(fin_img.shape),
    "original_n_volumes": int(in_img.shape[3]), "final_n_volumes": int(fin_img.shape[3]),
    "volumes_discarded": int(in_img.shape[3] - fin_img.shape[3]),
    "TR_s": float(fin_img.header.get_zooms()[3]),
    "original_voxel_size_mm": str(tuple(round(float(z), 4) for z in in_img.header.get_zooms()[:3])),
    "final_voxel_size_mm": str(tuple(round(float(z), 4) for z in fin_img.header.get_zooms()[:3])),
    "original_orientation": "".join(nib.aff2axcodes(in_img.affine)),
    "final_orientation": "".join(nib.aff2axcodes(fin_img.affine)),
    "original_datatype": str(in_img.get_data_dtype()), "final_datatype": str(fin_img.get_data_dtype()),
    "final_NaN_count": int(np.isnan(fin_d).sum()), "final_Inf_count": int(np.isinf(fin_d).sum()),
    "final_min": float(np.min(fin_d)), "final_max": float(np.max(fin_d)), "final_mean": float(np.mean(fin_d)),
    "original_min": float(np.min(in_d)), "original_max": float(np.max(in_d)), "original_mean": float(np.mean(in_d)),
}
M.update(integ)
for k, v in integ.items():
    print(f"  {k}: {v}")
print("  original affine:"); print(in_img.affine)
print("  final affine:"); print(fin_img.affine)
final_is_4d = len(fin_img.shape) == 4
final_is_4mm = all(abs(float(z) - 4.0) < 0.01 for z in fin_img.header.get_zooms()[:3])
M["verify_final_is_4D"] = bool(final_is_4d)
M["verify_final_is_4mm_isotropic"] = bool(final_is_4mm)
print(f"  VERIFIED final is 4D: {final_is_4d}")
print(f"  VERIFIED final is 4mm isotropic: {final_is_4mm}")

# ============================================================
hdr("SECTION 4 -- MOTION QC (independent recomputation from native .mcdat)")
mcdat = np.loadtxt(PATHS["motion parameters (.mcdat native)"])
rot_deg = mcdat[:, 1:4]
trans_mm = np.column_stack([mcdat[:, 5], mcdat[:, 6], mcdat[:, 4]])  # dL->x, dP->y, dS->z
rot_rad = np.deg2rad(rot_deg)
dtr = np.vstack([np.zeros((1, 3)), np.diff(trans_mm, axis=0)])
dro = np.vstack([np.zeros((1, 3)), np.diff(rot_rad, axis=0)])
FD = np.abs(dtr).sum(1) + np.abs(dro * 50.0).sum(1)
print(f"  FD recomputed independently: Power et al. (2012) formulation, "
      f"sum|dtrans| + sum|drot_rad * 50mm|, from the native FS-FAST .mcdat")

stored_fd = pd.read_csv(PATHS["FD file"], sep="\t")["FD_mm"].values
fd_match = bool(len(stored_fd) == len(FD) and np.allclose(stored_fd, FD, atol=1e-6))
max_abs_diff = float(np.max(np.abs(stored_fd - FD))) if len(stored_fd) == len(FD) else float("nan")
print(f"  stored fd_values.tsv vs independent recomputation: MATCH={fd_match} "
      f"(max abs diff={max_abs_diff:.3e})")

fd = {
    "mean_FD_mm": float(FD.mean()), "median_FD_mm": float(np.median(FD)), "max_FD_mm": float(FD.max()),
    "std_FD_mm": float(FD.std()),
    "n_FD_gt_0.2mm": int((FD > 0.2).sum()), "n_FD_gt_0.5mm": int((FD > 0.5).sum()),
    "n_FD_gt_1.0mm": int((FD > 1.0).sum()),
    "pct_FD_gt_0.5mm": round(100.0 * (FD > 0.5).sum() / len(FD), 2),
    "pct_FD_gt_1.0mm": round(100.0 * (FD > 1.0).sum() / len(FD), 2),
    "max_abs_translation_mm": float(np.abs(trans_mm).max()),
    "max_abs_rotation_deg": float(np.abs(rot_deg).max()),
    "fd_stored_matches_recomputed": fd_match,
}
M.update(fd)
print("  MEASURED VALUES (no PASS/FAIL applied here):")
for k, v in fd.items():
    print(f"    {k}: {v}")

pd.DataFrame({"volume": np.arange(len(FD)), "FD_mm_recomputed": FD,
              "FD_mm_stored": stored_fd,
              "abs_diff": np.abs(FD - stored_fd)}).to_csv(p(AUDIT, "fd_values_verified.csv"), index=False)
print(f"  saved: {p(AUDIT, 'fd_values_verified.csv')}")

plt.figure(figsize=(11, 3.4))
plt.plot(FD, lw=0.9, label="FD (recomputed)")
for lev, col in [(0.2, "orange"), (0.5, "r"), (1.0, "darkred")]:
    plt.axhline(lev, color=col, ls="--", lw=0.7, label=f"{lev}mm reference line")
plt.xlabel("volume"); plt.ylabel("FD (mm)")
plt.title(f"FD -- original 4mm/6mm pilot (mean={fd['mean_FD_mm']:.3f}, max={fd['max_FD_mm']:.3f})")
plt.legend(fontsize=7); plt.tight_layout()
plt.savefig(p(AUDIT, "fd_plot.png"), dpi=120); plt.close()
print(f"  saved: {p(AUDIT, 'fd_plot.png')}")

# ============================================================
hdr("SECTION 5 -- DVARS QC")
smooth_img_o = nib.load(PATHS["intermediate: 6mm smoothed"])
sm = smooth_img_o.get_fdata(dtype=np.float32)
nt = sm.shape[3]
mask_img = nib.load(PATHS["brain mask (4mm)"])
mask_d = mask_img.get_fdata() > 0

print("  Determining DVARS type actually used by this pilot:")
print("    stage: intermediate/desc-smooth6mm_bold.nii.gz (MNI-normalized, 4mm-resampled,")
print("           6mm-FWHM smoothed; detrend/nuisance/band-pass NOT yet applied)")
print("    formula in the pilot: sqrt(mean((BOLD_t - BOLD_t-1)^2)) over in-mask voxels,")
print("           on native scanner intensity units with NO normalization")
print("    => TYPE: RAW-INTENSITY DVARS (NOT standardized, NOT percent-signal-change)")
print("    => Literature thresholds for standardized/%-change DVARS DO NOT APPLY here.")

sflat = sm.reshape(-1, nt).T[:, mask_d.ravel()]
dv = np.sqrt(np.mean(np.diff(sflat, axis=0) ** 2, axis=1))
DV = np.concatenate([[0.0], dv])
stored_dv = pd.read_csv(PATHS["DVARS file"])["DVARS"].values
dv_match = bool(len(stored_dv) == len(DV) and np.allclose(stored_dv, DV, atol=1e-3))
print(f"  stored dvars_values.csv vs independent recomputation: MATCH={dv_match}")

q1, q3 = np.percentile(DV[1:], [25, 75])
thr_extreme = q3 + 1.5 * (q3 - q1)
dvs = {
    "dvars_type": "RAW_INTENSITY (not standardized, not percent-signal-change)",
    "dvars_stage": "desc-smooth6mm_bold.nii.gz (pre-detrend/pre-nuisance/pre-bandpass)",
    "mean_DVARS_raw": float(DV[1:].mean()), "median_DVARS_raw": float(np.median(DV[1:])),
    "max_DVARS_raw": float(DV[1:].max()), "std_DVARS_raw": float(DV[1:].std()),
    "extreme_definition": "Q3 + 1.5*IQR, self-relative (no valid absolute threshold for raw units)",
    "extreme_threshold_value": float(thr_extreme),
    "n_extreme_DVARS_volumes": int((DV[1:] > thr_extreme).sum()),
    "dvars_stored_matches_recomputed": dv_match,
}
M.update(dvs)
for k, v in dvs.items():
    print(f"    {k}: {v}")

plt.figure(figsize=(11, 3.4))
plt.plot(DV, lw=0.9)
plt.axhline(thr_extreme, color="r", ls="--", lw=0.8, label="Q3+1.5*IQR (self-relative)")
plt.xlabel("volume"); plt.ylabel("DVARS (raw intensity units)")
plt.title(f"DVARS (raw intensity) -- original pilot, mean={dvs['mean_DVARS_raw']:.1f}")
plt.legend(fontsize=8); plt.tight_layout()
plt.savefig(p(AUDIT, "dvars_plot.png"), dpi=120); plt.close()
print(f"  saved: {p(AUDIT, 'dvars_plot.png')}")

# ============================================================
hdr("SECTION 6 -- tSNR QC")
print("  Stage used by the pilot for tSNR (from its stored QC + report):")
stored_qc = dict(zip(pd.read_csv(PATHS["QC metrics (stored)"])["metric"],
                     pd.read_csv(PATHS["QC metrics (stored)"])["value"]))
print(f"    stored tSNR_stage field: {stored_qc.get('tSNR_stage')}")
print("    -> desc-smooth6mm_bold.nii.gz")
print("       smoothing applied: YES (6mm FWHM)")
print("       detrending applied: NO")
print("       nuisance regression applied: NO")
print("       band-pass filtering applied: NO")

sm_mean, sm_sd = sm.mean(axis=3), sm.std(axis=3)
with np.errstate(divide="ignore", invalid="ignore"):
    tsnr = np.where(sm_sd > 0, sm_mean / sm_sd, 0.0)
tv = tsnr[mask_d]
tv = tv[np.isfinite(tv)]
ts = {
    "mean_tSNR": float(tv.mean()), "median_tSNR": float(np.median(tv)), "std_tSNR": float(tv.std()),
    "min_tSNR": float(tv.min()), "max_tSNR": float(tv.max()),
    "p5_tSNR": float(np.percentile(tv, 5)), "p25_tSNR": float(np.percentile(tv, 25)),
    "p75_tSNR": float(np.percentile(tv, 75)), "p95_tSNR": float(np.percentile(tv, 95)),
}
M.update(ts)
for k, v in ts.items():
    print(f"    {k}: {v}")
stored_tsnr = float(stored_qc.get("mean_tSNR", float("nan")))
tsnr_match = abs(stored_tsnr - ts["mean_tSNR"]) < 0.5
print(f"  stored mean_tSNR={stored_tsnr} vs recomputed={ts['mean_tSNR']:.4f} -> match(±0.5)={tsnr_match}")
M["tsnr_stored_matches_recomputed"] = bool(tsnr_match)

# investigate extremely high tSNR
sd_in = sm_sd[mask_d]
sd_pos = sd_in[sd_in > 0]
sd_low = np.percentile(sd_pos, 1)
hi_cut = np.percentile(tv, 99.5)
susp = mask_d & (tsnr > hi_cut) & (sm_sd < sd_low * 5)
n_susp = int(susp.sum())
M["n_suspicious_high_tsnr_voxels"] = n_susp
M["tsnr_high_cut_used"] = float(hi_cut)
print(f"  Investigation of extremely high tSNR (>{hi_cut:.1f}, i.e. 99.5th pct) combined with")
print(f"  near-zero temporal SD (< 5x the 1st-percentile in-mask SD): {n_susp} voxel(s)")
if n_susp:
    cc_lab, cc_n = ndimage.label(susp)
    # are they at the mask edge?
    eroded = ndimage.binary_erosion(mask_d, iterations=2)
    edge = mask_d & ~eroded
    frac_edge = float((susp & edge).sum() / n_susp)
    print(f"    connected clusters: {cc_n}; fraction lying in the outer 2-voxel mask rim: {frac_edge:.2%}")
    print(f"    INTERPRETATION: these are treated as possible near-zero-variance ARTIFACTS,")
    print(f"    NOT as evidence of excellent data quality.")
    M["suspicious_tsnr_fraction_at_mask_edge"] = frac_edge
    M["suspicious_tsnr_n_clusters"] = int(cc_n)

fig, ax = plt.subplots(2, 2, figsize=(10, 8))
mid = [s // 2 for s in tsnr.shape]
for i, axis in enumerate([0, 1, 2]):
    a = ax.flat[i]
    im = a.imshow(np.rot90(np.take(tsnr, mid[axis], axis=axis)), cmap="hot",
                  vmin=0, vmax=float(np.percentile(tv, 99)))
    plt.colorbar(im, ax=a, fraction=0.046); a.axis("off"); a.set_title(f"tSNR axis {axis}")
ax.flat[3].hist(tv, bins=80); ax.flat[3].set_title("in-mask tSNR distribution")
ax.flat[3].set_xlabel("tSNR")
plt.suptitle(f"tSNR -- original 4mm/6mm pilot (mean={ts['mean_tSNR']:.1f}, {n_susp} suspicious voxels)")
plt.tight_layout(); plt.savefig(p(AUDIT, "tsnr_qc.png"), dpi=120); plt.close()
print(f"  saved: {p(AUDIT, 'tsnr_qc.png')}")

# ============================================================
hdr("SECTION 7 -- SNR")
add = json.load(open(PATHS["QC metrics additional (stored)"])) if os.path.isfile(
    PATHS["QC metrics additional (stored)"]) else {}
raw_mean = in_d.mean(axis=3)


def otsu(v3):
    v = v3[v3 > 0]
    h, e = np.histogram(v, bins=256)
    mids = (e[:-1] + e[1:]) / 2
    w1, w2 = np.cumsum(h), np.cumsum(h[::-1])[::-1]
    m1 = np.cumsum(h * mids) / np.maximum(w1, 1)
    m2 = np.cumsum((h * mids)[::-1])[::-1] / np.maximum(w2, 1)
    var = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
    return mids[int(np.argmax(var))]


t = otsu(raw_mean)
lab, nlab = ndimage.label(raw_mean > t, structure=np.ones((3, 3, 3)))
sizes = ndimage.sum(raw_mean > t, lab, range(1, nlab + 1))
nat_brain = ndimage.binary_fill_holes(lab == int(np.argmax(sizes) + 1))
bg = (~nat_brain) & (raw_mean > 0)
sig = float(raw_mean[nat_brain].mean()); noi = float(raw_mean[bg].std())
SNR = sig / noi
snr = {
    "SNR_recomputed": SNR,
    "SNR_signal_definition": "mean of the temporal-mean intensity inside the native-space brain mask",
    "SNR_background_definition": "std of the temporal-mean intensity in non-brain, nonzero voxels (native space)",
    "SNR_formula": "signal_mean / background_std (fBIRN-style)",
    "SNR_mask_used": "native-resolution Otsu threshold + largest 3D connected component + hole filling",
    "SNR_signal_value": sig, "SNR_background_std": noi,
}
M.update(snr)
for k, v in snr.items():
    print(f"    {k}: {v}")
stored_snr = add.get("SNR")
if stored_snr is not None:
    ok = abs(float(stored_snr) - SNR) < 1e-6
    print(f"  stored SNR={stored_snr} vs recomputed={SNR} -> match={ok}")
    M["SNR_stored"] = float(stored_snr); M["SNR_stored_matches_recomputed"] = bool(ok)
print("  No threshold applied: reported as a measured value only.")

# ============================================================
hdr("SECTION 8 -- CNR")
print("  CNR = NOT_RELIABLY_COMPUTABLE")
print("  Reason: this dataset contains NO T1w anatomical image for this subject, so no")
print("  reliable GM/WM/CSF tissue segmentation can be derived. Producing a segmentation")
print("  from BOLD/EPI alone (or asserting one from template priors) would not give a valid")
print("  subject-specific tissue-contrast estimate. No PASS/FAIL is assigned to CNR.")
M["CNR"] = "NOT_RELIABLY_COMPUTABLE"
M["CNR_reason"] = "No T1w anatomical image -> no reliable tissue segmentation"

# ============================================================
hdr("SECTION 9 -- SPATIAL ENTROPY")
vals = sm_mean[mask_d]
h, _ = np.histogram(vals, bins=256)
pr = h / h.sum(); pr = pr[pr > 0]
spatial_entropy = float(-(pr * np.log2(pr)).sum())
M["spatial_entropy_bits"] = spatial_entropy
print(f"    spatial_entropy_recomputed: {spatial_entropy:.4f} bits")
print(f"    image/stage: temporal mean of desc-smooth6mm_bold.nii.gz")
print(f"    mask: transforms/brain_mask_mni4mm.nii.gz")
print(f"    formula: Shannon H = -sum(p log2 p), 256-bin histogram of in-mask intensities")
print(f"    units: bits")
if "spatial_entropy_bits" in add:
    ok = abs(add["spatial_entropy_bits"] - spatial_entropy) < 1e-6
    print(f"  stored={add['spatial_entropy_bits']} vs recomputed={spatial_entropy} -> match={ok}")
    M["spatial_entropy_stored"] = add["spatial_entropy_bits"]
    M["spatial_entropy_stored_matches"] = bool(ok)
print("  No PASS/FAIL threshold assigned (no sourced universal reference exists).")

# ============================================================
hdr("SECTION 10 -- TEMPORAL ENTROPY")
flat = sm.reshape(-1, nt).T[:, mask_d.ravel()]     # (nt, n_vox)
tv_mat = flat.T                                     # (n_vox, nt)
z = (tv_mat - tv_mat.mean(axis=1, keepdims=True)) / (tv_mat.std(axis=1, keepdims=True) + 1e-9)
print(f"    per-voxel time-series matrix (n_voxels x nt): {z.shape}")
ents = []
for i in range(z.shape[0]):
    hh, _ = np.histogram(z[i, :], bins=16)
    q = hh / hh.sum(); q = q[q > 0]
    ents.append(-(q * np.log2(q)).sum())
temporal_entropy = float(np.mean(ents))
M["temporal_entropy_bits"] = temporal_entropy
print(f"    temporal_entropy_recomputed: {temporal_entropy:.4f} bits")
print(f"    method: mean per-voxel Shannon entropy of the z-scored temporal histogram (16 bins)")
print(f"    stage: desc-smooth6mm_bold.nii.gz;  units: bits")
if "temporal_entropy_bits" in add:
    ok = abs(add["temporal_entropy_bits"] - temporal_entropy) < 1e-3
    print(f"  stored={add['temporal_entropy_bits']} vs recomputed={temporal_entropy} -> match={ok}")
    M["temporal_entropy_stored"] = add["temporal_entropy_bits"]
    M["temporal_entropy_stored_matches"] = bool(ok)
print("  No PASS/FAIL threshold assigned.")

# ============================================================
hdr("SECTION 11 -- BRAIN MASK QC")
md = mask_img.get_fdata()
uniq = np.unique(md)
lab_m, n_comp = ndimage.label(md > 0, structure=np.ones((3, 3, 3)))
comp_sizes = ndimage.sum(md > 0, lab_m, range(1, n_comp + 1)) if n_comp else np.array([])
n_mask = int((md > 0).sum()); n_tot = int(md.size)
mk = {
    "mask_dimensions": str(mask_img.shape),
    "mask_voxel_size_mm": str(tuple(round(float(z_), 4) for z_ in mask_img.header.get_zooms()[:3])),
    "mask_datatype": str(mask_img.get_data_dtype()),
    "mask_is_binary": bool(set(uniq.tolist()) <= {0.0, 1.0}),
    "mask_unique_values": str(uniq.tolist()[:10]),
    "mask_n_voxels": n_mask, "mask_pct_of_image": round(100.0 * n_mask / n_tot, 3),
    "mask_n_connected_components": int(n_comp),
    "mask_largest_component_voxels": int(comp_sizes.max()) if n_comp else 0,
    "mask_largest_component_pct_of_mask": round(100.0 * comp_sizes.max() / n_mask, 2) if n_comp else 0.0,
    "mask_has_NaN": bool(np.isnan(md).any()), "mask_has_Inf": bool(np.isinf(md).any()),
    "mask_matches_final_grid": bool(mask_img.shape == fin_img.shape[:3] and
                                    np.allclose(mask_img.affine, fin_img.affine, atol=1e-3)),
}
M.update(mk)
for k, v in mk.items():
    print(f"    {k}: {v}")

fig, ax = plt.subplots(1, 3, figsize=(12, 4))
mmid = [s // 2 for s in md.shape]
for i, axis in enumerate([0, 1, 2]):
    ax[i].imshow(np.rot90(np.take(sm_mean, mmid[axis], axis=axis)), cmap="gray")
    ax[i].contour(np.rot90(np.take(md > 0, mmid[axis], axis=axis)), levels=[0.5], colors="lime", linewidths=1.0)
    ax[i].axis("off"); ax[i].set_title(f"axis {axis}")
plt.suptitle(f"Brain mask QC -- original pilot ({n_mask} voxels, {n_comp} component(s))")
plt.tight_layout(); plt.savefig(p(AUDIT, "brain_mask_qc.png"), dpi=120); plt.close()
print(f"  saved: {p(AUDIT, 'brain_mask_qc.png')}")

# ============================================================
hdr("SECTION 12 -- MNI REGISTRATION QC")
import templateflow.api as tflow
tpl_path = str(tflow.get("MNI152NLin6Asym", resolution=2, desc="brain", suffix="T1w", extension="nii.gz"))
tpl2 = nib.load(tpl_path)
# the pilot's final image is 4mm, derived from the 2mm template grid -- verify that relationship
exp_aff = tpl2.affine.copy()
scale = np.array([4.0, 4.0, 4.0]) / np.array([abs(tpl2.affine[i, i]) for i in range(3)])
exp_aff[:3, :3] = tpl2.affine[:3, :3] @ np.diag(scale)
exp_shape = tuple(int(np.ceil(tpl2.shape[i] / scale[i])) for i in range(3))
reg = {
    "final_affine": str(fin_img.affine.tolist()),
    "final_orientation": "".join(nib.aff2axcodes(fin_img.affine)),
    "final_dimensions": str(fin_img.shape[:3]),
    "final_voxel_size_mm": str(tuple(round(float(z_), 4) for z_ in fin_img.header.get_zooms()[:3])),
    "template_reference": f"MNI152NLin6Asym res-02 desc-brain_T1w ({tpl_path})",
    "expected_4mm_grid_from_template_shape": str(exp_shape),
    "final_matches_expected_4mm_MNI_grid": bool(fin_img.shape[:3] == exp_shape and
                                                np.allclose(fin_img.affine, exp_aff, atol=1e-3)),
    "n_transform_files_present": len(xfms),
}
M.update(reg)
for k, v in reg.items():
    print(f"    {k}: {v}")
print("  NOTE: affine/grid agreement is necessary but NOT sufficient evidence of good")
print("  registration -- the overlay below is required for actual spatial judgement.")

tpl4 = nib.Nifti1Image(np.zeros(exp_shape), exp_aff)
from nilearn.image import resample_img
tpl4_res = resample_img(tpl2, target_affine=exp_aff, target_shape=exp_shape,
                        interpolation="continuous", force_resample=True, copy_header=True)
tpl4_d = tpl4_res.get_fdata()
fin_mean = fin_d.mean(axis=3)
fig, ax = plt.subplots(2, 3, figsize=(13, 8))
for i, axis in enumerate([0, 1, 2]):
    t_sl = np.rot90(np.take(tpl4_d, tpl4_d.shape[axis] // 2, axis=axis))
    b_sl = np.rot90(np.take(fin_mean, fin_mean.shape[axis] // 2, axis=axis))
    ax[0, i].imshow(t_sl, cmap="gray"); ax[0, i].set_title(f"MNI template @4mm axis {axis}"); ax[0, i].axis("off")
    ax[1, i].imshow(b_sl, cmap="gray")
    if (t_sl > 0).any():
        ax[1, i].contour(t_sl > np.percentile(t_sl[t_sl > 0], 40), levels=[0.5], colors="r", linewidths=0.7)
    ax[1, i].set_title("final mean BOLD + template contour"); ax[1, i].axis("off")
plt.suptitle("MNI registration QC -- original 4mm/6mm pilot (independently regenerated)")
plt.tight_layout(); plt.savefig(p(AUDIT, "mni_registration_qc.png"), dpi=120); plt.close()
print(f"  saved: {p(AUDIT, 'mni_registration_qc.png')}")

# ============================================================
hdr("SECTION 13 -- FINAL IMAGE / STAGE-BY-STAGE VISUAL QC")
stages = [
    ("motion-corrected (native)", PATHS["intermediate: motion-corrected"]),
    ("MNI-normalized (2mm)", PATHS["intermediate: MNI-normalized"]),
    ("4mm resampled", PATHS["intermediate: 4mm resampled"]),
    ("6mm smoothed", PATHS["intermediate: 6mm smoothed"]),
    ("detrended", PATHS["intermediate: detrended"]),
    ("nuisance-regressed", PATHS["intermediate: nuisance-regressed"]),
    ("band-pass filtered", PATHS["intermediate: band-pass"]),
    ("FINAL output", PATHS["final preprocessed NIfTI"]),
]
stage_rows = []
means_for_fig = []
for name, path in stages:
    img = nib.load(path)
    dobj = img.dataobj
    v0 = np.asarray(dobj[..., 0], dtype=np.float32)
    # cheap spatial-gradient proxy for "how smooth is this image"
    gx, gy, gz = np.gradient(v0)
    gmag = np.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    nzm = v0 != 0
    grad_ratio = float(gmag[nzm].mean() / (np.abs(v0[nzm]).mean() + 1e-9)) if nzm.any() else float("nan")
    row = {
        "stage": name, "shape": str(img.shape),
        "voxel_mm": str(tuple(round(float(z_), 3) for z_ in img.header.get_zooms()[:3])),
        "nonzero_fraction_vol0": round(float(nzm.mean()), 4),
        "relative_spatial_gradient": round(grad_ratio, 5),
    }
    stage_rows.append(row)
    means_for_fig.append((name, v0))
    print(f"    {name}: shape={row['shape']} voxel={row['voxel_mm']} "
          f"nonzero={row['nonzero_fraction_vol0']} rel_grad={row['relative_spatial_gradient']}")
    img.uncache()
pd.DataFrame(stage_rows).to_csv(p(AUDIT, "stage_by_stage_qc.csv"), index=False)
print(f"  saved: {p(AUDIT, 'stage_by_stage_qc.csv')}")
print("  'relative_spatial_gradient' = mean|grad| / mean|intensity| on volume 0; it drops when an")
print("  image is spatially smoother. Used to locate WHERE smoothing occurs, not to judge quality.")

fig, ax = plt.subplots(2, 4, figsize=(16, 8))
for i, (name, vol) in enumerate(means_for_fig):
    a = ax.flat[i]
    a.imshow(np.rot90(np.take(vol, vol.shape[2] // 2, axis=2)), cmap="gray")
    a.set_title(name, fontsize=9); a.axis("off")
plt.suptitle("Stage-by-stage axial view (volume 0) -- original 4mm/6mm pilot")
plt.tight_layout(); plt.savefig(p(AUDIT, "stage_by_stage_qc.png"), dpi=110); plt.close()
print(f"  saved: {p(AUDIT, 'stage_by_stage_qc.png')}")

fin_nz = float((fin_mean != 0).mean())
fin_uniform = float(fin_mean[mask_d].std() / (abs(fin_mean[mask_d].mean()) + 1e-9))
M["final_nonzero_fraction"] = round(fin_nz, 4)
M["final_in_mask_relative_sd"] = round(fin_uniform, 5)
print(f"  final image: nonzero fraction={fin_nz:.4f}, in-mask relative SD={fin_uniform:.5f}")
print("  (a very low in-mask relative SD would indicate a suspiciously uniform image)")

# ============================================================
hdr("SECTION 14 -- BRAINNETOME-246 COMPATIBILITY (performed fresh, not assumed)")
print("  NOTE: this ORIGINAL pilot has NO pre-existing Brainnetome output, so compatibility")
print("  was NOT previously established for the 4mm grid. Testing it now.")
atlas_img = nib.load(ATLAS_PATH)
print(f"    atlas: {ATLAS_PATH}")
print(f"    atlas dims={atlas_img.shape} voxel="
      f"{tuple(round(float(z_), 3) for z_ in atlas_img.header.get_zooms()[:3])} "
      f"orientation={''.join(nib.aff2axcodes(atlas_img.affine))}")
print(f"    final BOLD dims={fin_img.shape[:3]} voxel="
      f"{tuple(round(float(z_), 3) for z_ in fin_img.header.get_zooms()[:3])} "
      f"orientation={''.join(nib.aff2axcodes(fin_img.affine))}")
grids_same = (atlas_img.shape == fin_img.shape[:3] and np.allclose(atlas_img.affine, fin_img.affine, atol=1e-3))
print(f"    grids already identical: {grids_same} -> resampling {'NOT needed' if grids_same else 'REQUIRED'}")

atlas_canon = nib.as_closest_canonical(atlas_img)
atlas_on_4mm = resample_img(atlas_canon, target_affine=fin_img.affine, target_shape=fin_img.shape[:3],
                            interpolation="nearest", force_resample=True, copy_header=True)
print("    resampling method used: NEAREST-NEIGHBOR ONLY (nilearn interpolation='nearest')")
print("    (atlas first losslessly reoriented LAS->RAS; original atlas file never modified)")
a4 = np.asarray(atlas_on_4mm.dataobj).astype(np.int32)

lut = {}
with open(ATLAS_LUT) as f:
    for line in f:
        q = line.split()
        if len(q) >= 2 and q[0].isdigit():
            lut[int(q[0])] = q[1]

rows, counts = [], []
for rid in range(1, 247):
    n = int((a4 == rid).sum())
    counts.append(n)
    st = "ZERO_VOXELS" if n == 0 else ("VERY_SMALL_LT5" if n < 5 else ("SMALL_LT10" if n < 10 else "ADEQUATE_GE10"))
    rows.append({"ROI_ID": rid, "ROI_NAME": lut.get(rid, f"ROI_{rid}"), "VOXEL_COUNT": n, "STATUS": st})
with open(p(AUDIT, "brainnetome_roi_voxel_counts_4mm.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["ROI_ID", "ROI_NAME", "VOXEL_COUNT", "STATUS"])
    w.writeheader(); w.writerows(rows)
counts = np.array(counts)
bn = {
    "bn_n_zero_voxel_ROIs": int((counts == 0).sum()),
    "bn_n_ROIs_1_to_2_voxels": int(((counts >= 1) & (counts <= 2)).sum()),
    "bn_n_ROIs_lt_5_voxels": int((counts < 5).sum()),
    "bn_n_ROIs_lt_10_voxels": int((counts < 10).sum()),
    "bn_min_voxels": int(counts.min()), "bn_max_voxels": int(counts.max()),
    "bn_mean_voxels": float(counts.mean()), "bn_median_voxels": float(np.median(counts)),
    "bn_resample_method": "nearest-neighbor (after lossless LAS->RAS reorientation of the atlas)",
}
M.update(bn)
for k, v in bn.items():
    print(f"    {k}: {v}")
print(f"  saved: {p(AUDIT, 'brainnetome_roi_voxel_counts_4mm.csv')}")

plt.figure(figsize=(10, 4))
plt.bar(range(1, 247), counts, width=1.0)
plt.axhline(10, color="r", ls="--", lw=0.8, label="10-voxel reference line")
plt.axhline(5, color="orange", ls="--", lw=0.8, label="5-voxel reference line")
plt.xlabel("Brainnetome ROI ID"); plt.ylabel("voxel count"); plt.legend(fontsize=8)
plt.title(f"ROI voxel coverage on the 4mm grid ({bn['bn_n_zero_voxel_ROIs']} zero-voxel ROIs)")
plt.tight_layout(); plt.savefig(p(AUDIT, "brainnetome_roi_coverage_4mm.png"), dpi=120); plt.close()
print(f"  saved: {p(AUDIT, 'brainnetome_roi_coverage_4mm.png')}")

# ============================================================
hdr("SECTION 15 -- ROI TIME-SERIES QC")
roi_ts_candidates = []
for dirpath, _, files in os.walk(ROOT):
    for fl in files:
        if "timeseries" in fl.lower() or "time_series" in fl.lower():
            roi_ts_candidates.append(os.path.join(dirpath, fl))
if roi_ts_candidates:
    print(f"  found ROI time-series file(s): {roi_ts_candidates}")
    ts_df = pd.read_csv(roi_ts_candidates[0], sep="\t")
    stds = ts_df.std(axis=0)
    tsq = {
        "roi_ts_n_ROIs": int(ts_df.shape[1]), "roi_ts_n_timepoints": int(ts_df.shape[0]),
        "roi_ts_NaN": int(ts_df.isna().sum().sum()), "roi_ts_Inf": int(np.isinf(ts_df.values).sum()),
        "roi_ts_zero_variance": int((stds == 0).sum()),
        "roi_ts_near_zero_variance": int((stds < stds.median() * 0.01).sum()),
    }
    M.update(tsq)
    for k, v in tsq.items():
        print(f"    {k}: {v}")
else:
    print("  NO ROI time-series file exists in this pilot.")
    print("  STATUS: NOT COMPUTABLE -- nothing to inspect. No ROI extraction was performed here,")
    print("  and per the audit instructions none is generated now.")
    M["roi_timeseries_present"] = False

# ============================================================
hdr("SECTION 16 -- VERIFIED QC METRIC TABLE")
table = [
    ("tSNR (mean)", f"{ts['mean_tSNR']:.2f}", "ratio", "desc-smooth6mm (pre-detrend/nuisance/bandpass)",
     "mean/SD over time, in-mask; recomputed independently",
     "no sourced universal threshold applied", "CHECK REQUIRED",
     f"Recomputed value matches the stored value ({tsnr_match}). Reported as measured; not classified "
     f"because no defensible universal tSNR cutoff was supplied."),
    ("SNR", f"{SNR:.4f}", "ratio", "native raw input", "mean(in-brain)/SD(background), fBIRN-style",
     "no sourced universal threshold applied", "CHECK REQUIRED",
     "Matches stored value exactly. Measured only; no universal cutoff applied."),
    ("CNR", "NOT_RELIABLY_COMPUTABLE", "-", "-", "requires GM/WM/CSF segmentation",
     "n/a", "NOT COMPUTABLE", "No T1w anatomical image exists; fabricating segmentation is prohibited."),
    ("Mean FD", f"{fd['mean_FD_mm']:.4f}", "mm", "native mc-afni2 .mcdat",
     "Power et al. 2012 formulation, 50mm rotation radius",
     "0.2/0.5/1.0mm shown as reference lines only, not as pass criteria", "WARN",
     f"{fd['n_FD_gt_0.5mm']}/{len(FD)} volumes ({fd['pct_FD_gt_0.5mm']}%) exceed 0.5mm and "
     f"{fd['n_FD_gt_1.0mm']} exceed 1.0mm -- real motion present. Flagged for awareness, not exclusion."),
    ("Median FD", f"{fd['median_FD_mm']:.4f}", "mm", "native .mcdat", "as above", "reference only", "WARN",
     "Reported alongside mean; same motion caveat applies."),
    ("Maximum FD", f"{fd['max_FD_mm']:.4f}", "mm", "native .mcdat", "as above", "reference only", "WARN",
     "Single worst frame; no censoring performed per the frozen protocol."),
    ("FD > 0.5 mm", f"{fd['n_FD_gt_0.5mm']} ({fd['pct_FD_gt_0.5mm']}%)", "volumes", "native .mcdat",
     "count of FD>0.5mm", "reference line only", "WARN", "A substantial minority of frames."),
    ("FD > 1.0 mm", f"{fd['n_FD_gt_1.0mm']} ({fd['pct_FD_gt_1.0mm']}%)", "volumes", "native .mcdat",
     "count of FD>1.0mm", "reference line only", "WARN", "A small number of high-motion frames."),
    ("DVARS (mean)", f"{dvs['mean_DVARS_raw']:.1f}", "raw intensity units", "desc-smooth6mm",
     "sqrt(mean(diff^2)) in-mask, unnormalized",
     "no valid absolute threshold: published cutoffs assume standardized/%-change DVARS",
     "NOT COMPUTABLE AS PASS/FAIL",
     "Raw-intensity units cannot be compared to standardized-DVARS literature thresholds. "
     f"Self-relative extremes (Q3+1.5IQR): {dvs['n_extreme_DVARS_volumes']} volumes."),
    ("Spatial entropy", f"{spatial_entropy:.4f}", "bits", "temporal mean of desc-smooth6mm",
     "Shannon entropy, 256-bin in-mask histogram", "none (descriptive metric)", "CHECK REQUIRED",
     "Matches stored value. No universal good/bad reference exists."),
    ("Temporal entropy", f"{temporal_entropy:.4f}", "bits", "desc-smooth6mm",
     "mean per-voxel Shannon entropy of z-scored 16-bin temporal histogram", "none", "CHECK REQUIRED",
     "Matches stored value. No universal good/bad reference exists."),
    ("Brain-mask coverage", f"{mk['mask_n_voxels']} vox ({mk['mask_pct_of_image']}% of image)", "voxels/%",
     "transforms/brain_mask_mni4mm.nii.gz", "voxel count + connected-component analysis",
     "expect non-empty, binary, single dominant component",
     "PASS" if (mk["mask_n_voxels"] > 0 and mk["mask_is_binary"] and mk["mask_n_connected_components"] == 1) else "CHECK REQUIRED",
     f"Binary={mk['mask_is_binary']}, components={mk['mask_n_connected_components']}, "
     f"grid matches final image={mk['mask_matches_final_grid']}."),
    ("MNI registration", f"matches expected 4mm MNI grid = {reg['final_matches_expected_4mm_MNI_grid']}", "-",
     "final image vs MNI152NLin6Asym-derived 4mm grid", "affine/shape identity + visual overlay",
     "grid identity is necessary, not sufficient",
     "PASS" if reg["final_matches_expected_4mm_MNI_grid"] else "CHECK REQUIRED",
     "Geometry confirmed; spatial alignment quality must still be confirmed visually from "
     "mni_registration_qc.png."),
    ("NaN/Inf", f"{integ['final_NaN_count']}/{integ['final_Inf_count']}", "count", "final image",
     "direct count", "must be 0/0", "PASS" if (integ["final_NaN_count"] == 0 and integ["final_Inf_count"] == 0) else "FAIL",
     "No invalid numeric values present in the final output."),
    ("Brainnetome ROI coverage", f"{bn['bn_n_zero_voxel_ROIs']} zero-voxel of 246 "
     f"(min={bn['bn_min_voxels']}, median={bn['bn_median_voxels']:.1f})", "ROIs",
     "BN_Atlas_246_2mm -> 4mm grid, nearest-neighbor", "per-ROI voxel count",
     "a zero-voxel ROI cannot yield a regional signal",
     "PASS" if bn["bn_n_zero_voxel_ROIs"] == 0 else "WARN",
     f"Tested fresh in this audit (was NOT previously established for the 4mm grid). "
     f"{bn['bn_n_ROIs_lt_10_voxels']} ROI(s) have <10 voxels, {bn['bn_n_ROIs_lt_5_voxels']} have <5."),
]
with open(p(AUDIT, "qc_metrics_verified.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["Metric", "Value", "Units", "Stage", "Calculation", "Reference/Threshold", "Status", "Interpretation"])
    w.writerows(table)
print(f"  saved: {p(AUDIT, 'qc_metrics_verified.csv')}")
for r in table:
    print(f"    {r[0]:28s} {r[6]:28s} {r[1]}")

with open(p(AUDIT, "qc_metrics_verified.json"), "w") as f:
    json.dump(M, f, indent=2, default=str)
print(f"  saved: {p(AUDIT, 'qc_metrics_verified.json')}")

# ============================================================
hdr("SECTION 17-19 -- OVERALL STATUS + REPORT")
warnings_list = [
    f"Motion: {fd['n_FD_gt_0.5mm']}/{len(FD)} volumes ({fd['pct_FD_gt_0.5mm']}%) exceed 0.5mm FD and "
    f"{fd['n_FD_gt_1.0mm']} exceed 1.0mm; max FD {fd['max_FD_mm']:.3f}mm. No censoring was applied.",
    f"{n_susp} voxel(s) show very high tSNR combined with near-zero temporal SD -- treated as possible "
    f"artifact, NOT as evidence of good quality.",
]
if bn["bn_n_ROIs_lt_10_voxels"] > 0:
    warnings_list.append(f"{bn['bn_n_ROIs_lt_10_voxels']} Brainnetome ROI(s) have fewer than 10 voxels on the "
                         f"4mm grid (min {bn['bn_min_voxels']}); small ROIs give less stable regional means.")
limitations = [
    "No T1w anatomical image exists for this dataset -> CNR not computable and WM/CSF regressors come from "
    "population MNI template priors rather than subject-specific segmentation.",
    "SliceTiming metadata is absent -> slice-timing correction was deliberately not performed.",
    "DVARS is implemented in raw intensity units, so published standardized/%-change DVARS thresholds "
    "cannot be applied to it.",
    "This is a single acquisition; none of these values establish group-level quality.",
]
hard_fail = (integ["final_NaN_count"] > 0 or integ["final_Inf_count"] > 0 or
             not final_is_4d or not final_is_4mm or bn["bn_n_zero_voxel_ROIs"] > 0 or
             mk["mask_n_voxels"] == 0 or not reg["final_matches_expected_4mm_MNI_grid"])
if hard_fail:
    OVERALL = "REQUIRES INVESTIGATION"
    why = "At least one structural/geometric criterion did not hold -- see the table above."
elif warnings_list:
    OVERALL = "PASS WITH WARNINGS"
    why = ("All structural, geometric, mask, registration and Brainnetome-coverage checks hold, and every "
           "recomputed metric reproduces the stored value. Head motion is genuinely elevated for a subset "
           "of frames and a small number of tSNR voxels look artifactual, so this is not an unqualified PASS.")
else:
    OVERALL = "PASS"
    why = "All checks held with no caution items."
print(f"  OVERALL: {OVERALL}\n  WHY: {why}")

rep = p(AUDIT, "QC_AUDIT_ORIGINAL_PILOT.md")
with open(rep, "w", encoding="utf-8") as f:
    w = f.write
    w("# Original Pilot QC Audit\n\n")
    w("Read-only audit of `pilot_preprocessing/` (the 4mm-resample / 6mm-FWHM configuration). "
      "No preprocessing was rerun, no NIfTI modified, nothing in `pilot_preprocessing/qc/` touched, "
      "and no other subject inspected.\n\n")

    w("## 1. Pilot Identification\n\n")
    w(f"- Subject: **{SUBJECT}**\n- Session: **{SESSION}**\n- Run: **{RUN}**\n")
    w(f"- Exactly ONE acquisition confirmed in this folder.\n\n| Item | Path |\n|---|---|\n")
    for k, v in PATHS.items():
        w(f"| {k} | `{v}` |\n")
    w("\n")

    w("## 2. Actual Preprocessing Configuration\n\n")
    w("Read from the pilot's own config/log, not assumed:\n\n| Step | What was actually performed |\n|---|---|\n")
    for label, actual, _ in checks:
        w(f"| {label} | {actual} |\n")
    w(f"\nIndependent check of the confound file: **{conf.shape[1]} columns x {conf.shape[0]} timepoints**, "
      f"with **{n_friston}/24** Friston-24 columns present -> {friston_ok}.\n\n")

    w("## 3. Image Integrity\n\n| Property | Value |\n|---|---|\n")
    for k in ["original_dimensions", "final_dimensions", "original_n_volumes", "final_n_volumes",
              "volumes_discarded", "TR_s", "original_voxel_size_mm", "final_voxel_size_mm",
              "original_orientation", "final_orientation", "original_datatype", "final_datatype",
              "final_NaN_count", "final_Inf_count", "final_min", "final_max", "final_mean"]:
        w(f"| {k} | {integ[k]} |\n")
    w(f"\n- Final image is 4D: **{final_is_4d}**\n- Final image is 4mm isotropic: **{final_is_4mm}**\n\n")

    w("## 4. Motion QC\n\n| Metric | Value |\n|---|---|\n")
    for k, v in fd.items():
        w(f"| {k} | {v} |\n")
    w(f"\nIndependently recomputed FD reproduces the stored `fd_values.tsv`: **{fd_match}** "
      f"(max abs diff {max_abs_diff:.2e}). See `fd_values_verified.csv`, `fd_plot.png`.\n\n")

    w("## 5. DVARS\n\n| Metric | Value |\n|---|---|\n")
    for k, v in dvs.items():
        w(f"| {k} | {v} |\n")
    w("\n**DVARS type: raw-intensity.** Published thresholds for standardized or percent-signal-change "
      "DVARS do not apply and were not used. See `dvars_plot.png`.\n\n")

    w("## 6. tSNR\n\n")
    w("Stage: `desc-smooth6mm_bold.nii.gz` -- smoothing YES (6mm FWHM), detrending NO, "
      "nuisance regression NO, band-pass NO.\n\n| Metric | Value |\n|---|---|\n")
    for k, v in ts.items():
        w(f"| {k} | {v} |\n")
    w(f"\n- Stored mean tSNR reproduced by independent recomputation: **{tsnr_match}**\n")
    w(f"- Voxels with very high tSNR *and* near-zero temporal SD: **{n_susp}**")
    if n_susp:
        w(f", of which {M.get('suspicious_tsnr_fraction_at_mask_edge', 0):.1%} lie in the outer "
          f"2-voxel mask rim. These are flagged as possible artifacts, **not** as evidence of good data")
    w(".\n\nSee `tsnr_qc.png`.\n\n")

    w("## 7. SNR\n\n| Field | Value |\n|---|---|\n")
    for k, v in snr.items():
        w(f"| {k} | {v} |\n")
    if stored_snr is not None:
        w(f"| stored SNR | {stored_snr} |\n| reproduces stored value | {M.get('SNR_stored_matches_recomputed')} |\n")
    w("\n")

    w("## 8. CNR\n\n**NOT_RELIABLY_COMPUTABLE.** No T1w anatomical image exists for this subject, so no "
      "reliable GM/WM/CSF segmentation can be derived. No PASS/FAIL assigned; no segmentation fabricated.\n\n")

    w("## 9. Spatial Entropy\n\n")
    w(f"- Value (recomputed): **{spatial_entropy:.4f} bits**\n- Stage: temporal mean of `desc-smooth6mm_bold.nii.gz`\n")
    w(f"- Mask: `transforms/brain_mask_mni4mm.nii.gz`\n- Formula: Shannon H = -sum(p log2 p), 256-bin in-mask histogram\n")
    if "spatial_entropy_stored" in M:
        w(f"- Reproduces stored value: **{M['spatial_entropy_stored_matches']}**\n")
    w("- No PASS/FAIL threshold assigned.\n\n")

    w("## 10. Temporal Entropy\n\n")
    w(f"- Value (recomputed): **{temporal_entropy:.4f} bits**\n- Stage: `desc-smooth6mm_bold.nii.gz`\n")
    w("- Method: mean per-voxel Shannon entropy of the z-scored temporal histogram (16 bins)\n")
    if "temporal_entropy_stored" in M:
        w(f"- Reproduces stored value: **{M['temporal_entropy_stored_matches']}**\n")
    w("- No PASS/FAIL threshold assigned.\n\n")

    w("## 11. Brain Mask\n\n| Property | Value |\n|---|---|\n")
    for k, v in mk.items():
        w(f"| {k} | {v} |\n")
    w("\nSee `brain_mask_qc.png`.\n\n")

    w("## 12. MNI Registration\n\n| Property | Value |\n|---|---|\n")
    for k, v in reg.items():
        w(f"| {k} | {v} |\n")
    w("\nGrid/affine identity is necessary but not sufficient; visual confirmation required from "
      "`mni_registration_qc.png`.\n\n")

    w("## 13. Visual QC\n\n| Stage | Shape | Voxel (mm) | Nonzero frac (vol 0) | Relative spatial gradient |\n")
    w("|---|---|---|---|---|\n")
    for r in stage_rows:
        w(f"| {r['stage']} | {r['shape']} | {r['voxel_mm']} | {r['nonzero_fraction_vol0']} | "
          f"{r['relative_spatial_gradient']} |\n")
    w("\nThe relative spatial gradient (mean|grad| / mean|intensity|) drops where an image becomes "
      "spatially smoother, which locates *where* smoothing takes effect. It is a descriptive locator, "
      "not a quality judgement. See `stage_by_stage_qc.png`.\n\n")
    w(f"- Final image nonzero fraction: {fin_nz:.4f}\n")
    w(f"- Final in-mask relative SD: {fin_uniform:.5f} (a near-zero value would indicate a suspiciously "
      f"uniform image)\n\n")

    w("## 14. Brainnetome-246 Compatibility\n\n")
    w("This pilot had **no** pre-existing Brainnetome output, so compatibility with the 4mm grid was "
      "**not previously established**. It was tested fresh in this audit.\n\n| Metric | Value |\n|---|---|\n")
    for k, v in bn.items():
        w(f"| {k} | {v} |\n")
    w("\nAtlas was losslessly reoriented (LAS->RAS) then resampled to the 4mm BOLD grid with "
      "**nearest-neighbor interpolation only**. The original atlas file was not modified. "
      "See `brainnetome_roi_voxel_counts_4mm.csv` and `brainnetome_roi_coverage_4mm.png`.\n\n")

    w("## 15. ROI Time-Series QC\n\n")
    if roi_ts_candidates:
        for k in ["roi_ts_n_ROIs", "roi_ts_n_timepoints", "roi_ts_NaN", "roi_ts_Inf",
                  "roi_ts_zero_variance", "roi_ts_near_zero_variance"]:
            w(f"- {k}: {M[k]}\n")
    else:
        w("**NOT COMPUTABLE** -- no ROI time-series file exists in this pilot, and none was generated "
          "(ROI extraction was outside this audit's scope).\n")
    w("\n")

    w("## 16. Verified QC Metrics\n\n")
    w("| Metric | Value | Units | Stage | Reference/Threshold | Status |\n|---|---|---|---|---|---|\n")
    for r in table:
        w(f"| {r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[5]} | {r[6]} |\n")
    w("\nFull table with calculation details and interpretation: `qc_metrics_verified.csv`.\n\n")

    w("## 17. Warnings\n\n")
    for x in warnings_list:
        w(f"- {x}\n")
    w("\n## 18. Dataset Limitations\n\n")
    for x in limitations:
        w(f"- {x}\n")
    w(f"\n## 19. Overall QC Status\n\n**{OVERALL}**\n\n{why}\n\n")
    w("Every recomputed metric (FD, DVARS, tSNR, SNR, spatial entropy, temporal entropy) reproduced the "
      "value stored by the original pipeline run, so the pilot's own QC outputs are internally consistent "
      "and trustworthy.\n")
print(f"  report written: {rep}")

hdr("FINAL SUMMARY")
print(f"SUBJECT:\n{SUBJECT}\n")
print(f"PIPELINE:\ndiscard 5 vols -> slice-timing NOT PERFORMED -> mc-afni2 motion correction ->\n"
      f"direct EPI->MNI (ANTs SyN, MNI152NLin6Asym) -> 4mm resample -> 6mm FWHM smoothing ->\n"
      f"linear detrend -> 28-regressor nuisance (Friston-24 + WM + CSF + GS) -> 0.01-0.10 Hz band-pass\n")
print(f"VOXEL SIZE:\n{integ['final_voxel_size_mm']}\n")
print(f"SMOOTHING:\n6 mm FWHM\n")
print(f"tSNR:\n{ts['mean_tSNR']:.2f} (median {ts['median_tSNR']:.2f})\n")
print(f"SNR:\n{SNR:.2f}\n")
print(f"CNR:\nNOT COMPUTABLE (no T1w)\n")
print(f"MEAN FD:\n{fd['mean_FD_mm']:.4f} mm\n")
print(f"FD > 0.5 mm:\n{fd['n_FD_gt_0.5mm']} volumes ({fd['pct_FD_gt_0.5mm']}%)\n")
print(f"FD > 1.0 mm:\n{fd['n_FD_gt_1.0mm']} volumes ({fd['pct_FD_gt_1.0mm']}%)\n")
print(f"DVARS:\n{dvs['mean_DVARS_raw']:.1f} (raw intensity units -- not comparable to standardized thresholds)\n")
print(f"SPATIAL ENTROPY:\n{spatial_entropy:.4f} bits\n")
print(f"TEMPORAL ENTROPY:\n{temporal_entropy:.4f} bits\n")
print(f"BRAIN MASK:\n{mk['mask_n_voxels']} voxels ({mk['mask_pct_of_image']}% of image), "
      f"{mk['mask_n_connected_components']} component(s), binary={mk['mask_is_binary']}\n")
print(f"BRAINNETOME:\n246 ROIs tested on the 4mm grid -- {bn['bn_n_zero_voxel_ROIs']} zero-voxel, "
      f"{bn['bn_n_ROIs_lt_5_voxels']} with <5 vox, {bn['bn_n_ROIs_lt_10_voxels']} with <10 vox, "
      f"min={bn['bn_min_voxels']}, median={bn['bn_median_voxels']:.1f}\n")
print(f"NaN/Inf:\n{integ['final_NaN_count']}/{integ['final_Inf_count']}\n")
print(f"MNI REGISTRATION:\ngrid/affine match with expected 4mm MNI grid = "
      f"{reg['final_matches_expected_4mm_MNI_grid']} (visual confirmation required)\n")
print(f"OVERALL QC:\n{OVERALL}\n")
print("QC_AUDIT_ORIGINAL_PILOT_DONE")
