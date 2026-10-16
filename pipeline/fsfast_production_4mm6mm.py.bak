"""
FINAL PRODUCTION preprocessing: 4mm MNI + 6mm FWHM + linear detrend + 28-regressor
nuisance (Friston-24 + WM + CSF + GS) + 0.01-0.10 Hz band-pass, for all six groups
(AD, CN_Final, EMCI, LMCI, MCI, SMC_Final).

This is the configuration validated by the single-subject pilot
(pilot_preprocessing/, sub-019S4549/ses-01) and its QC audit. Frozen parameters:
discard 5 vols, no slice-timing (no metadata), mc-afni2 motion correction, direct
EPI->MNI (ANTs SyN, MNI152NLin6Asym res-02), 4mm isotropic resample, 6mm FWHM
smoothing, linear detrend, Friston-24+WM+CSF+GS nuisance regression, 0.01-0.10Hz
band-pass.

Per-acquisition processing happens on the WSL-native filesystem (avoids the DrvFs
large-file write truncation found during pilot development) and only the final
required outputs (final BOLD, brain mask, motion file, design matrix, JSON, QC)
are copied to the Windows-visible derivatives/fsfast/<GROUP>/ tree -- NOT the full
set of debug intermediates the pilot kept, per this run's required directory
structure.

Usage: python3 fsfast_production_4mm6mm.py <GROUP>
"""
import os
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["FREESURFER_HOME"] = "/home/harish/freesurfer"
os.environ["FSFAST_HOME"] = "/home/harish/freesurfer/fsfast"
os.environ["SUBJECTS_DIR"] = "/home/harish/freesurfer/subjects"
os.environ["PATH"] = "/home/harish/freesurfer/bin:/home/harish/freesurfer/fsfast/bin:" + os.environ["PATH"]

import sys
import gc
import csv
import json
import time
import shutil
import hashlib
import platform
import subprocess
import datetime
import numpy as np
import nibabel as nib
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage, signal

BIDS_ROOT = "/mnt/c/Users/krish/FYP/BIDS"
MANIFEST_DIR = "/mnt/c/Users/krish/FYP/audit"
DERIV_ROOT = "/mnt/c/Users/krish/FYP/derivatives/fsfast"
SCRATCH_ROOT = "/home/harish/fyp_work/fsfast_4mm6mm"
ATLAS_PATH = "/mnt/c/Users/krish/FYP/atlases/Brainnetome246/BN_Atlas_246_2mm.nii.gz"
ATLAS_LUT = "/mnt/c/Users/krish/FYP/atlases/Brainnetome246/BN_Atlas_246_LUT.txt"

GROUP_MANIFEST = {
    "AD": ("ad_rerun/acquisition_manifest_final.csv", 25),
    "CN_Final": ("cn_final_rerun/acquisition_manifest_final.csv", 29),
    "EMCI": ("emci_rerun/acquisition_manifest_final.csv", 30),
    "LMCI": ("lmci_rerun/acquisition_manifest_final.csv", 22),
    "MCI": ("mci_rerun/acquisition_manifest_final.csv", 35),
    "SMC_Final": ("smc_final_rerun/acquisition_manifest_final.csv", 32),
}

TPL_NAME = "MNI152NLin6Asym"
N_DISCARD = 5
FWHM_MM = 6.0
LOW_HZ, HIGH_HZ = 0.01, 0.10
# Template-level result (subject-independent -- verified once during pilot development:
# correlating a candidate-transformed WM prior against template T1w intensity gave
# no_transform r=+0.784 vs xfm_forward r=+0.691; xfm_inverse fails to invert). This is a
# fixed relationship between two published MNI templates, not a per-subject computation,
# so it is reused for every acquisition rather than re-tested 173 times.
WM_CSF_MAPPING = "no_transform"


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def otsu(v3d):
    v = v3d[v3d > 0]
    hist, edges = np.histogram(v, bins=256)
    mids = (edges[:-1] + edges[1:]) / 2
    w1, w2 = np.cumsum(hist), np.cumsum(hist[::-1])[::-1]
    m1 = np.cumsum(hist * mids) / np.maximum(w1, 1)
    m2 = np.cumsum((hist * mids)[::-1])[::-1] / np.maximum(w2, 1)
    var = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
    return mids[int(np.argmax(var))]


def process_one_acquisition(group, sub, ses, run, bold_path, work_dir, out_dir,
                            tpl_brain, tpl_mask_path, wm_src, csf_src, log):
    """Returns dict summary_row. Writes only the required final outputs to out_dir."""
    import ants
    t_start = time.time()
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    qc_dir = os.path.join(out_dir, "qc")
    os.makedirs(qc_dir, exist_ok=True)

    row = {"group": group, "sub": sub, "ses": ses, "run": run, "status": "FAIL", "reason": ""}

    json_path = bold_path.replace(".nii.gz", ".json").replace(".nii", ".json")
    checksum_before = md5sum(bold_path)

    try:
        in_img = nib.load(bold_path)
        in_data = in_img.get_fdata(dtype=np.float32)
    except Exception as e:
        row["reason"] = f"cannot load input: {e}"
        log(f"FAIL {sub}/{ses}/{run}: {row['reason']}")
        return row

    errors = []
    if in_data.ndim != 4:
        errors.append("not 4D")
    if np.isnan(in_data).any():
        errors.append("NaN in input")
    if np.isinf(in_data).any():
        errors.append("Inf in input")
    if not np.isfinite(in_img.affine).all() or abs(np.linalg.det(in_img.affine)) < 1e-9:
        errors.append("invalid affine")
    TR = float(in_img.header.get_zooms()[3])
    if not (TR and np.isfinite(TR) and TR > 0):
        errors.append("invalid TR")
    if not np.any(in_data != 0):
        errors.append("empty image")
    if errors:
        row["reason"] = "; ".join(errors)
        log(f"FAIL {sub}/{ses}/{run}: INPUT_VALIDATION: {row['reason']}")
        return row

    N_ORIG = in_data.shape[3]
    if N_ORIG <= N_DISCARD:
        row["reason"] = f"only {N_ORIG} volumes, cannot discard {N_DISCARD}"
        log(f"FAIL {sub}/{ses}/{run}: {row['reason']}")
        return row

    d5 = in_data[..., N_DISCARD:]
    N_KEPT = d5.shape[3]
    del in_data
    p_d5 = os.path.join(work_dir, "d5.nii.gz")
    nib.save(nib.Nifti1Image(d5, in_img.affine, in_img.header), p_d5)

    with open(json_path) as f:
        meta = json.load(f)
    has_st = "SliceTiming" in meta and bool(meta.get("SliceTiming"))
    stc_status = "PERFORMED" if has_st else "NOT_PERFORMED"

    # motion correction
    p_mc = os.path.join(work_dir, "mc.nii.gz")
    p_mcdat = os.path.join(work_dir, "motion.mcdat")
    cmd = ["mc-afni2", "--i", p_d5, "--o", p_mc, "--mcdat", p_mcdat]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0 or not os.path.isfile(p_mcdat):
        row["reason"] = f"mc-afni2 failed: {r.stderr[-500:]}"
        log(f"FAIL {sub}/{ses}/{run}: {row['reason']}")
        return row

    mcdat = np.loadtxt(p_mcdat)
    if mcdat.ndim == 1:
        mcdat = mcdat.reshape(1, -1)
    if mcdat.shape[0] != N_KEPT:
        row["reason"] = f"motion rows ({mcdat.shape[0]}) != retained volumes ({N_KEPT})"
        log(f"FAIL {sub}/{ses}/{run}: {row['reason']}")
        return row
    rot_deg = mcdat[:, 1:4]
    trans_mm = np.column_stack([mcdat[:, 5], mcdat[:, 6], mcdat[:, 4]])
    rot_rad = np.deg2rad(rot_deg)
    dtr = np.vstack([np.zeros((1, 3)), np.diff(trans_mm, axis=0)])
    dro = np.vstack([np.zeros((1, 3)), np.diff(rot_rad, axis=0)])
    FD = np.abs(dtr).sum(1) + np.abs(dro * 50.0).sum(1)

    # MNI normalization -- direct EPI->MNI, single-shot onto the final 4mm grid target
    mc_img = nib.load(p_mc)
    mc_data = mc_img.get_fdata(dtype=np.float32)
    mean_bold = mc_data.mean(axis=3)
    thr = otsu(mean_bold)
    lab, ncomp = ndimage.label(mean_bold > thr, structure=np.ones((3, 3, 3)))
    if ncomp == 0:
        row["reason"] = "no connected component found for registration brain mask"
        log(f"FAIL {sub}/{ses}/{run}: {row['reason']}")
        return row
    sizes = ndimage.sum(mean_bold > thr, lab, range(1, ncomp + 1))
    native_mask = ndimage.binary_fill_holes(lab == int(np.argmax(sizes) + 1))
    p_ref = os.path.join(work_dir, "ref_brain.nii.gz")
    nib.save(nib.Nifti1Image((mean_bold * native_mask).astype(np.float32), mc_img.affine, mc_img.header), p_ref)
    del mean_bold

    fixed = ants.image_read(tpl_brain)
    moving = ants.image_read(p_ref)
    t_reg0 = time.time()
    try:
        reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="SyN", verbose=False)
    except Exception as e:
        row["reason"] = f"ANTs registration failed: {e}"
        log(f"FAIL {sub}/{ses}/{run}: {row['reason']}")
        return row
    reg_time = time.time() - t_reg0

    bold4d = ants.image_read(p_mc)
    del mc_data
    t_w0 = time.time()
    warped = ants.apply_transforms(fixed=fixed, moving=bold4d, transformlist=reg["fwdtransforms"],
                                   interpolator="linear", imagetype=3)
    warp_time = time.time() - t_w0
    p_mni2mm = os.path.join(work_dir, "mni2mm.nii.gz")
    ants.image_write(warped, p_mni2mm)
    del bold4d, warped
    gc.collect()

    mni_img = nib.load(p_mni2mm)
    if mni_img.shape[3] != N_KEPT:
        row["reason"] = "volume count changed during normalization"
        log(f"FAIL {sub}/{ses}/{run}: {row['reason']}")
        return row
    tpl_img = nib.load(tpl_brain)
    in_mni = bool(mni_img.shape[:3] == tpl_img.shape[:3] and np.allclose(mni_img.affine, tpl_img.affine, atol=1e-3))

    # 4mm resample (paper-validated method: nilearn continuous resample onto template-derived 4mm grid)
    from nilearn.image import resample_img, smooth_img
    tgt_aff = tpl_img.affine.copy()
    scale = np.array([4.0, 4.0, 4.0]) / np.array([abs(tpl_img.affine[i, i]) for i in range(3)])
    tgt_aff[:3, :3] = tpl_img.affine[:3, :3] @ np.diag(scale)
    tgt_shape = tuple(int(np.ceil(tpl_img.shape[i] / scale[i])) for i in range(3))
    r4 = resample_img(mni_img, target_affine=tgt_aff, target_shape=tgt_shape,
                      interpolation="continuous", force_resample=True, copy_header=True)
    z4 = tuple(round(float(z), 4) for z in r4.header.get_zooms()[:3])
    if not all(abs(z - 4.0) < 0.01 for z in z4):
        row["reason"] = f"resample voxel size not 4mm: {z4}"
        log(f"FAIL {sub}/{ses}/{run}: {row['reason']}")
        return row
    del mni_img
    tpl_mask_img = nib.load(tpl_mask_path)
    tpl_mask_4 = resample_img(tpl_mask_img, target_affine=tgt_aff, target_shape=tgt_shape,
                              interpolation="nearest", force_resample=True, copy_header=True)
    data4 = r4.get_fdata(dtype=np.float32)
    cover = data4.std(axis=3) > 0
    BRAIN = (tpl_mask_4.get_fdata() > 0) & cover
    n_mask = int(BRAIN.sum())
    if n_mask == 0:
        row["reason"] = "empty brain mask after 4mm resample"
        log(f"FAIL {sub}/{ses}/{run}: {row['reason']}")
        return row

    # 6mm smoothing
    sm = smooth_img(r4, fwhm=FWHM_MM)
    sm_data = sm.get_fdata(dtype=np.float32)
    nt = sm_data.shape[3]
    del data4

    # linear detrend
    flat = sm_data.reshape(-1, nt).T
    tvec = np.arange(nt, dtype=np.float64)
    X_lin = np.column_stack([np.ones(nt), (tvec - tvec.mean()) / tvec.std()])
    beta = np.linalg.lstsq(X_lin, flat, rcond=None)[0]
    vox_mean = flat.mean(axis=0)
    det = flat - X_lin @ beta + vox_mean
    det4 = det.T.reshape(sm_data.shape).astype(np.float32)
    del flat, det, beta, vox_mean, X_lin

    # nuisance regression: Friston-24 + WM + CSF (template priors, precomputed mapping) + GS
    R = np.column_stack([trans_mm, rot_deg])
    Rd = np.vstack([np.zeros((1, 6)), np.diff(R, axis=0)])
    F24 = np.column_stack([R, Rd, R ** 2, Rd ** 2])

    import ants as ants2
    tpl_ants = ants2.image_read(tpl_brain)
    wm_ants = ants2.image_read(wm_src)
    csf_ants = ants2.image_read(csf_src)
    wm_tpl_space = ants2.resample_image_to_target(wm_ants, tpl_ants, interp_type="linear").numpy()
    csf_tpl_space = ants2.resample_image_to_target(csf_ants, tpl_ants, interp_type="linear").numpy()

    def prior_to_mask(prior_tpl_space, thresh, erode_iter):
        m = prior_tpl_space >= thresh
        if erode_iter > 0:
            m = ndimage.binary_erosion(m, structure=np.ones((3, 3, 3)), iterations=erode_iter)
        img_t = nib.Nifti1Image(m.astype(np.float32), tpl_img.affine)
        rr = resample_img(img_t, target_affine=tgt_aff, target_shape=tgt_shape,
                          interpolation="continuous", force_resample=True, copy_header=True)
        m4 = (rr.get_fdata() >= 0.5) & BRAIN
        return m4

    WM_M = prior_to_mask(wm_tpl_space, 0.95, 1)
    CSF_M = prior_to_mask(csf_tpl_space, 0.90, 0)
    del wm_tpl_space, csf_tpl_space

    det_flat = det4.reshape(-1, nt).T
    GS = det_flat[:, BRAIN.ravel()].mean(axis=1)
    MIN_VOX = 20
    wm_ok, csf_ok = int(WM_M.sum()) >= MIN_VOX, int(CSF_M.sum()) >= MIN_VOX
    WM_SIG = det_flat[:, WM_M.ravel()].mean(axis=1) if wm_ok else None
    CSF_SIG = det_flat[:, CSF_M.ravel()].mean(axis=1) if csf_ok else None
    r_wm_gs = float(np.corrcoef(WM_SIG, GS)[0, 1]) if wm_ok else float("nan")
    r_csf_gs = float(np.corrcoef(CSF_SIG, GS)[0, 1]) if csf_ok else float("nan")
    DUP = 0.98
    wm_use = wm_ok and abs(r_wm_gs) < DUP
    csf_use = csf_ok and abs(r_csf_gs) < DUP

    regressors, names = [np.ones(nt)], ["intercept"]
    labels6 = ["trans_x", "trans_y", "trans_z", "roll", "pitch", "yaw"]
    for i, n in enumerate(labels6):
        regressors.append(F24[:, i]); names.append(f"friston24_{n}")
    for i, n in enumerate(labels6):
        regressors.append(F24[:, 6 + i]); names.append(f"friston24_{n}_deriv")
    for i, n in enumerate(labels6):
        regressors.append(F24[:, 12 + i]); names.append(f"friston24_{n}_sq")
    for i, n in enumerate(labels6):
        regressors.append(F24[:, 18 + i]); names.append(f"friston24_{n}_deriv_sq")
    WM_STATUS, CSF_STATUS = "NOT PERFORMED", "NOT PERFORMED"
    if wm_use:
        regressors.append(WM_SIG); names.append("white_matter"); WM_STATUS = "PERFORMED (template prior)"
    if csf_use:
        regressors.append(CSF_SIG); names.append("csf"); CSF_STATUS = "PERFORMED (template prior)"
    regressors.append(GS); names.append("global_signal")
    Xn = np.column_stack(regressors)

    bn = np.linalg.lstsq(Xn, det_flat, rcond=None)[0]
    res = det_flat - Xn @ bn + det_flat.mean(axis=0)
    nz4 = res.T.reshape(det4.shape).astype(np.float32)
    nz4[~BRAIN] = 0
    del det4, det_flat, bn, res

    # band-pass
    fs = 1.0 / TR
    nyq = fs / 2.0
    if not (0 < LOW_HZ < HIGH_HZ < nyq):
        row["reason"] = f"invalid filter band for TR={TR} (Nyquist={nyq})"
        log(f"FAIL {sub}/{ses}/{run}: {row['reason']}")
        return row
    b_, a_ = signal.butter(2, [LOW_HZ / nyq, HIGH_HZ / nyq], btype="bandpass")
    res_flat = nz4.reshape(-1, nt).T
    mean_keep = res_flat.mean(axis=0)
    bp = signal.filtfilt(b_, a_, res_flat, axis=0) + mean_keep
    bp4 = bp.T.reshape(nz4.shape).astype(np.float32)
    bp4[~BRAIN] = 0
    del nz4, res_flat, mean_keep, bp

    # ---- write required outputs ----
    final_fname = f"{sub}_{ses}_task-rest_{run}_desc-preproc_bold.nii.gz"
    p_final = os.path.join(out_dir, final_fname)
    nib.save(nib.Nifti1Image(bp4, r4.affine, r4.header), p_final)

    p_mask = os.path.join(out_dir, "brain_mask.nii.gz")
    nib.save(nib.Nifti1Image(BRAIN.astype(np.uint8), r4.affine, r4.header), p_mask)

    p_mot = os.path.join(out_dir, f"{sub}_{ses}_task-rest_{run}_motion_parameters.mcdat")
    shutil.copy2(p_mcdat, p_mot)

    p_design = os.path.join(out_dir, "design_matrix.txt")
    np.savetxt(p_design, Xn)

    checksum_after = md5sum(bold_path)
    checksum_match = checksum_before == checksum_after

    # ---- QC ----
    sm_mean, sm_sd = sm_data.mean(axis=3), sm_data.std(axis=3)
    with np.errstate(divide="ignore", invalid="ignore"):
        tsnr = np.where(sm_sd > 0, sm_mean / sm_sd, 0.0)
    tv = tsnr[BRAIN]; tv = tv[np.isfinite(tv)]
    sflat = sm_data.reshape(-1, nt).T[:, BRAIN.ravel()]
    dv = np.sqrt(np.mean(np.diff(sflat, axis=0) ** 2, axis=1))
    DVARS = np.concatenate([[0.0], dv])

    vals = sm_mean[BRAIN]
    hh, _ = np.histogram(vals, bins=256)
    pr = hh / hh.sum(); pr = pr[pr > 0]
    spatial_entropy = float(-(pr * np.log2(pr)).sum())

    tvals = sflat.T
    z = (tvals - tvals.mean(axis=1, keepdims=True)) / (tvals.std(axis=1, keepdims=True) + 1e-9)
    ent = []
    for i in range(z.shape[0]):
        hz, _ = np.histogram(z[i, :], bins=16)
        q = hz / hz.sum(); q = q[q > 0]
        ent.append(-(q * np.log2(q)).sum())
    temporal_entropy = float(np.mean(ent))
    del z, tvals, sflat

    raw_mean_native = nib.load(bold_path).get_fdata(dtype=np.float32).mean(axis=3)
    thr2 = otsu(raw_mean_native)
    lab2, nc2 = ndimage.label(raw_mean_native > thr2, structure=np.ones((3, 3, 3)))
    sizes2 = ndimage.sum(raw_mean_native > thr2, lab2, range(1, nc2 + 1)) if nc2 else []
    native_brain2 = ndimage.binary_fill_holes(lab2 == int(np.argmax(sizes2) + 1)) if nc2 else np.zeros_like(raw_mean_native, dtype=bool)
    bg2 = (~native_brain2) & (raw_mean_native > 0)
    SNR = float(raw_mean_native[native_brain2].mean() / raw_mean_native[bg2].std()) if bg2.any() and native_brain2.any() else float("nan")
    del raw_mean_native

    mean_fd, median_fd, max_fd = float(FD.mean()), float(np.median(FD)), float(FD.max())
    n_fd_02, n_fd_05, n_fd_10 = int((FD > 0.2).sum()), int((FD > 0.5).sum()), int((FD > 1.0).sum())

    warnings = []
    if n_fd_05 > 0:
        warnings.append(f"mean/frame FD: {n_fd_05}/{nt} volumes exceed 0.5mm ({100*n_fd_05/nt:.1f}%)")
    qc_status = "WARN" if warnings else "PASS"

    qc_metrics = {
        "subject": sub, "session": ses, "run": run,
        "mean_FD_mm": mean_fd, "median_FD_mm": median_fd, "max_FD_mm": max_fd,
        "FD_gt_0.2mm": n_fd_02, "FD_gt_0.5mm": n_fd_05, "FD_gt_1.0mm": n_fd_10,
        "mean_tSNR": float(tv.mean()), "median_tSNR": float(np.median(tv)),
        "SNR": SNR, "CNR": "NOT_RELIABLY_COMPUTABLE",
        "mean_DVARS_raw": float(DVARS[1:].mean()), "median_DVARS_raw": float(np.median(DVARS[1:])),
        "max_DVARS_raw": float(DVARS[1:].max()), "dvars_type": "RAW_INTENSITY",
        "spatial_entropy_bits": spatial_entropy, "temporal_entropy_bits": temporal_entropy,
        "mask_voxel_count": n_mask, "WM_regressor": WM_STATUS, "CSF_regressor": CSF_STATUS,
        "slice_timing": stc_status, "registration_in_MNI": in_mni,
        "checksum_before": checksum_before, "checksum_after": checksum_after, "checksum_match": checksum_match,
        "n_volumes_orig": N_ORIG, "n_volumes_retained": N_KEPT,
        "final_NaN": int(np.isnan(bp4).sum()), "final_Inf": int(np.isinf(bp4).sum()),
        "qc_status": qc_status, "qc_warnings": "; ".join(warnings) if warnings else "",
    }

    with open(os.path.join(qc_dir, "qc_metrics.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["Metric", "Value"])
        for k, v in qc_metrics.items():
            w.writerow([k, v])
    with open(os.path.join(qc_dir, "qc_metrics.txt"), "w") as f:
        for k, v in qc_metrics.items():
            f.write(f"{k}: {v}\n")
    with open(os.path.join(qc_dir, "qc_report.md"), "w") as f:
        f.write(f"# QC Report: {sub} {ses} {run}\n\n| Metric | Value | Status |\n|---|---|---|\n")
        f.write(f"| tSNR | {qc_metrics['mean_tSNR']:.2f} | - |\n")
        f.write(f"| SNR | {qc_metrics['SNR']:.2f} | - |\n")
        f.write(f"| CNR | NOT_RELIABLY_COMPUTABLE | - |\n")
        f.write(f"| FD (mean) | {mean_fd:.4f}mm | {'WARN' if n_fd_05>0 else 'PASS'} |\n")
        f.write(f"| DVARS (mean, raw) | {qc_metrics['mean_DVARS_raw']:.1f} | - |\n")
        f.write(f"| Spatial entropy | {spatial_entropy:.4f} bits | - |\n")
        f.write(f"| Temporal entropy | {temporal_entropy:.4f} bits | - |\n")
        f.write(f"\nOverall: **{qc_status}**\n")
        if warnings:
            f.write("\nWarnings:\n")
            for w_ in warnings:
                f.write(f"- {w_}\n")

    pd.DataFrame({"volume": np.arange(nt), "FD_mm": FD}).to_csv(os.path.join(qc_dir, "fd_values.csv"), index=False)
    pd.DataFrame({"volume": np.arange(nt), "DVARS": DVARS}).to_csv(os.path.join(qc_dir, "dvars_values.csv"), index=False)

    plt.figure(figsize=(9, 3))
    plt.plot(FD, lw=0.8); plt.axhline(0.5, color="r", ls="--", lw=0.7)
    plt.xlabel("volume"); plt.ylabel("FD (mm)"); plt.title(f"{sub} {ses} FD")
    plt.tight_layout(); plt.savefig(os.path.join(qc_dir, "fd_plot.png"), dpi=100); plt.close()

    plt.figure(figsize=(9, 3))
    plt.plot(DVARS, lw=0.8)
    plt.xlabel("volume"); plt.ylabel("DVARS"); plt.title(f"{sub} {ses} DVARS")
    plt.tight_layout(); plt.savefig(os.path.join(qc_dir, "dvars_plot.png"), dpi=100); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    mid = [s // 2 for s in tsnr.shape]
    for i, axis in enumerate([0, 1, 2]):
        im = ax[i].imshow(np.rot90(np.take(tsnr, mid[axis], axis=axis)), cmap="hot",
                          vmin=0, vmax=float(np.percentile(tv, 99)) if len(tv) else 1)
        ax[i].axis("off")
    plt.suptitle(f"{sub} {ses} tSNR"); plt.tight_layout()
    plt.savefig(os.path.join(qc_dir, "tsnr_map.png"), dpi=100); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    for i, axis in enumerate([0, 1, 2]):
        ax[i].imshow(np.rot90(np.take(sm_mean, mid[axis], axis=axis)), cmap="gray"); ax[i].axis("off")
    plt.suptitle(f"{sub} {ses} mean BOLD"); plt.tight_layout()
    plt.savefig(os.path.join(qc_dir, "mean_bold.png"), dpi=100); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    tpl_d = tpl_img.get_fdata()
    tpl4_res = resample_img(tpl_img, target_affine=tgt_aff, target_shape=tgt_shape,
                            interpolation="continuous", force_resample=True, copy_header=True)
    tpl4_d = tpl4_res.get_fdata()
    for i, axis in enumerate([0, 1, 2]):
        t_sl = np.rot90(np.take(tpl4_d, tpl4_d.shape[axis] // 2, axis=axis))
        b_sl = np.rot90(np.take(sm_mean, sm_mean.shape[axis] // 2, axis=axis))
        ax[i].imshow(b_sl, cmap="gray")
        if (t_sl > 0).any():
            ax[i].contour(t_sl > np.percentile(t_sl[t_sl > 0], 40), levels=[0.5], colors="r", linewidths=0.6)
        ax[i].axis("off")
    plt.suptitle(f"{sub} {ses} MNI registration"); plt.tight_layout()
    plt.savefig(os.path.join(qc_dir, "registration_qc.png"), dpi=100); plt.close()

    # provenance JSON
    provenance = {
        "subject": sub, "session": ses, "run": run,
        "source_bold": bold_path, "original_dimensions": [int(s) for s in in_img.shape],
        "final_dimensions": [int(s) for s in bp4.shape],
        "original_voxel_size_mm": [round(float(z), 4) for z in in_img.header.get_zooms()[:3]],
        "final_voxel_size_mm": list(z4), "TR_seconds": TR,
        "discarded_volumes": N_DISCARD, "retained_volumes": N_KEPT,
        "slice_timing_status": stc_status,
        "motion_correction": "mc-afni2", "mni_template": TPL_NAME,
        "registration_method": "ANTsPy ants.registration(type_of_transform='SyN'); direct EPI->MNI (no T1w)",
        "resampling": "4mm isotropic (nilearn continuous resample onto template-derived grid)",
        "smoothing": f"{FWHM_MM}mm FWHM (nilearn smooth_img)",
        "detrending": "linear (OLS intercept+linear removal, temporal mean restored)",
        "nuisance_regression": {
            "n_regressors": int(Xn.shape[1]), "regressor_names": names,
            "friston24_included": True, "WM_status": WM_STATUS, "CSF_status": CSF_STATUS,
            "WM_CSF_source": "template priors (MNI152NLin2009cAsym label-WM/CSF probseg), NOT subject T1w segmentation",
        },
        "bandpass_hz": [LOW_HZ, HIGH_HZ],
        "software_versions": {"python": platform.python_version(), "numpy": np.__version__,
                              "nibabel": nib.__version__, "antspyx": ants.__version__},
        "processing_date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "processing_status": qc_status,
        "checksum_before": checksum_before, "checksum_after": checksum_after, "checksum_match": checksum_match,
    }
    with open(os.path.join(out_dir, f"{sub}_{ses}_task-rest_{run}_desc-preproc_bold.json"), "w") as f:
        json.dump(provenance, f, indent=2)

    elapsed = time.time() - t_start
    row.update({"status": qc_status, "reason": "; ".join(warnings) if warnings else "OK",
               "reg_time_s": round(reg_time, 1), "warp_time_s": round(warp_time, 1),
               "elapsed_s": round(elapsed, 1)})
    log(f"{qc_status} {sub}/{ses}/{run}  reg={reg_time:.1f}s warp={warp_time:.1f}s total={elapsed:.1f}s "
        f"FD_mean={mean_fd:.3f} tSNR={tv.mean():.1f}")

    del sm_data, sm, r4, bp4, tsnr, DVARS, sm_mean, sm_sd
    gc.collect()
    shutil.rmtree(work_dir, ignore_errors=True)
    return row


CHECKPOINT_PATH = os.path.join(DERIV_ROOT, "PROCESSING_CHECKPOINT.json")


def load_checkpoint():
    if os.path.isfile(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH) as f:
                cp = json.load(f)
            for g in GROUP_MANIFEST:
                cp.setdefault(g, {"completed": [], "failed": [], "remaining": []})
            return cp
        except Exception:
            pass
    return {g: {"completed": [], "failed": [], "remaining": []} for g in GROUP_MANIFEST}


def save_checkpoint(cp):
    tmp = CHECKPOINT_PATH + ".tmp"
    with open(tmp, "w") as f:
        json.dump(cp, f, indent=2)
    os.replace(tmp, CHECKPOINT_PATH)


def validate_acquisition_complete(out_dir, sub, ses, run):
    """An acquisition is COMPLETED only if every required output exists AND passes
    basic validation -- not just 'the JSON file exists'."""
    final = os.path.join(out_dir, f"{sub}_{ses}_task-rest_{run}_desc-preproc_bold.nii.gz")
    mask = os.path.join(out_dir, "brain_mask.nii.gz")
    mot = os.path.join(out_dir, f"{sub}_{ses}_task-rest_{run}_motion_parameters.mcdat")
    design = os.path.join(out_dir, "design_matrix.txt")
    prov = os.path.join(out_dir, f"{sub}_{ses}_task-rest_{run}_desc-preproc_bold.json")
    qc_dir = os.path.join(out_dir, "qc")
    required_qc = ["qc_metrics.csv", "qc_metrics.txt", "qc_report.md", "fd_values.csv", "dvars_values.csv"]

    for p in [final, mask, mot, design, prov]:
        if not os.path.isfile(p):
            return False, f"missing {os.path.basename(p)}"
    for q in required_qc:
        if not os.path.isfile(os.path.join(qc_dir, q)):
            return False, f"missing qc/{q}"
    try:
        img = nib.load(final)
        if len(img.shape) != 4 or img.shape[3] < 2:
            return False, "final output not valid 4D"
    except Exception as e:
        return False, f"final output unreadable: {e}"
    try:
        mimg = nib.load(mask)
        if len(mimg.shape) != 3:
            return False, "mask not valid 3D"
    except Exception as e:
        return False, f"mask unreadable: {e}"
    try:
        with open(prov) as f:
            json.load(f)
    except Exception as e:
        return False, f"provenance JSON unreadable: {e}"
    return True, "OK"


def main(group):
    if group not in GROUP_MANIFEST:
        print(f"FATAL: unknown group {group}"); sys.exit(1)
    manifest_rel, expected_n = GROUP_MANIFEST[group]
    manifest_path = os.path.join(MANIFEST_DIR, manifest_rel)
    rows = list(csv.DictReader(open(manifest_path, newline="", encoding="utf-8")))
    if len(rows) != expected_n:
        print(f"FATAL: expected {expected_n} acquisitions for {group}, manifest has {len(rows)}")
        sys.exit(1)

    checkpoint = load_checkpoint()

    group_deriv = os.path.join(DERIV_ROOT, group)
    logs_dir = os.path.join(group_deriv, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    master_log_path = os.path.join(logs_dir, "preprocessing.log")
    master_log = open(master_log_path, "a")

    def log(msg):
        line = f"{datetime.datetime.now().isoformat()} {msg}"
        print(line, flush=True)
        master_log.write(line + "\n"); master_log.flush()

    log(f"=== {group}: {len(rows)} acquisitions to process (n_procs=1) ===")

    import templateflow.api as tflow
    tpl_brain = str(tflow.get(TPL_NAME, resolution=2, desc="brain", suffix="T1w", extension="nii.gz"))
    tpl_mask_path = str(tflow.get(TPL_NAME, resolution=2, desc="brain", suffix="mask", extension="nii.gz"))
    wm_src = str(tflow.get("MNI152NLin2009cAsym", resolution=2, label="WM", suffix="probseg", extension="nii.gz"))
    csf_src = str(tflow.get("MNI152NLin2009cAsym", resolution=2, label="CSF", suffix="probseg", extension="nii.gz"))
    log(f"template={tpl_brain}")
    log(f"WM/CSF mapping (template-level, precomputed): {WM_CSF_MAPPING}")

    all_keys = [f"{r['subject_id']}/{r['session_id']}/{r['run_id']}" for r in rows]
    grp_cp = checkpoint[group]

    summary_rows = []
    for r in rows:
        sub, ses, run = r["subject_id"], r["session_id"], r["run_id"]
        key = f"{sub}/{ses}/{run}"
        bold_path = r["input_path"]
        out_dir = os.path.join(group_deriv, sub, ses, run)

        ok, val_reason = validate_acquisition_complete(out_dir, sub, ses, run)
        if ok:
            log(f"SKIP {sub}/{ses}/{run} -- COMPLETED, validated on disk (resuming, not reprocessed)")
            with open(os.path.join(out_dir, f"{sub}_{ses}_task-rest_{run}_desc-preproc_bold.json")) as f:
                prov = json.load(f)
            summary_rows.append({"group": group, "sub": sub, "ses": ses, "run": run,
                                 "status": prov.get("processing_status", "PASS"), "reason": "SKIPPED_ALREADY_PROCESSED"})
            if key not in grp_cp["completed"]:
                grp_cp["completed"].append(key)
            if key in grp_cp["failed"]:
                grp_cp["failed"].remove(key)
            grp_cp["remaining"] = [k for k in all_keys if k not in grp_cp["completed"]]
            save_checkpoint(checkpoint)
            continue
        elif os.path.isdir(out_dir):
            log(f"INCOMPLETE {sub}/{ses}/{run} -- {val_reason} -- rebuilding this acquisition")

        work_dir = os.path.join(SCRATCH_ROOT, group, f"{sub}_{ses}_{run}")
        try:
            row = process_one_acquisition(group, sub, ses, run, bold_path, work_dir, out_dir,
                                          tpl_brain, tpl_mask_path, wm_src, csf_src, log)
        except Exception as e:
            import traceback
            log(f"FAIL {sub}/{ses}/{run}: UNHANDLED EXCEPTION: {e}\n{traceback.format_exc()}")
            row = {"group": group, "sub": sub, "ses": ses, "run": run, "status": "FAIL", "reason": str(e)}
            shutil.rmtree(work_dir, ignore_errors=True)
        summary_rows.append(row)

        # checkpoint immediately after every acquisition (success or fail)
        if row["status"] in ("PASS", "WARN"):
            if key not in grp_cp["completed"]:
                grp_cp["completed"].append(key)
            if key in grp_cp["failed"]:
                grp_cp["failed"].remove(key)
        else:
            if key not in grp_cp["failed"]:
                grp_cp["failed"].append(key)
        grp_cp["remaining"] = [k for k in all_keys if k not in grp_cp["completed"] and k not in grp_cp["failed"]]
        save_checkpoint(checkpoint)

        # release memory before next acquisition
        gc.collect()

    # group summary
    n_pass = sum(1 for r in summary_rows if r["status"] == "PASS")
    n_warn = sum(1 for r in summary_rows if r["status"] == "WARN")
    n_fail = sum(1 for r in summary_rows if r["status"] == "FAIL")
    n_skip = sum(1 for r in summary_rows if r["reason"] == "SKIPPED_ALREADY_PROCESSED")

    with open(os.path.join(group_deriv, f"{group}_preprocessing_QC_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["group", "sub", "ses", "run", "status", "reason"])
        w.writeheader()
        for r in summary_rows:
            w.writerow({k: r.get(k, "") for k in ["group", "sub", "ses", "run", "status", "reason"]})

    with open(os.path.join(group_deriv, f"{group}_preprocessing_QC_summary.txt"), "w") as f:
        f.write(f"Total: {len(summary_rows)}  PASS={n_pass}  WARN={n_warn}  FAIL={n_fail}  SKIPPED={n_skip}\n\n")
        for r in summary_rows:
            f.write(f"{r['sub']} {r['ses']} {r['run']}: {r['status']}  {r['reason']}\n")

    with open(os.path.join(group_deriv, f"{group}_preprocessing_QC_report.md"), "w") as f:
        f.write(f"# {group} Preprocessing QC Report\n\n")
        f.write(f"Total: {len(summary_rows)}  PASS={n_pass}  WARN={n_warn}  FAIL={n_fail}  SKIPPED={n_skip}\n\n")
        f.write("| Subject | Session | Run | Status | Reason |\n|---|---|---|---|---|\n")
        for r in summary_rows:
            f.write(f"| {r['sub']} | {r['ses']} | {r['run']} | {r['status']} | {r['reason']} |\n")

    log(f"=== {group} COMPLETE: {n_pass} PASS, {n_warn} WARN, {n_fail} FAIL, {n_skip} SKIPPED ===")
    log(f"GROUP_{group}_DONE")
    master_log.close()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python3 fsfast_production_4mm6mm.py <GROUP>"); sys.exit(1)
    main(sys.argv[1])
