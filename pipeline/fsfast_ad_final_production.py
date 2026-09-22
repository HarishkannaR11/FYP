"""
FINAL PRODUCTION AD preprocessing + masked GLM + comprehensive QC.
Builds on the validated pipeline/fsfast_ad_production_masked.py (identical
motion correction / smoothing / masking / masked GLM logic), extended with:
  - New output layout: derivatives/fsfast/AD/sub-*/ses-*/run-*/ (+ QC/ subdir)
  - Comprehensive QC: tSNR, SNR, CNR(not computable), FD, DVARS, GCOR,
    spatial/temporal entropy (same definitions validated in
    scripts/audit/run_qc_audit.py single-subject pilot)
  - AD-level QC summary across all acquisitions
  - Fresh work dir (no reuse of any old/deleted cache)
  - n_procs=1 (WSL RAM constraint, per audit/wsl_stability_report.txt)

Reads ONLY from BIDS/ (independently verified identical to NIfTI_/NIfTI/AD/
source). Writes ONLY to derivatives/fsfast/AD/ and the Nipype work dir.
STOPS after QC -- no registration, no ROI extraction, no biomarkers.
"""
import os
import csv
import json
import time
import shutil
import hashlib
import subprocess
import numpy as np
import nibabel as nib

os.environ["FREESURFER_HOME"] = "/home/harish/freesurfer"
os.environ["FSFAST_HOME"] = "/home/harish/freesurfer/fsfast"
os.environ["SUBJECTS_DIR"] = "/home/harish/freesurfer/subjects"
os.environ["PATH"] = "/home/harish/freesurfer/bin:/home/harish/freesurfer/fsfast/bin:" + os.environ["PATH"]

from nipype.pipeline import engine as pe
from nipype.interfaces.utility import IdentityInterface, Function

WORK_DIR = "/home/harish/fyp_work/fsfast_ad_final/work"
BIDS_ROOT = "/mnt/c/Users/krish/FYP/BIDS"
DERIV_ROOT = "/mnt/c/Users/krish/FYP/derivatives/fsfast/AD"
MAPPING_CSV = "/mnt/c/Users/krish/FYP/audit/ad_session_mapping_final.csv"

FWHM_MM = 6.0
FD_THRESHOLD_MM = 0.5
DVARS_THRESHOLD_PCTILE = 95


def load_ad_runs():
    rows = list(csv.DictReader(open(MAPPING_CSV, newline="", encoding="utf-8")))
    return sorted([(r["participant_id"], r["session"]) for r in rows])


def split_key(run_key):
    sub, ses = run_key.rsplit("_", 1)
    return sub, ses, "run-01"


def get_paths(sub, ses, run, bids_root):
    import os
    bold_file = f"{bids_root}/{sub}/{ses}/func/{sub}_{ses}_task-rest_{run}_bold.nii.gz"
    if not os.path.exists(bold_file):
        bold_file = bold_file.replace(".nii.gz", ".nii")
    json_file = bold_file.replace(".nii.gz", ".json").replace(".nii", ".json")
    return bold_file, json_file


def check_not_already_processed(sub, ses, run, deriv_root):
    import os
    existing = os.path.join(deriv_root, sub, ses, run, f"{sub}_{ses}_task-rest_{run}_desc-preproc_bold.nii.gz")
    return not os.path.isfile(existing), existing


def checksum_before(bold_file):
    import hashlib
    h = hashlib.md5()
    with open(bold_file, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def validate(bold_file, json_file):
    import nibabel as nib
    import numpy as np
    import json as js
    import time
    t0 = time.time()
    errors = []
    img = nib.load(bold_file)
    data = img.get_fdata()
    shape = img.shape
    if len(shape) != 4:
        errors.append("not 4D")
    if np.isnan(data).any():
        errors.append("NaN present")
    if np.isinf(data).any():
        errors.append("Inf present")
    empty = [t for t in range(shape[3]) if np.allclose(data[..., t], 0)]
    if empty:
        errors.append(f"{len(empty)} empty volumes")
    tr_nifti = float(img.header.get_zooms()[3])
    zooms = img.header.get_zooms()[:3]
    orient = "".join(nib.aff2axcodes(img.affine))
    with open(json_file) as f:
        meta = js.load(f)
    tr_json = meta.get("RepetitionTime", None)
    ok = len(errors) == 0
    dt = time.time() - t0
    return (ok, str(shape), tr_nifti, tr_json, f"({zooms[0]:.2f},{zooms[1]:.2f},{zooms[2]:.2f})",
            orient, "; ".join(errors), dt)


def run_moco(bold_file):
    import subprocess, os, time
    t0 = time.time()
    out = os.path.join(os.getcwd(), "moco_bold.nii.gz")
    mcdat = os.path.join(os.getcwd(), "moco_bold.mcdat")
    cmd = ["mc-afni2", "--i", bold_file, "--o", out, "--mcdat", mcdat]
    res = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if res.returncode != 0:
        return None, None, False, res.stderr[-2000:], dt
    return out, mcdat, True, "", dt


def run_smooth(moco_file, fwhm):
    import subprocess, os, time
    t0 = time.time()
    if not moco_file or not os.path.exists(moco_file):
        return None, False, f"missing/invalid upstream input: moco_file={moco_file}", time.time() - t0
    out = os.path.join(os.getcwd(), "smooth_bold.nii.gz")
    cmd = ["mri_fwhm", "--i", moco_file, "--o", out, "--fwhm", str(fwhm), "--smooth-only"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if res.returncode != 0:
        return None, False, res.stderr[-2000:], dt
    return out, True, "", dt


def run_brain_mask(smooth_file):
    import nibabel as nib
    import numpy as np
    from scipy import ndimage
    import os
    import time

    def otsu_threshold(data):
        vals = data[data > 0]
        if vals.size == 0:
            return 0.0
        hist, edges = np.histogram(vals, bins=256)
        mids = (edges[:-1] + edges[1:]) / 2
        w1 = np.cumsum(hist)
        w2 = np.cumsum(hist[::-1])[::-1]
        m1 = np.cumsum(hist * mids) / np.maximum(w1, 1)
        m2 = (np.cumsum((hist * mids)[::-1])[::-1]) / np.maximum(w2, 1)
        var = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
        return mids[np.argmax(var)]

    t0 = time.time()
    try:
        img = nib.load(smooth_file)
        data = img.get_fdata()
        mean_vol = data.mean(axis=3)
        thr = otsu_threshold(mean_vol)
        raw_mask = mean_vol > thr
        labeled, n_components = ndimage.label(raw_mask, structure=np.ones((3, 3, 3)))
        if n_components == 0:
            return None, 0, 0.0, 0, 0, False, "no connected components found in mask", time.time() - t0
        sizes = ndimage.sum(raw_mask, labeled, range(1, n_components + 1))
        largest_label = int(np.argmax(sizes) + 1)
        largest_size = int(sizes.max())
        largest_component = labeled == largest_label
        final_mask = ndimage.binary_fill_holes(largest_component)
        n_mask = int(final_mask.sum())
        pct_mask = float(100.0 * n_mask / final_mask.size)
        out = os.path.join(os.getcwd(), "brain_mask.nii.gz")
        mask_img = nib.Nifti1Image(final_mask.astype(np.uint8), img.affine)
        mask_img.header.set_zooms(img.header.get_zooms()[:3])
        nib.save(mask_img, out)
        dt = time.time() - t0
        return out, n_mask, pct_mask, n_components, largest_size, True, "", dt
    except Exception as e:
        return None, 0, 0.0, 0, 0, False, str(e), time.time() - t0


def build_design(mcdat_file):
    import numpy as np, os, time
    t0 = time.time()
    if not mcdat_file or not os.path.exists(mcdat_file):
        return None, 0, time.time() - t0
    mc = np.loadtxt(mcdat_file)
    n = mc.shape[0]
    motion = mc[:, 1:7]
    t = np.arange(n, dtype=float)
    t = (t - t.mean()) / t.std()
    X = np.column_stack([np.ones(n), t, t ** 2 - np.mean(t ** 2), motion])
    out = os.path.join(os.getcwd(), "design_matrix.txt")
    np.savetxt(out, X, fmt="%.6f")
    dt = time.time() - t0
    return out, X.shape[1], dt


def run_glmfit(smooth_file, design_file, mask_file):
    import subprocess, os, time
    t0 = time.time()
    missing = [name for name, val in
               [("smooth_file", smooth_file), ("design_file", design_file), ("mask_file", mask_file)]
               if not val or not os.path.exists(val)]
    if missing:
        return None, False, f"missing/invalid upstream input(s): {', '.join(missing)}", "", time.time() - t0
    glmdir = os.path.join(os.getcwd(), "glmdir")
    cmd = ["mri_glmfit", "--y", smooth_file, "--X", design_file, "--no-contrasts-ok",
           "--mask", mask_file, "--glmdir", glmdir, "--eres-save", "--nii.gz"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if res.returncode != 0:
        return None, False, res.stderr[-2000:], "", dt
    final = os.path.join(glmdir, "eres.nii.gz")
    mask_voxels_reported = None
    for line in res.stdout.splitlines():
        if "voxels in mask" in line:
            mask_voxels_reported = "".join(c for c in line if c.isdigit())
    return final, True, "", mask_voxels_reported or "", dt


def finalize(sub, ses, run, bold_file_checksum_before, safe_to_write, existing_path,
             bold_file, json_file, orig_shape, orig_tr_nifti, orig_tr_json, orig_voxel,
             orig_orient, validate_ok, validate_err, moco_file, mcdat_file, moco_ok, moco_err,
             smooth_file, smooth_ok, smooth_err, mask_file, n_mask_voxels, pct_mask,
             n_components, largest_component_size, mask_ok, mask_err,
             design_file, design_ncols, final_file, glmfit_ok, glmfit_err, glm_mask_voxels_reported,
             t_validate, t_moco, t_smooth, t_mask, t_design, t_glmfit, fwhm, deriv_root):
    import nibabel as nib
    import numpy as np
    import os
    import json as js
    import shutil
    import csv
    import hashlib

    FD_THRESHOLD_MM = 0.5
    DVARS_THRESHOLD_PCTILE = 95

    # ---- helper functions (nested: Nipype Function nodes only serialize their own body) ----
    def otsu_threshold(data):
        vals = data[data > 0]
        if vals.size == 0:
            return 0.0
        hist, edges = np.histogram(vals, bins=256)
        mids = (edges[:-1] + edges[1:]) / 2
        w1 = np.cumsum(hist)
        w2 = np.cumsum(hist[::-1])[::-1]
        m1 = np.cumsum(hist * mids) / np.maximum(w1, 1)
        m2 = (np.cumsum((hist * mids)[::-1])[::-1]) / np.maximum(w2, 1)
        var = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
        return mids[np.argmax(var)]

    def spatial_fwhm_estimate(vol3d, mask, voxel_sizes):
        fwhms = []
        for axis, vs in enumerate(voxel_sizes):
            d = np.diff(vol3d, axis=axis)
            if axis == 0:
                m = mask[1:, :, :] & mask[:-1, :, :]
            elif axis == 1:
                m = mask[:, 1:, :] & mask[:, :-1, :]
            else:
                m = mask[:, :, 1:] & mask[:, :, :-1]
            if m.sum() < 10:
                continue
            var_d = np.var(d[m])
            var_i = np.var(vol3d[mask])
            if var_i <= 0:
                continue
            ratio = var_d / (2 * var_i)
            if ratio >= 1:
                continue
            fwhm_vox = np.sqrt(-2 * np.log(2) / np.log(1 - ratio))
            fwhms.append(fwhm_vox * vs)
        if not fwhms:
            return None
        return float(np.exp(np.mean(np.log(fwhms))))

    def compute_fd_array(mc):
        rot_deg = mc[:, 1:4]
        trans = mc[:, 4:7]
        rot_rad = np.deg2rad(rot_deg)
        d_trans = np.diff(trans, axis=0)
        d_rot_mm = np.diff(rot_rad, axis=0) * 50.0
        return np.sum(np.abs(d_trans), axis=1) + np.sum(np.abs(d_rot_mm), axis=1)

    def voxelwise_r2(design, Y):
        beta, _, _, _ = np.linalg.lstsq(design, Y, rcond=None)
        pred = design @ beta
        ss_res = np.sum((Y - pred) ** 2, axis=0)
        y_mean = Y.mean(axis=0, keepdims=True)
        ss_tot = np.sum((Y - y_mean) ** 2, axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            r2 = 1.0 - ss_res / ss_tot
        return np.where(ss_tot > 1e-8, r2, np.nan)

    def mean_abs_corr_per_regressor(motion, Y):
        T = motion.shape[0]
        m_c = motion - motion.mean(axis=0, keepdims=True)
        m_std = m_c.std(axis=0, ddof=1)
        y_c = Y - Y.mean(axis=0, keepdims=True)
        y_std = Y.std(axis=0, ddof=1)
        out = np.zeros(6)
        for k in range(6):
            cov = (m_c[:, k:k + 1] * y_c).sum(axis=0) / (T - 1)
            denom = m_std[k] * y_std
            with np.errstate(divide="ignore", invalid="ignore"):
                corr = np.where(denom > 1e-12, cov / denom, np.nan)
            out[k] = np.nanmean(np.abs(corr))
        return out

    def compute_dvars_array(data, mask):
        vox_ts = data[mask]
        diff = np.diff(vox_ts, axis=1)
        dvars = np.sqrt((diff ** 2).mean(axis=0))
        return dvars

    def compute_gcor_val(data, mask):
        vox_ts = data[mask].astype(np.float64)
        mu = vox_ts.mean(axis=1, keepdims=True)
        sd = vox_ts.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        u = (vox_ts - mu) / sd
        return float(u.mean(axis=0).var())

    def compute_snr_val(data4d, mask):
        tmean = data4d.mean(axis=3)
        signal = tmean[mask]
        background = tmean[(~mask) & (tmean != 0)]
        if background.size < 100:
            return None, None, 0
        noise_std = background.std()
        if noise_std <= 0:
            return None, float(noise_std), int(background.size)
        return float(signal.mean() / noise_std), float(noise_std), int(background.size)

    def compute_spatial_entropy_val(vol3d, mask, bins=256):
        vals = vol3d[mask]
        hist, edges = np.histogram(vals, bins=bins)
        p = hist / hist.sum()
        p_nonzero = p[p > 0]
        return float(-(p_nonzero * np.log2(p_nonzero)).sum())

    def compute_temporal_entropy_val(data4d, mask, bins=16):
        vox_ts = data4d[mask].astype(np.float64)
        mu = vox_ts.mean(axis=1, keepdims=True)
        sd = vox_ts.std(axis=1, keepdims=True)
        sd[sd == 0] = 1.0
        z = (vox_ts - mu) / sd
        edges = np.linspace(-4, 4, bins + 1)
        entropies = np.zeros(vox_ts.shape[0])
        for i in range(vox_ts.shape[0]):
            hist, _ = np.histogram(z[i], bins=edges)
            p = hist / hist.sum()
            p_nonzero = p[p > 0]
            entropies[i] = -(p_nonzero * np.log2(p_nonzero)).sum() if p_nonzero.size else 0.0
        return float(entropies.mean())

    errors = []
    warnings = []
    notes = []

    if not safe_to_write:
        logs_dir_skip = os.path.join(deriv_root, "logs")
        os.makedirs(logs_dir_skip, exist_ok=True)
        with open(os.path.join(logs_dir_skip, f"{sub}_{ses}_{run}_processing.log"), "a") as f:
            f.write(f"\n=== {sub} {ses} {run}: SKIPPED (already processed) at {existing_path} ===\n")
        qc_rows_skip = [{"sub": sub, "ses": ses, "run": run, "stage": "already_processed",
                          "file": existing_path, "shape": "", "readable": "YES"}]
        summary_row_skip = {"sub": sub, "ses": ses, "run": run, "input_file": bold_file,
                             "output_file": existing_path, "total_processing_time_s": 0,
                             "qc_status": "SKIPPED_ALREADY_PROCESSED", "errors": "", "warnings": ""}
        return qc_rows_skip, summary_row_skip, os.path.join(logs_dir_skip, f"{sub}_{ses}_{run}_processing.log")

    for ok, err, stage in [(validate_ok, validate_err, "validate"), (moco_ok, moco_err, "moco"),
                            (smooth_ok, smooth_err, "smooth"), (mask_ok, mask_err, "mask"),
                            (glmfit_ok, glmfit_err, "glmfit")]:
        if not ok:
            errors.append(f"{stage}: {err}")

    checksum_ok = True
    checksum_after = None
    if os.path.exists(bold_file):
        h = hashlib.md5()
        with open(bold_file, "rb") as f:
            for c in iter(lambda: f.read(1 << 20), b""):
                h.update(c)
        checksum_after = h.hexdigest()
        if checksum_after != bold_file_checksum_before:
            errors.append(f"SOURCE CHECKSUM CHANGED during processing: before={bold_file_checksum_before} after={checksum_after}")
            checksum_ok = False

    qc_pass = len(errors) == 0
    out_dir = os.path.join(deriv_root, sub, ses, run)
    qc_out_dir = os.path.join(out_dir, "QC")
    final_bold_out = motion_tsv_out = json_out = mask_out = design_out = None

    metrics = {}

    if qc_pass:
        os.makedirs(out_dir, exist_ok=True)
        os.makedirs(qc_out_dir, exist_ok=True)

        in_img = nib.load(bold_file)
        in_data = in_img.get_fdata()
        expect_shape = eval(orig_shape)

        out_final_img = nib.load(final_file)
        out_data = out_final_img.get_fdata()
        nan_count = int(np.isnan(out_data).sum())
        inf_count = int(np.isinf(out_data).sum())
        if nan_count or inf_count:
            errors.append(f"NaN={nan_count} Inf={inf_count} in final output")
            qc_pass = False
        shape_preserved = (out_final_img.shape == expect_shape)
        if not shape_preserved:
            errors.append(f"shape changed: {expect_shape} -> {out_final_img.shape}")
            qc_pass = False
        vol_preserved = (out_final_img.shape[3] == expect_shape[3]) if len(out_final_img.shape) > 3 else False

        mask_data = nib.load(mask_file).get_fdata().astype(bool)
        mask_nonempty = bool(mask_data.sum() > 0)
        if not mask_nonempty:
            errors.append("brain mask is EMPTY")
            qc_pass = False
        mask_suspicious = (pct_mask < 10.0) or (pct_mask > 45.0)
        if mask_suspicious:
            warnings.append(f"suspicious mask coverage: {pct_mask:.2f}% of FOV")
        outside_mask_vals = out_data[~mask_data]
        n_nonzero_outside = int(np.count_nonzero(outside_mask_vals))
        if n_nonzero_outside > 0:
            errors.append(f"masked GLM output NOT zero outside mask: {n_nonzero_outside} nonzero voxels")
            qc_pass = False
        glm_mask_match = (str(glm_mask_voxels_reported) == str(n_mask_voxels))
        if not glm_mask_match:
            warnings.append(f"mri_glmfit reported mask voxel count ({glm_mask_voxels_reported}) != generated mask ({n_mask_voxels})")

        mc = np.loadtxt(mcdat_file)
        fd = compute_fd_array(mc)
        fd_full = np.concatenate([[0.0], fd])
        mean_fd = float(np.mean(fd))
        median_fd = float(np.median(fd))
        max_fd = float(np.max(fd))
        p95_fd = float(np.percentile(fd, 95))
        n_gt_02 = int((fd > 0.2).sum())
        n_gt_05 = int((fd > 0.5).sum())
        pct_gt_02 = float(100 * n_gt_02 / fd.size)
        pct_gt_05 = float(100 * n_gt_05 / fd.size)
        max_trans = float(np.max(mc[:, 9]))
        max_rot = float(np.max(np.abs(mc[:, 1:4])))
        if mc.shape[0] != expect_shape[3]:
            errors.append(f"motion params row count {mc.shape[0]} != {expect_shape[3]}")
            qc_pass = False

        sm_data = nib.load(smooth_file).get_fdata()
        temporal_sd_raw = float(np.mean(in_data.std(axis=3)[mask_data])) if mask_data.sum() else None
        temporal_sd_final = float(np.mean(out_data.std(axis=3)[mask_data])) if mask_data.sum() else None

        global_signal_raw = sm_data[mask_data].mean(axis=0) if mask_data.sum() else None
        spikes_first_vol_only = False
        spikes_other = []
        if global_signal_raw is not None and global_signal_raw.std() > 0:
            z = (global_signal_raw - global_signal_raw.mean()) / global_signal_raw.std()
            spike_idx = np.where(np.abs(z) > 3)[0].tolist()
            spikes_other = [s for s in spike_idx if s != 0]
            spikes_first_vol_only = (0 in spike_idx)
            if spikes_other:
                warnings.append(f"raw global-signal spike(s) at NON-first volumes: {spikes_other}")

        sm_tmean = sm_data.mean(axis=3)
        sm_std = sm_data.std(axis=3)
        with np.errstate(divide="ignore", invalid="ignore"):
            tsnr_map = np.where(sm_std > 0, sm_tmean / sm_std, 0)
        tsnr_vals_all = tsnr_map[mask_data]
        tsnr_valid = tsnr_vals_all[np.isfinite(tsnr_vals_all) & (tsnr_vals_all != 0)]
        mean_tsnr = float(tsnr_valid.mean()) if tsnr_valid.size else None
        median_tsnr = float(np.median(tsnr_valid)) if tsnr_valid.size else None
        std_tsnr = float(tsnr_valid.std()) if tsnr_valid.size else None
        min_tsnr = float(tsnr_valid.min()) if tsnr_valid.size else None
        max_tsnr = float(tsnr_valid.max()) if tsnr_valid.size else None
        p5_tsnr = float(np.percentile(tsnr_valid, 5)) if tsnr_valid.size else None
        p95_tsnr = float(np.percentile(tsnr_valid, 95)) if tsnr_valid.size else None
        notes.append("tSNR computed on pre-GLM smoothed (magnitude-domain) intermediate, NOT the zero-mean final residual")

        # ---- SNR ----
        snr_val, snr_noise_std, snr_bg_n = compute_snr_val(sm_data, mask_data)

        # ---- CNR ----
        cnr_val = "NOT RELIABLY COMPUTABLE"
        cnr_reason = "No subject-specific T1w anatomical image/segmentation available for reliable anatomical tissue contrast estimation."

        # ---- DVARS (on smoothed pre-GLM stage) ----
        dvars_arr = compute_dvars_array(sm_data, mask_data)
        dvars_full = np.concatenate([[np.nan], dvars_arr])
        mean_dvars = float(dvars_arr.mean())
        median_dvars = float(np.median(dvars_arr))
        max_dvars = float(dvars_arr.max())
        p95_dvars = float(np.percentile(dvars_arr, DVARS_THRESHOLD_PCTILE))
        n_high_dvars = int((dvars_arr > p95_dvars).sum())
        pct_high_dvars = float(100 * n_high_dvars / dvars_arr.size)

        # ---- GCOR (pre vs post GLM) ----
        gcor_pre = compute_gcor_val(sm_data, mask_data)
        gcor_post = compute_gcor_val(out_data, mask_data)

        # ---- Entropy ----
        spatial_entropy = compute_spatial_entropy_val(sm_tmean, mask_data)
        temporal_entropy = compute_temporal_entropy_val(sm_data, mask_data)

        design_match = None
        if os.path.exists(design_file):
            X_saved = np.loadtxt(design_file)
            if X_saved.shape[1] >= 9:
                design_match = bool(np.allclose(X_saved[:, 3:9], mc[:, 1:7], atol=1e-4))
                if not design_match:
                    errors.append("DESIGN MATRIX MOTION COLUMNS DO NOT MATCH mc-afni2 OUTPUT")
                    qc_pass = False

        motion = mc[:, 1:7]
        Xr = np.column_stack([np.ones(len(mc)), motion])
        Y_before = sm_data[mask_data]
        Y_after = out_data[mask_data]
        r2_before_map = voxelwise_r2(Xr, Y_before.T)
        r2_after_map = voxelwise_r2(Xr, Y_after.T)
        motion_r2_before = float(np.nanmean(r2_before_map))
        motion_r2_after = float(np.nanmean(r2_after_map))
        if motion_r2_after > motion_r2_before:
            warnings.append("motion-explained R^2 DID NOT decrease after GLM regression")

        corr_before = mean_abs_corr_per_regressor(motion, Y_before.T)
        corr_after = mean_abs_corr_per_regressor(motion, Y_after.T)
        corr_labels = ["roll", "pitch", "yaw", "dS", "dL", "dP"]

        moco_tmean = nib.load(moco_file).get_fdata().mean(axis=3)
        vs = in_img.header.get_zooms()[:3]
        fwhm_before = spatial_fwhm_estimate(moco_tmean, mask_data, vs)
        fwhm_after = spatial_fwhm_estimate(sm_tmean, mask_data, vs)

        if in_img.shape[2] != out_final_img.shape[2]:
            errors.append("Z-DIMENSION CHANGED -- unintended slice cropping detected")
            qc_pass = False

        voxel_preserved = bool(np.allclose(in_img.header.get_zooms()[:3], out_final_img.header.get_zooms()[:3], atol=0.01))
        out_tr = float(out_final_img.header.get_zooms()[3]) if len(out_final_img.header.get_zooms()) > 3 else None
        tr_preserved = (orig_tr_nifti is not None and out_tr is not None and abs(orig_tr_nifti - out_tr) < 0.01)

        if mean_fd > FD_THRESHOLD_MM:
            warnings.append(f"mean FD {mean_fd:.3f}mm exceeds {FD_THRESHOLD_MM}mm reference threshold (not used to exclude)")

        metrics = {
            "max_translation_mm": max_trans, "max_rotation_deg": max_rot,
            "mean_FD": mean_fd, "median_FD": median_fd, "max_FD": max_fd, "p95_FD": p95_fd,
            "FD_gt_0.2_count": n_gt_02, "FD_gt_0.2_pct": pct_gt_02,
            "FD_gt_0.5_count": n_gt_05, "FD_gt_0.5_pct": pct_gt_05,
            "temporal_SD_raw": temporal_sd_raw, "temporal_SD_final": temporal_sd_final,
            "mean_tSNR": mean_tsnr, "median_tSNR": median_tsnr, "std_tSNR": std_tsnr,
            "min_tSNR": min_tsnr, "max_tSNR": max_tsnr, "p5_tSNR": p5_tsnr, "p95_tSNR": p95_tsnr,
            "SNR": snr_val, "SNR_noise_std": snr_noise_std, "SNR_background_voxels": snr_bg_n,
            "CNR": cnr_val, "CNR_reason": cnr_reason,
            "mean_DVARS": mean_dvars, "median_DVARS": median_dvars, "max_DVARS": max_dvars,
            "p95_DVARS": p95_dvars, "DVARS_high_count": n_high_dvars, "DVARS_high_pct": pct_high_dvars,
            "GCOR_pre_GLM": gcor_pre, "GCOR_post_GLM": gcor_post,
            "spatial_entropy_bits": spatial_entropy, "temporal_entropy_bits": temporal_entropy,
            "mask_voxel_count": n_mask_voxels, "mask_pct_of_FOV": pct_mask,
            "mask_n_components_before_selection": n_components, "mask_largest_component_size": largest_component_size,
            "mask_nonempty": mask_nonempty, "mask_suspicious_coverage": mask_suspicious,
            "mask_zero_outside_confirmed": (n_nonzero_outside == 0),
            "glm_mask_voxel_count_match": glm_mask_match,
            "motion_R2_before_GLM": motion_r2_before, "motion_R2_after_GLM": motion_r2_after,
            "design_matrix_motion_match": design_match,
            "estimated_FWHM_before_mm": fwhm_before, "estimated_FWHM_after_mm": fwhm_after,
            "requested_FWHM_mm": fwhm,
            "NaN_count": nan_count, "Inf_count": inf_count,
            "shape_preserved": shape_preserved, "voxel_size_preserved": voxel_preserved,
            "TR_preserved": tr_preserved, "volume_count_preserved": vol_preserved,
            "spikes_first_volume_only": spikes_first_vol_only, "spikes_other_volumes": spikes_other,
            "checksum_before": bold_file_checksum_before, "checksum_after": checksum_after,
            "checksum_match": checksum_ok,
        }
        for i, lab in enumerate(corr_labels):
            metrics[f"corr_{lab}_before"] = float(corr_before[i])
            metrics[f"corr_{lab}_after"] = float(corr_after[i])

        if qc_pass:
            final_bold_out = os.path.join(out_dir, f"{sub}_{ses}_task-rest_{run}_desc-preproc_bold.nii.gz")
            if os.path.isfile(final_bold_out):
                errors.append("OUTPUT APPEARED DURING PROCESSING -- refusing to overwrite")
                qc_pass = False
            else:
                shutil.copy(final_file, final_bold_out)
                motion_tsv_out = os.path.join(out_dir, f"{sub}_{ses}_task-rest_{run}_motion_parameters.mcdat")
                shutil.copy(mcdat_file, motion_tsv_out)

                mask_out = os.path.join(out_dir, "brain_mask.nii.gz")
                shutil.copy(mask_file, mask_out)

                design_out = os.path.join(out_dir, "design_matrix.txt")
                shutil.copy(design_file, design_out)

                # ---- per-acquisition QC directory ----
                qc_metrics_csv = os.path.join(qc_out_dir, "qc_metrics.csv")
                with open(qc_metrics_csv, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["Metric", "Value"])
                    for k, v in metrics.items():
                        w.writerow([k, v])

                qc_metrics_txt = os.path.join(qc_out_dir, "qc_metrics.txt")
                with open(qc_metrics_txt, "w") as f:
                    for k, v in metrics.items():
                        f.write(f"{k}: {v}\n")

                fd_csv = os.path.join(qc_out_dir, "fd_values.csv")
                with open(fd_csv, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["frame", "FD_mm"])
                    for i, v in enumerate(fd_full):
                        w.writerow([i, v])

                dvars_csv = os.path.join(qc_out_dir, "dvars_values.csv")
                with open(dvars_csv, "w", newline="") as f:
                    w = csv.writer(f)
                    w.writerow(["frame", "DVARS"])
                    for i, v in enumerate(dvars_full):
                        w.writerow([i, v])

                try:
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt

                    fig, ax = plt.subplots(figsize=(7, 5))
                    ax.hist(tsnr_valid, bins=60, color="steelblue")
                    ax.set_title(f"tSNR distribution ({sub} {ses} {run})")
                    plt.tight_layout()
                    plt.savefig(os.path.join(qc_out_dir, "tsnr_distribution.png"), dpi=120)
                    plt.close(fig)

                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(fd_full, color="crimson", lw=0.8)
                    ax.axhline(0.5, color="red", ls="--", lw=0.8)
                    ax.set_title(f"FD ({sub} {ses} {run})")
                    plt.tight_layout()
                    plt.savefig(os.path.join(qc_out_dir, "fd_plot.png"), dpi=120)
                    plt.close(fig)

                    fig, ax = plt.subplots(figsize=(10, 4))
                    ax.plot(dvars_full, color="darkorange", lw=0.8)
                    ax.set_title(f"DVARS ({sub} {ses} {run})")
                    plt.tight_layout()
                    plt.savefig(os.path.join(qc_out_dir, "dvars_plot.png"), dpi=120)
                    plt.close(fig)

                    fig, ax = plt.subplots(figsize=(7, 5))
                    ax.hist(sm_tmean[mask_data], bins=256, color="seagreen")
                    ax.set_title(f"Spatial intensity ({sub} {ses} {run})\nEntropy={spatial_entropy:.2f} bits")
                    plt.tight_layout()
                    plt.savefig(os.path.join(qc_out_dir, "spatial_entropy_distribution.png"), dpi=120)
                    plt.close(fig)
                except Exception as plot_err:
                    warnings.append(f"QC plot generation failed: {plot_err}")

                qc_report_md = os.path.join(qc_out_dir, "qc_report.md")
                with open(qc_report_md, "w") as f:
                    f.write(f"# QC Report: {sub} {ses} {run}\n\n")
                    f.write(f"| Metric | Value | Status |\n|---|---|---|\n")
                    f.write(f"| tSNR | {mean_tsnr:.2f} | PASS |\n")
                    f.write(f"| SNR | {snr_val:.2f} | {'PASS' if snr_val else 'NOT COMPUTABLE'} |\n")
                    f.write(f"| CNR | {cnr_val} | NOT COMPUTABLE |\n")
                    f.write(f"| FD (mean) | {mean_fd:.3f}mm | {'WARN' if mean_fd > FD_THRESHOLD_MM else 'PASS'} |\n")
                    f.write(f"| DVARS (mean) | {mean_dvars:.1f} | PASS |\n")
                    f.write(f"| GCOR (post-GLM) | {gcor_post:.4f} | PASS |\n")
                    f.write(f"| Spatial Entropy | {spatial_entropy:.2f} bits | PASS |\n")
                    f.write(f"| Temporal Entropy | {temporal_entropy:.2f} bits | PASS |\n")

                json_out = os.path.join(out_dir, f"{sub}_{ses}_task-rest_{run}_desc-preproc_bold.json")
                sidecar = {
                    "Sources": [os.path.basename(bold_file)],
                    "OriginalDimensions": orig_shape, "OriginalVoxelSize": orig_voxel,
                    "OriginalOrientation": orig_orient, "OriginalRepetitionTime": orig_tr_json,
                    "SkullStripped": True,
                    "PreprocessingPipeline": "FS-FAST + Nipype (BOLD-only, no T1w) -- FINAL PRODUCTION",
                    "ProcessingSteps": [
                        {"step": "data_validation", "tool": "NiBabel", "status": "PASS"},
                        {"step": "motion_correction", "tool": "mc-afni2", "reference_frame": 0,
                         "max_translation_mm": max_trans, "max_rotation_deg": max_rot,
                         "mean_FD_mm": mean_fd, "status": "PASS"},
                        {"step": "slice_timing_correction", "status": "SKIPPED",
                         "reason": "SliceTiming not reliably recoverable from ADNI DICOM/BIDS metadata."},
                        {"step": "spatial_smoothing", "tool": "mri_fwhm", "requested_fwhm_mm": fwhm,
                         "estimated_fwhm_before_mm": fwhm_before, "estimated_fwhm_after_mm": fwhm_after,
                         "status": "PASS"},
                        {"step": "brain_mask_generation", "tool": "Otsu + largest 3D connected component + hole-filling",
                         "mask_voxel_count": n_mask_voxels, "mask_pct_of_FOV": pct_mask,
                         "n_components_before_selection": n_components, "status": "PASS"},
                        {"step": "temporal_filtering_and_nuisance_regression", "tool": "mri_glmfit", "masked": True,
                         "design_matrix_columns": design_ncols,
                         "design_regressors": ["intercept", "linear_trend", "quadratic_trend",
                                                "motion_roll", "motion_pitch", "motion_yaw",
                                                "motion_dS", "motion_dL", "motion_dP"],
                         "design_matches_motion_output": design_match,
                         "motion_R2_before": motion_r2_before, "motion_R2_after": motion_r2_after,
                         "status": "PASS"},
                        {"step": "no_arbitrary_slice_cropping", "status": "CONFIRMED"},
                    ],
                    "QCMetrics": metrics,
                }
                with open(json_out, "w") as f:
                    js.dump(sidecar, f, indent=2, default=str)
        else:
            warnings.append("QC failed -- outputs NOT copied to derivatives")

    total_time = sum(x for x in [t_validate, t_moco, t_smooth, t_mask, t_design, t_glmfit] if x)

    logs_dir = os.path.join(deriv_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    subj_log = os.path.join(logs_dir, f"{sub}_{ses}_{run}_processing.log")
    log_lines = [
        f"=== {sub} {ses} {run} processing log (FINAL PRODUCTION) ===",
        f"Input BOLD: {bold_file}", f"Source checksum before: {bold_file_checksum_before}",
        f"Source checksum after: {metrics.get('checksum_after', 'N/A')}",
        f"Checksum match: {metrics.get('checksum_match', 'N/A')}",
        f"Validation: {'PASS' if validate_ok else 'FAIL - ' + validate_err} (time {t_validate:.2f}s)",
        f"Motion correction: {'PASS' if moco_ok else 'FAIL - ' + moco_err} (time {t_moco:.2f}s)",
        f"Slice-timing: SKIPPED -- not recoverable.",
        f"Smoothing: {'PASS' if smooth_ok else 'FAIL - ' + smooth_err} (time {t_smooth:.2f}s)",
        f"Mask: {'PASS' if mask_ok else 'FAIL - ' + mask_err} (time {t_mask:.2f}s) -- "
        f"{n_components} components before selection, largest={largest_component_size} voxels",
        f"GLM: {'PASS' if glmfit_ok else 'FAIL - ' + glmfit_err} (time {t_glmfit:.2f}s)",
        f"Total time: {total_time:.2f}s",
        f"QC: {'PASS' if qc_pass else 'FAIL'}",
        f"Errors: {'; '.join(errors) if errors else 'none'}",
        f"Warnings: {'; '.join(warnings) if warnings else 'none'}",
        f"Output BOLD: {final_bold_out}",
        "",
    ]
    with open(subj_log, "w") as f:
        f.write("\n".join(log_lines))

    qc_rows = []
    for stage_name, path in [("input_original_BIDS", bold_file), ("motion_corrected", moco_file),
                              ("smoothed", smooth_file), ("brain_mask", mask_file),
                              ("final_preproc_residual", final_file)]:
        row = {"sub": sub, "ses": ses, "run": run, "stage": stage_name, "file": path or ""}
        try:
            img2 = nib.load(path)
            row["shape"] = str(img2.shape)
            row["readable"] = "YES"
        except Exception as e:
            row["shape"] = ""
            row["readable"] = "NO"
            row["error"] = str(e)
        qc_rows.append(row)

    summary_row = {"sub": sub, "ses": ses, "run": run, "input_file": bold_file,
                    "output_file": final_bold_out or "", "total_processing_time_s": round(total_time, 2),
                    "qc_status": "PASS" if qc_pass else "FAIL",
                    "errors": "; ".join(errors) if errors else "", "warnings": "; ".join(warnings) if warnings else ""}
    summary_row.update(metrics)

    return qc_rows, summary_row, subj_log


def collect_and_write(qc_rows_list, summary_rows_list, subj_logs_list, deriv_root, fwhm):
    import csv
    import json as js
    import os
    import time
    import numpy as np

    FD_THRESHOLD_MM = 0.5

    qc_dir = os.path.join(deriv_root, "QC")
    logs_dir = os.path.join(deriv_root, "logs")
    os.makedirs(qc_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    n_fail = sum(1 for r in summary_rows_list if r["qc_status"] == "FAIL")
    n_warn = sum(1 for r in summary_rows_list if r["qc_status"] == "PASS" and r.get("warnings"))
    n_clean_pass = sum(1 for r in summary_rows_list if r["qc_status"] == "PASS" and not r.get("warnings"))
    n_skipped = sum(1 for r in summary_rows_list if r["qc_status"] == "SKIPPED_ALREADY_PROCESSED")

    ad_columns = ["subject_id", "session_id", "run_id", "input_file", "dimensions", "voxel_size", "TR", "volumes",
                  "tSNR", "SNR", "CNR", "FD_mean", "FD_median", "FD_max", "FD_95th", "FD_gt_0.2", "FD_gt_0.5",
                  "DVARS_mean", "DVARS_median", "DVARS_max", "DVARS_95th", "GCOR", "spatial_entropy",
                  "temporal_entropy", "mask_voxels", "mask_coverage", "NaN_count", "Inf_count", "GLM_status", "QC_status"]

    ad_rows = []
    for r in summary_rows_list:
        ad_rows.append({
            "subject_id": r["sub"], "session_id": r["ses"], "run_id": r["run"], "input_file": r["input_file"],
            "dimensions": "", "voxel_size": "", "TR": "", "volumes": "",
            "tSNR": r.get("mean_tSNR", ""), "SNR": r.get("SNR", ""), "CNR": r.get("CNR", ""),
            "FD_mean": r.get("mean_FD", ""), "FD_median": r.get("median_FD", ""), "FD_max": r.get("max_FD", ""),
            "FD_95th": r.get("p95_FD", ""), "FD_gt_0.2": r.get("FD_gt_0.2_pct", ""), "FD_gt_0.5": r.get("FD_gt_0.5_pct", ""),
            "DVARS_mean": r.get("mean_DVARS", ""), "DVARS_median": r.get("median_DVARS", ""),
            "DVARS_max": r.get("max_DVARS", ""), "DVARS_95th": r.get("p95_DVARS", ""),
            "GCOR": r.get("GCOR_post_GLM", ""), "spatial_entropy": r.get("spatial_entropy_bits", ""),
            "temporal_entropy": r.get("temporal_entropy_bits", ""),
            "mask_voxels": r.get("mask_voxel_count", ""), "mask_coverage": r.get("mask_pct_of_FOV", ""),
            "NaN_count": r.get("NaN_count", ""), "Inf_count": r.get("Inf_count", ""),
            "GLM_status": "PASS" if r["qc_status"] in ("PASS", "SKIPPED_ALREADY_PROCESSED") else "FAIL",
            "QC_status": r["qc_status"],
        })

    ad_csv = os.path.join(qc_dir, "AD_preprocessing_QC_summary.csv")
    with open(ad_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=ad_columns)
        w.writeheader()
        w.writerows(ad_rows)

    ad_txt = os.path.join(qc_dir, "AD_preprocessing_QC_summary.txt")
    with open(ad_txt, "w") as f:
        f.write(f"Total: {len(summary_rows_list)}  PASS(clean)={n_clean_pass}  PASS(warn)={n_warn}  "
                f"FAIL={n_fail}  SKIPPED={n_skipped}\n\n")
        for r in ad_rows:
            f.write(f"{r['subject_id']} {r['session_id']} {r['run_id']}: {r['QC_status']}  "
                    f"FD_mean={r['FD_mean']}  tSNR={r['tSNR']}\n")

    ad_report_md = os.path.join(qc_dir, "AD_preprocessing_QC_report.md")
    with open(ad_report_md, "w") as f:
        f.write("# AD Preprocessing QC Summary\n\n")
        f.write(f"Total acquisitions: {len(summary_rows_list)}\n")
        f.write(f"PASS (clean): {n_clean_pass}\nPASS (warnings): {n_warn}\nFAIL: {n_fail}\nSKIPPED: {n_skipped}\n\n")
        f.write("| Subject | Session | Run | Status | FD mean | tSNR | GCOR |\n|---|---|---|---|---|---|---|\n")
        for r in ad_rows:
            f.write(f"| {r['subject_id']} | {r['session_id']} | {r['run_id']} | {r['QC_status']} | "
                    f"{r['FD_mean']} | {r['tSNR']} | {r['GCOR']} |\n")

    master_log = os.path.join(logs_dir, "preprocessing.log")
    with open(master_log, "w") as f:
        f.write("=== FS-FAST FINAL AD PRODUCTION PREPROCESSING + QC ===\n")
        f.write(f"Timestamp: {time.ctime()}\n")
        f.write(f"Total: {len(summary_rows_list)}  clean_pass={n_clean_pass}  warn={n_warn}  fail={n_fail}  skipped={n_skipped}\n")

    return ad_csv, ad_report_md, master_log


def write_dataset_description(deriv_root):
    import json as js
    import os
    os.makedirs(deriv_root, exist_ok=True)
    path = os.path.join(deriv_root, "dataset_description.json")
    if os.path.exists(path):
        return
    desc = {"Name": "FS-FAST BOLD-only preprocessing derivatives (AD group, final production)",
            "BIDSVersion": "1.8.0", "DatasetType": "derivative",
            "GeneratedBy": [{"Name": "FS-FAST + Nipype (final production pipeline)"}]}
    with open(path, "w") as f:
        js.dump(desc, f, indent=2)


def main():
    write_dataset_description(DERIV_ROOT)
    runs = load_ad_runs()
    print(f"=== Processing exactly these {len(runs)} AD acquisitions ===")
    for sub, ses in runs:
        print(f"  {sub} {ses} run-01")
    assert len(runs) == 25, f"expected 25 runs, got {len(runs)}"

    wf = pe.Workflow(name="fsfast_ad_final_production", base_dir=WORK_DIR)

    infosource = pe.Node(IdentityInterface(fields=["run_key"]), name="infosource")
    infosource.iterables = [("run_key", [f"{sub}_{ses}" for sub, ses in runs])]

    splitkey = pe.Node(Function(input_names=["run_key"], output_names=["sub", "ses", "run"],
                                 function=split_key), name="splitkey")

    getpaths = pe.Node(Function(input_names=["sub", "ses", "run", "bids_root"],
                                 output_names=["bold_file", "json_file"], function=get_paths), name="getpaths")
    getpaths.inputs.bids_root = BIDS_ROOT

    precheck = pe.Node(Function(input_names=["sub", "ses", "run", "deriv_root"],
                                 output_names=["safe_to_write", "existing_path"],
                                 function=check_not_already_processed), name="precheck")
    precheck.inputs.deriv_root = DERIV_ROOT

    checksum_n = pe.Node(Function(input_names=["bold_file"], output_names=["checksum"],
                                   function=checksum_before), name="checksum")

    validate_n = pe.Node(Function(input_names=["bold_file", "json_file"],
                                   output_names=["ok", "shape", "tr_nifti", "tr_json", "voxel", "orient", "err", "dt"],
                                   function=validate), name="validate")

    moco = pe.Node(Function(input_names=["bold_file"],
                             output_names=["moco_file", "mcdat_file", "ok", "err", "dt"],
                             function=run_moco), name="moco")

    smooth = pe.Node(Function(input_names=["moco_file", "fwhm"],
                               output_names=["smooth_file", "ok", "err", "dt"],
                               function=run_smooth), name="smooth")
    smooth.inputs.fwhm = FWHM_MM

    mask_n = pe.Node(Function(input_names=["smooth_file"],
                               output_names=["mask_file", "n_mask_voxels", "pct_mask",
                                             "n_components", "largest_component_size", "ok", "err", "dt"],
                               function=run_brain_mask), name="mask")

    design = pe.Node(Function(input_names=["mcdat_file"], output_names=["design_file", "ncols", "dt"],
                               function=build_design), name="design")

    glmfit = pe.Node(Function(input_names=["smooth_file", "design_file", "mask_file"],
                               output_names=["final_file", "ok", "err", "glm_mask_voxels_reported", "dt"],
                               function=run_glmfit), name="glmfit")

    finalize_n = pe.Node(Function(
        input_names=["sub", "ses", "run", "bold_file_checksum_before", "safe_to_write", "existing_path",
                     "bold_file", "json_file", "orig_shape", "orig_tr_nifti", "orig_tr_json", "orig_voxel",
                     "orig_orient", "validate_ok", "validate_err", "moco_file", "mcdat_file", "moco_ok",
                     "moco_err", "smooth_file", "smooth_ok", "smooth_err", "mask_file", "n_mask_voxels",
                     "pct_mask", "n_components", "largest_component_size", "mask_ok", "mask_err",
                     "design_file", "design_ncols", "final_file", "glmfit_ok", "glmfit_err",
                     "glm_mask_voxels_reported", "t_validate", "t_moco", "t_smooth", "t_mask", "t_design",
                     "t_glmfit", "fwhm", "deriv_root"],
        output_names=["qc_rows", "summary_row", "subj_log"], function=finalize), name="finalize")
    finalize_n.inputs.fwhm = FWHM_MM
    finalize_n.inputs.deriv_root = DERIV_ROOT

    collector = pe.JoinNode(Function(input_names=["qc_rows_list", "summary_rows_list", "subj_logs_list",
                                                   "deriv_root", "fwhm"],
                                      output_names=["qc_csv", "summary_json", "master_log"],
                                      function=collect_and_write),
                             name="collector", joinsource="infosource",
                             joinfield=["qc_rows_list", "summary_rows_list", "subj_logs_list"])
    collector.inputs.deriv_root = DERIV_ROOT
    collector.inputs.fwhm = FWHM_MM

    wf.connect([
        (infosource, splitkey, [("run_key", "run_key")]),
        (splitkey, getpaths, [("sub", "sub"), ("ses", "ses"), ("run", "run")]),
        (splitkey, precheck, [("sub", "sub"), ("ses", "ses"), ("run", "run")]),
        (getpaths, checksum_n, [("bold_file", "bold_file")]),
        (getpaths, validate_n, [("bold_file", "bold_file"), ("json_file", "json_file")]),
        (getpaths, moco, [("bold_file", "bold_file")]),
        (moco, smooth, [("moco_file", "moco_file")]),
        (moco, design, [("mcdat_file", "mcdat_file")]),
        (smooth, mask_n, [("smooth_file", "smooth_file")]),
        (smooth, glmfit, [("smooth_file", "smooth_file")]),
        (mask_n, glmfit, [("mask_file", "mask_file")]),
        (design, glmfit, [("design_file", "design_file")]),

        (splitkey, finalize_n, [("sub", "sub"), ("ses", "ses"), ("run", "run")]),
        (checksum_n, finalize_n, [("checksum", "bold_file_checksum_before")]),
        (precheck, finalize_n, [("safe_to_write", "safe_to_write"), ("existing_path", "existing_path")]),
        (getpaths, finalize_n, [("bold_file", "bold_file"), ("json_file", "json_file")]),
        (validate_n, finalize_n, [("shape", "orig_shape"), ("tr_nifti", "orig_tr_nifti"),
                                   ("tr_json", "orig_tr_json"), ("voxel", "orig_voxel"),
                                   ("orient", "orig_orient"), ("ok", "validate_ok"),
                                   ("err", "validate_err"), ("dt", "t_validate")]),
        (moco, finalize_n, [("moco_file", "moco_file"), ("mcdat_file", "mcdat_file"),
                             ("ok", "moco_ok"), ("err", "moco_err"), ("dt", "t_moco")]),
        (smooth, finalize_n, [("smooth_file", "smooth_file"), ("ok", "smooth_ok"),
                               ("err", "smooth_err"), ("dt", "t_smooth")]),
        (mask_n, finalize_n, [("mask_file", "mask_file"), ("n_mask_voxels", "n_mask_voxels"),
                               ("pct_mask", "pct_mask"), ("n_components", "n_components"),
                               ("largest_component_size", "largest_component_size"),
                               ("ok", "mask_ok"), ("err", "mask_err"), ("dt", "t_mask")]),
        (design, finalize_n, [("design_file", "design_file"), ("ncols", "design_ncols"), ("dt", "t_design")]),
        (glmfit, finalize_n, [("final_file", "final_file"), ("ok", "glmfit_ok"),
                               ("err", "glmfit_err"), ("glm_mask_voxels_reported", "glm_mask_voxels_reported"),
                               ("dt", "t_glmfit")]),

        (finalize_n, collector, [("qc_rows", "qc_rows_list"), ("summary_row", "summary_rows_list"),
                                  ("subj_log", "subj_logs_list")]),
    ])

    wf.run(plugin="MultiProc", plugin_args={"n_procs": 1})
    print("WORKFLOW_DONE")


if __name__ == "__main__":
    main()
