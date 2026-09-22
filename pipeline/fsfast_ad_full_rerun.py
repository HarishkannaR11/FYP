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

WORK_DIR = "/home/harish/fyp_work/fsfast_ad/work"
BIDS_ROOT = "/mnt/c/Users/krish/FYP/BIDS"
DERIV_ROOT = "/mnt/c/Users/krish/FYP/derivatives/fsfast"
MAPPING_CSV = "/mnt/c/Users/krish/FYP/audit/ad_session_mapping_final.csv"
CHECKSUM_BEFORE_CSV = "/mnt/c/Users/krish/FYP/audit/ad_rerun/source_checksums_before.csv"

FWHM_MM = 6.0
FD_THRESHOLD_MM = 0.5


def load_ad_runs():
    rows = list(csv.DictReader(open(MAPPING_CSV, newline="", encoding="utf-8")))
    return sorted([(r["participant_id"], r["session"]) for r in rows])


def split_key(run_key):
    sub, ses = run_key.rsplit("_", 1)
    return sub, ses, "run-01"


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def get_paths(sub, ses, run, bids_root):
    import os
    bold_file = f"{bids_root}/{sub}/{ses}/func/{sub}_{ses}_task-rest_{run}_bold.nii.gz"
    if not os.path.exists(bold_file):
        bold_file = bold_file.replace(".nii.gz", ".nii")
    json_file = bold_file.replace(".nii.gz", ".json").replace(".nii", ".json")
    return bold_file, json_file


def check_not_already_processed(sub, ses, run, deriv_root):
    import os
    existing = os.path.join(deriv_root, sub, ses, "func",
                             f"{sub}_{ses}_task-rest_{run}_desc-preproc_bold.nii.gz")
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
    out = os.path.join(os.getcwd(), "smooth_bold.nii.gz")
    cmd = ["mri_fwhm", "--i", moco_file, "--o", out, "--fwhm", str(fwhm), "--smooth-only"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if res.returncode != 0:
        return None, False, res.stderr[-2000:], dt
    return out, True, "", dt


def build_design(mcdat_file):
    import numpy as np, os, time
    t0 = time.time()
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


def run_glmfit(smooth_file, design_file):
    import subprocess, os, time
    t0 = time.time()
    glmdir = os.path.join(os.getcwd(), "glmdir")
    cmd = ["mri_glmfit", "--y", smooth_file, "--X", design_file, "--no-contrasts-ok",
           "--no-mask", "--glmdir", glmdir, "--eres-save", "--nii.gz"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    if res.returncode != 0:
        return None, False, res.stderr[-2000:], dt
    final = os.path.join(glmdir, "eres.nii.gz")
    dof_file = os.path.join(glmdir, "dof.dat")
    dof = None
    if os.path.exists(dof_file):
        try:
            dof = open(dof_file).read().strip()
        except Exception:
            pass
    return final, True, "", dt


def otsu_threshold(data):
    import numpy as np
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
    import numpy as np
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


def compute_fd(mc):
    import numpy as np
    rot_deg = mc[:, 1:4]
    trans = mc[:, 4:7]
    rot_rad = np.deg2rad(rot_deg)
    d_trans = np.diff(trans, axis=0)
    d_rot_mm = np.diff(rot_rad, axis=0) * 50.0
    return np.sum(np.abs(d_trans), axis=1) + np.sum(np.abs(d_rot_mm), axis=1)


def finalize(sub, ses, run, bold_file_checksum_before, safe_to_write, existing_path,
             bold_file, json_file, orig_shape, orig_tr_nifti, orig_tr_json, orig_voxel,
             orig_orient, validate_ok, validate_err, moco_file, mcdat_file, moco_ok, moco_err,
             smooth_file, smooth_ok, smooth_err, design_file, design_ncols, final_file,
             glmfit_ok, glmfit_err, t_validate, t_moco, t_smooth, t_design, t_glmfit,
             fwhm, deriv_root):
    import nibabel as nib
    import numpy as np
    import os
    import json as js
    import shutil
    import time
    import hashlib

    # NOTE: nipype Function nodes serialize only this function's own isolated
    # source -- sibling top-level helpers are NOT available at exec time, so
    # they must be defined here, nested, even though identical top-level
    # copies also exist in this file for readability/reuse elsewhere.
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

    def compute_fd(mc):
        rot_deg = mc[:, 1:4]
        trans = mc[:, 4:7]
        rot_rad = np.deg2rad(rot_deg)
        d_trans = np.diff(trans, axis=0)
        d_rot_mm = np.diff(rot_rad, axis=0) * 50.0
        return np.sum(np.abs(d_trans), axis=1) + np.sum(np.abs(d_rot_mm), axis=1)

    FD_THRESHOLD_MM = 0.5

    errors = []
    warnings = []
    notes = []

    if not safe_to_write:
        errors.append(f"OUTPUT ALREADY EXISTS at {existing_path} -- refusing to overwrite")

    for ok, err, stage in [(validate_ok, validate_err, "validate"), (moco_ok, moco_err, "moco"),
                            (smooth_ok, smooth_err, "smooth"), (glmfit_ok, glmfit_err, "glmfit")]:
        if not ok:
            errors.append(f"{stage}: {err}")

    # ---- checksum re-verification (source must not have changed during processing) ----
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
    out_dir = os.path.join(deriv_root, sub, ses, "func")
    final_bold_out = motion_tsv_out = json_out = None

    metrics = {}

    if qc_pass:
        os.makedirs(out_dir, exist_ok=True)

        in_img = nib.load(bold_file)
        in_data = in_img.get_fdata()
        expect_shape = eval(orig_shape)

        # ---- A. input/output integrity ----
        out_final_img = nib.load(final_file)
        out_data = out_final_img.get_fdata()
        nan_count = int(np.isnan(out_data).sum())
        inf_count = int(np.isinf(out_data).sum())
        empty_frames = [t for t in range(out_data.shape[3]) if np.allclose(out_data[..., t], 0)]
        if nan_count or inf_count:
            errors.append(f"NaN={nan_count} Inf={inf_count} in final output")
            qc_pass = False
        if empty_frames:
            errors.append(f"{len(empty_frames)} empty frames in final output")
            qc_pass = False
        shape_preserved = (out_final_img.shape == expect_shape)
        if not shape_preserved:
            errors.append(f"shape changed: {expect_shape} -> {out_final_img.shape}")
            qc_pass = False
        vol_preserved = (out_final_img.shape[3] == expect_shape[3]) if len(out_final_img.shape) > 3 else False

        # ---- B. motion QC ----
        mc = np.loadtxt(mcdat_file)
        fd = compute_fd(mc)
        mean_fd = float(np.mean(fd))
        median_fd = float(np.median(fd))
        max_fd = float(np.max(fd))
        high_fd_pct = float(100 * np.mean(fd > FD_THRESHOLD_MM))
        max_trans = float(np.max(mc[:, 9]))
        max_rot = float(np.max(np.abs(mc[:, 1:4])))
        if mc.shape[0] != expect_shape[3]:
            errors.append(f"motion params row count {mc.shape[0]} != {expect_shape[3]}")
            qc_pass = False

        # ---- C. temporal QC ----
        in_tmean = in_data.mean(axis=3)
        thresh_in = otsu_threshold(in_tmean)
        brain_mask = in_tmean > thresh_in
        global_signal_raw = in_data[brain_mask].mean(axis=0) if brain_mask.sum() else None
        temporal_sd_raw = float(np.mean(in_data.std(axis=3)[brain_mask])) if brain_mask.sum() else None
        temporal_sd_final = float(np.mean(out_data.std(axis=3)[brain_mask])) if brain_mask.sum() else None

        spikes_first_vol_only = False
        spikes_other = []
        if global_signal_raw is not None and global_signal_raw.std() > 0:
            z = (global_signal_raw - global_signal_raw.mean()) / global_signal_raw.std()
            spike_idx = np.where(np.abs(z) > 3)[0].tolist()
            spikes_other = [s for s in spike_idx if s != 0]
            spikes_first_vol_only = (0 in spike_idx)
            if spikes_other:
                warnings.append(f"raw global-signal spike(s) at NON-first volumes: {spikes_other}")

        # ---- D. tSNR on smoothed magnitude-domain intermediate (NOT the residual) ----
        sm_data = nib.load(smooth_file).get_fdata()
        sm_tmean = sm_data.mean(axis=3)
        sm_std = sm_data.std(axis=3)
        with np.errstate(divide="ignore", invalid="ignore"):
            tsnr_map = np.where(sm_std > 0, sm_tmean / sm_std, 0)
        tsnr_vals = tsnr_map[brain_mask]
        mean_tsnr = float(np.mean(tsnr_vals)) if tsnr_vals.size else None
        median_tsnr = float(np.median(tsnr_vals)) if tsnr_vals.size else None
        notes.append("tSNR computed on pre-GLM smoothed (magnitude-domain) intermediate, "
                     "NOT the zero-mean final residual -- standard tSNR is undefined for a zero-mean signal")

        # ---- E. motion-regression QC ----
        design_match = None
        motion_assoc_before = motion_assoc_after = None
        if os.path.exists(design_file):
            X_saved = np.loadtxt(design_file)
            if X_saved.shape[1] >= 9:
                design_match = bool(np.allclose(X_saved[:, 3:9], mc[:, 1:7], atol=1e-4))
                if not design_match:
                    errors.append("DESIGN MATRIX MOTION COLUMNS DO NOT MATCH mc-afni2 OUTPUT")
                    qc_pass = False
        if global_signal_raw is not None:
            Xr = np.column_stack([np.ones(len(mc)), mc[:, 1:7]])
            beta, *_ = np.linalg.lstsq(Xr, global_signal_raw, rcond=None)
            ss_res = np.sum((global_signal_raw - Xr @ beta) ** 2)
            ss_tot = np.sum((global_signal_raw - global_signal_raw.mean()) ** 2)
            motion_assoc_before = float(1 - ss_res / ss_tot) if ss_tot > 0 else None
            global_signal_final = out_data[brain_mask].mean(axis=0) if brain_mask.sum() else None
            if global_signal_final is not None:
                beta2, *_ = np.linalg.lstsq(Xr, global_signal_final, rcond=None)
                ss_res2 = np.sum((global_signal_final - Xr @ beta2) ** 2)
                ss_tot2 = np.sum((global_signal_final - global_signal_final.mean()) ** 2)
                motion_assoc_after = float(1 - ss_res2 / ss_tot2) if ss_tot2 > 0 else None
                if motion_assoc_before is not None and motion_assoc_after is not None and motion_assoc_after > motion_assoc_before:
                    warnings.append("motion association DID NOT decrease after regression")

        # ---- F. smoothing QC ----
        moco_tmean = nib.load(moco_file).get_fdata().mean(axis=3)
        vs = in_img.header.get_zooms()[:3]
        fwhm_before = spatial_fwhm_estimate(moco_tmean, brain_mask, vs)
        fwhm_after = spatial_fwhm_estimate(sm_tmean, brain_mask, vs)
        notes.append(f"requested smoothing kernel: {fwhm}mm FWHM; independently ESTIMATED "
                     f"(Forman et al. spatial-derivative method) before={fwhm_before} after={fwhm_after} "
                     f"-- NOT assumed to equal {fwhm}mm exactly")

        # ---- G. spatial coverage (NO cropping ever applied -- check only) ----
        thresh_sm = otsu_threshold(sm_tmean)
        cov_in = np.array([(in_tmean[:, :, z] > thresh_in).sum() / in_tmean[:, :, z].size for z in range(in_tmean.shape[2])])
        cov_sm = np.array([(sm_tmean[:, :, z] > thresh_sm).sum() / sm_tmean[:, :, z].size for z in range(sm_tmean.shape[2])])
        lost = set(np.where(cov_in >= 0.05)[0]) - set(np.where(cov_sm >= 0.05)[0])
        coverage_status = "OK" if len(lost) <= 1 else f"CHECK({sorted(lost)})"
        if coverage_status.startswith("CHECK"):
            warnings.append(f"spatial coverage difference beyond single boundary slice: {sorted(lost)}")
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
            "mean_FD": mean_fd, "median_FD": median_fd, "max_FD": max_fd,
            "high_FD_percent": high_fd_pct,
            "temporal_SD_raw": temporal_sd_raw, "temporal_SD_final": temporal_sd_final,
            "mean_tSNR": mean_tsnr, "median_tSNR": median_tsnr,
            "motion_association_before": motion_assoc_before, "motion_association_after": motion_assoc_after,
            "design_matrix_motion_match": design_match,
            "estimated_FWHM_before_mm": fwhm_before, "estimated_FWHM_after_mm": fwhm_after,
            "requested_FWHM_mm": fwhm,
            "spatial_coverage_status": coverage_status,
            "NaN_count": nan_count, "Inf_count": inf_count,
            "shape_preserved": shape_preserved, "voxel_size_preserved": voxel_preserved,
            "TR_preserved": tr_preserved, "volume_count_preserved": vol_preserved,
            "spikes_first_volume_only": spikes_first_vol_only, "spikes_other_volumes": spikes_other,
            "checksum_before": bold_file_checksum_before, "checksum_after": checksum_after,
            "checksum_match": checksum_ok,
        }

        if qc_pass:
            final_bold_out = os.path.join(out_dir, f"{sub}_{ses}_task-rest_{run}_desc-preproc_bold.nii.gz")
            if os.path.isfile(final_bold_out):
                errors.append(f"OUTPUT APPEARED DURING PROCESSING -- refusing to overwrite")
                qc_pass = False
            else:
                shutil.copy(final_file, final_bold_out)
                motion_tsv_out = os.path.join(out_dir, f"{sub}_{ses}_task-rest_{run}_desc-motion_timeseries.tsv")
                with open(mcdat_file) as fin, open(motion_tsv_out, "w") as fout:
                    fout.write("n_TR\troll\tpitch\tyaw\tdS\tdL\tdP\trmsold\trmsnew\ttrans_mm\n")
                    for line in fin:
                        fout.write("\t".join(line.split()) + "\n")

                json_out = os.path.join(out_dir, f"{sub}_{ses}_task-rest_{run}_desc-preproc_bold.json")
                sidecar = {
                    "Sources": [os.path.basename(bold_file)],
                    "OriginalDimensions": orig_shape, "OriginalVoxelSize": orig_voxel,
                    "OriginalOrientation": orig_orient, "OriginalRepetitionTime": orig_tr_json,
                    "SkullStripped": False,
                    "PreprocessingPipeline": "FS-FAST + Nipype (BOLD-only, no T1w) -- FULL AD RERUN",
                    "ProcessingSteps": [
                        {"step": "data_validation", "tool": "NiBabel/PyBIDS", "status": "PASS"},
                        {"step": "motion_correction", "tool": "mc-afni2", "reference_frame": 0,
                         "max_translation_mm": max_trans, "max_rotation_deg": max_rot,
                         "mean_FD_mm": mean_fd, "status": "PASS"},
                        {"step": "slice_timing_correction", "status": "SKIPPED",
                         "reason": "Slice-timing correction skipped: acquisition timing/order not "
                                   "recoverable from available BIDS/DICOM metadata."},
                        {"step": "spatial_smoothing", "tool": "mri_fwhm", "requested_fwhm_mm": fwhm,
                         "estimated_fwhm_before_mm": fwhm_before, "estimated_fwhm_after_mm": fwhm_after,
                         "status": "PASS"},
                        {"step": "temporal_filtering_and_nuisance_regression", "tool": "mri_glmfit",
                         "design_matrix_columns": design_ncols,
                         "design_regressors": ["intercept", "linear_trend", "quadratic_trend",
                                                "motion_roll", "motion_pitch", "motion_yaw",
                                                "motion_dS", "motion_dL", "motion_dP"],
                         "design_matches_motion_output": design_match,
                         "motion_association_R2_before": motion_assoc_before,
                         "motion_association_R2_after": motion_assoc_after,
                         "wm_csf_regression": "UNAVAILABLE (no T1w segmentation)", "status": "PASS"},
                        {"step": "no_arbitrary_slice_cropping", "status": "CONFIRMED",
                         "note": "Original spatial dimensions preserved; no first/last slices removed"},
                    ],
                    "QCMetrics": metrics,
                }
                with open(json_out, "w") as f:
                    js.dump(sidecar, f, indent=2, default=str)
        else:
            warnings.append("QC failed -- outputs NOT copied to derivatives")

    total_time = sum(x for x in [t_validate, t_moco, t_smooth, t_design, t_glmfit] if x)

    logs_dir = os.path.join(deriv_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    subj_log = os.path.join(logs_dir, f"{sub}_{ses}_processing.log")
    log_lines = [
        f"=== {sub} {ses} {run} processing log (FULL AD RERUN) ===",
        f"Input BOLD: {bold_file}", f"Input JSON: {json_file}",
        f"Source checksum before: {bold_file_checksum_before}",
        f"Source checksum after: {metrics.get('checksum_after', 'N/A')}",
        f"Checksum match: {metrics.get('checksum_match', 'N/A')}",
        f"Original dimensions: {orig_shape}  TR(nifti)={orig_tr_nifti}  TR(json)={orig_tr_json}",
        f"Voxel size: {orig_voxel}  Orientation: {orig_orient}",
        f"Validation: {'PASS' if validate_ok else 'FAIL - ' + validate_err} (time {t_validate:.2f}s)",
        f"Motion correction (mc-afni2): {'PASS' if moco_ok else 'FAIL - ' + moco_err} (time {t_moco:.2f}s)",
        f"  max_translation={metrics.get('max_translation_mm')}mm  max_rotation={metrics.get('max_rotation_deg')}deg",
        f"  mean_FD={metrics.get('mean_FD')}mm  median_FD={metrics.get('median_FD')}mm  max_FD={metrics.get('max_FD')}mm",
        f"  high_FD_percent(>{FD_THRESHOLD_MM}mm, reference only)={metrics.get('high_FD_percent')}%",
        f"Slice-timing correction: SKIPPED -- acquisition timing/order not recoverable from available BIDS/DICOM metadata.",
        f"Spatial smoothing (mri_fwhm, requested FWHM={fwhm}mm): {'PASS' if smooth_ok else 'FAIL - ' + smooth_err} (time {t_smooth:.2f}s)",
        f"  estimated FWHM before={metrics.get('estimated_FWHM_before_mm')}mm after={metrics.get('estimated_FWHM_after_mm')}mm (independently measured, not assumed == {fwhm}mm)",
        f"Design matrix: {design_ncols} columns (intercept, linear, quadratic, 6 motion params) (time {t_design:.2f}s)",
        f"  design matrix motion columns match mc-afni2 output: {metrics.get('design_matrix_motion_match')}",
        f"GLM fit (mri_glmfit): {'PASS' if glmfit_ok else 'FAIL - ' + glmfit_err} (time {t_glmfit:.2f}s)",
        f"  motion-signal association R2 before={metrics.get('motion_association_before')} after={metrics.get('motion_association_after')}",
        f"  WM/CSF nuisance regression: UNAVAILABLE (no T1w-derived segmentation)",
        f"tSNR (on pre-GLM smoothed magnitude-domain intermediate): mean={metrics.get('mean_tSNR')} median={metrics.get('median_tSNR')}",
        f"Temporal SD: raw={metrics.get('temporal_SD_raw')} final={metrics.get('temporal_SD_final')}",
        f"Raw global-signal spikes: first-volume-only(expected/benign)={metrics.get('spikes_first_volume_only')}  other-volumes={metrics.get('spikes_other_volumes')}",
        f"Spatial coverage: {metrics.get('spatial_coverage_status')} -- no arbitrary slice cropping performed",
        f"NaN={metrics.get('NaN_count')}  Inf={metrics.get('Inf_count')}",
        f"shape_preserved={metrics.get('shape_preserved')}  voxel_preserved={metrics.get('voxel_size_preserved')}  TR_preserved={metrics.get('TR_preserved')}  volumes_preserved={metrics.get('volume_count_preserved')}",
        f"Total processing time: {total_time:.2f}s",
        f"QC: {'PASS' if qc_pass else 'FAIL'}",
        f"Errors: {'; '.join(errors) if errors else 'none'}",
        f"Warnings: {'; '.join(warnings) if warnings else 'none'}",
        f"Output BOLD: {final_bold_out}", f"Output motion TSV: {motion_tsv_out}", f"Output JSON: {json_out}",
        "",
    ]
    with open(subj_log, "w") as f:
        f.write("\n".join(log_lines))

    qc_rows = []
    for stage_name, path in [("input_original_BIDS", bold_file), ("motion_corrected", moco_file),
                              ("smoothed", smooth_file), ("final_preproc_residual", final_file)]:
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
    from collections import Counter

    FD_THRESHOLD_MM = 0.5

    qc_dir = os.path.join(deriv_root, "qc")
    logs_dir = os.path.join(deriv_root, "logs")
    os.makedirs(qc_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    all_fields = set()
    for rows in qc_rows_list:
        for r in rows:
            all_fields.update(r.keys())
    fieldnames = sorted(all_fields)
    qc_csv = os.path.join(qc_dir, "preprocessing_qc.csv")
    with open(qc_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for rows in qc_rows_list:
            for r in rows:
                for fn in fieldnames:
                    r.setdefault(fn, "")
                writer.writerow(r)

    def numeric(field):
        return [float(r[field]) for r in summary_rows_list if r.get(field) not in ("", None)]

    def outliers(field):
        vals = numeric(field)
        if len(vals) < 4:
            return []
        arr = np.array(vals)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        return [(r["sub"], r["ses"], float(r[field])) for r in summary_rows_list
                if r.get(field) not in ("", None) and (float(r[field]) < lo or float(r[field]) > hi)]

    motion_outliers = outliers("mean_FD")
    tsnr_outliers = outliers("mean_tSNR")
    fwhm_outliers = outliers("estimated_FWHM_after_mm")

    outlier_map = {}
    for s, se, v in motion_outliers:
        outlier_map.setdefault((s, se), []).append(f"statistical outlier: mean FD={v:.3f}mm (IQR-based)")
    for s, se, v in tsnr_outliers:
        outlier_map.setdefault((s, se), []).append(f"statistical outlier: mean tSNR={v:.2f} (IQR-based)")
    for s, se, v in fwhm_outliers:
        outlier_map.setdefault((s, se), []).append(f"statistical outlier: FWHM after={v:.2f}mm (IQR-based)")

    for r in summary_rows_list:
        key = (r["sub"], r["ses"])
        if key in outlier_map:
            for reason in outlier_map[key]:
                r["warnings"] = (r.get("warnings", "") + f"; {reason}").strip("; ")
            if r["qc_status"] == "PASS" and r["warnings"]:
                pass  # PASS with warnings noted -- overall status stays PASS unless errors present; outliers become WARN below

    n_fail = sum(1 for r in summary_rows_list if r["qc_status"] == "FAIL")
    n_with_warnings = sum(1 for r in summary_rows_list if r["qc_status"] == "PASS" and r.get("warnings"))
    n_clean_pass = sum(1 for r in summary_rows_list if r["qc_status"] == "PASS" and not r.get("warnings"))

    summary = {
        "total_acquisitions": len(summary_rows_list),
        "clean_pass": n_clean_pass, "pass_with_warnings": n_with_warnings, "fail": n_fail,
        "requested_smoothing_fwhm_mm": fwhm,
        "FD_threshold_mm_reference_only": FD_THRESHOLD_MM,
        "slice_timing_correction": "SKIPPED -- acquisition timing/order not recoverable from available BIDS/DICOM metadata.",
        "arbitrary_slice_cropping": "NOT PERFORMED -- original spatial dimensions preserved for all runs.",
        "wm_csf_nuisance_regression": "UNAVAILABLE (no T1w)",
        "motion_outliers": [{"sub": s, "ses": se, "mean_FD": v} for s, se, v in motion_outliers],
        "tsnr_outliers": [{"sub": s, "ses": se, "mean_tSNR": v} for s, se, v in tsnr_outliers],
        "fwhm_outliers": [{"sub": s, "ses": se, "FWHM_after": v} for s, se, v in fwhm_outliers],
        "runs": summary_rows_list,
    }
    with open(os.path.join(qc_dir, "preprocessing_qc_summary.json"), "w") as f:
        js.dump(summary, f, indent=2, default=str)

    master_log = os.path.join(logs_dir, "preprocessing.log")
    with open(master_log, "w") as f:
        f.write("=== FS-FAST FULL AD-GROUP RERUN ===\n")
        f.write(f"Timestamp: {time.ctime()}\n")
        f.write(f"Total: {len(summary_rows_list)}  clean_pass={n_clean_pass}  pass_with_warnings={n_with_warnings}  fail={n_fail}\n\n")
        for s in summary_rows_list:
            f.write(f"{s['sub']} {s['ses']}: QC={s['qc_status']} time={s['total_processing_time_s']}s "
                    f"mean_FD={s.get('mean_FD')} mean_tSNR={s.get('mean_tSNR')} "
                    f"errors={s['errors'] or 'none'} warnings={s['warnings'] or 'none'}\n")

    return qc_csv, os.path.join(qc_dir, "preprocessing_qc_summary.json"), master_log


def write_dataset_description(deriv_root):
    import json as js
    import os
    path = os.path.join(deriv_root, "dataset_description.json")
    if os.path.exists(path):
        return
    desc = {"Name": "FS-FAST BOLD-only preprocessing derivatives", "BIDSVersion": "1.8.0",
            "DatasetType": "derivative",
            "GeneratedBy": [{"Name": "FS-FAST + Nipype (custom pipeline)"}]}
    with open(path, "w") as f:
        js.dump(desc, f, indent=2)


def main():
    write_dataset_description(DERIV_ROOT)
    runs = load_ad_runs()
    print(f"=== Processing exactly these {len(runs)} AD acquisitions ===")
    for sub, ses in runs:
        print(f"  {sub} {ses} run-01")
    assert len(runs) == 25, f"expected 25 runs, got {len(runs)}"

    wf = pe.Workflow(name="fsfast_ad_full_rerun", base_dir=WORK_DIR)

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

    design = pe.Node(Function(input_names=["mcdat_file"], output_names=["design_file", "ncols", "dt"],
                               function=build_design), name="design")

    glmfit = pe.Node(Function(input_names=["smooth_file", "design_file"],
                               output_names=["final_file", "ok", "err", "dt"],
                               function=run_glmfit), name="glmfit")

    finalize_n = pe.Node(Function(
        input_names=["sub", "ses", "run", "bold_file_checksum_before", "safe_to_write", "existing_path",
                     "bold_file", "json_file", "orig_shape", "orig_tr_nifti", "orig_tr_json", "orig_voxel",
                     "orig_orient", "validate_ok", "validate_err", "moco_file", "mcdat_file", "moco_ok",
                     "moco_err", "smooth_file", "smooth_ok", "smooth_err", "design_file", "design_ncols",
                     "final_file", "glmfit_ok", "glmfit_err", "t_validate", "t_moco", "t_smooth", "t_design",
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
        (smooth, glmfit, [("smooth_file", "smooth_file")]),
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
        (design, finalize_n, [("design_file", "design_file"), ("ncols", "design_ncols"), ("dt", "t_design")]),
        (glmfit, finalize_n, [("final_file", "final_file"), ("ok", "glmfit_ok"),
                               ("err", "glmfit_err"), ("dt", "t_glmfit")]),

        (finalize_n, collector, [("qc_rows", "qc_rows_list"), ("summary_row", "summary_rows_list"),
                                  ("subj_log", "subj_logs_list")]),
    ])

    wf.run(plugin="MultiProc", plugin_args={"n_procs": 2})
    print("WORKFLOW_DONE")


if __name__ == "__main__":
    main()
