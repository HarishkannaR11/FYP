import os
import csv
import json
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BIDS_ROOT = "/mnt/c/Users/krish/FYP/BIDS"
DERIV_ROOT = "/mnt/c/Users/krish/FYP/derivatives/fsfast"
WORK_AD5 = "/home/harish/fyp_work/ad5/work/fsfast_5ad_preprocessing"
WORK_AD20 = "/home/harish/fyp_work/ad20/work/fsfast_20ad_preprocessing"
MAPPING_CSV = "/mnt/c/Users/krish/FYP/audit/ad_session_mapping_final.csv"

OUT_DIR = "/mnt/c/Users/krish/FYP/audit/ad_preprocessing_quantitative_qc"
FIG = {k: os.path.join(OUT_DIR, "figures", k) for k in ["motion", "tsnr", "temporal", "smoothing", "summary"]}
CSV_OUT = os.path.join(OUT_DIR, "quantitative_qc.csv")
TXT_OUT = os.path.join(OUT_DIR, "quantitative_qc_summary.txt")
JSON_OUT = os.path.join(OUT_DIR, "quantitative_qc.json")

FD_THRESHOLD_MM = 0.5   # Power et al. 2012 commonly-cited adult resting-state threshold; reported, not used to exclude
LOWFREQ_CUTOFF_HZ = 0.01  # standard scanner-drift/low-frequency band cutoff; reported explicitly

CSV_FIELDS = [
    "participant_id", "session", "mean_FD", "max_FD", "max_translation_mm", "mean_translation_mm",
    "max_rotation_deg", "high_motion_percent", "mean_tSNR", "median_tSNR", "tSNR_p25", "tSNR_p75",
    "temporal_SD_raw", "temporal_SD_final", "low_frequency_power_before", "low_frequency_power_after",
    "motion_association_before", "motion_association_after", "estimated_FWHM_before_mm",
    "estimated_FWHM_after_mm", "NaN_count", "Inf_count", "shape_preserved", "voxel_size_preserved",
    "TR_preserved", "volume_count_preserved", "spatial_coverage_status", "overall_QC", "notes",
]


def find_workdir(sub, ses):
    c1 = os.path.join(WORK_AD5, f"_sub_{sub}")
    if os.path.isdir(c1):
        return c1
    c2 = os.path.join(WORK_AD20, f"_run_key_{sub}_{ses}")
    if os.path.isdir(c2):
        return c2
    return None


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


def compute_fd(mc):
    """Power et al. 2012 framewise displacement: sum of abs frame-to-frame
    differences of 3 translations (mm) + 3 rotations converted to mm via a
    50mm assumed head radius (rotation_rad * 50). mc columns (mcdat):
    0=n, 1=roll,2=pitch,3=yaw (deg), 4=dS,5=dL,6=dP (mm), 7=rmsold,8=rmsnew,9=trans."""
    rot_deg = mc[:, 1:4]
    trans = mc[:, 4:7]
    rot_rad = np.deg2rad(rot_deg)
    d_trans = np.diff(trans, axis=0)
    d_rot_mm = np.diff(rot_rad, axis=0) * 50.0
    fd = np.sum(np.abs(d_trans), axis=1) + np.sum(np.abs(d_rot_mm), axis=1)
    return fd  # length N-1


def spatial_fwhm_estimate(vol3d, mask, voxel_sizes):
    """Forman et al. 1995 / FSL 'smoothest'-style spatial-derivative FWHM
    estimator, per-axis, combined via geometric mean. Applied within mask."""
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
    return float(np.exp(np.mean(np.log(fwhms))))  # geometric mean


def lowfreq_power_fraction(signal, tr, cutoff_hz):
    n = len(signal)
    signal = signal - np.mean(signal)
    freqs = np.fft.rfftfreq(n, d=tr)
    power = np.abs(np.fft.rfft(signal)) ** 2
    total = power.sum()
    if total == 0:
        return 0.0, freqs, power
    low = power[freqs <= cutoff_hz].sum()
    return float(low / total), freqs, power


def validate_run(sub, ses, run, idx, total):
    print(f"[{idx}/{total}] {sub} {ses}")
    notes = []

    input_file = f"{BIDS_ROOT}/{sub}/{ses}/func/{sub}_{ses}_task-rest_{run}_bold.nii.gz"
    if not os.path.exists(input_file):
        input_file = input_file.replace(".nii.gz", ".nii")
    output_file = f"{DERIV_ROOT}/{sub}/{ses}/func/{sub}_{ses}_task-rest_{run}_desc-preproc_bold.nii.gz"

    row = {"participant_id": sub, "session": ses}

    if not os.path.exists(input_file) or not os.path.exists(output_file):
        row["overall_QC"] = "FAIL"
        row["notes"] = "input or output file missing"
        return row, None

    in_img = nib.load(input_file)
    out_img = nib.load(output_file)
    in_data = in_img.get_fdata()
    out_data = out_img.get_fdata()
    in_shape, out_shape = in_img.shape, out_img.shape
    in_zooms, out_zooms = in_img.header.get_zooms(), out_img.header.get_zooms()
    in_tr = float(in_zooms[3]) if len(in_zooms) > 3 else None
    out_tr = float(out_zooms[3]) if len(out_zooms) > 3 else None

    workdir = find_workdir(sub, ses)
    moco_file = os.path.join(workdir, "moco", "moco_bold.nii.gz") if workdir else None
    mcdat_file = os.path.join(workdir, "moco", "moco_bold.mcdat") if workdir else None
    smooth_file = os.path.join(workdir, "smooth", "smooth_bold.nii.gz") if workdir else None
    design_file = os.path.join(workdir, "design", "design_matrix.txt") if workdir else None

    # ============ 1. MOTION QC ============
    mean_fd = max_fd = high_motion_pct = None
    max_trans = mean_trans = max_rot = None
    fd = None
    if mcdat_file and os.path.exists(mcdat_file):
        mc = np.loadtxt(mcdat_file)
        fd = compute_fd(mc)
        mean_fd = float(np.mean(fd))
        max_fd = float(np.max(fd))
        high_motion_pct = float(100 * np.mean(fd > FD_THRESHOLD_MM))
        max_trans = float(np.max(mc[:, 9]))
        mean_trans = float(np.mean(mc[:, 9]))
        max_rot = float(np.max(np.abs(mc[:, 1:4])))
    else:
        notes.append("motion .mcdat unavailable")

    # ============ 6. DATA INTEGRITY (computed early, needed elsewhere) ============
    nan_count = int(np.isnan(out_data).sum())
    inf_count = int(np.isinf(out_data).sum())
    empty_frames = [t for t in range(out_shape[3]) if np.allclose(out_data[..., t], 0)]
    constant_frames = [t for t in range(out_shape[3]) if np.std(out_data[..., t]) == 0]
    shape_preserved = (in_shape == out_shape)
    voxel_preserved = bool(np.allclose(in_zooms[:3], out_zooms[:3], atol=0.01))
    tr_preserved = (in_tr is not None and out_tr is not None and abs(in_tr - out_tr) < 0.01)
    volcount_preserved = (in_shape[3] == out_shape[3]) if len(in_shape) > 3 and len(out_shape) > 3 else False

    in_tmean = in_data.mean(axis=3)
    thresh_in = otsu_threshold(in_tmean)
    brain_mask = in_tmean > thresh_in

    coverage_status = "OK"
    if smooth_file and os.path.exists(smooth_file):
        smooth_mean = nib.load(smooth_file).get_fdata().mean(axis=3)
        thresh_sm = otsu_threshold(smooth_mean)
        cov_in = np.array([(in_tmean[:, :, z] > thresh_in).sum() / in_tmean[:, :, z].size for z in range(in_shape[2])])
        cov_sm = np.array([(smooth_mean[:, :, z] > thresh_sm).sum() / smooth_mean[:, :, z].size for z in range(smooth_mean.shape[2])])
        lost = set(np.where(cov_in >= 0.05)[0]) - set(np.where(cov_sm >= 0.05)[0])
        coverage_status = "OK" if len(lost) <= 1 else f"CHECK({sorted(lost)})"
    else:
        coverage_status = "UNVERIFIED(no smoothed intermediate)"

    # ============ 2. SIGNAL QC (tSNR) ============
    # NOTE (per instruction not to misinterpret metrics): the final glmfit
    # output (eres.nii.gz) is a zero-mean GLM residual after intercept
    # removal -- its voxelwise temporal MEAN is ~0 by construction, so a
    # standard tSNR = mean/std computed on it is not meaningful (near-zero
    # numerator regardless of data quality). The scientifically appropriate
    # "after preprocessing" comparison for tSNR is the pre-GLM, still-
    # magnitude-domain smoothed/motion-corrected intermediate. This is
    # applied here and stated explicitly, not silently substituted.
    in_std = in_data.std(axis=3)
    with np.errstate(divide="ignore", invalid="ignore"):
        in_tsnr_map = np.where(in_std > 0, in_tmean / in_std, 0)
    in_tsnr_vals = in_tsnr_map[brain_mask]
    mean_tsnr = float(np.mean(in_tsnr_vals)) if in_tsnr_vals.size else None
    median_tsnr = float(np.median(in_tsnr_vals)) if in_tsnr_vals.size else None
    tsnr_p25 = float(np.percentile(in_tsnr_vals, 25)) if in_tsnr_vals.size else None
    tsnr_p75 = float(np.percentile(in_tsnr_vals, 75)) if in_tsnr_vals.size else None

    final_tsnr_map = None
    if smooth_file and os.path.exists(smooth_file):
        sm_data = nib.load(smooth_file).get_fdata()
        sm_tmean = sm_data.mean(axis=3)
        sm_std = sm_data.std(axis=3)
        with np.errstate(divide="ignore", invalid="ignore"):
            final_tsnr_map = np.where(sm_std > 0, sm_tmean / sm_std, 0)
        notes.append(f"tSNR 'after' computed on pre-GLM smoothed intermediate (magnitude domain), "
                     f"NOT the zero-mean residual -- standard tSNR is undefined for a zero-mean signal")
    else:
        notes.append("smoothed intermediate unavailable -- tSNR after-preprocessing not computed")

    # ============ 3. TEMPORAL QC ============
    temporal_sd_raw = float(np.mean(in_std[brain_mask])) if brain_mask.sum() else None
    out_std = out_data.std(axis=3)
    temporal_sd_final = float(np.mean(out_std[brain_mask])) if brain_mask.sum() else None

    global_signal_raw = in_data[brain_mask].mean(axis=0) if brain_mask.sum() else None
    global_signal_final = out_data[brain_mask].mean(axis=0) if brain_mask.sum() else None

    lf_before = lf_after = None
    psd_data = None
    if global_signal_raw is not None and in_tr:
        lf_before, freqs_b, power_b = lowfreq_power_fraction(global_signal_raw, in_tr, LOWFREQ_CUTOFF_HZ)
    if global_signal_final is not None and out_tr:
        lf_after, freqs_a, power_a = lowfreq_power_fraction(global_signal_final, out_tr, LOWFREQ_CUTOFF_HZ)
        psd_data = (freqs_b, power_b, freqs_a, power_a) if global_signal_raw is not None else None

    # abnormal temporal behavior: spike detection on raw global signal.
    # NOTE: a spike at volume 0 ONLY is the well-documented non-steady-state
    # effect (longitudinal magnetization has not reached equilibrium at the
    # start of an EPI run) -- this is a property of the RAW ACQUISITION
    # itself, present before any preprocessing, and is expected/benign, not
    # a data-quality defect. It is reported for transparency but does NOT by
    # itself drive WARN status. A spike at any OTHER volume index is a
    # genuinely unexpected mid-run event and DOES drive WARN status.
    spikes = []
    spikes_excl_first = []
    if global_signal_raw is not None:
        gsr = global_signal_raw
        z = (gsr - gsr.mean()) / gsr.std() if gsr.std() > 0 else np.zeros_like(gsr)
        spikes = np.where(np.abs(z) > 3)[0].tolist()
        spikes_excl_first = [s for s in spikes if s != 0]
        if spikes:
            first_vol_note = " (includes volume 0 -- expected non-steady-state first-volume effect, not a defect)" if 0 in spikes else ""
            notes.append(f"raw global-signal spikes (|z|>3) at volumes: {spikes}{first_vol_note}")

    # ============ 4. MOTION-REGRESSION VALIDATION ============
    motion_assoc_before = motion_assoc_after = None
    design_match = None
    if mcdat_file and os.path.exists(mcdat_file) and global_signal_raw is not None:
        mc = np.loadtxt(mcdat_file)
        motion_regs = mc[:, 1:7]
        X = np.column_stack([np.ones(len(motion_regs)), motion_regs])
        beta, *_ = np.linalg.lstsq(X, global_signal_raw, rcond=None)
        pred = X @ beta
        ss_res = np.sum((global_signal_raw - pred) ** 2)
        ss_tot = np.sum((global_signal_raw - global_signal_raw.mean()) ** 2)
        motion_assoc_before = float(1 - ss_res / ss_tot) if ss_tot > 0 else None

        if global_signal_final is not None:
            beta2, *_ = np.linalg.lstsq(X, global_signal_final, rcond=None)
            pred2 = X @ beta2
            ss_res2 = np.sum((global_signal_final - pred2) ** 2)
            ss_tot2 = np.sum((global_signal_final - global_signal_final.mean()) ** 2)
            motion_assoc_after = float(1 - ss_res2 / ss_tot2) if ss_tot2 > 0 else None

        if design_file and os.path.exists(design_file):
            X_saved = np.loadtxt(design_file)
            if X_saved.shape[1] >= 9:
                design_match = bool(np.allclose(X_saved[:, 3:9], motion_regs, atol=1e-4))
                if not design_match:
                    notes.append("DESIGN MATRIX MOTION COLUMNS DO NOT MATCH SAVED MOTION OUTPUT")
    else:
        notes.append("cannot compute motion/signal association -- motion file or signal unavailable")

    # ============ 5. SPATIAL SMOOTHING VALIDATION ============
    fwhm_before = fwhm_after = None
    if moco_file and smooth_file and os.path.exists(moco_file) and os.path.exists(smooth_file):
        moco_tmean = nib.load(moco_file).get_fdata().mean(axis=3)
        smooth_tmean = nib.load(smooth_file).get_fdata().mean(axis=3)
        vs = in_zooms[:3]
        fwhm_before = spatial_fwhm_estimate(moco_tmean, brain_mask, vs)
        fwhm_after = spatial_fwhm_estimate(smooth_tmean, brain_mask, vs)
        if fwhm_before and fwhm_after:
            notes.append(f"estimated FWHM before={fwhm_before:.2f}mm after={fwhm_after:.2f}mm "
                         f"(requested 6mm; independently ESTIMATED via Forman et al. spatial-derivative "
                         f"method, not assumed to equal 6mm exactly)")
    else:
        notes.append("pre/post-smoothing intermediates unavailable -- FWHM not estimated")

    # ============ assemble row ============
    row.update({
        "mean_FD": round(mean_fd, 4) if mean_fd is not None else "",
        "max_FD": round(max_fd, 4) if max_fd is not None else "",
        "max_translation_mm": round(max_trans, 4) if max_trans is not None else "",
        "mean_translation_mm": round(mean_trans, 4) if mean_trans is not None else "",
        "max_rotation_deg": round(max_rot, 4) if max_rot is not None else "",
        "high_motion_percent": round(high_motion_pct, 2) if high_motion_pct is not None else "",
        "mean_tSNR": round(mean_tsnr, 2) if mean_tsnr is not None else "",
        "median_tSNR": round(median_tsnr, 2) if median_tsnr is not None else "",
        "tSNR_p25": round(tsnr_p25, 2) if tsnr_p25 is not None else "",
        "tSNR_p75": round(tsnr_p75, 2) if tsnr_p75 is not None else "",
        "temporal_SD_raw": round(temporal_sd_raw, 2) if temporal_sd_raw is not None else "",
        "temporal_SD_final": round(temporal_sd_final, 2) if temporal_sd_final is not None else "",
        "low_frequency_power_before": round(lf_before, 4) if lf_before is not None else "",
        "low_frequency_power_after": round(lf_after, 4) if lf_after is not None else "",
        "motion_association_before": round(motion_assoc_before, 4) if motion_assoc_before is not None else "",
        "motion_association_after": round(motion_assoc_after, 4) if motion_assoc_after is not None else "",
        "estimated_FWHM_before_mm": round(fwhm_before, 3) if fwhm_before is not None else "",
        "estimated_FWHM_after_mm": round(fwhm_after, 3) if fwhm_after is not None else "",
        "NaN_count": nan_count,
        "Inf_count": inf_count,
        "shape_preserved": shape_preserved,
        "voxel_size_preserved": voxel_preserved,
        "TR_preserved": tr_preserved,
        "volume_count_preserved": volcount_preserved,
        "spatial_coverage_status": coverage_status,
    })
    if design_match is not None:
        row["notes_design_match"] = design_match

    # ---- overall QC determination ----
    fail_reasons = []
    if nan_count > 0 or inf_count > 0:
        fail_reasons.append("NaN/Inf present")
    if empty_frames:
        fail_reasons.append(f"{len(empty_frames)} empty frames")
    if not shape_preserved or not voxel_preserved or not tr_preserved or not volcount_preserved:
        fail_reasons.append("geometry/TR/volume-count not preserved")
    if design_match is False:
        fail_reasons.append("design matrix motion columns do not match saved motion output")

    warn_reasons = []
    if coverage_status.startswith("CHECK"):
        warn_reasons.append("spatial coverage difference beyond single boundary slice")
    if mean_fd is not None and mean_fd > FD_THRESHOLD_MM:
        warn_reasons.append(f"mean FD {mean_fd:.3f}mm exceeds {FD_THRESHOLD_MM}mm reference threshold")
    if spikes_excl_first:
        warn_reasons.append(f"raw global-signal spike(s) at NON-first volumes: {spikes_excl_first}")
    if motion_assoc_after is not None and motion_assoc_before is not None and motion_assoc_after > motion_assoc_before:
        warn_reasons.append("motion association DID NOT decrease after regression")

    if fail_reasons:
        overall = "FAIL"
    elif warn_reasons:
        overall = "WARN"
    else:
        overall = "PASS"
    row["overall_QC"] = overall
    row["notes"] = "; ".join(notes + [f"FAIL:{r}" for r in fail_reasons] + [f"WARN:{r}" for r in warn_reasons])

    fig_data = {
        "sub": sub, "ses": ses, "fd": fd, "in_tsnr_map": in_tsnr_map, "final_tsnr_map": final_tsnr_map,
        "mid_slice": in_shape[2] // 2, "psd": psd_data, "mean_fd": mean_fd,
        "fwhm_before": fwhm_before, "fwhm_after": fwhm_after,
        "roughness_note": None,
    }
    return row, fig_data


def make_motion_fig(fd, sub, ses):
    if fd is None:
        return
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(fd)
    ax.axhline(FD_THRESHOLD_MM, color="r", linestyle="--", label=f"{FD_THRESHOLD_MM}mm reference")
    ax.set_xlabel("frame (t)")
    ax.set_ylabel("FD (mm)")
    ax.set_title(f"{sub} {ses} framewise displacement")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG["motion"], f"{sub}_{ses}_FD.png"), dpi=90)
    plt.close(fig)


def make_tsnr_fig(fd_data):
    in_map, final_map, mid = fd_data["in_tsnr_map"], fd_data["final_tsnr_map"], fd_data["mid_slice"]
    fig, axes = plt.subplots(1, 2 if final_map is not None else 1, figsize=(10, 4))
    if final_map is not None:
        im0 = axes[0].imshow(np.rot90(np.clip(in_map[:, :, mid], 0, 200)), cmap="viridis")
        axes[0].set_title("raw tSNR")
        axes[0].axis("off")
        im1 = axes[1].imshow(np.rot90(np.clip(final_map[:, :, mid], 0, 200)), cmap="viridis")
        axes[1].set_title("after moco+smooth tSNR")
        axes[1].axis("off")
    else:
        axes.imshow(np.rot90(np.clip(in_map[:, :, mid], 0, 200)), cmap="viridis")
        axes.set_title("raw tSNR")
        axes.axis("off")
    fig.suptitle(f"{fd_data['sub']} {fd_data['ses']}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG["tsnr"], f"{fd_data['sub']}_{fd_data['ses']}_tSNR.png"), dpi=90)
    plt.close(fig)


def make_psd_fig(fd_data):
    if fd_data["psd"] is None:
        return
    freqs_b, power_b, freqs_a, power_a = fd_data["psd"]
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.semilogy(freqs_b, power_b, label="raw", alpha=0.7)
    ax.semilogy(freqs_a, power_a, label="final (residual)", alpha=0.7)
    ax.axvline(LOWFREQ_CUTOFF_HZ, color="gray", linestyle="--", label=f"{LOWFREQ_CUTOFF_HZ}Hz cutoff")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power")
    ax.set_title(f"{fd_data['sub']} {fd_data['ses']} global-signal power spectrum")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG["temporal"], f"{fd_data['sub']}_{fd_data['ses']}_PSD.png"), dpi=90)
    plt.close(fig)


def make_smoothing_fig(fd_data):
    fb, fa = fd_data["fwhm_before"], fd_data["fwhm_after"]
    if fb is None or fa is None:
        return
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.bar(["before", "after"], [fb, fa], color=["gray", "steelblue"])
    ax.axhline(6.0, color="r", linestyle="--", label="requested 6mm")
    ax.set_ylabel("estimated FWHM (mm)")
    ax.set_title(f"{fd_data['sub']} {fd_data['ses']}")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG["smoothing"], f"{fd_data['sub']}_{fd_data['ses']}_FWHM.png"), dpi=90)
    plt.close(fig)


def make_summary_figs(rows):
    import numpy as np

    def col(name):
        return np.array([float(r[name]) for r in rows if r.get(name) not in ("", None)])

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    data_specs = [
        ("mean_FD", "Mean FD (mm)"), ("max_translation_mm", "Max translation (mm)"),
        ("mean_tSNR", "Mean tSNR (raw)"), ("temporal_SD_final", "Temporal SD (final)"),
        ("estimated_FWHM_after_mm", "Estimated FWHM after (mm)"), ("motion_association_before", "Motion R2 before"),
    ]
    for ax, (field, label) in zip(axes.flat, data_specs):
        vals = col(field)
        if vals.size:
            ax.hist(vals, bins=12, color="steelblue", edgecolor="black")
        ax.set_title(label)
    fig.suptitle("Cross-run distributions (n=25 AD acquisitions)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG["summary"], "cross_run_distributions.png"), dpi=100)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ma_before = col("motion_association_before")
    ma_after = col("motion_association_after")
    n = min(len(ma_before), len(ma_after))
    ax.scatter(ma_before[:n], ma_after[:n])
    lims = [0, max(ma_before.max() if ma_before.size else 1, ma_after.max() if ma_after.size else 1)]
    ax.plot(lims, lims, "r--", label="no change")
    ax.set_xlabel("Motion association R2 BEFORE regression (raw)")
    ax.set_ylabel("Motion association R2 AFTER regression (final)")
    ax.set_title("Motion-signal association: before vs after")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG["summary"], "motion_association_before_after.png"), dpi=100)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    fb = col("estimated_FWHM_before_mm")
    fa = col("estimated_FWHM_after_mm")
    n = min(len(fb), len(fa))
    ax.scatter(fb[:n], fa[:n])
    ax.axhline(6.0, color="r", linestyle="--", label="requested 6mm")
    ax.set_xlabel("Estimated FWHM BEFORE smoothing (mm)")
    ax.set_ylabel("Estimated FWHM AFTER smoothing (mm)")
    ax.set_title("Spatial smoothness: before vs after mri_fwhm")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIG["summary"], "smoothness_before_after.png"), dpi=100)
    plt.close(fig)


def main():
    mapping = list(csv.DictReader(open(MAPPING_CSV, newline="", encoding="utf-8")))
    runs = [(m["participant_id"], m["session"]) for m in mapping]
    print(f"Quantitative QC for {len(runs)} AD acquisitions")

    all_rows = []
    fig_data_list = []
    for i, (sub, ses) in enumerate(runs, 1):
        row, fd_data = validate_run(sub, ses, "run-01", i, len(runs))
        all_rows.append(row)
        if fd_data:
            fig_data_list.append(fd_data)
            try:
                make_motion_fig(fd_data["fd"], sub, ses)
                make_tsnr_fig(fd_data)
                make_smoothing_fig(fd_data)
            except Exception as e:
                row["notes"] = (row.get("notes", "") + f"; per-run figure error: {e}").strip("; ")

    # representative runs for PSD: lowest / median / highest mean_FD, plus 2 more spread out
    fd_sorted = sorted([r for r in all_rows if r.get("mean_FD") not in ("", None)], key=lambda r: r["mean_FD"])
    rep_keys = set()
    if fd_sorted:
        picks = [0, len(fd_sorted) // 4, len(fd_sorted) // 2, 3 * len(fd_sorted) // 4, -1]
        for p in picks:
            r = fd_sorted[p]
            rep_keys.add((r["participant_id"], r["session"]))
    for fd_data in fig_data_list:
        if (fd_data["sub"], fd_data["ses"]) in rep_keys:
            try:
                make_psd_fig(fd_data)
            except Exception:
                pass

    try:
        make_summary_figs(all_rows)
    except Exception as e:
        print("summary figure error:", e)

    def numeric(field):
        return [float(r[field]) for r in all_rows if r.get(field) not in ("", None)]

    def outliers(field, label):
        vals = numeric(field)
        if len(vals) < 4:
            return []
        arr = np.array(vals)
        q1, q3 = np.percentile(arr, [25, 75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        out = []
        for r in all_rows:
            v = r.get(field)
            if v in ("", None):
                continue
            v = float(v)
            if v < lo or v > hi:
                out.append((r["participant_id"], r["session"], v))
        return out

    motion_outliers = outliers("mean_FD", "mean FD")
    tsnr_outliers = outliers("mean_tSNR", "mean tSNR")
    tempsd_outliers = outliers("temporal_SD_final", "temporal SD")
    fwhm_outliers = outliers("estimated_FWHM_after_mm", "FWHM after")

    # Cross-reference outlier detection back into each row's status/notes --
    # required explicitly (item 7): outliers must be marked WARN with an
    # explanation, not just listed separately and disconnected from the
    # per-run overall_QC.
    outlier_map = {}
    for s, se, v in motion_outliers:
        outlier_map.setdefault((s, se), []).append(f"statistical outlier: mean FD={v:.3f}mm (IQR-based, vs this dataset's own distribution)")
    for s, se, v in tsnr_outliers:
        outlier_map.setdefault((s, se), []).append(f"statistical outlier: mean tSNR={v:.2f} (IQR-based)")
    for s, se, v in tempsd_outliers:
        outlier_map.setdefault((s, se), []).append(f"statistical outlier: temporal SD={v:.2f} (IQR-based)")
    for s, se, v in fwhm_outliers:
        outlier_map.setdefault((s, se), []).append(f"statistical outlier: FWHM after={v:.2f}mm (IQR-based)")

    for r in all_rows:
        key = (r["participant_id"], r["session"])
        if key in outlier_map:
            for reason in outlier_map[key]:
                r["notes"] = (r.get("notes", "") + f"; WARN:{reason}").strip("; ")
            if r["overall_QC"] == "PASS":
                r["overall_QC"] = "WARN"

    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for r in all_rows:
            for fn in CSV_FIELDS:
                r.setdefault(fn, "")
            writer.writerow(r)

    n_pass = sum(1 for r in all_rows if r["overall_QC"] == "PASS")
    n_warn = sum(1 for r in all_rows if r["overall_QC"] == "WARN")
    n_fail = sum(1 for r in all_rows if r["overall_QC"] == "FAIL")

    summary = {
        "total_acquisitions": len(all_rows),
        "PASS": n_pass, "WARN": n_warn, "FAIL": n_fail,
        "FD_threshold_mm_reference_only": FD_THRESHOLD_MM,
        "low_frequency_cutoff_hz": LOWFREQ_CUTOFF_HZ,
        "motion_outliers_IQR": motion_outliers,
        "tsnr_outliers_IQR": tsnr_outliers,
        "temporal_sd_outliers_IQR": tempsd_outliers,
        "fwhm_outliers_IQR": fwhm_outliers,
        "runs": all_rows,
    }
    with open(JSON_OUT, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    with open(TXT_OUT, "w", encoding="utf-8") as f:
        f.write("=== AD PREPROCESSING QUANTITATIVE QC SUMMARY ===\n\n")
        f.write(f"Total acquisitions: {len(all_rows)}\n")
        f.write(f"PASS: {n_pass}  WARN: {n_warn}  FAIL: {n_fail}\n\n")
        f.write(f"FD threshold used for reporting (NOT for exclusion): {FD_THRESHOLD_MM} mm "
                f"(Power et al. 2012 commonly-cited adult resting-state value)\n")
        f.write(f"Low-frequency power cutoff: {LOWFREQ_CUTOFF_HZ} Hz\n\n")

        f.write("--- Motion outliers (IQR-based, computed across this dataset's own distribution) ---\n")
        for s, se, v in motion_outliers:
            f.write(f"  {s} {se}: mean FD = {v:.3f} mm\n")
        if not motion_outliers:
            f.write("  None\n")
        f.write("\n--- tSNR outliers (IQR-based) ---\n")
        for s, se, v in tsnr_outliers:
            f.write(f"  {s} {se}: mean tSNR = {v:.2f}\n")
        if not tsnr_outliers:
            f.write("  None\n")
        f.write("\n--- Temporal SD outliers (IQR-based) ---\n")
        for s, se, v in tempsd_outliers:
            f.write(f"  {s} {se}: temporal SD (final) = {v:.2f}\n")
        if not tempsd_outliers:
            f.write("  None\n")
        f.write("\n--- Spatial smoothness (FWHM after) outliers (IQR-based) ---\n")
        for s, se, v in fwhm_outliers:
            f.write(f"  {s} {se}: estimated FWHM after = {v:.2f} mm\n")
        if not fwhm_outliers:
            f.write("  None\n")
        f.write("\n")

        f.write("--- Per-run overall status ---\n")
        for r in all_rows:
            f.write(f"  {r['participant_id']} {r['session']}: {r['overall_QC']}"
                    + (f" -- {r['notes']}" if r["overall_QC"] != "PASS" else "") + "\n")

    print(f"\nCSV: {CSV_OUT}\nTXT: {TXT_OUT}\nJSON: {JSON_OUT}")
    print(f"PASS={n_pass} WARN={n_warn} FAIL={n_fail}")


if __name__ == "__main__":
    main()
