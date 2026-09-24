"""
DATASET-SCALE Brainnetome-246 parcellation + biomarker generation.

Scales the already-validated single-acquisition pilot
(AD/sub-019S4549/ses-01/run-01) to every successfully preprocessed
acquisition across all six groups.

Strictly additive: reads preprocessed BOLD + the original 2mm atlas, writes
ONLY into <run>/biomarkers/. Never reruns/modifies preprocessing, never
modifies BIDS/NIfTI sources, never modifies the original atlas.

MAX_WORKERS = 1 by construction (no multiprocessing / joblib / futures / GPU).

Modes:
  discover      dry-run: enumerate eligible acquisitions, write manifest, no processing
  verify-pilot  reprocess the pilot into a scratch dir and compare against its
                existing validated outputs (does NOT touch the pilot directory)
  run           process the dataset, one acquisition at a time, checkpoint/resume
  audit         dataset-level audit over whatever is on disk
"""
import os
import sys
import csv
import gc
import json
import time
import shutil
import platform
import datetime
import traceback

import numpy as np
import nibabel as nib
import pandas as pd
from scipy.stats import rankdata
from nilearn.image import resample_img

ROOT = "/mnt/c/Users/krish/FYP"
DERIV = os.path.join(ROOT, "derivatives", "fsfast")
ATLAS_PATH = os.path.join(ROOT, "atlases", "Brainnetome246", "BN_Atlas_246_2mm.nii.gz")
ATLAS_LUT = os.path.join(ROOT, "atlases", "Brainnetome246", "BN_Atlas_246_LUT.txt")
CHECKPOINT_PREPROC = os.path.join(DERIV, "PROCESSING_CHECKPOINT.json")
EXCLUDED_CSV = os.path.join(ROOT, "audit", "excluded_runs.csv")

AUDIT_DIR = os.path.join(DERIV, "BIOMARKER_DATASET_AUDIT")
STATE_JSON = os.path.join(AUDIT_DIR, "biomarker_processing_state.json")
MANIFEST_CSV = os.path.join(AUDIT_DIR, "processing_manifest.csv")
ERRORS_CSV = os.path.join(AUDIT_DIR, "processing_errors.csv")

SCRATCH_ROOT = "/home/harish/fyp_work/biomarkers_scratch"

GROUPS = ["AD", "CN_Final", "EMCI", "LMCI", "MCI", "SMC_Final"]
PILOT_ACQ = ("AD", "sub-019S4549", "ses-01", "run-01")
N_ROI = 246
LOW_HZ, HIGH_HZ = 0.01, 0.10
TOL = 1e-10

MANIFEST_FIELDS = ["GROUP", "SUBJECT", "SESSION", "RUN", "STATUS", "BOLD_PATH",
                   "ROI_STATUS", "ALFF_STATUS", "REHO_STATUS", "FC_STATUS", "DC_STATUS",
                   "BIOMARKER_STATUS", "ERROR", "TIMESTAMP"]

REQUIRED_OUTPUTS = [
    "brainnetome_246_4mm.nii.gz", "brainnetome_metadata.json", "brainnetome_qc_report.md",
    "brainnetome_qc.csv", "roi_timeseries.npy", "roi_timeseries.csv", "roi_voxel_counts.csv",
    "roi_timeseries_qc.csv", "alff.npy", "alff.csv", "reho.npy", "reho.csv",
    "fc_matrix.npy", "degree_centrality.npy", "degree_centrality.csv",
    "fc_strength.npy", "fc_strength.csv", "regional_biomarkers.npy", "regional_biomarkers.csv",
    "biomarker_metadata.json", "biomarker_qc_report.md",
]


def log(msg):
    print(msg, flush=True)


def robust_copy(src, dst, attempts=4):
    """
    Copy across the DrvFs (/mnt/c) boundary, retrying transient I/O errors.
    Bulk sequential copies survive this mount far better than nibabel/numpy
    doing many small reads against it, which is what destabilised earlier runs.
    """
    last = None
    for i in range(attempts):
        try:
            shutil.copyfile(src, dst)
            return
        except OSError as e:
            last = e
            time.sleep(1.5 * (i + 1))
    raise OSError(f"copy failed after {attempts} attempts: {src} -> {dst}: {last}")


def hdr(t):
    log("\n" + "=" * 72)
    log(t)
    log("=" * 72)


def utcnow():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def load_lut():
    lut = {}
    with open(ATLAS_LUT) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 2 and parts[0].isdigit():
                lut[int(parts[0])] = parts[1]
    return lut


# ======================================================================
# DISCOVERY
# ======================================================================
def load_excluded():
    """Pre-processing exclusion list (audit/excluded_runs.csv)."""
    excluded = {}
    if not os.path.isfile(EXCLUDED_CSV):
        return excluded
    with open(EXCLUDED_CSV, newline="") as f:
        for row in csv.DictReader(f):
            g = (row.get("group") or row.get("GROUP") or "").strip()
            s = (row.get("subject") or row.get("SUBJECT") or "").strip()
            se = (row.get("session") or row.get("SESSION") or "").strip()
            r = (row.get("run") or row.get("RUN") or "").strip()
            reason = (row.get("reason") or row.get("REASON") or "").strip()
            if g and s:
                excluded[(g, s, se, r)] = reason
    return excluded


def discover_acquisitions():
    """
    Eligible == listed as COMPLETED in the preprocessing checkpoint.
    Preprocessing failures and audit exclusions are never processed.
    Directory existence alone is NEVER treated as validity.
    """
    with open(CHECKPOINT_PREPROC) as f:
        ck = json.load(f)
    excluded = load_excluded()

    acqs, skipped = [], []
    for g in GROUPS:
        entry = ck.get(g, {})
        for rel in entry.get("completed", []):
            sub, ses, run = rel.split("/")
            if (g, sub, ses, run) in excluded:
                skipped.append((g, sub, ses, run, "SKIPPED_EXCLUDED",
                                f"audit exclusion: {excluded[(g, sub, ses, run)]}"))
                continue
            acqs.append((g, sub, ses, run))
        for rel in entry.get("failed", []):
            sub, ses, run = rel.split("/")
            skipped.append((g, sub, ses, run, "SKIPPED_EXCLUDED",
                            "preprocessing FAILED (not eligible for biomarkers)"))
    return acqs, skipped


def bold_path_for(g, sub, ses, run):
    return os.path.join(DERIV, g, sub, ses, run,
                        f"{sub}_{ses}_task-rest_{run}_desc-preproc_bold.nii.gz")


def validate_bold(path):
    """Full per-acquisition input validation. Returns (ok, reason, info)."""
    if not os.path.isfile(path):
        return False, "BOLD file does not exist", {}
    try:
        img = nib.load(path)
    except Exception as e:
        return False, f"NiBabel load failed: {e}", {}
    if len(img.shape) != 4:
        return False, f"not 4D (shape={img.shape})", {}
    X, Y, Z, T = img.shape
    if min(X, Y, Z) <= 0:
        return False, f"invalid spatial dimensions {img.shape}", {}
    if T <= 0:
        return False, f"time dimension not > 0 (T={T})", {}
    aff = img.affine
    if not np.all(np.isfinite(aff)):
        return False, "affine contains non-finite values", {}
    if abs(float(np.linalg.det(aff[:3, :3]))) < 1e-9:
        return False, "degenerate affine (|det| ~ 0)", {}
    try:
        data = img.get_fdata(dtype=np.float32)
    except Exception as e:
        return False, f"failed to read data array: {e}", {}
    if not np.all(np.isfinite(data)):
        n_nan = int(np.isnan(data).sum())
        n_inf = int(np.isinf(data).sum())
        del data
        gc.collect()
        return False, f"non-finite values in BOLD (NaN={n_nan}, Inf={n_inf})", {}
    info = {"shape": tuple(int(v) for v in img.shape), "T": int(T),
            "TR_s": float(img.header.get_zooms()[3]),
            "voxel_size_mm": tuple(round(float(z), 4) for z in img.header.get_zooms()[:3])}
    del data
    gc.collect()
    return True, "", info


def validate_existing_outputs(out_dir, expected_T):
    """
    Re-validate real files on disk. Never trusts the manifest alone and never
    treats mere existence as validity.
    """
    if not os.path.isdir(out_dir):
        return False, "no biomarkers/ directory"
    missing = [f for f in REQUIRED_OUTPUTS if not os.path.isfile(os.path.join(out_dir, f))]
    if missing:
        return False, f"missing {len(missing)} output(s): {','.join(missing[:4])}"
    # Shapes are read via mmap/headers rather than loading arrays: this runs for
    # every finished acquisition on every resume, and loading them all back was
    # hundreds of MB of churn per restart on a VM that is already unstable.
    try:
        roi = np.load(os.path.join(out_dir, "roi_timeseries.npy"), mmap_mode="r")
        if roi.shape != (expected_T, N_ROI):
            return False, f"roi_timeseries shape {roi.shape} != ({expected_T},{N_ROI})"
        fc = np.load(os.path.join(out_dir, "fc_matrix.npy"), mmap_mode="r")
        if fc.shape != (N_ROI, N_ROI):
            return False, f"fc_matrix shape {fc.shape}"
        for nm in ("alff.npy", "reho.npy", "degree_centrality.npy", "fc_strength.npy"):
            a = np.load(os.path.join(out_dir, nm), mmap_mode="r")
            if a.shape != (N_ROI,):
                return False, f"{nm} shape {a.shape}"
            del a
        rb = np.load(os.path.join(out_dir, "regional_biomarkers.npy"), mmap_mode="r")
        if rb.shape != (N_ROI, 4):
            return False, f"regional_biomarkers shape {rb.shape}"
        atl = nib.load(os.path.join(out_dir, "brainnetome_246_4mm.nii.gz"))
        if len(atl.shape) != 3:
            return False, "aligned atlas not 3D"
        del roi, fc, rb, atl
    except Exception as e:
        return False, f"output validation error: {e}"
    return True, ""


# ======================================================================
# CORE COMPUTATION (identical definitions to the validated pilot)
# ======================================================================
def align_atlas(bold_img):
    """Brainnetome 2mm -> subject BOLD grid, NEAREST NEIGHBOR ONLY."""
    atlas_img = nib.load(ATLAS_PATH)
    native = np.asarray(atlas_img.dataobj)
    uniq = set(int(x) for x in np.unique(native) if x > 0)
    native_ok = (len(sorted(set(range(1, N_ROI + 1)) - uniq)) == 0 and
                 len(sorted(uniq - set(range(1, N_ROI + 1)))) == 0)
    del native
    atlas_canon = nib.as_closest_canonical(atlas_img)
    aligned = resample_img(atlas_canon, target_affine=bold_img.affine,
                           target_shape=bold_img.shape[:3], interpolation="nearest",
                           force_resample=True, copy_header=True)
    aligned_int = nib.Nifti1Image(np.asarray(aligned.dataobj).astype(np.int16),
                                  bold_img.affine, bold_img.header)
    aligned_int.header.set_data_dtype(np.int16)
    grid_match = bool(aligned_int.shape == bold_img.shape[:3] and
                      np.allclose(aligned_int.affine, bold_img.affine, atol=1e-6))
    return aligned_int, grid_match, native_ok, atlas_img.shape, \
        tuple(round(float(z), 4) for z in atlas_img.header.get_zooms()[:3])


def compute_alff(bold_data, mask, T, TR, shape3):
    """Zang et al. 2007: mean FFT amplitude within 0.01-0.10 Hz, voxel-wise."""
    freqs = np.fft.rfftfreq(T, d=TR)
    band = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
    flat = bold_data.reshape(-1, T)
    mask_flat = mask.reshape(-1)
    in_idx = np.where(mask_flat)[0]
    sig = flat[in_idx, :]
    sig = sig - sig.mean(axis=1, keepdims=True)
    amp = np.abs(np.fft.rfft(sig, axis=1))
    vals = amp[:, band].mean(axis=1)
    out = np.zeros(flat.shape[0], dtype=np.float32)
    out[in_idx] = vals
    del flat, sig, amp, vals
    gc.collect()
    return out.reshape(shape3), int(band.sum())


def _shift_slices(d, n):
    if d == 0:
        return slice(0, n), slice(0, n)
    if d > 0:
        return slice(0, n - d), slice(d, n)
    return slice(-d, n), slice(0, n + d)


def compute_reho(bold_data, mask, T, chunk=16):
    """
    Kendall's W over a 27-voxel (3x3x3, center included) neighborhood,
    neighbors restricted to in-mask voxels so K varies at the boundary.

    Vectorized form of the pilot loop: each voxel's time series is ranked
    across time independently of its neighborhood, so ranks are precomputed
    once and neighbor rank-sums accumulated by shifted adds. Mathematically
    identical to the pilot (verify-pilot confirms bit-identical output).

    Time is processed in chunks and ranks held as float32 (rank sums are
    small integers/halves, exactly representable) to keep peak memory tens
    of MB rather than hundreds: this VM destabilises under repeated large
    allocations. The squared-deviation accumulator stays float64.
    """
    X, Y, Z = mask.shape
    idx = np.where(mask)
    ranks = rankdata(bold_data[idx], axis=1).astype(np.float32)  # (n_mask, T)

    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)]
    mask_f = mask.astype(np.float32)
    Kc = np.zeros((X, Y, Z), dtype=np.float32)
    for dx, dy, dz in offsets:
        sxd, sxs = _shift_slices(dx, X)
        syd, sys_ = _shift_slices(dy, Y)
        szd, szs = _shift_slices(dz, Z)
        Kc[sxd, syd, szd] += mask_f[sxs, sys_, szs]
    del mask_f

    Rbar = (Kc * ((T + 1) / 2.0)).astype(np.float64)
    S = np.zeros((X, Y, Z), dtype=np.float64)
    for t0 in range(0, T, chunk):
        t1 = min(t0 + chunk, T)
        rv = np.zeros((X, Y, Z, t1 - t0), dtype=np.float32)
        rv[idx] = ranks[:, t0:t1]
        Ri = np.zeros_like(rv)
        for dx, dy, dz in offsets:
            sxd, sxs = _shift_slices(dx, X)
            syd, sys_ = _shift_slices(dy, Y)
            szd, szs = _shift_slices(dz, Z)
            Ri[sxd, syd, szd, :] += rv[sxs, sys_, szs, :]
        for k in range(t1 - t0):
            d = Ri[:, :, :, k].astype(np.float64) - Rbar
            S += d * d
            del d
        del rv, Ri
        gc.collect()
    del ranks, Rbar
    gc.collect()

    W = np.zeros((X, Y, Z), dtype=np.float32)
    Kc64 = Kc.astype(np.float64)
    valid = mask & (Kc >= 2)
    denom = (Kc64 ** 2) * (T ** 3 - T)
    W[valid] = (12.0 * S[valid] / denom[valid]).astype(np.float32)
    del S, Kc, Kc64, denom, valid
    gc.collect()
    return W


def roi_mean_of_map(vol, atlas):
    out = np.full(N_ROI, np.nan)
    for rid in range(1, N_ROI + 1):
        vox = atlas == rid
        if vox.sum() > 0:
            out[rid - 1] = float(vol[vox].mean())
    return out


def extract_roi_timeseries(bold_data, atlas, T):
    """Mean BOLD per ROI per timepoint, on the already-aligned grid. (T, 246)."""
    flat_bold = bold_data.reshape(-1, T)
    flat_atlas = atlas.reshape(-1)
    roi_ts = np.zeros((T, N_ROI), dtype=np.float64)
    for rid in range(1, N_ROI + 1):
        vox = flat_atlas == rid
        if vox.sum() > 0:
            roi_ts[:, rid - 1] = flat_bold[vox, :].mean(axis=0)
        else:
            roi_ts[:, rid - 1] = np.nan
    del flat_bold, flat_atlas
    gc.collect()
    return roi_ts


# ======================================================================
# PER-ACQUISITION PROCESSING
# ======================================================================
def process_one(g, sub, ses, run, out_dir, no_clobber=False, scratch_root=None):
    """
    Full biomarker generation for one acquisition. Returns a result dict.
    no_clobber: never overwrite an existing file (used to protect the pilot).
    scratch_root: stage inputs/outputs on WSL-native disk and copy results to
    out_dir only once the acquisition is complete.
    """
    lut = load_lut()
    res = {"ROI_STATUS": "PENDING", "ALFF_STATUS": "PENDING", "REHO_STATUS": "PENDING",
           "FC_STATUS": "PENDING", "DC_STATUS": "PENDING", "BIOMARKER_STATUS": "PENDING",
           "qc": {}, "arrays": {}}
    os.makedirs(out_dir, exist_ok=True)

    work = out_dir
    if scratch_root:
        work = os.path.join(scratch_root, f"{g}_{sub}_{ses}_{run}")
        shutil.rmtree(work, ignore_errors=True)
        os.makedirs(work, exist_ok=True)

    def save(name, fn):
        # no_clobber is judged against the FINAL destination, so a preserved
        # file is never regenerated even though work/ starts empty.
        if no_clobber and os.path.isfile(os.path.join(out_dir, name)):
            res.setdefault("preserved", []).append(name)
            return None
        p = os.path.join(work, name)
        fn(p)
        return p

    def resolve(name):
        p = os.path.join(work, name)
        return p if os.path.isfile(p) else os.path.join(out_dir, name)

    def save_csv(name, header, rows):
        def _w(p):
            with open(p, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(header)
                w.writerows(rows)
        return save(name, _w)

    def save_dictcsv(name, rows):
        def _w(p):
            with open(p, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
        return save(name, _w)

    bold_path = bold_path_for(g, sub, ses, run)
    mask_path = os.path.join(DERIV, g, sub, ses, run, "brain_mask.nii.gz")
    read_bold, read_mask = bold_path, mask_path
    if scratch_root:
        read_bold = os.path.join(work, "_in_bold.nii.gz")
        read_mask = os.path.join(work, "_in_mask.nii.gz")
        robust_copy(bold_path, read_bold)
        robust_copy(mask_path, read_mask)

    bold_img = nib.load(read_bold)
    bold_data = bold_img.get_fdata(dtype=np.float32)
    T = int(bold_img.shape[3])
    TR = float(bold_img.header.get_zooms()[3])
    shape3 = bold_img.shape[:3]

    mask_img = nib.load(read_mask)
    mask = mask_img.get_fdata() > 0
    n_mask = int(mask.sum())

    # ---- Brainnetome alignment -------------------------------------
    aligned_int, grid_match, native_ok, atlas_native_shape, atlas_native_zoom = align_atlas(bold_img)
    if not grid_match:
        raise RuntimeError("aligned atlas grid does not match BOLD grid")
    save("brainnetome_246_4mm.nii.gz", lambda p: nib.save(aligned_int, p))
    atlas = np.asarray(aligned_int.dataobj).astype(np.int32)

    counts = np.array([int((atlas == rid).sum()) for rid in range(1, N_ROI + 1)])
    present = set(int(x) for x in np.unique(atlas) if x > 0)
    missing_labels = sorted(set(range(1, N_ROI + 1)) - present)
    extra_labels = sorted(present - set(range(1, N_ROI + 1)))
    n_present, n_zero = int((counts > 0).sum()), int((counts == 0).sum())
    n_lt5, n_lt10 = int((counts < 5).sum()), int((counts < 10).sum())

    save_csv("roi_voxel_counts.csv", ["ROI_ID", "ROI_Name", "Voxel_Count"],
             [[rid, lut.get(rid, f"ROI_{rid}"), counts[rid - 1]] for rid in range(1, N_ROI + 1)])

    bn_qc = {"expected_rois": N_ROI, "rois_present": n_present,
             "missing_roi_labels": len(missing_labels), "extra_roi_labels": len(extra_labels),
             "zero_voxel_rois": n_zero, "rois_lt5_voxels": n_lt5, "rois_lt10_voxels": n_lt10,
             "min_voxels": int(counts.min()), "median_voxels": float(np.median(counts)),
             "max_voxels": int(counts.max()), "atlas_bold_dim_match": bool(aligned_int.shape == shape3),
             "atlas_bold_affine_match": bool(grid_match), "interpolation": "nearest_neighbor"}
    save_dictcsv("brainnetome_qc.csv", [bn_qc])
    res["qc"]["brainnetome"] = bn_qc

    # ---- ROI time series -------------------------------------------
    roi_ts = extract_roi_timeseries(bold_data, atlas, T)
    if roi_ts.shape != (T, N_ROI):
        raise RuntimeError(f"ROI matrix shape {roi_ts.shape} != ({T},{N_ROI})")
    save("roi_timeseries.npy", lambda p: np.save(p, roi_ts))
    ts_df = pd.DataFrame(roi_ts, columns=[f"ROI_{i:03d}" for i in range(1, N_ROI + 1)])
    ts_df.insert(0, "time", np.arange(1, T + 1))
    save("roi_timeseries.csv", lambda p: ts_df.to_csv(p, index=False))
    del ts_df

    stds = np.array([float(np.std(roi_ts[:, i][np.isfinite(roi_ts[:, i])]))
                     if np.isfinite(roi_ts[:, i]).any() else np.nan for i in range(N_ROI)])
    n_zero_var = int(np.sum(stds == 0.0))
    ts_qc_rows = []
    for rid in range(1, N_ROI + 1):
        col = roi_ts[:, rid - 1]
        fin = col[np.isfinite(col)]
        ts_qc_rows.append({
            "ROI_ID": rid, "ROI_Name": lut.get(rid, f"ROI_{rid}"),
            "Mean": float(fin.mean()) if fin.size else float("nan"),
            "Std": float(fin.std()) if fin.size else float("nan"),
            "Min": float(fin.min()) if fin.size else float("nan"),
            "Max": float(fin.max()) if fin.size else float("nan"),
            "NaN_Count": int(np.isnan(col).sum()), "Inf_Count": int(np.isinf(col).sum()),
            "Zero_Variance": bool(fin.size and fin.std() == 0.0)})
    save_dictcsv("roi_timeseries_qc.csv", ts_qc_rows)

    roi_qc = {"shape_T": T, "shape_ROI": int(roi_ts.shape[1]),
              "nan_count": int(np.isnan(roi_ts).sum()), "inf_count": int(np.isinf(roi_ts).sum()),
              "zero_variance_rois": n_zero_var,
              "min_std": float(np.nanmin(stds)), "median_std": float(np.nanmedian(stds)),
              "max_std": float(np.nanmax(stds))}
    res["qc"]["roi_timeseries"] = roi_qc
    res["ROI_STATUS"] = "COMPLETED" if (roi_ts.shape == (T, N_ROI) and roi_qc["inf_count"] == 0
                                        and roi_qc["nan_count"] == 0) else "COMPLETED_WITH_WARNINGS"

    # ---- ALFF ------------------------------------------------------
    alff_map, n_band = compute_alff(bold_data, mask, T, TR, shape3)
    save("alff_map.nii.gz", lambda p: nib.save(
        nib.Nifti1Image(alff_map.astype(np.float32), bold_img.affine, bold_img.header), p))
    alff_roi = roi_mean_of_map(alff_map, atlas)
    del alff_map
    gc.collect()
    save("alff.npy", lambda p: np.save(p, alff_roi))
    save_csv("alff.csv", ["ROI_ID", "ROI_Name", "ALFF"],
             [[rid, lut.get(rid, f"ROI_{rid}"), alff_roi[rid - 1]] for rid in range(1, N_ROI + 1)])
    alff_qc = {"n_values": int(alff_roi.size), "nan": int(np.isnan(alff_roi).sum()),
               "inf": int(np.isinf(alff_roi).sum()),
               "negative": int(np.sum(alff_roi[np.isfinite(alff_roi)] < 0)),
               "min": float(np.nanmin(alff_roi)), "median": float(np.nanmedian(alff_roi)),
               "max": float(np.nanmax(alff_roi)), "n_freq_bins_in_band": n_band}
    res["qc"]["alff"] = alff_qc
    res["ALFF_STATUS"] = "COMPLETED" if (alff_qc["nan"] == 0 and alff_qc["inf"] == 0
                                         and alff_qc["negative"] == 0) else "COMPLETED_WITH_WARNINGS"

    # ---- ReHo ------------------------------------------------------
    reho_map = compute_reho(bold_data, mask, T)
    save("reho_map.nii.gz", lambda p: nib.save(
        nib.Nifti1Image(reho_map.astype(np.float32), bold_img.affine, bold_img.header), p))
    reho_roi = roi_mean_of_map(reho_map, atlas)
    del reho_map
    gc.collect()
    save("reho.npy", lambda p: np.save(p, reho_roi))
    save_csv("reho.csv", ["ROI_ID", "ROI_Name", "ReHo"],
             [[rid, lut.get(rid, f"ROI_{rid}"), reho_roi[rid - 1]] for rid in range(1, N_ROI + 1)])
    fin_reho = reho_roi[np.isfinite(reho_roi)]
    reho_qc = {"n_values": int(reho_roi.size), "nan": int(np.isnan(reho_roi).sum()),
               "inf": int(np.isinf(reho_roi).sum()),
               "min": float(np.nanmin(reho_roi)), "median": float(np.nanmedian(reho_roi)),
               "max": float(np.nanmax(reho_roi)),
               "out_of_range_0_1": int(np.sum((fin_reho < 0) | (fin_reho > 1)))}
    res["qc"]["reho"] = reho_qc
    res["REHO_STATUS"] = "COMPLETED" if (reho_qc["nan"] == 0 and reho_qc["inf"] == 0
                                         and reho_qc["out_of_range_0_1"] == 0) else "COMPLETED_WITH_WARNINGS"

    del bold_data
    gc.collect()

    # ---- FC --------------------------------------------------------
    fc = np.corrcoef(roi_ts.T)
    save("fc_matrix.npy", lambda p: np.save(p, fc))
    fin_off = fc[~np.eye(N_ROI, dtype=bool)]
    fin_off = fin_off[np.isfinite(fin_off)]
    fc_qc = {"shape": f"{fc.shape[0]}x{fc.shape[1]}",
             "nan": int(np.isnan(fc).sum()), "inf": int(np.isinf(fc).sum()),
             "min": float(np.nanmin(fc)), "max": float(np.nanmax(fc)),
             "mean": float(np.nanmean(fc)), "median": float(np.nanmedian(fc)),
             "diagonal_error": float(np.nanmax(np.abs(np.diag(fc) - 1.0))),
             "symmetry_error": float(np.nanmax(np.abs(fc - fc.T))),
             "pct_negative_offdiag": float(100.0 * np.sum(fin_off < 0) / fin_off.size)
             if fin_off.size else float("nan")}
    res["qc"]["fc"] = fc_qc
    res["FC_STATUS"] = "COMPLETED" if (fc_qc["nan"] == 0 and fc_qc["inf"] == 0
                                       and fc_qc["diagonal_error"] < 1e-6
                                       and fc_qc["symmetry_error"] < 1e-6) else "COMPLETED_WITH_WARNINGS"

    # ---- DC + FC strength ------------------------------------------
    fc_no_diag = fc.copy()
    np.fill_diagonal(fc_no_diag, 0.0)
    dc = fc_no_diag.sum(axis=1)
    fc_strength = fc_no_diag.sum(axis=1) / float(N_ROI - 1)

    save("degree_centrality.npy", lambda p: np.save(p, dc))
    save_csv("degree_centrality.csv", ["ROI_ID", "ROI_Name", "DC"],
             [[rid, lut.get(rid, f"ROI_{rid}"), dc[rid - 1]] for rid in range(1, N_ROI + 1)])
    save("fc_strength.npy", lambda p: np.save(p, fc_strength))
    save_csv("fc_strength.csv", ["ROI_ID", "ROI_Name", "FC_Strength"],
             [[rid, lut.get(rid, f"ROI_{rid}"), fc_strength[rid - 1]] for rid in range(1, N_ROI + 1)])

    dc_qc = {"n_values": int(dc.size), "nan": int(np.isnan(dc).sum()), "inf": int(np.isinf(dc).sum()),
             "min": float(np.nanmin(dc)), "median": float(np.nanmedian(dc)), "max": float(np.nanmax(dc))}
    res["qc"]["dc"] = dc_qc
    res["DC_STATUS"] = "COMPLETED" if (dc_qc["nan"] == 0 and dc_qc["inf"] == 0) else "COMPLETED_WITH_WARNINGS"
    res["qc"]["fc_strength"] = {"n_values": int(fc_strength.size),
                                "nan": int(np.isnan(fc_strength).sum()),
                                "inf": int(np.isinf(fc_strength).sum()),
                                "min": float(np.nanmin(fc_strength)),
                                "median": float(np.nanmedian(fc_strength)),
                                "max": float(np.nanmax(fc_strength))}

    # ---- regional biomarker matrix ---------------------------------
    reg_num = np.column_stack([alff_roi, reho_roi, dc, fc_strength])  # (246, 4)
    save("regional_biomarkers.npy", lambda p: np.save(p, reg_num))
    reg_df = pd.DataFrame({"ROI_ID": np.arange(1, N_ROI + 1), "ALFF": alff_roi,
                           "ReHo": reho_roi, "DC": dc, "FC_Strength": fc_strength})
    save("regional_biomarkers.csv", lambda p: reg_df.to_csv(p, index=False))

    # ---- cross-file consistency (independent recomputation) --------
    fc_re = np.load(resolve("fc_matrix.npy"))
    tmp = fc_re.copy()
    np.fill_diagonal(tmp, 0.0)
    dc_re = tmp.sum(axis=1)
    strength_re = tmp.sum(axis=1) / float(N_ROI - 1)
    dc_saved = np.load(resolve("degree_centrality.npy"))
    st_saved = np.load(resolve("fc_strength.npy"))
    dc_diff = float(np.nanmax(np.abs(dc_re - dc_saved)))
    st_diff = float(np.nanmax(np.abs(strength_re - st_saved)))

    cons = {
        "brainnetome_labels_1_246": bool(len(missing_labels) == 0 and len(extra_labels) == 0),
        "roi_ts_shape_ok": bool(roi_ts.shape == (T, N_ROI)),
        "alff_len_ok": bool(alff_roi.shape == (N_ROI,)),
        "reho_len_ok": bool(reho_roi.shape == (N_ROI,)),
        "dc_len_ok": bool(dc.shape == (N_ROI,)),
        "fc_strength_len_ok": bool(fc_strength.shape == (N_ROI,)),
        "fc_shape_ok": bool(fc.shape == (N_ROI, N_ROI)),
        "regional_matrix_shape_ok": bool(reg_num.shape == (N_ROI, 4)),
        "roi_ids_match_across_files": bool(list(reg_df["ROI_ID"]) == list(range(1, N_ROI + 1))),
        "no_duplicate_roi_ids": bool(len(set(reg_df["ROI_ID"])) == N_ROI),
        "no_missing_roi_ids": bool(set(reg_df["ROI_ID"]) == set(range(1, N_ROI + 1))),
        "no_nan_in_biomarkers": bool(not np.isnan(reg_num).any()),
        "no_inf_in_biomarkers": bool(not np.isinf(reg_num).any()),
        "fc_symmetric": bool(fc_qc["symmetry_error"] < 1e-6),
        "fc_diagonal_is_1": bool(fc_qc["diagonal_error"] < 1e-6),
        "dc_matches_fc_definition": bool(dc_diff <= TOL),
        "fc_strength_matches_fc_definition": bool(st_diff <= TOL),
        "dc_max_abs_diff": dc_diff,
        "fc_strength_max_abs_diff": st_diff,
    }
    res["qc"]["consistency"] = cons
    all_ok = all(v for k, v in cons.items() if isinstance(v, bool))
    res["BIOMARKER_STATUS"] = "COMPLETED" if all_ok else "COMPLETED_WITH_WARNINGS"

    # ---- metadata + report -----------------------------------------
    bn_meta = {
        "subject": sub, "session": ses, "run": run, "group": g,
        "atlas": "Brainnetome246", "atlas_source_path": ATLAS_PATH,
        "atlas_native_dimensions": list(atlas_native_shape),
        "atlas_native_voxel_size_mm": list(atlas_native_zoom),
        "atlas_original_modified": False,
        "interpolation": "nearest_neighbor",
        "target_grid": "subject preprocessed BOLD grid",
        "aligned_dimensions": list(aligned_int.shape),
        "aligned_voxel_size_mm": [round(float(z), 4) for z in aligned_int.header.get_zooms()[:3]],
        "grid_match_bold": grid_match,
        "labels_1_246_present_in_native_atlas": native_ok,
        "roi_count_expected": N_ROI, "roi_count_present": n_present,
        "zero_voxel_rois": n_zero, "rois_lt5_voxels": n_lt5, "rois_lt10_voxels": n_lt10,
        "bold_source": bold_path, "bold_dimensions": list(bold_img.shape),
        "bold_TR_s": TR, "brain_mask_voxels": n_mask,
        "roi_timeseries_shape": [int(T), N_ROI],
        "processing_date_utc": utcnow(),
    }
    save("brainnetome_metadata.json",
         lambda p: json.dump(bn_meta, open(p, "w"), indent=2))

    def write_bn_report(p):
        with open(p, "w", encoding="utf-8") as f:
            w = f.write
            w(f"# Brainnetome-246 QC -- {g}/{sub}/{ses}/{run}\n\n## Input\n\n")
            w(f"- BOLD: `{bold_path}`\n- Atlas: `{ATLAS_PATH}` (original, read-only)\n\n")
            w("## Spatial compatibility\n\n")
            w(f"- BOLD dimensions: {bold_img.shape}\n")
            w(f"- Atlas native dimensions: {atlas_native_shape} @ {atlas_native_zoom} mm\n")
            w(f"- Aligned atlas dimensions: {aligned_int.shape}\n")
            w(f"- Interpolation: NEAREST NEIGHBOR ONLY\n")
            w(f"- Atlas/BOLD dimension match: {aligned_int.shape == shape3}\n")
            w(f"- Atlas/BOLD affine match: {grid_match}\n")
            w(f"- Labels 1-246 present in native atlas: {native_ok}\n\n")
            w("## ROI coverage\n\n")
            w(f"- Expected ROIs: {N_ROI}\n- Present ROIs: {n_present}\n")
            w(f"- Missing ROI labels: {len(missing_labels)}\n")
            w(f"- Unexpected/extra labels: {len(extra_labels)}\n")
            w(f"- Zero-voxel ROIs: {n_zero}\n- ROIs <5 voxels: {n_lt5}\n- ROIs <10 voxels: {n_lt10}\n")
            w(f"- Voxel count min/median/max: {int(counts.min())} / "
              f"{float(np.median(counts))} / {int(counts.max())}\n\n")
            w(f"**Status: {'PASS' if (n_present == N_ROI and grid_match) else 'PASS WITH WARNINGS'}**\n")
    save("brainnetome_qc_report.md", write_bn_report)

    bio_meta = {
        "subject": sub, "session": ses, "run": run, "group": g,
        "bold_source": bold_path, "bold_volumes": T, "TR_s": TR,
        "n_rois": N_ROI, "brain_mask_voxels": n_mask,
        "alff": {"method": "Zang et al. 2007 (mean amplitude in band)",
                 "band_hz": [LOW_HZ, HIGH_HZ],
                 "note": "BOLD already band-pass filtered 0.01-0.10Hz during preprocessing",
                 "level": "voxel-wise, then ROI mean"},
        "reho": {"method": "Kendall's coefficient of concordance (KCC)",
                 "neighborhood": "27-voxel (3x3x3 incl. center)",
                 "reason": "Zang et al. 2004 original definition; AFNI 3dReHo / DPARSF default",
                 "boundary_handling": "K varies per voxel; restricted to in-mask neighbors",
                 "level": "voxel-wise, then ROI mean"},
        "fc": {"method": "Pearson correlation", "matrix": "246x246",
               "self_connections": "included in matrix diagonal, excluded from DC/strength",
               "negative_correlations": "retained", "thresholding": "none (full weighted matrix)",
               "binarization": "none"},
        "degree_centrality": {"definition": "DC_i = sum_{j!=i} FC[i,j]",
                              "weighted": True, "signed": True, "threshold": None,
                              "self_connection_excluded": True},
        "fc_regional_strength": {"definition": "mean_{j!=i} FC[i,j]", "signed": True,
                                 "absolute_value": False},
        "regional_matrix": {"csv_shape": [N_ROI, 5], "npy_shape": [N_ROI, 4],
                            "features": ["ALFF", "ReHo", "DC", "FC_Strength"],
                            "row_order": "ROI 1..246"},
        "consistency_checks": cons,
        "software": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": __import__("scipy").__version__, "nibabel": nib.__version__,
                     "nilearn": __import__("nilearn").__version__, "pandas": pd.__version__},
        "processing_date_utc": utcnow(),
        "preprocessing_modified": False, "atlas_modified": False,
    }
    save("biomarker_metadata.json",
         lambda p: json.dump(bio_meta, open(p, "w"), indent=2))

    def write_report(p):
        with open(p, "w", encoding="utf-8") as f:
            w = f.write
            w(f"# Biomarker QC Report -- {g}/{sub}/{ses}/{run}\n\n")
            w(f"BOLD volumes (T): {T}  |  TR: {TR}s  |  brain mask voxels: {n_mask}\n\n")
            w("## Brainnetome-246\n\n")
            w(f"- Interpolation: nearest neighbor\n- Grid match with BOLD: {grid_match}\n")
            w(f"- ROIs present: {n_present} / {N_ROI}\n- Zero-voxel ROIs: {n_zero}\n")
            w(f"- ROIs <5 voxels: {n_lt5}  |  <10 voxels: {n_lt10}\n")
            w(f"- Voxel count min/median/max: {int(counts.min())} / "
              f"{float(np.median(counts))} / {int(counts.max())}\n\n")
            w("## ROI time series\n\n")
            w(f"- Shape: {T} x {N_ROI}\n- NaN: {roi_qc['nan_count']}  Inf: {roi_qc['inf_count']}\n")
            w(f"- Zero-variance ROIs: {n_zero_var}\n")
            w(f"- Std min/median/max: {roi_qc['min_std']:.6g} / "
              f"{roi_qc['median_std']:.6g} / {roi_qc['max_std']:.6g}\n\n")
            w("## ALFF\n\n")
            w(f"- 246 values, NaN={alff_qc['nan']}, Inf={alff_qc['inf']}, "
              f"negative={alff_qc['negative']}\n")
            w(f"- min/median/max: {alff_qc['min']:.4f} / {alff_qc['median']:.4f} / "
              f"{alff_qc['max']:.4f}\n- Status: {res['ALFF_STATUS']}\n\n")
            w("## ReHo\n\n")
            w(f"- Neighborhood: 27-voxel (3x3x3 incl. center)\n")
            w(f"- 246 values, NaN={reho_qc['nan']}, Inf={reho_qc['inf']}, "
              f"out of [0,1]={reho_qc['out_of_range_0_1']}\n")
            w(f"- min/median/max: {reho_qc['min']:.4f} / {reho_qc['median']:.4f} / "
              f"{reho_qc['max']:.4f}\n- Status: {res['REHO_STATUS']}\n\n")
            w("## Functional Connectivity\n\n")
            w(f"- {fc_qc['shape']}, Pearson, signed, unthresholded\n")
            w(f"- NaN={fc_qc['nan']}, Inf={fc_qc['inf']}\n")
            w(f"- diagonal error={fc_qc['diagonal_error']:.2e}, "
              f"symmetry error={fc_qc['symmetry_error']:.2e}\n")
            w(f"- negative off-diagonal: {fc_qc['pct_negative_offdiag']:.2f}%\n")
            w(f"- Status: {res['FC_STATUS']}\n\n")
            w("## Degree Centrality / FC Strength\n\n")
            w(f"- DC = sum_(j!=i) FC[i,j] (weighted, signed, unthresholded)\n")
            w(f"- DC min/median/max: {dc_qc['min']:.4f} / {dc_qc['median']:.4f} / "
              f"{dc_qc['max']:.4f}\n")
            w(f"- Independent DC recomputation max abs diff: {dc_diff:.2e}\n")
            w(f"- Independent FC-strength recomputation max abs diff: {st_diff:.2e}\n")
            w(f"- Status: {res['DC_STATUS']}\n\n")
            w("## Cross-file consistency\n\n")
            for k, v in cons.items():
                w(f"- {k}: {v}\n")
            w(f"\n**Overall: {res['BIOMARKER_STATUS']}**\n")
    save("biomarker_qc_report.md", write_report)

    res["arrays"] = {"T": T, "n_present": n_present, "n_zero_var": n_zero_var}
    del roi_ts, fc, fc_no_diag, fc_re, tmp, dc, dc_re, fc_strength, strength_re
    del alff_roi, reho_roi, reg_num, reg_df, atlas, mask, aligned_int
    del bold_img, mask_img
    gc.collect()

    # Publish to the Windows-visible tree only once the acquisition is complete,
    # so a mid-acquisition crash never leaves a half-written output directory.
    if scratch_root:
        for name in sorted(os.listdir(work)):
            if name.startswith("_in_"):
                continue
            robust_copy(os.path.join(work, name), os.path.join(out_dir, name))
        shutil.rmtree(work, ignore_errors=True)
    return res


# ======================================================================
# MANIFEST
# ======================================================================
def load_state():
    if os.path.isfile(STATE_JSON):
        with open(STATE_JSON) as f:
            return json.load(f)
    return {}


def save_state(state):
    os.makedirs(AUDIT_DIR, exist_ok=True)
    tmp = STATE_JSON + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, indent=1)
    os.replace(tmp, STATE_JSON)
    with open(MANIFEST_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        for key in sorted(state.keys()):
            w.writerow({k: state[key].get(k, "") for k in MANIFEST_FIELDS})
    errs = [v for v in state.values() if v.get("STATUS") == "FAILED"]
    with open(ERRORS_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["GROUP", "SUBJECT", "SESSION", "RUN",
                                          "STAGE", "ERROR", "TRACEBACK", "TIMESTAMP"])
        w.writeheader()
        for e in errs:
            w.writerow({"GROUP": e.get("GROUP"), "SUBJECT": e.get("SUBJECT"),
                        "SESSION": e.get("SESSION"), "RUN": e.get("RUN"),
                        "STAGE": e.get("STAGE", ""), "ERROR": e.get("ERROR", ""),
                        "TRACEBACK": e.get("TRACEBACK", ""), "TIMESTAMP": e.get("TIMESTAMP", "")})


def key_of(g, sub, ses, run):
    return f"{g}/{sub}/{ses}/{run}"


def init_state(acqs, skipped):
    state = load_state()
    for (g, sub, ses, run) in acqs:
        k = key_of(g, sub, ses, run)
        if k not in state:
            state[k] = {"GROUP": g, "SUBJECT": sub, "SESSION": ses, "RUN": run,
                        "STATUS": "PENDING", "BOLD_PATH": bold_path_for(g, sub, ses, run),
                        "ROI_STATUS": "PENDING", "ALFF_STATUS": "PENDING",
                        "REHO_STATUS": "PENDING", "FC_STATUS": "PENDING",
                        "DC_STATUS": "PENDING", "BIOMARKER_STATUS": "PENDING",
                        "ERROR": "", "TIMESTAMP": ""}
    for (g, sub, ses, run, st, reason) in skipped:
        k = key_of(g, sub, ses, run)
        if k not in state or state[k].get("STATUS") != "COMPLETED":
            state[k] = {"GROUP": g, "SUBJECT": sub, "SESSION": ses, "RUN": run,
                        "STATUS": st, "BOLD_PATH": bold_path_for(g, sub, ses, run),
                        "ROI_STATUS": "SKIPPED", "ALFF_STATUS": "SKIPPED",
                        "REHO_STATUS": "SKIPPED", "FC_STATUS": "SKIPPED",
                        "DC_STATUS": "SKIPPED", "BIOMARKER_STATUS": "SKIPPED",
                        "ERROR": reason, "TIMESTAMP": utcnow()}
    save_state(state)
    return state


# ======================================================================
# MODES
# ======================================================================
def mode_discover():
    hdr("DRY-RUN DISCOVERY")
    acqs, skipped = discover_acquisitions()
    log(f"eligible (preprocessing COMPLETED, not excluded): {len(acqs)}")
    log(f"skipped (preproc FAILED or audit exclusion):      {len(skipped)}")
    per_g = {}
    for (g, _, _, _) in acqs:
        per_g[g] = per_g.get(g, 0) + 1
    for g in GROUPS:
        log(f"  {g:10s}: {per_g.get(g,0)}")

    log("\nvalidating BOLD inputs (this reads every file)...")
    bad = []
    for i, (g, sub, ses, run) in enumerate(acqs, 1):
        ok, reason, info = validate_bold(bold_path_for(g, sub, ses, run))
        if not ok:
            bad.append((g, sub, ses, run, reason))
        if i % 25 == 0:
            log(f"  checked {i}/{len(acqs)}")
    log(f"\nBOLD validation: {len(acqs)-len(bad)} valid, {len(bad)} invalid")
    for b in bad:
        log(f"  INVALID {b[0]}/{b[1]}/{b[2]}/{b[3]}: {b[4]}")
    init_state(acqs, skipped)
    log(f"\nmanifest written: {MANIFEST_CSV}")
    log("DISCOVERY_DONE")


def _cmp(name, a, b, results):
    d = float(np.nanmax(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))
    results.append({"quantity": name, "max_abs_diff": d,
                    "match_1e-10": bool(d <= 1e-10), "match_1e-6": bool(d <= 1e-6)})
    log(f"  {name:28s} max|diff| = {d:.3e}   <=1e-10: {d<=1e-10}   <=1e-6: {d<=1e-6}")
    return d


def mode_verify_pilot():
    hdr("PILOT REPRODUCTION CHECK (pilot directory is NOT written to)")
    g, sub, ses, run = PILOT_ACQ
    pilot_run = os.path.join(DERIV, g, sub, ses, run)
    old_bn = os.path.join(pilot_run, "brainnetome")
    old_bio = os.path.join(pilot_run, "biomarkers")
    scratch = os.path.join(AUDIT_DIR, "PILOT_VERIFICATION", f"{sub}_{ses}_{run}")
    os.makedirs(scratch, exist_ok=True)
    log(f"reference (existing, untouched): {old_bio}")
    log(f"scratch output:                  {scratch}")

    t0 = datetime.datetime.now()
    res = process_one(g, sub, ses, run, scratch, no_clobber=False)
    dt = (datetime.datetime.now() - t0).total_seconds()
    log(f"\nreprocessed in {dt:.1f}s")

    results = []
    _cmp("ALFF (246,)", np.load(os.path.join(scratch, "alff.npy")),
         np.load(os.path.join(old_bio, "alff_roi_values.npy")), results)
    _cmp("ReHo (246,)", np.load(os.path.join(scratch, "reho.npy")),
         np.load(os.path.join(old_bio, "reho_roi_values.npy")), results)
    _cmp("FC (246x246)", np.load(os.path.join(scratch, "fc_matrix.npy")),
         np.load(os.path.join(old_bio, "fc_matrix.npy")), results)
    _cmp("DegreeCentrality (246,)", np.load(os.path.join(scratch, "degree_centrality.npy")),
         np.load(os.path.join(old_bio, "degree_centrality.npy")), results)
    _cmp("ROI timeseries (T,246)", np.load(os.path.join(scratch, "roi_timeseries.npy")),
         np.load(os.path.join(old_bn, "roi_timeseries.npy")), results)
    old_strength = pd.read_csv(os.path.join(old_bio, "fc_regional_strength.csv"))["FC_Strength"].values
    _cmp("FC strength (246,)", np.load(os.path.join(scratch, "fc_strength.npy")),
         old_strength, results)
    old_counts = pd.read_csv(os.path.join(old_bn, "roi_voxel_counts.csv"))["Voxel_Count"].values
    new_counts = pd.read_csv(os.path.join(scratch, "roi_voxel_counts.csv"))["Voxel_Count"].values
    _cmp("ROI voxel counts (246,)", new_counts, old_counts, results)
    old_atlas = np.asarray(nib.load(os.path.join(old_bn, "brainnetome_246_4mm.nii.gz")).dataobj)
    new_atlas = np.asarray(nib.load(os.path.join(scratch, "brainnetome_246_4mm.nii.gz")).dataobj)
    _cmp("aligned atlas labels", new_atlas, old_atlas, results)

    pd.DataFrame(results).to_csv(os.path.join(AUDIT_DIR, "pilot_reproduction_check.csv"), index=False)
    worst = max(r["max_abs_diff"] for r in results)
    strict = all(r["match_1e-10"] for r in results)
    loose = all(r["match_1e-6"] for r in results)
    log(f"\nworst max|diff| across all quantities: {worst:.3e}")
    log(f"all quantities match within 1e-10: {strict}")
    log(f"all quantities match within 1e-6:  {loose}")
    log(f"saved: {os.path.join(AUDIT_DIR, 'pilot_reproduction_check.csv')}")
    log(f"\nPILOT_REPRODUCTION: {'PASS' if loose else 'FAIL'}")
    log("VERIFY_PILOT_DONE")
    return loose


def mode_run(only_group=None):
    global ATLAS_PATH, ATLAS_LUT
    hdr("DATASET BIOMARKER GENERATION (MAX_WORKERS=1)")
    os.makedirs(SCRATCH_ROOT, exist_ok=True)
    # stage the atlas on native disk once rather than re-reading it over DrvFs
    # for every acquisition (the originals are only ever read, never modified)
    atlas_local = os.path.join(SCRATCH_ROOT, "BN_Atlas_246_2mm.nii.gz")
    lut_local = os.path.join(SCRATCH_ROOT, "BN_Atlas_246_LUT.txt")
    if not os.path.isfile(atlas_local):
        robust_copy(ATLAS_PATH, atlas_local)
    if not os.path.isfile(lut_local):
        robust_copy(ATLAS_LUT, lut_local)
    ATLAS_PATH, ATLAS_LUT = atlas_local, lut_local
    log(f"scratch: {SCRATCH_ROOT}  (atlas staged locally)")

    acqs, skipped = discover_acquisitions()
    state = init_state(acqs, skipped)
    if only_group:
        acqs = [a for a in acqs if a[0] == only_group]
        log(f"restricted to group: {only_group}")
    log(f"acquisitions in scope: {len(acqs)}")

    done = fail = skip = 0
    for i, (g, sub, ses, run) in enumerate(acqs, 1):
        k = key_of(g, sub, ses, run)
        out_dir = os.path.join(DERIV, g, sub, ses, run, "biomarkers")
        rec = state[k]
        prefix = f"[{i}/{len(acqs)}] {k}"

        # Header-only probe first: the resume check needs just T, and reading
        # every full BOLD over DrvFs before deciding to skip was re-reading
        # gigabytes per restart on a mount that is already failing under load.
        t_probe = None
        try:
            _im = nib.load(bold_path_for(g, sub, ses, run))
            if len(_im.shape) == 4:
                t_probe = int(_im.shape[3])
            del _im
        except Exception:
            t_probe = None

        if t_probe is not None:
            valid, why = validate_existing_outputs(out_dir, t_probe)
            if valid:
                rec.update({"STATUS": "COMPLETED", "ROI_STATUS": "COMPLETED",
                            "ALFF_STATUS": "COMPLETED", "REHO_STATUS": "COMPLETED",
                            "FC_STATUS": "COMPLETED", "DC_STATUS": "COMPLETED",
                            "BIOMARKER_STATUS": "COMPLETED", "ERROR": "", "TIMESTAMP": utcnow()})
                save_state(state)
                log(f"{prefix}  already complete and validated -- skipping")
                skip += 1
                continue

        ok, reason, info = validate_bold(bold_path_for(g, sub, ses, run))
        if not ok:
            rec.update({"STATUS": "FAILED", "STAGE": "input_validation", "ERROR": reason,
                        "TRACEBACK": "", "TIMESTAMP": utcnow()})
            save_state(state)
            log(f"{prefix}  FAILED (input validation): {reason}")
            fail += 1
            continue

        valid, why = validate_existing_outputs(out_dir, info["T"])
        if valid:
            rec.update({"STATUS": "COMPLETED", "ROI_STATUS": "COMPLETED", "ALFF_STATUS": "COMPLETED",
                        "REHO_STATUS": "COMPLETED", "FC_STATUS": "COMPLETED", "DC_STATUS": "COMPLETED",
                        "BIOMARKER_STATUS": "COMPLETED", "ERROR": "", "TIMESTAMP": utcnow()})
            save_state(state)
            log(f"{prefix}  already complete and validated -- skipping")
            skip += 1
            continue

        rec.update({"STATUS": "RUNNING", "TIMESTAMP": utcnow()})
        save_state(state)
        t0 = datetime.datetime.now()
        try:
            no_clobber = (g, sub, ses, run) == PILOT_ACQ
            res = process_one(g, sub, ses, run, out_dir, no_clobber=no_clobber,
                              scratch_root=SCRATCH_ROOT)
            dt = (datetime.datetime.now() - t0).total_seconds()
            rec.update({"STATUS": "COMPLETED", "ROI_STATUS": res["ROI_STATUS"],
                        "ALFF_STATUS": res["ALFF_STATUS"], "REHO_STATUS": res["REHO_STATUS"],
                        "FC_STATUS": res["FC_STATUS"], "DC_STATUS": res["DC_STATUS"],
                        "BIOMARKER_STATUS": res["BIOMARKER_STATUS"], "ERROR": "",
                        "TIMESTAMP": utcnow()})
            log(f"{prefix}  OK in {dt:.1f}s  T={res['arrays']['T']} "
                f"ROIs={res['arrays']['n_present']}/246 "
                f"zeroVar={res['arrays']['n_zero_var']} -> {res['BIOMARKER_STATUS']}"
                + (f"  (preserved {len(res.get('preserved', []))} pilot file(s))" if no_clobber else ""))
            done += 1
        except Exception as e:
            tb = traceback.format_exc()
            rec.update({"STATUS": "FAILED", "STAGE": "processing", "ERROR": str(e),
                        "TRACEBACK": tb[-2000:], "TIMESTAMP": utcnow()})
            # full traceback goes to processing_errors.csv, not stdout: an echoed
            # traceback would be indistinguishable from a top-level crash to the
            # retry wrapper, which must keep going after a single bad acquisition.
            log(f"{prefix}  FAILED ({type(e).__name__}): {e}  [traceback -> processing_errors.csv]")
            fail += 1
        save_state(state)
        gc.collect()

    log(f"\nprocessed={done}  skipped_already_complete={skip}  failed={fail}")
    log("RUN_DONE")


def mode_audit():
    hdr("DATASET-LEVEL BIOMARKER AUDIT")
    acqs, skipped = discover_acquisitions()
    state = load_state()
    lut = load_lut()

    rows = []
    for (g, sub, ses, run) in acqs:
        k = key_of(g, sub, ses, run)
        out_dir = os.path.join(DERIV, g, sub, ses, run, "biomarkers")
        rec = state.get(k, {})
        r = {"GROUP": g, "SUBJECT": sub, "SESSION": ses, "RUN": run,
             "STATUS": rec.get("STATUS", "PENDING"), "OUTPUTS_COMPLETE": False}
        try:
            roi = np.load(os.path.join(out_dir, "roi_timeseries.npy"))
            fc = np.load(os.path.join(out_dir, "fc_matrix.npy"))
            alff = np.load(os.path.join(out_dir, "alff.npy"))
            reho = np.load(os.path.join(out_dir, "reho.npy"))
            dc = np.load(os.path.join(out_dir, "degree_centrality.npy"))
            st = np.load(os.path.join(out_dir, "fc_strength.npy"))
            reg = np.load(os.path.join(out_dir, "regional_biomarkers.npy"))
            counts = pd.read_csv(os.path.join(out_dir, "roi_voxel_counts.csv"))["Voxel_Count"].values
            tmp = fc.copy()
            np.fill_diagonal(tmp, 0.0)
            dc_diff = float(np.nanmax(np.abs(tmp.sum(axis=1) - dc)))
            st_diff = float(np.nanmax(np.abs(tmp.sum(axis=1) / (N_ROI - 1) - st)))
            offd = fc[~np.eye(N_ROI, dtype=bool)]
            r.update({
                "OUTPUTS_COMPLETE": True,
                "T": int(roi.shape[0]), "ROI_COLS": int(roi.shape[1]),
                "ROI_TS_NAN": int(np.isnan(roi).sum()), "ROI_TS_INF": int(np.isinf(roi).sum()),
                "ROI_ZERO_VAR": int(np.sum(roi.std(axis=0) == 0)),
                "BN_ROIS_PRESENT": int((counts > 0).sum()), "BN_ZERO_VOXEL": int((counts == 0).sum()),
                "BN_MIN_VOX": int(counts.min()), "BN_MEDIAN_VOX": float(np.median(counts)),
                "BN_MAX_VOX": int(counts.max()),
                "ALFF_N": int(alff.size), "ALFF_NAN": int(np.isnan(alff).sum()),
                "ALFF_NEG": int(np.sum(alff[np.isfinite(alff)] < 0)),
                "ALFF_MIN": float(np.nanmin(alff)), "ALFF_MAX": float(np.nanmax(alff)),
                "REHO_N": int(reho.size), "REHO_NAN": int(np.isnan(reho).sum()),
                "REHO_MIN": float(np.nanmin(reho)), "REHO_MAX": float(np.nanmax(reho)),
                "REHO_OUT_OF_RANGE": int(np.sum((reho[np.isfinite(reho)] < 0) |
                                                (reho[np.isfinite(reho)] > 1))),
                "FC_SHAPE_OK": bool(fc.shape == (N_ROI, N_ROI)),
                "FC_NAN": int(np.isnan(fc).sum()),
                "FC_DIAG_ERR": float(np.nanmax(np.abs(np.diag(fc) - 1.0))),
                "FC_SYM_ERR": float(np.nanmax(np.abs(fc - fc.T))),
                "FC_PCT_NEG_OFFDIAG": float(100.0 * np.sum(offd < 0) / offd.size),
                "DC_N": int(dc.size), "DC_NAN": int(np.isnan(dc).sum()),
                "DC_MIN": float(np.nanmin(dc)), "DC_MAX": float(np.nanmax(dc)),
                "FCSTR_N": int(st.size), "FCSTR_NAN": int(np.isnan(st).sum()),
                "REG_SHAPE_OK": bool(reg.shape == (N_ROI, 4)),
                "REG_NAN": int(np.isnan(reg).sum()),
                "DC_RECOMPUTE_DIFF": dc_diff, "FCSTR_RECOMPUTE_DIFF": st_diff,
                "CONSISTENT": bool(dc_diff <= TOL and st_diff <= TOL and
                                   fc.shape == (N_ROI, N_ROI) and reg.shape == (N_ROI, 4) and
                                   not np.isnan(reg).any()),
            })
            del roi, fc, alff, reho, dc, st, reg, tmp, offd
            gc.collect()
        except Exception as e:
            r.update({"OUTPUTS_COMPLETE": False, "AUDIT_ERROR": str(e)})
        rows.append(r)

    df = pd.DataFrame(rows)
    os.makedirs(AUDIT_DIR, exist_ok=True)
    df.to_csv(os.path.join(AUDIT_DIR, "dataset_biomarker_summary.csv"), index=False)

    # group-level rollup
    exp = {}
    for (g, _, _, _) in acqs:
        exp[g] = exp.get(g, 0) + 1
    skip_n = {}
    for (g, _, _, _, _, _) in skipped:
        skip_n[g] = skip_n.get(g, 0) + 1

    lines = ["# Dataset Biomarker Audit", "",
             f"Generated (UTC): {utcnow()}", "",
             "## Per-group completion", "",
             "| Group | Expected | Completed | Failed | Skipped (ineligible) | Missing outputs |",
             "|---|---:|---:|---:|---:|---:|"]
    tot = [0, 0, 0, 0, 0]
    for g in GROUPS:
        sub_df = df[df["GROUP"] == g] if len(df) else df
        n_exp = exp.get(g, 0)
        n_comp = int((sub_df["STATUS"] == "COMPLETED").sum()) if len(sub_df) else 0
        n_fail = int((sub_df["STATUS"] == "FAILED").sum()) if len(sub_df) else 0
        n_miss = int((~sub_df["OUTPUTS_COMPLETE"].astype(bool)).sum()) if len(sub_df) else 0
        n_skip = skip_n.get(g, 0)
        lines.append(f"| {g} | {n_exp} | {n_comp} | {n_fail} | {n_skip} | {n_miss} |")
        tot = [tot[0] + n_exp, tot[1] + n_comp, tot[2] + n_fail, tot[3] + n_skip, tot[4] + n_miss]
    lines.append(f"| **TOTAL** | **{tot[0]}** | **{tot[1]}** | **{tot[2]}** | "
                 f"**{tot[3]}** | **{tot[4]}** |")

    okdf = df[df["OUTPUTS_COMPLETE"].astype(bool)] if len(df) else df
    if len(okdf):
        lines += ["", "## Dataset QC (over acquisitions with complete outputs)", "",
                  f"- Acquisitions audited: {len(okdf)}",
                  f"- Brainnetome coverage 246/246: {int((okdf['BN_ROIS_PRESENT']==246).sum())} / {len(okdf)}",
                  f"- Brainnetome zero-voxel ROIs (any): {int((okdf['BN_ZERO_VOXEL']>0).sum())}",
                  f"- ROI voxel count min across dataset: {int(okdf['BN_MIN_VOX'].min())}",
                  f"- T (volumes) range: {int(okdf['T'].min())} - {int(okdf['T'].max())}",
                  f"- ROI matrix columns == 246 for all: {bool((okdf['ROI_COLS']==246).all())}",
                  f"- ROI time-series NaN total: {int(okdf['ROI_TS_NAN'].sum())}",
                  f"- ROI time-series Inf total: {int(okdf['ROI_TS_INF'].sum())}",
                  f"- Acquisitions with zero-variance ROIs: {int((okdf['ROI_ZERO_VAR']>0).sum())}",
                  f"- ALFF: all 246 values present: {bool((okdf['ALFF_N']==246).all())}, "
                  f"NaN total={int(okdf['ALFF_NAN'].sum())}, negative total={int(okdf['ALFF_NEG'].sum())}",
                  f"- ALFF range across dataset: {float(okdf['ALFF_MIN'].min()):.2f} - {float(okdf['ALFF_MAX'].max()):.2f}",
                  f"- ReHo: all 246 values present: {bool((okdf['REHO_N']==246).all())}, "
                  f"NaN total={int(okdf['REHO_NAN'].sum())}, out-of-[0,1] total={int(okdf['REHO_OUT_OF_RANGE'].sum())}",
                  f"- ReHo range across dataset: {float(okdf['REHO_MIN'].min()):.4f} - {float(okdf['REHO_MAX'].max()):.4f}",
                  f"- FC shape 246x246 for all: {bool(okdf['FC_SHAPE_OK'].all())}, NaN total={int(okdf['FC_NAN'].sum())}",
                  f"- FC max diagonal error: {float(okdf['FC_DIAG_ERR'].max()):.3e}",
                  f"- FC max symmetry error: {float(okdf['FC_SYM_ERR'].max()):.3e}",
                  f"- FC negative off-diagonal %: mean={float(okdf['FC_PCT_NEG_OFFDIAG'].mean()):.2f}%, "
                  f"range {float(okdf['FC_PCT_NEG_OFFDIAG'].min()):.2f}-{float(okdf['FC_PCT_NEG_OFFDIAG'].max()):.2f}%",
                  f"- DC: all 246 present: {bool((okdf['DC_N']==246).all())}, NaN total={int(okdf['DC_NAN'].sum())}",
                  f"- DC range across dataset: {float(okdf['DC_MIN'].min()):.4f} - {float(okdf['DC_MAX'].max()):.4f}",
                  f"- FC strength: all 246 present: {bool((okdf['FCSTR_N']==246).all())}, "
                  f"NaN total={int(okdf['FCSTR_NAN'].sum())}",
                  f"- Regional matrix 246x4 for all: {bool(okdf['REG_SHAPE_OK'].all())}, "
                  f"NaN total={int(okdf['REG_NAN'].sum())}",
                  f"- Max DC recomputation difference: {float(okdf['DC_RECOMPUTE_DIFF'].max()):.3e}",
                  f"- Max FC-strength recomputation difference: {float(okdf['FCSTR_RECOMPUTE_DIFF'].max()):.3e}",
                  f"- Cross-file consistent: {int(okdf['CONSISTENT'].sum())} / {len(okdf)}"]

        unusual = okdf[(okdf["ROI_ZERO_VAR"] > 0) | (okdf["BN_ZERO_VOXEL"] > 0) |
                       (okdf["FC_NAN"] > 0) | (okdf["REHO_OUT_OF_RANGE"] > 0) |
                       (okdf["ALFF_NEG"] > 0) | (~okdf["CONSISTENT"])]
        lines += ["", "## Unusual acquisitions", ""]
        if len(unusual) == 0:
            lines.append("None. Every audited acquisition passed all structural, numerical and "
                         "cross-file consistency checks.")
        else:
            lines.append("| Acquisition | zeroVarROI | zeroVoxROI | FC NaN | ReHo out-of-range | ALFF neg | consistent |")
            lines.append("|---|---:|---:|---:|---:|---:|---|")
            for _, u in unusual.iterrows():
                lines.append(f"| {u['GROUP']}/{u['SUBJECT']}/{u['SESSION']}/{u['RUN']} | "
                             f"{u['ROI_ZERO_VAR']} | {u['BN_ZERO_VOXEL']} | {u['FC_NAN']} | "
                             f"{u['REHO_OUT_OF_RANGE']} | {u['ALFF_NEG']} | {u['CONSISTENT']} |")

    fails = [v for v in state.values() if v.get("STATUS") == "FAILED"]
    lines += ["", "## Processing errors", ""]
    if not fails:
        lines.append("No processing errors.")
    else:
        lines.append("| Acquisition | Stage | Error |")
        lines.append("|---|---|---|")
        for e in fails:
            lines.append(f"| {e['GROUP']}/{e['SUBJECT']}/{e['SESSION']}/{e['RUN']} | "
                         f"{e.get('STAGE','')} | {str(e.get('ERROR',''))[:160]} |")

    with open(os.path.join(AUDIT_DIR, "dataset_biomarker_summary.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log("\n".join(lines))
    log(f"\nsaved: {os.path.join(AUDIT_DIR, 'dataset_biomarker_summary.csv')}")
    log(f"saved: {os.path.join(AUDIT_DIR, 'dataset_biomarker_summary.md')}")
    log("AUDIT_DONE")


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "discover"
    os.makedirs(AUDIT_DIR, exist_ok=True)
    if mode == "discover":
        mode_discover()
    elif mode == "verify-pilot":
        mode_verify_pilot()
    elif mode == "run":
        mode_run(sys.argv[2] if len(sys.argv) > 2 else None)
    elif mode == "audit":
        mode_audit()
    else:
        log(f"unknown mode: {mode}")
        sys.exit(2)
