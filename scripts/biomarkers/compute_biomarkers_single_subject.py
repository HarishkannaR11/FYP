"""
TEST/PILOT: compute ALFF (fALFF), ReHo, and Degree Centrality (DC) for ONE
AD acquisition (sub-019S5012 ses-01), in NATIVE space (no atlas parcellation
yet -- that requires spatial normalization to MNI, which is blocked pending
the Brainnetome atlas file + a registration tool decision).

Inputs (read-only):
  - smooth_bold.nii.gz  (post-moco, pre-GLM smoothed intermediate, from the
    Nipype work-dir cache) -- used for ALFF. NOT the final GLM residual,
    because polynomial detrending removes exactly the low-frequency power
    ALFF measures.
  - moco_bold.nii.gz  (post-moco, UNSMOOTHED intermediate, same cache) --
    used for ReHo, per standard practice (smoothing inflates local
    homogeneity artificially).
  - desc-preproc_bold.nii.gz  (final masked GLM residual, from
    derivatives/fsfast/AD/) -- used for DC, since nuisance regression is
    standard/expected before connectivity-based measures.
  - desc-brain_mask.nii.gz  (from derivatives/fsfast/AD/) -- restricts all
    three computations to brain voxels.

Outputs -> test/biomarkers/ only. Touches nothing else.
"""
import os
import time
import numpy as np
import nibabel as nib

SUB, SES = "sub-019S5012", "ses-01"
TR = 3.0  # seconds, confirmed from header in prior QC

WORK_DIR = f"/home/harish/fyp_work/fsfast_ad_masked/work/fsfast_ad_production_masked/_run_key_{SUB}_{SES}"
SMOOTH_FILE = os.path.join(WORK_DIR, "smooth", "smooth_bold.nii.gz")
MOCO_FILE = os.path.join(WORK_DIR, "moco", "moco_bold.nii.gz")

DERIV_DIR = f"/mnt/c/Users/krish/FYP/derivatives/fsfast/AD/{SUB}/{SES}/func"
FINAL_BOLD = os.path.join(DERIV_DIR, f"{SUB}_{SES}_task-rest_run-01_desc-preproc_bold.nii.gz")
MASK_FILE = os.path.join(DERIV_DIR, f"{SUB}_{SES}_task-rest_run-01_desc-brain_mask.nii.gz")

OUT_DIR = "/mnt/c/Users/krish/FYP/test/biomarkers"
os.makedirs(OUT_DIR, exist_ok=True)


def log(msg, lines):
    print(msg)
    lines.append(str(msg))


def compute_falff(data, mask, tr):
    """fALFF: low-frequency (0.01-0.08 Hz) power / total power, per voxel."""
    T = data.shape[3]
    freqs = np.fft.rfftfreq(T, d=tr)
    low_band = (freqs >= 0.01) & (freqs <= 0.08)

    vox = data[mask]  # (Nvox, T)
    vox = vox - vox.mean(axis=1, keepdims=True)
    fft_amp = np.abs(np.fft.rfft(vox, axis=1))
    total_power = fft_amp.sum(axis=1)
    low_power = fft_amp[:, low_band].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        falff = np.where(total_power > 0, low_power / total_power, 0.0)

    falff_map = np.zeros(mask.shape, dtype=np.float32)
    falff_map[mask] = falff
    return falff_map, freqs, low_band


def compute_reho(data, mask):
    """Kendall's coefficient of concordance (KCC) over each voxel's 27-voxel
    (3x3x3) neighborhood, across time. Computed on UNSMOOTHED data."""
    x, y, z, T = data.shape
    reho_map = np.zeros((x, y, z), dtype=np.float32)

    # rank each voxel's time series once (KCC uses ranks across time)
    ranks = np.argsort(np.argsort(data, axis=3), axis=3).astype(np.float32) + 1

    mask_idx = np.array(np.where(mask)).T
    n = 27  # neighborhood size (3x3x3, edge voxels get fewer -- handled below)

    for (i, j, k) in mask_idx:
        i0, i1 = max(i - 1, 0), min(i + 2, x)
        j0, j1 = max(j - 1, 0), min(j + 2, y)
        k0, k1 = max(k - 1, 0), min(k + 2, z)
        neighborhood_mask = mask[i0:i1, j0:j1, k0:k1]
        if neighborhood_mask.sum() < 7:  # require a minimally reasonable neighborhood
            continue
        R = ranks[i0:i1, j0:j1, k0:k1, :][neighborhood_mask]  # (Nneighbors, T)
        Nn, Tt = R.shape
        Rj = R.sum(axis=0)  # sum of ranks per timepoint across neighbors
        mean_R = Nn * (Tt + 1) / 2.0
        SS = np.sum((Rj - mean_R) ** 2)
        W = 12 * SS / (Nn ** 2 * (Tt ** 3 - Tt)) if Tt > 1 else 0.0
        reho_map[i, j, k] = W
    return reho_map


def compute_dc(data, mask, r_thresh=0.25, chunk=500):
    """Weighted degree centrality: for each brain voxel, sum of correlations
    with all other brain voxels exceeding r_thresh. Computed in chunks to
    keep memory bounded (given the known WSL RAM ceiling)."""
    vox = data[mask]  # (Nvox, T)
    vox = vox - vox.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(vox, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vox_n = vox / norms  # unit-norm rows -> dot product = correlation

    Nvox = vox_n.shape[0]
    degree = np.zeros(Nvox, dtype=np.float32)
    for start in range(0, Nvox, chunk):
        end = min(start + chunk, Nvox)
        corr_block = vox_n[start:end] @ vox_n.T  # (chunk, Nvox)
        corr_block[np.arange(end - start), np.arange(start, end)] = 0  # zero self-corr
        thresholded = np.where(corr_block > r_thresh, corr_block, 0.0)
        degree[start:end] = thresholded.sum(axis=1)

    dc_map = np.zeros(mask.shape, dtype=np.float32)
    dc_map[mask] = degree
    return dc_map


def main():
    lines = []
    log("=== Single-subject biomarker generation (native space, no atlas parcellation) ===", lines)
    log(f"Subject: {SUB} {SES}\n", lines)

    log("--- Inputs (read-only) ---", lines)
    log(f"ALFF input (pre-GLM smoothed): {SMOOTH_FILE}", lines)
    log(f"ReHo input (unsmoothed moco): {MOCO_FILE}", lines)
    log(f"DC input (final masked residual): {FINAL_BOLD}", lines)
    log(f"Mask: {MASK_FILE}\n", lines)

    for p in [SMOOTH_FILE, MOCO_FILE, FINAL_BOLD, MASK_FILE]:
        assert os.path.exists(p), f"missing required input: {p}"

    mask = nib.load(MASK_FILE).get_fdata().astype(bool)
    affine = nib.load(FINAL_BOLD).affine
    log(f"Brain mask: {mask.sum()} / {mask.size} voxels ({100*mask.sum()/mask.size:.2f}%)\n", lines)

    # ---- ALFF / fALFF ----
    log("=== Computing fALFF ===", lines)
    t0 = time.time()
    smooth_data = nib.load(SMOOTH_FILE).get_fdata(dtype=np.float32)
    falff_map, freqs, low_band = compute_falff(smooth_data, mask, TR)
    log(f"  TR={TR}s, {smooth_data.shape[3]} timepoints, freq resolution={freqs[1]-freqs[0]:.5f} Hz", lines)
    log(f"  low-freq band 0.01-0.08 Hz covers {low_band.sum()} of {len(freqs)} frequency bins", lines)
    log(f"  fALFF within-mask: mean={falff_map[mask].mean():.4f} std={falff_map[mask].std():.4f} "
        f"min={falff_map[mask].min():.4f} max={falff_map[mask].max():.4f}", lines)
    falff_out = os.path.join(OUT_DIR, f"{SUB}_{SES}_falff.nii.gz")
    nib.save(nib.Nifti1Image(falff_map, affine), falff_out)
    log(f"  Saved: {falff_out}  (time {time.time()-t0:.1f}s)\n", lines)

    # ---- ReHo ----
    log("=== Computing ReHo (Kendall's W, 3x3x3 neighborhood) ===", lines)
    t0 = time.time()
    moco_data = nib.load(MOCO_FILE).get_fdata(dtype=np.float32)
    reho_map = compute_reho(moco_data, mask)
    log(f"  ReHo within-mask: mean={reho_map[mask].mean():.4f} std={reho_map[mask].std():.4f} "
        f"min={reho_map[mask].min():.4f} max={reho_map[mask].max():.4f}", lines)
    reho_out = os.path.join(OUT_DIR, f"{SUB}_{SES}_reho.nii.gz")
    nib.save(nib.Nifti1Image(reho_map, affine), reho_out)
    log(f"  Saved: {reho_out}  (time {time.time()-t0:.1f}s)\n", lines)

    # ---- Degree Centrality ----
    log("=== Computing weighted Degree Centrality (r > 0.25) ===", lines)
    t0 = time.time()
    final_data = nib.load(FINAL_BOLD).get_fdata(dtype=np.float32)
    dc_map = compute_dc(final_data, mask, r_thresh=0.25)
    log(f"  DC within-mask: mean={dc_map[mask].mean():.4f} std={dc_map[mask].std():.4f} "
        f"min={dc_map[mask].min():.4f} max={dc_map[mask].max():.4f}", lines)
    dc_out = os.path.join(OUT_DIR, f"{SUB}_{SES}_dc.nii.gz")
    nib.save(nib.Nifti1Image(dc_map, affine), dc_out)
    log(f"  Saved: {dc_out}  (time {time.time()-t0:.1f}s)\n", lines)

    log("=== NOTE ===", lines)
    log("FC (functional connectivity) is not computed here -- standard practice computes it", lines)
    log("at the ATLAS-REGION level (246x246 matrix), which requires spatial normalization to", lines)
    log("MNI + the Brainnetome atlas, neither of which is available yet (see conversation).", lines)
    log("All three maps above are in NATIVE subject space -- parcellation into the 246", lines)
    log("Brainnetome regions is a separate, still-blocked step for ALL FOUR biomarkers.", lines)

    report_path = os.path.join(OUT_DIR, f"{SUB}_{SES}_biomarker_report.txt")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport: {report_path}")


if __name__ == "__main__":
    main()
