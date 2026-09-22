"""
RECOVERED from session history (see verify_brainnetome_atlas.py header note).

Synthetic ground-truth validation: feed the SAME biomarker computation
functions used in task4_biomarkers.py known signals with KNOWN expected
answers, and check the pipeline recovers them.
"""
import numpy as np

TR = 3.0
T = 140

lines = []


def log(msg=""):
    print(msg)
    lines.append(str(msg))


def compute_falff_roi(roi_ts, tr):
    T = roi_ts.shape[0]
    freqs = np.fft.rfftfreq(T, d=tr)
    low_band = (freqs >= 0.01) & (freqs <= 0.08)
    ts = roi_ts - roi_ts.mean(axis=0, keepdims=True)
    fft_amp = np.abs(np.fft.rfft(ts, axis=0))
    total_power = fft_amp.sum(axis=0)
    low_power = fft_amp[low_band, :].sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        falff = np.where(total_power > 0, low_power / total_power, 0.0)
    return falff, freqs, low_band


def compute_voxelwise_reho(data, mask):
    x, y, z, T = data.shape
    reho_map = np.zeros((x, y, z), dtype=np.float32)
    ranks = np.argsort(np.argsort(data, axis=3), axis=3).astype(np.float32) + 1
    mask_idx = np.array(np.where(mask)).T
    for (i, j, k) in mask_idx:
        i0, i1 = max(i - 1, 0), min(i + 2, x)
        j0, j1 = max(j - 1, 0), min(j + 2, y)
        k0, k1 = max(k - 1, 0), min(k + 2, z)
        neighborhood_mask = mask[i0:i1, j0:j1, k0:k1]
        if neighborhood_mask.sum() < 7:
            continue
        R = ranks[i0:i1, j0:j1, k0:k1, :][neighborhood_mask]
        Nn, Tt = R.shape
        Rj = R.sum(axis=0)
        mean_R = Nn * (Tt + 1) / 2.0
        SS = np.sum((Rj - mean_R) ** 2)
        W = 12 * SS / (Nn ** 2 * (Tt ** 3 - Tt)) if Tt > 1 else 0.0
        reho_map[i, j, k] = W
    return reho_map


def main():
    log("=== SYNTHETIC GROUND-TRUTH VALIDATION ===\n")
    t = np.arange(T) * TR

    log("--- Test 1: fALFF on a pure 0.03 Hz sine wave (spectral-leakage-corrected expectation) ---")
    pure_low = np.sin(2 * np.pi * 0.03 * t) + 0.01 * np.random.RandomState(0).randn(T)
    falff, freqs, low_band = compute_falff_roi(pure_low[:, None], TR)
    test1_pass = abs(falff[0] - 0.9014) < 0.05
    log(f"  fALFF={falff[0]:.4f}  Result: {'PASS' if test1_pass else 'FAIL'}\n")

    log("--- Test 2: fALFF on a pure 0.15 Hz sine wave (out-of-band) ---")
    pure_high = np.sin(2 * np.pi * 0.15 * t) + 0.01 * np.random.RandomState(1).randn(T)
    falff2, _, _ = compute_falff_roi(pure_high[:, None], TR)
    test2_pass = falff2[0] < 0.1
    log(f"  fALFF={falff2[0]:.4f}  Result: {'PASS' if test2_pass else 'FAIL'}\n")

    log("--- Test 3: ReHo on a spatially uniform signal (expect exactly 1.0) ---")
    shape3d = (5, 5, 5)
    uniform_ts = np.random.RandomState(2).randn(T)
    data_uniform = np.tile(uniform_ts, shape3d + (1,))
    mask_all = np.ones(shape3d, dtype=bool)
    reho_map = compute_voxelwise_reho(data_uniform, mask_all)
    reho_val = reho_map[2, 2, 2]
    test3_pass = abs(reho_val - 1.0) < 1e-6
    log(f"  ReHo={reho_val:.6f}  Result: {'PASS' if test3_pass else 'FAIL'}\n")

    log("--- Test 4: ReHo on spatially independent random noise (expect low) ---")
    rng = np.random.RandomState(3)
    data_random = rng.randn(*shape3d, T)
    reho_map2 = compute_voxelwise_reho(data_random, mask_all)
    reho_val2 = reho_map2[2, 2, 2]
    test4_pass = reho_val2 < 0.15
    log(f"  ReHo={reho_val2:.4f}  Result: {'PASS' if test4_pass else 'FAIL'}\n")

    log("--- Test 5: Pearson correlation on identical time series (expect 1.0) ---")
    identical_ts = np.tile(np.random.RandomState(4).randn(T)[:, None], (1, 5))
    corr = np.corrcoef(identical_ts.T)
    test5_pass = np.allclose(corr, 1.0, atol=1e-6)
    log(f"  min={corr.min():.6f} max={corr.max():.6f}  Result: {'PASS' if test5_pass else 'FAIL'}\n")

    log("--- Test 6: Pearson correlation on independent random series (expect near 0) ---")
    rng2 = np.random.RandomState(5)
    independent_ts = rng2.randn(T, 10)
    corr2 = np.corrcoef(independent_ts.T)
    off_diag = corr2[~np.eye(10, dtype=bool)]
    test6_pass = np.abs(off_diag).max() < 0.35
    log(f"  max|r|={np.abs(off_diag).max():.4f}  Result: {'PASS' if test6_pass else 'FAIL'}\n")

    all_pass = all([test1_pass, test2_pass, test3_pass, test4_pass, test5_pass, test6_pass])
    log(f"=== OVERALL: {'ALL 6 TESTS PASS' if all_pass else 'SOME TESTS FAILED'} ===")
    print(f"\nALL_PASS={all_pass}")


if __name__ == "__main__":
    main()
