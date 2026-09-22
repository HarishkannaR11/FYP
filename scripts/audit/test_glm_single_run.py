"""
ONE-OFF TEST (not part of the production pipeline): run mri_glmfit on the
already-existing test/ desc-preproc-NO-GLM_bold.nii.gz for sub-019S4549
ses-01 run-01, using the exact same 9-column design (intercept, linear
trend, quadratic trend, 6 mc-afni2 motion params) and the exact same
mri_glmfit options as pipeline/fsfast_ad_full_rerun.py (build_design() /
run_glmfit(), reused verbatim below -- not reimplemented from scratch).

Reads only from test/ (already a copy, not the original BIDS/derivatives
file) and writes only into test/. Touches nothing else.
"""
import os
import sys
import subprocess
import time
import numpy as np
import nibabel as nib

TEST_DIR = "/mnt/c/Users/krish/FYP/test"
INPUT_NOGLM = os.path.join(TEST_DIR, "sub-019S4549_ses-01_task-rest_run-01_desc-preproc-NO-GLM_bold.nii.gz")
MCDAT = os.path.join(TEST_DIR, "sub-019S4549_ses-01_task-rest_run-01_desc-motion_timeseries.mcdat")
GLMDIR = os.path.join(TEST_DIR, "glmdir_sub-019S4549_ses-01")
DESIGN_OUT = os.path.join(TEST_DIR, "sub-019S4549_ses-01_task-rest_run-01_design_matrix.txt")
FINAL_OUT = os.path.join(TEST_DIR, "sub-019S4549_ses-01_task-rest_run-01_desc-preproc-GLM_bold.nii.gz")
LOG_OUT = os.path.join(TEST_DIR, "sub-019S4549_ses-01_task-rest_run-01_glm_test_log.txt")


def build_design(mcdat_file, out_path):
    """Verbatim copy of build_design() in pipeline/fsfast_ad_full_rerun.py."""
    mc = np.loadtxt(mcdat_file)
    n = mc.shape[0]
    motion = mc[:, 1:7]
    t = np.arange(n, dtype=float)
    t = (t - t.mean()) / t.std()
    X = np.column_stack([np.ones(n), t, t ** 2 - np.mean(t ** 2), motion])
    np.savetxt(out_path, X, fmt="%.6f")
    return X.shape[1], n


def main():
    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(msg)

    log("=== TEST-ONLY GLM run (NOT part of production pipeline) ===")
    log(f"Input (existing NO-GLM test derivative, read-only): {INPUT_NOGLM}")
    log(f"Motion file (existing test copy, read-only): {MCDAT}")
    assert os.path.exists(INPUT_NOGLM), "NO-GLM input missing"
    assert os.path.exists(MCDAT), "mcdat file missing"

    # ---- pre-GLM validation of the input ----
    img_in = nib.load(INPUT_NOGLM)
    shape_in = img_in.shape
    zooms_in = img_in.header.get_zooms()
    log(f"Input shape: {shape_in}, voxel size: {zooms_in[:3]}, TR: {zooms_in[3] if len(zooms_in) > 3 else 'N/A'}")
    assert len(shape_in) == 4, "input is not 4D"
    assert shape_in[:3] == (64, 64, 48), f"unexpected spatial dims {shape_in[:3]}"
    assert shape_in[3] == 140, f"unexpected timepoints {shape_in[3]}"
    data_in = img_in.get_fdata(dtype=np.float32)
    assert not np.isnan(data_in).any(), "NaN in input"
    assert not np.isinf(data_in).any(), "Inf in input"
    log("Input validation: PASS (4D, 64x64x48, 140 timepoints, no NaN/Inf)")

    # ---- design matrix (identical construction to production pipeline) ----
    ncols, n = build_design(MCDAT, DESIGN_OUT)
    log(f"Design matrix: {ncols} columns, {n} rows -> {DESIGN_OUT}")
    assert ncols == 9, f"expected 9 design columns, got {ncols}"
    assert n == 140, f"expected 140 rows in design matrix, got {n}"

    # ---- mri_glmfit (identical command/options to production pipeline) ----
    os.makedirs(TEST_DIR, exist_ok=True)
    cmd = ["mri_glmfit", "--y", INPUT_NOGLM, "--X", DESIGN_OUT, "--no-contrasts-ok",
           "--no-mask", "--glmdir", GLMDIR, "--eres-save", "--nii.gz"]
    log(f"Command: {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    dt = time.time() - t0
    log(f"mri_glmfit exit code: {res.returncode} (time {dt:.2f}s)")
    log("--- stdout (tail) ---")
    log(res.stdout[-3000:])
    log("--- stderr (tail) ---")
    log(res.stderr[-3000:])
    assert res.returncode == 0, "mri_glmfit failed"

    eres = os.path.join(GLMDIR, "eres.nii.gz")
    assert os.path.exists(eres), "eres.nii.gz not produced"

    # ---- copy to clearly-named final test output ----
    import shutil
    shutil.copy(eres, FINAL_OUT)
    log(f"Copied {eres} -> {FINAL_OUT}")

    # ---- post-GLM validation ----
    img_out = nib.load(FINAL_OUT)
    shape_out = img_out.shape
    zooms_out = img_out.header.get_zooms()
    log(f"Output shape: {shape_out}, voxel size: {zooms_out[:3]}")
    assert len(shape_out) == 4, "output is not 4D"
    assert shape_out[:3] == (64, 64, 48), f"output spatial dims changed: {shape_out[:3]}"
    assert shape_out[3] == 140, f"output timepoints changed: {shape_out[3]}"
    assert tuple(round(float(v), 2) for v in zooms_out[:3]) == tuple(round(float(v), 2) for v in zooms_in[:3]), \
        "voxel size changed"
    data_out = img_out.get_fdata(dtype=np.float32)
    n_nan = int(np.isnan(data_out).sum())
    n_inf = int(np.isinf(data_out).sum())
    log(f"Output NaN count: {n_nan}, Inf count: {n_inf}")
    assert n_nan == 0, "NaN in output"
    assert n_inf == 0, "Inf in output"

    log("=== VALIDATION SUMMARY ===")
    log("Input 4D: PASS")
    log("Output 4D: PASS")
    log(f"Dimensions preserved (64x64x48): PASS")
    log(f"Timepoints preserved (140): PASS")
    log(f"Voxel size preserved ({zooms_in[:3]}): PASS")
    log("No NaN/Inf in output: PASS")
    log("GLM completed successfully: PASS")
    log(f"Exact command used: {' '.join(cmd)}")
    log(f"Final output path: {FINAL_OUT}")
    log(f"Design matrix path: {DESIGN_OUT}")
    log(f"GLM working dir (full mri_glmfit output incl. logs): {GLMDIR}")

    with open(LOG_OUT, "w") as f:
        f.write("\n".join(log_lines))
    print(f"\nLog written to {LOG_OUT}")


if __name__ == "__main__":
    main()
