import os
import csv
import json
import time
import shutil
import subprocess
import numpy as np
import nibabel as nib

os.environ["FREESURFER_HOME"] = "/home/harish/freesurfer"
os.environ["FSFAST_HOME"] = "/home/harish/freesurfer/fsfast"
os.environ["SUBJECTS_DIR"] = "/home/harish/freesurfer/subjects"
os.environ["PATH"] = "/home/harish/freesurfer/bin:/home/harish/freesurfer/fsfast/bin:" + os.environ["PATH"]

from nipype.pipeline import engine as pe
from nipype.interfaces.utility import IdentityInterface, Function

WORK_DIR = "/home/harish/fyp_work/ad20/work"
BIDS_ROOT = "/mnt/c/Users/krish/FYP/BIDS"
DERIV_ROOT = "/mnt/c/Users/krish/FYP/derivatives/fsfast"

# Fixed, pre-verified selection -- the 20 remaining AD-group runs identified as
# NOT_YET_PROCESSED in audit/ad_session_mapping_final.csv (session mapping
# already verified: exactly one session per acquisition, no collisions).
RUNS = [
    ("sub-002S5018", "ses-02"), ("sub-018S4733", "ses-01"), ("sub-019S4252", "ses-01"),
    ("sub-019S4477", "ses-01"), ("sub-019S4477", "ses-02"), ("sub-019S4549", "ses-01"),
    ("sub-019S4549", "ses-02"), ("sub-019S5012", "ses-01"), ("sub-019S5019", "ses-01"),
    ("sub-031S4024", "ses-01"), ("sub-130S4641", "ses-01"), ("sub-130S4660", "ses-01"),
    ("sub-130S4730", "ses-01"), ("sub-130S4730", "ses-02"), ("sub-130S4971", "ses-01"),
    ("sub-130S4982", "ses-01"), ("sub-130S4982", "ses-02"), ("sub-130S4984", "ses-01"),
    ("sub-130S4990", "ses-01"), ("sub-130S5059", "ses-01"),
]

FWHM_MM = 6.0


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
    """Hard safeguard: refuse to proceed if an output already exists for this
    exact subject+session -- never overwrite another session's or an
    already-processed run's derivative."""
    import os
    existing = os.path.join(deriv_root, sub, ses, "func",
                             f"{sub}_{ses}_task-rest_{run}_desc-preproc_bold.nii.gz")
    already_exists = os.path.isfile(existing)
    return not already_exists, existing


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
    tr_nifti = float(img.header.get_zooms()[3])
    with open(json_file) as f:
        meta = js.load(f)
    tr_json = meta.get("RepetitionTime", None)
    ok = len(errors) == 0
    dt = time.time() - t0
    return ok, str(shape), tr_nifti, tr_json, "; ".join(errors), dt


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
    return final, True, "", dt


def finalize(sub, ses, run, safe_to_write, existing_path, bold_file, json_file, orig_shape,
             orig_tr_nifti, orig_tr_json, validate_ok, validate_err, moco_file, mcdat_file,
             moco_ok, moco_err, smooth_file, smooth_ok, smooth_err, design_file, design_ncols,
             final_file, glmfit_ok, glmfit_err, t_validate, t_moco, t_smooth, t_design, t_glmfit,
             fwhm, deriv_root):
    import nibabel as nib
    import numpy as np
    import os
    import json as js
    import shutil
    import time

    errors = []
    warnings = []

    if not safe_to_write:
        errors.append(f"OUTPUT ALREADY EXISTS at {existing_path} -- refusing to overwrite")

    for ok, err, stage in [(validate_ok, validate_err, "validate"), (moco_ok, moco_err, "moco"),
                            (smooth_ok, smooth_err, "smooth"), (glmfit_ok, glmfit_err, "glmfit")]:
        if not ok:
            errors.append(f"{stage}: {err}")

    qc_rows = []
    max_trans = None
    qc_pass = len(errors) == 0

    out_dir = os.path.join(deriv_root, sub, ses, "func")
    final_bold_out = None
    motion_tsv_out = None
    json_out = None

    if qc_pass:
        os.makedirs(out_dir, exist_ok=True)
        mc = np.loadtxt(mcdat_file)
        max_trans = float(np.max(mc[:, 9]))
        if mc.shape[0] != 140:
            errors.append(f"motion params row count {mc.shape[0]} != 140")
            qc_pass = False

        expect_shape = eval(orig_shape)
        for stage_name, path in [("input_original_BIDS", bold_file), ("motion_corrected", moco_file),
                                  ("smoothed", smooth_file), ("final_preproc_eres", final_file)]:
            row = {"sub": sub, "ses": ses, "run": run, "stage": stage_name, "file": path}
            try:
                img = nib.load(path)
                data = img.get_fdata()
                shp = img.shape
                row["shape"] = str(shp)
                row["shape_match"] = "YES" if shp == expect_shape else "NO"
                row["has_nan"] = str(bool(np.isnan(data).any()))
                row["has_inf"] = str(bool(np.isinf(data).any()))
                row["readable"] = "YES"
                row["status"] = "PASS" if row["shape_match"] == "YES" and row["has_nan"] == "False" and row["has_inf"] == "False" else "FAIL"
                if row["status"] == "FAIL":
                    qc_pass = False
            except Exception as e:
                row["readable"] = "NO"
                row["status"] = "FAIL"
                row["error"] = str(e)
                qc_pass = False
            qc_rows.append(row)

        if qc_pass:
            final_bold_out = os.path.join(out_dir, f"{sub}_{ses}_task-rest_{run}_desc-preproc_bold.nii.gz")
            if os.path.isfile(final_bold_out):
                # second safeguard, in case of a race -- never overwrite
                errors.append(f"OUTPUT APPEARED DURING PROCESSING at {final_bold_out} -- refusing to overwrite")
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
                    "OriginalDimensions": orig_shape,
                    "OriginalRepetitionTime": orig_tr_json,
                    "SkullStripped": False,
                    "PreprocessingPipeline": "FS-FAST + Nipype (BOLD-only, no T1w)",
                    "ProcessingSteps": [
                        {"step": "data_validation", "tool": "NiBabel/PyBIDS", "status": "PASS"},
                        {"step": "motion_correction", "tool": "mc-afni2", "reference_frame": 0,
                         "max_translation_mm": max_trans, "status": "PASS"},
                        {"step": "slice_timing_correction", "status": "SKIPPED",
                         "reason": "SliceTiming not recoverable -- confirmed by exhaustive BIDS+DICOM audit "
                                   "(audit/slice_timing_final_exhaustive_audit.txt). Not fabricated."},
                        {"step": "spatial_smoothing", "tool": "mri_fwhm", "fwhm_mm": fwhm, "status": "PASS"},
                        {"step": "temporal_filtering_and_nuisance_regression", "tool": "mri_glmfit",
                         "design_matrix_columns": design_ncols,
                         "design_regressors": ["intercept", "linear_trend", "quadratic_trend",
                                                "motion_roll", "motion_pitch", "motion_yaw",
                                                "motion_dS", "motion_dL", "motion_dP"],
                         "wm_csf_regression": "UNAVAILABLE (no T1w segmentation)", "status": "PASS"},
                    ],
                }
                with open(json_out, "w") as f:
                    js.dump(sidecar, f, indent=2)
        if not qc_pass:
            warnings.append("QC failed on one or more stages -- outputs NOT copied to derivatives")
    else:
        qc_pass = False

    total_time = sum(x for x in [t_validate, t_moco, t_smooth, t_design, t_glmfit] if x)

    log_lines = [
        f"=== {sub} {ses} {run} processing log ===",
        f"Input BOLD: {bold_file}",
        f"Input JSON: {json_file}",
        f"Original dimensions: {orig_shape}",
        f"Original TR (NIfTI header): {orig_tr_nifti}",
        f"Original TR (JSON RepetitionTime): {orig_tr_json}",
        f"Pre-check (no overwrite): {'PASS -- no existing output' if safe_to_write else 'FAIL -- output already existed at ' + existing_path}",
        f"Validation: {'PASS' if validate_ok else 'FAIL - ' + validate_err} (time {t_validate:.2f}s)",
        f"Motion correction (mc-afni2): {'PASS' if moco_ok else 'FAIL - ' + moco_err} (time {t_moco:.2f}s)",
        f"  Max translation (mm): {max_trans}",
        f"Slice-timing correction: SKIPPED -- SliceTiming not recoverable (exhaustive audit); not fabricated.",
        f"Spatial smoothing (mri_fwhm, FWHM={fwhm}mm): {'PASS' if smooth_ok else 'FAIL - ' + smooth_err} (time {t_smooth:.2f}s)",
        f"Design matrix: {design_ncols} columns (intercept, linear, quadratic, 6 motion params) (time {t_design:.2f}s)",
        f"GLM fit (mri_glmfit, temporal filter + motion nuisance regression): {'PASS' if glmfit_ok else 'FAIL - ' + glmfit_err} (time {t_glmfit:.2f}s)",
        f"WM/CSF nuisance regression: UNAVAILABLE (no T1w-derived segmentation)",
        f"Total processing time: {total_time:.2f}s",
        f"QC: {'PASS' if qc_pass else 'FAIL'}",
        f"Errors: {'; '.join(errors) if errors else 'none'}",
        f"Warnings: {'; '.join(warnings) if warnings else 'none'}",
        f"Output BOLD: {final_bold_out}",
        f"Output motion TSV: {motion_tsv_out}",
        f"Output JSON sidecar: {json_out}",
        "",
    ]

    logs_dir = os.path.join(deriv_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    # Session-qualified log filename -- prevents a second session for the
    # same subject from overwriting the first session's log.
    subj_log = os.path.join(logs_dir, f"{sub}_{ses}_processing.log")
    with open(subj_log, "w") as f:
        f.write("\n".join(log_lines))

    summary_row = {
        "sub": sub, "ses": ses, "run": run,
        "input_file": bold_file, "output_file": final_bold_out or "",
        "original_dimensions": orig_shape, "original_TR": orig_tr_json,
        "motion_correction_status": "PASS" if moco_ok else "FAIL",
        "max_translation_mm": max_trans,
        "smoothing_fwhm_mm": fwhm,
        "design_matrix_columns": design_ncols,
        "slice_timing_status": "SKIPPED_NOT_RECOVERABLE",
        "total_processing_time_s": round(total_time, 2),
        "qc_status": "PASS" if qc_pass else "FAIL",
        "errors": "; ".join(errors) if errors else "",
        "warnings": "; ".join(warnings) if warnings else "",
    }

    return qc_rows, summary_row, subj_log


def collect_and_write(qc_rows_list, summary_rows_list, subj_logs_list, deriv_root, fwhm):
    import csv
    import json as js
    import os
    import time

    qc_dir = os.path.join(deriv_root, "qc")
    logs_dir = os.path.join(deriv_root, "logs")
    os.makedirs(qc_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    qc_csv = os.path.join(qc_dir, "preprocessing_qc.csv")
    fieldnames = ["sub", "ses", "run", "stage", "file", "shape", "shape_match", "has_nan", "has_inf",
                  "readable", "status", "error"]

    # Merge with existing QC rows (from the earlier 5-run batch) instead of
    # overwriting -- never destroy prior derivatives' QC records.
    existing_rows = []
    if os.path.isfile(qc_csv):
        with open(qc_csv, newline="", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))

    new_rows = []
    for rows in qc_rows_list:
        for r in rows:
            for fn in fieldnames:
                r.setdefault(fn, "")
            new_rows.append(r)

    all_rows = existing_rows + new_rows
    with open(qc_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    n_pass_new = sum(1 for s in summary_rows_list if s["qc_status"] == "PASS")
    n_fail_new = len(summary_rows_list) - n_pass_new

    summary_json = os.path.join(qc_dir, "summary.json")
    existing_summary = {"runs": []}
    if os.path.isfile(summary_json):
        with open(summary_json) as f:
            existing_summary = js.load(f)

    all_runs = existing_summary.get("runs", []) + summary_rows_list
    n_pass_total = sum(1 for r in all_runs if r["qc_status"] == "PASS")
    n_fail_total = len(all_runs) - n_pass_total

    summary = {
        "total_runs_processed": len(all_runs),
        "qc_pass": n_pass_total,
        "qc_fail": n_fail_total,
        "smoothing_fwhm_mm": fwhm,
        "slice_timing_correction": "SKIPPED (not recoverable, confirmed by exhaustive audit)",
        "wm_csf_nuisance_regression": "UNAVAILABLE (no T1w)",
        "anatomical_processing": "NONE (BOLD-only pipeline, no T1w/recon-all)",
        "runs": all_runs,
    }
    with open(summary_json, "w") as f:
        js.dump(summary, f, indent=2)

    master_log = os.path.join(logs_dir, "preprocessing.log")
    with open(master_log, "w") as f:
        f.write(f"=== FS-FAST AD-GROUP PREPROCESSING (CUMULATIVE) ===\n")
        f.write(f"Last updated: {time.ctime()}\n")
        f.write(f"This batch: {len(summary_rows_list)} runs, PASS: {n_pass_new}, FAIL: {n_fail_new}\n")
        f.write(f"Cumulative total: {len(all_runs)} runs, PASS: {n_pass_total}, FAIL: {n_fail_total}\n\n")
        for s in all_runs:
            f.write(f"{s['sub']} {s['ses']} {s['run']}: QC={s['qc_status']} "
                    f"time={s['total_processing_time_s']}s max_trans={s['max_translation_mm']}mm "
                    f"errors={s['errors'] or 'none'}\n")

    return qc_csv, summary_json, master_log


def main():
    wf = pe.Workflow(name="fsfast_20ad_preprocessing", base_dir=WORK_DIR)

    infosource = pe.Node(IdentityInterface(fields=["run_key"]), name="infosource")
    infosource.iterables = [("run_key", [f"{sub}_{ses}" for sub, ses in RUNS])]

    splitkey = pe.Node(Function(input_names=["run_key"], output_names=["sub", "ses", "run"],
                                 function=split_key), name="splitkey")

    getpaths = pe.Node(Function(input_names=["sub", "ses", "run", "bids_root"],
                                 output_names=["bold_file", "json_file"], function=get_paths), name="getpaths")
    getpaths.inputs.bids_root = BIDS_ROOT

    precheck = pe.Node(Function(input_names=["sub", "ses", "run", "deriv_root"],
                                 output_names=["safe_to_write", "existing_path"],
                                 function=check_not_already_processed), name="precheck")
    precheck.inputs.deriv_root = DERIV_ROOT

    validate_n = pe.Node(Function(input_names=["bold_file", "json_file"],
                                   output_names=["ok", "shape", "tr_nifti", "tr_json", "err", "dt"],
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
        input_names=["sub", "ses", "run", "safe_to_write", "existing_path", "bold_file", "json_file",
                     "orig_shape", "orig_tr_nifti", "orig_tr_json", "validate_ok", "validate_err",
                     "moco_file", "mcdat_file", "moco_ok", "moco_err", "smooth_file", "smooth_ok",
                     "smooth_err", "design_file", "design_ncols", "final_file", "glmfit_ok", "glmfit_err",
                     "t_validate", "t_moco", "t_smooth", "t_design", "t_glmfit", "fwhm", "deriv_root"],
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
        (getpaths, validate_n, [("bold_file", "bold_file"), ("json_file", "json_file")]),
        (getpaths, moco, [("bold_file", "bold_file")]),
        (moco, smooth, [("moco_file", "moco_file")]),
        (moco, design, [("mcdat_file", "mcdat_file")]),
        (smooth, glmfit, [("smooth_file", "smooth_file")]),
        (design, glmfit, [("design_file", "design_file")]),

        (splitkey, finalize_n, [("sub", "sub"), ("ses", "ses"), ("run", "run")]),
        (precheck, finalize_n, [("safe_to_write", "safe_to_write"), ("existing_path", "existing_path")]),
        (getpaths, finalize_n, [("bold_file", "bold_file"), ("json_file", "json_file")]),
        (validate_n, finalize_n, [("shape", "orig_shape"), ("tr_nifti", "orig_tr_nifti"),
                                   ("tr_json", "orig_tr_json"), ("ok", "validate_ok"),
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
    print(f"=== Processing exactly these {len(RUNS)} runs ===")
    for sub, ses in RUNS:
        print(f"  {sub} {ses} run-01")
    main()
