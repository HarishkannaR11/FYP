import os
import csv
import subprocess
import numpy as np
import nibabel as nib

os.environ["FREESURFER_HOME"] = "/home/harish/freesurfer"
os.environ["FSFAST_HOME"] = "/home/harish/freesurfer/fsfast"
os.environ["SUBJECTS_DIR"] = "/home/harish/freesurfer/subjects"
os.environ["PATH"] = "/home/harish/freesurfer/bin:/home/harish/freesurfer/fsfast/bin:" + os.environ["PATH"]

from nipype.pipeline import engine as pe
from nipype.interfaces.utility import IdentityInterface, Function

WORK_DIR = "/home/harish/fyp_work/pilot2w/work"
BIDS_ROOT = "/mnt/c/Users/krish/FYP/BIDS"
DERIV_ROOT = "/mnt/c/Users/krish/FYP/derivatives/fsfast"

RUNS = [
    {"sub": "sub-002S0413", "ses": "ses-01", "run": "run-01"},
    {"sub": "sub-002S0685", "ses": "ses-01", "run": "run-01"},
]


def get_paths(sub, ses, run, bids_root):
    bold_file = f"{bids_root}/{sub}/{ses}/func/{sub}_{ses}_task-rest_{run}_bold.nii.gz"
    return bold_file


def validate(bold_file):
    import nibabel as nib
    import numpy as np
    img = nib.load(bold_file)
    data = img.get_fdata()
    shape = img.shape
    ok = (len(shape) == 4 and not np.isnan(data).any() and not np.isinf(data).any())
    return bool(ok), str(shape)


def run_moco(bold_file):
    import subprocess, os
    out = os.path.join(os.getcwd(), "moco_bold.nii.gz")
    mcdat = os.path.join(os.getcwd(), "moco_bold.mcdat")
    cmd = ["mc-afni2", "--i", bold_file, "--o", out, "--mcdat", mcdat]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"mc-afni2 failed: {res.stderr}")
    return out, mcdat


def run_smooth(moco_file):
    import subprocess, os
    out = os.path.join(os.getcwd(), "smooth_bold.nii.gz")
    cmd = ["mri_fwhm", "--i", moco_file, "--o", out, "--fwhm", "6", "--smooth-only"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"mri_fwhm failed: {res.stderr}")
    return out


def build_design(mcdat_file):
    import numpy as np, os
    mc = np.loadtxt(mcdat_file)
    n = mc.shape[0]
    motion = mc[:, 1:7]
    t = np.arange(n, dtype=float)
    t = (t - t.mean()) / t.std()
    X = np.column_stack([np.ones(n), t, t ** 2 - np.mean(t ** 2), motion])
    out = os.path.join(os.getcwd(), "design_matrix.txt")
    np.savetxt(out, X, fmt="%.6f")
    return out


def run_glmfit(smooth_file, design_file):
    import subprocess, os
    glmdir = os.path.join(os.getcwd(), "glmdir")
    cmd = ["mri_glmfit", "--y", smooth_file, "--X", design_file, "--no-contrasts-ok",
           "--no-mask", "--glmdir", glmdir, "--eres-save", "--nii.gz"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"mri_glmfit failed: {res.stderr}")
    final = os.path.join(glmdir, "eres.nii.gz")
    return final


def do_qc(sub, ses, run, bold_file, moco_file, smooth_file, final_file, mcdat_file):
    import nibabel as nib
    import numpy as np
    rows = []
    expect = None
    for stage, path in [("input_original_BIDS", bold_file), ("motion_corrected", moco_file),
                         ("smoothed", smooth_file), ("final_preproc_eres", final_file)]:
        row = {"sub": sub, "ses": ses, "run": run, "stage": stage, "file": path}
        try:
            img = nib.load(path)
            data = img.get_fdata()
            shape = img.shape
            if expect is None:
                expect = shape
            row["shape"] = str(shape)
            row["shape_match"] = "YES" if shape == expect else "NO"
            row["has_nan"] = str(bool(np.isnan(data).any()))
            row["has_inf"] = str(bool(np.isinf(data).any()))
            row["readable"] = "YES"
            row["status"] = "PASS" if row["has_nan"] == "False" and row["has_inf"] == "False" and row["shape_match"] == "YES" else "FAIL"
        except Exception as e:
            row["readable"] = "NO"
            row["status"] = "FAIL"
            row["error"] = str(e)
        rows.append(row)

    mc = np.loadtxt(mcdat_file)
    rows.append({
        "sub": sub, "ses": ses, "run": run, "stage": "motion_parameters", "file": mcdat_file,
        "shape": f"({mc.shape[0]} rows, {mc.shape[1]} cols)",
        "shape_match": "YES" if mc.shape[0] == expect[3] else "NO",
        "has_nan": str(bool(np.isnan(mc).any())), "has_inf": str(bool(np.isinf(mc).any())),
        "readable": "YES",
        "status": "PASS" if mc.shape[0] == expect[3] and not np.isnan(mc).any() else "FAIL",
    })
    max_trans = float(np.max(mc[:, 9]))
    return rows, max_trans


def copy_to_derivatives(sub, ses, run, final_file, mcdat_file, deriv_root):
    import shutil, os
    out_dir = os.path.join(deriv_root, sub, ses, "func")
    os.makedirs(out_dir, exist_ok=True)
    final_out = os.path.join(out_dir, f"{sub}_{ses}_task-rest_{run}_desc-preproc_bold.nii.gz")
    shutil.copy(final_file, final_out)
    motion_out = os.path.join(out_dir, f"{sub}_{ses}_task-rest_{run}_desc-motion_timeseries.tsv")
    with open(mcdat_file) as fin, open(motion_out, "w") as fout:
        fout.write("n_TR\troll\tpitch\tyaw\tdS\tdL\tdP\trmsold\trmsnew\ttrans_mm\n")
        for line in fin:
            fout.write("\t".join(line.split()) + "\n")
    return final_out


def collect_and_write(qc_rows_list, max_trans_list, sub_list):
    import csv
    QC_CSV = "/mnt/c/Users/krish/FYP/audit/fsfast_2worker_pilot_qc.csv"
    fieldnames = ["sub", "ses", "run", "stage", "file", "shape", "shape_match", "has_nan", "has_inf", "readable", "status", "error"]
    with open(QC_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rows in qc_rows_list:
            for r in rows:
                for fn in fieldnames:
                    r.setdefault(fn, "")
                writer.writerow(r)
    overall = all(r["status"] == "PASS" for rows in qc_rows_list for r in rows)
    return QC_CSV, overall


def main():
    wf = pe.Workflow(name="fsfast_2worker_pilot", base_dir=WORK_DIR)

    infosource = pe.Node(IdentityInterface(fields=["sub", "ses", "run"]), name="infosource")
    infosource.iterables = [("sub", [r["sub"] for r in RUNS])]
    # pair ses/run with sub via a lookup function instead of separate iterables (avoids cross-product)
    lookup = {r["sub"]: (r["ses"], r["run"]) for r in RUNS}

    def get_ses_run(sub):
        lookup = {"sub-002S0413": ("ses-01", "run-01"), "sub-002S0685": ("ses-01", "run-01")}
        return lookup[sub]

    getses = pe.Node(Function(input_names=["sub"], output_names=["ses", "run"], function=get_ses_run), name="getses")

    getpaths = pe.Node(Function(input_names=["sub", "ses", "run", "bids_root"],
                                 output_names=["bold_file"], function=get_paths), name="getpaths")
    getpaths.inputs.bids_root = BIDS_ROOT

    validate_n = pe.Node(Function(input_names=["bold_file"], output_names=["ok", "shape"],
                                   function=validate), name="validate")

    moco = pe.Node(Function(input_names=["bold_file"], output_names=["moco_file", "mcdat_file"],
                             function=run_moco), name="moco")

    smooth = pe.Node(Function(input_names=["moco_file"], output_names=["smooth_file"],
                               function=run_smooth), name="smooth")

    design = pe.Node(Function(input_names=["mcdat_file"], output_names=["design_file"],
                               function=build_design), name="design")

    glmfit = pe.Node(Function(input_names=["smooth_file", "design_file"], output_names=["final_file"],
                               function=run_glmfit), name="glmfit")

    qc = pe.Node(Function(input_names=["sub", "ses", "run", "bold_file", "moco_file", "smooth_file",
                                        "final_file", "mcdat_file"],
                           output_names=["qc_rows", "max_trans"], function=do_qc), name="qc")

    copyderiv = pe.Node(Function(input_names=["sub", "ses", "run", "final_file", "mcdat_file", "deriv_root"],
                                  output_names=["final_out"], function=copy_to_derivatives), name="copyderiv")
    copyderiv.inputs.deriv_root = DERIV_ROOT

    collector = pe.JoinNode(Function(input_names=["qc_rows_list", "max_trans_list", "sub_list"],
                                      output_names=["qc_csv", "overall"], function=collect_and_write),
                             name="collector", joinsource="infosource", joinfield=["qc_rows_list", "max_trans_list", "sub_list"])

    wf.connect([
        (infosource, getses, [("sub", "sub")]),
        (infosource, getpaths, [("sub", "sub")]),
        (getses, getpaths, [("ses", "ses"), ("run", "run")]),
        (getpaths, validate_n, [("bold_file", "bold_file")]),
        (getpaths, moco, [("bold_file", "bold_file")]),
        (moco, smooth, [("moco_file", "moco_file")]),
        (moco, design, [("mcdat_file", "mcdat_file")]),
        (smooth, glmfit, [("smooth_file", "smooth_file")]),
        (design, glmfit, [("design_file", "design_file")]),
        (infosource, qc, [("sub", "sub")]),
        (getses, qc, [("ses", "ses"), ("run", "run")]),
        (getpaths, qc, [("bold_file", "bold_file")]),
        (moco, qc, [("moco_file", "moco_file"), ("mcdat_file", "mcdat_file")]),
        (smooth, qc, [("smooth_file", "smooth_file")]),
        (glmfit, qc, [("final_file", "final_file")]),
        (infosource, copyderiv, [("sub", "sub")]),
        (getses, copyderiv, [("ses", "ses"), ("run", "run")]),
        (glmfit, copyderiv, [("final_file", "final_file")]),
        (moco, copyderiv, [("mcdat_file", "mcdat_file")]),
        (qc, collector, [("qc_rows", "qc_rows_list"), ("max_trans", "max_trans_list")]),
        (infosource, collector, [("sub", "sub_list")]),
    ])

    result = wf.run(plugin="MultiProc", plugin_args={"n_procs": 2})
    print("WORKFLOW_DONE")


if __name__ == "__main__":
    main()
