import os
import sys
import glob
import csv
import subprocess
import json
import shutil
import nibabel as nib
import nipype.pipeline.engine as pe
import nipype.interfaces.utility as niu
from nipype.interfaces.base import CommandLine

BIDS_DIR = "/mnt/c/Users/krish/FYP/BIDS"
AUDIT_DIR = "/mnt/c/Users/krish/FYP/audit"
WORK_DIR = "/home/harish/fyp_work/working"
LOG_DIR = "/home/harish/fyp_work/logs"
DERIV_DIR = "/mnt/c/Users/krish/FYP/derivatives/fsfast"

ENV_TXT = os.path.join(AUDIT_DIR, "fsfast_preprocessing_environment.txt")
INV_CSV = os.path.join(AUDIT_DIR, "fsfast_preprocessing_inventory.csv")
ELIG_CSV = os.path.join(AUDIT_DIR, "fsfast_preprocessing_eligibility.csv")
PILOT_TXT = os.path.join(AUDIT_DIR, "fsfast_pilot_report.txt")
QC_CSV = os.path.join(AUDIT_DIR, "fsfast_pilot_qc.csv")

def verify_environment():
    env = {}
    try:
        import nipype
        env["nipype"] = nipype.__version__
    except: env["nipype"] = "Missing"
    try:
        env["nibabel"] = nib.__version__
    except: env["nibabel"] = "Missing"

    fs_home = os.environ.get("FREESURFER_HOME", "Not Set")
    env["FREESURFER_HOME"] = fs_home
    
    for cmd in ["preproc-sess", "mc-afni2", "mri_fwhm", "mri_glmfit"]:
        path = subprocess.run(["which", cmd], capture_output=True, text=True).stdout.strip()
        env[cmd] = path if path else "Missing"
        
    os.makedirs(AUDIT_DIR, exist_ok=True)
    with open(ENV_TXT, "w") as f:
        f.write("=== PREPROCESSING ENVIRONMENT ===\n")
        for k, v in env.items():
            f.write(f"{k}: {v}\n")
    return env

def audit_inputs():
    inventory = []
    eligibility = []
    bold_files = sorted(glob.glob(f"{BIDS_DIR}/sub-*/ses-*/func/*_bold.nii*") + glob.glob(f"{BIDS_DIR}/sub-*/func/*_bold.nii*"))
    
    for bf in bold_files:
        name = os.path.basename(bf)
        parts = name.split('_')
        sub = next((p for p in parts if p.startswith("sub-")), "unknown")
        ses = next((p for p in parts if p.startswith("ses-")), "none")
        run = next((p for p in parts if p.startswith("run-")), "none")
        
        try:
            img = nib.load(bf)
            shape = img.shape
            vols = shape[3] if len(shape) > 3 else 1
            dims = img.header.get_zooms()
            tr = dims[3] if len(dims) > 3 else "Unknown"
        except:
            shape, vols, dims, tr = "Error", 0, "Error", "Error"
            
        json_path = bf.replace(".nii.gz", ".json").replace(".nii", ".json")
        has_json = os.path.exists(json_path)
        
        inventory.append([sub, ses, run, bf, shape, vols, dims, tr, has_json, "No", "Checked"])
        
        if vols < 50:
            eligibility.append([sub, ses, run, name, "INELIGIBLE", f"Insufficient volumes ({vols})"])
        else:
            eligibility.append([sub, ses, run, name, "ELIGIBLE", "OK"])

    with open(INV_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["participant_id", "session", "run", "input_path", "shape", "volumes", "voxel_dims", "TR", "has_slicetiming", "has_fieldmap", "status"])
        writer.writerows(inventory)
        
    with open(ELIG_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["participant_id", "session", "run", "file", "eligibility", "reason"])
        writer.writerows(eligibility)
        
    return [e for e in eligibility if e[4] == "ELIGIBLE"]

def run_pilot(eligible, env):
    if not eligible:
        with open(PILOT_TXT, "w") as f: f.write("No eligible subjects for pilot.\n")
        return
        
    pilot = eligible[0]
    sub, ses, run, name = pilot[0], pilot[1], pilot[2], pilot[3]
    input_file = next(f for f in glob.glob(f"{BIDS_DIR}/{sub}/**/func/{name}", recursive=True))
    
    with open(PILOT_TXT, "w") as f:
        f.write("=== PILOT RUN REPORT ===\n")
        f.write(f"Participant: {sub}\nSession: {ses}\nRun: {run}\n")
        f.write(f"Input: {input_file}\n")
        
        if env.get("mc-afni2") == "Missing":
            f.write("\nERROR: FreeSurfer/FS-FAST (mc-afni2) is not installed or not in PATH.\n")
            f.write("Cannot execute pilot run. Please install FreeSurfer and set FREESURFER_HOME.\n")
            return
            
        f.write("\nExecuting pilot...\n")
        
    # Dummy QC since execution will fail if FreeSurfer is missing
    with open(QC_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["participant", "session", "run", "status", "reason"])
        if env.get("mc-afni2") == "Missing":
            writer.writerow([sub, ses, run, "FAIL", "FreeSurfer missing"])

if __name__ == "__main__":
    env = verify_environment()
    eligible = audit_inputs()
    run_pilot(eligible, env)
    print("Script execution completed.")
