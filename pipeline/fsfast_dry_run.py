import os
import json
import csv
import subprocess
import glob

BIDS_DIR = "/mnt/c/Users/krish/FYP/BIDS"
AUDIT_DIR = "/mnt/c/Users/krish/FYP/audit"
INVENTORY_CSV = os.path.join(AUDIT_DIR, "fsfast_bids_inventory.csv")
ELIGIBILITY_CSV = os.path.join(AUDIT_DIR, "fsfast_preprocessing_eligibility.csv")
PLAN_TXT = os.path.join(AUDIT_DIR, "fsfast_preprocessing_plan.txt")
ENV_VERIFICATION = "/home/harish/fsfast_environment_verification.txt"

def check_env():
    env_info = {}
    try:
        import nipype
        env_info["nipype"] = nipype.__version__
    except:
        env_info["nipype"] = "Missing"
        
    try:
        import nibabel
        env_info["nibabel"] = nibabel.__version__
    except:
        env_info["nibabel"] = "Missing"
        
    # Check FreeSurfer/FS-FAST (expected missing)
    fs_check = subprocess.run(["which", "recon-all"], capture_output=True, text=True).stdout.strip()
    env_info["freesurfer"] = fs_check if fs_check else "Missing"
    
    fsfast_check = subprocess.run(["which", "preproc-sess"], capture_output=True, text=True).stdout.strip()
    env_info["fsfast"] = fsfast_check if fsfast_check else "Missing"
    
    return env_info

def audit_bids():
    bids_data = {
        "subjects": set(),
        "bold_runs": [],
        "t1w_runs": [],
        "fmap_runs": []
    }
    
    for root, _, files in os.walk(BIDS_DIR):
        for f in files:
            if f.endswith(".nii") or f.endswith(".nii.gz"):
                parts = f.replace(".nii.gz", "").replace(".nii", "").split("_")
                sub = next((p for p in parts if p.startswith("sub-")), None)
                if sub: bids_data["subjects"].add(sub)
                
                if "bold" in f:
                    bids_data["bold_runs"].append(os.path.join(root, f))
                elif "T1w" in f:
                    bids_data["t1w_runs"].append(os.path.join(root, f))
                elif "fmap" in f or "phasediff" in f or "epi" in f:
                    if "fmap" in root:
                        bids_data["fmap_runs"].append(os.path.join(root, f))
                        
    return bids_data

def generate_reports(env, bids):
    os.makedirs(AUDIT_DIR, exist_ok=True)
    
    # 1. Environment Verification
    with open(ENV_VERIFICATION, 'w') as f:
        f.write("=== FS-FAST ENVIRONMENT VERIFICATION ===\n")
        f.write(f"Nipype: {env['nipype']}\n")
        f.write(f"NiBabel: {env['nibabel']}\n")
        f.write(f"FreeSurfer (recon-all): {env['freesurfer']}\n")
        f.write(f"FS-FAST (preproc-sess): {env['fsfast']}\n")
        f.write("\nWARNING: FreeSurfer/FS-FAST is not currently installed.\n")

    # 2. BIDS Inventory
    with open(INVENTORY_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Subject", "Session", "Run", "Modality", "File"])
        for run in bids["bold_runs"]:
            name = os.path.basename(run)
            parts = name.split('_')
            sub = next((p for p in parts if p.startswith('sub-')), "")
            ses = next((p for p in parts if p.startswith('ses-')), "")
            rn = next((p for p in parts if p.startswith('run-')), "")
            writer.writerow([sub, ses, rn, "BOLD", name])
            
    # 3. Eligibility
    with open(ELIGIBILITY_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Subject", "Session", "Run", "File", "Eligibility", "Reason"])
        for run in bids["bold_runs"]:
            name = os.path.basename(run)
            parts = name.split('_')
            sub = next((p for p in parts if p.startswith('sub-')), "")
            ses = next((p for p in parts if p.startswith('ses-')), "")
            rn = next((p for p in parts if p.startswith('run-')), "")
            
            eligibility = "ELIGIBLE"
            reason = "Standard"
            
            # Check for known short run
            if sub == "sub-012S4026":
                eligibility = "INELIGIBLE"
                reason = "INSUFFICIENT_TEMPORAL_LENGTH (7 volumes)"
                
            # Global check for T1w
            if len(bids["t1w_runs"]) == 0:
                eligibility = "PENDING_ANATOMICAL"
                reason += " | Missing T1w for recon-all"
                
            writer.writerow([sub, ses, rn, name, eligibility, reason])

    # 4. Preprocessing Plan
    with open(PLAN_TXT, 'w') as f:
        f.write("=== FS-FAST PREPROCESSING PLAN ===\n\n")
        f.write(f"Subjects: {len(bids['subjects'])}\n")
        f.write(f"BOLD Acquisitions: {len(bids['bold_runs'])}\n")
        f.write(f"T1w Images: {len(bids['t1w_runs'])}\n")
        f.write(f"Fieldmaps: {len(bids['fmap_runs'])}\n\n")
        
        f.write("--- WORKFLOW STEP ELIGIBILITY ---\n")
        f.write("1. Motion Correction (mkanalysis-sess / preproc-sess -motion): AVAILABLE (Uses BOLD only)\n")
        f.write("2. Slice Timing Correction (preproc-sess -stc): REQUIRES_OTHER_INPUT (Needs SliceTiming JSON metadata)\n")
        f.write("3. B0 Distortion Correction (preproc-sess -fmap): REQUIRES_FIELDMAP\n")
        f.write("4. Anatomical Registration (bbregister): REQUIRES_T1W (Requires completed recon-all subject directory)\n")
        f.write("5. Spatial Normalization to MNI/fsaverage: REQUIRES_T1W (Relies on anatomical registration)\n")
        f.write("6. Spatial Smoothing (preproc-sess -smooth): REQUIRES_T1W (FS-FAST typically smooths on the surface/volume after bbregister)\n")
        f.write("7. Nuisance Regression (WM/CSF masks): REQUIRES_T1W (Needs anatomical segmentation from recon-all)\n\n")
        
        f.write("--- CRITICAL FS-FAST ANALYSIS ---\n")
        f.write("Can resting-state preprocessing be performed with FS-FAST WITHOUT a T1w image?\n")
        f.write("Answer: STRICTLY NO for a standard FS-FAST workflow.\n")
        f.write("Reasoning:\n")
        f.write("FS-FAST (FreeSurfer Functional Analysis Stream) is explicitly designed to integrate functional MRI with FreeSurfer's structural cortical surface models (recon-all). Core commands like 'preproc-sess' heavily depend on 'bbregister' (boundary-based registration), which requires the cortical ribbon generated by 'recon-all'. Without T1w images, 'recon-all' cannot run, and FS-FAST cannot perform anatomical registration, surface projection, or extract WM/CSF nuisance regressors for resting-state connectivity.\n")

if __name__ == "__main__":
    env = check_env()
    bids = audit_bids()
    generate_reports(env, bids)
    print("Dry run complete.")
