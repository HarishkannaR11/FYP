import os
import json

BIDS_DIR = r"C:\Users\krish\FYP\BIDS"
AUDIT_DIR = r"C:\Users\krish\FYP\audit"
REPORT_TXT = os.path.join(AUDIT_DIR, "bold_only_fsfast_report.txt")

def main():
    print("Generating BOLD-only FS-FAST Nipype Dry Run Report...")

    report_content = """=== BOLD-ONLY FS-FAST + NIPYPE PREPROCESSING PLAN ===

--- 1. FS-FAST COMMANDS TO BE USED ---
Because T1w images are missing, standard 'preproc-sess' (which expects a valid FreeSurfer 'recon-all' subject hierarchy) cannot be used in its default form. Furthermore, FSL (MCFLIRT, FLIRT, etc.) is strictly prohibited. 
We will use native FreeSurfer/FS-FAST binaries via Nipype to process the 4D BOLD data directly in native space:

- Motion Correction: 'mri_robust_register' or 'mri_vol2vol' (FreeSurfer's native rigid-body registration to a reference volume, bypassing FSL).
- Spatial Smoothing: 'mri_fwhm' or 'mri_smooth' (FreeSurfer's native spatial smoothing).
- Temporal Filtering/Detrending: 'mri_glmfit' (Using polynomials/fourier regressors for high-pass/low-pass filtering in native space).
- Nuisance Regression: Without T1w, we cannot generate accurate subject-specific White Matter or CSF masks using FreeSurfer. We will rely solely on motion parameters (extracted from mri_robust_register) and potentially global signal regression if desired, performed via 'mri_glmfit'.

--- 2. NIPYPE NODES & INTERFACES ---
- nipype.interfaces.freesurfer.RobustRegister (Motion Correction)
- nipype.interfaces.freesurfer.Smooth (Spatial Smoothing)
- nipype.interfaces.freesurfer.GLMFit (Temporal Filtering / Nuisance Regression)
- nipype.interfaces.freesurfer.MRIConvert (Format conversion if needed)

--- 3. REQUIRED INPUTS ---
- Functional Data: 4D NIfTI BOLD files (*_bold.nii or *_bold.nii.gz).

--- 4. UNAVAILABLE INPUTS & HANDLING ---
- T1w Anatomical: Unavailable. Handled by operating strictly in functional native space. Anatomical registration (bbregister) is OMITTED.
- Fieldmaps: Unavailable. Handled by omitting B0 distortion correction.
- Slice Timing JSON: Unreliable/missing. Handled by omitting slice-timing correction to prevent interpolating over incorrect temporal spacing.

--- 5. EXPECTED OUTPUTS ---
- <prefix>_desc-moco_bold.nii.gz (Motion Corrected)
- <prefix>_desc-smooth_bold.nii.gz (Smoothed)
- <prefix>_desc-filtered_bold.nii.gz (Filtered & Nuisance Regressed)
- <prefix>_motion_params.txt (Rigid body transforms)

--- 6. TRACEABILITY (SUBJECT/SESSION/RUN) ---
Nipype's `IdentityInterface` or `BIDSDataGrabber` will be used to loop over exactly:
participant_id -> session -> run.
The output filenames will preserve the exact BIDS entities to guarantee 100% strict traceability.

--- 7. HANDLING OF sub-012S4026 (7 volumes) ---
This run is explicitly removed from the Nipype iterable node. It is mathematically impossible to temporally filter or properly regress nuisance signals from 7 volumes. It will be logged in `excluded_runs.csv` and bypassed in the workflow.

--- 8. OUTPUT DIRECTORY STRUCTURE ---
/home/harish/fyp_work/working/ (Nipype Crash/Working Cache - WSL)
/home/harish/fyp_work/logs/ (Nipype Execution Logs - WSL)
C:\\Users\\krish\\FYP\\derivatives\\fsfast\\ (Final Windows-Accessible Output)
  └── sub-XXXX\\
       └── ses-XX\\
            └── func\\
                 ├── sub-XXXX_ses-XX_task-rest_run-XX_desc-preproc_bold.nii.gz
                 └── sub-XXXX_ses-XX_task-rest_run-XX_desc-motion_timeseries.tsv

--- 9. QC / VALIDATION STRATEGY ---
A custom Nipype `Function` node will run post-processing QC:
- Verifies output NIfTI readability using NiBabel.
- Checks shape (must match original X,Y,Z and T).
- Verifies absence of NaNs/Infs.
- Produces a final `audit/preprocessing_qc_report.csv` detailing SUCCESS/FAILURE for all 174 eligible runs.
"""

    os.makedirs(AUDIT_DIR, exist_ok=True)
    with open(REPORT_TXT, "w", encoding="utf-8") as f:
        f.write(report_content)

    print(f"\nReport written to {REPORT_TXT}")

if __name__ == "__main__":
    main()
