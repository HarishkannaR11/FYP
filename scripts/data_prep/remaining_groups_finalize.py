import os
import csv
import zipfile
import tempfile
import shutil
import subprocess
import re
import argparse
from datetime import datetime
import nibabel as nib

BASE_DIR = r"C:\Users\krish\FYP"
NIFTI_ORIG = os.path.join(BASE_DIR, r"NIfTI\NIfTI")
RECOVERED_DIR = os.path.join(BASE_DIR, "recovered_fmri")
RAW_DIR = os.path.join(BASE_DIR, "raw_data")
AUDIT_DIR = os.path.join(BASE_DIR, "audit")

TARGET_GROUPS = ["AD", "EMCI", "LMCI", "CN_Final"]
LMCI_RAW_ZIP = os.path.join(RAW_DIR, "LMCI.zip")
DCM2NIIX_PATH = r"C:\Users\krish\FYP\tools\dcm2niix.exe"

DRY_RUN_CSV = os.path.join(AUDIT_DIR, "remaining_groups_merge_dry_run.csv")
DRY_RUN_SUMMARY = os.path.join(AUDIT_DIR, "remaining_groups_merge_summary.txt")
FINAL_CSV = os.path.join(AUDIT_DIR, "final_dataset_verification.csv")
FINAL_SUMMARY = os.path.join(AUDIT_DIR, "final_dataset_verification_summary.txt")

def format_datetime(dt_str):
    if not dt_str: return ""
    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S.%f")
        return dt.strftime("%Y%m%d%H%M%S")
    except:
        pass
    try:
        dt = datetime.strptime(dt_str, "%Y-%m-%dT%H:%M:%S")
        return dt.strftime("%Y%m%d%H%M%S")
    except:
        return re.sub(r'[^0-9]', '', dt_str)

def extract_dicom_series(zip_path, internal_dir, temp_dir):
    extracted_files = []
    with zipfile.ZipFile(zip_path, 'r') as z:
        for info in z.infolist():
            if info.filename.startswith(internal_dir + "/") and info.filename.endswith(".dcm"):
                z.extract(info, temp_dir)
                extracted_files.append(os.path.join(temp_dir, info.filename))
    return os.path.join(temp_dir, internal_dir)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Perform actual copy and convert operations")
    args = parser.parse_args()

    print("========================================================")
    print(f"FINAL DATASET COMPLETION - {'EXECUTE' if args.execute else 'DRY RUN'}")
    print("========================================================")

    # 1. Map existing JSONs and NIfTIs in primary dataset
    json_map = {}
    existing_nifti_map = {}
    
    for g in TARGET_GROUPS:
        orig_group_dir = os.path.join(NIFTI_ORIG, g)
        if not os.path.exists(orig_group_dir):
            continue
        for root, _, files in os.walk(orig_group_dir):
            for f in files:
                if f.endswith(".json"):
                    stem = f[:-5]
                    json_map[stem] = {
                        "group": g,
                        "path": os.path.join(root, f),
                        "dir": root,
                        "filename": f,
                        "subject_id": os.path.basename(root)
                    }
                elif f.endswith(".nii") or f.endswith(".nii.gz"):
                    stem = f.replace(".nii.gz", "").replace(".nii", "")
                    existing_nifti_map[stem] = os.path.join(root, f)

    # 2. Map recovered NIfTIs
    recovered_map = {}
    for g in TARGET_GROUPS:
        rec_group_dir = os.path.join(RECOVERED_DIR, g)
        if not os.path.exists(rec_group_dir):
            continue
        for root, _, files in os.walk(rec_group_dir):
            for f in files:
                if f.endswith(".nii"):
                    stem = f[:-4]
                    recovered_map[stem] = {
                        "group": g,
                        "path": os.path.join(root, f)
                    }

    # 3. Build Action List
    results = []
    group_stats = {g: {
        "Total JSON": 0,
        "Existing NIfTI": 0,
        "Recovered and ready to merge": 0,
        "Requires LMCI conversion": 0,
        "RAW Unavailable (AD)": 0
    } for g in TARGET_GROUPS}

    for stem, j_info in json_map.items():
        g = j_info["group"]
        group_stats[g]["Total JSON"] += 1
        
        row = {
            "group": g,
            "subject_id": j_info["subject_id"],
            "json_stem": stem,
            "json_path": j_info["path"],
            "source_nii": "",
            "target_dir": j_info["dir"],
            "action": ""
        }

        if stem in existing_nifti_map:
            row["action"] = "EXISTING_NIFTI"
            group_stats[g]["Existing NIfTI"] += 1
        elif stem in recovered_map:
            row["source_nii"] = recovered_map[stem]["path"]
            row["action"] = "RECOVERED_READY_TO_MERGE"
            group_stats[g]["Recovered and ready to merge"] += 1
        else:
            if g == "LMCI":
                row["action"] = "REQUIRES_NEW_CONVERSION"
                group_stats[g]["Requires LMCI conversion"] += 1
            elif g == "AD":
                row["action"] = "RAW_UNAVAILABLE"
                group_stats[g]["RAW Unavailable (AD)"] += 1
            else:
                row["action"] = "UNEXPECTED_MISSING"
                
        results.append(row)

    if not args.execute:
        # Generate Dry Run Reports
        with open(DRY_RUN_CSV, 'w', newline='', encoding='utf-8') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
                
        with open(DRY_RUN_SUMMARY, 'w', encoding='utf-8') as f:
            f.write("--- REMAINING GROUPS DRY RUN SUMMARY ---\n\n")
            for g in TARGET_GROUPS:
                f.write(f"{g}\n")
                f.write("-" * len(g) + "\n")
                for k, v in group_stats[g].items():
                    f.write(f"{k}: {v}\n")
                f.write("\n")
                
        print(f"Dry run complete. Check {DRY_RUN_SUMMARY}")

    else:
        # EXECUTE COPY & CONVERSION
        print("Executing copy of recovered files...")
        copied = 0
        converted = 0
        
        # 1. Copy Recovered
        for r in results:
            if r["action"] == "RECOVERED_READY_TO_MERGE":
                src = r["source_nii"]
                dst = os.path.join(r["target_dir"], r["json_stem"] + ".nii")
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    copied += 1

        print(f"Copied {copied} files.")

        # 2. Convert LMCI
        print("Executing LMCI conversions...")
        lmci_to_convert = [r for r in results if r["action"] == "REQUIRES_NEW_CONVERSION"]
        
        if lmci_to_convert:
            # We need to map RAW DICOMs for these 4 LMCI scans
            print("Mapping RAW DICOMs for LMCI...")
            raw_map = {}
            with zipfile.ZipFile(LMCI_RAW_ZIP, 'r') as z:
                dirs = set([os.path.dirname(n) for n in z.namelist() if n.endswith('.dcm')])
                for d in dirs:
                    parts = d.split('/')
                    if len(parts) >= 5 and 'ADNI' in parts[0]:
                        subj = parts[1]
                        desc = parts[2]
                        dt = parts[3]
                        fmt_dt = re.sub(r'[^0-9]', '', dt)
                        if len(fmt_dt) == 15 and fmt_dt.endswith('0'):
                            fmt_dt = fmt_dt[:-1]
                        
                        key = f"{subj}_{fmt_dt}"
                        if key not in raw_map:
                            raw_map[key] = []
                        raw_map[key].append({"dir": d, "desc": desc})
                        
            # We need to extract datetime from the json metadata of these 4 files
            for r in lmci_to_convert:
                stem = r["json_stem"]
                subj = r["subject_id"]
                target_dir = r["target_dir"]
                target_nii = os.path.join(target_dir, stem + ".nii")
                
                acq_dt = ""
                try:
                    with open(r["json_path"], 'r') as jp:
                        jd = json.load(jp)
                        dt = jd.get("AcquisitionDateTime", "")
                        acq_dt = format_datetime(dt)
                except:
                    pass
                
                if not acq_dt:
                    # try audit_metadata (if we had it), but they are known from previous dry run
                    # we can also just parse from filename because it's standard 
                    # 130_S_4250_WIP_Resting_State_fMRI_20120411101533_601 -> 20120411101533
                    m = re.search(r'_(\d{14})_', stem)
                    if m:
                        acq_dt = m.group(1)
                
                key = f"{subj}_{acq_dt}"
                if key in raw_map:
                    best = raw_map[key][0]
                    temp_dir = tempfile.mkdtemp()
                    try:
                        ext_path = extract_dicom_series(LMCI_RAW_ZIP, best["dir"], temp_dir)
                        cmd = [DCM2NIIX_PATH, "-z", "n", "-b", "n", "-f", stem, "-o", target_dir, ext_path]
                        subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                        
                        if not os.path.exists(target_nii):
                            possible = [f for f in os.listdir(target_dir) if f.startswith(stem) and f.endswith(".nii")]
                            if possible:
                                os.rename(os.path.join(target_dir, possible[0]), target_nii)
                                
                        if os.path.exists(target_nii):
                            converted += 1
                    finally:
                        shutil.rmtree(temp_dir, ignore_errors=True)
                        
        print(f"Converted {converted} LMCI files.")

        # 3. Final Verification
        print("Executing Final Verification...")
        final_stats = {
            "AD": {"Total": 0, "EXISTING_NIFTI": 0, "RECOVERED_AND_MERGED": 0, "NEWLY_CONVERTED_LMCI": 0, "RAW_UNAVAILABLE": 0, "MAPPING_ERROR": 0},
            "EMCI": {"Total": 0, "EXISTING_NIFTI": 0, "RECOVERED_AND_MERGED": 0, "NEWLY_CONVERTED_LMCI": 0, "RAW_UNAVAILABLE": 0, "MAPPING_ERROR": 0},
            "LMCI": {"Total": 0, "EXISTING_NIFTI": 0, "RECOVERED_AND_MERGED": 0, "NEWLY_CONVERTED_LMCI": 0, "RAW_UNAVAILABLE": 0, "MAPPING_ERROR": 0},
            "CN_Final": {"Total": 0, "EXISTING_NIFTI": 0, "RECOVERED_AND_MERGED": 0, "NEWLY_CONVERTED_LMCI": 0, "RAW_UNAVAILABLE": 0, "MAPPING_ERROR": 0},
        }

        final_rows = []
        for stem, j_info in json_map.items():
            g = j_info["group"]
            final_stats[g]["Total"] += 1
            
            expected_nii = os.path.join(j_info["dir"], stem + ".nii")
            expected_nii_gz = os.path.join(j_info["dir"], stem + ".nii.gz")
            
            row = {
                "group": g,
                "json_stem": stem,
                "status": ""
            }
            
            if os.path.exists(expected_nii_gz) or (os.path.exists(expected_nii) and stem in existing_nifti_map):
                row["status"] = "EXISTING_NIFTI"
                final_stats[g]["EXISTING_NIFTI"] += 1
            elif os.path.exists(expected_nii):
                # Validate
                try:
                    img = nib.load(expected_nii)
                    if len(img.shape) >= 4:
                        if g == "LMCI" and any(r["json_stem"] == stem for r in lmci_to_convert):
                            row["status"] = "NEWLY_CONVERTED_LMCI"
                            final_stats[g]["NEWLY_CONVERTED_LMCI"] += 1
                        else:
                            row["status"] = "RECOVERED_AND_MERGED"
                            final_stats[g]["RECOVERED_AND_MERGED"] += 1
                    else:
                        row["status"] = "MAPPING_ERROR"
                        final_stats[g]["MAPPING_ERROR"] += 1
                except:
                    row["status"] = "MAPPING_ERROR"
                    final_stats[g]["MAPPING_ERROR"] += 1
            else:
                if g == "AD":
                    row["status"] = "RAW_UNAVAILABLE"
                    final_stats[g]["RAW_UNAVAILABLE"] += 1
                else:
                    row["status"] = "MAPPING_ERROR"
                    final_stats[g]["MAPPING_ERROR"] += 1
                    
            final_rows.append(row)

        with open(FINAL_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=final_rows[0].keys())
            writer.writeheader()
            writer.writerows(final_rows)
            
        with open(FINAL_SUMMARY, 'w', encoding='utf-8') as f:
            f.write("--- FINAL DATASET VERIFICATION SUMMARY ---\n\n")
            for g in TARGET_GROUPS:
                f.write(f"{g}\n")
                f.write("-" * len(g) + "\n")
                for k, v in final_stats[g].items():
                    f.write(f"{k}: {v}\n")
                f.write("\n")
                
        print("Done. Check audit/final_dataset_verification_summary.txt")

if __name__ == "__main__":
    main()
