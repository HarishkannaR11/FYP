import os
import json
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

DRY_RUN_CSV = os.path.join(AUDIT_DIR, "remaining_groups_recovery_dry_run.csv")
DRY_RUN_SUMMARY = os.path.join(AUDIT_DIR, "remaining_groups_recovery_summary.txt")

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

def is_resting_state_fmri(desc):
    desc_lower = desc.lower()
    if 'rest' in desc_lower and 'fmri' in desc_lower: return True
    if 'rs-fmri' in desc_lower or 'rsfmri' in desc_lower: return True
    if 'resting' in desc_lower: return True
    if 'extended resting state' in desc_lower: return True
    if 't1' in desc_lower or 't2' in desc_lower or 'flair' in desc_lower or 'dti' in desc_lower or 'field_mapping' in desc_lower or 'cal' in desc_lower or 'localizer' in desc_lower:
        return False
    return False

def validate_nifti(nifti_path):
    try:
        img = nib.load(nifti_path)
        shape = img.shape
        zooms = img.header.get_zooms()
        TR = zooms[3] if len(zooms) > 3 else "N/A"
        timepoints = shape[3] if len(shape) > 3 else "N/A"
        is_4d = len(shape) >= 4 and timepoints != "N/A" and timepoints > 1
        return is_4d, f"{shape}", f"{timepoints}", f"{zooms}", str(TR), ""
    except Exception as e:
        return False, "ERROR", "ERROR", "ERROR", "ERROR", str(e)

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
    parser.add_argument("--execute_conversion", action="store_true", help="Perform actual conversion")
    parser.add_argument("--execute_merge", action="store_true", help="Perform actual merge into dataset")
    args = parser.parse_args()

    print("========================================================")
    print(f"REMAINING GROUPS RECOVERY - {'CONVERSION' if args.execute_conversion else ('MERGE' if args.execute_merge else 'DRY RUN')}")
    print("========================================================")
    
    # Load fallback metadata
    audit_metadata = {}
    audit_csv_path = os.path.join(AUDIT_DIR, "data_audit.csv")
    if os.path.exists(audit_csv_path):
        with open(audit_csv_path, 'r', encoding='utf-8') as f:
            for row in csv.DictReader(f):
                fname = row.get('filename', '')
                if fname:
                    audit_metadata[fname] = row
                    audit_metadata[fname.replace('.json', '')] = row

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
                    
                    json_path = os.path.join(root, f)
                    subj_id = ""
                    acq_dt = ""
                    scan_desc = ""
                    
                    try:
                        with open(json_path, 'r') as jp:
                            jdata = json.load(jp)
                            subj_id = jdata.get("PatientID", jdata.get("PatientName", ""))
                            dt = jdata.get("AcquisitionDateTime", "")
                            acq_dt = format_datetime(dt)
                            scan_desc = jdata.get("SeriesDescription", "")
                    except:
                        pass
                    
                    if f in audit_metadata:
                        row = audit_metadata[f]
                        if not subj_id: subj_id = row.get('subject_id', '')
                        if not acq_dt: acq_dt = row.get('acquisition_datetime', '')
                        if not scan_desc: scan_desc = row.get('series_description', '')
                    
                    if not subj_id:
                        m = re.search(r'(\d{3}_S_\d{4})', f)
                        if m: subj_id = m.group(1)
                        
                    json_map[stem] = {
                        "group": g,
                        "path": json_path,
                        "dir": root,
                        "filename": f,
                        "subject_id": subj_id,
                        "acquisition_datetime": acq_dt,
                        "scan_description": scan_desc
                    }
                elif f.endswith(".nii") or f.endswith(".nii.gz"):
                    stem = f.replace(".nii.gz", "").replace(".nii", "")
                    existing_nifti_map[stem] = "EXISTING"

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
                    recovered_map[stem] = "RECOVERED"

    # 3. Match missing acquisitions
    acquisitions = []
    missing_stems = []
    
    for stem, j_info in json_map.items():
        if stem in existing_nifti_map:
            status = "EXISTING_NIFTI"
        elif stem in recovered_map:
            status = "RECOVERED_NIFTI"
        else:
            status = "MISSING_NIFTI"
            missing_stems.append(stem)
            
        j_info['status'] = status
        acquisitions.append(j_info)

    print(f"Found {len(missing_stems)} missing acquisitions for AD, EMCI, LMCI, CN_Final.")

    # 4. Search RAW ADNI for missing
    raw_map = {}
    if missing_stems:
        print("Scanning raw archives...")
        for root, _, files in os.walk(RAW_DIR):
            for f in files:
                if f.endswith(".zip"):
                    zip_path = os.path.join(root, f)
                    try:
                        with zipfile.ZipFile(zip_path, 'r') as z:
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
                                    raw_map[key].append({
                                        "zip_path": zip_path,
                                        "internal_dir": d,
                                        "series_desc": desc
                                    })
                    except Exception as e:
                        pass
                        
    # 5. Process matches and dry run counts
    results = []
    group_stats = {}
    for g in TARGET_GROUPS:
        group_stats[g] = {
            "Total JSON acquisitions": 0,
            "Existing NIfTI": 0,
            "Already recovered NIfTI": 0,
            "Missing NIfTI": 0,
            "Raw DICOM available": 0,
            "Raw DICOM unavailable": 0,
            "Ambiguous": 0,
            "Proposed for conversion": 0
        }

    for j_info in acquisitions:
        g = j_info['group']
        group_stats[g]["Total JSON acquisitions"] += 1
        
        row = {
            "group": g,
            "subject_id": j_info['subject_id'],
            "json_filename": j_info['filename'],
            "acquisition_datetime": j_info['acquisition_datetime'],
            "matching_nii": "",
            "status": j_info['status'],
            "raw_zip": "",
            "raw_dir": ""
        }
        
        if j_info['status'] == "EXISTING_NIFTI":
            group_stats[g]["Existing NIfTI"] += 1
            row["matching_nii"] = "EXISTING"
        elif j_info['status'] == "RECOVERED_NIFTI":
            group_stats[g]["Already recovered NIfTI"] += 1
            row["matching_nii"] = "RECOVERED"
        else:
            group_stats[g]["Missing NIfTI"] += 1
            
            # Find RAW DICOM
            subj = j_info['subject_id']
            acq_dt = j_info['acquisition_datetime']
            key = f"{subj}_{acq_dt}"
            
            if not subj or not acq_dt:
                row["status"] = "AMBIGUOUS"
                group_stats[g]["Ambiguous"] += 1
            elif key in raw_map:
                cands = raw_map[key]
                best = cands[0]
                for c in cands:
                    if j_info['scan_description'] and j_info['scan_description'] in c['series_desc']:
                        best = c
                        break
                        
                row["raw_zip"] = best["zip_path"]
                row["raw_dir"] = best["internal_dir"]
                
                if not is_resting_state_fmri(best['series_desc']) and not is_resting_state_fmri(j_info['scan_description']):
                    row["status"] = "NOT_FMRI"
                    group_stats[g]["Raw DICOM unavailable"] += 1
                else:
                    row["status"] = "MISSING_NIFTI_RAW_AVAILABLE"
                    group_stats[g]["Raw DICOM available"] += 1
                    group_stats[g]["Proposed for conversion"] += 1
            else:
                row["status"] = "MISSING_NIFTI_RAW_NOT_FOUND"
                group_stats[g]["Raw DICOM unavailable"] += 1
                
        results.append(row)

    if not args.execute_conversion and not args.execute_merge:
        with open(DRY_RUN_CSV, 'w', newline='', encoding='utf-8') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
                
        with open(DRY_RUN_SUMMARY, 'w', encoding='utf-8') as f:
            for g in TARGET_GROUPS:
                f.write(f"{g}\n----\n")
                for k, v in group_stats[g].items():
                    f.write(f"{k}: {v}\n")
                f.write("\n")
                
        print("Dry run complete. Check audit/remaining_groups_recovery_dry_run_summary.txt")

    elif args.execute_conversion:
        # Phase 6: Conversion
        print("Executing conversion...")
        converted = 0
        failed = 0
        for r in results:
            if r["status"] == "MISSING_NIFTI_RAW_AVAILABLE":
                g = r["group"]
                subj = r["subject_id"]
                stem = r["json_filename"][:-5]
                target_dir = os.path.join(RECOVERED_DIR, g, subj)
                os.makedirs(target_dir, exist_ok=True)
                target_nii = os.path.join(target_dir, stem + ".nii")
                
                temp_dir = tempfile.mkdtemp()
                try:
                    ext_path = extract_dicom_series(r["raw_zip"], r["raw_dir"], temp_dir)
                    dcm2niix_path = r"C:\Users\krish\FYP\tools\dcm2niix.exe"
                    cmd = [dcm2niix_path, "-z", "n", "-b", "n", "-f", stem, "-o", target_dir, ext_path]
                    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    
                    if not os.path.exists(target_nii):
                        possible = [f for f in os.listdir(target_dir) if f.startswith(stem) and f.endswith(".nii")]
                        if possible:
                            os.rename(os.path.join(target_dir, possible[0]), target_nii)
                            
                    if os.path.exists(target_nii):
                        is_4d, dims, tps, vox, tr, err = validate_nifti(target_nii)
                        if is_4d:
                            converted += 1
                        else:
                            os.remove(target_nii)
                            failed += 1
                    else:
                        failed += 1
                except Exception as e:
                    failed += 1
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    
        print(f"Conversion complete. Successfully converted: {converted}, Failed: {failed}")
        
    elif args.execute_merge:
        # Phase 7: Merge
        print("Executing merge...")
        copied = 0
        for stem, j_info in json_map.items():
            expected_nii = os.path.join(j_info["dir"], stem + ".nii")
            expected_nii_gz = os.path.join(j_info["dir"], stem + ".nii.gz")
            
            if os.path.exists(expected_nii) or os.path.exists(expected_nii_gz):
                continue # Already present
                
            g = j_info["group"]
            subj = j_info["subject_id"]
            recovered_nii = os.path.join(RECOVERED_DIR, g, subj, stem + ".nii")
            
            if os.path.exists(recovered_nii):
                if not os.path.exists(expected_nii):
                    shutil.copy2(recovered_nii, expected_nii)
                    copied += 1
                    
        print(f"Merge complete. Copied {copied} NIfTIs.")

if __name__ == "__main__":
    main()
