import os
import json
import csv
import zipfile
import tempfile
import shutil
import subprocess
import re
from datetime import datetime
import nibabel as nib

BASE_DIR = r"C:\Users\krish\FYP"
NIFTI_ORIG = os.path.join(BASE_DIR, r"NIfTI\NIfTI")
RECOVERED_DIR = os.path.join(BASE_DIR, "recovered_fmri")
RAW_DIR = os.path.join(BASE_DIR, "raw_data")
AUDIT_DIR = os.path.join(BASE_DIR, "audit")

TARGET_GROUPS = ["MCI", "SMC", "SMC_Final"]

REPORT_CSV = os.path.join(AUDIT_DIR, "mci_smc_recovery_report.csv")
SUMMARY_TXT = os.path.join(AUDIT_DIR, "mci_smc_recovery_summary.txt")
MISSING_CSV = os.path.join(AUDIT_DIR, "mci_smc_missing_acquisitions.csv")

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
    print("========================================================")
    print("STEP 1-3: INVENTORY AND ACQUISITION-LEVEL MATCHING")
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

    # Track existing NIfTIs in NIfTI and recovered
    existing_niftis = {}
    
    for root, _, files in os.walk(NIFTI_ORIG):
        for f in files:
            if f.endswith(".nii") or f.endswith(".nii.gz"):
                base = f.split('.nii')[0]
                existing_niftis[base] = "EXISTING"
                
    if os.path.exists(RECOVERED_DIR):
        for root, _, files in os.walk(RECOVERED_DIR):
            for f in files:
                if f.endswith(".nii") or f.endswith(".nii.gz"):
                    base = f.split('.nii')[0]
                    existing_niftis[base] = "RECOVERED"
                    
    # Scan JSONs
    acquisitions = []
    missing_csv_data = []
    
    for root, _, files in os.walk(NIFTI_ORIG):
        group = None
        for g in TARGET_GROUPS:
            if f"\\{g}\\" in root or f"/{g}/" in root:
                group = g
                break
        if not group:
            continue
            
        for f in files:
            if f.endswith(".json"):
                json_path = os.path.join(root, f)
                base_no_ext = f[:-5]
                
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
                
                # Fallback to audit metadata
                if f in audit_metadata:
                    row = audit_metadata[f]
                    if not subj_id: subj_id = row.get('subject_id', '')
                    if not acq_dt: acq_dt = row.get('acquisition_datetime', '')
                    if not scan_desc: scan_desc = row.get('series_description', '')
                
                if not subj_id:
                    m = re.search(r'(\d{3}_S_\d{4})', f)
                    if m: subj_id = m.group(1)
                    
                has_nifti = base_no_ext in existing_niftis
                status = "NIFTI_PRESENT" if has_nifti else "MISSING_NIFTI"
                
                acq = {
                    "group": group,
                    "subject_id": subj_id,
                    "json_filename": f,
                    "json_path": json_path,
                    "base_name": base_no_ext,
                    "acquisition_datetime": acq_dt,
                    "scan_description": scan_desc,
                    "status": status,
                    "existing_status": existing_niftis.get(base_no_ext, "NONE")
                }
                acquisitions.append(acq)
                missing_csv_data.append({
                    "group": group,
                    "subject_id": subj_id,
                    "acquisition_datetime": acq_dt,
                    "json_path": json_path,
                    "json_filename": f,
                    "status": status
                })

    with open(MISSING_CSV, 'w', newline='', encoding='utf-8') as f:
        if missing_csv_data:
            writer = csv.DictWriter(f, fieldnames=missing_csv_data[0].keys())
            writer.writeheader()
            writer.writerows(missing_csv_data)

    missing_acqs = [a for a in acquisitions if a['status'] == "MISSING_NIFTI"]
    print(f"Found {len(missing_acqs)} missing acquisitions for MCI/SMC.")

    print("========================================================")
    print("STEP 4-6: SEARCH RAW DATA AND MATCH DICOM SERIES")
    print("========================================================")
    
    # Only scan RAW if there is something missing
    raw_map = {}
    if missing_acqs:
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
                        
    print("========================================================")
    print("STEP 7-12: CONVERT, VALIDATE, REPORT")
    print("========================================================")
    
    results = []
    
    # Initialize group summaries
    group_stats = {}
    for g in TARGET_GROUPS:
        group_stats[g] = {
            "Total JSON acquisitions": 0,
            "Existing NIfTI": 0,
            "Already recovered": 0,
            "Missing NIfTI": 0,
            "Raw DICOM matches": 0,
            "Successfully converted": 0,
            "Conversion failures": 0,
            "No raw match": 0,
            "Ambiguous": 0,
            "Invalid NIfTI": 0,
            "Not fMRI": 0
        }

    for acq in acquisitions:
        g = acq['group']
        group_stats[g]["Total JSON acquisitions"] += 1
        
        subj = acq['subject_id']
        acq_dt = acq['acquisition_datetime']
        base = acq['base_name']
        
        row = {
            "group": g,
            "subject_id": subj,
            "acquisition_datetime": acq_dt,
            "original_json": acq['json_filename'],
            "raw_archive": "",
            "raw_series_path": "",
            "match_confidence": "NONE",
            "conversion_status": "",
            "nifti_path": "",
            "nifti_valid": "NO",
            "dimensions": "",
            "timepoints": "",
            "TR": "",
            "series_description": "",
            "reason": "",
            "notes": ""
        }
        
        if acq['status'] == "NIFTI_PRESENT":
            row['conversion_status'] = "ALREADY_PRESENT"
            row['reason'] = "NIfTI exists"
            if acq['existing_status'] == "EXISTING":
                group_stats[g]["Existing NIfTI"] += 1
            else:
                group_stats[g]["Already recovered"] += 1
            results.append(row)
            continue
            
        group_stats[g]["Missing NIfTI"] += 1
        
        # Search RAW
        key = f"{subj}_{acq_dt}"
        if key in raw_map:
            cands = raw_map[key]
            best = cands[0] # Default to first if multiple
            # Prefer ones matching series desc roughly
            for c in cands:
                if acq['scan_description'] and acq['scan_description'] in c['series_desc']:
                    best = c
                    break
                    
            row['raw_archive'] = best['zip_path']
            row['raw_series_path'] = best['internal_dir']
            row['series_description'] = best['series_desc']
            row['match_confidence'] = "HIGH"
            
            # Verify if it's fMRI
            if not is_resting_state_fmri(best['series_desc']) and not is_resting_state_fmri(acq['scan_description']):
                row['conversion_status'] = "NOT_FMRI"
                row['reason'] = "Raw series is not resting-state fMRI"
                group_stats[g]["Not fMRI"] += 1
            else:
                group_stats[g]["Raw DICOM matches"] += 1
                
                # Extract and Convert
                temp_dir = tempfile.mkdtemp()
                target_dir = os.path.join(RECOVERED_DIR, g, subj)
                os.makedirs(target_dir, exist_ok=True)
                target_nii = os.path.join(target_dir, base + ".nii")
                
                try:
                    ext_path = extract_dicom_series(best['zip_path'], best['internal_dir'], temp_dir)
                    
                    dcm2niix_path = r"C:\Users\krish\FYP\tools\dcm2niix.exe"
                    cmd = [dcm2niix_path, "-z", "n", "-b", "n", "-f", base, "-o", target_dir, ext_path]
                    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    
                    if not os.path.exists(target_nii):
                        possible = [f for f in os.listdir(target_dir) if f.startswith(base) and f.endswith(".nii")]
                        if possible:
                            os.rename(os.path.join(target_dir, possible[0]), target_nii)
                            
                    if os.path.exists(target_nii):
                        is_4d, dims, tps, vox, tr, err = validate_nifti(target_nii)
                        row['nifti_path'] = target_nii
                        row['dimensions'] = dims
                        row['timepoints'] = tps
                        row['TR'] = tr
                        if is_4d:
                            row['conversion_status'] = "RECOVERED"
                            row['nifti_valid'] = "YES"
                            row['reason'] = "Success"
                            group_stats[g]["Successfully converted"] += 1
                        else:
                            row['conversion_status'] = "INVALID_NIFTI"
                            row['reason'] = f"Validation failed: {err}"
                            group_stats[g]["Invalid NIfTI"] += 1
                            # Delete invalid
                            os.remove(target_nii)
                    else:
                        row['conversion_status'] = "CONVERSION_FAILED"
                        row['reason'] = "dcm2niix produced no output"
                        row['notes'] = p.stderr.strip()
                        group_stats[g]["Conversion failures"] += 1
                        
                except Exception as e:
                    row['conversion_status'] = "CONVERSION_FAILED"
                    row['reason'] = str(e)
                    group_stats[g]["Conversion failures"] += 1
                finally:
                    shutil.rmtree(temp_dir, ignore_errors=True)
        else:
            if not subj or not acq_dt:
                row['conversion_status'] = "AMBIGUOUS"
                row['reason'] = "Missing subject or datetime"
                group_stats[g]["Ambiguous"] += 1
            else:
                row['conversion_status'] = "NO_RAW_MATCH"
                row['reason'] = "No corresponding series in RAW data"
                group_stats[g]["No raw match"] += 1
                
        results.append(row)

    print("========================================================")
    print("STEP 12-14: REPORTING")
    print("========================================================")
    
    with open(REPORT_CSV, 'w', newline='', encoding='utf-8') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
            
    with open(SUMMARY_TXT, 'w', encoding='utf-8') as f:
        for g in TARGET_GROUPS:
            f.write(f"{g}\n----\n")
            stats = group_stats[g]
            for k, v in stats.items():
                f.write(f"{k}: {v}\n")
            f.write("\n")
            
        f.write("--- Successfully Recovered Files ---\n")
        for r in results:
            if r['conversion_status'] == "RECOVERED":
                f.write(f"{r['subject_id']} | {r['acquisition_datetime']} | {r['nifti_path']}\n")

    print("Complete! Check mci_smc_recovery_report.csv and mci_smc_recovery_summary.txt")

if __name__ == "__main__":
    main()
