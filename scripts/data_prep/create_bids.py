import os
import json
import csv
import re
import shutil
from datetime import datetime
import nibabel as nib

BASE_DIR = r"C:\Users\krish\FYP"
NIFTI_ORIG = os.path.join(BASE_DIR, r"NIfTI\NIfTI")
BIDS_DIR = os.path.join(BASE_DIR, "BIDS")
AUDIT_DIR = os.path.join(BASE_DIR, "audit")

GROUPS = ["AD", "EMCI", "LMCI", "CN_Final", "MCI", "SMC_Final"]

MAP_CSV = os.path.join(AUDIT_DIR, "bids_source_mapping.csv")
SESS_CSV = os.path.join(AUDIT_DIR, "bids_session_run_mapping.csv")
MISSING_CSV = os.path.join(AUDIT_DIR, "bids_missing_metadata.csv")
VAL_CSV = os.path.join(AUDIT_DIR, "bids_validation_report.csv")
VAL_SUMMARY = os.path.join(AUDIT_DIR, "bids_validation_summary.txt")

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

def parse_metadata_from_stem(stem):
    m = re.search(r'(\d{3}_S_\d{4})', stem)
    subj = m.group(1) if m else ""
    m = re.search(r'_(\d{14})_', stem)
    dt = m.group(1) if m else ""
    m = re.search(r'_(\d+)$', stem)
    series = m.group(1) if m else ""
    return subj, dt, series

def bidsify_subj(subj):
    return "sub-" + subj.replace("_", "")

def main():
    print("Starting BIDS creation process...")
    
    # Ensure BIDS dir exists
    os.makedirs(BIDS_DIR, exist_ok=True)
    
    # 1. Audit Source
    print("Phase 1: Source Audit & Mapping...")
    jsons = []
    niftis = []
    
    for g in GROUPS:
        g_dir = os.path.join(NIFTI_ORIG, g)
        if not os.path.exists(g_dir): continue
        for root, _, files in os.walk(g_dir):
            subj_id = os.path.basename(root)
            if not re.match(r'\d{3}_S_\d{4}', subj_id): continue
            for f in files:
                path = os.path.join(root, f)
                if f.endswith(".json"):
                    stem = f[:-5]
                    jsons.append({
                        "group": g,
                        "subject_id": subj_id,
                        "stem": stem,
                        "filename": f,
                        "path": path
                    })
                elif f.endswith(".nii") or f.endswith(".nii.gz"):
                    stem = f.replace(".nii.gz", "").replace(".nii", "")
                    niftis.append({
                        "stem": stem,
                        "path": path,
                        "filename": f
                    })
                    
    mapped_acquisitions = []
    map_rows = []
    
    for j in jsons:
        cands = [n for n in niftis if n["stem"] == j["stem"]]
        if len(cands) == 1:
            n = cands[0]
            
            # Read JSON for metadata
            jdata = {}
            try:
                with open(j["path"], 'r') as jp:
                    jdata = json.load(jp)
            except:
                pass
                
            subj, dt, series = parse_metadata_from_stem(j["stem"])
            acq_dt = format_datetime(jdata.get("AcquisitionDateTime", ""))
            if not acq_dt: acq_dt = dt
            
            snum = str(jdata.get("SeriesNumber", ""))
            if not snum: snum = series
            
            desc = jdata.get("SeriesDescription", "")
            
            mapped_acquisitions.append({
                "group": j["group"],
                "subject_id": j["subject_id"],
                "json_path": j["path"],
                "nii_path": n["path"],
                "acq_dt": acq_dt,
                "acq_date": acq_dt[:8] if len(acq_dt) >= 8 else "00000000",
                "desc": desc,
                "series": snum,
                "original_jdata": jdata
            })
            
            map_rows.append({
                "group": j["group"],
                "subject_id": j["subject_id"],
                "json_path": j["path"],
                "nii_path": n["path"],
                "acquisition_datetime": acq_dt,
                "scan_description": desc,
                "series_number": snum,
                "mapping_status": "EXACT_MATCH"
            })
        else:
            map_rows.append({
                "group": j["group"],
                "subject_id": j["subject_id"],
                "json_path": j["path"],
                "nii_path": "",
                "acquisition_datetime": "",
                "scan_description": "",
                "series_number": "",
                "mapping_status": "MISSING_OR_AMBIGUOUS_NIFTI"
            })
            
    with open(MAP_CSV, 'w', newline='', encoding='utf-8') as f:
        if map_rows:
            writer = csv.DictWriter(f, fieldnames=map_rows[0].keys())
            writer.writeheader()
            writer.writerows(map_rows)
            
    # 2. Session and Run Logic
    print("Phase 2: Session and Run Logic...")
    sess_rows = []
    
    subj_dict = {}
    for acq in mapped_acquisitions:
        sid = acq["subject_id"]
        if sid not in subj_dict: subj_dict[sid] = []
        subj_dict[sid].append(acq)
        
    for sid, acqs in subj_dict.items():
        # Sort chronologically (by full datetime, then series)
        acqs.sort(key=lambda x: (x["acq_dt"], x["series"]))
        
        # Group by exact Date (YYYYMMDD) to define sessions
        sessions = {}
        for acq in acqs:
            date = acq["acq_date"]
            if date not in sessions:
                sessions[date] = []
            sessions[date].append(acq)
            
        # Assign session and run numbers
        # Sort dates chronologically to assign ses-01, ses-02...
        sorted_dates = sorted(list(sessions.keys()))
        
        for sess_idx, date in enumerate(sorted_dates):
            bids_ses = f"{sess_idx + 1:02d}"
            
            # Within session, assign run numbers based on chronological order
            sess_acqs = sessions[date]
            for run_idx, acq in enumerate(sess_acqs):
                bids_run = f"{run_idx + 1:02d}"
                
                acq["bids_sub"] = bidsify_subj(sid)
                acq["bids_ses"] = bids_ses
                acq["bids_run"] = bids_run
                
                sess_rows.append({
                    "subject_id": sid,
                    "original_json": os.path.basename(acq["json_path"]),
                    "acquisition_datetime": acq["acq_dt"],
                    "series_number": acq["series"],
                    "BIDS_session": f"ses-{bids_ses}",
                    "BIDS_run": f"run-{bids_run}",
                    "reason": f"Date {date} -> Session {bids_ses}, Sequence {run_idx+1} -> Run {bids_run}"
                })
                
    with open(SESS_CSV, 'w', newline='', encoding='utf-8') as f:
        if sess_rows:
            writer = csv.DictWriter(f, fieldnames=sess_rows[0].keys())
            writer.writeheader()
            writer.writerows(sess_rows)
            
    # 3. Create BIDS Dataset
    print("Phase 3: Building BIDS Architecture...")
    
    # dataset_description.json
    ds_desc = {
        "Name": "ADNI Resting State fMRI",
        "BIDSVersion": "1.8.0",
        "DatasetType": "raw",
        "License": "ADNI Data Use Agreement"
    }
    with open(os.path.join(BIDS_DIR, "dataset_description.json"), 'w') as f:
        json.dump(ds_desc, f, indent=4)
        
    # participants.tsv
    with open(os.path.join(BIDS_DIR, "participants.tsv"), 'w', newline='') as f:
        writer = csv.writer(f, delimiter='\t')
        writer.writerow(["participant_id", "group"])
        for sid, acqs in subj_dict.items():
            # Use the group from the first acquisition
            writer.writerow([bidsify_subj(sid), acqs[0]["group"]])
            
    # Copy Data
    missing_meta_rows = []
    
    for acq in mapped_acquisitions:
        sub = acq["bids_sub"]
        ses = f"ses-{acq['bids_ses']}"
        run = f"run-{acq['bids_run']}"
        
        dest_dir = os.path.join(BIDS_DIR, sub, ses, "func")
        os.makedirs(dest_dir, exist_ok=True)
        
        bids_base = f"{sub}_{ses}_task-rest_{run}_bold"
        
        # Copy NIfTI
        nii_ext = ".nii.gz" if acq["nii_path"].endswith(".nii.gz") else ".nii"
        dest_nii = os.path.join(dest_dir, bids_base + nii_ext)
        if not os.path.exists(dest_nii):
            shutil.copy2(acq["nii_path"], dest_nii)
            
        # Write JSON
        jdata = acq["original_jdata"].copy()
        jdata["TaskName"] = "rest"
        
        # Check required fields
        required = ["RepetitionTime", "TaskName"]
        for r in required:
            if r not in jdata:
                missing_meta_rows.append({
                    "subject": sub,
                    "session": ses,
                    "run": run,
                    "missing_field": r,
                    "original_json": os.path.basename(acq["json_path"])
                })
                
        dest_json = os.path.join(dest_dir, bids_base + ".json")
        with open(dest_json, 'w') as f:
            json.dump(jdata, f, indent=4)
            
    with open(MISSING_CSV, 'w', newline='', encoding='utf-8') as f:
        if missing_meta_rows:
            writer = csv.DictWriter(f, fieldnames=missing_meta_rows[0].keys())
            writer.writeheader()
            writer.writerows(missing_meta_rows)
            
    # 4. Final Validation
    print("Phase 4: Final Validation...")
    val_rows = []
    val_counts = {
        "Total Subjects": 0,
        "Total Sessions": 0,
        "Total BIDS NIfTI": 0,
        "Total BIDS JSON": 0,
        "Valid NIfTI (nibabel)": 0,
        "Invalid NIfTI": 0,
        "Orphan NIfTI": 0,
        "Orphan JSON": 0
    }
    
    bids_niftis = []
    bids_jsons = []
    
    for root, dirs, files in os.walk(BIDS_DIR):
        if "sub-" in os.path.basename(root) and len(os.path.basename(root)) > 4:
            val_counts["Total Subjects"] += 1
        if "ses-" in os.path.basename(root):
            val_counts["Total Sessions"] += 1
            
        for f in files:
            p = os.path.join(root, f)
            if f.endswith(".json") and f != "dataset_description.json":
                bids_jsons.append(p)
            elif f.endswith(".nii") or f.endswith(".nii.gz"):
                bids_niftis.append(p)
                
    val_counts["Total BIDS JSON"] = len(bids_jsons)
    val_counts["Total BIDS NIfTI"] = len(bids_niftis)
    
    for nj in bids_jsons:
        stem = nj[:-5]
        expected_nii = stem + ".nii"
        expected_nii_gz = stem + ".nii.gz"
        has_nii = expected_nii in bids_niftis or expected_nii_gz in bids_niftis
        
        if not has_nii:
            val_counts["Orphan JSON"] += 1
            val_rows.append({"file": nj, "status": "ORPHAN_JSON", "notes": ""})
            
    for nn in bids_niftis:
        stem = nn.replace(".nii.gz", "").replace(".nii", "")
        expected_json = stem + ".json"
        has_json = expected_json in bids_jsons
        
        if not has_json:
            val_counts["Orphan NIfTI"] += 1
            val_rows.append({"file": nn, "status": "ORPHAN_NIFTI", "notes": ""})
            
        # Validate nibabel
        try:
            img = nib.load(nn)
            s = img.shape
            if len(s) >= 4:
                val_counts["Valid NIfTI (nibabel)"] += 1
                val_rows.append({"file": nn, "status": "VALID_NIFTI", "notes": f"Shape: {s}"})
            else:
                val_counts["Invalid NIfTI"] += 1
                val_rows.append({"file": nn, "status": "INVALID_NIFTI", "notes": f"Invalid shape: {s}"})
        except Exception as e:
            val_counts["Invalid NIfTI"] += 1
            val_rows.append({"file": nn, "status": "INVALID_NIFTI", "notes": f"Nibabel err: {str(e)}"})
            
    with open(VAL_CSV, 'w', newline='', encoding='utf-8') as f:
        if val_rows:
            writer = csv.DictWriter(f, fieldnames=val_rows[0].keys())
            writer.writeheader()
            writer.writerows(val_rows)
            
    with open(VAL_SUMMARY, 'w', encoding='utf-8') as f:
        f.write("=== BIDS VALIDATION SUMMARY ===\n\n")
        for k, v in val_counts.items():
            f.write(f"{k}: {v}\n")
            
    print("BIDS conversion complete. Please review audit reports.")

if __name__ == "__main__":
    main()
