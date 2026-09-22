import os
import json
import csv
import re

BIDS_DIR = r"C:\Users\krish\FYP\BIDS"
AUDIT_DIR = r"C:\Users\krish\FYP\audit"
VAL_REPORT = os.path.join(AUDIT_DIR, "bids_validation_mapping.csv")
VAL_SUMMARY = os.path.join(AUDIT_DIR, "bids_validation_report.txt")

def main():
    print("Starting comprehensive BIDS validation audit...")
    
    # 1. Load participants.tsv
    participants = {}
    tsv_path = os.path.join(BIDS_DIR, "participants.tsv")
    try:
        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            headers = next(reader)
            for row in reader:
                if len(row) >= 2:
                    participants[row[0]] = row[1]
    except Exception as e:
        print(f"Error reading participants.tsv: {e}")
        return

    # 2. Inventory BIDS Dir
    bids_subjects = set()
    bids_files = []
    
    for root, dirs, files in os.walk(BIDS_DIR):
        rel_path = os.path.relpath(root, BIDS_DIR)
        parts = rel_path.split(os.sep)
        
        # We only care about sub- directories for file mapping
        if parts[0].startswith("sub-"):
            subj = parts[0]
            bids_subjects.add(subj)
            sess = parts[1] if len(parts) > 1 and parts[1].startswith("ses-") else ""
            
            for f in files:
                bids_files.append({
                    "subject": subj,
                    "session": sess,
                    "filename": f,
                    "path": os.path.join(root, f)
                })
                
    # 3. Cross-check participants.tsv vs actual subject folders
    tsv_subs = set(participants.keys())
    missing_in_dir = tsv_subs - bids_subjects
    missing_in_tsv = bids_subjects - tsv_subs
    
    # 4. Map JSON and NIfTI
    acquisitions = {}
    orphans = []
    
    for bf in bids_files:
        f = bf["filename"]
        if not (f.endswith(".json") or f.endswith(".nii") or f.endswith(".nii.gz")):
            continue
            
        stem = f.replace(".nii.gz", "").replace(".nii", "").replace(".json", "")
        
        if stem not in acquisitions:
            acquisitions[stem] = {
                "subject": bf["subject"],
                "session": bf["session"],
                "stem": stem,
                "group": participants.get(bf["subject"], "UNKNOWN"),
                "has_json": False,
                "has_nifti": False,
                "json_path": "",
                "nifti_path": "",
                "is_bids_compliant": True,
                "bids_error": ""
            }
            
        acq = acquisitions[stem]
        
        # Check BIDS compliance of filename
        bids_pattern = r"^sub-[a-zA-Z0-9]+(_ses-[a-zA-Z0-9]+)?_task-[a-zA-Z0-9]+(_run-\d+)?_bold$"
        if not re.match(bids_pattern, stem):
            acq["is_bids_compliant"] = False
            acq["bids_error"] = "Non-compliant filename pattern"
            
        if f.endswith(".json"):
            acq["has_json"] = True
            acq["json_path"] = bf["path"]
        else:
            if acq["has_nifti"]:
                acq["bids_error"] = "Multiple NIfTI files for same stem"
            acq["has_nifti"] = True
            acq["nifti_path"] = bf["path"]

    # 5. Group Statistics
    group_stats = {}
    for g in set(participants.values()):
        group_stats[g] = {"subjects": set(), "sessions": set(), "runs": 0}
        
    csv_rows = []
    
    for stem, acq in acquisitions.items():
        g = acq["group"]
        if g in group_stats:
            group_stats[g]["subjects"].add(acq["subject"])
            if acq["session"]:
                group_stats[g]["sessions"].add(f"{acq['subject']}_{acq['session']}")
            else:
                group_stats[g]["sessions"].add(f"{acq['subject']}_none")
                
            if acq["has_nifti"] and acq["has_json"]:
                group_stats[g]["runs"] += 1
                
        status = "OK"
        if not acq["has_json"]:
            status = "ORPHAN_NIFTI"
        elif not acq["has_nifti"]:
            status = "ORPHAN_JSON"
        elif not acq["is_bids_compliant"]:
            status = "INVALID_BIDS_NAME"
            
        csv_rows.append({
            "subject": acq["subject"],
            "session": acq["session"],
            "group": g,
            "stem": stem,
            "has_json": acq["has_json"],
            "has_nifti": acq["has_nifti"],
            "bids_compliant": acq["is_bids_compliant"],
            "status": status,
            "notes": acq["bids_error"]
        })

    # 6. Generate Reports
    print("Writing reports...")
    
    with open(VAL_REPORT, 'w', newline='', encoding='utf-8') as f:
        if csv_rows:
            writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
            writer.writeheader()
            writer.writerows(csv_rows)
            
    with open(VAL_SUMMARY, 'w', encoding='utf-8') as f:
        f.write("=== BIDS VALIDATION AUDIT SUMMARY ===\n\n")
        
        f.write("--- PARTICIPANTS.TSV MATCHING ---\n")
        f.write(f"Subjects in TSV: {len(tsv_subs)}\n")
        f.write(f"Subject directories: {len(bids_subjects)}\n")
        f.write(f"Missing from dir (in TSV only): {missing_in_dir if missing_in_dir else 'None'}\n")
        f.write(f"Missing from TSV (in dir only): {missing_in_tsv if missing_in_tsv else 'None'}\n\n")
        
        f.write("--- FILE INTEGRITY ---\n")
        total_acqs = len(acquisitions)
        perfect_pairs = sum(1 for a in acquisitions.values() if a["has_nifti"] and a["has_json"])
        orphan_nifti = sum(1 for a in acquisitions.values() if a["has_nifti"] and not a["has_json"])
        orphan_json = sum(1 for a in acquisitions.values() if a["has_json"] and not a["has_nifti"])
        non_compliant = sum(1 for a in acquisitions.values() if not a["is_bids_compliant"])
        
        f.write(f"Total Acquisition Stems: {total_acqs}\n")
        f.write(f"Perfect JSON+NIfTI Pairs: {perfect_pairs}\n")
        f.write(f"Orphan NIfTIs: {orphan_nifti}\n")
        f.write(f"Orphan JSONs: {orphan_json}\n")
        f.write(f"BIDS Non-Compliant Names: {non_compliant}\n\n")
        
        f.write("--- DIAGNOSTIC GROUP STATISTICS ---\n")
        for g, stats in group_stats.items():
            f.write(f"\n{g}:\n")
            f.write(f"  Subjects: {len(stats['subjects'])}\n")
            f.write(f"  Sessions: {len(stats['sessions'])}\n")
            f.write(f"  Valid Runs/Files: {stats['runs']}\n")

    print(f"Validation complete. Check {VAL_SUMMARY}")

if __name__ == "__main__":
    main()
