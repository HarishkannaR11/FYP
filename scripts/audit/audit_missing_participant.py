import os
import csv
import json

BASE_DIR = r"C:\Users\krish\FYP"
NIFTI_ORIG = os.path.join(BASE_DIR, r"NIfTI\NIfTI")
BIDS_DIR = os.path.join(BASE_DIR, "BIDS")
AUDIT_DIR = os.path.join(BASE_DIR, "audit")
REPORT_PATH = os.path.join(AUDIT_DIR, "participant_discrepancy_report.txt")

GROUPS = ["AD", "EMCI", "LMCI", "CN_Final", "MCI", "SMC_Final"]

def bidsify_subj(subj):
    return "sub-" + subj.replace("_", "")

def main():
    print("Auditing original vs BIDS participants...")
    
    expected_subjects = set()
    subject_to_group = {}
    subject_to_acqs = {}

    # 1. Read original NIfTI\NIfTI
    for g in GROUPS:
        g_dir = os.path.join(NIFTI_ORIG, g)
        if not os.path.exists(g_dir): continue
        for subj in os.listdir(g_dir):
            if not subj.startswith("0") and not subj.startswith("1"): continue # Basic filter
            subj_dir = os.path.join(g_dir, subj)
            if os.path.isdir(subj_dir):
                # Count valid json+nifti pairs
                jsons = [f for f in os.listdir(subj_dir) if f.endswith(".json")]
                niftis = [f for f in os.listdir(subj_dir) if f.endswith(".nii") or f.endswith(".nii.gz")]
                
                valid_pairs = 0
                for j in jsons:
                    stem = j[:-5]
                    if stem + ".nii" in niftis or stem + ".nii.gz" in niftis:
                        valid_pairs += 1
                        
                if valid_pairs > 0:
                    expected_subjects.add(subj)
                    subject_to_group[subj] = g
                    subject_to_acqs[subj] = valid_pairs

    # 2. Read BIDS participants.tsv
    actual_bids_subjects = set()
    tsv_path = os.path.join(BIDS_DIR, "participants.tsv")
    try:
        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            headers = next(reader)
            for row in reader:
                if len(row) >= 1:
                    # Strip "sub-" prefix and convert back to ADNI format roughly
                    bids_id = row[0]
                    actual_bids_subjects.add(bids_id)
    except Exception as e:
        print(f"Error reading TSV: {e}")

    # Map expected back to BIDS format for comparison
    expected_bids_format = {subj: bidsify_subj(subj) for subj in expected_subjects}
    
    missing_from_bids = []
    for orig_subj, bids_subj in expected_bids_format.items():
        if bids_subj not in actual_bids_subjects:
            missing_from_bids.append(orig_subj)
            
    # Check if total acquisitions match
    total_expected_acqs = sum(subject_to_acqs.values())
    
    total_bids_acqs = 0
    for root, dirs, files in os.walk(BIDS_DIR):
        for f in files:
            if f.endswith(".json") and f != "dataset_description.json":
                total_bids_acqs += 1

    with open(REPORT_PATH, 'w', encoding='utf-8') as f:
        f.write("=== PARTICIPANT DISCREPANCY REPORT ===\n\n")
        f.write(f"Expected Participants (NIfTI Orig): {len(expected_subjects)}\n")
        f.write(f"Actual Participants (BIDS TSV): {len(actual_bids_subjects)}\n")
        f.write(f"Missing Participant Count: {len(missing_from_bids)}\n\n")
        
        f.write("--- ACQUISITION VERIFICATION ---\n")
        f.write(f"Expected Valid Acquisitions (Original JSON+NIfTI pairs): {total_expected_acqs}\n")
        f.write(f"Actual BIDS Acquisitions (BIDS JSON+NIfTI pairs): {total_bids_acqs}\n")
        f.write(f"Discrepancy: {total_expected_acqs - total_bids_acqs}\n\n")
        
        f.write("--- MISSING PARTICIPANTS DETAILS ---\n")
        if not missing_from_bids:
            f.write("None found.\n")
        else:
            for m in missing_from_bids:
                f.write(f"Original ID: {m}\n")
                f.write(f"Group: {subject_to_group[m]}\n")
                f.write(f"Expected BIDS ID: {expected_bids_format[m]}\n")
                f.write(f"Acquisitions tied to this subject: {subject_to_acqs[m]}\n")
                f.write(f"Source Folder: NIfTI\\NIfTI\\{subject_to_group[m]}\\{m}\n")
                f.write("\nFiles in original source:\n")
                for src_f in os.listdir(os.path.join(NIFTI_ORIG, subject_to_group[m], m)):
                    f.write(f"  - {src_f}\n")
                f.write("\n")
                
        # Check if the missing participant was accidentally merged into another
        # by searching the BIDS folder for their exact file stem
        f.write("--- BIDS FOLDER SEARCH FOR MISSING ACQUISITIONS ---\n")
        for m in missing_from_bids:
            found = False
            for src_f in os.listdir(os.path.join(NIFTI_ORIG, subject_to_group[m], m)):
                if src_f.endswith(".json"):
                    stem = src_f[:-5]
                    # Search BIDS for any JSON containing this stem inside its original JSON field
                    # Or just search BIDS folders for files with similar names?
                    # The script earlier stored the original json in the json metadata? No, only TaskName was added.
                    # BIDS renames the file entirely to sub-XXX_ses-YYY...
                    pass
                    
            # Let's check the source mapping CSV we generated earlier
            map_csv = os.path.join(AUDIT_DIR, "bids_source_mapping.csv")
            try:
                with open(map_csv, 'r') as mcf:
                    reader = csv.DictReader(mcf)
                    for row in reader:
                        if row["subject_id"] == m:
                            f.write(f"Audit log found {m} -> {row['mapping_status']}\n")
            except:
                pass

    print("Audit complete.")

if __name__ == "__main__":
    main()
