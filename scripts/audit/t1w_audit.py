import os
import csv
import json

BIDS_DIR = r"C:\Users\krish\FYP\BIDS"
AUDIT_DIR = r"C:\Users\krish\FYP\audit"
CSV_OUT = os.path.join(AUDIT_DIR, "t1w_inventory.csv")
TXT_OUT = os.path.join(AUDIT_DIR, "t1w_inventory_summary.txt")

def main():
    print("Starting T1w audit...")
    t1w_files = []
    participants_with_t1w = set()
    all_participants = set()

    # Get all participants from participants.tsv
    tsv_path = os.path.join(BIDS_DIR, "participants.tsv")
    try:
        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            headers = next(reader)
            for row in reader:
                if len(row) >= 1 and row[0].startswith("sub-"):
                    all_participants.add(row[0])
    except Exception as e:
        print(f"Warning: Could not read participants.tsv: {e}")

    # Crawl the BIDS directory
    for root, dirs, files in os.walk(BIDS_DIR):
        for f in files:
            # Look for sub-* directories to capture any stray participants not in TSV
            if f.endswith(".nii") or f.endswith(".nii.gz"):
                stem = f.replace(".nii.gz", "").replace(".nii", "")
                parts = stem.split("_")
                sub = next((p for p in parts if p.startswith("sub-")), "UNKNOWN")
                all_participants.add(sub)
                
                if "T1w" in f:
                    ses = next((p for p in parts if p.startswith("ses-")), "NONE")
                    json_file = os.path.join(root, stem + ".json")
                    has_json = "YES" if os.path.exists(json_file) else "NO"
                    
                    t1w_files.append({
                        "participant_id": sub,
                        "session": ses,
                        "filename": f,
                        "has_json": has_json,
                        "filepath": os.path.join(root, f)
                    })
                    participants_with_t1w.add(sub)

    os.makedirs(AUDIT_DIR, exist_ok=True)
    
    # Write CSV
    with open(CSV_OUT, "w", newline="", encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Participant_ID", "Session", "Filename", "Has_JSON", "Filepath"])
        for t1 in t1w_files:
            writer.writerow([t1["participant_id"], t1["session"], t1["filename"], t1["has_json"], t1["filepath"]])

    # Write TXT Summary
    participants_missing_t1w = sorted(list(all_participants - participants_with_t1w))
    
    # Check for duplicates (same participant and session)
    seen = set()
    duplicates = []
    for t1 in t1w_files:
        key = f"{t1['participant_id']}_{t1['session']}"
        if key in seen:
            duplicates.append(t1)
        else:
            seen.add(key)

    with open(TXT_OUT, "w", encoding='utf-8') as f:
        f.write("=== T1w INVENTORY SUMMARY ===\n\n")
        f.write(f"Total T1w NIfTI files found: {len(t1w_files)}\n")
        f.write(f"Total participants with T1w: {len(participants_with_t1w)}\n")
        f.write(f"Total participants missing T1w: {len(participants_missing_t1w)}\n")
        f.write(f"Total duplicate T1w acquisitions: {len(duplicates)}\n\n")
        
        f.write("--- DUPLICATES ---\n")
        if duplicates:
            for d in duplicates:
                f.write(f"Duplicate: {d['participant_id']} (Session: {d['session']}) - {d['filename']}\n")
        else:
            f.write("None\n")
            
        f.write("\n--- PARTICIPANTS MISSING T1w ---\n")
        for p in participants_missing_t1w:
            if p != "UNKNOWN":
                f.write(f"{p}\n")
                
        f.write("\n--- T1w FILE LIST ---\n")
        for t1 in t1w_files:
            f.write(f"Participant: {t1['participant_id']} | Session: {t1['session']} | File: {t1['filename']} | JSON: {t1['has_json']}\n")

    print(f"T1w audit complete. Found {len(t1w_files)} files. Written to {TXT_OUT}")

if __name__ == "__main__":
    main()
