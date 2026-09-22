import os
import json
import csv
import re
import shutil
import argparse
import nibabel as nib

BASE_DIR = r"C:\Users\krish\FYP"
NIFTI_ORIG = os.path.join(BASE_DIR, r"NIfTI\NIfTI")
RECOVERED_DIR = os.path.join(BASE_DIR, "recovered_fmri")
AUDIT_DIR = os.path.join(BASE_DIR, "audit")

GROUPS = ["AD", "EMCI", "LMCI", "CN_Final", "MCI", "SMC_Final"]

DRY_RUN_CSV = os.path.join(AUDIT_DIR, "recovered_json_mapping_dry_run.csv")
DRY_RUN_SUMMARY = os.path.join(AUDIT_DIR, "recovered_json_mapping_dry_run_summary.txt")

FINAL_CSV = os.path.join(AUDIT_DIR, "recovered_json_mapping_final.csv")
FINAL_SUMMARY = os.path.join(AUDIT_DIR, "recovered_json_mapping_final_summary.txt")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Execute the actual copy operation")
    args = parser.parse_args()

    print("========================================================")
    print(f"JSON-FIRST MERGE - {'EXECUTE' if args.execute else 'DRY RUN'}")
    print("========================================================")

    jsons = []
    recovered_niftis = []

    # 1. Inventory Existing JSONs
    print("Scanning primary dataset for JSONs...")
    for g in GROUPS:
        g_dir = os.path.join(NIFTI_ORIG, g)
        if not os.path.exists(g_dir): continue
        for root, _, files in os.walk(g_dir):
            subj_id = os.path.basename(root)
            if not re.match(r'\d{3}_S_\d{4}', subj_id):
                continue
            for f in files:
                if f.endswith(".json"):
                    stem = f[:-5]
                    jsons.append({
                        "group": g,
                        "subject_id": subj_id,
                        "stem": stem,
                        "filename": f,
                        "dir": root,
                        "path": os.path.join(root, f)
                    })

    # 2. Inventory Recovered NIfTIs
    print("Scanning recovered dataset for NIfTIs...")
    for g in GROUPS:
        rec_g_dir = os.path.join(RECOVERED_DIR, g)
        if not os.path.exists(rec_g_dir): continue
        for root, _, files in os.walk(rec_g_dir):
            subj_id = os.path.basename(root)
            for f in files:
                if f.endswith(".nii") or f.endswith(".nii.gz"):
                    stem = f.replace(".nii.gz", "").replace(".nii", "")
                    # Extract subj_id from stem if folder is weird
                    m = re.search(r'(\d{3}_S_\d{4})', stem)
                    actual_subj = m.group(1) if m else subj_id
                    recovered_niftis.append({
                        "group": g,
                        "subject_id": actual_subj,
                        "stem": stem,
                        "filename": f,
                        "dir": root,
                        "path": os.path.join(root, f),
                        "mapped": False
                    })

    # 3. Dry Run Mapping
    print("Mapping JSONs to recovered NIfTIs...")
    
    group_counts = {g: {
        "READY_TO_MERGE": 0,
        "ALREADY_PRESENT": 0,
        "JSON_WITHOUT_RECOVERED_NIFTI": 0,
        "ORPHAN_RECOVERED_NIFTI": 0,
        "AMBIGUOUS": 0,
        "DUPLICATE": 0
    } for g in GROUPS}

    results = []
    
    for j in jsons:
        g = j["group"]
        
        # Check if already present in NIfTI_ORIG
        expected_nii = os.path.join(j["dir"], j["stem"] + ".nii")
        expected_nii_gz = os.path.join(j["dir"], j["stem"] + ".nii.gz")
        
        already_present = os.path.exists(expected_nii) or os.path.exists(expected_nii_gz)
        
        cands = [n for n in recovered_niftis if n["stem"] == j["stem"]]
        
        if len(cands) == 1:
            n = cands[0]
            n["mapped"] = True
            
            status = "ALREADY_PRESENT" if already_present else "READY_TO_MERGE"
            group_counts[g][status] += 1
            
            results.append({
                "group": g,
                "subject_id": j["subject_id"],
                "recovered_nii": n["path"],
                "recovered_nii_stem": n["stem"],
                "matched_json": j["filename"],
                "destination_directory": j["dir"],
                "destination_nii": os.path.join(j["dir"], n["filename"]),
                "status": status,
                "notes": ""
            })
        elif len(cands) == 0:
            group_counts[g]["JSON_WITHOUT_RECOVERED_NIFTI"] += 1
            results.append({
                "group": g,
                "subject_id": j["subject_id"],
                "recovered_nii": "",
                "recovered_nii_stem": "",
                "matched_json": j["filename"],
                "destination_directory": j["dir"],
                "destination_nii": "",
                "status": "JSON_WITHOUT_RECOVERED_NIFTI",
                "notes": "No recovered NIfTI found."
            })
        else:
            group_counts[g]["DUPLICATE"] += 1
            for n in cands:
                n["mapped"] = True
            results.append({
                "group": g,
                "subject_id": j["subject_id"],
                "recovered_nii": "MULTIPLE",
                "recovered_nii_stem": j["stem"],
                "matched_json": j["filename"],
                "destination_directory": j["dir"],
                "destination_nii": "",
                "status": "DUPLICATE",
                "notes": f"Found {len(cands)} recovered files with this stem."
            })

    # Find orphans
    for n in recovered_niftis:
        if not n["mapped"]:
            g = n["group"]
            group_counts[g]["ORPHAN_RECOVERED_NIFTI"] += 1
            results.append({
                "group": g,
                "subject_id": n["subject_id"],
                "recovered_nii": n["path"],
                "recovered_nii_stem": n["stem"],
                "matched_json": "",
                "destination_directory": "",
                "destination_nii": "",
                "status": "ORPHAN_RECOVERED_NIFTI",
                "notes": "Recovered file does not match any JSON stem."
            })

    if not args.execute:
        # Write Dry Run Outputs
        print("Writing dry run reports...")
        with open(DRY_RUN_CSV, 'w', newline='', encoding='utf-8') as f:
            if results:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)

        with open(DRY_RUN_SUMMARY, 'w', encoding='utf-8') as f:
            f.write("=== RECOVERED JSON MAPPING DRY RUN SUMMARY ===\n\n")
            for g in GROUPS:
                f.write(f"--- {g} ---\n")
                for k, v in group_counts[g].items():
                    f.write(f"{k}: {v}\n")
                f.write("\n")
                
        print(f"Dry run complete. Check {DRY_RUN_SUMMARY}")

    else:
        # Execute Copy & Verify
        print("Executing copy operations...")
        copied = 0
        for r in results:
            if r["status"] == "READY_TO_MERGE":
                src = r["recovered_nii"]
                dst = r["destination_nii"]
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    copied += 1
                    
        print(f"Copied {copied} NIfTIs. Running final verification...")
        
        final_counts = {
            "TOTAL JSON": len(jsons),
            "RECOVERED NIFTI FOUND": sum(1 for r in results if r["status"] in ("READY_TO_MERGE", "ALREADY_PRESENT")),
            "SUCCESSFULLY MERGED": 0,
            "ALREADY PRESENT": sum(1 for r in results if r["status"] == "ALREADY_PRESENT"),
            "JSON WITHOUT RECOVERED NIFTI": sum(1 for r in results if r["status"] == "JSON_WITHOUT_RECOVERED_NIFTI"),
            "ORPHAN RECOVERED NIFTI": sum(1 for r in results if r["status"] == "ORPHAN_RECOVERED_NIFTI"),
            "AMBIGUOUS": sum(1 for r in results if r["status"] == "AMBIGUOUS"),
            "DUPLICATE": sum(1 for r in results if r["status"] == "DUPLICATE"),
            "INVALID NIFTI": 0
        }
        
        final_rows = []
        for r in results:
            if r["status"] == "READY_TO_MERGE":
                dst = r["destination_nii"]
                if os.path.exists(dst):
                    # Validate
                    try:
                        img = nib.load(dst)
                        if len(img.shape) >= 4:
                            r["status"] = "SUCCESSFULLY_MERGED"
                            final_counts["SUCCESSFULLY MERGED"] += 1
                        else:
                            r["status"] = "INVALID_NIFTI"
                            final_counts["INVALID NIFTI"] += 1
                            r["notes"] = "Not a 4D NIfTI"
                    except:
                        r["status"] = "INVALID_NIFTI"
                        final_counts["INVALID NIFTI"] += 1
                        r["notes"] = "nibabel failed to open"
                else:
                    r["status"] = "COPY_FAILED"
            final_rows.append(r)
            
        print("Writing final reports...")
        with open(FINAL_CSV, 'w', newline='', encoding='utf-8') as f:
            if final_rows:
                writer = csv.DictWriter(f, fieldnames=final_rows[0].keys())
                writer.writeheader()
                writer.writerows(final_rows)

        with open(FINAL_SUMMARY, 'w', encoding='utf-8') as f:
            f.write("=== RECOVERED JSON MAPPING FINAL SUMMARY ===\n\n")
            for k, v in final_counts.items():
                f.write(f"{k}: {v}\n")
                
        print(f"Final execution complete. Check {FINAL_SUMMARY}")

if __name__ == "__main__":
    main()
