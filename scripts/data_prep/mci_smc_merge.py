import os
import csv
import argparse
import shutil
import nibabel as nib

BASE_DIR = r"C:\Users\krish\FYP"
NIFTI_ORIG = os.path.join(BASE_DIR, r"NIfTI\NIfTI")
RECOVERED_DIR = os.path.join(BASE_DIR, "recovered_fmri")
AUDIT_DIR = os.path.join(BASE_DIR, "audit")

TARGET_GROUPS = ["MCI", "SMC_Final"]

DRY_RUN_CSV = os.path.join(AUDIT_DIR, "mci_smc_merge_dry_run.csv")
DRY_RUN_SUMMARY = os.path.join(AUDIT_DIR, "mci_smc_merge_dry_run_summary.txt")

FINAL_CSV = os.path.join(AUDIT_DIR, "mci_smc_merge_final.csv")
FINAL_SUMMARY = os.path.join(AUDIT_DIR, "mci_smc_merge_final_summary.txt")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Perform actual merge")
    args = parser.parse_args()

    print(f"========================================================")
    print(f"MCI/SMC DATASET MERGE - {'EXECUTE' if args.execute else 'DRY RUN'}")
    print(f"========================================================")

    # 1. Map existing JSONs and NIfTIs
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
                        "path": os.path.join(root, f),
                        "subject_id": os.path.basename(root)
                    }

    # 3. Match and determine status
    results = []
    summary_counts = {
        "READY_TO_COPY": 0,
        "ALREADY_PRESENT": 0,
        "JSON_NOT_FOUND": 0,
        "AMBIGUOUS": 0
    }

    # First handle all recovered files
    for stem, rec_info in recovered_map.items():
        row = {
            "subject_id": rec_info['subject_id'],
            "group": rec_info['group'],
            "json_path": "",
            "recovered_nii_path": rec_info['path'],
            "destination_path": "",
            "matching_stem": stem,
            "status": ""
        }

        if stem in json_map:
            row["json_path"] = json_map[stem]["path"]
            row["destination_path"] = os.path.join(json_map[stem]["dir"], stem + ".nii")
            
            if stem in existing_nifti_map:
                row["status"] = "ALREADY_PRESENT"
                summary_counts["ALREADY_PRESENT"] += 1
            else:
                row["status"] = "READY_TO_COPY"
                summary_counts["READY_TO_COPY"] += 1
        else:
            row["status"] = "JSON_NOT_FOUND"
            summary_counts["JSON_NOT_FOUND"] += 1

        results.append(row)

    if not args.execute:
        # Write dry run reports
        with open(DRY_RUN_CSV, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

        with open(DRY_RUN_SUMMARY, 'w', encoding='utf-8') as f:
            f.write("--- MCI/SMC Merge Dry Run Summary ---\n")
            for k, v in summary_counts.items():
                f.write(f"{k}: {v}\n")
                
        print("Dry run complete. Check audit/mci_smc_merge_dry_run_summary.txt")
    else:
        # EXECUTE COPY
        print("Executing copy...")
        copied_count = 0
        
        for r in results:
            if r["status"] == "READY_TO_COPY":
                src = r["recovered_nii_path"]
                dst = r["destination_path"]
                
                # Copy file safely
                if not os.path.exists(dst):
                    shutil.copy2(src, dst)
                    copied_count += 1
                else:
                    print(f"Warning: {dst} suddenly exists, skipping.")
                    
        print(f"Copied {copied_count} NIfTI files.")
        
        # FINAL VERIFICATION
        print("Performing final verification...")
        
        final_counts = {
            "total_json_acquisitions": 0,
            "niftis_already_present": 0,
            "successfully_mapped_and_copied": 0,
            "unmatched": summary_counts["JSON_NOT_FOUND"],
            "ambiguous": summary_counts["AMBIGUOUS"],
            "invalid_nifti": 0,
            "remaining_missing": 0
        }
        
        final_results = []
        
        # Re-scan NIfTI dataset
        for stem, j_info in json_map.items():
            final_counts["total_json_acquisitions"] += 1
            row = {
                "subject_id": j_info["subject_id"],
                "group": j_info["group"],
                "json_path": j_info["path"],
                "nifti_path": "",
                "status": ""
            }
            
            expected_nii = os.path.join(j_info["dir"], stem + ".nii")
            expected_nii_gz = os.path.join(j_info["dir"], stem + ".nii.gz")
            
            if os.path.exists(expected_nii_gz) or (os.path.exists(expected_nii) and stem in existing_nifti_map):
                row["status"] = "ALREADY_PRESENT"
                row["nifti_path"] = existing_nifti_map.get(stem, expected_nii_gz)
                final_counts["niftis_already_present"] += 1
            elif os.path.exists(expected_nii):
                # Need to validate it using nibabel since we just copied it
                try:
                    img = nib.load(expected_nii)
                    shape = img.shape
                    if len(shape) >= 4 and shape[3] > 1:
                        row["status"] = "SUCCESSFULLY_COPIED"
                        row["nifti_path"] = expected_nii
                        final_counts["successfully_mapped_and_copied"] += 1
                    else:
                        row["status"] = "INVALID_NIFTI"
                        final_counts["invalid_nifti"] += 1
                except:
                    row["status"] = "INVALID_NIFTI"
                    final_counts["invalid_nifti"] += 1
            else:
                row["status"] = "MISSING_NIFTI"
                final_counts["remaining_missing"] += 1
                
            final_results.append(row)
            
        with open(FINAL_CSV, 'w', newline='', encoding='utf-8') as f:
            if final_results:
                writer = csv.DictWriter(f, fieldnames=final_results[0].keys())
                writer.writeheader()
                writer.writerows(final_results)

        with open(FINAL_SUMMARY, 'w', encoding='utf-8') as f:
            f.write("--- FINAL MCI/SMC MERGE VERIFICATION ---\n")
            for k, v in final_counts.items():
                f.write(f"{k}: {v}\n")
                
        print("Merge and verification complete! Check audit/mci_smc_merge_final_summary.txt")

if __name__ == "__main__":
    main()
