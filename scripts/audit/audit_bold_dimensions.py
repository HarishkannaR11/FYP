import os
import json
import csv
import nibabel as nib

BIDS_DIR = r"C:\Users\krish\FYP\BIDS"
AUDIT_DIR = r"C:\Users\krish\FYP\audit"
CSV_REPORT = os.path.join(AUDIT_DIR, "bold_dimensions_inventory.csv")
TXT_SUMMARY = os.path.join(AUDIT_DIR, "bold_dimensions_summary.txt")

def main():
    print("Starting BOLD dimensional audit...")

    # Load participants.tsv to get groups
    groups = {}
    tsv_path = os.path.join(BIDS_DIR, "participants.tsv")
    try:
        with open(tsv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter='\t')
            headers = next(reader)
            for row in reader:
                if len(row) >= 2:
                    groups[row[0]] = row[1]
    except Exception as e:
        print(f"Error reading participants.tsv: {e}")

    inventory = []
    
    unique_spatial = set()
    unique_voxel = set()
    unique_volumes = set()
    unique_tr = set()
    
    group_stats = {
        "AD": {"count": 0, "4D": 0, "anomalies": 0},
        "LMCI": {"count": 0, "4D": 0, "anomalies": 0},
        "MCI": {"count": 0, "4D": 0, "anomalies": 0},
        "EMCI": {"count": 0, "4D": 0, "anomalies": 0},
        "CN_Final": {"count": 0, "4D": 0, "anomalies": 0},
        "SMC_Final": {"count": 0, "4D": 0, "anomalies": 0},
        "UNKNOWN": {"count": 0, "4D": 0, "anomalies": 0}
    }

    # Crawl BIDS
    for root, _, files in os.walk(BIDS_DIR):
        for f in files:
            if f.endswith(".nii") or f.endswith(".nii.gz"):
                nii_path = os.path.join(root, f)
                stem = f.replace(".nii.gz", "").replace(".nii", "")
                
                # Parse BIDS filename
                parts = stem.split('_')
                subject = next((p for p in parts if p.startswith('sub-')), "UNKNOWN")
                session = next((p for p in parts if p.startswith('ses-')), "UNKNOWN")
                run = next((p for p in parts if p.startswith('run-')), "UNKNOWN")
                group = groups.get(subject, "UNKNOWN")
                
                json_path = os.path.join(root, stem + ".json")
                
                # Default values
                is_readable = False
                dimensionality = "ERROR"
                shape_str = ""
                T = 0
                vox_dim = ""
                nii_tr = "UNKNOWN"
                json_tr = "UNKNOWN"
                tr_agree = "N/A"
                anomaly = False
                
                # Read JSON TR
                if os.path.exists(json_path):
                    try:
                        with open(json_path, 'r') as jf:
                            jdata = json.load(jf)
                            json_tr = jdata.get("RepetitionTime", "UNKNOWN")
                    except:
                        pass
                
                # Read NIfTI
                try:
                    img = nib.load(nii_path)
                    hdr = img.header
                    shape = img.shape
                    
                    is_readable = True
                    if len(shape) >= 4:
                        dimensionality = "4D"
                        T = shape[3]
                        shape_str = f"({shape[0]}, {shape[1]}, {shape[2]}, {shape[3]})"
                    elif len(shape) == 3:
                        dimensionality = "3D"
                        T = 1
                        shape_str = f"({shape[0]}, {shape[1]}, {shape[2]})"
                        anomaly = True
                    
                    # Zoom/Voxel Sizes
                    zooms = hdr.get_zooms()
                    if len(zooms) >= 3:
                        vox_dim = f"({zooms[0]:.2f}, {zooms[1]:.2f}, {zooms[2]:.2f})"
                    
                    # NIfTI TR
                    try:
                        nii_tr_val = zooms[3] if len(zooms) >= 4 else "UNKNOWN"
                        if isinstance(nii_tr_val, (int, float)):
                            nii_tr = round(nii_tr_val, 4)
                        else:
                            nii_tr = nii_tr_val
                    except:
                        nii_tr = "UNKNOWN"
                        
                    # Agreement
                    if json_tr != "UNKNOWN" and nii_tr != "UNKNOWN":
                        try:
                            if abs(float(json_tr) - float(nii_tr)) < 0.01:
                                tr_agree = "YES"
                            else:
                                tr_agree = "NO"
                                anomaly = True
                        except:
                            tr_agree = "ERROR"
                            
                    unique_spatial.add(f"({shape[0]}, {shape[1]}, {shape[2]})")
                    unique_voxel.add(vox_dim)
                    unique_volumes.add(T)
                    if json_tr != "UNKNOWN": unique_tr.add(json_tr)
                    
                except Exception as e:
                    anomaly = True
                    print(f"Error reading {f}: {e}")
                
                # Record
                inventory.append({
                    "participant_id": subject,
                    "group": group,
                    "session": session,
                    "run": run,
                    "filename": f,
                    "dimensionality": dimensionality,
                    "shape": shape_str,
                    "volumes": T,
                    "voxel_dimensions": vox_dim,
                    "nifti_tr": nii_tr,
                    "json_tr": json_tr,
                    "tr_match": tr_agree,
                    "readable": is_readable
                })
                
                # Stats
                if group in group_stats:
                    group_stats[group]["count"] += 1
                    if dimensionality == "4D": group_stats[group]["4D"] += 1
                    if anomaly: group_stats[group]["anomalies"] += 1

    print("Writing reports...")
    
    # CSV
    with open(CSV_REPORT, 'w', newline='', encoding='utf-8') as f:
        if inventory:
            writer = csv.DictWriter(f, fieldnames=inventory[0].keys())
            writer.writeheader()
            writer.writerows(inventory)

    # Calculate Majorities
    spatial_list = [i["shape"] for i in inventory if "readable" in i and i["readable"]]
    vol_list = [i["volumes"] for i in inventory if "readable" in i and i["readable"]]
    
    maj_spatial = max(set(spatial_list), key=spatial_list.count) if spatial_list else "None"
    maj_vol = max(set(vol_list), key=vol_list.count) if vol_list else "None"
    
    anomalies = [i for i in inventory if i["shape"] != maj_spatial or i["volumes"] != maj_vol or i["tr_match"] == "NO" or not i["readable"]]

    # TXT Summary
    with open(TXT_SUMMARY, 'w', encoding='utf-8') as f:
        f.write("=== BOLD DIMENSIONAL AUDIT SUMMARY ===\n\n")
        f.write(f"Total NIfTI Files: {len(inventory)}\n")
        f.write(f"Readable NIfTIs: {sum(1 for x in inventory if x['readable'])}\n")
        f.write(f"3D Files: {sum(1 for x in inventory if x['dimensionality'] == '3D')}\n")
        f.write(f"4D Files: {sum(1 for x in inventory if x['dimensionality'] == '4D')}\n\n")
        
        f.write("--- DATASET VARIANCE ---\n")
        f.write(f"Unique Spatial Dimensions: {sorted(list(unique_spatial))}\n")
        f.write(f"Unique Voxel Sizes: {sorted(list(unique_voxel))}\n")
        f.write(f"Unique Volume Counts (T): {sorted(list(unique_volumes))}\n")
        f.write(f"Unique TR Values: {sorted(list(unique_tr))}\n\n")
        
        f.write("--- GROUP STATISTICS ---\n")
        for g, stats in group_stats.items():
            if stats["count"] > 0:
                f.write(f"{g}:\n")
                f.write(f"  Total Files: {stats['count']}\n")
                f.write(f"  4D Files: {stats['4D']}\n")
                f.write(f"  Dimensional Anomalies: {stats['anomalies']}\n")
        
        f.write("\n--- DEVIATIONS & ANOMALIES ---\n")
        f.write(f"Majority Shape: {maj_spatial}\n")
        f.write(f"Majority Volumes: {maj_vol}\n\n")
        
        if not anomalies:
            f.write("No files deviate from the majority pattern.\n")
        else:
            for a in anomalies:
                f.write(f"Participant: {a['participant_id']} | Session: {a['session']} | File: {a['filename']}\n")
                f.write(f"  Shape: {a['shape']} | Volumes: {a['volumes']} | TR Match: {a['tr_match']}\n")

    print(f"Audit complete. Summary written to {TXT_SUMMARY}")

if __name__ == "__main__":
    main()
