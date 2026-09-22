"""
READ-ONLY: build the AD acquisition inventory for the 4mm/6mm production run.
Independently verifies every acquisition's input validity (not just existence).
"""
import csv
import os
import numpy as np
import nibabel as nib

MANIFEST = "/mnt/c/Users/krish/FYP/audit/ad_rerun/acquisition_manifest_final.csv"
OUT = "/mnt/c/Users/krish/FYP/derivatives/fsfast/AD_preprocessing_inventory.csv"

rows = list(csv.DictReader(open(MANIFEST, newline="", encoding="utf-8")))
print(f"Manifest rows: {len(rows)}")

out_rows = []
for r in rows:
    sub, ses, run = r["subject_id"], r["session_id"], r["run_id"]
    bold_path = r["input_path"]
    json_path = r["input_json"]
    status, reason = "PROCESSABLE", "OK"
    try:
        img = nib.load(bold_path)
        d = img.get_fdata(dtype=np.float32)
        errs = []
        if d.ndim != 4:
            errs.append("not 4D")
        if np.isnan(d).any():
            errs.append("NaN")
        if np.isinf(d).any():
            errs.append("Inf")
        if not np.isfinite(img.affine).all() or abs(np.linalg.det(img.affine)) < 1e-9:
            errs.append("invalid affine")
        TR = float(img.header.get_zooms()[3])
        if not (TR and np.isfinite(TR) and TR > 0):
            errs.append("invalid TR")
        if not np.any(d != 0):
            errs.append("empty")
        if not os.path.isfile(json_path):
            errs.append("missing JSON sidecar")
        if errs:
            status, reason = "FAIL", "; ".join(errs)
    except Exception as e:
        status, reason = "FAIL", f"cannot load: {e}"

    out_rows.append({
        "subject": sub, "session": ses, "run": run,
        "source_nifti": bold_path, "source_json": json_path,
        "status": status, "reason": reason,
    })

with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["subject", "session", "run", "source_nifti", "source_json", "status", "reason"])
    w.writeheader()
    w.writerows(out_rows)

n_proc = sum(1 for r in out_rows if r["status"] == "PROCESSABLE")
n_fail = sum(1 for r in out_rows if r["status"] == "FAIL")
print(f"Total: {len(out_rows)}  PROCESSABLE: {n_proc}  FAIL: {n_fail}")
print(f"Saved: {OUT}")

claimed_unrecoverable = [
    "002_S_5018_Resting_State_fMRI_20130516122033_501",
    "006_S_4153_WIP_Resting_State_fMRI_20120327121013_501",
    "019_S_4477_Resting_State_fMRI_20130213122116_501",
    "019_S_4549_Resting_State_fMRI_20130313113703_601",
    "130_S_4730_WIP_Resting_State_fMRI_20130612085835_701",
    "130_S_4982_Resting_State_fMRI_20131025101621_501",
]
print("\nVerification of the 6 acquisitions previously claimed unrecoverable "
      "(checked against current BIDS-derived inventory, not assumed):")
for c in claimed_unrecoverable:
    hit = next((r for r in out_rows if r["subject"].replace("sub-", "").replace("S", "_S_", 1) in c
                or c.split("_", 2)[0] + "S" + c.split("_", 3)[2] in r["subject"]), None)
    if hit:
        print(f"  {c}: FOUND in BIDS as {hit['subject']}/{hit['session']} -> status={hit['status']}")
    else:
        print(f"  {c}: no clear match found by name heuristic -- see full CSV for manual cross-check")
print("INVENTORY_BUILD_DONE")
