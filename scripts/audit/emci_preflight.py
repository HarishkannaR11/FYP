import os, csv, hashlib, json
from collections import Counter, defaultdict
import nibabel as nib
import numpy as np

PARTICIPANTS = "/mnt/c/Users/krish/FYP/BIDS/participants.tsv"
MAPPING = "/mnt/c/Users/krish/FYP/audit/bids_session_run_mapping.csv"
BIDS = "/mnt/c/Users/krish/FYP/BIDS"
DERIV_FSFAST = "/mnt/c/Users/krish/FYP/derivatives/fsfast"
DERIV_EMCI = "/mnt/c/Users/krish/FYP/derivatives/fsfast_emci"
EXCL = "/mnt/c/Users/krish/FYP/audit/excluded_runs.csv"
OUT = "/mnt/c/Users/krish/FYP/audit/emci_rerun"

def adni_to_bids(a):
    return "sub-" + a.replace("_", "")

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1<<20), b""):
            h.update(c)
    return h.hexdigest()

stop = []
warn = []

# 1. group membership -- ONLY participants.tsv
emci = []
with open(PARTICIPANTS, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["group"] == "EMCI":
            emci.append(r["participant_id"])
emci_set = set(emci)

# 2. authoritative session/run mapping, filtered to EMCI subjects
all_map = list(csv.DictReader(open(MAPPING, newline="", encoding="utf-8")))
emci_map = [r for r in all_map if adni_to_bids(r["subject_id"]) in emci_set]
if not emci_map:
    stop.append("No EMCI rows found in authoritative bids_session_run_mapping.csv")

# exclusions
excluded = set()
if os.path.exists(EXCL):
    for r in csv.DictReader(open(EXCL, newline="", encoding="utf-8")):
        excluded.add((r["participant_id"], r["session"], r["run"]))

plan = []
seen_out = set()
seen_subj_ses_run = set()
for r in sorted(emci_map, key=lambda x: (x["subject_id"], x["BIDS_session"])):
    sub = adni_to_bids(r["subject_id"])
    ses = r["BIDS_session"]
    run = r["BIDS_run"]
    key = (sub, ses, run)
    if key in seen_subj_ses_run:
        stop.append("DUPLICATE MAPPING ROW: " + str(key))
    seen_subj_ses_run.add(key)

    if key in excluded:
        warn.append("mapped run is in excluded_runs.csv: " + str(key))

    src = BIDS + "/" + sub + "/" + ses + "/func/" + sub + "_" + ses + "_task-rest_" + run + "_bold.nii.gz"
    if not os.path.exists(src):
        src = src.replace(".nii.gz", ".nii")
    if not os.path.exists(src):
        stop.append("SOURCE MISSING: " + src)
        continue
    sjson = src.replace(".nii.gz", ".json").replace(".nii", ".json")
    if not os.path.exists(sjson):
        stop.append("SOURCE JSON MISSING: " + sjson)

    out_fsfast = DERIV_FSFAST + "/" + sub + "/" + ses + "/func/" + sub + "_" + ses + "_task-rest_" + run + "_desc-preproc_bold.nii.gz"
    out_emci = DERIV_EMCI + "/" + sub + "/" + ses + "/func/" + sub + "_" + ses + "_task-rest_" + run + "_desc-preproc_bold.nii.gz"

    if out_fsfast in seen_out or out_emci in seen_out:
        stop.append("OUTPUT COLLISION: " + sub + " " + ses + " " + run)
    seen_out.add(out_fsfast); seen_out.add(out_emci)

    try:
        img = nib.load(src)
        shp = img.shape
        z = img.header.get_zooms()
        data = img.get_fdata()
        is4d = len(shp) == 4
        nan = int(np.isnan(data).sum()); inf = int(np.isinf(data).sum())
        vox = "(%.2f,%.2f,%.2f)" % (z[0], z[1], z[2])
        tr = round(float(z[3]), 4) if len(z) > 3 else None
        vols = shp[3] if is4d else 1
        if not is4d:
            stop.append("NOT 4D: " + src)
        if nan or inf:
            stop.append("NaN/Inf in source: " + src)
    except Exception as e:
        stop.append("UNREADABLE: " + src + " -- " + str(e))
        continue

    with open(sjson) as f:
        meta = json.load(f)
    tr_json = meta.get("RepetitionTime")
    has_slicetiming = "SliceTiming" in meta

    already_fsfast = os.path.exists(out_fsfast)
    already_emci = os.path.exists(out_emci)

    plan.append({
        "participant_id": sub, "session": ses, "run": run,
        "source_file": src, "source_json": sjson,
        "shape": str(shp), "volumes": vols, "voxel_size": vox,
        "TR_nifti": tr, "TR_json": tr_json, "has_SliceTiming": has_slicetiming,
        "md5_before": md5(src),
        "output_fsfast": out_fsfast, "output_emci": out_emci,
        "already_processed": "YES" if (already_fsfast or already_emci) else "NO",
        "excluded": "YES" if key in excluded else "NO",
    })

# 3. slice-timing re-check across EMCI JSON sidecars specifically
n_with_slicetiming = sum(1 for p in plan if p["has_SliceTiming"])

# 4. existing derivatives check -- subject+session+run level, both possible dirs
existing_count = sum(1 for p in plan if p["already_processed"] == "YES")

# 5. resources
import shutil, subprocess
disk = shutil.disk_usage("/home/harish")
mem = subprocess.run(["free", "-m"], capture_output=True, text=True).stdout
ncpu = os.cpu_count()

os.makedirs(OUT, exist_ok=True)
with open(OUT + "/emci_execution_plan.csv", "w", newline="") as f:
    fieldnames = list(plan[0].keys()) if plan else []
    w = csv.DictWriter(f, fieldnames=fieldnames)
    w.writeheader()
    for p in plan:
        w.writerow(p)

with open(OUT + "/emci_source_checksums_before.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["participant_id","session","run","source_file","md5_before"])
    for p in plan:
        w.writerow([p["participant_id"], p["session"], p["run"], p["source_file"], p["md5_before"]])

print("Total EMCI subjects (participants.tsv):", len(emci))
print("Total EMCI rows in authoritative mapping:", len(emci_map))
print("Total EMCI acquisitions planned:", len(plan))
print("Unique output paths:", len(seen_out)//2 if seen_out else 0)
print()
print("%-14s %-7s %-7s %-45s %-20s %-5s %-8s %-8s %-6s %s" % ("Subject","Session","Run","SourceFile","Dimensions","Vols","TR_nii","TR_json","SlcTim","AlreadyProc"))
for p in plan:
    print("%-14s %-7s %-7s %-45s %-20s %-5s %-8s %-8s %-6s %s" % (p["participant_id"], p["session"], p["run"], os.path.basename(p["source_file"]), p["shape"], p["volumes"], p["TR_nifti"], p["TR_json"], p["has_SliceTiming"], p["already_processed"]))
print()
print("Distinct dimensions:", dict(Counter(p["shape"] for p in plan)))
print("Distinct volumes:", dict(Counter(p["volumes"] for p in plan)))
print("Distinct TR (json):", dict(Counter(str(p["TR_json"]) for p in plan)))
print("Sessions per subject:", dict(Counter(Counter(p["participant_id"] for p in plan).values())))
print()
print("Subjects with SliceTiming present:", n_with_slicetiming, "of", len(plan), "acquisitions")
print()
print("Already processed (subject+session+run level):", existing_count)
print("Remaining:", len(plan) - existing_count)
print("Excluded (in excluded_runs.csv):", sum(1 for p in plan if p["excluded"]=="YES"))
print()
print("=== RESOURCES ===")
print("CPUs:", ncpu)
print("Disk free (GB):", round(disk.free / 1e9, 1))
print(mem)
print("=== STOP CONDITIONS ===")
print("\n".join(stop) if stop else "none")
print("=== WARNINGS ===")
print("\n".join(warn) if warn else "none")
