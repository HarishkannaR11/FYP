import os, csv, hashlib, json
from collections import Counter, defaultdict
import nibabel as nib
import numpy as np

PARTICIPANTS = "/mnt/c/Users/krish/FYP/BIDS/participants.tsv"
MAPPING = "/mnt/c/Users/krish/FYP/audit/ad_session_mapping_final.csv"
BIDS = "/mnt/c/Users/krish/FYP/BIDS"
DERIV = "/mnt/c/Users/krish/FYP/derivatives/fsfast"
EXCL = "/mnt/c/Users/krish/FYP/audit/excluded_runs.csv"
OUT = "/mnt/c/Users/krish/FYP/audit/ad_rerun"

EXPECTED = ["sub-002S5018","sub-006S4153","sub-006S4192","sub-013S5071","sub-018S4696",
"sub-018S4733","sub-019S4252","sub-019S4477","sub-019S4549","sub-019S5012","sub-019S5019",
"sub-031S4024","sub-130S4641","sub-130S4660","sub-130S4730","sub-130S4971","sub-130S4982",
"sub-130S4984","sub-130S4990","sub-130S5059"]

stop = []
warn = []

def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1<<20), b""):
            h.update(c)
    return h.hexdigest()

# 1. group membership -- ONLY participants.tsv
ad = []
with open(PARTICIPANTS, newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        if r["group"] == "AD":
            ad.append(r["participant_id"])
ad_sorted = sorted(ad)
if ad_sorted != sorted(EXPECTED):
    stop.append("AD subject list from participants.tsv does not match expected list: extra=" + str(sorted(set(ad_sorted)-set(EXPECTED))) + " missing=" + str(sorted(set(EXPECTED)-set(ad_sorted))))

# 2. authoritative mapping
rows = [r for r in csv.DictReader(open(MAPPING, newline="", encoding="utf-8")) if r["participant_id"] in ad]
if len(rows) != 25:
    stop.append("expected 25 AD acquisitions in mapping, found " + str(len(rows)))

# exclusions
excluded = set()
if os.path.exists(EXCL):
    for r in csv.DictReader(open(EXCL, newline="", encoding="utf-8")):
        excluded.add((r["participant_id"], r["session"], r["run"]))
ad_excluded = [k for k in excluded if k[0] in ad]

# 3. per-acquisition header check + checksum
plan = []
seen_out = set()
for r in sorted(rows, key=lambda x: (x["participant_id"], x["session"])):
    sub, ses = r["participant_id"], r["session"]
    run = "run-01"
    key = (sub, ses, run)
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
    out = DERIV + "/" + sub + "/" + ses + "/func/" + sub + "_" + ses + "_task-rest_" + run + "_desc-preproc_bold.nii.gz"
    if out in seen_out:
        stop.append("OUTPUT COLLISION: " + out)
    seen_out.add(out)

    try:
        img = nib.load(src)
        shp = img.shape
        z = img.header.get_zooms()
        data = img.get_fdata()
        readable = True
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

    existing_out = os.path.exists(out)
    plan.append({
        "participant_id": sub, "session": ses, "run": run,
        "input": src, "input_json": sjson,
        "dimensions": str(shp), "volumes": vols, "voxel_size": vox,
        "TR_nifti": tr, "TR_json": tr_json,
        "md5_before": md5(src),
        "output": out, "existing_output": "YES" if existing_out else "NO",
    })

# 4. existing derivatives inventory for AD subjects
existing = {"subject_dirs": [], "func_files": [], "logs": [], "qc": []}
for sub in ad:
    d = DERIV + "/" + sub
    if os.path.isdir(d):
        existing["subject_dirs"].append(d)
        for root, _, files in os.walk(d):
            for fn in files:
                existing["func_files"].append(os.path.join(root, fn))
logs_dir = DERIV + "/logs"
if os.path.isdir(logs_dir):
    for fn in sorted(os.listdir(logs_dir)):
        if any(fn.startswith(s) for s in ad) or fn == "preprocessing.log":
            existing["logs"].append(os.path.join(logs_dir, fn))
qc_dir = DERIV + "/qc"
if os.path.isdir(qc_dir):
    for fn in sorted(os.listdir(qc_dir)):
        existing["qc"].append(os.path.join(qc_dir, fn))
non_ad_dirs = sorted(d for d in os.listdir(DERIV) if d.startswith("sub-") and d not in ad)

# 5. resources
import shutil, subprocess
disk = shutil.disk_usage("/home/harish")
mem = subprocess.run(["free", "-m"], capture_output=True, text=True).stdout
ncpu = os.cpu_count()

# write outputs
os.makedirs(OUT, exist_ok=True)
with open(OUT + "/source_checksums_before.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["participant_id","session","run","input","md5_before"])
    w.writeheader()
    for p in plan:
        w.writerow({k: p[k] for k in ["participant_id","session","run","input","md5_before"]})

with open(OUT + "/execution_plan.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(plan[0].keys()))
    w.writeheader(); w.writerows(plan)

with open(OUT + "/existing_ad_derivatives.json", "w") as f:
    json.dump(existing, f, indent=2)

# print plan table
print("AD subjects from participants.tsv:", len(ad), "-- matches expected:", ad_sorted == sorted(EXPECTED))
print("Acquisitions in verified mapping:", len(rows))
print("Acquisitions planned:", len(plan))
print("Unique output paths:", len(seen_out))
print()
print("%-14s %-7s %-45s %-20s %-5s %-8s %-8s %s" % ("Subject","Session","Input","Dimensions","Vols","TR_nii","TR_json","ExistingOut"))
for p in plan:
    print("%-14s %-7s %-45s %-20s %-5s %-8s %-8s %s" % (p["participant_id"], p["session"], os.path.basename(p["input"]), p["dimensions"], p["volumes"], p["TR_nifti"], p["TR_json"], p["existing_output"]))
print()
print("Distinct dimensions:", dict(Counter(p["dimensions"] for p in plan)))
print("Distinct volumes:", dict(Counter(p["volumes"] for p in plan)))
print("Distinct TR (json):", dict(Counter(str(p["TR_json"]) for p in plan)))
print("Sessions per subject:", dict(Counter(Counter(p["participant_id"] for p in plan).values())))
print()
print("=== EXISTING AD DERIVATIVES THAT WOULD BE AFFECTED ===")
print("AD subject dirs:", len(existing["subject_dirs"]))
print("files under those dirs:", len(existing["func_files"]))
print("AD-related log files:", len(existing["logs"]))
print("qc files (shared, contain AD rows):", [os.path.basename(x) for x in existing["qc"]])
print("NON-AD subject dirs in derivatives (must NOT be touched):", non_ad_dirs)
print()
print("=== RESOURCES ===")
print("CPUs:", ncpu)
print("Disk free (GB):", round(disk.free / 1e9, 1))
print(mem)
print("=== EXCLUSIONS affecting AD ===", ad_excluded if ad_excluded else "none")
print()
print("=== STOP CONDITIONS ===")
print("\n".join(stop) if stop else "none")
print("=== WARNINGS ===")
print("\n".join(warn) if warn else "none")
