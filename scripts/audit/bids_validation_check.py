import os
import csv
from collections import defaultdict, Counter

BIDS = "/mnt/c/Users/krish/FYP/BIDS"
OUT_DIR = "/mnt/c/Users/krish/FYP/audit/bids_conversion"
MAPPING_CSV = os.path.join(OUT_DIR, "bids_mapping.csv")
REPORT = os.path.join(OUT_DIR, "validation_report.txt")

lines = []

def w(s=""):
    lines.append(s)
    print(s)

w("=== BIDS DATASET VALIDATION REPORT ===")
w()
w("Dataset validated: " + BIDS)
w("Scope: the EXISTING BIDS dataset, cross-checked against an independently")
w("re-derived mapping built from the raw NIfTI_/NIfTI source tree.")
w()

mapping = list(csv.DictReader(open(MAPPING_CSV, newline="", encoding="utf-8")))
mapped = [r for r in mapping if r["mapping_status"] == "MAPPED"]

w("--- 1. Required / recommended top-level files ---")
checks = {
    "dataset_description.json": os.path.exists(os.path.join(BIDS, "dataset_description.json")),
    "participants.tsv": os.path.exists(os.path.join(BIDS, "participants.tsv")),
    "participants.json": os.path.exists(os.path.join(BIDS, "participants.json")),
    "README": os.path.exists(os.path.join(BIDS, "README")),
}
for k, v in checks.items():
    status = "PRESENT" if v else "MISSING"
    req = "REQUIRED" if k == "dataset_description.json" else "RECOMMENDED"
    w("  " + k + ": " + status + " (" + req + ")")
w()

bold_nii = []
bold_json = []
for root, _, files in os.walk(BIDS):
    for f in files:
        if f.endswith("_bold.nii") or f.endswith("_bold.nii.gz"):
            bold_nii.append(os.path.join(root, f))
        elif f.endswith("_bold.json"):
            bold_json.append(os.path.join(root, f))

w("--- 2. File counts ---")
w("  Source NIfTI files (NIfTI_/NIfTI tree): " + str(len(mapping)))
w("  BOLD NIfTI files in BIDS: " + str(len(bold_nii)))
w("  BOLD JSON sidecars in BIDS: " + str(len(bold_json)))
w("  Successfully mapped source -> BIDS target: " + str(len(mapped)))
missing_targets = [r for r in mapped if r["matches_existing_bids"] == "NOT_PRESENT"]
w("  Mapped targets NOT present in BIDS: " + str(len(missing_targets)))
w("  Orphan JSON (json without matching nii): " + str(max(0, len(bold_json) - len(bold_nii))))
w()

w("--- 3. BIDS filename compliance ---")
try:
    from bids_validator import BIDSValidator
    v = BIDSValidator()
    bad = []
    for p in bold_nii + bold_json:
        rel = "/" + os.path.relpath(p, BIDS).replace(os.sep, "/")
        if not v.is_bids(rel):
            bad.append(rel)
    w("  Files checked: " + str(len(bold_nii) + len(bold_json)))
    w("  Non-compliant filenames: " + str(len(bad)))
    for b in bad[:20]:
        w("      " + b)
    w("  NOTE: this is the Python bids_validator module, which validates FILENAME/PATH")
    w("  compliance against the BIDS spec only. It is NOT the full official Node.js")
    w("  bids-validator (which additionally checks JSON field contents, cross-file")
    w("  consistency and dataset-level rules). The official validator is not installed")
    w("  here -- reported honestly rather than claiming full validation.")
except Exception as e:
    w("  bids_validator unavailable or failed: " + str(e))
w()

w("--- 4. participants.tsv consistency ---")
ptsv = {}
with open(os.path.join(BIDS, "participants.tsv"), newline="", encoding="utf-8") as f:
    for r in csv.DictReader(f, delimiter="\t"):
        ptsv[r["participant_id"]] = r.get("group", "")
subj_dirs = set(d for d in os.listdir(BIDS) if d.startswith("sub-") and os.path.isdir(os.path.join(BIDS, d)))
w("  participants.tsv entries: " + str(len(ptsv)))
w("  sub-* directories on disk: " + str(len(subj_dirs)))
in_tsv_not_disk = sorted(set(ptsv) - subj_dirs)
on_disk_not_tsv = sorted(subj_dirs - set(ptsv))
w("  In participants.tsv but no directory: " + str(len(in_tsv_not_disk)) + " " + str(in_tsv_not_disk[:10]))
w("  Directory present but not in participants.tsv: " + str(len(on_disk_not_tsv)) + " " + str(on_disk_not_tsv[:10]))

src_group = {}
for r in mapping:
    src_group.setdefault(r["participant_id"], r["group"])
mismatched = [(s, ptsv.get(s), src_group.get(s)) for s in sorted(src_group) if s in ptsv and ptsv[s] != src_group[s]]
w("  Group label mismatches (participants.tsv vs source folder): " + str(len(mismatched)))
for m in mismatched[:10]:
    w("      " + str(m[0]) + ": participants.tsv=" + str(m[1]) + " source_folder=" + str(m[2]))
w()

w("--- 5. Duplicate content (identical MD5) ---")
dups = [r for r in mapping if r["duplicate_status"].startswith("DUPLICATE")]
if dups:
    w("  " + str(len(dups)) + " source files have byte-identical duplicates:")
    for r in dups:
        w("      " + r["participant_id"] + " " + r["session"] + " " + r["run"] + " (" + r["group"] + ") -- " + os.path.basename(r["source_file"]))
    w("  IMPACT: these are currently represented as SEPARATE runs in the BIDS dataset,")
    w("  which inflates the acquisition count and would feed the same scan twice into")
    w("  any downstream analysis. Flagged for review -- NOT auto-removed.")
else:
    w("  None.")
w()

w("--- 6. Metadata consistency across mapped BOLD files ---")
shapes = Counter(r["shape"] for r in mapped)
trs = Counter(str(r["TR"]) for r in mapped)
vols = Counter(str(r["volumes"]) for r in mapped)
w("  Distinct shapes: " + str(len(shapes)))
for s, c in shapes.most_common():
    w("      " + s + ": " + str(c) + " runs")
w("  Distinct TR values: " + str(len(trs)))
for t, c in trs.most_common():
    w("      TR=" + t + ": " + str(c) + " runs")
w("  Distinct volume counts: " + str(len(vols)))
for v2, c in vols.most_common():
    w("      " + v2 + " volumes: " + str(c) + " runs")
w()

w("--- 7. Session / run structure ---")
sess = defaultdict(set)
runs = defaultdict(int)
for r in mapped:
    sess[r["participant_id"]].add(r["session"])
    runs[(r["participant_id"], r["session"])] += 1
w("  Subjects with BOLD data: " + str(len(sess)))
dist = Counter(len(v) for v in sess.values())
for n in sorted(dist):
    w("      " + str(dist[n]) + " subjects with " + str(n) + " session(s)")
rdist = Counter(runs.values())
for n in sorted(rdist):
    w("      " + str(rdist[n]) + " sessions with " + str(n) + " run(s)")
w()
multi = dict((s, v) for s, v in sess.items() if len(v) > 1)
multi_run = dict((k, v) for k, v in runs.items() if v > 1)
w("  Subjects with MULTIPLE sessions: " + str(len(multi)))
w("  Sessions with MULTIPLE runs: " + str(len(multi_run)))
for k, v in multi_run.items():
    w("      " + k[0] + " " + k[1] + ": " + str(v) + " runs")
w()

w("--- 8. Overall ---")
problems = []
if not checks["dataset_description.json"]:
    problems.append("dataset_description.json missing (REQUIRED)")
if not checks["participants.json"]:
    problems.append("participants.json missing (recommended)")
if not checks["README"]:
    problems.append("README missing (recommended)")
if missing_targets:
    problems.append(str(len(missing_targets)) + " mapped source files have no BIDS target")
if dups:
    problems.append(str(len(dups)) + " duplicate-content files represented as distinct runs")
if on_disk_not_tsv or in_tsv_not_disk:
    problems.append("participants.tsv / directory mismatch")
if mismatched:
    problems.append(str(len(mismatched)) + " group-label mismatches")

if problems:
    w("  ISSUES FOUND:")
    for p in problems:
        w("    - " + p)
else:
    w("  No issues found.")

with open(REPORT, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print("")
print("Written: " + REPORT)
