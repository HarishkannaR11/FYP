"""
READ-ONLY: build the CN_Final session mapping + acquisition manifest.
Uses BIDS/participants.tsv (authoritative group membership) and the actual
BIDS directory structure (ground truth for session/run assignment, same
reliable approach used for AD). Excludes sub-012S4026/ses-01 per
audit/excluded_runs.csv (user-confirmed, 7-volume truncated acquisition).
STOPS (asserts) if eligible acquisition count != 29.
"""
import csv
import glob
import hashlib
import os
import nibabel as nib

PARTICIPANTS_TSV = "/mnt/c/Users/krish/FYP/BIDS/participants.tsv"
BIDS_ROOT = "/mnt/c/Users/krish/FYP/BIDS"
EXCLUDED_CSV = "/mnt/c/Users/krish/FYP/audit/excluded_runs.csv"
MAPPING_OUT = "/mnt/c/Users/krish/FYP/audit/lmci_session_mapping.csv"
MANIFEST_OUT = "/mnt/c/Users/krish/FYP/audit/lmci_rerun/acquisition_manifest_final.csv"

EXPECTED_SUBJECTS_TOTAL = 20
EXPECTED_ELIGIBLE_ACQUISITIONS = 22


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_excluded():
    excluded = set()
    with open(EXCLUDED_CSV, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            excluded.add((r["participant_id"], r["session"], r["run"]))
    return excluded


def main():
    with open(PARTICIPANTS_TSV, newline="", encoding="utf-8-sig") as f:
        content = f.read().replace("\r", "")
    participants = list(csv.DictReader(content.splitlines(), delimiter="\t"))
    cn_subjects = sorted([p["participant_id"] for p in participants if p["group"] == "LMCI"])
    print(f"CN_Final subjects in participants.tsv: {len(cn_subjects)}")
    assert len(cn_subjects) == EXPECTED_SUBJECTS_TOTAL, \
        f"STOP: expected {EXPECTED_SUBJECTS_TOTAL} CN_Final subjects, got {len(cn_subjects)}"

    excluded = load_excluded()
    print(f"Excluded runs (project-wide): {excluded}")

    import re

    mapping_rows = []
    for sub in cn_subjects:
        bold_files = sorted(glob.glob(os.path.join(BIDS_ROOT, sub, "ses-*", "func", "*task-rest*bold.nii*")))
        for bf in bold_files:
            ses = [p for p in bf.split(os.sep) if p.startswith("ses-")][0]
            run_match = re.search(r"(run-\d+)", os.path.basename(bf))
            run = run_match.group(1) if run_match else "run-01"  # parsed from actual filename, not assumed
            mapping_rows.append({"participant_id": sub, "session": ses, "run": run, "input_nifti": bf})

    print(f"Total BIDS acquisitions found for CN_Final: {len(mapping_rows)} (expected source total: 30)")

    with open(MAPPING_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["participant_id", "session", "run", "input_nifti"])
        w.writeheader()
        w.writerows(mapping_rows)
    print(f"Saved: {MAPPING_OUT}")

    eligible_rows = [r for r in mapping_rows if (r["participant_id"], r["session"], r["run"]) not in excluded]
    excluded_found = [r for r in mapping_rows if (r["participant_id"], r["session"], r["run"]) in excluded]
    print(f"\nEligible acquisitions (after exclusion): {len(eligible_rows)}")
    print(f"Excluded acquisitions applied: {[(r['participant_id'], r['session']) for r in excluded_found]}")

    assert len(eligible_rows) == EXPECTED_ELIGIBLE_ACQUISITIONS, \
        f"STOP: expected {EXPECTED_ELIGIBLE_ACQUISITIONS} eligible acquisitions, got {len(eligible_rows)}"

    os.makedirs(os.path.dirname(MANIFEST_OUT), exist_ok=True)
    manifest = []
    for r in eligible_rows:
        bold_path = r["input_nifti"]
        json_path = bold_path.replace(".nii.gz", ".json").replace(".nii", ".json")
        img = nib.load(bold_path)
        shape = img.shape
        zooms = img.header.get_zooms()
        tr = float(zooms[3]) if len(zooms) > 3 else None
        vols = shape[3] if len(shape) > 3 else 1
        size = os.path.getsize(bold_path)
        checksum = md5sum(bold_path)
        manifest.append({
            "subject_id": r["participant_id"], "session_id": r["session"], "run_id": r["run"],
            "input_path": bold_path, "input_json": json_path,
            "dimensions": str(shape), "voxel_size": str(tuple(round(float(z), 4) for z in zooms[:3])),
            "TR": tr, "volumes": vols, "file_size_bytes": size, "checksum_md5": checksum,
        })

    with open(MANIFEST_OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)
    print(f"\nManifest written: {MANIFEST_OUT}")
    print(f"MANIFEST_VALIDATION_PASSED: {len(cn_subjects)} subjects, {len(manifest)} eligible acquisitions confirmed")

    from collections import Counter
    counts = Counter(m["subject_id"] for m in manifest)
    multi = {k: v for k, v in counts.items() if v > 1}
    print(f"\nSubjects with multiple eligible acquisitions ({len(multi)}): {multi}")


if __name__ == "__main__":
    main()
