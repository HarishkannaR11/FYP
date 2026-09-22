"""
READ-ONLY: build the AD acquisition manifest before any production processing.
Uses BIDS/participants.tsv (authoritative group membership) and the existing
verified session mapping (audit/ad_session_mapping_final.csv) rather than
re-deriving from scratch. STOPS (asserts) if acquisition count != 25.
"""
import csv
import hashlib
import nibabel as nib

MAPPING_CSV = "/mnt/c/Users/krish/FYP/audit/ad_session_mapping_final.csv"
OUT_CSV = "/mnt/c/Users/krish/FYP/audit/ad_rerun/acquisition_manifest_final.csv"

EXPECTED_SUBJECTS = 20
EXPECTED_ACQUISITIONS = 25


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main():
    rows = list(csv.DictReader(open(MAPPING_CSV, newline="", encoding="utf-8")))
    print(f"Loaded {len(rows)} rows from {MAPPING_CSV}")

    manifest = []
    subjects = set()
    for r in rows:
        sub, ses = r["participant_id"], r["session"]
        run = "run-01"
        subjects.add(sub)
        bold_path = r["input_nifti"]
        json_path = r["input_json"]

        img = nib.load(bold_path)
        shape = img.shape
        zooms = img.header.get_zooms()
        tr = float(zooms[3]) if len(zooms) > 3 else None
        vols = shape[3] if len(shape) > 3 else 1
        import os
        size = os.path.getsize(bold_path)
        checksum = md5sum(bold_path)

        manifest.append({
            "subject_id": sub, "session_id": ses, "run_id": run,
            "input_path": bold_path, "input_json": json_path,
            "dimensions": str(shape), "voxel_size": str(tuple(round(float(z), 4) for z in zooms[:3])),
            "TR": tr, "volumes": vols, "file_size_bytes": size, "checksum_md5": checksum,
        })

    print(f"\nUnique subjects: {len(subjects)} (expected {EXPECTED_SUBJECTS})")
    print(f"Total acquisitions: {len(manifest)} (expected {EXPECTED_ACQUISITIONS})")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
        w.writeheader()
        w.writerows(manifest)
    print(f"\nManifest written: {OUT_CSV}")

    assert len(subjects) == EXPECTED_SUBJECTS, f"STOP: expected {EXPECTED_SUBJECTS} unique subjects, got {len(subjects)}"
    assert len(manifest) == EXPECTED_ACQUISITIONS, f"STOP: expected {EXPECTED_ACQUISITIONS} acquisitions, got {len(manifest)}"
    print("\nMANIFEST_VALIDATION_PASSED: 20 subjects, 25 acquisitions confirmed")

    # subjects with 2 acquisitions
    from collections import Counter
    counts = Counter(m["subject_id"] for m in manifest)
    multi = {k: v for k, v in counts.items() if v > 1}
    print(f"\nSubjects with multiple acquisitions ({len(multi)}):")
    for k, v in multi.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
