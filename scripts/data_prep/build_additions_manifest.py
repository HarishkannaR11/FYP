"""
Build an acquisition manifest for newly added acquisitions, in the exact
format the production pipeline (pipeline/fsfast_production_4mm6mm.py)
expects.

Every field is computed from the BIDS file on disk -- nothing is copied
from another manifest and nothing is invented. The existing group manifests
are never modified.

    python scripts/data_prep/build_additions_manifest.py
"""
import csv
import hashlib
import json
import os

import nibabel as nib

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIDS = os.path.join(ROOT, "BIDS")
OUTDIR = os.path.join(ROOT, "audit", "mci_additions")
OUT = os.path.join(OUTDIR, "acquisition_manifest_final.csv")
WSL_BIDS = "/mnt/c/Users/krish/FYP/BIDS"

COLUMNS = ["subject_id", "session_id", "run_id", "input_path", "input_json",
           "dimensions", "voxel_size", "TR", "volumes", "file_size_bytes",
           "checksum_md5"]

TARGETS = [("sub-100S0296", "ses-01", "run-01"),
           ("sub-007S2394", "ses-01", "run-01")]


def md5sum(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rows = []
    for sub, ses, run in TARGETS:
        base = f"{sub}_{ses}_task-rest_{run}_bold"
        nii = os.path.join(BIDS, sub, ses, "func", base + ".nii.gz")
        js = os.path.join(BIDS, sub, ses, "func", base + ".json")
        if not os.path.isfile(nii):
            raise SystemExit(f"missing BIDS input: {nii}")
        if not os.path.isfile(js):
            raise SystemExit(f"missing sidecar: {js}")

        img = nib.load(nii)
        zooms = img.header.get_zooms()
        with open(js, encoding="utf-8") as fh:
            meta = json.load(fh)
        tr = meta.get("RepetitionTime")
        if tr is None:
            raise SystemExit(f"RepetitionTime absent in {js} -- not inventing one")

        row = {
            "subject_id": sub,
            "session_id": ses,
            "run_id": run,
            "input_path": f"{WSL_BIDS}/{sub}/{ses}/func/{base}.nii.gz",
            "input_json": f"{WSL_BIDS}/{sub}/{ses}/func/{base}.json",
            "dimensions": str(tuple(int(x) for x in img.shape)),
            "voxel_size": str(tuple(round(float(z), 4) for z in zooms[:3])),
            "TR": tr,
            "volumes": int(img.shape[3]),
            "file_size_bytes": os.path.getsize(nii),
            "checksum_md5": md5sum(nii),
        }
        rows.append(row)
        nyq = 1.0 / (2.0 * float(tr))
        print(f"{sub} {ses} {run}")
        print(f"   dims {row['dimensions']}  voxel {row['voxel_size']}")
        print(f"   TR {tr}s  Nyquist {nyq:.4f} Hz  "
              f"band-pass 0.01-0.10 viable: {nyq > 0.10}")
        print(f"   volumes {row['volumes']}  after discarding 5: "
              f"{row['volumes'] - 5}")
        print(f"   md5 {row['checksum_md5']}")

    with open(OUT, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS)
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)} with {len(rows)} rows")
    print("Existing group manifests were not modified.")


if __name__ == "__main__":
    main()
