import os
import re
import csv
import json
import hashlib
import nibabel as nib

NIFTI_ROOT = "/mnt/c/Users/krish/FYP/NIfTI_/NIfTI"
PARTICIPANTS_TSV = "/mnt/c/Users/krish/FYP/BIDS/participants.tsv"
OUT_DIR = "/mnt/c/Users/krish/FYP/audit/bids_conversion"
INVENTORY_CSV = os.path.join(OUT_DIR, "complete_nifti_inventory.csv")

GROUPS = ["AD", "CN_Final", "EMCI", "LMCI", "MCI", "SMC_Final"]

FIELDS = [
    "group", "adni_id", "participant_id", "source_file", "source_json",
    "acquisition_datetime", "series_number", "shape", "volumes",
    "voxel_size", "TR_nifti", "TR_json", "orientation", "modality_guess",
    "series_description", "protocol_name", "manufacturer", "md5",
    "readable", "notes",
]


def adni_to_bids(adni_id):
    return "sub-" + adni_id.replace("_", "")


def md5sum(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def extract_datetime(fname):
    m = re.search(r"(\d{14})", fname)
    return m.group(1) if m else None


def classify_modality(fname, meta):
    fl = fname.lower()
    sd = (meta.get("SeriesDescription") or "").lower()
    pn = (meta.get("ProtocolName") or "").lower()
    text = fl + " " + sd + " " + pn
    if "t1" in text or "mprage" in text or "spgr" in text:
        return "T1w"
    if "fieldmap" in text or "fmap" in text:
        return "fieldmap"
    if "dti" in text or "diffusion" in text:
        return "dwi"
    if "resting" in text or "rest" in text or "bold" in text:
        return "BOLD_resting_state"
    return "UNKNOWN"


def main():
    rows = []
    participant_group = {}
    if os.path.exists(PARTICIPANTS_TSV):
        with open(PARTICIPANTS_TSV, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f, delimiter="\t"):
                participant_group[r["participant_id"]] = r.get("group", "")

    for group in GROUPS:
        group_dir = os.path.join(NIFTI_ROOT, group)
        if not os.path.isdir(group_dir):
            print(f"WARNING: group dir not found: {group_dir}")
            continue
        subj_dirs = sorted([d for d in os.listdir(group_dir) if os.path.isdir(os.path.join(group_dir, d))])
        print(f"{group}: {len(subj_dirs)} subject folders")

        for adni_id in subj_dirs:
            sdir = os.path.join(group_dir, adni_id)
            bids_sub = adni_to_bids(adni_id)
            nii_files = sorted([f for f in os.listdir(sdir) if f.endswith(".nii") or f.endswith(".nii.gz")])

            for fname in nii_files:
                nii_path = os.path.join(sdir, fname)
                stem = fname.replace(".nii.gz", "").replace(".nii", "")
                json_path = os.path.join(sdir, stem + ".json")
                notes = []

                meta = {}
                if os.path.exists(json_path):
                    try:
                        with open(json_path) as jf:
                            meta = json.load(jf)
                    except Exception as e:
                        notes.append(f"JSON parse error: {e}")
                else:
                    notes.append("JSON sidecar missing")

                shape = tr_nifti = voxel = orient = "N/A"
                volumes = "N/A"
                readable = "NO"
                try:
                    img = nib.load(nii_path)
                    shape = str(img.shape)
                    zooms = img.header.get_zooms()
                    volumes = img.shape[3] if len(img.shape) > 3 else 1
                    voxel = f"({zooms[0]:.2f},{zooms[1]:.2f},{zooms[2]:.2f})" if len(zooms) >= 3 else "N/A"
                    tr_nifti = round(float(zooms[3]), 4) if len(zooms) > 3 else "N/A"
                    orient = "".join(nib.aff2axcodes(img.affine))
                    readable = "YES"
                except Exception as e:
                    notes.append(f"NIfTI read error: {e}")

                try:
                    checksum = md5sum(nii_path)
                except Exception as e:
                    checksum = "ERROR"
                    notes.append(f"checksum error: {e}")

                modality = classify_modality(fname, meta)
                acq_dt = extract_datetime(fname)

                rows.append({
                    "group": group,
                    "adni_id": adni_id,
                    "participant_id": bids_sub,
                    "source_file": nii_path,
                    "source_json": json_path if os.path.exists(json_path) else "MISSING",
                    "acquisition_datetime": acq_dt or "UNKNOWN",
                    "series_number": meta.get("SeriesNumber", ""),
                    "shape": shape,
                    "volumes": volumes,
                    "voxel_size": voxel,
                    "TR_nifti": tr_nifti,
                    "TR_json": meta.get("RepetitionTime", ""),
                    "orientation": orient,
                    "modality_guess": modality,
                    "series_description": meta.get("SeriesDescription", ""),
                    "protocol_name": meta.get("ProtocolName", ""),
                    "manufacturer": meta.get("Manufacturer", ""),
                    "md5": checksum,
                    "readable": readable,
                    "notes": "; ".join(notes),
                })

    os.makedirs(OUT_DIR, exist_ok=True)
    with open(INVENTORY_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nTotal NIfTI files inventoried: {len(rows)}")
    print(f"Written: {INVENTORY_CSV}")


if __name__ == "__main__":
    main()
