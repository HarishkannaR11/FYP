import csv
import os
import re
import json
import nibabel as nib

PARTICIPANTS_TSV = "/mnt/c/Users/krish/FYP/BIDS/participants.tsv"
NIFTI_AD_DIR = "/mnt/c/Users/krish/FYP/NIfTI_/NIfTI/AD"
DERIV_ROOT = "/mnt/c/Users/krish/FYP/derivatives/fsfast"

CSV_OUT = "/mnt/c/Users/krish/FYP/audit/ad_nifti_mapping.csv"
TXT_OUT = "/mnt/c/Users/krish/FYP/audit/ad_nifti_mapping_summary.txt"

CSV_FIELDS = [
    "participant_id", "nifti_filename", "json_filename", "dimensions", "volumes",
    "TR", "voxel_size", "series_description", "protocol_name", "manufacturer",
    "classification", "reason", "inferred_session", "already_in_derivatives", "derivatives_path",
]


def extract_datetime(fname):
    m = re.search(r"(\d{14})", fname)
    return m.group(1) if m else None


def load_ad_subjects():
    subs = []
    with open(PARTICIPANTS_TSV, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if row.get("group") == "AD":
                subs.append(row["participant_id"])
    return subs


def find_subject_folder(sub):
    # sub-002S5018 -> try common variants actually present on disk
    candidates = [
        sub,  # sub-002S5018
        sub.replace("sub-", ""),  # 002S5018
    ]
    # ADNI-style with underscores: 002S5018 -> 002_S_5018
    bare = sub.replace("sub-", "")
    if len(bare) >= 4:
        # pattern: SSS S NNNN  (site3 + 'S' + subjid)
        import re
        m = re.match(r"^(\d{3})S(\d+)$", bare)
        if m:
            candidates.append(f"{m.group(1)}_S_{m.group(2)}")
    for c in candidates:
        p = os.path.join(NIFTI_AD_DIR, c)
        if os.path.isdir(p):
            return p, c
    return None, None


def classify(nii_path, json_path, meta, shape):
    """Classify image type from filename + metadata, no assumptions from count alone."""
    fname = os.path.basename(nii_path).lower()
    series_desc = (meta.get("SeriesDescription") or "").lower() if meta else ""
    protocol = (meta.get("ProtocolName") or "").lower() if meta else ""
    modality = (meta.get("Modality") or "").upper() if meta else ""

    is_4d = len(shape) == 4
    text = fname + " " + series_desc + " " + protocol

    if "t1" in text or "mprage" in text or "spgr" in text:
        return "ANATOMICAL_T1W", "Filename/metadata indicates T1-weighted structural scan"
    if "fieldmap" in text or "b0" in text or "fmap" in text:
        return "FIELDMAP", "Filename/metadata indicates a fieldmap acquisition"
    if "dti" in text or "diffusion" in text:
        return "DIFFUSION", "Filename/metadata indicates a DTI/diffusion acquisition"
    if ("resting" in text or "rest" in text or "bold" in text) and is_4d:
        return "BOLD_RESTING_STATE", "Filename/metadata indicates resting-state BOLD and image is 4D"
    if is_4d and modality == "MR":
        return "BOLD_LIKELY", "4D MR image without explicit non-BOLD keyword in filename/metadata -- treated as BOLD candidate"
    if not is_4d:
        return "NON_BOLD_3D", "Image is 3D, not a functional time series"
    return "UNKNOWN", "Could not classify from available filename/metadata"


def main():
    ad_subjects = load_ad_subjects()
    print(f"AD subjects from participants.tsv: {len(ad_subjects)}")

    rows = []
    subjects_with_folder = 0
    subjects_missing_folder = []
    total_nifti_files = 0
    subjects_bold_count = {}

    for sub in ad_subjects:
        folder, matched_name = find_subject_folder(sub)
        if folder is None:
            subjects_missing_folder.append(sub)
            continue
        subjects_with_folder += 1

        nii_files = []
        for root, _, files in os.walk(folder):
            for f in files:
                if f.endswith(".nii") or f.endswith(".nii.gz"):
                    nii_files.append(os.path.join(root, f))

        # Determine session order chronologically from each file's embedded
        # acquisition datetime (YYYYMMDDHHMMSS in filename) -- same convention
        # already established in this project (earliest = ses-01, next = ses-02,
        # ...). This is read directly from real filename metadata, not fabricated.
        dated = [(extract_datetime(os.path.basename(p)), p) for p in nii_files]
        dated_with_dt = sorted([d for d in dated if d[0] is not None], key=lambda x: x[0])
        dated_without_dt = [d for d in dated if d[0] is None]
        session_map = {}
        for i, (dt, p) in enumerate(dated_with_dt, start=1):
            session_map[p] = f"ses-{i:02d}"
        for dt, p in dated_without_dt:
            session_map[p] = "ses-UNKNOWN"

        nii_files = [p for _, p in dated_with_dt] + [p for _, p in dated_without_dt]
        total_nifti_files += len(nii_files)

        bold_count = 0
        for nii_path in nii_files:
            stem = nii_path.replace(".nii.gz", "").replace(".nii", "")
            json_path = stem + ".json"
            meta = {}
            if os.path.exists(json_path):
                try:
                    with open(json_path) as jf:
                        meta = json.load(jf)
                except Exception:
                    meta = {}

            try:
                img = nib.load(nii_path)
                shape = img.shape
                zooms = img.header.get_zooms()
                vol = shape[3] if len(shape) > 3 else 1
            except Exception as e:
                shape = ("READ_ERROR",)
                zooms = ()
                vol = "N/A"

            cls, reason = classify(nii_path, json_path, meta, shape)
            if cls in ("BOLD_RESTING_STATE", "BOLD_LIKELY"):
                bold_count += 1

            tr = meta.get("RepetitionTime", "")
            voxel = f"({zooms[0]:.2f},{zooms[1]:.2f},{zooms[2]:.2f})" if len(zooms) >= 3 else ""

            # check derivatives -- PER SESSION, not just per subject, so a
            # subject with one processed session doesn't wrongly mark their
            # OTHER, unprocessed session as already done.
            inferred_ses = session_map.get(nii_path, "ses-UNKNOWN")
            already = "NO"
            deriv_path = ""
            expected_deriv = os.path.join(
                DERIV_ROOT, sub, inferred_ses, "func",
                f"{sub}_{inferred_ses}_task-rest_run-01_desc-preproc_bold.nii.gz"
            )
            if os.path.isfile(expected_deriv):
                already = "YES"
                deriv_path = expected_deriv

            rows.append({
                "participant_id": sub,
                "nifti_filename": os.path.basename(nii_path),
                "json_filename": os.path.basename(json_path) if os.path.exists(json_path) else "MISSING",
                "dimensions": str(shape),
                "volumes": vol,
                "TR": tr,
                "voxel_size": voxel,
                "series_description": meta.get("SeriesDescription", ""),
                "protocol_name": meta.get("ProtocolName", ""),
                "manufacturer": meta.get("Manufacturer", ""),
                "classification": cls,
                "reason": reason,
                "inferred_session": inferred_ses,
                "already_in_derivatives": already,
                "derivatives_path": deriv_path,
            })
        subjects_bold_count[sub] = bold_count

    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    bold_rows = [r for r in rows if r["classification"] in ("BOLD_RESTING_STATE", "BOLD_LIKELY")]
    subjects_2plus_bold = [s for s, c in subjects_bold_count.items() if c >= 2]
    subjects_missing_bold = [s for s, c in subjects_bold_count.items() if c == 0]
    subjects_processed = sorted(set(r["participant_id"] for r in rows if r["already_in_derivatives"] == "YES"))

    with open(TXT_OUT, "w", encoding="utf-8") as f:
        f.write("=== AD SUBJECT / NIfTI MAPPING AUDIT ===\n\n")
        f.write(f"Group-membership source (ONLY source used): {PARTICIPANTS_TSV}\n")
        f.write(f"NIfTI source directory: {NIFTI_AD_DIR}\n\n")

        f.write("NOTE ON SESSION INFERENCE: raw NIfTI filenames encode a real acquisition\n")
        f.write("datetime (YYYYMMDDHHMMSS) but not an explicit ses-XX label. Where a subject\n")
        f.write("has multiple dated files, session number was assigned by chronological order\n")
        f.write("(earliest datetime = ses-01, next = ses-02, ...) -- the same convention already\n")
        f.write("established and cross-checked elsewhere in this project (DICOM audit). This is\n")
        f.write("read from real embedded metadata, not fabricated. The already_in_derivatives\n")
        f.write("check is done PER INFERRED SESSION, not per subject -- an earlier draft of this\n")
        f.write("script incorrectly checked at the subject level only, which wrongly marked an\n")
        f.write("unprocessed second session as already done for a multi-session subject; this was\n")
        f.write("caught and fixed before this report was finalized.\n\n")

        f.write(f"Total AD subjects listed in participants.tsv: {len(ad_subjects)}\n")
        f.write(f"AD subjects with a matching folder under NIfTI_/NIfTI/AD: {subjects_with_folder}\n")
        f.write(f"AD subjects with NO matching folder found: {len(subjects_missing_folder)}\n")
        if subjects_missing_folder:
            f.write(f"  Missing: {subjects_missing_folder}\n")
        f.write(f"\nTotal NIfTI files found across all AD subject folders: {total_nifti_files}\n")
        f.write(f"Total classified as BOLD resting-state candidates: {len(bold_rows)}\n\n")

        f.write("--- Classification breakdown (all NIfTI files) ---\n")
        from collections import Counter
        cls_counts = Counter(r["classification"] for r in rows)
        for cls, cnt in cls_counts.items():
            f.write(f"  {cls}: {cnt}\n")
        f.write("\n")

        f.write(f"--- Subjects with 2+ BOLD files ---\n")
        for s in subjects_2plus_bold:
            f.write(f"  {s}: {subjects_bold_count[s]} BOLD files\n")
        if not subjects_2plus_bold:
            f.write("  None\n")
        f.write("\n")

        f.write(f"--- Subjects with missing/zero BOLD data ---\n")
        for s in subjects_missing_bold:
            f.write(f"  {s}\n")
        if not subjects_missing_bold:
            f.write("  None\n")
        f.write("\n")

        f.write(f"--- Subjects already processed in derivatives/fsfast ---\n")
        for s in subjects_processed:
            f.write(f"  {s}\n")
        if not subjects_processed:
            f.write("  None\n")
        f.write("\n")

        unprocessed_bold = [r for r in bold_rows if r["already_in_derivatives"] == "NO"]
        f.write(f"--- BOLD runs NOT yet in derivatives (candidates for next processing) ---\n")
        for r in unprocessed_bold:
            f.write(f"  {r['participant_id']}: {r['nifti_filename']} (TR={r['TR']}, dims={r['dimensions']})\n")
        if not unprocessed_bold:
            f.write("  None\n")
        f.write("\n")

        f.write("=== FINAL SUMMARY ===\n\n")
        f.write(f"Total AD subjects from CSV: {len(ad_subjects)}\n")
        f.write(f"Total NIfTI files: {total_nifti_files}\n")
        f.write(f"Total eligible resting-state BOLD runs: {len(bold_rows)}\n")
        f.write(f"Subjects with 2+ BOLD files: {len(subjects_2plus_bold)} {subjects_2plus_bold}\n")
        f.write(f"Subjects with missing BOLD data: {len(subjects_missing_bold)} {subjects_missing_bold}\n")
        f.write(f"Subjects already processed (in derivatives/fsfast): {len(subjects_processed)} {subjects_processed}\n")
        f.write(f"Runs that should be processed next: {len(unprocessed_bold)}\n")
        for r in unprocessed_bold:
            f.write(f"  {r['participant_id']} -- {r['nifti_filename']}\n")

    print(f"CSV: {CSV_OUT}")
    print(f"TXT: {TXT_OUT}")
    print(f"Total NIfTI: {total_nifti_files}, BOLD candidates: {len(bold_rows)}, "
          f"already processed subjects: {len(subjects_processed)}, "
          f"unprocessed BOLD runs: {len(unprocessed_bold)}")


if __name__ == "__main__":
    main()
