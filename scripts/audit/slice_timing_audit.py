import os
import json
import csv
import nibabel as nib

BIDS_DIR = "/mnt/c/Users/krish/FYP/BIDS"
AUDIT_DIR = "/mnt/c/Users/krish/FYP/audit"
CSV_OUT = os.path.join(AUDIT_DIR, "slice_timing_inventory.csv")
TXT_OUT = os.path.join(AUDIT_DIR, "slice_timing_summary.txt")

FIELDNAMES = [
    "participant_id", "session", "run", "task",
    "bold_file", "json_file",
    "z_dimension", "volumes",
    "TR",
    "has_slice_timing", "slice_timing_count", "slice_timing_values",
    "slice_timing_zdim_match",
    "number_of_slices_field", "acquisition_time_field",
    "slice_encoding_direction", "metadata_status",
]


def parse_entities(stem):
    parts = stem.split("_")
    sub = next((p for p in parts if p.startswith("sub-")), "UNKNOWN")
    ses = next((p for p in parts if p.startswith("ses-")), "UNKNOWN")
    run = next((p for p in parts if p.startswith("run-")), "UNKNOWN")
    task = next((p for p in parts if p.startswith("task-")), "UNKNOWN")
    return sub, ses, run, task


def main():
    print("Starting slice-timing metadata audit (read-only)...")
    rows = []

    n_total = 0
    n_with_st = 0
    n_without_st = 0
    n_mismatch = 0
    n_json_missing = 0
    n_read_error = 0
    examples_missing = []
    examples_mismatch = []
    examples_json_missing = []

    for root, _, files in os.walk(BIDS_DIR):
        for f in files:
            if not (f.endswith("_bold.nii") or f.endswith("_bold.nii.gz")):
                continue

            n_total += 1
            bold_path = os.path.join(root, f)
            stem = f.replace(".nii.gz", "").replace(".nii", "")
            sub, ses, run, task = parse_entities(stem)
            json_path = os.path.join(root, stem + ".json")

            z_dim = "ERROR"
            volumes = "ERROR"
            tr_val = "UNKNOWN"
            has_st = "NO"
            st_count = 0
            st_values = ""
            st_match = "N/A"
            nslices_field = "NOT_PRESENT"
            acqtime_field = "NOT_PRESENT"
            slice_enc_dir = "NOT_PRESENT"
            status_flags = []

            # Read NIfTI header (shape only, no data load beyond header/shape)
            try:
                img = nib.load(bold_path)
                shape = img.shape
                if len(shape) >= 3:
                    z_dim = shape[2]
                if len(shape) >= 4:
                    volumes = shape[3]
                else:
                    volumes = 1
            except Exception as e:
                z_dim = "READ_ERROR"
                volumes = "READ_ERROR"
                status_flags.append("NIFTI_READ_ERROR")
                n_read_error += 1

            # Read JSON sidecar
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as jf:
                        meta = json.load(jf)

                    tr_val = meta.get("RepetitionTime", "UNKNOWN")

                    if "SliceTiming" in meta and isinstance(meta["SliceTiming"], list):
                        st_list = meta["SliceTiming"]
                        has_st = "YES"
                        st_count = len(st_list)
                        st_values = ";".join(str(v) for v in st_list)
                        n_with_st += 1

                        if isinstance(z_dim, int):
                            if st_count == z_dim:
                                st_match = "MATCH"
                            else:
                                st_match = "MISMATCH"
                                status_flags.append("SLICETIMING_COUNT_MISMATCH")
                                n_mismatch += 1
                                if len(examples_mismatch) < 10:
                                    examples_mismatch.append(
                                        f"{sub} {ses} {run}: SliceTiming count={st_count} vs Z dimension={z_dim} ({f})"
                                    )
                        else:
                            st_match = "UNKNOWN_Z"
                    else:
                        has_st = "NO"
                        n_without_st += 1
                        status_flags.append("MISSING_SLICETIMING")
                        if len(examples_missing) < 10:
                            examples_missing.append(f"{sub} {ses} {run}: no 'SliceTiming' key in {stem}.json")

                    if "NumberOfSlices" in meta:
                        nslices_field = meta.get("NumberOfSlices")
                    if "AcquisitionTime" in meta:
                        acqtime_field = meta.get("AcquisitionTime")
                    if "SliceEncodingDirection" in meta:
                        slice_enc_dir = meta.get("SliceEncodingDirection")

                except Exception as e:
                    status_flags.append(f"JSON_READ_ERROR:{e}")
                    n_read_error += 1
            else:
                status_flags.append("JSON_MISSING")
                n_json_missing += 1
                n_without_st += 1
                if len(examples_json_missing) < 10:
                    examples_json_missing.append(f"{sub} {ses} {run}: no JSON sidecar found for {f}")

            metadata_status = ";".join(status_flags) if status_flags else "OK"

            rows.append({
                "participant_id": sub,
                "session": ses,
                "run": run,
                "task": task,
                "bold_file": bold_path,
                "json_file": json_path if os.path.exists(json_path) else "MISSING",
                "z_dimension": z_dim,
                "volumes": volumes,
                "TR": tr_val,
                "has_slice_timing": has_st,
                "slice_timing_count": st_count,
                "slice_timing_values": st_values,
                "slice_timing_zdim_match": st_match,
                "number_of_slices_field": nslices_field,
                "acquisition_time_field": acqtime_field,
                "slice_encoding_direction": slice_enc_dir,
                "metadata_status": metadata_status,
            })

    os.makedirs(AUDIT_DIR, exist_ok=True)

    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    valid_st_count = sum(1 for r in rows if r["has_slice_timing"] == "YES" and r["slice_timing_zdim_match"] == "MATCH")

    with open(TXT_OUT, "w", encoding="utf-8") as f:
        f.write("=== SLICE-TIMING METADATA AUDIT SUMMARY ===\n\n")
        f.write(f"Total BOLD runs checked: {n_total}\n")
        f.write(f"Runs with 'SliceTiming' key present: {n_with_st}\n")
        f.write(f"Runs without 'SliceTiming' key: {n_without_st}\n")
        f.write(f"  - of which missing JSON sidecar entirely: {n_json_missing}\n")
        f.write(f"Runs with SliceTiming present but count/Z-dimension mismatch: {n_mismatch}\n")
        f.write(f"Runs with SliceTiming present and count == Z-dimension (valid): {valid_st_count}\n")
        f.write(f"Read errors (NIfTI or JSON): {n_read_error}\n\n")

        f.write("--- CONCLUSION ---\n")
        if n_with_st == 0:
            f.write(
                "No BOLD run in this dataset contains a 'SliceTiming' field in its JSON sidecar. "
                "Slice-timing correction is NOT technically supported by the available metadata for any run. "
                "No SliceTiming values were inferred or fabricated; this reflects the literal contents of the sidecars.\n"
            )
        elif n_with_st == n_total and n_mismatch == 0:
            f.write(
                "All BOLD runs contain a valid 'SliceTiming' field consistent with their Z dimension. "
                "Slice-timing correction is technically supported by the available metadata for all runs.\n"
            )
        else:
            f.write(
                f"SliceTiming metadata is PARTIAL/INCONSISTENT across the dataset: {n_with_st} of {n_total} runs "
                f"have a 'SliceTiming' field, and {n_mismatch} of those have a count that does not match the Z "
                "dimension. Slice-timing correction is only technically supportable, on a per-run basis, for runs "
                "flagged MATCH in slice_timing_inventory.csv. It cannot be applied uniformly across the dataset "
                "without either excluding the non-compliant runs or omitting slice-timing correction entirely -- "
                "this is a decision for the mentor/pipeline owner, not something this audit resolves.\n"
            )

        f.write("\n--- EXAMPLES: RUNS MISSING SliceTiming KEY (JSON present, key absent) ---\n")
        if examples_missing:
            for e in examples_missing:
                f.write(e + "\n")
        else:
            f.write("None\n")

        f.write("\n--- EXAMPLES: RUNS MISSING JSON SIDECAR ENTIRELY ---\n")
        if examples_json_missing:
            for e in examples_json_missing:
                f.write(e + "\n")
        else:
            f.write("None\n")

        f.write("\n--- EXAMPLES: SliceTiming COUNT MISMATCH vs Z DIMENSION ---\n")
        if examples_mismatch:
            for e in examples_mismatch:
                f.write(e + "\n")
        else:
            f.write("None\n")

    print(f"Done. {n_total} runs checked. SliceTiming present: {n_with_st}. Missing: {n_without_st}. Mismatch: {n_mismatch}.")
    print(f"CSV:  {CSV_OUT}")
    print(f"TXT:  {TXT_OUT}")


if __name__ == "__main__":
    main()
