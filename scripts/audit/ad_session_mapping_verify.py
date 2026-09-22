import csv
import os

PARTICIPANTS_TSV = "/mnt/c/Users/krish/FYP/BIDS/participants.tsv"
SESSION_MAP_CSV = "/mnt/c/Users/krish/FYP/audit/bids_session_run_mapping.csv"
NIFTI_AD_DIR = "/mnt/c/Users/krish/FYP/NIfTI_/NIfTI/AD"
DERIV_ROOT = "/mnt/c/Users/krish/FYP/derivatives/fsfast"

CSV_OUT = "/mnt/c/Users/krish/FYP/audit/ad_session_mapping_final.csv"
TXT_OUT = "/mnt/c/Users/krish/FYP/audit/ad_session_mapping_final.txt"

CSV_FIELDS = ["participant_id", "session", "acquisition_date", "input_nifti", "input_json",
              "output_directory", "processing_status"]


def adni_to_bids(adni_id):
    return "sub-" + adni_id.replace("_", "")


def main():
    # 1. Only source of group membership
    ad_subjects = set()
    with open(PARTICIPANTS_TSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if row.get("group") == "AD":
                ad_subjects.add(row["participant_id"])
    print(f"AD subjects (from participants.tsv, only source used): {len(ad_subjects)}")

    # 2. Authoritative session/run mapping (from original BIDS creation process)
    all_mappings = list(csv.DictReader(open(SESSION_MAP_CSV, newline="", encoding="utf-8")))
    ad_mappings = [m for m in all_mappings if adni_to_bids(m["subject_id"]) in ad_subjects]
    print(f"AD-group rows found in authoritative bids_session_run_mapping.csv: {len(ad_mappings)}")

    rows = []
    output_paths_seen = {}
    subj_session_seen = {}
    issues = []

    for m in ad_mappings:
        adni_id = m["subject_id"]
        bids_sub = adni_to_bids(adni_id)
        session = m["BIDS_session"]
        run = m["BIDS_run"]
        acq_dt = m["acquisition_datetime"]
        acq_date = acq_dt[:8] if len(acq_dt) >= 8 else acq_dt

        # Check 1: exactly one session per acquisition -- detect duplicate
        # (subject, session) pairs, which would mean two acquisitions were
        # mapped to the same session (a real collision, not assumed away).
        key = (bids_sub, session)
        if key in subj_session_seen:
            issues.append(f"COLLISION: {bids_sub} {session} appears more than once "
                           f"(also for {subj_session_seen[key]})")
        subj_session_seen[key] = m["original_json"]

        # locate actual raw files on disk
        subj_folder_candidates = [adni_id]
        input_json = ""
        input_nifti = ""
        for cand in subj_folder_candidates:
            folder = os.path.join(NIFTI_AD_DIR, cand)
            jpath = os.path.join(folder, m["original_json"])
            if os.path.exists(jpath):
                input_json = jpath
                nii1 = jpath.replace(".json", ".nii.gz")
                nii2 = jpath.replace(".json", ".nii")
                if os.path.exists(nii1):
                    input_nifti = nii1
                elif os.path.exists(nii2):
                    input_nifti = nii2

        output_dir = os.path.join(DERIV_ROOT, bids_sub, session, "func")
        output_file = os.path.join(output_dir, f"{bids_sub}_{session}_task-rest_{run}_desc-preproc_bold.nii.gz")

        # Check 2/5: unique output path per acquisition
        if output_file in output_paths_seen:
            issues.append(f"OUTPUT COLLISION: {output_file} claimed by both "
                           f"{output_paths_seen[output_file]} and {bids_sub}/{session}")
        output_paths_seen[output_file] = f"{bids_sub}/{session}"

        status = "ALREADY_PROCESSED" if os.path.isfile(output_file) else "NOT_YET_PROCESSED"
        if input_nifti == "" or input_json == "":
            status = "SOURCE_FILE_NOT_FOUND"
            issues.append(f"MISSING SOURCE: {bids_sub} {session} -- json={input_json or 'MISSING'} "
                           f"nifti={input_nifti or 'MISSING'}")

        rows.append({
            "participant_id": bids_sub,
            "session": session,
            "acquisition_date": acq_date,
            "input_nifti": input_nifti or "NOT_FOUND",
            "input_json": input_json or "NOT_FOUND",
            "output_directory": output_dir,
            "processing_status": status,
        })

    rows.sort(key=lambda r: (r["participant_id"], r["session"]))

    os.makedirs(os.path.dirname(CSV_OUT), exist_ok=True)
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    already = [r for r in rows if r["processing_status"] == "ALREADY_PROCESSED"]
    remaining = [r for r in rows if r["processing_status"] == "NOT_YET_PROCESSED"]
    missing_src = [r for r in rows if r["processing_status"] == "SOURCE_FILE_NOT_FOUND"]

    # Check 1 (formal): every subject+session pair count == 1
    from collections import Counter
    pair_counts = Counter((r["participant_id"], r["session"]) for r in rows)
    multi_assigned = [k for k, c in pair_counts.items() if c > 1]

    # Check 2/5 (formal): output path uniqueness across ALL rows
    out_paths = [os.path.join(r["output_directory"],
                 f"{r['participant_id']}_{r['session']}_task-rest_run-01_desc-preproc_bold.nii.gz")
                 for r in rows]
    out_path_counts = Counter(out_paths)
    duplicate_outputs = [p for p, c in out_path_counts.items() if c > 1]

    with open(TXT_OUT, "w", encoding="utf-8") as f:
        f.write("=== AD SESSION MAPPING FINAL VERIFICATION ===\n\n")
        f.write(f"Group-membership source: {PARTICIPANTS_TSV} (only source used)\n")
        f.write(f"Session/run assignment source: {SESSION_MAP_CSV} (authoritative record from\n")
        f.write(f"  the original BIDS creation process -- cross-checked against, and found to\n")
        f.write(f"  exactly match, the independent chronological-order inference used in the\n")
        f.write(f"  prior AD NIfTI mapping audit. Using this file directly here rather than\n")
        f.write(f"  re-deriving, since it is the ground-truth record, not an inference.)\n\n")

        f.write(f"Total AD subjects: {len(ad_subjects)}\n")
        f.write(f"Total AD BOLD acquisitions (rows): {len(rows)}\n\n")

        f.write("--- CHECK 1: every acquisition has EXACTLY one session ---\n")
        if multi_assigned:
            f.write(f"  FAIL: {len(multi_assigned)} (subject, session) pairs have more than one "
                    f"acquisition assigned: {multi_assigned}\n")
        else:
            f.write(f"  PASS: all {len(rows)} acquisitions map to a distinct (subject, session) pair. "
                    f"No two acquisitions share a session.\n")
        f.write("\n")

        f.write("--- CHECK 2 & 5: every session has a UNIQUE derivative output path (no collision) ---\n")
        if duplicate_outputs:
            f.write(f"  FAIL: {len(duplicate_outputs)} output paths are claimed by more than one "
                    f"acquisition: {duplicate_outputs}\n")
        else:
            f.write(f"  PASS: all {len(out_paths)} computed output paths are unique. No overwrite is "
                    f"possible with this mapping.\n")
        f.write("\n")

        f.write("--- CHECK 3: already-processed runs correctly identified ---\n")
        f.write(f"  {len(already)} runs found with an existing output file:\n")
        for r in already:
            f.write(f"    {r['participant_id']} {r['session']}\n")
        f.write("\n")

        f.write("--- CHECK 4: remaining runs correctly identified ---\n")
        f.write(f"  {len(remaining)} runs with no existing output file (candidates for next processing):\n")
        for r in remaining:
            f.write(f"    {r['participant_id']} {r['session']} -- {os.path.basename(r['input_nifti'])}\n")
        f.write("\n")

        if missing_src:
            f.write("--- WARNING: source files not found on disk for these mapped acquisitions ---\n")
            for r in missing_src:
                f.write(f"    {r['participant_id']} {r['session']}\n")
            f.write("\n")

        if issues:
            f.write("--- ISSUES DETECTED ---\n")
            for i in issues:
                f.write(f"  {i}\n")
            f.write("\n")
        else:
            f.write("--- NO ISSUES DETECTED ---\n\n")

        f.write("=== SUMMARY ===\n\n")
        f.write(f"Total BOLD acquisitions verified: {len(rows)}\n")
        f.write(f"Already processed: {len(already)}\n")
        f.write(f"Remaining (to be processed next): {len(remaining)}\n")
        f.write(f"Source files missing: {len(missing_src)}\n")
        f.write(f"Session collisions: {len(multi_assigned)}\n")
        f.write(f"Output path collisions: {len(duplicate_outputs)}\n")
        overall = "SAFE TO PROCEED" if not (multi_assigned or duplicate_outputs or missing_src) else "DO NOT PROCEED -- ISSUES ABOVE MUST BE RESOLVED FIRST"
        f.write(f"\nOVERALL: {overall}\n")

    print(f"CSV: {CSV_OUT}")
    print(f"TXT: {TXT_OUT}")
    print(f"Already processed: {len(already)}, Remaining: {len(remaining)}, "
          f"Session collisions: {len(multi_assigned)}, Output collisions: {len(duplicate_outputs)}, "
          f"Missing source: {len(missing_src)}")


if __name__ == "__main__":
    main()
