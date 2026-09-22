import os
import csv
import json
from collections import defaultdict, Counter

OUT_DIR = "/mnt/c/Users/krish/FYP/audit/bids_conversion"
INVENTORY_CSV = os.path.join(OUT_DIR, "complete_nifti_inventory.csv")
MAPPING_CSV = os.path.join(OUT_DIR, "bids_mapping.csv")
SESSION_RUN_CSV = os.path.join(OUT_DIR, "session_run_mapping.csv")
SUMMARY_TXT = os.path.join(OUT_DIR, "bids_conversion_summary.txt")

EXISTING_BIDS = "/mnt/c/Users/krish/FYP/BIDS"

MAP_FIELDS = [
    "group", "participant_id", "source_file", "source_json", "acquisition_datetime",
    "session", "run", "shape", "volumes", "voxel_size", "TR", "modality",
    "duplicate_status", "mapping_status", "target_bids_nifti", "target_bids_json",
    "matches_existing_bids",
]


def main():
    rows = list(csv.DictReader(open(INVENTORY_CSV, newline="", encoding="utf-8")))
    print(f"Loaded {len(rows)} inventory rows")

    # ---- duplicate detection by md5 ----
    md5_groups = defaultdict(list)
    for r in rows:
        md5_groups[r["md5"]].append(r)
    dup_md5 = {m: rs for m, rs in md5_groups.items() if len(rs) > 1 and m != "ERROR"}

    # ---- group files by participant ----
    by_subject = defaultdict(list)
    for r in rows:
        by_subject[r["participant_id"]].append(r)

    mapping_rows = []
    ambiguous = []

    for sub, items in sorted(by_subject.items()):
        # Only BOLD resting-state files get BIDS func/ placement
        bold_items = [i for i in items if i["modality_guess"] == "BOLD_resting_state"]
        non_bold = [i for i in items if i["modality_guess"] != "BOLD_resting_state"]

        # Group by acquisition DATE (first 8 chars of YYYYMMDDHHMMSS) -> session
        dated = []
        undated = []
        for i in bold_items:
            dt = i["acquisition_datetime"]
            if dt and dt != "UNKNOWN" and len(dt) >= 8:
                dated.append((dt, i))
            else:
                undated.append(i)

        # session index by unique DATE, chronological
        unique_dates = sorted({dt[:8] for dt, _ in dated})
        date_to_session = {d: f"ses-{n:02d}" for n, d in enumerate(unique_dates, start=1)}

        # within a session (same date), multiple acquisitions -> run-01, run-02...
        session_counter = defaultdict(int)
        for dt, item in sorted(dated, key=lambda x: x[0]):
            ses = date_to_session[dt[:8]]
            session_counter[ses] += 1
            run = f"run-{session_counter[ses]:02d}"

            dup_status = "UNIQUE"
            if item["md5"] in dup_md5:
                others = [o["source_file"] for o in dup_md5[item["md5"]]
                          if o["source_file"] != item["source_file"]]
                dup_status = f"DUPLICATE_MD5 (same content as: {'; '.join(os.path.basename(o) for o in others)})"

            target_nii = f"{EXISTING_BIDS}/{sub}/{ses}/func/{sub}_{ses}_task-rest_{run}_bold.nii.gz"
            target_json = target_nii.replace(".nii.gz", ".json")

            # does an equivalent file already exist in the current BIDS dataset?
            existing_nii_gz = target_nii
            existing_nii = target_nii.replace(".nii.gz", ".nii")
            if os.path.exists(existing_nii_gz):
                matches_existing = "EXISTS(.nii.gz)"
            elif os.path.exists(existing_nii):
                matches_existing = "EXISTS(.nii)"
            else:
                matches_existing = "NOT_PRESENT"

            mapping_rows.append({
                "group": item["group"], "participant_id": sub,
                "source_file": item["source_file"], "source_json": item["source_json"],
                "acquisition_datetime": dt, "session": ses, "run": run,
                "shape": item["shape"], "volumes": item["volumes"],
                "voxel_size": item["voxel_size"],
                "TR": item["TR_json"] or item["TR_nifti"],
                "modality": item["modality_guess"],
                "duplicate_status": dup_status,
                "mapping_status": "MAPPED",
                "target_bids_nifti": target_nii, "target_bids_json": target_json,
                "matches_existing_bids": matches_existing,
            })

        # undated BOLD -> cannot assign session from metadata -> flag, do NOT invent
        for item in undated:
            ambiguous.append((sub, item["source_file"], "no acquisition datetime in filename"))
            mapping_rows.append({
                "group": item["group"], "participant_id": sub,
                "source_file": item["source_file"], "source_json": item["source_json"],
                "acquisition_datetime": "UNKNOWN", "session": "AMBIGUOUS", "run": "AMBIGUOUS",
                "shape": item["shape"], "volumes": item["volumes"],
                "voxel_size": item["voxel_size"],
                "TR": item["TR_json"] or item["TR_nifti"],
                "modality": item["modality_guess"],
                "duplicate_status": "UNIQUE" if item["md5"] not in dup_md5 else "DUPLICATE_MD5",
                "mapping_status": "AMBIGUOUS_NO_DATETIME_FLAGGED_FOR_REVIEW",
                "target_bids_nifti": "", "target_bids_json": "",
                "matches_existing_bids": "",
            })

        # non-BOLD files -> not placed in func/
        for item in non_bold:
            mapping_rows.append({
                "group": item["group"], "participant_id": sub,
                "source_file": item["source_file"], "source_json": item["source_json"],
                "acquisition_datetime": item["acquisition_datetime"],
                "session": "", "run": "",
                "shape": item["shape"], "volumes": item["volumes"],
                "voxel_size": item["voxel_size"],
                "TR": item["TR_json"] or item["TR_nifti"],
                "modality": item["modality_guess"],
                "duplicate_status": "UNIQUE" if item["md5"] not in dup_md5 else "DUPLICATE_MD5",
                "mapping_status": "NOT_BOLD_NOT_MAPPED_TO_FUNC",
                "target_bids_nifti": "", "target_bids_json": "",
                "matches_existing_bids": "",
            })

    with open(MAPPING_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=MAP_FIELDS)
        w.writeheader()
        w.writerows(mapping_rows)

    # ---- session/run mapping table ----
    with open(SESSION_RUN_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["group", "participant_id", "session", "run", "acquisition_date",
                    "source_filename", "target_bids_filename"])
        for r in mapping_rows:
            if r["mapping_status"] == "MAPPED":
                w.writerow([r["group"], r["participant_id"], r["session"], r["run"],
                            r["acquisition_datetime"][:8], os.path.basename(r["source_file"]),
                            os.path.basename(r["target_bids_nifti"])])

    # ---- summary ----
    mapped = [r for r in mapping_rows if r["mapping_status"] == "MAPPED"]
    amb = [r for r in mapping_rows if r["mapping_status"].startswith("AMBIGUOUS")]
    nonbold = [r for r in mapping_rows if r["mapping_status"] == "NOT_BOLD_NOT_MAPPED_TO_FUNC"]
    missing_json = [r for r in mapping_rows if r["source_json"] == "MISSING"]
    dups = [r for r in mapping_rows if r["duplicate_status"].startswith("DUPLICATE")]

    multi_acq = {}
    for r in mapped:
        multi_acq.setdefault(r["participant_id"], []).append(r)
    multi_acq = {k: v for k, v in multi_acq.items() if len(v) > 1}

    subj_per_group = defaultdict(set)
    for r in mapped:
        subj_per_group[r["group"]].add(r["participant_id"])

    sess_per_subj = defaultdict(set)
    runs_per_session = defaultdict(int)
    for r in mapped:
        sess_per_subj[r["participant_id"]].add(r["session"])
        runs_per_session[(r["participant_id"], r["session"])] += 1

    not_present = [r for r in mapped if r["matches_existing_bids"] == "NOT_PRESENT"]

    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write("=== BIDS CONVERSION INVENTORY & MAPPING SUMMARY ===\n\n")
        f.write("READ-ONLY audit. No source NIfTI/JSON file was read-modified, renamed, or deleted.\n")
        f.write("No BIDS dataset has been written by this script -- this is the mapping PLAN only.\n\n")

        f.write(f"Source root: /mnt/c/Users/krish/FYP/NIfTI_/NIfTI\n")
        f.write(f"Total source NIfTI files found: {len(rows)}\n")
        f.write(f"Total subjects (folders): {len(by_subject)}\n\n")

        f.write(f"Classified as BOLD resting-state (mappable): {len(mapped)}\n")
        f.write(f"Ambiguous (no acquisition datetime -- FLAGGED, not invented): {len(amb)}\n")
        f.write(f"Non-BOLD (not mapped to func/): {len(nonbold)}\n")
        f.write(f"Missing JSON sidecar: {len(missing_json)}\n")
        f.write(f"Duplicate content (identical MD5): {len(dups)}\n\n")

        f.write("--- Subjects per group (BOLD-mapped) ---\n")
        for g in sorted(subj_per_group):
            f.write(f"  {g}: {len(subj_per_group[g])} subjects\n")
        f.write("\n")

        f.write("--- Sessions per subject distribution ---\n")
        dist = Counter(len(v) for v in sess_per_subj.values())
        for n_ses in sorted(dist):
            f.write(f"  {dist[n_ses]} subjects have {n_ses} session(s)\n")
        f.write("\n")

        f.write("--- Runs per session distribution ---\n")
        rdist = Counter(runs_per_session.values())
        for n_run in sorted(rdist):
            f.write(f"  {rdist[n_run]} sessions have {n_run} run(s)\n")
        f.write("\n")

        f.write(f"--- Subjects with MULTIPLE acquisitions ({len(multi_acq)}) ---\n")
        for sub in sorted(multi_acq):
            items = sorted(multi_acq[sub], key=lambda r: (r["session"], r["run"]))
            f.write(f"  {sub} ({items[0]['group']}): {len(items)} acquisitions\n")
            for it in items:
                f.write(f"      {it['session']} {it['run']}  date={it['acquisition_datetime'][:8]}  "
                        f"{os.path.basename(it['source_file'])}\n")
        f.write("\n")

        f.write("--- Duplicate-content files (identical MD5) ---\n")
        if dups:
            for r in dups:
                f.write(f"  {r['participant_id']} {r['session']} {r['run']}: {r['duplicate_status']}\n")
        else:
            f.write("  None -- every source NIfTI has unique content.\n")
        f.write("\n")

        f.write("--- Ambiguous session assignment (flagged for review, NOT invented) ---\n")
        if amb:
            for r in amb:
                f.write(f"  {r['participant_id']}: {os.path.basename(r['source_file'])}\n")
        else:
            f.write("  None -- every BOLD file carried an acquisition datetime in its filename.\n")
        f.write("\n")

        f.write("--- Comparison against the EXISTING BIDS dataset ---\n")
        f.write(f"  Planned BOLD targets already present in existing BIDS/: "
                f"{len(mapped) - len(not_present)} of {len(mapped)}\n")
        f.write(f"  Planned BOLD targets NOT present in existing BIDS/: {len(not_present)}\n")
        if not_present:
            for r in not_present[:30]:
                f.write(f"      {r['participant_id']} {r['session']} {r['run']} ({r['group']})\n")
        f.write("\n")

    print(f"Mapped (BOLD): {len(mapped)}")
    print(f"Ambiguous: {len(amb)}  Non-BOLD: {len(nonbold)}  Missing JSON: {len(missing_json)}  Duplicates: {len(dups)}")
    print(f"Subjects with multiple acquisitions: {len(multi_acq)}")
    print(f"Planned targets NOT already in existing BIDS: {len(not_present)}")
    print(f"\nWritten:\n  {MAPPING_CSV}\n  {SESSION_RUN_CSV}\n  {SUMMARY_TXT}")


if __name__ == "__main__":
    main()
