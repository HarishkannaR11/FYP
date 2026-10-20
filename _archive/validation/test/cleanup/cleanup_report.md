# AD Preprocessing/Biomarker Cleanup Report (Phase 2 — Final)

**Deletion timestamps:** Phase 1: 2026-09-18 13:26:40 UTC (Windows-side derivatives/test). Phase 2: 2026-09-18 (Nipype WSL-native caches + pilot/pilot2w + reusable script recovery).
**Purpose:** Full reset ahead of restarting AD preprocessing from the original BOLD NIfTI files.

---

## IMPORTANT NOTE ON SEQUENCING

The user's script-preservation instruction ("before deleting test/biomarkers/, copy reusable
.py files to scripts/biomarkers/") arrived **after** `test/` had already been deleted in
Phase 1 (per the user's own prior confirmation to delete "all of test/ (test data)"). This
was flagged immediately and verified before any further action. The 11 reusable scripts
were **recreated verbatim from this session's own history** (not fabricated — exact content
this assistant authored earlier in the same conversation) rather than copied from disk,
since the source files no longer existed. One script (`run_qc_audit.py`) was recreated in
condensed form (core tSNR/FD/DVARS/GCOR logic) rather than its full original scope
(SNR/CNR/entropy sections omitted) — flagged explicitly, not silently.

---

## Deleted paths

### Phase 1 (Windows-mounted project paths, ~1.08 GB)
- `derivatives/biomarkers/AD/`
- `derivatives/fsfast/AD/`
- `test/` (entire directory: biomarker pilots, registration pilots, GLM test outputs, and their scripts)

### Phase 2 (WSL-native Nipype caches, ~19.3 GB)
- `/home/harish/fyp_work/fsfast_ad/` (old unmasked-GLM AD rerun cache)
- `/home/harish/fyp_work/fsfast_ad_masked/` (masked-GLM production cache)
- `/home/harish/fyp_work/ad5/`, `/home/harish/fyp_work/ad20/` (earliest AD pilot caches)
- `/home/harish/fyp_work/pilot/`, `/home/harish/fyp_work/pilot2w/` (confirmed non-AD, subject sub-002S0413)
- 8 loose `.log` files tied to the above runs

**`/home/harish/fyp_work/` is now completely empty** (verified: 4.0K, directory itself only).

## Preserved reusable scripts

Recreated in `scripts/biomarkers/` (12 files total):
- `compute_biomarkers_single_subject.py` (never deleted — was already outside `test/`)
- `verify_brainnetome_atlas.py`
- `reslice_atlas_to_bold_grid.py`
- `task1_geometry_verification.py`
- `task2_roi_coverage.py`
- `task3_transform_and_extract_roi.py`
- `task4_biomarkers.py`
- `task5_qc.py`
- `run_pilot_steps1to4.py` (superseded by run_registration_mni152.py, preserved for provenance)
- `run_registration_mni152.py`
- `run_qc_audit.py` (condensed — see note above)
- `synthetic_ground_truth.py`

## Original AD NIfTI count

**Before cleanup: 25. After cleanup: 25.** Unchanged. All 25 independently verified readable via NiBabel post-deletion.

## Preserved (verified untouched)

- `NIfTI_/NIfTI/AD/` — 25/25 files, all readable
- `BIDS/` — 175 task-rest bold files (unchanged)
- `raw_data/` — 6 group zip archives present
- `audit/` — 95 items present, not touched
- `participants.tsv`, `dataset_description.json` — not modified
- `atlases/Brainnetome246/BN_Atlas_246_2mm.nii.gz` — MD5 checksum identical (`9ddef1f57609f4f45a109757bec05ed3`)
- `atlases/Brainnetome246/BN_Atlas_246_LUT.txt` — 247 lines, unchanged
- `scripts/`, `pipeline/`, `tools/` — not touched

## Post-deletion verification results (all 12 checks)

| # | Check | Result |
|---|---|---|
| 1 | Original AD NIfTI count = 25 | PASS |
| 2 | All 25 original AD NIfTIs readable | PASS |
| 3 | BIDS untouched (175 files) | PASS |
| 4 | raw_data untouched (6 zips) | PASS |
| 5 | audit/ untouched (95 items) | PASS |
| 6 | Brainnetome atlas exists and readable | PASS |
| 7 | Atlas checksum/LUT line count unchanged | PASS |
| 8 | Reusable scripts exist under scripts/biomarkers/ | PASS (12 files) |
| 9 | Old AD preprocessing derivatives gone | PASS |
| 10 | Old AD biomarker outputs gone | PASS |
| 11 | Old pilot/test GLM outputs gone | PASS |
| 12 | Old Nipype caches gone | PASS |

**Disk usage under /home/harish/fyp_work/: 4.0K (empty)**

## Final cleanup status: **SUCCESS**

Complete reset achieved. Original source data, BIDS, raw_data, audit trail, the official
Brainnetome atlas, and all pipeline/scripts source code verified untouched. All old
generated preprocessing, biomarker, registration, and pilot outputs — both on the
Windows-mounted project and the WSL-native Nipype work directory — removed. Reusable
biomarker/QC/registration scripts preserved in `scripts/biomarkers/`.

**Not started:** the new preprocessing pipeline, as instructed. Ready for you to begin
the AD production restart from `NIfTI_/NIfTI/AD/` whenever you give the go-ahead.
