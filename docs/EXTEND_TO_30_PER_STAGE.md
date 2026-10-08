# Extending the cohort to 30 processed scans per stage

Why: the Review II slides and report say "about 30 scans per stage" because that is
what the pipeline has processed. They can say "30 per stage" only after the scans
below have gone through the same frozen pipeline and audit.

This runbook is assembled from the scripts' headers, constants and wrappers. It has
**not been executed**: BIDS, the DICOM archives and the WSL toolchain are not in the
repository, so run it on the project machine and check each script's output.

## 1. The gap today (computed from the audit files)

`python3 scripts/audit/stage_target_check.py 30`

| Stage | Raw in BIDS | Processed | Subjects | Need more processed | Above 30 |
|---|---:|---:|---:|---:|---:|
| CN   | 30 | 27 | 26 | **3** | 0 |
| SMC  | 32 | 31 | 21 | 0 | 1 |
| EMCI | 30 | 29 | 26 | **1** | 0 |
| MCI  | 35 | 32 | 14 | 0 | 2 |
| LMCI | 23 | 21 | 20 | **9** | 0 |
| AD   | 25 | 25 | 20 | **5** | 0 |

18 more processed scans are needed. A few raw scans will fail on the way (duplicate,
truncated, or TR 6.02 s), so bring in a small margin.

## 2. What to add

- **AD (+5):** five AD subjects are in your ADNI export (`Comorbidity/AD_9_01_2026.csv`)
  but not in BIDS: 006S4546 (I336551), 006S4867 (I362021), 053S5070 (I389296),
  130S4589 (I347092), 130S5006 (I398911). Download and convert these.
- **LMCI (+9 processed, so about +10 raw):** no LMCI export is in the repo. Re-export the
  LMCI image collection from ADNI, original resting-state fMRI, same search settings as
  the other groups.
- **CN (+3) and EMCI (+1):** their exports list as many scans as BIDS holds (30 each), so the
  new ones must come from a fresh ADNI search with the same settings.

## 3. Steps (project machine, WSL)

1. DICOM to NIfTI (dcm2niix, as at Review I) into `NIfTI/NIfTI/<GROUP>/`, then add the new
   scans to BIDS the way the originals were added (`scripts/data_prep/create_bids.py` built
   BIDS and the mapping files in `audit/`; BIDS is meant to stay read-only, so check what it
   overwrites before re-running it).
2. Group manifests: edit the expected counts, then run `scripts/audit/build_<group>_manifest.py`
   for each changed group. AD is different: its builder reads `audit/ad_session_mapping_final.csv`
   (see `scripts/audit/ad_session_mapping_verify.py`), so extend that mapping with the five new
   subjects before building the AD manifest.
3. Update the hard-coded expectations (the scripts assert on them and will stop otherwise):

   | File | Constant |
   |---|---|
   | `pipeline/fsfast_production_4mm6mm.py` | `GROUP_MANIFEST` (second value per group) |
   | `scripts/audit/build_<group>_manifest.py` | `EXPECTED_SUBJECTS*`, `EXPECTED_ELIGIBLE_ACQUISITIONS` / `EXPECTED_ACQUISITIONS` |
   | `scripts/audit/final_6group_audit.py` | `EXPECTED`, `KNOWN_EXCLUSIONS` |
   | `scripts/biomarkers/normalize_within_subject.py` | `EXPECTED_PER_GROUP`, `EXPECTED_TOTAL` |
   | `scripts/clinical/clean_clinical_data.py` | `EXPECTED_SUBJECTS` (and the "127" / "165" in its report text) |
   | `audit/excluded_runs.csv` | add any new duplicate/truncated scans |

4. Keep every processing parameter frozen. Run, in order (all resume and skip finished scans):
   ```
   bash scripts/setup/retry_all_groups_4mm6mm.sh
   bash scripts/setup/retry_final_audit.sh
   bash scripts/setup/retry_biomarker_dataset.sh discover DISCOVERY_DONE
   bash scripts/setup/retry_biomarker_dataset.sh run RUN_DONE
   bash scripts/setup/retry_biomarker_dataset.sh audit AUDIT_DONE
   python3 scripts/biomarkers/normalize_within_subject.py run
   python3 scripts/biomarkers/normalize_within_subject.py audit
   python3 scripts/biomarkers/stage_heatmaps.py
   python3 scripts/clinical/clean_clinical_data.py
   ```
5. Check: `python3 scripts/audit/stage_target_check.py 30` should show 0 in "Need more".
6. Fix exactly 30 per stage with a rule that never looks at imaging results or labels:
   `python3 scripts/audit/select_balanced_cohort.py 30`
   (writes `derivatives/cohort/balanced_30_index.csv`; every subject's first scan comes
   before anyone's second; it refuses to write if any stage is below 30. Add
   `--exclude sub-018S4313/ses-01/run-01` to drop the artefact scan before selecting.)

## 4. After the run

Commit the audit and derivative outputs and tell me. I will then update, from the new
audit only: the stages slide, the dataset slide and table, the verification slide, the
speaker notes, and the report's cohort, QC, heatmap and abstract numbers.

Two things that do not change by adding imaging scans: the clinical covariates for the
LMCI group (their source tables are still missing), and the 127-subject statistics, which
will grow with the new subjects.
