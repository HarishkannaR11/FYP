# Phase-1 Cohort — 116 Subjects

The analysis cohort. One resting-state fMRI acquisition per subject, with
preprocessing, biomarkers and clinical data already extracted. Every feature
matrix, statistic and model in this project is built from here.

Verified against the files in this directory on 2026-10-20. Numbers that could
not be verified are marked, not estimated.

```
CN → SMC → EMCI → MCI → LMCI → AD
 0     1      2     3      4     5
```

This order is encoded in `derivatives/model_ready/y.npy`, not just documented.
Do not reorder it.

---

## 1. Data selection

```
                    ADNI rs-fMRI download
                   (no T1w scan for any subject)
                              │
                              ▼
                  ┌───────────────────────┐
                  │  167 acquisitions     │
                  │  129 distinct subjects│
                  └───────────┬───────────┘
                              │   preprocessing + biomarker extraction
                              │   0 invalid, 0 missing, 0 non-finite,
                              │   0 duplicate acquisition ids
                              ▼
                  ┌───────────────────────┐
                  │  167 VALID_UNIQUE     │
                  └───────────┬───────────┘
                              │
                              │   SELECTION RULE — deterministic, label-blind
                              │   per stage:
                              │     1. keep VALID_UNIQUE only
                              │     2. sort by (subject, session, run)
                              │        ascending — lexicographic, stable,
                              │        independent of any biomarker value
                              │     3. take the first acquisition of each
                              │        distinct subject until 20 are held
                              ▼
     ┌────────────────────────────────────────────────────────┐
     │  CN 20   SMC 20   EMCI 20   MCI 16   LMCI 20   AD 20    │
     │                      = 116 subjects                     │
     │         1 acquisition each, 0 repeat acquisitions       │
     └────────────────────────────┬───────────────────────────┘
                                  │
         ┌────────────────────────┼────────────────────────┐
         ▼                        ▼                        ▼
   ┌───────────┐          ┌──────────────┐        ┌──────────────┐
   │  imaging  │          │   clinical   │        │ comorbidity  │
   │  116/116  │          │  MMSE 116    │        │  19 flags    │
   │           │          │  CDR  116    │        │   114/116    │
   │ 4 biomark.│          │  GDS  115    │        │              │
   │ 246 ROIs  │          │  APOE 115    │        │  scan-matched│
   └─────┬─────┘          └──────┬───────┘        └──────┬───────┘
         └────────────────────────┼───────────────────────┘
                                  ▼
                   derivatives/model_ready/*.npy
                     116 rows, one per subject
                        ↓            ↓
                  train_pool 81   test 35
                  (5 CV folds)    (locked)
```

**Why MCI is 16, not 20.** Only 16 distinct MCI subjects had valid
preprocessed acquisitions. The stage had 34 acquisitions but they repeat
across subjects, and the rule allows one acquisition per subject. The target
of 20 per stage was not met and was **not** topped up with repeat scans.

**Why the rule is label-blind.** Sorting is lexicographic on identifiers only.
No biomarker value, quality metric or outcome influences which acquisition is
kept, so selection cannot bias any group difference reported later.

Source: `reports/dataset_selection_report.md`, built by
`scripts/data_prep/build_phase1_dataset.py`. Source trees were opened
read-only; nothing was converted, preprocessed or re-extracted.

---

## 2. Layout

```
phase1_selected_20/                                   1.9 GB
├── CN/ SMC/ EMCI/ MCI/ LMCI/ AD/     the 116 subjects
│   └── sub-<id>/ses-01/run-01/
├── comorbidity/                      clinical + comorbidity tables
├── manifests/                        what was selected, and from where
├── reports/                          dataset_selection_report.md
└── _audit/
```

### Per subject — `sub-<id>/ses-01/run-01/`

| path | contents |
|---|---|
| `*_desc-preproc_bold.nii.gz` | preprocessed 4-D BOLD |
| `*_motion_parameters.mcdat`, `design_matrix.txt`, `brain_mask.nii.gz` | preprocessing products |
| `biomarkers/` | per-ROI ALFF, ReHo, degree centrality, FC matrix, ROI timeseries, Brainnetome-246 parcellation + QC |
| `normalized/` | **the model inputs** — mALFF, mReHo, DC_z, Fisher-z FC |
| `qc/` | FD and DVARS traces, tSNR map, registration check, `qc_metrics.csv` |

`normalized/` is what feeds the feature matrices. `biomarkers/` holds the
unnormalised originals and is **read-only** — do not modify anything under it.

### `manifests/`

`selected_acquisitions.csv` (what was chosen), `acquisition_inventory.csv`
(everything considered), `subject_acquisition_mapping.csv`, `duplicate_audit.csv`,
`copy_mapping.csv`, `selection_summary.json`.

### `comorbidity/`

See `comorbidity/SOURCE.md`. The table to use is
**`model_clinical_116.csv`** — 116 rows, in `subjects.npy` order, so row *i*
is row *i* of `X_regional.npy` / `X_fc.npy` / `y.npy` with no join.

---

## 3. Features built from this cohort

`derivatives/model_ready/`, all in the same row order:

| file | shape | contents |
|---|---|---|
| `X_regional.npy` | (116, 246, 3) | mALFF, mReHo, DC_z |
| `X_fc.npy` | (116, 246, 246) | Fisher-z FC |
| `X_fc_roi_unsigned.npy` | (116, 246) | FC strength per ROI |
| `X_fc_roi_dispersion.npy` | (116, 246) | FC dispersion — **never evaluated** |
| `y.npy`, `stages.npy`, `subjects.npy`, `acq_ids.npy` | (116,) | labels and ids |
| `index.csv` | 116 rows | row ↔ acquisition ↔ `source_dir` here |
| `splits/split_assignment.csv` | 116 rows | partition + CV fold |

**Split:** 81 train pool (5 stratified folds of 16–17), 35 locked test.
Fingerprint `6f478fbcf56364e2`.

---

## 4. Clinical and comorbidity coverage

| field | available | missing |
|---|---|---|
| imaging, MRI date | **116/116** | — |
| MMSE, CDR global | **116/116** | — |
| GDS | 115/116 | `130_S_4990` |
| APOE genotype | 115/116 | `019_S_5180` |
| CDR sum-of-boxes | 114/116 | `007_S_2394`, `100_S_0296` |
| 19 comorbidity flags | 114/116 | `007_S_2394`, `100_S_0296` |
| everything | **113/116** | |

Usable *n* depends on which fields you take: MMSE + CDR gives 116, adding GDS
115, adding APOE 114, adding comorbidity flags 113, adding CDR-SB 112.

`007_S_2394` and `100_S_0296` entered through a separate download
(`NIfTI_extra`, image ids `I680900` and `I422066`) and appear in no ADNI
clinical table in the project.

---

## 5. Four things to know before using this

**MMSE and CDR partly define the label.** ADNI's diagnostic criteria use their
cut-offs to assign the stage being predicted. Global CDR alone reaches 0.386
balanced accuracy against the rs-fMRI model's 0.307 — that measures the
enrolment rule, not the biology. Use them as a baseline, not as imaging-model
features. `GDS_total`, the `MH_*` flags and `APOE4_count` are not part of the
criteria and are the honest covariates.

**No ROI × stage test survives FDR correction.** Across all four biomarkers
the strongest effect reaches q = 0.118; q = 0.05 needs a P-score of 1.301 and
the largest value anywhere is 0.928. The cohort is underpowered for
1,476-test correction.

**The test set has been evaluated more than once.** `final_heldout_results.json`
records one evaluation, and a later feature-selection study ran three more
configurations on the same 35 subjects. Report all four and keep the
pre-declared primary as the headline.

**Missing is never zero.** `-4` is ADNI's missing code and is stored blank
throughout. Nothing in the clinical tables is imputed.

---

## 6. Related documents

| | |
|---|---|
| `comorbidity/SOURCE.md` | clinical and comorbidity tables in detail |
| `reports/dataset_selection_report.md` | the selection, in full |
| `docs/DATASET_AND_COHORT.md` | cohort provenance and acquisition inventory |
| `docs/PREPROCESSING_PIPELINE.md` | how the preprocessed BOLD was produced |
| `docs/FEATURES_AND_STATISTICAL_ANALYSIS.md` | biomarker definitions and statistics |
| `docs/MODEL_ARCHITECTURE.md` | models trained on these features |
| `docs/LIMITATIONS_AND_FUTURE_WORK.md` | what this cohort cannot support |

**ADNI Data Use Agreement:** the imaging and clinical data here are not
redistributable. `*.nii` / `*.nii.gz` are gitignored; the derived clinical
tables are deliberately kept out of version control and regenerate from
`scripts/comorbidity/build_model_clinical.py`.
