# Dataset and Cohort

Every count in this document was verified against the files in this
repository on the date of writing. Where a number could not be verified it
is marked as such rather than estimated.

---

## 1. Source

**ADNI** (Alzheimer's Disease Neuroimaging Initiative), accessed through the
LONI Image & Data Archive. Resting-state fMRI only — **no T1-weighted
structural scan exists for any subject in this download**, which constrains
the preprocessing design (see [PREPROCESSING_PIPELINE.md](PREPROCESSING_PIPELINE.md)).

Six diagnostic groups are used. The directory names `CN_Final` and
`SMC_Final` are historical; they map to the canonical stages `CN` and `SMC`.

### Canonical stage order

```
CN → SMC → EMCI → MCI → LMCI → AD
 0     1      2     3      4     5
```

This ordering is **encoded in the data**, not merely in documentation:
`derivatives/model_ready/y.npy` uses MCI = 3 and LMCI = 4, and
`derivatives/model_ready/metadata.json` records `stage_to_index`. The
label-to-index agreement was verified by counting rows both by string label
and by integer index — all six agree exactly
([FINAL_STATISTICAL_AUDIT.md](../derivatives/INTRA_INTER_CLASS_HEATMAPS/v2_cohort116/FINAL_STATISTICAL_AUDIT.md), §1).

A request to place LMCI before MCI was received and **not applied**, because
it contradicts the encoding used by every downstream output.

---

## 2. Acquisition inventory

| Stage | Preprocessed acquisitions | Distinct subjects |
|---|---:|---:|
| CN | 27 | 26 |
| SMC | 31 | 21 |
| EMCI | 29 | 26 |
| MCI | 34 | 16 |
| LMCI | 21 | 20 |
| AD | 25 | 20 |
| **Total** | **167** | **129** |

Source: `derivatives/fsfast/FINAL_6GROUP_AUDIT/dataset_qc_summary.csv`.

Note how far acquisitions exceed subjects — ADNI is longitudinal, so many
subjects contribute several visits. **MCI is the extreme case: 34
acquisitions from only 16 people.** This is why the analysis cohort is
built per subject, not per acquisition.

### Attrition

| Step | Count |
|---|---:|
| Raw acquisitions audited | 175 |
| Pre-excluded | 2 |
| Eligible | 173 |
| Failed preprocessing | 8 |
| Successfully preprocessed (original run) | 165 |
| Added later (2 new MCI subjects) | 2 |
| **Total preprocessed** | **167** |

**Pre-exclusions** (`dataset_exclusions.csv`):

- `CN_Final/sub-012S4026/ses-01/run-01` — 7-volume truncated acquisition
- `LMCI/sub-006S4363/ses-01/run-02` — byte-identical duplicate of run-01

**Preprocessing failures** (`dataset_failures.csv`) — all 8 share one cause:

| Group | Acquisition | TR | Nyquist |
|---|---|---:|---:|
| CN_Final | sub-002S0413/ses-01/run-01 | 6.0216 s | 0.0830 Hz |
| CN_Final | sub-002S0685/ses-01/run-01 | 6.0216 s | 0.0830 Hz |
| EMCI | sub-002S2043/ses-01/run-01 | 6.0216 s | 0.0830 Hz |
| LMCI | sub-002S4171/ses-01/run-01 | 6.0216 s | 0.0830 Hz |
| MCI | sub-002S0729/ses-01/run-01 | 6.0216 s | 0.0830 Hz |
| MCI | sub-002S1155/ses-02/run-01 | 6.0216 s | 0.0830 Hz |
| MCI | sub-002S4799/ses-01/run-01 | 6.0216 s | 0.0830 Hz |
| SMC_Final | sub-002S5256/ses-02/run-01 | 6.0235 s | 0.0830 Hz |

The Nyquist frequency `1/(2·TR)` = 0.083 Hz falls **below** the 0.10 Hz
upper band-pass cutoff, so the configured filter is not defined for these
acquisitions. The pipeline refused them rather than silently narrowing the
band. All 8 come from site 002.

---

## 3. Analysis cohort (116 subjects)

`derivatives/phase1_selected_20/` and `derivatives/model_ready/`

| Stage | Acquisitions | Subjects |
|---|---:|---:|
| CN | 20 | 20 |
| SMC | 20 | 20 |
| EMCI | 20 | 20 |
| **MCI** | **16** | **16** |
| LMCI | 20 | 20 |
| AD | 20 | 20 |
| **Total** | **116** | **116** |

**Verified**: 116 unique subject IDs, 116 acquisitions, zero subjects
contributing more than one acquisition.

Feature array shapes (`derivatives/model_ready/`):

```
X_regional.npy   (116, 246, 3)    float32   channels = mALFF, mReHo, DC_z
X_fc.npy         (116, 246, 246)  float32   Fisher-z, signed, diagonal 0
y.npy            (116,)           int64     CN=0 … AD=5
subjects.npy, stages.npy, acq_ids.npy, index.csv, metadata.json
```

Cohort fingerprint `6ab68ae6e80f61c2` (SHA-256 of sorted subject IDs plus
stage labels), used by the statistical checkpointing to reject results
computed on any other cohort.

### Inclusion rules

An acquisition is eligible only if **all** hold:

1. Preprocessing completed and recorded in `PROCESSING_CHECKPOINT.json`.
2. Not in `dataset_exclusions.csv` and not in `dataset_failures.csv`.
3. A preprocessed BOLD image exists on disk.
4. All four normalised features exist: `malff.npy`, `mreho.npy`,
   `dc_z.npy`, `fc_fisher_z.npy`.
5. All raw biomarker arrays exist.
6. 246 ROI values per regional feature; FC is exactly 246 × 246.
7. No NaN or infinite values.

All 167 acquisitions passed these checks (`VALID_UNIQUE`); none was excluded
at this stage.

### Selection rule — one acquisition per subject

Applied per stage, deterministic and label-blind:

1. Consider only `VALID_UNIQUE` acquisitions.
2. Sort by `(subject, session, run)` ascending, stable sort.
3. Take the **first acquisition of each distinct subject** in that order,
   until 20 are held.
4. If fewer than 20 distinct subjects exist, **stop and report the
   shortfall**. A second acquisition from an already-selected subject is
   taken only if explicitly enabled (it is not, by default).

No QC metric, biomarker value or stage label enters the ordering, so no
target information leaks into later evaluation.

**Why 116 and not 120**: MCI has only 16 subjects with valid data. The
other five stages reach the cap of 20. 116 is the arithmetic consequence of
the rule, not a target that was imposed.

### Consequence for validation

Because every subject appears exactly once, observations are independent and
**no repeated-measures model is required**. Subject-grouped cross-validation
is trivially satisfied: no subject can straddle a fold.

---

## 4. Clinical and comorbidity data

Sourced separately from the imaging data. The ADNI **image** search exports
(`Comorbidity/*_9_01_2026.csv`) contain only 12 columns — Image Data ID,
Subject, Group, Sex, Age, Visit, Modality, Description, Type, Acq Date,
Format, Downloaded — and **no clinical scores at all**. Scores come from the
assessment tables in `Comorbidity/Tables_01Sep2026*.zip`:

```
Resting_State_fMRI_<GROUP>_Cohort_CDR_01Sep2026.csv
Resting_State_fMRI_<GROUP>_Cohort_MMSE_01Sep2026.csv
Resting_State_fMRI_<GROUP>_Cohort_GDSCALE_01Sep2026.csv
Resting_State_fMRI_<GROUP>_Cohort_MEDHIST_01Sep2026.csv
Resting_State_fMRI_<GROUP>_Cohort_APOERES_01Sep2026.csv
Resting_State_fMRI_<GROUP>_Cohort_Functional_MRI_Images_01Sep2026.csv
```

plus `Comorbidity/APOERES_01Sep2026.csv`, the full ADNI genetics table
(3,216 subjects).

### How clinical data is joined

`scripts/clinical/clean_clinical_data.py`:

1. Reads loose CSVs and CSVs inside ZIP archives from `Comorbidity/`.
2. Maps ADNI `PTID` (`NNN_S_NNNN`) to the project's `sub-NNNSNNNN`.
3. Establishes each subject's MRI date from
   `Functional_MRI_Images`, then the image-collection exports, then
   provenance timestamps — in that order.
4. For each assessment, selects the visit **nearest in time to the MRI**,
   recording the chosen visit, the day offset, and the match method in
   dedicated columns.
5. Decodes medical-history flags: 1 → present, 0 → absent, sentinel → NaN.
6. Counts APOE ε4 alleles from the genotype string.

**No imputation is performed.** Missing values remain missing. Conflicting
duplicate records are flagged for review rather than silently resolved
(`duplicate_conflict_report.csv`).

### Coverage — verified within the 116-subject cohort

**Do not assume every subject has complete comorbidity data. They do not.**

| Variable | CN | SMC | EMCI | MCI | LMCI | AD | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| MMSE | 20/20 | 20/20 | 20/20 | 14/16 | **0/20** | 20/20 | **94/116** |
| CDR global | 20/20 | 20/20 | 20/20 | 14/16 | **0/20** | 20/20 | **94/116** |
| GDS | 20/20 | 20/20 | 20/20 | 14/16 | **0/20** | 20/20 | **94/116** |
| Medical history | 20/20 | 20/20 | 20/20 | 14/16 | **0/20** | 20/20 | **94/116** |
| APOE ε4 count | 20/20 | 19/20 | 20/20 | 14/16 | 20/20 | 20/20 | **113/116** |

Two findings matter:

1. **LMCI has no MMSE, CDR, GDS or medical-history data at all.** The LMCI
   assessment tables were never downloaded from ADNI; only APOE (which comes
   from the cohort-wide table) is available. Any comorbidity-aware model
   would currently exclude LMCI entirely or require that download.
2. **Two cohort subjects have no clinical row of any kind**:
   `sub-100S0296` and `sub-007S2394` — the two MCI subjects added most
   recently. Their imaging is complete; their assessment tables have not
   been downloaded.

Outputs: `derivatives/clinical_cleaned/` — `master_subject_clinical.csv`
(127 subjects × 90 columns), per-stage tables in `stages/`,
`clinical_data_dictionary.md`, `clinical_audit_report.md`,
`duplicate_conflict_report.csv`, `visit_audit.csv`.

---

## 5. Limitations

- **MCI is under-represented**: 16 subjects against 20 elsewhere. The
  intended 20-per-stage target is not met and was not forced.
- **Small absolute size**: 116 subjects for a six-class ordinal problem is
  roughly 19 per class.
- **Site concentration**: 14 scanning sites, unevenly distributed. One site
  contributes 40% of AD acquisitions and two contribute 68%, while no single
  site exceeds 19% of CN. Any CN–AD contrast is partly a site contrast. No
  harmonisation (e.g. ComBat) has been applied.
- **Protocol heterogeneity**: the original cohort is `Resting State fMRI`
  (140 volumes); the two newest MCI subjects are `Extended Resting State
  fMRI` and `Axial rsfMRI (Eyes Open)` (200 volumes). All are TR ≈ 3 s.
  `sub-007S2394` is the only **GE** scanner and the only acquisition whose
  sidecar contains real `SliceTiming`.
- **No structural scan**, so no subject-specific tissue segmentation; the
  CSF regressor failed its reliability gate in every acquisition.
- **Longitudinal data discarded by design**: repeat visits exist but are not
  used, since the analysis is cross-sectional and one-per-subject.

---

## 6. Data access and ethics

ADNI data is **not redistributable**. This repository contains derived
features and statistics; raw imaging must be obtained directly from
<https://adni.loni.usc.edu> under ADNI's Data Use Agreement, which requires
registration and approval.

ADNI participants gave written informed consent, and the study was approved
by the institutional review boards of the participating sites. Users of ADNI
data are required to follow the ADNI Data Sharing and Publication Policy,
including its acknowledgement requirements.

No attempt is made in this project to re-identify participants. Subject
identifiers are ADNI's own pseudonymous codes.

---

## 7. Provenance and reproducibility

Each preprocessed acquisition carries a JSON provenance record with the
source path, MD5 checksum before and after processing, every parameter,
software versions, and a UTC timestamp. Checksum equality
(`checksum_match: true`) is recorded for all acquisitions, demonstrating the
source files were never modified.

Audit tables in `derivatives/fsfast/FINAL_6GROUP_AUDIT/` are regenerated by
`scripts/audit/final_6group_audit.py`, which recomputes every check from the
files on disk rather than copying from processing logs.
