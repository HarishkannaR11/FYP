# Assessment of Alzheimer's Disease Progression Using Comorbidity-Aware Meta-Bayesian Framework with Resting-State fMRI

Final Year Project. Predicting Alzheimer's disease progression
(CN → SMC → EMCI → LMCI → MCI → AD) from resting-state fMRI (rs-fMRI)
functional biomarkers, using a Brainnetome-246 parcellation, a multi-attention
transformer, meta-learning, comorbidity fusion, and a Bayesian ordinal
classifier for staging. Source dataset: ADNI, BOLD-only (no T1w, no
fieldmaps, no reliable SliceTiming metadata).

**Full methodology, verified line-by-line against the actual code/logs/outputs:**
[`docs/FYP_RSFMRI_COMPLETE_METHODOLOGY_AND_IMPLEMENTATION.md`](docs/FYP_RSFMRI_COMPLETE_METHODOLOGY_AND_IMPLEMENTATION.md)
(exhaustive, Parts A–Y) and
[`docs/FYP_RSFMRI_METHODOLOGY_SUMMARY.md`](docs/FYP_RSFMRI_METHODOLOGY_SUMMARY.md)
(condensed). This README is a pointer/orientation layer on top of those —
treat the docs as the source of truth for methodological detail.

## Project structure

```
FYP/
├── pipeline/             Production preprocessing pipelines
│   └── fsfast_production_4mm6mm.py   CURRENT pipeline — all 6 groups
│                                       (see "Current pipeline" below)
│   (older/, superseded, kept for provenance: fsfast_pipeline.py,
│    fsfast_ad_*.py, fsfast_<group>_production.py, paper_pipeline_pilot*.py)
├── scripts/
│   ├── audit/             Dataset/DICOM/preprocessing/QC audit scripts
│   ├── biomarkers/        Brainnetome parcellation + biomarker generation
│   │                       (biomarker_dataset_pipeline.py = dataset-scale)
│   ├── data_prep/         BIDS conversion and DICOM-recovery scripts
│   └── setup/             WSL/FreeSurfer setup + resilient retry wrapper
│                           scripts (*.sh) used to run every long job
├── docs/                  Methodology documentation + environment setup
├── atlases/
│   └── Brainnetome246/    BN_Atlas_246_2mm.nii.gz + ROI lookup table
├── BIDS/                  BIDS-formatted dataset (original, READ-ONLY, gitignored)
├── raw_data/               Original ADNI DICOM archives, grouped by diagnosis
├── NIfTI_/                 Raw converted NIfTI files (pre-BIDS), gitignored
├── derivatives/fsfast/     Pipeline outputs
│   ├── <GROUP>/sub-<ID>/ses-<N>/run-<N>/   preprocessed BOLD, motion file,
│   │                                        design matrix, qc/, and
│   │                                        biomarkers/ (all 165 acquisitions)
│   ├── PROCESSING_CHECKPOINT.json           per-group completed/failed/
│   │                                         remaining, updated per acquisition
│   ├── FINAL_6GROUP_AUDIT/                  preprocessing audit report + CSVs
│   └── BIOMARKER_DATASET_AUDIT/             biomarker manifest, errors,
│                                             per-acquisition QC + audit report
├── audit/                 Read-only audit reports (txt/csv/json)
└── test/                  Validation/cleanup scripts
```

Groups: `AD`, `CN_Final`, `EMCI`, `LMCI`, `MCI`, `SMC_Final`.
Progression ordering used throughout: `CN → SMC → EMCI → LMCI → MCI → AD`.

## Current pipeline

The frozen, current production pipeline is
[`pipeline/fsfast_production_4mm6mm.py`](pipeline/fsfast_production_4mm6mm.py).
Earlier native-space scripts in `pipeline/` (`fsfast_pipeline.py`,
`fsfast_ad_*.py`, `fsfast_<group>_production.py`) are superseded and kept only
for provenance — do not use them for new work.

```
Raw 4D BOLD (BIDS)
      ↓
Input validation (4D, finite, valid affine/TR, non-empty)
      ↓
Discard first 5 volumes
      ↓
[Slice-timing correction: NOT PERFORMED — exhaustively confirmed
 unrecoverable from BIDS JSON or raw DICOM for this entire dataset;
 see docs/..._IMPLEMENTATION.md Part C. Nothing fabricated.]
      ↓
Motion correction (FS-FAST mc-afni2 / AFNI 3dvolreg, rigid 6-DOF,
 reference = frame 0; native .mcdat preserved)
      ↓
Direct EPI → MNI152NLin6Asym registration (ANTsPy, SyN; no T1w exists
 for this dataset, so registration is BOLD-mean → template directly)
      ↓
4mm isotropic resample (continuous interpolation)
      ↓
6mm FWHM Gaussian smoothing
      ↓
Linear detrending (OLS, temporal mean restored)
      ↓
Nuisance regression — 26-28 regressors: Friston-24 (motion + derivatives
 + squares + squared-derivatives) + WM + CSF (MNI152NLin2009cAsym template
 priors, reliability-gated per acquisition) + Global Signal
      ↓
Band-pass filter, 0.01-0.10 Hz (Butterworth order-2 + filtfilt,
 Nyquist-validated per acquisition — never silently altered)
      ↓
Preprocessed BOLD + QC (tSNR, SNR, FD, DVARS, spatial/temporal entropy;
 CNR = NOT_RELIABLY_COMPUTABLE, no T1w)
```

**Known dataset constraints, by design, not oversight:** no T1w anatomical
images anywhere in this dataset (BOLD-only), no fieldmaps, no reliable
SliceTiming metadata (exhaustively audited, not merely assumed absent). No
anatomical/T1w-dependent processing (recon-all, subject-specific WM/CSF
segmentation) is used anywhere.

## Downstream: parcellation + biomarkers (complete, all 165 acquisitions)

Run across the entire preprocessed dataset by
[`scripts/biomarkers/biomarker_dataset_pipeline.py`](scripts/biomarkers/biomarker_dataset_pipeline.py)
(modes: `discover`, `verify-pilot`, `run`, `audit`; `MAX_WORKERS = 1`,
checkpoint/resume, one acquisition at a time).

```
Preprocessed BOLD
      ↓
Brainnetome-246 parcellation (per subject, nearest-neighbor only)
      ↓
ROI time series (T × 246)
      ↓
ALFF · ReHo · FC · DC · FC regional strength
      ↓
Regional biomarker matrix (246 × 4)
```

- **Brainnetome-246 parcellation** — the original 2mm atlas is reoriented
  LAS→RAS losslessly and nearest-neighbor-resampled onto each subject's own
  BOLD grid (never linear/cubic: labels are categorical). **246/246 ROI
  coverage in all 165 acquisitions**, zero zero-voxel ROIs dataset-wide.
- **ROI time series** — mean BOLD per ROI per timepoint by direct index
  masking on the already-aligned grid (no second resampling). Orientation is
  **time × ROI**, `(135, 246)` for every acquisition. 0 NaN, 0 Inf,
  0 zero-variance ROIs.
- **ALFF / ReHo** — computed voxel-wise, then averaged within each ROI. ALFF
  is the mean FFT amplitude in 0.01–0.10 Hz (Zang et al. 2007); ReHo is
  Kendall's W over a 27-voxel (3×3×3, centre included) neighbourhood, with K
  reduced at the mask boundary.
- **FC / DC / FC strength** — FC is the full signed, unthresholded 246×246
  Pearson matrix (diagonal preserved); DC is the weighted signed row-sum
  excluding self (`DC_i = Σ_{j≠i} FC[i,j]`); FC strength is the corresponding
  *mean*. The full matrix is always kept alongside the scalar summaries.
- **Per-acquisition consistency checks** — DC and FC strength are
  independently recomputed from each saved FC matrix and compared
  (tolerance 1e-10). Dataset-wide worst difference: **0.000e+00** for DC,
  6.9e-18 for FC strength; 165/165 consistent.
- **Pilot protection** — the original pilot (`AD/sub-019S4549/ses-01/run-01`)
  was reproduced in a scratch directory first: ALFF, ReHo, ROI time series,
  atlas labels and voxel counts came back **bit-identical** (0.000e+00), FC/DC
  within 1.8e-15 (`BIOMARKER_DATASET_AUDIT/pilot_reproduction_check.csv`). Its
  directory was then written no-clobber, so all original pilot files and the
  separate `brainnetome/` folder are untouched.

Per-acquisition outputs (21 files) live in
`derivatives/fsfast/<GROUP>/sub-*/ses-*/run-*/biomarkers/`:

```
brainnetome_246_4mm.nii.gz   brainnetome_metadata.json   brainnetome_qc_report.md
brainnetome_qc.csv           roi_timeseries.npy / .csv   roi_voxel_counts.csv
roi_timeseries_qc.csv        alff.npy / .csv             reho.npy / .csv
fc_matrix.npy                degree_centrality.npy/.csv  fc_strength.npy / .csv
regional_biomarkers.npy/.csv biomarker_metadata.json     biomarker_qc_report.md
```

`regional_biomarkers.npy` is 246 × 4 (ALFF, ReHo, DC, FC_Strength), row order
ROI 1…246; the `.csv` adds `ROI_ID` as a 5th column. Voxel-level `alff_map.nii.gz`
and `reho_map.nii.gz` are also retained.

> **Note for the modelling stage:** ALFF is stored in **raw FFT amplitude
> units**, so it spans ~1.5e3–9.1e5 across acquisitions (it scales with each
> scan's raw BOLD intensity). ReHo, being bounded, is tight (0.235–0.823).
> ALFF therefore requires subject-level normalisation before cross-subject
> modelling. No normalisation has been applied — the values are raw by design.

Full detail: `docs/FYP_RSFMRI_COMPLETE_METHODOLOGY_AND_IMPLEMENTATION.md`,
Parts O–S.

## Setup

See [`docs/FREESURFER_WSL_SETUP.md`](docs/FREESURFER_WSL_SETUP.md) for the
full WSL + FreeSurfer/FS-FAST + Python environment setup guide.

Long-running jobs (preprocessing, biomarker generation, dataset audits) are
launched through the resilient retry wrappers in `scripts/setup/*.sh`
(`retry_*.sh`), which detect WSL2 VM crashes vs. genuine Python errors and
restart/retry automatically rather than requiring a manual re-run.

The biomarker pipeline additionally stages each acquisition on the WSL-native
filesystem and copies results to `/mnt/c` only once complete: sustained numpy
I/O directly against the DrvFs mount triggered `Errno 5` failures and VM-level
crashes. Combined with checkpoint/resume (which re-validates real files on
disk, never trusting the manifest alone), an interrupted run resumes without
redoing or corrupting completed work.

## Status

| Stage | Status |
|---|---|
| Dataset audit, DICOM recovery, exclusions | Completed |
| Slice-timing investigation | Completed — `SLICE_TIMING_NOT_RECOVERABLE` |
| Preprocessing, all 6 groups | Completed — 165 / 173 eligible acquisitions |
| Preprocessing QC (per-acquisition + dataset-wide) | Completed |
| Brainnetome-246 parcellation | Completed — 165 / 165, 246/246 ROI coverage |
| ROI time-series extraction | Completed — 165 / 165, all `(135, 246)` |
| ALFF / ReHo / FC / DC / FC-strength | Completed — 165 / 165, 0 failures |
| Biomarker cross-file consistency + dataset audit | Completed — 165 / 165 consistent |
| Feature normalization / fusion | Not yet implemented |
| Multi-Attention Transformer | Not yet implemented |
| Meta-learning | Not yet implemented |
| Comorbidity fusion | Not yet implemented |
| Bayesian ordinal classifier | Not yet implemented |
| Uncertainty estimation / explainability | Not yet implemented |

Dataset-wide audit reports:
[`FINAL_6GROUP_AUDIT/FINAL_6GROUP_AUDIT_REPORT.md`](derivatives/fsfast/FINAL_6GROUP_AUDIT/FINAL_6GROUP_AUDIT_REPORT.md)
(preprocessing) and
[`BIOMARKER_DATASET_AUDIT/dataset_biomarker_summary.md`](derivatives/fsfast/BIOMARKER_DATASET_AUDIT/dataset_biomarker_summary.md)
(biomarkers).

### Per-group acquisition counts

| | AD | CN_Final | EMCI | LMCI | MCI | SMC_Final | Total |
|---|---:|---:|---:|---:|---:|---:|---:|
| Eligible for preprocessing | 25 | 29 | 30 | 22 | 35 | 32 | 173 |
| Preprocessed | 25 | 27 | 29 | 21 | 32 | 31 | **165** |
| Biomarkers generated | 25 | 27 | 29 | 21 | 32 | 31 | **165** |

8 of the 173 eligible acquisitions (spread across 5 groups) failed
preprocessing for a single, shared, root-caused reason: TR≈6.02s makes the
fixed 0.01–0.10Hz band-pass invalid (Nyquist≈0.083Hz) for that protocol
variant — correctly refused rather than silently reprocessed with altered
frequencies. Those 8 have no preprocessed BOLD, so they are not eligible for
biomarker generation either; every acquisition that *does* have preprocessed
BOLD (165/165) has a complete biomarker set.

## Excluded runs

Of the 175 total BOLD acquisitions, **2 are excluded before preprocessing**
(source files retained, never deleted). Canonical list:
[`audit/excluded_runs.csv`](audit/excluded_runs.csv) — any batch-processing
script consults it.

| Run | Reason |
|---|---|
| `sub-012S4026` ses-01 run-01 | Only 7 volumes (truncated at acquisition) |
| `sub-006S4363` ses-01 run-02 | Byte-identical duplicate of run-01 |

→ 173 eligible runs entered preprocessing (see table above for the per-group
breakdown and the 8 additional TR/Nyquist failures).
