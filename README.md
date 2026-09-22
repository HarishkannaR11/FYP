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
│   ├── biomarkers/        Brainnetome parcellation + biomarker pilot scripts
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
│   │                                        design matrix, qc/, and (pilot
│   │                                        acquisition only) brainnetome/
│   │                                        + biomarkers/
│   ├── PROCESSING_CHECKPOINT.json           per-group completed/failed/
│   │                                         remaining, updated per acquisition
│   └── FINAL_6GROUP_AUDIT/                  dataset-wide audit report + CSVs
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

## Downstream: parcellation + biomarkers (pilot stage)

Implemented and validated on a **single pilot acquisition**
(`AD/sub-019S4549/ses-01/run-01`) — not yet run across the full dataset:

- **Brainnetome-246 parcellation** (`scripts/biomarkers/`, `scripts/audit/brainnetome_pilot_single.py`):
  atlas reoriented LAS→RAS losslessly, nearest-neighbor-resampled onto the
  BOLD grid, 246/246 ROI coverage confirmed. ROI time series extracted by
  direct index-masking → verified shape `(135, 246)` (time × ROI).
- **Biomarkers** (`scripts/audit/biomarker_pilot_single.py`): ALFF and ReHo
  computed voxel-wise then ROI-aggregated; ReHo uses a 27-voxel (3×3×3)
  Kendall's-W neighborhood; FC is the full signed, unthresholded 246×246
  Pearson matrix; DC is the weighted signed row-sum of FC (self excluded).
- **Independent validation** (`scripts/audit/biomarker_pilot_validation.py`):
  separate process, reads only saved files, recomputes DC and FC-strength
  from scratch — exact match (DC diff 0.00e+00, FC-strength diff 9.89e-17).

Full detail: `docs/FYP_RSFMRI_COMPLETE_METHODOLOGY_AND_IMPLEMENTATION.md`,
Parts O–S.

## Setup

See [`docs/FREESURFER_WSL_SETUP.md`](docs/FREESURFER_WSL_SETUP.md) for the
full WSL + FreeSurfer/FS-FAST + Python environment setup guide.

Long-running jobs (a full group's preprocessing, dataset audits, pilots) are
launched through the resilient retry wrappers in `scripts/setup/*.sh`
(`retry_*.sh`), which detect WSL2 VM crashes vs. genuine Python errors and
restart/retry automatically rather than requiring a manual re-run.

## Status

| Stage | Status |
|---|---|
| Dataset audit, DICOM recovery, exclusions | Completed |
| Slice-timing investigation | Completed — `SLICE_TIMING_NOT_RECOVERABLE` |
| Preprocessing, all 6 groups | Completed — 165 / 173 eligible acquisitions |
| Preprocessing QC (per-acquisition + dataset-wide) | Completed |
| Brainnetome-246 parcellation | Pilot completed (1 acquisition) |
| ALFF / ReHo / FC / DC biomarkers | Pilot completed (1 acquisition) |
| Independent biomarker validation | Completed (pilot) |
| Full-dataset biomarker generation | Not yet implemented |
| Feature normalization / fusion | Not yet implemented |
| Multi-Attention Transformer | Not yet implemented |
| Meta-learning | Not yet implemented |
| Comorbidity fusion | Not yet implemented |
| Bayesian ordinal classifier | Not yet implemented |
| Uncertainty estimation / explainability | Not yet implemented |

Dataset-wide audit report:
[`derivatives/fsfast/FINAL_6GROUP_AUDIT/FINAL_6GROUP_AUDIT_REPORT.md`](derivatives/fsfast/FINAL_6GROUP_AUDIT/FINAL_6GROUP_AUDIT_REPORT.md).

### Per-group eligible acquisitions

| AD | CN_Final | EMCI | LMCI | MCI | SMC_Final | Total eligible |
|---:|---:|---:|---:|---:|---:|---:|
| 25 | 29 | 30 | 22 | 35 | 32 | 173 |

8 of the 173 eligible acquisitions (spread across 5 groups) failed
preprocessing for a single, shared, root-caused reason: TR≈6.02s makes the
fixed 0.01–0.10Hz band-pass invalid (Nyquist≈0.083Hz) for that protocol
variant — correctly refused rather than silently reprocessed with altered
frequencies. 165 acquisitions completed successfully.

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
