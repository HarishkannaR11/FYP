# Assessment of Alzheimer's Disease Progression Using Comorbidity-Aware Meta-Bayesian Framework with Resting-State fMRI

## Complete Methodology and Implementation Documentation

**Scope of this document:** this describes the *actual implementation* as it exists in this project's codebase, logs, configuration files, and generated outputs as of the date below. Every claim below has been cross-checked against real code, audit files, provenance JSON sidecars, or QC reports — not reconstructed from memory or textbook convention. Where the implementation does not provide enough evidence for a detail, this document says so explicitly rather than guessing.

Stages implemented and validated: **dataset audit → preprocessing → Brainnetome-246 parcellation → ROI time-series extraction → single-subject biomarker generation pilot (ALFF, ReHo, DC, FC) → independent biomarker validation.**

Stages implemented and verified: **dataset audit → preprocessing → Brainnetome-246 parcellation → ROI time-series extraction → biomarker generation (ALFF, ReHo, FC, DC) → within-subject normalization**, each across all 165 preprocessed acquisitions.

Stages **NOT YET IMPLEMENTED**: cross-subject (train-fold-only) scaling, the subject-level cohort index and cross-validation folds, feature fusion, the Multi-Attention Transformer, meta-learning, comorbidity fusion, the Bayesian ordinal classifier, uncertainty estimation, and explainability. These are described only as planned future stages, never as completed work.

Document generated: 2026-09-22.

---

## PART A — Dataset

### A.1 Dataset source

The dataset is derived from **ADNI (Alzheimer's Disease Neuroimaging Initiative)** resting-state fMRI acquisitions, supplied to this project as a mixture of raw DICOM archives (`raw_data/`) and pre-converted NIfTI files (`NIfTI_/NIfTI/<GROUP>/`), later reorganized into a BIDS-like structure under `BIDS/`.

### A.2 Imaging data available

Only resting-state BOLD EPI acquisitions are available. Verified directly (`audit/data_audit_summary.txt`, item H): **0 subjects have a T1w anatomical image** anywhere in this dataset. This absence of T1w is a hard constraint that shapes multiple downstream methodological decisions (documented in Parts H, L, N) — direct EPI→MNI registration instead of T1w-mediated registration, template-prior WM/CSF regressors instead of subject-specific tissue segmentation, and CNR marked not reliably computable.

### A.3 Subject/group organization and BIDS structure

Subjects are organized under `BIDS/sub-<ID>/ses-<NN>/func/`, with group membership recorded **only** in `BIDS/participants.tsv` (a `group` column). Session numbering was assigned by chronological acquisition datetime embedded in the original filenames (earliest = ses-01, next = ses-02, …) where a subject had multiple dated acquisitions — a convention established during the DICOM/NIfTI audit and re-used consistently, not invented per group (see `audit/ad_nifti_mapping_summary.txt`, "NOTE ON SESSION INFERENCE").

### A.4 The six diagnostic groups and progression ordering

The six groups implemented in this project, in the clinical progression ordering used throughout:

```
CN_Final → SMC_Final → EMCI → MCI → LMCI → AD
```

(Cognitively Normal → Subjective Memory Concern → Early MCI → MCI → Late MCI → Alzheimer's Disease)

This ordering is the one fixed in the Review-I report: the three mild-cognitive-impairment cohorts run early (EMCI), intermediate (MCI), late (LMCI). An earlier revision of this document recorded `EMCI → LMCI → MCI`, which was incorrect. The ordinal classifier's loss structure depends on this ordering, so it is held fixed.

### A.5 Number of subjects / acquisitions per group

These are the counts of acquisitions actually processed in the final production run (`derivatives/fsfast/FINAL_6GROUP_AUDIT/dataset_inventory.csv`), independently re-verified against `BIDS/participants.tsv` and per-group manifest files before processing began:

| Group | Subjects | Eligible acquisitions (post-exclusion) | Raw acquisitions before exclusion |
|---|---:|---:|---:|
| AD | 20 | 25 | 25 |
| CN_Final | 29 | 29 | 30 |
| EMCI | 30 | 30 | 30 |
| LMCI | 20 | 22 | 23 |
| MCI | 14 | 35 | 35 |
| SMC_Final | 21 | 32 | 32 |
| **Total** | — | **173** | 175 |

(175 total raw acquisitions minus 2 known exclusions — see Part B — equals 173 eligible acquisitions, which is the figure that entered production preprocessing.)

---

## PART B — Data Audit and Recovery

### B.1 Initial dataset audit

Before any preprocessing, a full inventory audit was performed (`audit/data_audit_summary.txt`). Verified initial state:

- Total subjects: 131
- Valid NIfTI files present: 72
- **JSON-only records (metadata present, NIfTI missing): 103**
- Zero-byte/invalid NIfTI files: 0
- Subjects with multiple acquisitions: 35
- Subjects with T1w: 0
- Subjects flagged for manual review: 39

This confirms the dataset initially arrived with a substantial fraction of acquisitions having only DICOM-derived JSON sidecar metadata but no corresponding NIfTI — these required a recovery step (raw DICOM → NIfTI conversion) before they could enter preprocessing.

### B.2 Recovery process (`audit/mci_smc_recovery_summary.txt`)

Recovery was performed per group by matching missing JSON records against the raw DICOM archives and converting the matched series.

| Group | Total JSON acquisitions | Existing NIfTI (pre-recovery) | Missing NIfTI | Raw DICOM matches found | Successfully converted | Conversion failures |
|---|---:|---:|---:|---:|---:|---:|
| MCI | 35 | 26 | 9 | 9 | 9 | 0 |
| SMC_Final | 32 | 12 | 20 | 20 | 20 | 0 |

For SMC_Final, this means **20 of the group's 32 final acquisitions (62.5%) were recovered from raw DICOM**, not originally present as NIfTI. Zero conversion failures, zero ambiguous matches, zero "no raw match" cases for either group.

### B.3 AD group — recovery status (corrected)

A previous working note in this project's history stated that AD had "19 processable / 6 unrecoverable (no matching raw DICOM)" acquisitions. **This figure was independently re-verified against the current BIDS source before the final production run and found to be incorrect/stale.** Direct verification (`audit/ad_nifti_mapping_summary.txt`, final summary):

- AD subjects listed in `participants.tsv`: 20
- AD subjects with a matching NIfTI folder: 20 (0 missing)
- Total BOLD NIfTI files found: 25
- Total classified as resting-state BOLD candidates: 25
- **Subjects with missing BOLD data: 0**

All 6 acquisitions previously flagged as "unrecoverable" (`002_S_5018_...20130516122033_501`, `006_S_4153_...20120327121013_501`, `019_S_4477_...20130213122116_501`, `019_S_4549_...20130313113703_601`, `130_S_4730_...20130612085835_701`, `130_S_4982_...20131025101621_501`) were located as valid, checksummed NIfTI files during this re-audit and were included in the final production run. All 25 AD acquisitions were processed; none were treated as unrecoverable in the final run.

### B.4 Duplicate / orphan / hash-collision audit (`audit/duplicate_nifti_summary.txt`)

A dataset-wide duplicate audit was run across all groups:

- Total JSONs analyzed: 175
- Total NIfTIs analyzed: 175
- JSONs with multiple matching NIfTIs: 0
- **Orphan NIfTIs (no matching JSON): 0**
- **Hash collisions (identical content, different filename): 1**

The single hash collision: LMCI subject `006_S_4363`, two NIfTI files with **byte-identical MD5** (`e7a650bfacca1fbf40e9a3ffa6f16797`), same acquisition datetime (`20120619135852`), exported twice under DICOM series 501 and 601.

### B.5 Documented exclusions (`audit/excluded_runs.csv`)

| Group | Acquisition | Reason | Decided by | Date |
|---|---|---|---|---|
| CN_Final | sub-012S4026/ses-01/run-01 | Only 7 volumes present (all other runs have 140); truncation confirmed to originate at the source DICOM archive/export, not at any processing step | user | 2026-09-13 |
| LMCI | sub-006S4363/ses-01/run-02 | Byte-identical duplicate of ses-01/run-01 (see B.4); run-01 retained for processing | user | 2026-09-16 |

### B.6 Final accounting

| Category | Definition | Count |
|---|---|---|
| AVAILABLE | Present as valid NIfTI from the start | 72 (initial) → 175 (post-recovery, dataset-wide) |
| RECOVERED | Converted from raw DICOM during the recovery step | 9 (MCI) + 20 (SMC_Final) = 29 |
| UNRECOVERABLE | Acquisitions with no raw DICOM match anywhere | 0 (dataset-wide, per B.2/B.3; the earlier AD "6 unrecoverable" figure was superseded, see B.3) |
| EXCLUDED (pre-processing) | Documented data-quality exclusions | 2 (B.5) |
| DUPLICATE | Hash-identical NIfTI pairs | 1 pair (B.4), resolved via the LMCI exclusion above |

---

## PART C — Slice-Timing Investigation

### C.1 BIDS JSON sidecar inspection

An exhaustive scan of all 175 BOLD JSON sidecars (`audit/slice_timing_summary.txt`) found:

- Runs with a `SliceTiming` key present: **0 / 175**
- Runs missing the key entirely (but JSON sidecar itself present): 175 / 175
- Runs missing the JSON sidecar entirely: 0

**Conclusion (verbatim from the audit): "No BOLD run in this dataset contains a 'SliceTiming' field in its JSON sidecar. Slice-timing correction is NOT technically supported by the available metadata for any run. No SliceTiming values were inferred or fabricated; this reflects the literal contents of the sidecars."**

A broader keyword scan of the same 175 sidecars for any slice-timing-adjacent field (multiband/SMS metadata, partial field names, etc.) found none beyond `SliceThickness`, `SpacingBetweenSlices`, and `PixelBandwidth` — none of which encode temporal slice order.

### C.2 DICOM-level investigation

A representative sample across all **17 distinct protocol signatures** (by TR, slice count, and software version) present in the raw DICOM archives was inspected element-by-element (`audit/slice_timing_final_exhaustive_audit.txt`), including private vendor groups (2001/2005), typically 150–200+ elements per header. Findings:

1. **Slice acquisition timing: NOT FOUND.** No standard DICOM tag (this is classic single-frame DICOM, not enhanced multi-frame with per-frame timing) and no private tag encodes a distinct acquisition instant per slice. `AcquisitionTime` and `TriggerTime` were directly compared across different slices within the same volume for multiple series (Part 3 of the audit) and confirmed **identical across every slice sampled within a volume** — i.e., the available timestamps have volume-level granularity only, not slice-level.

2. **Slice acquisition order: NOT ESTABLISHED.** `(2001,100A) Slice Number MR` (a Philips private tag) gives a per-slice **spatial** index and was confirmed present. **No tag anywhere — standard or private — states the temporal acquisition sequence** (ascending, descending, interleaved, or otherwise). Per explicit project instruction, this spatial slice index was **not** used to infer or guess a temporal order — doing so would fabricate information the header does not contain. This is the specific reason **Philips' spatial slice index must not be treated as temporal slice order**: it answers "where is this slice in space" and says nothing about "when was this slice acquired relative to the others."

3. **Multiband/SMS factor: NOT FOUND among named tags.** No standard or private tag with a decodable name matching multiband/SMS/acceleration/simultaneous terminology was present. Combined with `MRAcquisitionType=2D` and the acquisition era/protocol (ADNI2 resting-state, Philips Achieva/Intera, 2011–2015), this is *consistent with* conventional non-multiband 2D EPI — but each header also contains dozens of private-group elements with no decodable name for this vendor, whose raw values were **deliberately not inspected or pattern-matched** to guess a multiband factor. The audit explicitly reports this as "no named tag found," not as proof that no multiband factor exists anywhere in the file.

4. **Number of slices: found and reliable** — `(2001,1018) Number of Slices MR` is present and consistent with the Z-dimension of the converted NIfTI for every matched run (48 or 36 depending on protocol).

5. **Repetition Time:** present in every header, but for the TR≈6.02s protocol subgroup, the DICOM `RepetitionTime` tag under-reports the true effective volume TR by roughly 2× (cross-confirmed via `AcquisitionDuration`/`NumberOfTemporalPositions` and empirical inter-volume `AcquisitionTime` deltas). The BIDS JSON sidecar's own `RepetitionTime` value — not the raw DICOM tag — is the value used throughout this pipeline.

### C.3 Final determination

```
SLICE_TIMING_NOT_RECOVERABLE
```

**No SliceTiming values were inferred, derived, or fabricated at any point in this audit or in preprocessing.** Consequently, in every preprocessing run in this project:

```
Slice-timing correction = NOT PERFORMED
```

This is recorded per-acquisition in every provenance JSON as `"slice_timing_status": "NOT_PERFORMED"`.

---

## PART D — Preprocessing Pipeline (Step-by-Step)

The final, frozen preprocessing pipeline, as implemented in `pipeline/fsfast_production_4mm6mm.py` (function `process_one_acquisition`):

```
Raw BOLD
  ↓
Input validation
  ↓
Remove first 5 volumes
  ↓
Slice-timing correction — NOT PERFORMED
  ↓
Motion correction — FreeSurfer FS-FAST mc-afni2
  ↓
Direct EPI → MNI152NLin6Asym registration (ANTsPy, SyN)
  ↓
4 mm isotropic resampling
  ↓
6 mm FWHM Gaussian smoothing
  ↓
Linear detrending
  ↓
28-regressor nuisance regression (Friston-24 + WM + CSF + Global Signal)
  ↓
0.01–0.10 Hz band-pass filtering
  ↓
Final preprocessed BOLD
  ↓
QC
```

Each step is documented in detail in Parts E–N below, following the same purpose / input / algorithm / parameters / output / QC / rationale / limitations structure.

---

## PART E — Input Validation

**Purpose:** reject malformed or corrupted acquisitions before any processing time is spent on them, and record the exact reason rather than silently skipping.

**Checks performed** (verified directly in code, `process_one_acquisition`):

- 4D NIfTI shape check (`in_data.ndim != 4`)
- No NaN values (`np.isnan(in_data).any()`)
- No Inf values (`np.isinf(in_data).any()`)
- Valid affine: all finite AND `abs(determinant) > 1e-9` (rejects degenerate/singular affines)
- Valid TR: finite and `> 0`
- Non-empty image: `np.any(in_data != 0)`

**Handling of invalid acquisitions:** if any check fails, the acquisition is marked `status = FAIL` with the exact failing condition(s) recorded as the `reason` string, written to the group's `logs/preprocessing.log` and to `<GROUP>_preprocessing_QC_summary.csv`. Processing continues to the next acquisition — **invalid acquisitions are never silently skipped**; every FAIL is logged with its specific cause. Orientation validation is implicit in the affine check (a valid, non-degenerate affine implies a decodable orientation); no separate explicit orientation-string check beyond this was implemented in the production script.

---

## PART F — Volume Removal

**Implementation:** `N_DISCARD = 5` (fixed constant). The first 5 volumes are removed via `in_data[..., N_DISCARD:]`.

- Original volume count: read from the input NIfTI header per acquisition (typically 140 for the standard TR≈3.0s protocol; varies for the TR≈6.02s protocol subgroup — see Part M).
- Retained volume count: `original - 5` (135 for the standard protocol).
- **Verification:** every provenance JSON records both `"discarded_volumes": 5` and `"retained_volumes"` explicitly; the retained count is cross-checked against the final image's actual 4th dimension in every downstream QC pass performed in this project (e.g. `derivatives/fsfast/FINAL_6GROUP_AUDIT/dataset_data_integrity.csv`).
- **Rationale:** discarding initial volumes to allow for T1-equilibration/signal stabilization is standard rs-fMRI practice; the exact count of 5 was adopted from the frozen project specification. No further justification beyond the frozen specification is present in the project's own documentation — stated here rather than invented.

---

## PART G — Motion Correction

**Tool:** FreeSurfer FS-FAST `mc-afni2` (a FreeSurfer wrapper around AFNI's `3dvolreg`).

**What it does:** rigid-body (6 degrees of freedom) volume-to-volume realignment, estimating and correcting head motion across the time series.

**Reference frame:** frame 0 of the post-discard series (i.e., the first retained volume after the 5-volume discard in Part F).

**Six rigid-body parameters estimated:** roll, pitch, yaw (rotations, degrees) and dS, dL, dP (superior, left, posterior translations, mm) — confirmed from the `.mcdat` file's own embedded column structure (10 columns total: index, roll, pitch, yaw, dS, dL, dP, rmsold, rmsnew, trans).

**Motion output preserved:** the native `.mcdat` file produced by `mc-afni2` is copied verbatim into each acquisition's output directory as `<sub>_<ses>_task-rest_<run>_motion_parameters.mcdat` — **it is the real tool output, never synthesized or replaced.**

**Downstream use of motion parameters:**
1. **Framewise Displacement (FD):** computed per volume as `sum(|Δtranslation|) + sum(|Δrotation_rad| × 50mm)` (Power et al. 2012 convention, 50mm assumed head radius), directly from the `.mcdat` translations/rotations.
2. **Nuisance regression:** the 6 raw motion parameters feed directly into the Friston-24 expansion (Part L).

---

## PART H — MNI Normalization

**Method:** direct EPI (BOLD) → MNI template registration — **not** T1w-mediated registration, because **no subject T1w anatomical image exists in this dataset** (Part A.2).

**Registration tool:** ANTsPy, `ants.registration(type_of_transform='SyN')` — a rigid + affine + symmetric normalization (SyN) diffeomorphic registration.

**Moving image:** the temporal mean of the motion-corrected BOLD series, skull-stripped via an Otsu threshold + largest 3D connected component + hole-filling mask computed independently per acquisition (no atlas-based or template-derived brain extraction is used at this stage — the mask is derived purely from the acquisition's own intensity distribution).

**Fixed image / target template:** `MNI152NLin6Asym`, resolution 2mm, `desc-brain_T1w`, obtained via TemplateFlow.

**Transform application:** the full 4D motion-corrected series is warped onto the fixed template's native 2mm grid in a single interpolation step (`ants.apply_transforms`, linear interpolation, `imagetype=3` for time series) — not warped in two separate passes, minimizing cumulative interpolation blur.

**Final spatial grid after this step:** the exact MNI152NLin6Asym res-2mm grid (91×109×91 voxels, 2mm isotropic) before the subsequent 4mm resampling in Part I.

This is explicitly **not** subject-specific anatomical registration: there is no T1w image being registered to a template and no BOLD-to-T1w co-registration step anywhere in this pipeline. Every provenance JSON records this explicitly: `"registration_method": "ANTsPy ants.registration(type_of_transform='SyN'); direct EPI->MNI (no T1w)"`.

---

## PART I — Resampling

**Final resolution: 4 × 4 × 4 mm isotropic.**

**Target grid:** derived from the MNI152NLin6Asym template's own affine, scaled from its native 2mm to 4mm (`nilearn.image.resample_img`, `target_affine`/`target_shape` computed from the template).

**Interpolation for BOLD (continuous-valued signal):** **continuous/linear interpolation** (`interpolation="continuous"` in `nilearn.image.resample_img`) — appropriate because BOLD signal is a continuous physical quantity where interpolation between neighboring values is physically meaningful.

**This is explicitly distinct from Brainnetome label interpolation** (Part O), which uses **nearest-neighbor only**, because atlas labels are categorical integers where linear interpolation would invent meaningless fractional/blended label values.

**Affine verification:** every acquisition's final image affine is checked against the expected 4mm-scaled template affine; this check is part of the acquisition's own QC pass and is independently re-verified for the completed dataset in `derivatives/fsfast/FINAL_6GROUP_AUDIT/dataset_data_integrity.csv` (`is_4mm_isotropic` column, header-only NIfTI reads, no full-array loading, for memory safety).

---

## PART J — Smoothing

**Method: 6 mm FWHM Gaussian spatial smoothing.**

**Tool:** `nilearn.image.smooth_img(fwhm=6.0)`.

**Purpose:** increase SNR by averaging spatially correlated signal across neighboring voxels, at the cost of spatial resolution; also partially compensates for residual inter-subject anatomical variability not corrected by the EPI-only (non-T1w) normalization in Part H.

**Position in pipeline:** applied immediately after 4mm resampling (Part I) and before detrending (Part K) — i.e., on the already-resampled 4mm grid, not on the higher-resolution 2mm registered image.

**Relationship to spatial resolution:** the smoothing kernel (6mm FWHM) is larger than the 4mm voxel size, so it meaningfully blurs fine spatial detail by design — this is the expected, intended effect of this step, not an artifact.

---

## PART K — Detrending

**Method: linear temporal detrending only.**

**What is removed:** the per-voxel intercept and linear trend across time, estimated via ordinary least squares (`numpy.linalg.lstsq` on a design matrix of `[ones(nt), normalized_time_vector]`). The voxel's own temporal mean is added back after trend removal, so the detrended signal remains in native intensity units rather than being mean-centered to zero.

**Where it occurs:** immediately after smoothing (Part J), before nuisance regression (Part L).

**Quadratic detrending: NOT used.** Only the linear (first-order) term is modeled and removed; no quadratic or higher-order polynomial trend term is included anywhere in this implementation.

---

## PART L — Nuisance Regression

**Total regressors: up to 28**, composed of:

```
Friston-24 (24 columns)
+ White Matter signal (1 column, conditional)
+ CSF signal (1 column, conditional)
+ Global Signal (1 column)
+ intercept (1 column)
```

### L.1 Friston-24, explicitly

Built from the 6 raw motion parameters (Part G) as:

- 6 raw motion parameters (roll, pitch, yaw, dS→x, dL→y, dP→z translations)
- 6 temporal derivatives (first-order differences of the above, first frame = 0)
- 6 squared original motion parameters
- 6 squared temporal derivatives

= 24 columns total, verified programmatically in code (`F24.shape == (nt, 24)` assertion) for every acquisition.

### L.2 White matter and CSF regressors

**There are no T1w anatomical images in this dataset, so subject-specific tissue segmentation is not possible and was not attempted.** WM and CSF signals are instead derived from **published population template tissue probability priors**:

- Source: `MNI152NLin2009cAsym` resolution-2mm `label-WM`/`label-CSF` `probseg` maps (TemplateFlow).
- WM mask: prior probability ≥ 0.95, eroded by 1 iteration (3×3×3 structuring element) to reduce grey-matter partial-volume contamination.
- CSF mask: prior probability ≥ 0.90, **not eroded** (the ventricular CSF prior is thin enough at this resolution that erosion would remove nearly the entire mask — this was empirically verified during pilot development, not assumed).
- **Reliability gate:** a candidate WM or CSF regressor is only included if (a) its mask has ≥20 voxels **and** (b) its correlation with the global signal has `|r| < 0.98` (i.e., it is not simply a near-duplicate of the global signal, which would carry no independent information). If either condition fails for a given acquisition, that regressor is marked `"NOT PERFORMED"` for that specific acquisition and the design matrix has fewer columns — this varies per acquisition (see example below) and is recorded explicitly, never silently substituted with a fabricated regressor.

**These WM/CSF regressors must not be described as subject-specific T1w segmentation — they were not implemented that way and this dataset provides no basis for that method.**

### L.3 Global signal

Mean BOLD signal across all voxels within the acquisition's own brain mask (Part D pipeline's mask, derived during the 4mm-resample stage as `template_brain_mask AND non-constant-BOLD-coverage`), computed per timepoint.

### L.4 Design matrix dimensions — worked example

For `sub-019S4549/ses-01/run-01` (actual provenance JSON, verified): 27 columns were used (`intercept + 24 Friston + WM + GS`) — **CSF was excluded for this specific acquisition** because it failed the reliability gate (`"CSF_status": "NOT PERFORMED"`). This demonstrates the design is **not a fixed 28 columns for every acquisition** — it is 26–28 depending on whether WM and/or CSF individually pass the reliability check for that acquisition. The maximum possible (all components present) is 28; fewer columns occur when the gate excludes WM and/or CSF.

### L.5 Implementation and residual

Nuisance regression is implemented as OLS (`numpy.linalg.lstsq`) of the detrended signal (Part K output) against the full nuisance design matrix; **the regression residual, with the per-voxel temporal mean added back, becomes the input to band-pass filtering (Part M).** This residual is not yet the final output — band-pass filtering is applied afterward.

---

## PART M — Band-Pass Filtering

**Band: 0.01–0.10 Hz**, applied via `scipy.signal.butter` (2nd-order Butterworth) combined with `scipy.signal.filtfilt` for zero-phase (non-time-shifting) filtering.

**TR usage:** the actual TR is read from each acquisition's own NIfTI header (`bold_img.header.get_zooms()[3]`) — **never assumed**. Sampling frequency `fs = 1/TR`; Nyquist `= fs/2`.

**Nyquist validation:** every acquisition is checked for `0 < 0.01 < 0.10 < Nyquist` **before** the filter is applied. If this fails, the acquisition is marked `FAIL` with the exact reason (`"invalid filter band for TR=... (Nyquist=...)"`) and **the frequencies are never silently altered** to make an otherwise-invalid filter "work."

**Real consequence, verified in the completed dataset:** 8 acquisitions across 5 groups (CN_Final ×2, EMCI ×1, LMCI ×1, MCI ×3, SMC_Final ×1) failed this exact check. All 8 share the same TR≈6.02s (roughly double the standard ≈3.0s protocol used elsewhere in this dataset — a genuine, pre-existing dual-TR protocol variant, independently confirmed by direct header inspection, not a processing artifact). At TR=6.02s, Nyquist≈0.083Hz, which is below the 0.10Hz upper cutoff, making the fixed band mathematically invalid for these acquisitions. Per the frozen protocol, these were correctly refused rather than silently reprocessed with a different band.

**Position in pipeline:** the final step before the acquisition is considered "preprocessed." The band-pass-filtered residual (mean restored, masked to zero outside the brain mask) **is the final preprocessed BOLD** written to `<sub>_<ses>_task-rest_<run>_desc-preproc_bold.nii.gz`.

---

## PART N — Preprocessing QC

All QC metrics below are computed and stored per acquisition in `qc/qc_metrics.csv` / `.txt` / `qc_report.md`, and aggregated across the completed dataset in `derivatives/fsfast/FINAL_6GROUP_AUDIT/dataset_qc_summary.csv`. **GCOR is not computed in this final production pipeline** (it was used in an earlier single-subject pilot QC script but is not part of `fsfast_production_4mm6mm.py`'s QC set — not claimed here as implemented in production).

### N.1 tSNR (temporal Signal-to-Noise Ratio)

- **Definition:** `mean(|temporal signal|) / temporal standard deviation`, computed per voxel, within the brain mask.
- **Stage used:** the **smoothed, pre-detrend, pre-nuisance, pre-band-pass** intermediate (`sm_data` in code) — **not** the final band-pass-filtered residual. This is deliberate: the final residual is close to zero-mean by construction (most temporal variance has been regressed/filtered out), so tSNR computed there would be numerically unstable and not reflect genuine signal stability.
- **Units:** dimensionless ratio.
- **Direction:** higher generally indicates greater temporal signal stability.
- **Limitation:** near-zero-variance voxels (e.g., at mask edges under high motion) can produce artificially inflated tSNR; this was independently investigated and confirmed in this project (e.g., `sub-018S4313/ses-01` in CN_Final, see the FINAL_6GROUP_AUDIT report) rather than accepted at face value.

### N.2 SNR

- **Definition:** `mean(signal in native-space brain mask) / std(native-space background, non-brain nonzero voxels)` — an fBIRN-style SNR.
- **Stage used:** the **native, pre-registration** raw input, using an Otsu-threshold brain mask computed independently for this purpose.
- **Limitation:** SNR cannot be meaningfully computed on the *final* MNI-space output, because that image is exactly zero outside its brain mask by construction (masked during production) — there is no background region left to estimate noise from at that stage. This was directly verified (not assumed) during the AD single-subject independent QC audit.

### N.3 CNR

**`CNR = NOT_RELIABLY_COMPUTABLE`** for every acquisition in this project. Reason, stated consistently everywhere it appears: **no T1w anatomical image exists for tissue segmentation**, and CNR requires a valid GM/WM/CSF tissue-contrast definition that this dataset cannot provide. No tissue segmentation was fabricated to force a CNR number.

### N.4 FD (Framewise Displacement)

- **Definition:** Power et al. (2012): `sum(|Δtranslation|) + sum(|Δrotation_rad| × 50mm)` per volume.
- **Source:** the native `.mcdat` motion parameters (Part G) — recomputed independently multiple times in this project's audit history and confirmed to match the pipeline's own stored value **exactly** (e.g., `AD_INDEPENDENT_QC_AUDIT.md`: diff = 1.11e-16).
- **Reported:** mean, median, max, and counts/percentages of volumes exceeding 0.2mm/0.5mm/1.0mm reference lines.
- **Handling of high motion:** acquisitions are **never automatically excluded** for high FD — flagged `WARN` with the exact reason, per the frozen project policy.

### N.5 DVARS

- **Definition:** `sqrt(mean((BOLD_t - BOLD_{t-1})^2))` across in-mask voxels, per timepoint (first volume set to 0, no preceding volume to difference against).
- **Type: RAW-INTENSITY DVARS** — computed directly on native scanner intensity units, with **no normalization** to percent-signal-change or standardization. This is stated explicitly in every metadata/QC file (`"dvars_type": "RAW_INTENSITY"`).
- **Stage used:** same pre-detrend smoothed stage as tSNR (N.1), for the same reason.
- **Explicit limitation, stated everywhere it is used in this project:** raw-intensity DVARS values must **not** be compared against literature thresholds (e.g., the commonly-cited ~5/~20 guidance), which assume standardized or percent-signal-change DVARS. No such comparison is made anywhere in this project's QC reporting.

### N.6 Spatial entropy

- **Definition:** Shannon entropy, `H = -Σ p·log2(p)`, of a 256-bin histogram of in-mask voxel intensities from the temporal-mean of the pre-detrend smoothed stage.
- **Units:** bits.
- **Interpretation:** treated as an exploratory, context-dependent descriptive metric — **no universal "good/bad" direction is assigned anywhere in this project.**

### N.7 Temporal entropy

- **Definition:** mean, across in-mask voxels, of the Shannon entropy (16-bin histogram) of each voxel's own z-scored time series (pre-detrend smoothed stage).
- **Units:** bits.
- **Interpretation:** same exploratory/context-dependent treatment as spatial entropy.

### N.8 NaN/Inf checks

Every final image is checked for `np.isnan().sum()` and `np.isinf().sum()`, recorded per acquisition and re-verified independently at the dataset level (0/165 completed acquisitions have any NaN or Inf, per `dataset_data_integrity.csv`).

### N.9 Brain-mask checks

Per-acquisition: voxel count, binary check, dimension/affine match against the final BOLD grid. Dataset-level: independently re-verified for structural validity (3D, matches BOLD grid) across all completed acquisitions.

### N.10 Registration / grid checks

Final image affine and shape checked against the expected template-derived 4mm grid for every acquisition; independently confirmed **identical across all six groups** (not merely per-group) by direct affine comparison in the FINAL_6GROUP_AUDIT, since every acquisition registers onto the same fixed template-derived target regardless of source group.

---

## PART O — Brainnetome-246 Parcellation

**Status: COMPLETE across all 165 preprocessed acquisitions.** Implemented first as a single-acquisition pilot (`sub-019S4549/ses-01/run-01`, AD group), then scaled by `scripts/biomarkers/biomarker_dataset_pipeline.py` after that pipeline was required to reproduce the pilot bit-identically (Part S.2). The atlas is re-aligned per subject onto that subject's own BOLD grid; it is not computed once and copied.

### O.1 Atlas

- File: `atlases/Brainnetome246/BN_Atlas_246_2mm.nii.gz`
- LUT: `atlases/Brainnetome246/BN_Atlas_246_LUT.txt` (ROI_ID → ROI_Name mapping)
- Native resolution: 2mm isotropic, dimensions (91, 109, 91), orientation LAS
- Native affine: `[[-2,0,0,90],[0,2,0,-126],[0,0,2,-72],[0,0,0,1]]`
- **Labels verified, not assumed:** all integers 1–246 confirmed present (0 missing, 0 unexpected/extra) by direct enumeration of unique voxel values in the pilot run.

### O.2 Alignment process

```
Brainnetome 2mm (LAS)
  ↓ lossless canonical reorientation (LAS→RAS, exact array flip, zero interpolation)
  ↓ nearest-neighbor resampling onto the target BOLD grid
4mm Brainnetome atlas aligned to BOLD (RAS, matches BOLD affine exactly)
  ↓ ROI voxel-count validation
  ↓ ROI time-series extraction
```

**Why nearest-neighbor is required:** the atlas is a categorical label image — each voxel's value is a discrete ROI identifier, not a continuous physical quantity. Linear/cubic/spline interpolation would blend adjacent labels into fractional, semantically meaningless values (e.g., "37.4"); nearest-neighbor is the only interpolation mode that preserves label integrity. This distinction from BOLD interpolation (Part I) is enforced explicitly in code (`interpolation="nearest"`) and documented per instruction not to conflate the two.

**Grid verified identical to BOLD:** same dimensions, same affine (`np.allclose` check), confirmed in the pilot and additionally confirmed across a sample from all six production groups in the FINAL_6GROUP_AUDIT (every acquisition registers onto the identical fixed template-derived grid, so this check generalizes).

### O.3 Validation result — all 165 acquisitions

Recomputed per acquisition from each saved aligned atlas file (never assumed, never extrapolated from the pilot):

| Check | Result |
|---|---|
| Acquisitions with 246/246 ROIs present | **165 / 165** |
| Acquisitions with any zero-voxel ROI | **0** |
| Acquisitions with any ROI <5 voxels | **0** |
| Acquisitions with any ROI <10 voxels | **0** |
| Smallest ROI anywhere in the dataset | **10 voxels** |
| Atlas/BOLD dimension match | 165 / 165 |
| Atlas/BOLD affine match | 165 / 165 |

Pilot acquisition specifically: min 10 / median 68.0 / max 186 voxels.

Complete coverage is expected by construction — every acquisition is registered to the same template-derived 4mm grid, so the aligned atlas is identical across the cohort — but it was verified per acquisition rather than assumed, and the per-acquisition results are stored in each `biomarkers/brainnetome_qc.csv` and aggregated in `BIOMARKER_DATASET_AUDIT/dataset_biomarker_summary.csv`.

---

## PART P — ROI Time-Series Extraction

**Implemented method:** direct index-masking on the already-aligned atlas grid — **not** routed through `nilearn.input_data.NiftiLabelsMasker`, specifically to avoid that tool's own internal (and potentially silent) re-resampling of the label image against its own default target affine logic. Since the atlas was already verified grid-identical to the BOLD (Part O.2), direct masking is mathematically equivalent and removes any risk of an unintended second resampling pass.

**Per ROI, per timepoint:** `ROI_value = mean(BOLD signal across all voxels labeled with that ROI ID)`.

**Verified output shape: (135, 246)** — **135 time points (rows) × 246 Brainnetome ROIs (columns).** This orientation (T × ROI) was explicitly verified via direct NumPy inspection (`x.shape == (135, 246)` confirmed True, `(246, 135)` confirmed False) and is used consistently as the project's fixed convention.

**Validation performed:**
- NaN count: 0
- Inf count: 0
- Zero-variance ROIs: 0
- Time-point count cross-checked against the current preprocessed BOLD's own volume count: **135 == 135, exact match**

**Output formats:** `roi_timeseries.csv` (with a `time` index column and `ROI_001`…`ROI_246` columns) and `roi_timeseries.npy` (raw `(135, 246)` float64 array, no extra columns).

---

## PART Q — Biomarker Generation

**Status: COMPLETE across all 165 preprocessed acquisitions, 0 failures.** Definitions below are unchanged from the validated pilot; the dataset-scale implementation reproduces the pilot bit-identically for ALFF, ReHo, ROI time series and atlas labels (Part S.2).

### Q.1 ALFF

- **Definition/concept:** amplitude of low-frequency fluctuation — the amplitude spectrum of the voxel time series, summarized within a specified frequency band.
- **Frequency range:** 0.01–0.10 Hz.
- **Implementation:** computed **voxel-wise** from the *current preprocessed BOLD* (not from the 246-ROI time-series matrix) — per the explicit methodological requirement that ALFF must use voxel-level data. Per-voxel: mean-center the time series, FFT (`numpy.fft.rfft`), take the amplitude spectrum (`abs`), and average the amplitude across frequency bins falling within [0.01, 0.10] Hz (Zang et al. 2007 mean-amplitude convention).
- **Documented methodological note:** because the current preprocessed BOLD was *already* band-pass filtered to 0.01–0.10Hz during preprocessing (Part M), computing ALFF on its full available spectrum is effectively equivalent to the traditional definition, since negligible power exists outside that band post-filtering. This is a documented consequence of the frozen pipeline design, not a substitution of methods.
- **ROI aggregation:** mean of the voxel-level ALFF map within each of the 246 Brainnetome ROIs (using the already-aligned atlas from Part O, not re-resampled).
- **Output shape:** `(246,)`.
- **Units:** raw FFT amplitude units (not normalized to mALFF/fALFF).
- **QC (pilot):** 246 values, 0 NaN, 0 Inf, 0 negative values (independently confirmed).

### Q.2 ReHo

- **Definition:** Regional Homogeneity, via Kendall's Coefficient of Concordance (KCC) among a voxel and its spatial neighbors' time series.
- **Neighborhood actually implemented: 27-voxel (3×3×3 cube, including the center voxel).** This is the original Zang et al. (2004) definition and the default used by AFNI `3dReHo` and DPARSF/DPABI — chosen as the most common, best-validated option in the literature, not arbitrarily.
- **Boundary handling:** neighbors are restricted to those also inside the brain mask; near the mask boundary this means fewer than 27 series contribute (K varies per voxel), and the KCC formula explicitly uses the *actual* K for that voxel, not a fixed 27.
- **Implementation:** computed **voxel-wise** on the current preprocessed BOLD (not the ROI matrix), for all 28,549 in-mask voxels in the pilot acquisition, completing in ~68 seconds.
- **ROI aggregation:** mean of the voxel-level ReHo map within each of the 246 ROIs.
- **Output shape:** `(246,)`.
- **Units:** Kendall's W, theoretically bounded [0, 1].
- **QC (pilot):** 246 values, 0 NaN, 0 Inf, all values independently confirmed within [0,1] (min=0.3198, max=0.6803).

### Q.3 Functional Connectivity

```
ROI time series (135 × 246)
  ↓ Pearson correlation (every ROI pair)
246 × 246 FC matrix
```

- **Method:** Pearson correlation (`numpy.corrcoef`) between every pair of the 246 ROI time series.
- **Self-connections:** diagonal is included in the matrix (≈1.0 by construction) but **excluded** from degree centrality and regional-strength sums (Q.4/R).
- **Negative correlations: retained** — not clipped, not absolute-valued.
- **Thresholding: none.** The full weighted matrix is preserved, per the explicit instruction to prefer a weighted representation over an arbitrary/unvalidated threshold.
- **Output shape:** `(246, 246)`.
- **QC (pilot, independently recomputed):** 0 NaN, 0 Inf, diagonal mean/min/max all exactly 1.00000000, symmetry error `max(|FC-FC.T|)` = **1.11e-16** (floating-point noise).

### Q.4 Degree Centrality

- **Exact implemented definition:** `DC_i = Σ_{j≠i} FC[i,j]` — **weighted** (raw correlation magnitudes, not binarized), **signed** (negative correlations retained), **no threshold applied**, self-connection (diagonal) explicitly excluded from the sum.
- **Consequence, stated explicitly:** because correlations are signed and unthresholded, a region with many strong negative connections contributes negatively to its own DC value, potentially producing negative DC — this is an expected mathematical consequence of the chosen definition, confirmed in the pilot (DC range: -16.84 to +12.94), not an error.
- **Output shape:** `(246,)`.
- **Not implemented:** binary degree centrality (thresholded, unweighted counting of "connected/not connected" edges) was **not** used.

---

## PART R — Regional Biomarker Matrix

A combined per-ROI feature table is generated per acquisition:

| Column | Source |
|---|---|
| ROI_ID | 1–246 |
| ROI_Name | `BN_Atlas_246_LUT.txt` |
| ALFF | Part Q.1 |
| ReHo | Part Q.2 |
| DC | Part Q.4 |
| FC_Strength | derived summary, see below |

**FC_Strength** (a derived regional summary, distinct from the full FC matrix): `FC_strength_i = mean_{j≠i} FC[i,j]` — the *mean* (not sum) of ROI i's off-diagonal connectivity.

**Verified shape: 246 rows × 6 columns** (ROI_ID, ROI_Name + 4 biomarker columns).

**The full 246×246 FC matrix is retained separately** (`fc_matrix.npy`/`.csv`) — it is never replaced or discarded in favor of the single-value FC_Strength summary; both are kept as distinct artifacts for different future uses.

---

## PART S — Biomarker Validation (Independent)

### S.1 Pilot acquisition — independent validation

An independent validation pass was run as a **separate script and process** from biomarker generation, reading only the saved output files (never reusing in-memory values from generation, never modifying the source files). Results:

| Check | Result |
|---|---|
| ALFF shape | (246,) — confirmed |
| ALFF NaN / Inf / negative | 0 / 0 / 0 |
| ReHo shape | (246,) — confirmed |
| ReHo NaN / Inf / missing ROI | 0 / 0 / 0 |
| FC shape | (246, 246) — confirmed |
| FC NaN / Inf | 0 / 0 |
| FC diagonal | exactly 1.00000000 (mean, min, max) |
| FC symmetry error | 1.11e-16 |
| DC — independently recomputed from `fc_matrix.npy` using the exact documented definition | max absolute difference vs. saved: **0.00e+00 (exact match, all 246 ROIs)** |
| FC regional strength — independently recomputed | max absolute difference vs. saved: **9.89e-17 (exact match)** |
| Regional biomarker matrix | 246 rows, ROI IDs 1–246 exactly once, 0 duplicates, 0 missing, all 4 columns exactly match their source files |

**Final validation status: PASS.** Every mathematical/structural check (shapes, NaN/Inf, symmetry, diagonal, DC agreement, FC-strength agreement, cross-file internal consistency) passed. This validation is a mathematical/structural verification only — it makes no claim about the biological or diagnostic meaningfulness of the values (see Part X, Limitations).

### S.2 Pilot reproduction gate (before dataset scale-up)

Before any other acquisition was processed, the dataset-scale implementation was required to reproduce the already-validated pilot. It was run into a scratch directory (`BIOMARKER_DATASET_AUDIT/PILOT_VERIFICATION/`) and compared against the pilot's existing outputs, which were never written to:

| Quantity | Max absolute difference vs. pilot |
|---|---|
| ALFF (246,) | **0.000e+00** (bit-identical) |
| ReHo (246,) | **0.000e+00** (bit-identical) |
| ROI time series (135, 246) | **0.000e+00** (bit-identical) |
| Aligned atlas labels | **0.000e+00** (bit-identical) |
| ROI voxel counts (246,) | **0.000e+00** (bit-identical) |
| FC (246, 246) | 3.331e-16 |
| Degree centrality (246,) | 1.776e-15 |
| FC regional strength (246,) | 9.888e-17 |

Recorded in `BIOMARKER_DATASET_AUDIT/pilot_reproduction_check.csv`. This gate matters because the dataset implementation computes ReHo by a different, vectorized route (precomputing per-voxel ranks once and accumulating neighbour rank-sums by shifted array addition, rather than looping over ~28,500 voxels). The result being bit-identical proves the optimization did not alter the definition — it reduced ReHo runtime from ~68s to ~3s per acquisition.

The pilot directory was then written in **no-clobber** mode: files that already existed were preserved, not regenerated. All eight original pilot files and the separate `brainnetome/` folder remain untouched.

### S.3 Dataset-level verification — all 165 acquisitions

Every acquisition carries its own cross-file consistency check, in which DC and FC regional strength are recomputed from that acquisition's stored FC matrix and compared against the stored vectors (tolerance 1e-10):

| Check | Result |
|---|---|
| Acquisitions audited | 165 / 165 |
| ROI time series shape (135, 246) | 165 / 165 |
| Non-finite values in regional time series | 0 |
| Regions with zero temporal variance | 0 |
| ALFF non-finite or negative values | 0 |
| ReHo values outside valid [0,1] | 0 |
| ReHo range across dataset | 0.2352 – 0.8231 |
| FC non-finite values | 0 |
| FC max diagonal error | 2.220e-16 |
| FC max asymmetry | 2.220e-16 |
| FC negative off-diagonal share | mean 50.09% (range 45.55–53.12%) |
| **DC recomputed from stored FC, max difference** | **0.000e+00** |
| FC-strength recomputed, max difference | 6.939e-18 |
| Acquisitions passing all consistency checks | **165 / 165** |
| Processing failures | **0** |

The DC result is the single strongest piece of evidence in this stage: it shows not merely that the files are well formed, but that the stored values are exactly what the documented definition produces, for every acquisition.

---

## PART S2 — Within-Subject Normalization

**Status: COMPLETE across all 165 acquisitions, 0 failures.** Implemented in `scripts/biomarkers/normalize_within_subject.py`.

### S2.1 Why this stage exists

Raw ALFF is expressed in FFT amplitude units proportional to each acquisition's raw BOLD intensity. Across the dataset the per-acquisition mean ALFF spans **850.29 to 215,172.31** — a factor of roughly 250. ReHo, being a bounded coefficient, spans only 0.311 to 0.589. Feeding both unnormalized into a single model would let scanner and protocol scaling dominate any biological signal.

### S2.2 Leakage-free by construction

Every statistic used is computed from **that one acquisition's own 246 regional values**. No cross-subject mean, no cross-subject standard deviation, no ROI-wise statistic across subjects, and no diagnostic label is used anywhere in this stage. Cross-subject scaling is deliberately deferred to the modelling stage, where it must be fitted on training subjects only, after a subject-level split.

### S2.3 Transforms as implemented

| Feature | Transform | Rationale |
|---|---|---|
| ALFF → **mALFF** | `ALFF_i / mean(ALFF)` | Removes raw-intensity scaling. A ratio, not a z-score, because ALFF is strictly positive and the ratio form is what the literature reports |
| ReHo → **mReHo** | `ReHo_i / mean(ReHo)` | Removes each subject's global coherence offset |
| DC → **DC_z** | `(DC_i − mean(DC)) / std(DC, ddof=0)` | A z-score because DC is signed, which rules out min–max scaling. `ddof=0` chosen and applied identically to all 165 |
| FC → **Fisher-z FC** | `arctanh(clip(r, ±0.999999))` on off-diagonal; diagonal exactly 0 | Variance-stabilizes correlations. Signed, unthresholded, negatives retained, **not** z-scored within subject |

**Diagonal handling.** The FC diagonal is exactly 1.0, and `arctanh(1) = inf`. The implementation zeroes the diagonal **before** applying `arctanh`, so an infinity is never produced at any point (no transient `inf`, no RuntimeWarning), then sets the diagonal to exactly 0.0. Clipping is applied to off-diagonal entries only.

### S2.4 FC_Strength excluded as redundant

The pipeline stores a regional connectivity strength defined as the **mean** of each region's off-diagonal connectivity, while DC is the **sum** over the same entries. They therefore differ only by the constant factor 245:

- Measured Pearson correlation between DC and FC_Strength: **1.000000000000** in every acquisition checked
- Max absolute deviation from the exact relation `FC_Strength = DC / 245`: **6.939e-18**
- Max relative deviation: **7.187e-16**

Under any scale-invariant normalization the two become numerically identical. FC_Strength is therefore **excluded from the model feature set** as perfectly collinear. Its raw files (`fc_strength.npy`, `fc_strength.csv`) are retained unaltered. This means the independent regional feature count is **three**, not four — a correction to the Review-I design, which specified a 246 × 4 input matrix.

### S2.5 Verification — all 165 acquisitions

| Check | Result |
|---|---|
| Acquisitions normalized | **165 / 165** |
| Failures | **0** |
| Discrepancy vs. expected per-group counts | none |
| Input / output non-finite values | 0 / 0 |
| mean(mALFF) per acquisition | 1.000000 for all 165 |
| mean(mReHo) per acquisition | 1.000000 for all 165 |
| mean(DC_z) | ≤ 1.6e-16 (target 0) |
| std(DC_z) | 1.000000 for all 165 |
| Fisher-z FC symmetric | 165 / 165, max asymmetry 5.329e-15 |
| Fisher-z FC diagonal exactly 0 | **165 / 165** |
| Negative correlations preserved negative | 165 / 165 |
| Off-diagonal entries requiring clipping | **0** across the entire dataset |
| Fisher-z range across dataset | −2.2653 to 2.5924 |

The clipping count of zero confirms no region pair anywhere in the dataset is numerically perfectly correlated — the ±0.999999 bound was a safeguard that never engaged.

### S2.6 Raw-data integrity

Outputs are written to a **separate tree** (`derivatives/biomarkers_normalized/`); nothing inside any `biomarkers/` directory is modified. The script carries a hard guard that raises if a write path contains a `biomarkers` component or falls outside the output root. Independently confirmed by `git status` over `derivatives/fsfast`: **zero modified or deleted tracked files**.

### S2.7 Resulting model feature set

| Kind | Feature | Shape |
|---|---|---|
| Regional | mALFF | (246,) |
| Regional | mReHo | (246,) |
| Regional | DC_z | (246,) |
| Regional (combined) | `regional_biomarkers_normalized.npy` | (246, 3) |
| Connectivity | Fisher-z FC | (246, 246) |
| *Excluded* | *FC_Strength (= DC/245)* | — |

---

## PART T — Current File Structure

Verified directory tree for the completed pilot acquisition (`AD/sub-019S4549/ses-01/run-01/`):

```
derivatives/fsfast/AD/sub-019S4549/ses-01/run-01/
│
├── sub-019S4549_ses-01_task-rest_run-01_desc-preproc_bold.nii.gz   [FINAL]
├── sub-019S4549_ses-01_task-rest_run-01_desc-preproc_bold.json     [METADATA/provenance]
├── sub-019S4549_ses-01_task-rest_run-01_motion_parameters.mcdat    [INTERMEDIATE, native tool output]
├── brain_mask.nii.gz                                                [FINAL]
├── design_matrix.txt                                                [FINAL, nuisance design]
│
├── qc/                                                               [QC]
│   ├── qc_metrics.csv / .txt / qc_report.md
│   ├── fd_values.csv, dvars_values.csv
│   └── fd_plot.png, dvars_plot.png, tsnr_map.png, mean_bold.png, registration_qc.png
│
├── brainnetome/                                                      [derived: parcellation]
│   ├── brainnetome_246_4mm.nii.gz            [INTERMEDIATE — aligned atlas]
│   ├── roi_timeseries.csv / .npy             [FINAL — ROI signal matrix]
│   ├── roi_voxel_counts.csv                  [QC]
│   ├── roi_timeseries_qc.csv                 [QC]
│   ├── brainnetome_metadata.json             [METADATA]
│   └── brainnetome_qc_report.md              [QC]
│
└── biomarkers/                                                       [derived: biomarkers]
    ├── alff_map.nii.gz, alff_roi_values.csv/.npy       [FINAL]
    ├── reho_map.nii.gz, reho_roi_values.csv/.npy       [FINAL]
    ├── fc_matrix.npy / .csv                             [FINAL]
    ├── fc_regional_strength.csv                         [FINAL, derived]
    ├── degree_centrality.npy / .csv                     [FINAL]
    ├── regional_biomarker_matrix.csv                    [FINAL, combined]
    ├── biomarker_qc.csv                                 [QC]
    ├── validation_dc_comparison.csv                     [QC, validation-only]
    ├── validation_crosscheck_table.csv                  [QC, validation-only]
    ├── biomarker_metadata.json                          [METADATA]
    └── biomarker_qc_report.md                           [QC]
```

**Note on the pilot directory.** The listing above is the *pilot* acquisition, which retains its original filenames (`alff_roi_values.npy`, `regional_biomarker_matrix.csv`, etc.) because it was written in no-clobber mode. **The other 164 acquisitions use the dataset-scale naming**, and the pilot additionally received those files where they did not collide:

```
derivatives/fsfast/<GROUP>/sub-<ID>/ses-<NN>/run-<NN>/biomarkers/   [all 165]
├── brainnetome_246_4mm.nii.gz    brainnetome_metadata.json
├── brainnetome_qc.csv            brainnetome_qc_report.md
├── roi_timeseries.npy / .csv     roi_voxel_counts.csv
├── roi_timeseries_qc.csv
├── alff.npy / .csv               alff_map.nii.gz
├── reho.npy / .csv               reho_map.nii.gz
├── fc_matrix.npy
├── degree_centrality.npy / .csv
├── fc_strength.npy / .csv        [retained, excluded from features]
├── regional_biomarkers.npy (246x4) / .csv
├── biomarker_metadata.json       biomarker_qc_report.md
```

```
derivatives/biomarkers_normalized/<GROUP>/sub-<ID>/ses-<NN>/run-<NN>/   [all 165]
├── malff.npy / .csv
├── mreho.npy / .csv
├── dc_z.npy / .csv
├── fc_fisher_z.npy                              (246x246)
├── regional_biomarkers_normalized.npy (246x3) / .csv
└── normalization_metadata.json
```

Dataset-level audit directories:

```
derivatives/fsfast/FINAL_6GROUP_AUDIT/              preprocessing audit (10 files)
derivatives/fsfast/BIOMARKER_DATASET_AUDIT/
├── dataset_biomarker_summary.csv / .md             per-acquisition + rollup QC
├── processing_manifest.csv                         14-column per-acquisition manifest
├── processing_errors.csv                           header-only (no errors)
├── pilot_reproduction_check.csv                    Part S.2 gate
└── biomarker_processing_state.json                 resume state
derivatives/biomarkers_normalized/
├── normalization_qc.csv                            35-column per-acquisition QC
├── normalization_qc_report.md
└── normalization_state.json
```

All six groups (`AD, CN_Final, EMCI, LMCI, MCI, SMC_Final`) now carry `biomarkers/` and a mirrored `biomarkers_normalized/` entry for **every one of their 165 acquisitions**.

---

## PART U — Complete Pipeline Flow (End-to-End)

```
ADNI rs-fMRI (raw DICOM + partial NIfTI)
  ↓
Dataset audit (missing-NIfTI detection, 103 JSON-only records found)
  ↓
DICOM recovery (MCI: 9 recovered; SMC_Final: 20 recovered)
  ↓
Duplicate/hash-collision audit (1 collision found, LMCI)
  ↓
Known-exclusion documentation (CN_Final truncated run; LMCI duplicate run)
  ↓
BIDS organization (participants.tsv as sole group-membership source)
  ↓
Slice-timing investigation → SLICE_TIMING_NOT_RECOVERABLE
  ↓
Input validation (4D, finite, valid affine/TR, non-empty)
  ↓
Discard first 5 volumes
  ↓
Slice-timing correction — NOT PERFORMED
  ↓
mc-afni2 motion correction (native .mcdat preserved)
  ↓
Direct EPI → MNI152NLin6Asym (ANTsPy SyN, no T1w)
  ↓
4mm isotropic resampling (continuous interpolation)
  ↓
6mm FWHM smoothing
  ↓
Linear detrending
  ↓
26-28 regressor nuisance regression (Friston-24 + [WM] + [CSF] + GS, gated)
  ↓
0.01-0.10 Hz band-pass filtering (Nyquist-validated per acquisition; 8 acquisitions correctly rejected)
  ↓
Final preprocessed BOLD (165/173 acquisitions across 6 groups)
  ↓
Preprocessing QC (tSNR, SNR, CNR=N/A, FD, DVARS, entropy)
  ↓
[ALL 165 ACQUISITIONS:]
Brainnetome-246 alignment (nearest-neighbor, 246/246 coverage)
  ↓
ROI voxel-count validation
  ↓
ROI time-series extraction → 135 × 246 matrix
  ↓
ALFF (voxel-wise → ROI-aggregated)
ReHo (voxel-wise, 27-neighbor KCC → ROI-aggregated)
FC (246×246 Pearson, ROI-level)
DC (weighted signed sum from FC)
  ↓
Regional biomarker matrix (246 × 4) + full FC matrix (246×246) retained separately
  ↓
Per-acquisition cross-file consistency (DC recompute diff 0.000e+00) — 165/165 PASS
  ↓
Within-subject normalization (leakage-free)
  mALFF · mReHo · DC_z · Fisher-z FC   — 165/165
  FC_Strength excluded (= DC/245, perfectly collinear)
  ↓
Model feature set: 246×3 regional + 246×246 connectivity, per acquisition
  ↓
[NOT YET IMPLEMENTED:]
Comorbidity table cleaning (LMCI cohort tables still missing)
  ↓
Subject-level cohort index + stratified GROUPED CV folds (127 subjects, not 165 scans)
  ↓
Cross-subject scaling — fitted on TRAIN FOLD ONLY
  ↓
Feature fusion → Multi-Attention Transformer → Meta-learning
  ↓
Comorbidity fusion → Bayesian ordinal classifier (CN→SMC→EMCI→MCI→LMCI→AD)
  ↓
Uncertainty estimation → Explainability
```

---

## PART V — Methods Table

| Stage | Method | Parameters | Input | Output | Validation |
|---|---|---|---|---|---|
| Input validation | Header/array checks | 4D, finite, affine det≠0, TR>0 | Raw BOLD | PASS/FAIL flag | Logged per-acquisition |
| Volume removal | Array slicing | Discard first 5 | Raw BOLD | (X,Y,Z,T-5) | Retained count recorded |
| Slice timing | — | N/A | — | NOT_PERFORMED | Exhaustive DICOM+JSON audit |
| Motion correction | FS-FAST mc-afni2 (AFNI 3dvolreg) | ref=frame 0, 6-DOF rigid | Discard-5 BOLD | Motion-corrected BOLD + native `.mcdat` | Row count == volumes |
| MNI normalization | ANTsPy SyN (rigid+affine+SyN) | fixed=MNI152NLin6Asym res-2 | Motion-corrected BOLD | 2mm MNI-space BOLD | Affine/grid match to template |
| Resampling | nilearn `resample_img` | 4mm isotropic, continuous interp. | 2mm MNI BOLD | 4mm BOLD | Voxel size == 4mm exactly |
| Smoothing | nilearn `smooth_img` | 6mm FWHM Gaussian | 4mm BOLD | Smoothed 4mm BOLD | — |
| Detrending | OLS (`lstsq`) | linear only, mean restored | Smoothed BOLD | Detrended BOLD | Timepoints unchanged |
| Nuisance regression | OLS (`lstsq`) | Friston-24+[WM]+[CSF]+GS, 26-28 cols | Detrended BOLD | Residual BOLD | Design shape logged |
| Band-pass | `scipy.signal.butter`+`filtfilt` | 0.01-0.10Hz, order 2 | Residual BOLD | Final preprocessed BOLD | Nyquist check per acquisition |
| Brainnetome alignment | nilearn `resample_img`, nearest-neighbor | 2mm→4mm | Atlas + final BOLD | Aligned 4mm atlas | 246/246 label check |
| ROI extraction | Direct index masking | mean per ROI per timepoint | Final BOLD + aligned atlas | (T,246) matrix | NaN/Inf/zero-var check |
| ALFF | FFT amplitude, band-mean | 0.01-0.10Hz | Final BOLD (voxel) | (246,) | NaN/Inf/negative check |
| ReHo | Kendall's W (KCC) | 27-neighbor (3×3×3 incl. center) | Final BOLD (voxel) | (246,) | Bounded [0,1] check |
| FC | Pearson correlation | unthresholded, signed | ROI matrix (T,246) | (246,246) | Symmetry, diagonal check |
| DC | Weighted signed sum | Σ_j≠i FC[i,j] | FC matrix | (246,) | Independently recomputed, exact match |
| Biomarker validation | Independent re-read + recompute | separate process | All saved outputs | PASS/FAIL table | Cross-file consistency |

---

## PART W — Reproducibility

Verified directly from provenance JSON sidecars and environment checks (not assumed):

| Component | Version |
|---|---|
| Python | 3.12.3 |
| NumPy | 2.3.5 |
| SciPy | 1.15.3 |
| NiBabel | 5.4.2 |
| Nilearn | 0.14.1 |
| Pandas | 3.0.5 |
| Matplotlib | 3.11.2 |
| ANTsPy (antspyx) | 0.6.3 |
| TemplateFlow | 25.1.2 |
| FreeSurfer | 7.4.1 (build `freesurfer-linux-ubuntu22_x86_64-7.4.1-20230614-7eb8460`) |
| Operating environment | WSL2 (Ubuntu) on Windows 11 |

**Resource constraints:**
- **Single worker throughout**: `n_procs=1`, no multiprocessing/threading/joblib parallelism anywhere in the production scripts; ANTs/ITK/OMP/OpenBLAS/MKL thread counts explicitly pinned to 1 via environment variables at process start.
- **One acquisition processed at a time**, with explicit `gc.collect()` and scratch-directory cleanup after each acquisition to control peak memory.
- **Checkpoint/resume strategy:** `derivatives/fsfast/PROCESSING_CHECKPOINT.json` tracks `completed`/`failed`/`remaining` per group, updated after every single acquisition (not just per group). Resume logic **validates actual output files on disk** (final BOLD loadable and 4D, mask 3D, motion/design/JSON files present and readable) before treating an acquisition as complete — not merely checking that a file exists — so a crash mid-acquisition never causes a corrupted partial result to be silently accepted as done.
- **WSL2 stability:** this project's processing environment experienced recurring host-RAM-driven WSL2 VM crashes (`Wsl/Service/0x80072746` and related), root-caused to host memory exhaustion rather than the pipeline itself (`audit/wsl_stability_report.txt`). All production runs were executed via a resilient retry-wrapper that detects a crash (no completion marker, no Python traceback) versus a genuine error (traceback present) and automatically restarts WSL and resumes from the last validated checkpoint.

**Output naming convention:** BIDS-inspired but not strictly BIDS-compliant — `sub-<ID>/ses-<NN>/run-<NN>/<sub>_<ses>_task-rest_<run>_desc-preproc_bold.nii.gz`, with `desc-preproc`/`desc-` suffixes used consistently for derived files.

**Provenance:** every final preprocessed BOLD has a paired JSON sidecar recording source path, original/final dimensions and voxel size, TR, discarded volumes, slice-timing status, motion correction tool, MNI template, registration method, resampling/smoothing/detrending/nuisance/band-pass parameters, software versions, processing timestamp (UTC), processing status, and source checksum before/after (verified unchanged for every completed acquisition).

---

## PART X — Limitations

- **No T1w anatomical images exist for any subject in this dataset.** This is the single most consequential limitation, propagating into: direct (non-T1w-mediated) EPI→MNI registration (Part H); WM/CSF regressors sourced from population template priors rather than subject-specific tissue segmentation (Part L.2); and CNR being permanently marked `NOT_RELIABLY_COMPUTABLE` (Part N.3) for every acquisition in this project.
- **Slice-timing metadata is genuinely unavailable** (Part C) — not merely unused. Slice-timing correction is skipped for the entire dataset; residual slice-timing-related signal bias, if any, is not corrected.
- **Direct EPI-to-MNI registration** is inherently less precise than T1w-mediated registration (EPI has lower anatomical contrast and geometric distortion characteristics different from T1w); the registration quality achieved here has not been benchmarked against a T1w-based alternative, since none is possible with this dataset.
- **Raw-intensity DVARS** (Part N.5) is not directly comparable to the standardized/percent-signal-change DVARS thresholds commonly cited in the literature; no such comparison is made anywhere in this project.
- **Signed, unthresholded FC and DC** (Parts Q.3/Q.4) mean these values differ conceptually from binary/thresholded graph-theoretic measures common in some connectivity literature; this was a deliberate choice (preferring a fully weighted representation over an unvalidated threshold) and should be interpreted accordingly by any downstream consumer.
- **Validation to date is mathematical and structural, not biological.** The 165/165 PASS documented in Parts S and S2 establishes that shapes, finiteness, symmetry and the stored values' agreement with their documented definitions are all correct. It makes no claim that the values are biologically or diagnostically meaningful — that can only be established by the modelling stage, which is not yet built.
- **Effective sample size is 127 subjects, not 165 acquisitions.** 96 subjects contribute one acquisition, 25 contribute two, 5 contribute three and 1 contributes four. The smallest class by subject count is MCI at **14 subjects**, despite being the largest by acquisition count (32) — it is session-inflated at 2.3 scans/subject. Because validation splits must be grouped by subject, 14 bounds what can be claimed about that class, and a single held-out test split would place roughly 3 MCI subjects in test. Repeated stratified grouped cross-validation is therefore required. No subject appears in more than one group, so labels are subject-consistent.
- **Comorbidity data is incomplete and the gap is group-structured.** Cohort tables were obtained for five of six groups; the **LMCI tables are absent entirely**, leaving 20 subjects with no CDR, MMSE, GDS, medical-history or APOE4 data. Of 21 subjects lacking APOE4, 20 are the whole LMCI group and only 1 is a genuine individual gap. Because the missingness is group-structured rather than random, an availability indicator would act as a near-perfect proxy for LMCI membership and must **not** be used as a feature. The correct remedy is to obtain the missing tables.
- **Hypertension and diabetes are not directly available.** The ADNI MEDHIST tables supplied carry only broad categories — `MH4CARD` (cardiovascular, which includes hypertension but also other conditions) and `MH9ENDO` (endocrine-metabolic, which includes diabetes but also thyroid disease). Labelling these as "hypertension" and "diabetes" would overstate what the data supports; the specific conditions require ADNI's detailed medical-history recode tables.
- **One nominal biomarker is redundant.** FC_Strength is exactly DC/245 (Part S2.4), so the independent regional feature count is three, not the four specified in the Review-I design.
- **8 acquisitions (of 173 eligible) could not be preprocessed** due to a genuine TR≈6.02s / Nyquist incompatibility with the frozen 0.01–0.10Hz band-pass specification (Part M) — these are permanently excluded from the current derivatives unless the project's methodology is explicitly revisited for that TR subgroup.
- **Undocumented private DICOM tags** (Part C.2) could not be ruled out as carrying multiband/slice-timing information; the audit explicitly reports "no named tag found," not "no such information exists in the file."

None of the above limitations invalidate the pipeline — they are properties of the source dataset (ADNI rs-fMRI without T1w) that the implementation handles by explicit, documented exclusion or substitution rather than fabrication.

---

## PART Y — What Has and Has Not Been Completed

| Stage | Status |
|---|---|
| Dataset audit (missing-NIfTI, recovery, duplicates, exclusions) | **COMPLETED** |
| Slice-timing investigation | **COMPLETED** |
| Preprocessing (all 6 groups) | **COMPLETED** (165/173 eligible acquisitions; 8 excluded for TR/Nyquist reasons, see Part M) |
| Preprocessing QC (per-acquisition + 6-group audit) | **COMPLETED** |
| Brainnetome-246 parcellation | **COMPLETED** — 165/165, 246/246 ROI coverage every acquisition |
| ROI time-series extraction | **COMPLETED** — 165/165, all (135, 246) |
| ALFF generation | **COMPLETED** — 165/165, 0 NaN/Inf/negative |
| ReHo generation | **COMPLETED** — 165/165, all within [0,1] |
| FC generation | **COMPLETED** — 165/165, max diagonal/symmetry error 2.2e-16 |
| DC generation | **COMPLETED** — 165/165, independent recompute diff **0.000e+00** |
| Pilot reproduction gate before scale-up | **COMPLETED** — ALFF/ReHo/ROI-TS/atlas bit-identical |
| Per-acquisition cross-file consistency | **COMPLETED** — 165/165 PASS, 0 failures |
| Within-subject normalization (mALFF, mReHo, DC_z, Fisher-z FC) | **COMPLETED** — 165/165, 0 failures |
| Comorbidity data cleaning | **NOT STARTED** — LMCI cohort tables missing (20 subjects) |
| Subject-level cohort index + grouped CV folds | **NOT YET IMPLEMENTED** |
| Cross-subject scaling (train-fold only) | **NOT YET IMPLEMENTED** — deliberately deferred |
| Feature fusion | **NOT YET IMPLEMENTED** |
| Multi-Attention Transformer | **NOT YET IMPLEMENTED** |
| Meta-learning | **NOT YET IMPLEMENTED** |
| Comorbidity fusion | **NOT YET IMPLEMENTED** |
| Bayesian ordinal classifier | **NOT YET IMPLEMENTED** |
| Uncertainty estimation | **NOT YET IMPLEMENTED** |
| Explainability | **NOT YET IMPLEMENTED** |

---

### Summary of the completed data foundation

| | Count |
|---|---:|
| Raw acquisitions audited | 175 |
| Documented exclusions | 2 |
| Eligible for preprocessing | 173 |
| Successfully preprocessed | **165** |
| With complete biomarkers | **165** |
| With complete normalized features | **165** |
| Distinct subjects represented | **127** |
| Processing failures at biomarker or normalization stage | **0** |

Per acquisition, the model-ready feature set is **246 × 3** regional features (mALFF, mReHo, DC_z) plus a **246 × 246** Fisher-z connectivity matrix.

---

*End of document. This document describes the implementation as verified against project code, logs, provenance metadata, and generated outputs on 2026-10-05. Preprocessing, parcellation, biomarker generation and within-subject normalization are complete for all 165 acquisitions; it will require updating as the modelling stages are implemented.*
