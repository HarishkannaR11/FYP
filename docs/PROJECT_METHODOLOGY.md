# Project Methodology

Research-paper style description of the methods. **Each section states
whether the component is implemented or proposed.** Nothing is described as
done unless source code and execution evidence exist.

---

## 1. Data acquisition and subject selection — IMPLEMENTED

Resting-state fMRI from the Alzheimer's Disease Neuroimaging Initiative
(ADNI), obtained via the LONI Image & Data Archive under ADNI's Data Use
Agreement. Six diagnostic groups spanning the progression
**CN → SMC → EMCI → MCI → LMCI → AD**.

**No T1-weighted structural scan exists for any subject in this download.**
This single fact constrains the entire preprocessing design (§5).

175 raw acquisitions were audited. After 2 pre-exclusions and 8 preprocessing
failures, 165 were processed; 2 further subjects were added later, giving
**167 preprocessed acquisitions from 129 subjects**.

The analysis cohort is **116 subjects, one acquisition each** — see
[DATASET_AND_COHORT.md](DATASET_AND_COHORT.md).

---

## 2. Dataset audit — IMPLEMENTED

`scripts/data_prep/stage_dicom_inventory.py`,
`scripts/audit/final_6group_audit.py`

The ADNI image search returns `Modality = fMRI` records that are **not all
resting-state BOLD**. Descriptions are therefore classified into three tiers
before anything is converted:

| Tier | Descriptions | Action |
|---|---|---|
| TIER1 | `Resting State fMRI`, `Extended Resting State fMRI`, `Axial rsfMRI/fcMRI (Eyes Open)` | convert |
| REVIEW | unrecognised variants | flag, decide manually |
| EXCLUDE | `MoCoSeries`, `Perfusion_Weighted`, `relCBF` | never convert |

`MoCoSeries` is the scanner's own motion-corrected copy of the same EPI run.
Using it would double-correct the data and, critically, would leave no motion
parameters from which to build the Friston-24 regressors — the design would
collapse from 27 regressors to 3. `Perfusion_Weighted` and `relCBF` are ASL
derivatives, not BOLD.

Duplicate detection uses **(subject, acquisition date)**, not filename and
not file hash. Different filenames do not prove different acquisitions, and
different hashes do not either.

Outputs: `derivatives/fsfast/FINAL_6GROUP_AUDIT/` — inventory, processing
status, QC summary, data integrity, motion summary, Brainnetome coverage,
failures, exclusions, duplicates.

---

## 3. DICOM → NIfTI conversion — IMPLEMENTED

`scripts/data_prep/convert_extra_dicom.py`, **dcm2niix v1.0.20260724**

```
dcm2niix -z y -b y -f "%n_%d_%t_%s" -o <out> <series_dir>
```

One series at a time: extract from the archive → convert → validate → delete
staging. Peak disk ≈ 150 MB instead of the full uncompressed archive.
Original ZIPs are never modified.

Validation per output: readable by NiBabel; 4-D with time dimension ≥ 2;
voxel sizes finite and within 0.5–8 mm; affine finite and non-singular; data
finite and non-constant; JSON sidecar parses and contains `RepetitionTime`;
**volume count matches DICOM count** (e.g. 9,600 / 48 slices = 200 volumes);
dcm2niix warnings captured verbatim.

**No metadata is fabricated.** Where `SliceTiming` is absent it stays absent.

Resumable: each result is appended to a JSONL state file, and on restart an
acquisition is skipped only if its outputs are still present on disk.

---

## 4. BIDS organisation — IMPLEMENTED

`scripts/data_prep/create_bids.py`,
`scripts/data_prep/bidsify_new_acquisitions.py`

```
BIDS/<sub>/<ses>/func/<sub>_<ses>_task-rest_<run>_bold.nii[.gz]  (+ .json)
```

ADNI `NNN_S_NNNN` maps to `sub-NNNSNNNN`. Files are **copied, never moved**;
existing destinations are never overwritten; `participants.tsv` is appended
to only for new subjects; a source→destination mapping is written to
`audit/bids_additions_mapping.csv`.

The preprocessing pipeline reads **only** from `BIDS/`, so an acquisition is
invisible to it until BIDS-ified.

---

## 5. Preprocessing — IMPLEMENTED

`pipeline/fsfast_production_4mm6mm.py`. Full detail in
[PREPROCESSING_PIPELINE.md](PREPROCESSING_PIPELINE.md).

| Step | Implementation | Parameters |
|---|---|---|
| Volume discard | first N dropped | **N = 5** |
| Slice timing | **not performed** | §5.1 |
| Motion correction | FS-FAST `mc-afni2` | `.mcdat` retained |
| Registration | ANTsPy `SyN`, **direct EPI→MNI** | `MNI152NLin6Asym` res-02 |
| Resampling | nilearn, continuous | **4 mm isotropic** |
| Smoothing | `nilearn.image.smooth_img` | **6 mm FWHM** |
| Detrending | OLS intercept + linear | mean restored |
| Nuisance regression | masked GLM | **27 regressors** |
| Band-pass | Butterworth + `filtfilt` | **0.01–0.10 Hz** |
| QC | per acquisition | tSNR, SNR, FD, DVARS, GCOR, entropy |

**Input** `BIDS/.../*_bold.nii[.gz]`, typically 64×64×48×140 at ~3.31 mm.
**Output** `derivatives/fsfast/<GROUP>/<sub>/<ses>/<run>/*_desc-preproc_bold.nii.gz`,
46×55×46×(T−5) at 4 mm in MNI space, plus brain mask, design matrix, motion
parameters, QC and full provenance.

### 5.1 Slice-timing policy

**No slice-timing correction is applied to any acquisition.** Verified by
source inspection: the pipeline contains no `slicetimer`, `3dTshift` or
equivalent; it proceeds directly from volume discard to motion correction.

Rationale: ADNI's metadata does not reliably recover slice order for this
dataset, and nothing is inferred.

**Documented exception in labelling only.** `sub-007S2394` — the only GE
acquisition — has a sidecar containing real `SliceTiming`, and the pipeline's
flag is computed as *"SliceTiming present in metadata"*, so its provenance
reads `PERFORMED`. **No correction was applied to it.** The underlying
processing is identical to the other 166. The field is a metadata-presence
indicator, not a processing record.

### 5.2 Nuisance regression

27 regressors: intercept + Friston-24 (6 motion parameters, their temporal
derivatives, and the squares of both) + white-matter signal + global signal.

The design was specified as 28 including CSF, but **the CSF regressor failed
its reliability gate in every one of the 167 acquisitions**, so all carry
exactly 27 — uniformity verified in the provenance. WM and CSF priors are
template-level, a direct consequence of having no T1w for subject-specific
segmentation.

### 5.3 TR / Nyquist validation

Before filtering, the pipeline checks `1/(2·TR) > 0.10 Hz`. Eight
acquisitions with TR ≈ 6.02 s give a Nyquist of 0.083 Hz and were **refused**
rather than silently filtered with a narrowed band. All eight are from site
002 and are recorded with their measured TR in `dataset_failures.csv`.

---

## 6. Parcellation — IMPLEMENTED

Brainnetome-246 atlas (Fan et al., 2016): 210 cortical + 36 subcortical
regions. The atlas is reoriented LAS→RAS losslessly, then resampled onto each
acquisition's own BOLD grid with **nearest-neighbour interpolation only** —
continuous interpolation would invent fractional labels.

Verified: **246/246 ROIs present in every acquisition**, 0 zero-voxel, 0
below 10 voxels, voxel counts 10/68/186 (min/median/max). A region's time
series is the mean of its voxels, giving **T × 246**.

---

## 7. Biomarker extraction — IMPLEMENTED

`scripts/biomarkers/biomarker_dataset_pipeline.py`. Definitions and equations
in [FEATURES_AND_STATISTICAL_ANALYSIS.md](FEATURES_AND_STATISTICAL_ANALYSIS.md) §2.

| Biomarker | Output | Definition |
|---|---|---|
| ALFF | 246 | mean FFT amplitude in 0.01–0.10 Hz |
| ReHo | 246 | Kendall's *W*, 27-voxel neighbourhood |
| FC | 246×246 | Pearson *r*, signed, unthresholded |
| DC | 246 | $\sum_{j\neq i}\mathrm{FC}_{ij}$ |

**FC_Strength is excluded.** It equals DC/245 exactly — maximum absolute
difference 6.939 × 10⁻¹⁸, r = 1.000000000000 — and carries no independent
information. The raw files are kept for traceability but are never a model
feature.

---

## 8. Within-acquisition normalisation — IMPLEMENTED

`scripts/biomarkers/normalize_within_subject.py`

| Feature | Transform |
|---|---|
| mALFF | ALFF / mean(ALFF) — ratio, because ALFF is strictly positive |
| mReHo | ReHo / mean(ReHo) |
| DC_z | (DC − mean) / std, ddof = 0 — z-score, because DC is signed |
| FC | artanh(clip(r, ±0.999999)), **diagonal zeroed first** |

Every statistic comes from **one acquisition's own 246 values** — no
cross-subject statistic, no label. Leakage-free by construction.

Verified: mean(mALFF) = mean(mReHo) = 1.000000, std(DC_z) = 1.000000, 0
entries required clipping, Fisher-z diagonal exactly 0, 0 NaN/Inf.

---

## 9. Clinical and comorbidity integration — IMPLEMENTED (with gaps)

`scripts/clinical/clean_clinical_data.py`

ADNI **image** exports carry no clinical scores. Assessments come from
separate tables (CDR, MMSE, GDSCALE, MEDHIST, APOERES). For each subject the
MRI date is established, then for each assessment the **visit nearest in time
to the MRI** is selected, recording the chosen visit, day offset and match
method.

Medical history is decoded 1 → present, 0 → absent, sentinel → NaN. APOE ε4
alleles are counted from the genotype string. **No imputation.** Conflicting
duplicates are flagged, not resolved.

**Coverage**, within the 116-subject cohort: MMSE and CDR global **116/116**,
GDS 115, APOE 115, CDR sum-of-boxes and the 19 medical-history flags 114, all
of it together **113/116**. The earlier figure of 94/116 with LMCI at 0/20
reflected an ADNI table that had not been downloaded; it has since been
supplied. `derivatives/clinical_cleaned/` predates that and is stale — use
`derivatives/comorbidity_features/model_clinical_116.csv`.

**What is material is circularity, not coverage.** MMSE and CDR are inputs to
the ADNI criteria that assign the stage being predicted, so they cannot serve
as model features. See [DATASET_AND_COHORT.md](DATASET_AND_COHORT.md) §4.

---

## 10. Disease-stage mapping — IMPLEMENTED

```
CN = 0   SMC = 1   EMCI = 2   MCI = 3   LMCI = 4   AD = 5
```

Encoded in `derivatives/model_ready/y.npy`. Directory names `CN_Final` and
`SMC_Final` map to `CN` and `SMC`; the originals are preserved in the source
inventory. Label-to-index agreement was verified by counting rows both ways.

---

## 11. Cohort selection — IMPLEMENTED

`scripts/data_prep/build_phase1_dataset.py`

Deterministic and label-blind: consider only acquisitions passing all
validity checks; sort by `(subject, session, run)` with a stable sort; take
the first acquisition of each distinct subject until 20 are held; if fewer
than 20 subjects exist, **stop and report the shortfall**.

Result: **116 subjects, 116 acquisitions, one per subject**. MCI contributes
16 — the shortfall is recorded, not padded.

---

## 12. Statistical analysis — IMPLEMENTED

Three analyses, each with its own declared hypothesis family. Full
specification in [FEATURES_AND_STATISTICAL_ANALYSIS.md](FEATURES_AND_STATISTICAL_ANALYSIS.md) §5.

| Analysis | Design | Family | Test |
|---|---|---|---|
| Pairwise | 246 ROI × 15 comparisons | 3,690 per biomarker | Mann-Whitney *U* |
| ROI × stage | 246 ROI × 6 one-vs-rest | 1,476 per biomarker | Mann-Whitney *U* |
| Stage-level | 4 biomarkers × 6 stages | 24 total | permutation pseudo-F |

Effect sizes: Hedges' *g* (with rank-biserial stored alongside, as the effect
size matching the rank test). Correction: Benjamini–Hochberg throughout.
Permutations: 9,999, seed `20261013`, permuted **across subjects**.

**Results: nothing reaches q < 0.05 in any analysis.** See
[RESULTS_AND_INTERPRETATION.md](RESULTS_AND_INTERPRETATION.md).

---

## 13. Multi-Attention Transformer — ARCHITECTURE IMPLEMENTED, NOT TRAINED

`scripts/model/multi_attention_transformer.py`,
`scripts/model/attention_modules.py`

Three parallel attention branches over 246 ROI tokens — **global**
(unrestricted), **local** (restricted to a caller-supplied neighbourhood
mask), **network** (biased by measured FC, with separate learnable scales for
positive and negative connectivity) — combined by a per-ROI softmax gate, then
a feed-forward network, with pre-norm residuals.

**24 unit tests pass.** Smoke test on real cohort features: 196,374
parameters, correct output shapes, all finite.

**No training has occurred and no weights exist.** The `local_mask` input is
**not provided** by this repository and is deliberately not invented.

Full detail in [MODEL_ARCHITECTURE.md](MODEL_ARCHITECTURE.md).

---

## 14. Feature fusion, meta-learning, Bayesian head — PROPOSED

**No code exists for any of the following.**

- **Comorbidity fusion**: $H = [Z' \Vert C]$, concatenating the brain
  embedding with clinical covariates. Blocked by LMCI having no clinical
  data.
- **Meta-learning**: each biomarker as a task, MAML-style inner/outer loops,
  support and query sets subject-disjoint within training folds. Empirically
  motivated by the biomarkers sharing only 5.7–11.3% of their variance.
- **Bayesian ordinal head**: cumulative-link model,
  $P(y \le k) = \sigma(\tau_k - g_w(H))$, with a distribution over weights.
- **Uncertainty**: predictive entropy over weight samples.

---

## 15. Training, validation and evaluation — PROPOSED

**Nothing in this section has been implemented or run. No metric exists.**

Planned: subject-grouped cross-validation (trivially satisfied, one
acquisition per subject); any cross-subject scaler fitted on **training folds
only**; mean FD as a covariate, since 129/246 ROIs still correlate with it
after nuisance regression; scanning site considered given its concentration;
an ordinal loss consistent with the cumulative-link head; evaluation by mean
absolute error in stage units, quadratic-weighted kappa, per-class recall and
calibration; comparison against simple baselines **before** any claim of
benefit.

With ~19 subjects per class and ~200k parameters at the smallest tested
configuration, overfitting is the dominant risk — see
[LIMITATIONS_AND_FUTURE_WORK.md](LIMITATIONS_AND_FUTURE_WORK.md).

---

## References

- Fan, L. et al. (2016). The Human Brainnetome Atlas. *Cerebral Cortex*, 26(8), 3508–3526.
- Jack, C. R. Jr. et al. (2008). The Alzheimer's Disease Neuroimaging Initiative (ADNI): MRI methods. *Journal of Magnetic Resonance Imaging*, 27(4), 685–691.
- Petersen, R. C. et al. (2010). Alzheimer's Disease Neuroimaging Initiative (ADNI): Clinical characterization. *Neurology*, 74(3), 201–209.
- Zang, Y. et al. (2004). Regional homogeneity approach to fMRI data analysis. *NeuroImage*, 22(1), 394–400.
- Zang, Y.-F. et al. (2007). Altered baseline brain activity in children with ADHD. *Brain and Development*, 29(2), 83–91.
- Zou, Q.-H. et al. (2008). An improved approach to detection of ALFF: fractional ALFF. *Journal of Neuroscience Methods*, 172(1), 137–141.
- Zuo, X.-N. et al. (2012). Network centrality in the human functional connectome. *Cerebral Cortex*, 22(8), 1862–1875.
- Avants, B. B. et al. (2008). Symmetric diffeomorphic image registration with cross-correlation. *Medical Image Analysis*, 12(1), 26–41.
- Friston, K. J. et al. (1996). Movement-related effects in fMRI time-series. *Magnetic Resonance in Medicine*, 35(3), 346–355.
- Power, J. D. et al. (2012). Spurious but systematic correlations in functional connectivity MRI networks arise from subject motion. *NeuroImage*, 59(3), 2142–2154.
- Benjamini, Y. & Hochberg, Y. (1995). Controlling the False Discovery Rate. *JRSS-B*, 57(1), 289–300.
- Anderson, M. J. (2001). A new method for non-parametric multivariate analysis of variance. *Austral Ecology*, 26(1), 32–46.
- Vaswani, A. et al. (2017). Attention is all you need. *NeurIPS* 30.
- Finn, C., Abbeel, P. & Levine, S. (2017). Model-Agnostic Meta-Learning. *ICML*.
- Gal, Y. & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation. *ICML*.
- McCullagh, P. (1980). Regression models for ordinal data. *JRSS-B*, 42(2), 109–142.
- Cox, R. W. (1996). AFNI: Software for analysis and visualization of functional magnetic resonance neuroimages. *Computers and Biomedical Research*, 29(3), 162–173.
- Fischl, B. (2012). FreeSurfer. *NeuroImage*, 62(2), 774–781.
