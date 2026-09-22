# FYP rs-fMRI Pipeline — Methodology Summary

**Project:** Comorbidity-Aware Meta-Bayesian Multi-Attention Transformer for Discovering Shared Alzheimer's Disease Progression Representations Across Multiple Resting-State fMRI Biomarkers
**Scope of this summary:** condensed version of `FYP_RSFMRI_COMPLETE_METHODOLOGY_AND_IMPLEMENTATION.md`. Only implemented, verified work is described. Where evidence is insufficient, this document says "Not explicitly documented/verified in the current implementation" rather than guessing.
**Date:** 2026-09-22

---

## 1. Dataset

- Source: ADNI resting-state fMRI, raw DICOM + partial NIfTI, reorganized into BIDS (`participants.tsv` is the sole source of group membership).
- **No T1w anatomical image exists for any subject** — this shapes registration, nuisance regression, and CNR throughout the pipeline.
- Six groups, progression order: `CN_Final → SMC_Final → EMCI → LMCI → MCI → AD`.
- 175 raw acquisitions → 2 documented exclusions (1 truncated run, 1 byte-identical duplicate) → 173 eligible → **165 successfully preprocessed** (8 excluded for a genuine TR≈6.02s/Nyquist incompatibility, see §4).

| Group | AD | CN_Final | EMCI | LMCI | MCI | SMC_Final |
|---|---:|---:|---:|---:|---:|---:|
| Eligible acquisitions | 25 | 29 | 30 | 22 | 35 | 32 |

Data recovery: 103 acquisitions initially had JSON metadata but no NIfTI; 29 were recovered from raw DICOM (9 in MCI, 20 in SMC_Final), 0 recovery failures. A prior claim of "6 unrecoverable AD acquisitions" was checked against current source data and found false — all 25 AD acquisitions were located and processed.

## 2. Slice Timing

Exhaustive audit of all 175 JSON sidecars and representative raw DICOM headers across 17 protocol signatures: **no SliceTiming field, no per-slice timestamp, no temporal slice-order tag, and no decodable multiband tag exists anywhere in this dataset.** Conclusion: `SLICE_TIMING_NOT_RECOVERABLE`. No values were inferred or fabricated. Slice-timing correction is **not performed** anywhere in this project.

## 3. Preprocessing Pipeline

```
Raw BOLD → discard first 5 volumes → [slice-timing: NOT PERFORMED]
  → mc-afni2 motion correction (FS-FAST, rigid 6-DOF, ref=frame 0)
  → direct EPI→MNI152NLin6Asym registration (ANTsPy SyN, no T1w)
  → 4mm isotropic resample (continuous interpolation)
  → 6mm FWHM smoothing
  → linear detrend (OLS, mean restored)
  → nuisance regression (26-28 cols: Friston-24 + [WM] + [CSF] + Global Signal)
  → 0.01-0.10 Hz band-pass (Butterworth order-2 + filtfilt, Nyquist-validated per acquisition)
  → final preprocessed BOLD
```

Key facts:
- WM/CSF regressors come from **MNI152NLin2009cAsym population template priors** (WM: p≥0.95 + 1× erosion; CSF: p≥0.90, no erosion), not subject-specific segmentation (impossible without T1w). Each is included only if its mask has ≥20 voxels and `|r with global signal| < 0.98`; excluded and logged otherwise. Design matrix is therefore 26–28 columns, not fixed at 28 (verified example: sub-019S4549/ses-01/run-01 used 27 columns, CSF excluded for that acquisition).
- 8 acquisitions across 5 groups share TR≈6.02s, making Nyquist (≈0.083Hz) incompatible with the fixed 0.10Hz upper cutoff. These were correctly failed rather than silently reprocessed with altered frequencies.
- High-motion acquisitions are **never auto-excluded** — flagged WARN only.

## 4. QC

Computed on the smoothed, pre-nuisance/pre-bandpass stage: tSNR, SNR (native-space, fBIRN-style), FD (Power et al. 2012, from native `.mcdat`), DVARS (**raw-intensity, not standardized — not comparable to literature thresholds**), spatial entropy, temporal entropy. **CNR is `NOT_RELIABLY_COMPUTABLE` for every acquisition** (no T1w). **GCOR is not part of this production pipeline's QC.** Dataset-wide audit (165 acquisitions): 0 NaN/Inf, 0 duplicate outputs, Brainnetome-compatible grid confirmed identical across all six groups.

## 5. Brainnetome-246 Parcellation — Single-Acquisition Pilot

Status: **pilot only** (`sub-019S4549/ses-01/run-01`), not yet dataset-wide.

- Atlas reoriented LAS→RAS (lossless canonical flip) then resampled to the BOLD grid with **nearest-neighbor interpolation only** (categorical labels — never linear).
- Result: **246/246 ROIs covered, 0 zero-voxel ROIs**, min/median/max voxel count 10/68/186.
- ROI time series extracted by direct index-masking (mean per ROI per timepoint): verified shape **(135, 246)** — time × ROI. 0 NaN/Inf, 0 zero-variance ROIs, timepoint count matches source BOLD exactly.

## 6. Biomarkers — Single-Acquisition Pilot

Status: **pilot only**, same acquisition. All computed at voxel level first (where applicable), then ROI-aggregated.

| Biomarker | Method | Key parameters | Result (pilot) |
|---|---|---|---|
| ALFF | Mean FFT amplitude in-band | 0.01–0.10Hz, voxel-wise → ROI mean | 246 values, min 28292.46 / median 65980.05 / max 127465.90 |
| ReHo | Kendall's W (KCC) | **27-voxel (3×3×3) neighborhood**, boundary-adaptive K, voxel-wise → ROI mean | 246 values, bounded [0,1], min 0.3198 / median 0.4685 / max 0.6803 |
| FC | Pearson correlation, ROI×ROI | Full 246×246, signed, unthresholded | diagonal exact 1.0, symmetry error 1.11e-16 |
| DC | `Σ_{j≠i} FC[i,j]` | Weighted, signed, no threshold, self-excluded | min −16.84 / median 2.83 / max 12.94 |

The full 246×246 FC matrix is kept as a primary output; a separate derived **mean**-based `FC_Strength` per ROI is stored alongside it for the combined 246×4 regional biomarker matrix — the two are not interchangeable and both are retained.

## 7. Independent Validation

A separate script, reading only saved files (no reuse of generation-time memory), recomputed DC and FC-strength from the saved FC matrix from scratch:

- DC: max difference vs. saved = **0.00e+00** (exact)
- FC-strength: max difference vs. saved = **9.89e-17** (exact, floating-point noise only)
- All shapes, NaN/Inf, symmetry, and cross-file ROI-ID consistency checks: **PASS**

This is a mathematical/structural validation only; it makes no claim about biological/diagnostic meaningfulness.

## 8. Reproducibility

Python 3.12.3, NumPy 2.3.5, SciPy 1.15.3, NiBabel 5.4.2, Nilearn 0.14.1, ANTsPy 0.6.3, TemplateFlow 25.1.2, FreeSurfer 7.4.1, WSL2/Ubuntu. Single-worker throughout (`n_procs=1`, all thread env vars pinned to 1). Checkpoint/resume via `PROCESSING_CHECKPOINT.json`, validated against actual on-disk files (not trusted blindly) before skipping any acquisition. Every final BOLD has a paired provenance JSON (parameters, versions, timestamps, checksums).

## 9. Key Limitations

- No T1w anywhere → direct EPI-MNI registration, template-prior (not subject-specific) WM/CSF, CNR unavailable.
- Slice-timing genuinely unrecoverable, not merely skipped by choice.
- Raw-intensity DVARS is not comparable to standardized-DVARS literature thresholds.
- Biomarker pipeline validated on **one acquisition only** — not yet proven to run cleanly at full-dataset scale.
- 8/173 eligible acquisitions permanently excluded from current derivatives (TR/Nyquist incompatibility).

## 10. Status

| Stage | Status |
|---|---|
| Dataset audit, recovery, exclusions | Completed |
| Slice-timing investigation | Completed |
| Preprocessing, all 6 groups | Completed (165/173) |
| Preprocessing QC | Completed |
| Brainnetome parcellation | Pilot completed (1 acquisition) |
| ALFF / ReHo / FC / DC | Pilot completed (1 acquisition) |
| Independent biomarker validation | Completed (pilot) |
| Full-dataset biomarker generation | Not yet implemented |
| Normalization / fusion | Not yet implemented |
| Multi-Attention Transformer | Not yet implemented |
| Meta-learning | Not yet implemented |
| Comorbidity fusion | Not yet implemented |
| Bayesian ordinal classifier | Not yet implemented |
| Uncertainty estimation / explainability | Not yet implemented |

Full detail, per-part rationale, and complete audit trail: see `FYP_RSFMRI_COMPLETE_METHODOLOGY_AND_IMPLEMENTATION.md`.
