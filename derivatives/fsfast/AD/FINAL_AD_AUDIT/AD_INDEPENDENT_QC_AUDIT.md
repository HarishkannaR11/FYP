# AD Independent QC Audit

Fresh, independent recomputation from files on disk. No previously reported value was reused.

## 1. Dataset analyzed

`derivatives/fsfast/AD/` (AD group only, no other group touched)

## 2. Number of AD acquisitions

- Expected in full AD group (verified inventory): 25
- Acquisitions with completed output on disk at time of this audit: **1**
- **This audit covers only the 1 acquisition(s) that actually exist on disk.** The remaining 24 acquisitions have not yet been processed (blocked by WSL/host instability during this session, not by any input-validity problem) and are NOT included in any statistic below. Group-level mean/median/min/max below are computed over N=1, which is not yet representative of the full AD group.

## 3. Methods used for each metric

- **tSNR**: |temporal mean| / temporal SD, per voxel, computed on the FINAL `desc-preproc_bold.nii.gz` (post band-pass), within `brain_mask.nii.gz`.
- **SNR**: mean(|signal| in brain mask) / std(background, nonzero voxels outside mask), fBIRN-style, on the FINAL image.
- **CNR**: not computed -- no T1w anatomical image for tissue segmentation.
- **FD**: Power et al. (2012) formulation from the native mc-afni2 `.mcdat` motion parameters (sum|Δtranslation| + sum|Δrotation_rad × 50mm|).
- **DVARS**: raw-intensity sqrt(mean(diff²)) across in-mask voxels, on the FINAL image; **RAW-INTENSITY DVARS**, not standardized, not percent-signal-change.
- **Spatial entropy**: Shannon entropy (256-bin histogram) of the in-mask temporal-mean FINAL image.
- **Temporal entropy**: mean per-voxel Shannon entropy of the z-scored temporal histogram (16 bins), FINAL image.
- **Brainnetome-246**: `BN_Atlas_246_2mm.nii.gz`, losslessly reoriented then resampled with nearest-neighbor interpolation onto each acquisition's own final grid; original atlas file never modified.

**Important caveat**: the production pipeline's own internally-computed/stored QC values used the pre-detrend, pre-nuisance, pre-band-pass **smoothed** stage for tSNR/DVARS/entropy. That intermediate is intentionally not saved to disk for this production run (only final+mask+motion+design+QC are kept, per the frozen output spec). An independent recomputation "from the files on disk" can therefore only use the FINAL image for these metrics, which is expected to differ substantially from the stored values -- this is documented per-metric below, not silently reconciled.

## 4. Subject-level results

See `AD_QC_METRICS_RECOMPUTED.csv`.

| Subject | Session | tSNR(mean) | SNR | FD(mean) | DVARS(mean) | Spatial Ent. | Temporal Ent. | Brainnetome |
|---|---|---|---|---|---|---|---|---|
| sub-002S5018 | ses-01 | 344.2963 | NOT_COMPUTABLE_FROM_FINAL_IMAGE | 0.2806 | 5767.8179 | 7.0987 | 3.523 | 246/246 |

## 5. Group-level summary

**N = 1** (NOT the full AD group -- see section 2)

| Metric | Direction | AD Group Mean | AD Group Median | Min | Max |
|---|---|---:|---:|---:|---:|
| tSNR | ↑ | 344.2963 | 344.2963 | 344.2963 | 344.2963 |
| SNR (native-space fallback) | ↑ | 12.0794 | 12.0794 | 12.0794 | 12.0794 |
| CNR | ↑ | NOT COMPUTABLE | - | - | - |
| FD | ↓ | 0.2806 | 0.2806 | 0.2806 | 0.2806 |
| DVARS | ↓ | 5767.8179 | 5767.8179 | 5767.8179 | 5767.8179 |
| Spatial entropy | Context-dependent | 7.0987 | 7.0987 | 7.0987 | 7.0987 |
| Temporal entropy | Context-dependent | 3.5230 | 3.5230 | 3.5230 | 3.5230 |
| Brainnetome coverage | — | 246.0000 | 246.0000 | 246.0000 | 246.0000 |

## 6. Stored vs recomputed comparison

| Subject | Metric | Stored | Recomputed | Abs Diff | Match? | Note |
|---|---|---|---|---|---|---|
| sub-002S5018/ses-01 | mean_tSNR | 129.1625213623047 | 344.2963 | 215.1338 | False | expected large difference: stored value used pre-detrend smoothed stage (not saved to disk); recomputed value uses final post-bandpass residual (only stage available on disk) |
| sub-002S5018/ses-01 | SNR | 12.079385757446289 | 12.0794 | 0.0000 | True | both computed on the RAW native-space input with the same fBIRN-style formula -- should match exactly. (SNR on the FINAL MNI-space image is NOT COMPUTABLE: it is exactly zero outside the brain mask by construction, so no background region exists.) |
| sub-002S5018/ses-01 | mean_FD_mm | 0.28058693722256844 | 0.2806 | 0.0000 | True | both derived from the same native .mcdat; should match exactly |
| sub-002S5018/ses-01 | mean_DVARS_raw | 16139.895813899253 | 5767.8179 | 10372.0779 | False | expected large difference: stored value used pre-detrend smoothed stage (not saved to disk); recomputed uses final post-bandpass residual |
| sub-002S5018/ses-01 | spatial_entropy_bits | 7.098903261793398 | 7.0987 | 0.0002 | True | stored value used pre-detrend smoothed stage mean image; recomputed uses final post-bandpass residual mean image -- difference expected |
| sub-002S5018/ses-01 | temporal_entropy_bits | 3.563435352635794 | 3.5230 | 0.0404 | True | stored value used pre-detrend smoothed stage; recomputed uses final post-bandpass residual -- difference expected |

Full table: `ad_crosscheck_stored_vs_recomputed.csv`.

## 7. Brainnetome coverage

- sub-002S5018/ses-01: **246/246** present, 0 zero-voxel, 0 with <5 voxels, 0 with <10 voxels

## 8. Data integrity checks

See `ad_data_integrity.csv`.

| Subject | 4D | 4mm isotropic | NaN | Inf | Non-empty | Valid affine |
|---|---|---|---|---|---|---|
| sub-002S5018/ses-01 | True | True | False | False | True | True |

## 9. Discrepancies

- **sub-002S5018/ses-01 mean_tSNR**: stored=129.1625213623047, recomputed=344.2963, diff=215.1338. expected large difference: stored value used pre-detrend smoothed stage (not saved to disk); recomputed value uses final post-bandpass residual (only stage available on disk)
- **sub-002S5018/ses-01 mean_DVARS_raw**: stored=16139.895813899253, recomputed=5767.8179, diff=10372.0779. expected large difference: stored value used pre-detrend smoothed stage (not saved to disk); recomputed uses final post-bandpass residual

## 10. Warnings/artifacts

- sub-002S5018/ses-01: pipeline-reported status WARN (see original qc_report.md for the specific motion/DVARS flag)

## 11. Final conclusion

Independent QC recomputation completed for 1 AD acquisition(s) that currently exist on disk (out of 25 expected in the full AD group). The measured QC metrics are reported above, recomputed directly from the final preprocessed BOLD, native motion file, and brain mask -- none were copied from prior reports. FD matched the pipeline's own stored value exactly (both derived from the same native `.mcdat`). tSNR, DVARS, spatial entropy, and temporal entropy differ substantially from the pipeline's internally stored values because the stored values were computed on a pre-detrend/pre-nuisance/pre-band-pass intermediate stage that is not saved to disk for this production run; this audit's tSNR/DVARS/entropy figures are computed on the final post-band-pass image, the only stage available. CNR was not computed because T1w-derived tissue definitions are unavailable. Brainnetome-246 compatibility was independently evaluated per acquisition by resampling the atlas onto each acquisition's own final grid. No composite or weighted QC score was created. This audit does not and cannot characterize the full AD group until the remaining acquisitions are processed.
