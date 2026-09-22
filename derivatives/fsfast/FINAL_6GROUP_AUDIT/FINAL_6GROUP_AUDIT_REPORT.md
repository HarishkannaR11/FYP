# Final 6-Group Preprocessing Audit

Read-only audit of the completed 4mm/6mm production run across all six diagnostic groups. No preprocessing outputs were modified; BIDS/raw source untouched (verified via per-acquisition checksum_match, see `dataset_processing_status.csv`).

## Pipeline

```
Discard first 5 volumes -> Slice timing not performed -> mc-afni2 motion correction ->
direct EPI->MNI MNI152NLin6Asym -> 4mm isotropic resampling -> 6mm FWHM smoothing ->
linear detrending -> Friston-24 + WM + CSF + GS (28 regressors) -> 0.01-0.10Hz band-pass -> QC
```

## Group Summary

| Group | Expected | Processed | Failed | Excluded |
|---|---:|---:|---:|---:|
| AD | 25 | 25 | 0 | 0 |
| CN_Final | 29 | 27 | 2 | 1 |
| EMCI | 30 | 29 | 1 | 0 |
| LMCI | 22 | 21 | 1 | 1 |
| MCI | 35 | 32 | 3 | 0 |
| SMC_Final | 32 | 31 | 1 | 0 |

**Total: 173 expected, 165 processed, 8 failed, 2 pre-excluded.**

## Final QC Table (measured from completed acquisitions only, not copied from prior reports)

| Group | tSNR | SNR | CNR | FD | DVARS | Spatial Entropy | Temporal Entropy | Brainnetome |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| AD | 114.0588 | 13.4887 | N/A | 0.4394 | 18145.5833 | 7.2238 | 3.5337 | 246/246 |
| CN_Final | 12064.3619 | 13.0265 | N/A | 0.3793 | 14451.8613 | 7.0902 | 3.5469 | 246/246 |
| EMCI | 137.4854 | 13.7615 | N/A | 0.3546 | 16407.1233 | 7.1455 | 3.5292 | 246/246 |
| LMCI | 127.537 | 12.9948 | N/A | 0.3224 | 13564.45 | 7.182 | 3.5122 | 246/246 |
| MCI | 137.0515 | 12.9356 | N/A | 0.3207 | 16510.2114 | 7.1716 | 3.5629 | 246/246 |
| SMC_Final | 125.7105 | 13.1538 | N/A | 0.3755 | 13799.8246 | 7.1531 | 3.5677 | 246/246 |

No composite/weighted quality score was created; each metric stands on its own.

## Root Cause of All 8 Failures

Every single failure across all six groups shares **one identical root cause**: these acquisitions were scanned with **TR ≈ 6.02s** (a real, pre-existing protocol variant in this dataset) instead of the standard TR ≈ 3.0s. At TR=6.02s, the Nyquist frequency is ≈0.083 Hz, which is *below* the frozen band-pass upper cutoff of 0.10 Hz -- making the 0.01-0.10 Hz filter mathematically invalid for these acquisitions. Per the frozen protocol ("do not change these frequencies... if the TR makes the requested filter invalid, STOP and report the issue instead of silently changing the frequencies"), the pipeline correctly refused to process these acquisitions rather than silently altering the filter band. This is a genuine dataset characteristic, not a pipeline defect -- see `dataset_failures.csv` for every instance.

## Known Pre-Exclusions

- **CN_Final sub-012S4026/ses-01/run-01**: 7-volume truncated acquisition (pre-excluded from manifest)
- **LMCI sub-006S4363/ses-01/run-02**: byte-identical duplicate of ses-01/run-01 (pre-excluded from manifest)

## Data Integrity

- 165/165 completed acquisitions: 4D, 4mm isotropic, valid affine, valid binary 3D mask.
- See `dataset_data_integrity.csv` for the full per-acquisition table.

## Brainnetome-246 Compatibility

Computed once (the final grid is identical across every acquisition in every group, verified by direct affine comparison): **246/246 ROIs present**, 0 zero-voxel, 0 with <5 voxels, 0 with <10 voxels (min=10, median=68.0, max=186).

## Duplicates

Duplicate acquisition directories found: **0**

## Source Data Integrity

- 165/165 acquisitions: source BOLD checksum verified unchanged before vs. after processing (per-acquisition, recorded during production).
- BIDS/, raw_data/, and the Brainnetome atlas were never written to by this pipeline.

## Processing Constraints Honored

- Single worker throughout (`n_procs=1`, no multiprocessing/threading anywhere).
- One acquisition processed at a time; scratch directory cleared after each.
- Resume logic re-validated every acquisition's actual output files on disk before skipping (not just checkpoint presence) -- no completed acquisition was reprocessed.
- Group QC/summary files were written only after each group's full manifest was exhausted.
