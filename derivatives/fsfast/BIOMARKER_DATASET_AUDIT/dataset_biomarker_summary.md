# Dataset Biomarker Audit

Generated (UTC): 2026-09-23T10:58:05.000592+00:00

## Per-group completion

| Group | Expected | Completed | Failed | Skipped (ineligible) | Missing outputs |
|---|---:|---:|---:|---:|---:|
| AD | 25 | 25 | 0 | 0 | 0 |
| CN_Final | 27 | 27 | 0 | 2 | 0 |
| EMCI | 29 | 29 | 0 | 1 | 0 |
| LMCI | 21 | 21 | 0 | 1 | 0 |
| MCI | 32 | 32 | 0 | 3 | 0 |
| SMC_Final | 31 | 31 | 0 | 1 | 0 |
| **TOTAL** | **165** | **165** | **0** | **8** | **0** |

## Dataset QC (over acquisitions with complete outputs)

- Acquisitions audited: 165
- Brainnetome coverage 246/246: 165 / 165
- Brainnetome zero-voxel ROIs (any): 0
- ROI voxel count min across dataset: 10
- T (volumes) range: 135 - 135
- ROI matrix columns == 246 for all: True
- ROI time-series NaN total: 0
- ROI time-series Inf total: 0
- Acquisitions with zero-variance ROIs: 0
- ALFF: all 246 values present: True, NaN total=0, negative total=0
- ALFF range across dataset: 34.72 - 913706.06
- ReHo: all 246 values present: True, NaN total=0, out-of-[0,1] total=0
- ReHo range across dataset: 0.2352 - 0.8231
- FC shape 246x246 for all: True, NaN total=0
- FC max diagonal error: 2.220e-16
- FC max symmetry error: 2.220e-16
- FC negative off-diagonal %: mean=50.09%, range 45.55-53.12%
- DC: all 246 present: True, NaN total=0
- DC range across dataset: -41.7756 - 40.0963
- FC strength: all 246 present: True, NaN total=0
- Regional matrix 246x4 for all: True, NaN total=0
- Max DC recomputation difference: 0.000e+00
- Max FC-strength recomputation difference: 6.939e-18
- Cross-file consistent: 165 / 165

## Unusual acquisitions

None. Every audited acquisition passed all structural, numerical and cross-file consistency checks.

## Processing errors

No processing errors.
