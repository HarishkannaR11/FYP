# Biomarker QC Report -- LMCI/sub-130S4817/ses-01/run-01

BOLD volumes (T): 135  |  TR: 3.0s  |  brain mask voxels: 28549

## Brainnetome-246

- Interpolation: nearest neighbor
- Grid match with BOLD: True
- ROIs present: 246 / 246
- Zero-voxel ROIs: 0
- ROIs <5 voxels: 0  |  <10 voxels: 0
- Voxel count min/median/max: 10 / 68.0 / 186

## ROI time series

- Shape: 135 x 246
- NaN: 0  Inf: 0
- Zero-variance ROIs: 0
- Std min/median/max: 1459.15 / 3296.8 / 9382.26

## ALFF

- 246 values, NaN=0, Inf=0, negative=0
- min/median/max: 41044.3711 / 77058.3164 / 182822.5469
- Status: COMPLETED

## ReHo

- Neighborhood: 27-voxel (3x3x3 incl. center)
- 246 values, NaN=0, Inf=0, out of [0,1]=0
- min/median/max: 0.3033 / 0.4571 / 0.7048
- Status: COMPLETED

## Functional Connectivity

- 246x246, Pearson, signed, unthresholded
- NaN=0, Inf=0
- diagonal error=2.22e-16, symmetry error=1.11e-16
- negative off-diagonal: 48.54%
- Status: COMPLETED

## Degree Centrality / FC Strength

- DC = sum_(j!=i) FC[i,j] (weighted, signed, unthresholded)
- DC min/median/max: -25.1552 / 5.8958 / 22.9805
- Independent DC recomputation max abs diff: 0.00e+00
- Independent FC-strength recomputation max abs diff: 0.00e+00
- Status: COMPLETED

## Cross-file consistency

- brainnetome_labels_1_246: True
- roi_ts_shape_ok: True
- alff_len_ok: True
- reho_len_ok: True
- dc_len_ok: True
- fc_strength_len_ok: True
- fc_shape_ok: True
- regional_matrix_shape_ok: True
- roi_ids_match_across_files: True
- no_duplicate_roi_ids: True
- no_missing_roi_ids: True
- no_nan_in_biomarkers: True
- no_inf_in_biomarkers: True
- fc_symmetric: True
- fc_diagonal_is_1: True
- dc_matches_fc_definition: True
- fc_strength_matches_fc_definition: True
- dc_max_abs_diff: 0.0
- fc_strength_max_abs_diff: 0.0

**Overall: COMPLETED**
