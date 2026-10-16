# Biomarker QC Report -- SMC_Final/sub-053S5272/ses-01/run-01

BOLD volumes (T): 135  |  TR: 2.999997854232788s  |  brain mask voxels: 28549

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
- Std min/median/max: 1025.24 / 2422.77 / 7849.97

## ALFF

- 246 values, NaN=0, Inf=0, negative=0
- min/median/max: 26980.9355 / 52138.4844 / 128692.6797
- Status: COMPLETED

## ReHo

- Neighborhood: 27-voxel (3x3x3 incl. center)
- 246 values, NaN=0, Inf=0, out of [0,1]=0
- min/median/max: 0.3590 / 0.5041 / 0.7258
- Status: COMPLETED

## Functional Connectivity

- 246x246, Pearson, signed, unthresholded
- NaN=0, Inf=0
- diagonal error=2.22e-16, symmetry error=2.22e-16
- negative off-diagonal: 49.50%
- Status: COMPLETED

## Degree Centrality / FC Strength

- DC = sum_(j!=i) FC[i,j] (weighted, signed, unthresholded)
- DC min/median/max: -18.7288 / 3.6500 / 17.6469
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
