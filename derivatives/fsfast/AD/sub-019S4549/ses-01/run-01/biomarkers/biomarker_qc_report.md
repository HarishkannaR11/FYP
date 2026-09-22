# Biomarker QC Report

Subject: sub-019S4549
Session: ses-01
Run: run-01

BOLD volumes: 135
ROIs: 246

## ALFF

- Method: mean amplitude spectrum in 0.01-0.1 Hz (Zang et al. 2007)
- TR: 3.0s (Nyquist=0.1667 Hz)
- 246 ROI values, NaN=0, Inf=0
- Status: PASS

## ReHo

- Neighborhood: 27-voxel (3x3x3 incl. center), Zang et al. 2004 / AFNI 3dReHo default
- 246 ROI values, NaN=0, Inf=0
- Status: PASS

## Degree Centrality

- Correlation method: Pearson
- Definition: weighted, signed, DC_i = sum_j!=i FC[i,j], no threshold
- 246 values, NaN=0, Inf=0
- Status: PASS

## Functional Connectivity

- Method: Pearson correlation, 246x246
- diagonal~=1: True, symmetry_error=1.11e-16
- NaN=0, Inf=0
- Status: PASS

## Regional biomarker matrix

- Shape: 246 x 6
- Status: PASS
