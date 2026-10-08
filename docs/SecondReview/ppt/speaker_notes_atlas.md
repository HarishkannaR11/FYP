# Speaker notes: the 246-region atlas slide and its two backups

For slide 11 ("Brainnetome 246-ROI Parcellation for Feature Extraction") and the two
backup slides 3/8 ("The 246-ROI Extraction Space Is Verified") and 4/8 ("246 Regions as
Connectivity Nodes"). About 45-60 seconds for slide 11. Every number below is reproduced
by `python3 scripts/audit/roi_space_checks.py`.

## Slide 11 - say this (45-60 s)

"The atlas divides the brain into 246 named regions: 210 on the cortex and 36 deeper
inside. For every scan we do the same thing for every region: we take its signal over
time, which gives a table of 135 time points by 246 regions, and from it we compute
ALFF, ReHo and degree centrality, plus the 246-by-246 connectivity table. We checked
that this is complete: all 246 regions are present in all 165 scans, none is empty, and
none is flat. We keep every region on purpose. We do not claim all 246 are equally
related to Alzheimer's disease. We claim the set is complete and reproducible, so
nothing is thrown away before the model has a chance to decide what matters."

The four number tiles: **246** regions (210 cortical + 36 subcortical); **165**
acquisitions; **135 x 246** = volumes kept (140 minus 5 discarded) by regions, per scan;
**246 x 246** = functional connectivity per scan (30,135 unique region pairs).

Wording to keep: the model is a design, so say "will learn" / "will be learned", never
"learns". No region has been shown to be predictive of disease anywhere in this work.

## If the panel asks: "Why 246 regions, why not fewer?"

"Three reasons, and none of them is a claim about disease. First, starting fine is the
safe direction: regions can be merged later, but a coarse region cannot be split. Second,
it is verified: every region is sampled in every scan, so the 246 are a complete,
reproducible extraction space. Third, we are not choosing regions by hand. With 127
people, a hand-picked list would be a guess we would then have to defend. Which regions
and connections matter is for the model to learn, and for validation that keeps each
person's scans together to test. We do not yet know which ones matter, and we do not claim
to."

If asked whether the 246 regions are just copies of each other: on mALFF across all 165
scans, two regions share about **4 %** of their variance on average (mean r-squared), and
**79** principal components are needed for 95 % of the variance. So the regions carry
largely separate information. That says nothing about whether they predict disease.

## If the panel asks: "Why are half of the connectivity values negative? Is that a bug?"

"No. Our cleaning includes global signal regression: the 27 regressors are 24 motion terms,
white matter and the global signal. Removing the whole-brain average signal shifts the
correlations toward zero, so about half become negative: 50.1 % across all 165 scans, and
between 45.6 % and 53.1 % in any single scan. It is a known trade-off of that method, and
negative values after it should not be read as true anti-correlation between networks. We
keep the matrix signed and unthresholded. We have not rerun the pipeline without global
signal regression, so we cannot say how the percentage would change; that would be a
sensitivity check for later."

## Where each number on slide 11 and the backups comes from

| On the slide | Value | Source |
|---|---|---|
| 246 ROIs, 210 cortical + 36 subcortical | IDs 1-210 and 211-246 | `atlases/Brainnetome246/BN_Atlas_246_LUT.txt`, `dataset_brainnetome_summary.csv` |
| 246/246 present, none empty | all 246 "ADEQUATE"; voxels per ROI min 10, median 68, max 186 | `derivatives/fsfast/FINAL_6GROUP_AUDIT/dataset_brainnetome_summary.csv` |
| 165 acquisitions | 165 ROI time-series files and 165 FC files | `derivatives/fsfast/*/sub-*/ses-*/run-*/biomarkers/` |
| 135 x 246 | every time-series file has shape (135, 246) | same folder, `roi_timeseries.npy` |
| 246 x 246, 30,135 edges | every FC file has shape (246, 246); 246*245/2 = 30,135 | same folder, `fc_matrix.npy` |
| no zero-variance, no NaN/Inf | 0 and 0 over all 165 x 246 series | `scripts/audit/roi_space_checks.py` |
| 50.1 % negative | 50.09 % mean over scans; range 45.6-53.1 % | `scripts/audit/roi_space_checks.py` |

The picture on slide 11 is six axial slices read straight from the atlas file
(`figures/make_atlas_axial.py`); each colour is one ROI. The two backup figures are
slide-sized redraws of `fig_roi_qc.png` and `fig_fc_matrix.png`
(`figures/make_atlas_backup_figs.py`). Panels (b) and (c) of the first and the matrix and
histogram of the second are **one representative CN scan** (highest mean tSNR among CN
scans, one flagged artefact scan excluded: sub-010S4442, ses-02, run-01). Everything else
is over all 165 scans.

## Careful with these numbers (if you reuse the other write-up)

- "About 5 % shared variance between regions" is true only for the 69 scans of the 31
  people with repeat scans (the subset used in the granularity figure's caption). Over all
  165 scans it is **4.0 %**. Use 4 % unless you say which subset.
- "Lowest variance 1.98 x 10^5" is the lowest region in the one representative scan, not a
  dataset floor. Over all scans and regions the smallest variance is 4.6; still not zero.
- "50.1 % of edges negative across the cohort" is correct; the figure itself shows one scan
  (50.0 %), so say "across all 165 scans" when you quote 50.1 %.
- Findings 3 and 4, "Why 246 Regions?" and "Residual Motion and Site Effects" are now in the
  deck. Their numbers were re-checked and a few differ from `new_findings_slides.tex`: see
  `speaker_notes_findings.md`.
