# Diagnostic-Group Hierarchical Clustering — P-Score Profiles

One combined four-panel figure (A mALFF, B mReHo, C DC_z, D FC) showing the
**statistical significance** of each diagnostic group across the 246
Brainnetome ROIs.

| | |
|---|---|
| Figure | `figures/diagnostic_group_hierarchical_pscore_four_biomarkers.png` (300 dpi) |
| | `figures/diagnostic_group_hierarchical_pscore_four_biomarkers.pdf` (vector) |
| Script | `scripts/audit/diagnostic_group_pscore_four_biomarkers.py` |
| Leaf order | `csv/diagnostic_group_hierarchical_pscore_four_biomarkers_leaf_order.csv` |
| Branch | `feature/diagnostic-group-pscore-heatmap` |

---

## 1. P-score definition

$$\text{P-score}_{i,s} = -\log_{10} q_{i,s}$$

for ROI *i* and diagnostic group *s*, where *q* is the **Benjamini–Hochberg
FDR-adjusted** p-value from the stored ROI × stage analysis:

- test: two-sided **Mann–Whitney U**, each stage against the pooled other five
- correction: **BH over 246 × 6 = 1,476 tests per biomarker**
- threshold: **q < 0.05 ⇔ P-score > 1.3010**

A P-score is **not** an unadjusted p-value. It is the negative log of an
already-corrected q-value, so it already carries the multiplicity correction.

Numerical floor 1 × 10⁻¹² before the logarithm; **never engaged** here, since
the smallest q anywhere is 0.1181.

## 2. Source statistical files

| panel | source |
|---|---|
| A mALFF | `../mALFF_roi_stage_qvalues.csv` |
| B mReHo | `../mReHo_roi_stage_qvalues.csv` |
| C DC_z | `../DC_z_roi_stage_qvalues.csv` |
| D FC | `../FC_roi_stage_qvalues.csv` |

Read, never recomputed. Each file was checked before plotting for exactly 246
rows, ROI_IDs 1…246 in order, and column headers matching
`CN, SMC, EMCI, MCI, LMCI, AD`.

**Matrix dimensions: 246 ROIs × 6 diagnostic groups per panel.**

Stage encoding verified from `derivatives/model_ready/metadata.json`:
`CN=0, SMC=1, EMCI=2, MCI=3, LMCI=4, AD=5`.

## 3. FC data source

The FC panel uses genuine ROI-level FC q-values, not an arbitrary scalar
stand-in. Per `csv/fc_transformation_details.csv`:

| property | value |
|---|---|
| source array | `derivatives/model_ready/X_fc.npy` (116, 246, 246) |
| ROI-level summary | `FCstr_i = mean_{j≠i} |z_ij|` |
| signed or absolute | **absolute** (unsigned magnitude) |
| diagonal | zeroed at construction, **excluded** from the mean |
| missing edges | none; the matrix is unthresholded |

So the FC column is the **unsigned mean Fisher-z connectivity strength per
ROI**, tested identically to the three regional biomarkers. It is an ROI-level
summary of the connectivity matrix, **not** the full 246 × 246 pattern — the
246 values compress 30,135 unique edges, and that compression is a real loss
of information, documented here rather than glossed over. The signed row mean
was rejected because it equals DC/245 and would duplicate the DC_z panel.

`FC_roi_stage_qvalues.csv` genuinely supplies 246 × 6 values, so no data had to
be fabricated and no limitation blocks this panel.

## 4. Clustering

Per biomarker independently, reusing the configuration documented in
`DIAGNOSTIC_GROUP_HIERARCHICAL_README.md`:

| | |
|---|---|
| Input | the six groups' **246-D P-score profiles** |
| Distance | **correlation** (1 − Pearson *r*) |
| Linkage | **average** (UPGMA) |
| Leaf ordering | `scipy.cluster.hierarchy.optimal_leaf_ordering`, deterministic |

The clustering input is the P-score profile — **never raw biomarker values**.
Heatmap columns are reordered to match each panel's dendrogram leaves exactly.

### Leaf ordering

| panel | leaf order (left → right) | cophenetic *r* | max P-score |
|---|---|---:|---:|
| A mALFF | MCI, EMCI, LMCI, CN, AD, SMC | 0.757 | 0.652 |
| B mReHo | CN, LMCI, EMCI, AD, SMC, MCI | 0.858 | **0.928** |
| C DC_z | LMCI, AD, CN, SMC, MCI, EMCI | 0.768 | 0.137 |
| D FC | EMCI, AD, SMC, MCI, LMCI, CN | 0.582 | 0.414 |

**Clinical progression order is `CN → SMC → EMCI → MCI → LMCI → AD`** and is
printed on the figure. The dendrogram order differs from it in every panel;
that is expected and was not forced either way.

**No numerical branch labels are drawn.** Branch heights are merge distances,
not P-scores or p-values.

## 5. Colour scale and threshold

Sequential **YlOrRd**, fixed **0.0–1.40**, identical in all four panels so they
are directly comparable. Colourbar labelled
`P-score = -log10(FDR-adjusted q-value)`, with **P-score = 1.301 (q = 0.05)**
marked by a black line.

The diverging red–blue scale of the reference figure is deliberately **not**
reused: P-scores are non-negative with no meaningful midpoint, so a map centred
on zero would imply a sign distinction that does not exist and would
visually exaggerate near-zero values.

## 6. Verified result

| | |
|---|---|
| ROI-stage tests at q < 0.05 | **0 of 5,904** |
| Largest P-score anywhere | **0.9278** (mReHo) |
| Threshold | **1.3010** |
| Missing values | **0** |

All three previously audited figures are confirmed against source. Plotted
values match the source CSVs to **4.8 × 10⁻⁷** (their stored 6-decimal
precision). Nothing was rescaled and no statistic was altered.

## 7. Layout

- Top dendrogram per panel, heatmap immediately below
- **No left-side ROI dendrogram** — the space carries ROI identifiers instead
- All 246 ROI rows retained in every panel; every 10th labelled (ID + name in
  panel A, ID only in B–D for space)
- All six diagnostic groups per panel, bold and colour-coded, aligned to both
  the heatmap columns and the dendrogram leaves

## 8. Limitations of interpreting P-score-profile clustering

- **Nothing is significant.** With no cell passing FDR correction, the surface
  being clustered is dominated by sampling noise; the dendrogram is a display
  ordering, not a result.
- **A tree is not a test.** Branch heights are distances; no p-value can be
  read from them.
- **Similar evidence is not similar biology.** Two groups can cluster together
  because both produced weak evidence in the same regions.
- **Column positions are not comparable across panels** — each is clustered
  independently. The colour scale *is* comparable.
- **FC's tree is the weakest supported** (cophenetic r = 0.582).
- **DC_z appears almost blank** because its strongest evidence anywhere is
  P = 0.137. That is the honest result, not a rendering fault.
- Sample size: 116 subjects, 16–20 per stage, against a 1,476-test family per
  biomarker. Cross-sectional design — **no progression claim is supported.**

## 9. Reproducing

```bash
python scripts/audit/diagnostic_group_pscore_four_biomarkers.py
```

Deterministic. The script **refuses to overwrite** an existing figure at its
output paths. No dataset, feature matrix, statistical output, checkpoint,
manifest or pre-existing figure is modified.
