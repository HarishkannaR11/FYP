# Z-Score Clustering and Significance Heatmaps

Generated for the 116-subject analysis cohort of the project *Comorbidity-Aware
Meta-Bayesian Multi-Attention Transformer for Alzheimer's Disease Progression
Prediction Using Resting-State fMRI Biomarkers*.

Verified by `scripts/audit/validate_zscore_pscore_outputs.py` — **108/108 checks
pass**.

---

## The one distinction that matters

| | Figure A | Figures B and C |
|---|---|---|
| **Kind** | **descriptive** | **inferential** |
| Shows | where each subject sits relative to the cohort | whether a difference survives multiple-comparison correction |
| Involves a hypothesis test | **no** | yes |
| A red cell means | this subject is above the cohort mean for this ROI | a smaller q-value — **none reaches the threshold** |

A Z-score is a position, not a test. **A dark red cell in Figure A is not a
finding.** Clusters in Figure A are produced by an unsupervised algorithm that
never sees the stage labels; they are **not** claimed to correspond to clinical
stages.

**No result in this folder is statistically significant.** Across every family
tested, zero hypotheses reach q < 0.05. That is reported as it stands; no value
was altered, floored, or rescaled to produce a visible finding.

---

## 1. Cohort

Read from `derivatives/model_ready/`, with the stage mapping taken from the data
rather than reassigned.

| CN | SMC | EMCI | MCI | LMCI | AD | Total |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 20 | 20 | **16** | 20 | 20 | **116** |

116 unique subjects, 116 acquisitions, one per subject. Progression order is
fixed at `CN → SMC → EMCI → MCI → LMCI → AD` (encoded MCI = 3, LMCI = 4).
Cohort fingerprint `6ab68ae6e80f61c2`.

---

## 2. Figure A — Z-score hierarchical clustering

`zscore_hierarchical/zscore_hierarchical_{mALFF,mReHo,DC_z,FC}.{png,pdf}`

246 ROIs (rows) × 116 subjects (columns), with dendrograms on both axes and a
disease-stage colour strip above the columns.

### Z-score formula and scope

$$z_{ij} = \frac{x_{ij} - \overline{x_{i\cdot}}}{s_{i\cdot}}, \qquad s_{i\cdot} \text{ with } \mathrm{ddof}=1$$

*i* indexes ROI (1–246), *j* indexes subject (1–116). The standardisation scope
is **across subjects, within each ROI**. A value of +2 means the subject sits two
subject-level standard deviations above the cohort mean **for that ROI**.

Verified: every ROI has mean 0 (max |mean| 5.2 × 10⁻¹⁶) and sd exactly
1.000000000000. The pooled sd over all cells reads 0.995698 rather than 1; this
is arithmetic, not an error — it equals `sqrt(246 × 115 / (N−1))` with
N = 28,536.

### Relationship to the existing normalisation

The inputs are already normalised **within each acquisition**:

| Feature | Existing within-acquisition normalisation |
|---|---|
| mALFF, mReHo | ratio to that acquisition's own 246-ROI mean |
| DC_z | z-score across the 246 ROIs of that acquisition |
| FC | Fisher-z, signed, unthresholded, diagonal zeroed |

The Z-score applied here is a **second, orthogonal standardisation along the
other axis**. For DC_z this is a z-score of a z-score: the first across ROIs
within a subject, the second across subjects within an ROI. Different axes,
different operations — not redundant, but the double standardisation is stated
on the figure so it cannot be misread.

### Zero-variance ROIs

Detected explicitly, written to `csv/zero_variance_rois.csv`, excluded from
clustering and left blank rather than set to zero. **In this cohort there are
none** (0 for all four biomarkers), so the file is empty apart from its header.

### Clustering

| Setting | Value |
|---|---|
| Distance | correlation distance, 1 − Pearson *r* |
| Linkage | average (UPGMA) |
| Leaf ordering | `scipy.cluster.hierarchy.optimal_leaf_ordering` |
| Determinism | fully deterministic — no random initialisation |

Correlation distance is used because the question is whether ROIs vary
*together* across subjects: a question about profile shape, not magnitude.
Average linkage is used because it does not assume the spherical, equally sized
clusters that Ward's method implies.

Cophenetic correlation — how faithfully the dendrogram preserves the original
distances:

| Biomarker | ROI tree | Subject tree |
|---|---:|---:|
| mALFF | 0.646 | 0.486 |
| mReHo | 0.653 | 0.513 |
| DC_z | 0.605 | 0.504 |
| FC | 0.780 | 0.572 |

The subject trees are the weaker of the two throughout (0.49–0.57), meaning the
subject dendrogram is a loose summary of the real distances. **This is a further
reason not to read the subject clusters as disease groupings.**

### Colour scale

Diverging `RdBu_r`: **blue = below the cohort mean, red = above**, white near
zero. Symmetric, clipped at the 99th percentile of |Z| for that biomarker (the
exact value is printed on each colourbar). Clipping affects display only; the
CSVs hold unclipped values.

### Column order

Clustering reorders the subjects. `csv/clustering_order_<bm>_subjects.csv` maps
every displayed column to its subject ID, stage, stage index and original index;
`csv/clustering_order_<bm>_rois.csv` does the same for rows. Both were verified
to be exact permutations.

---

## 3. Figure B — ROI × stage significance

`roi_stage_significance/pscore_<bm>_roi_stage.{png,pdf}` — FDR-adjusted, **the
inferential figure**
`roi_stage_significance/pscore_<bm>_roi_stage_rawp.{png,pdf}` — raw, **exploratory only**

246 ROIs × 6 stages per biomarker.

These figures **do not recompute** the statistics. They read the stored output of
`scripts/audit/roi_stage_significance.py` and verify it before plotting: shapes
are 246 × 6, no cell is missing, and BH recomputed from the stored raw p-values
reproduces the stored q-values to within 1 × 10⁻⁵. That tolerance — not an exact
match — is the correct test, because the source CSVs store values rounded to 6
decimal places. Observed discrepancies are ~2.6 × 10⁻⁶, consistent with rounding
alone.

### Test and effect size

| | |
|---|---|
| Test | two-sided **Mann–Whitney U**, stage vs the pooled other five |
| Group sizes | 16–20 versus 96–100 |
| Effect size | **Hedges' *g***, bias-corrected, stage minus rest |
| Correction | **Benjamini–Hochberg over 246 × 6 = 1,476 tests, declared per biomarker** |
| Threshold | q < 0.05 |

Mann–Whitney is used because normality cannot be assumed at these group sizes.

### Colour scales — never mixed

The primary figure plots **−log10(q)** on `YlOrRd`. The exploratory figure plots
**−log10(p)** on `YlGnBu`, in a separate file. Raw and adjusted values are never
placed on the same scale, and p and q are carried in separate arrays and separate
CSVs throughout. Validation confirms q ≥ p in every cell of every matrix.

The colour axis is extended to at least −log10(0.05) = 1.301 so the threshold is
always visible on the colourbar, marked by a black line labelled `q = 0.05`.
If nothing reaches it, that is visible directly rather than hidden by
autoscaling.

### Numerical floor

A floor of 1 × 10⁻¹² is applied before taking log10, since −log10(0) is
undefined. **The floor is never reached in this cohort** — the smallest raw
p-value anywhere is 9.65 × 10⁻⁵ — so it affects nothing and is documented only
for completeness.

### Significance marks

Cells below the threshold are marked with a black star. In the FDR-adjusted
figures **no cell is marked**, and each figure states so explicitly in red:
*"no cell reaches q < 0.05 (0 of 1476 ROI × stage tests)"*. Stars appear only on
the raw exploratory figures, where they carry no inferential weight — 121 of
1,476 mALFF cells fall below p < 0.05, against ~74 expected by chance alone at
that family size.

### Top annotation strip

The strip above each heatmap shows the **stage-level multivariate** one-vs-rest
permutation result for that biomarker. It is a **different statistic** from the
ROI tests beneath it: it asks whether a stage is separable in the full
246-dimensional space, not whether any single ROI differs. It is drawn in a
different colormap (`Purples`) with its own colourbar, labelled
*"permutation test on all 246 ROIs jointly — NOT an ROI-specific p-value"*. It is
never substituted for an ROI-level p-value.

### Results

| Biomarker | smallest raw p | smallest q | cells with q < 0.05 | max \|g\| |
|---|---:|---:|---:|---:|
| mALFF | 1.509 × 10⁻⁴ | 0.2227 | **0** | 0.963 |
| mReHo | 9.649 × 10⁻⁵ | 0.1181 | **0** | 1.063 |
| DC_z | 1.073 × 10⁻³ | 0.7297 | **0** | 1.099 |
| FC | 6.167 × 10⁻⁴ | 0.3858 | **0** | 0.797 |

**0 of 5,904 ROI × stage tests reach q < 0.05.**

---

## 4. Figure C — overall 4 × 6 summary

`overall_summary/overall_zscore_significance_heatmap.{png,pdf,csv}`

4 biomarkers × 6 stages, exactly 24 cells.

Values are the **existing** stage-level multivariate one-vs-rest results from
`one_vs_rest_statistics.csv`. Nothing is recomputed. Before plotting, the script
verifies that all 24 tests are `COMPLETED`, share one cohort fingerprint
(`6ab68ae6e80f61c2`) and one seed (`20261013`), and that BH over the **full
24-test family** reproduces the stored q-values exactly (max difference 0.0).

| | |
|---|---|
| Statistic | distance-based **pseudo-F**, two-group (stage vs pooled rest) |
| Inference | permutation, **9,999** permutations |
| Smallest attainable p | 1 × 10⁻⁴ (with the +1 convention), so −log10(p) ≤ 4 |
| Correction family | **24 tests corrected together** |

Each cell prints the q-value, with the raw p and the pseudo-F statistic beneath,
so the adjusted and unadjusted values are both legible and clearly distinguished.

**Result: 0 of 24 reach q < 0.05. The smallest is q = 0.139 (mReHo at EMCI, raw
p = 0.0058).**

---

## 5. Additional figures

**`zscore_hierarchical/fc_stage_mean_connectivity_246x246.{png,pdf}`** — the mean
246 × 246 Fisher-z connectivity matrix within each stage. This is a connectivity
*pattern*, labelled separately from the 246 × 6 significance heatmap, and it
carries **no statistical test**. Off-diagonal mean z by stage: CN +0.0187,
SMC +0.0175, EMCI +0.0139, MCI +0.0150, LMCI +0.0171, AD +0.0187.

**`overall_summary/combined_summary_panels.{png,pdf}`** — panels A–C are the
regional Z-score clustering heatmaps (reusing the exact clustered orders of the
standalone figures), panel D is the 4 × 6 summary. The full-resolution 246-ROI
figures remain separate files; this composite is for orientation only, and its
caption states that panels A–C are descriptive while panel D is inferential.

---

## 6. How FC is represented

FC is a 246 × 246 matrix, not a regional scalar, so a representation had to be
chosen before any ROI-level comparison.

$$\mathrm{FCstr}_i = \operatorname{mean}_{j \neq i} \lvert z_{ij} \rvert$$

| Question | Answer |
|---|---|
| Signed or absolute? | **absolute** (unsigned magnitude) |
| Diagonal? | zeroed at construction, **excluded** from the mean |
| Missing edges? | **none** — the matrix is unthresholded, so every edge exists |
| Source | `derivatives/model_ready/X_fc_roi_unsigned.npy` |

**Why unsigned.** The obvious alternative — the signed row mean — equals DC/245
and correlates with DC_z at **r = +0.994** across this cohort. Using it would
re-plot DC_z under another name. The unsigned strength correlates with DC_z at
**r = −0.090**, so it carries different information: *total connectivity
magnitude* rather than *net signed connectivity*. A region strongly
anti-correlated with many others has high unsigned strength but low signed
degree, and in resting-state fMRI an anti-correlation is a real relationship, not
an absence of one.

This is also why **FC_Strength was excluded from the model feature set
entirely**: it equals DC/245 exactly (max difference 6.939 × 10⁻¹⁸,
r = 1.000000000000).

---

## 7. Limitations

- **Sample size.** 116 subjects across six ordinal classes, ~19 per class. This
  is the dominant constraint on every p-value here.
- **Stage imbalance.** MCI has 16 subjects, not 20, because only 16 MCI subjects
  have valid data. It was **not** padded with repeat visits, which would break
  the one-acquisition-per-subject property.
- **Null results are not proof of absence.** Nothing reaching q < 0.05 does not
  prove no biological difference exists. Effect sizes reach |g| = 1.10 in the ROI
  tests; they do not survive correction for 1,476 simultaneous comparisons at
  this *n*. The cause is effect size relative to between-subject variance, not a
  p-value resolution limit.
- **Significance and prediction are different questions.** These null results do
  not establish that the features lack predictive information — nor that they
  carry any. No model has been trained.
- **One-vs-rest ignores ordinality.** Pooling the other five stages discards the
  CN → AD ordering. It answers "is this stage separable", not "do the stages
  progress".
- **Cross-sectional.** Labels are the diagnostic group at the scanned visit, so
  **no progression claim is supported**.
- **Confounds not modelled here.** Residual motion still correlates with 129/246
  (mALFF) and 131/246 (mReHo) ROIs after nuisance regression, and one site
  contributes 40% of AD acquisitions. Neither is included as a covariate in
  these tests.
- **Clusters are not diagnoses.** Subject-tree cophenetic correlations of
  0.49–0.57 mean the subject dendrogram is a loose summary of the real distances.

---

## 8. Reproducing these outputs

Windows Python, single worker, a few minutes in total. The scripts read existing
derivatives; preprocessing and feature extraction are not re-run.

```bash
python scripts/audit/zscore_hierarchical_heatmaps.py    # Figure A + FC stage means
python scripts/audit/roi_stage_pscore_heatmaps.py       # Figure B
python scripts/audit/overall_and_combined_summary.py    # Figures C and D
python scripts/audit/validate_zscore_pscore_outputs.py  # 108 checks
```

Order matters only in that the combined figure reads the clustering-order CSVs
written by the first script, and validation must run last. Everything is
deterministic: clustering has no random initialisation, and the permutation
results are read from a stored, seeded analysis rather than regenerated.

### Inputs

```
derivatives/model_ready/X_regional.npy           116 x 246 x 3  (mALFF, mReHo, DC_z)
derivatives/model_ready/X_fc_roi_unsigned.npy    116 x 246      (FC strength)
derivatives/model_ready/X_fc.npy                 116 x 246 x 246
derivatives/model_ready/{y,subjects,metadata}    labels and provenance
derivatives/.../v2_cohort116/<bm>_roi_stage_*.csv        stored ROI x stage statistics
derivatives/.../v2_cohort116/one_vs_rest_statistics.csv  stored stage-level statistics
derivatives/fsfast/FINAL_6GROUP_AUDIT/dataset_brainnetome_summary.csv  ROI names
```

### Outputs

```
zscore_and_pscore_heatmaps/
├── zscore_hierarchical/     5 figures x 2 formats
├── roi_stage_significance/  8 figures x 2 formats (4 FDR + 4 raw exploratory)
├── overall_summary/         2 figures x 2 formats + the 4 x 6 CSV
├── csv/                     Z-scores, clustering orders, p, q, g, -log10 matrices
├── logs/                    full run logs for all three scripts
└── Z_SCORE_AND_PSCORE_HEATMAPS_README.md
```

Nothing outside this folder was modified. Validation confirms that no
pre-existing result in `v2_cohort116`, and no earlier-cohort result in the parent
directory, was touched, and that `HEATMAP_README.md`,
`HEATMAP_README.v1_uncorrected.md`, `FINAL_STATISTICAL_AUDIT.md` and
`one_vs_rest_checkpoint.jsonl` are all preserved.

---

## 9. What these figures do not show

- They do **not** show a statistically significant difference between any stages.
- They do **not** show disease progression; the design is cross-sectional.
- They do **not** show that unsupervised clusters correspond to clinical stages.
- They do **not** show model performance. **No model has been trained**, and no
  accuracy, sensitivity, specificity or AUC exists anywhere in this project.
- They do **not** establish that all 246 ROIs are individually disease-relevant.
