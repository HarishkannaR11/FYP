# Diagnostic-Group Hierarchical Clustering

Hierarchical clustering of the **six diagnostic groups** by their mean regional
biomarker profiles, for the 116-subject cohort.

Validated by `scripts/audit/validate_diagnostic_group_hierarchical.py` —
**93/93 checks pass**.

For the plain-language explanation written for the project review, see
[`DIAGNOSTIC_GROUP_CLUSTERING_INTERPRETATION.md`](DIAGNOSTIC_GROUP_CLUSTERING_INTERPRETATION.md).

---

## 1. What is new here

| Analysis | What is clustered |
|---|---|
| `zscore_hierarchical/` (existing) | 246 ROIs **and** 116 subjects |
| `pscore_hierarchical/` (existing) | 246 ROIs, by **significance profile** |
| **this folder** | **the 6 diagnostic groups**, by mean biomarker profile |

The existing figures answer "which regions behave alike" and "which regions
have similar evidence". This one answers the question actually asked: **which
diagnostic groups look like each other.**

---

## 2. Cohort and labels, verified at run time

| CN | SMC | EMCI | MCI | LMCI | AD | Total |
|---:|---:|---:|---:|---:|---:|---:|
| 20 | 20 | 20 | **16** | 20 | 20 | **116** |

Stage encoding read from `derivatives/model_ready/metadata.json` and checked
against the script's assumption before anything is computed:
`CN=0, SMC=1, EMCI=2, MCI=3, LMCI=4, AD=5`. **MCI and LMCI are not relabelled.**
Clinical progression order `CN → SMC → EMCI → MCI → LMCI → AD` is preserved in
every CSV and printed on every figure.

---

## 3. The group matrix

For each biomarker, a **246 × 6** matrix whose cell *(i, g)* is the mean of ROI
*i* over the subjects in diagnostic group *g*, computed from the subject-level
features. Validation recomputes every cell from `X_regional.npy` /
`X_fc_roi_unsigned.npy` and requires agreement to 1 × 10⁻¹².

### Standardisation — display only

$$zg_{i,g} = \frac{m_{i,g} - \overline{m_{i\cdot}}}{s_{i\cdot}}, \qquad \mathrm{ddof}=1$$

Each ROI is standardised **across the six groups**. A red cell means this group
is above that ROI's six-group average; blue means below.

**Raw unstandardised group means are preserved** in
`csv/group_means_raw_<bm>.csv` and are never overwritten.

### Two consequences of row-centring, stated up front

1. **Induced negative correlation.** Because every row is centred across the six
   groups, the six column profiles are linearly constrained and their pairwise
   correlations are biased downward, with expected mean −1/(k−1) = **−0.2** for
   k = 6. Observed correlation *distances* therefore sit near or above 1.0
   (range 0.80–1.44 across biomarkers). **This is a property of row-centring,
   not a finding**, and it is why distances are not interpreted in absolute
   terms.
2. **A hard bound on |z|.** With k = 6 and ddof = 1, |z| ≤ (k−1)/√k = **2.041**.
   The colour scale is therefore fixed at **±2.05** for all four biomarkers,
   covering every attainable value and making panels directly comparable.
   Validation confirms no value exceeds the bound.

Zero-variance ROIs would be flagged in `zero_variance`, left blank, and excluded
from clustering — never set to zero. **There are none** (0 for all four
biomarkers).

---

## 4. Clustering configuration

Recorded machine-readably in `csv/clustering_configuration.csv`.

| Objects | Distance | Linkage | Input space |
|---|---|---|---|
| **6 diagnostic groups** | correlation | average | standardised 246-D profile |
| **246 ROIs** | Euclidean | average | standardised 6-D profile |
| 6 groups (FC matrix-level check) | correlation | average | 30,135-D mean edge vector |
| 116 subjects (Figure C) | correlation | average | 738-D concatenated regional |

Leaf ordering is `scipy.cluster.hierarchy.optimal_leaf_ordering` throughout;
everything is deterministic with no random initialisation.

**Why correlation distance for the groups.** With 246 observations per group the
correlation is well determined, and the question is whether two groups deviate
in the *same pattern* across regions rather than by the same amount. Average
linkage is used because it makes no assumption of spherical, equally sized
clusters the way Ward's method does.

**Why Euclidean for the ROIs.** Each ROI row has only 6 values, where a
correlation is unstable and undefined for a constant row. Note that because rows
are standardised, ‖a−b‖² = 2(k−1)(1−r) exactly, so for rows this *is* a monotone
transform of correlation distance — Euclidean is simply the stable way to
compute it.

---

## 5. What the dendrograms actually show

Leaf order is the dendrogram's, **not** the clinical order. Reordering is
expected and does not imply anything about progression.

| Biomarker | Group leaf order | Branches found | Cophenetic *r* |
|---|---|---|---:|
| mALFF | EMCI, SMC, CN, AD, LMCI, MCI | {LMCI,AD} {SMC,EMCI} {CN,SMC,EMCI} {MCI,LMCI,AD} | 0.763 |
| mReHo | AD, CN, MCI, SMC, LMCI, EMCI | {CN,AD} {SMC,LMCI} {SMC,MCI,LMCI} {SMC,EMCI,MCI,LMCI} | 0.554 |
| DC_z | MCI, AD, LMCI, SMC, CN, EMCI | {CN,SMC} {LMCI,AD} {MCI,LMCI,AD} {CN,SMC,EMCI} | 0.674 |
| FC | CN, MCI, AD, LMCI, SMC, EMCI | {CN,MCI} {SMC,EMCI} {LMCI,AD} {SMC,EMCI,LMCI,AD} | 0.863 |

### Bootstrap stability

Because a single dendrogram from 6 group means is easy to over-read, each branch
was re-estimated by **1,000 bootstrap resamples of subjects within groups**
(seed 20261014), recomputing the group means and reclustering each time. The
frequency with which each observed branch reappears is in
`csv/group_cluster_stability_bootstrap.csv`.

| Branch | mALFF | mReHo | DC_z | FC |
|---|---:|---:|---:|---:|
| {LMCI, AD} | **46%** | — | 26% | 38% |
| {CN, SMC} | — | — | **49%** | — |
| {SMC, EMCI} | 25% | — | — | **45%** |
| {CN, MCI} | — | — | — | **72%** |
| {CN, AD} | — | 36% | — | — |
| {CN, SMC, EMCI} | 25% | — | 19% | — |
| {MCI, LMCI, AD} | 19% | — | 17% | — |

**This is a descriptive stability frequency, not a p-value and not a
significance test.**

Read honestly: **most branches are unstable.** Only one exceeds 50%
({CN, MCI} in FC, 72%). The adjacent-severity pairing {LMCI, AD} is the one
grouping that recurs across three of the four biomarkers, but never above 46%.
A branch appearing in fewer than half of resamples should not be presented as a
result.

---

## 6. FC representation

`csv/fc_transformation_details.csv` records this machine-readably.

$$\mathrm{FCstr}_i = \operatorname{mean}_{j \neq i} \lvert z_{ij} \rvert$$

| Property | Value |
|---|---|
| Source | `X_fc.npy` (116, 246, 246), Fisher-z, signed, unthresholded, diagonal zeroed |
| ROI summary | unsigned mean of off-diagonal \|Fisher-z\| |
| Signed or absolute | **absolute** |
| Diagonal | zeroed at construction, **excluded** from the mean |
| Missing edges | **none** — unthresholded |

Validation recomputes this from the raw 246 × 246 matrix for a subject and
confirms it matches `X_fc_roi_unsigned.npy`.

**Signed versus unsigned, stated conceptually.** Signed connectivity lets
positive and negative edges **cancel**: a region with many strong positive and
many strong negative connections can have a near-zero signed mean. Unsigned
connectivity measures **total connection magnitude** and does not cancel. They
answer different questions, and **neither is superior**. The signed row mean was
not used here only because it equals DC/245 exactly and would duplicate the DC_z
panel (r = +0.994); its low correlation with DC_z (r = −0.090) shows the unsigned
summary carries *different* information, which is not the same as better
information.

### Matrix-level cross-check

Spec option 2 was also computed: clustering the groups on the full **30,135-edge
upper triangle** of each group's mean 246 × 246 connectivity matrix
(`figures/fc_representation_comparison.*`).

- ROI-summary leaf order: CN, MCI, AD, LMCI, SMC, EMCI
- Matrix-level leaf order: MCI, CN, EMCI, SMC, LMCI, AD
- **Shared branches: 3 of 4** — {LMCI, AD}, {SMC, EMCI}, {SMC, EMCI, LMCI, AD}

Both representations place CN and MCI together at the deepest split. The
absolute distances differ greatly (0.80–1.44 for the ROI summary versus
0.07–0.13 for the matrix level) because the spaces differ; only the **topology**
is comparable, and it largely agrees.

**Limitation:** the ROI summary discards which connections changed, keeping only
how much total connectivity each region has. The matrix level keeps every edge
but estimates 30,135 numbers from 16–20 subjects per group.

---

## 7. Subject-level clustering (Figure C)

`figures/subject_level_clustering.*` — subjects clustered on the 738-dimensional
concatenation of standardised mALFF + mReHo + DC_z. **Diagnostic labels are used
only as the colour strip and are never an input to the clustering.**

If subjects of one group clustered together, the strip would show contiguous
blocks of one colour. **It does not** — the strip is visibly interleaved.

Quantified by the nearest-neighbour same-group rate
(`csv/subject_neighbour_purity.csv`):

| Scope | Rate | Chance |
|---|---:|---:|
| **Overall** | **0.233** | 0.160 |
| CN | 0.30 | 0.160 |
| SMC | 0.25 | 0.160 |
| EMCI | 0.20 | 0.160 |
| MCI | 0.19 | 0.160 |
| LMCI | 0.30 | 0.160 |
| AD | 0.15 | 0.160 |

Cophenetic *r* = 0.472, so the subject dendrogram is itself a loose summary of
the real distances.

23.3% against a 16.0% chance baseline is **weakly above chance and far from
separation**. AD (0.15) sits at or below chance. **This is a descriptive
summary, not a classification accuracy**: there is no train/test split, no
held-out data, and no trained model, so it is not evidence of diagnostic
discrimination.

---

## 8. Statistical status

Everything in this folder is **descriptive**. For every figure:

| Layer | What it is |
|---|---|
| Observed values | subject-level mALFF / mReHo / DC_z / FC strength |
| Transformation | group mean, then z across the six groups (ddof = 1) |
| Distance | correlation (groups) / Euclidean (ROIs) |
| Linkage | average |
| Branch height | **a distance** — never a p-value |
| Inference | **none performed here** |

**No p-value is derived from any branch height. No cluster is claimed to be
statistically significant.**

The project's existing FDR-corrected analyses remain the inferential record, and
they are null:

- 0 of 24 stage-level multivariate tests reach q < 0.05 (smallest q = 0.139,
  mReHo at EMCI)
- 0 of 5,904 ROI × stage tests reach q < 0.05 (highest P-score 0.928 < 1.301)
- earlier pairwise regional and FC analyses: also no FDR-significant findings

A visually distinct column in these heatmaps is an **exploratory observation**,
not a discovery.

---

## 9. Files

```
diagnostic_group_hierarchical/
├── figures/
│   ├── group_hierarchical_{mALFF,mReHo,DC_z,FC}.{png,pdf}   Figure A
│   ├── group_hierarchical_combined.{png,pdf}                Figure B
│   ├── fc_representation_comparison.{png,pdf}               FC check
│   └── subject_level_clustering.{png,pdf}                   Figure C
├── csv/
│   ├── group_means_raw_<bm>.csv              246 x 6, unstandardised
│   ├── group_means_standardized_<bm>.csv     246 x 6, + zero_variance flag
│   ├── group_dendrogram_leaf_order.csv       display order + source index
│   ├── roi_dendrogram_leaf_order.csv         4 x 246
│   ├── group_cluster_stability_bootstrap.csv branch frequencies
│   ├── clustering_configuration.csv          metric/linkage per figure
│   ├── fc_transformation_details.csv         FC derivation
│   └── subject_neighbour_purity.csv          rate vs chance
├── logs/diagnostic_group_hierarchical.log
├── DIAGNOSTIC_GROUP_HIERARCHICAL_README.md
└── DIAGNOSTIC_GROUP_CLUSTERING_INTERPRETATION.md
```

## 10. Reproducing

```bash
python scripts/audit/diagnostic_group_hierarchical.py
python scripts/audit/validate_diagnostic_group_hierarchical.py
```

Single worker, low memory, a few minutes including 1,000 bootstrap resamples per
biomarker. Validation confirms no existing dataset, feature, statistical result,
checkpoint or figure was modified.
