# P-Score Hierarchical Clustering Heatmaps

ROI-level stage-versus-rest statistical significance for the 116-subject cohort,
clustered by significance profile.

Validated by `scripts/audit/validate_pscore_hierarchical.py` — **70/70 checks
pass**. Companion to
[`../Z_SCORE_AND_PSCORE_HEATMAPS_README.md`](../Z_SCORE_AND_PSCORE_HEATMAPS_README.md).

---

## Read this before reading the figures

**No ROI passes FDR correction in any biomarker: 0 of 5,904 ROI × stage tests
reach q < 0.05.** The highest P-score observed anywhere is **0.928**, against a
threshold of **1.301**. Every figure here states that on its face.

The dendrograms therefore order rows by similarity of *statistical evidence* on a
surface where that evidence is uniformly weak. **A tight cluster is not a
finding.** Clustering is used here as a display ordering, not as a result.

---

## 1. The P-score

$$\text{P-score}_{i,s} = -\log_{10} q_{i,s}$$

for ROI *i* and stage *s*, where *q* is the Benjamini–Hochberg FDR-adjusted
p-value for that ROI × stage hypothesis.

| | |
|---|---|
| Test | two-sided **Mann–Whitney U**, stage vs the pooled other five |
| Group sizes | 16–20 versus 96–100 |
| Correction | **Benjamini–Hochberg over 246 × 6 = 1,476 tests, per biomarker** |
| Threshold | q < 0.05, i.e. **P-score > 1.3010** |
| Source | `../csv/roi_stage_qvalues_<bm>.csv` |

Nothing is recomputed. The q-values are read from the stored ROI × stage
analysis, which earlier validation confirmed to be bit-identical to the output of
`scripts/audit/roi_stage_significance.py`. Validation re-derives every P-score
from those q-values and requires agreement to 1 × 10⁻¹².

**The stage-level multivariate q-values are not used here.** Those test whether a
stage is separable in the full 246-dimensional space and are a different
statistic; treating them as ROI-specific would be a category error. They appear
only as a labelled annotation strip in the separate
`../roi_stage_significance/` figures.

### Numerical floor

A floor of 1 × 10⁻¹² would apply if any q were exactly 0. **No q-value is zero**
— the smallest is 0.118 — so the floor is never engaged. It exists only so the
logarithm is total. `−log10(1.0)` returns negative zero in IEEE arithmetic; the
sign of zero is normalised so the CSVs read `0.0` rather than `-0.0`.

Missing q-values would be carried as blanks and drawn in grey. **There are
none**; all 1,476 cells are present for all four biomarkers, and validation
confirms the blank count in the P-score CSVs equals the blank count in the source
q-value CSVs. No missing value is replaced by zero anywhere.

---

## 2. Fixed colour scale

All four biomarker figures use **one fixed scale, 0.0 to 1.40**, on `YlOrRd`.
It is never autoscaled per biomarker.

This matters in a dataset where every result is null. Independent autoscaling
would stretch each biomarker's noise across the full colour range and make four
equally non-significant panels look like four different findings. With the scale
fixed, DC_z (max P-score 0.137) correctly appears almost blank beside mReHo
(max 0.928), and the comparison between panels is meaningful.

The upper bound of 1.40 sits **above** the significance threshold, so
`q = 0.05 → 1.301` is always visible on the colourbar, marked by a black line.
That nothing reaches it is visible directly rather than hidden by autoscaling.

Because a fixed scale compresses the observed values into the lower part of the
range, **each figure also carries a histogram of its own P-score distribution**
across all 1,476 cells, with the threshold marked. The fixed scale therefore
conceals nothing about where the values actually lie.

---

## 3. Clustering

| Setting | Value | Why |
|---|---|---|
| Objects | 246 ROIs, each a 6-D P-score vector | one row per ROI |
| Distance | **Euclidean** | a P-score profile is a vector of magnitudes |
| Linkage | **Ward** | the standard pairing with Euclidean distance |
| Ordering | `optimal_leaf_ordering` | deterministic, improves readability |
| Columns | **never clustered** | stages stay in clinical order |

Euclidean distance is used here rather than the correlation distance used for the
Z-score figures. A P-score profile is a vector of *magnitudes*: two ROIs are alike
when their evidence is of similar strength at the same stages. Correlation
distance would discard magnitude entirely — calling an ROI with q ≈ 1 everywhere
"identical" to one with its strongest evidence at the same stage — and it is
undefined for an ROI whose six q-values are constant.

### Column order is fixed

The six stage columns are **never** reordered by clustering. They remain in the
established order `CN → SMC → EMCI → MCI → LMCI → AD` as encoded in
`derivatives/model_ready/metadata.json` (MCI = 3, LMCI = 4), verified at run time.
Letting a clustering algorithm permute disease stages would destroy the ordinal
structure that the whole project is built around.

### Cophenetic correlation

How faithfully each dendrogram preserves the original distances:

| Biomarker | P-score tree | min q | max P-score | median P-score |
|---|---:|---:|---:|---:|
| mALFF | 0.657 | 0.2227 | 0.652 | 0.027 |
| mReHo | **0.724** | 0.1181 | **0.928** | 0.035 |
| DC_z | 0.576 | 0.7297 | 0.137 | 0.009 |
| FC | **0.403** | 0.3858 | 0.414 | 0.074 |

FC's 0.403 is low: its dendrogram is a loose summary of the real distances, and
its row groupings should be given correspondingly little weight.

---

## 4. What this clustering does and does not mean

**It groups ROIs with similar statistical significance profiles.** That is all.

It does **not** group ROIs that are:

- biologically similar,
- anatomically adjacent,
- functionally connected,
- or part of a shared network.

Two ROIs can land in the same cluster purely because both produced weak evidence
at the same stage, which at this sample size is the expected behaviour of noise.

Specifically, **none of the following may be claimed from these figures**:

- that any ROI or cluster is statistically significant — nothing reaches q < 0.05;
- that a cluster corresponds to a disease process or progression pathway;
- that cluster membership is evidence of anything about an ROI;
- that the stages differ, in any direction, by these tests.

The original q-values are preserved unchanged in `../csv/roi_stage_qvalues_<bm>.csv`
and in the source analysis; the P-score is a monotone transformation for display
and nothing is overwritten.

---

## 5. Is clustering significance scores appropriate here?

**Partly, and with a caveat worth stating plainly.** Clustering a significance
surface is a legitimate way to display which ROIs behave alike across stages, and
it is what the requested figure shows. But a q-value is a *decision statistic*,
not a measurement: it compresses effect size and sample size into one number,
it is bounded, and under a true null it is close to uniform. In this cohort,
where nothing is significant, the P-score surface being clustered is dominated by
sampling noise, so the dendrogram structure is largely noise structure.

**The defensible complement is to cluster effect size instead**, which is why
this folder also contains:

`effectsize_hierarchical_alternative.{png,pdf}`

Hedges' *g* (stage minus rest) is a **measured, signed magnitude** that exists
whether or not a test is significant. Clustering it groups ROIs by the direction
and size of their stage-vs-rest difference — a statement about the data rather
than about a decision procedure. Shared symmetric scale ±1.1 (observed max
|g| = 1.099). Cophenetic correlations: mALFF 0.485, mReHo 0.422, DC_z 0.434,
FC 0.432.

**Effect size is still not significance.** None of those differences survives FDR
correction at this sample size. The alternative figure is shown for structure,
not for inference, and says so on its face.

---

## 6. FC representation

FC is a 246 × 246 connectivity matrix and has no scalar per ROI, so the ROI-level
statistic was computed on:

$$\mathrm{FCstr}_i = \operatorname{mean}_{j \neq i} \lvert z_{ij} \rvert$$

| Question | Answer |
|---|---|
| Signed or absolute? | **absolute** (unsigned magnitude) |
| Diagonal? | zeroed at construction, **excluded** from the mean |
| Missing edges? | **none** — unthresholded, every edge exists |
| Source | `derivatives/model_ready/X_fc_roi_unsigned.npy` |

This was **verified**, not assumed: the FC q-values used here come from
`roi_stage_significance.py`, whose documented ROI-level summary is the unsigned
mean Fisher-z strength.

**No substitution was made.** The signed mean was not used: it equals DC/245 and
correlates with DC_z at r = +0.994, so it would re-plot DC_z under another name.
DC_z was not substituted either — it is a separate panel. The unsigned strength
correlates with DC_z at r = −0.090 and therefore carries different information:
*total connectivity magnitude* rather than *net signed connectivity*.

Valid ROI-level q-values exist for all 246 ROIs × 6 stages, so no 246 × 6 matrix
had to be manufactured and no limitation applies on that count.

---

## 7. Limitations

- **Nothing is significant.** 0 of 5,904 ROI × stage tests reach q < 0.05. The
  maximum P-score (0.928) falls short of the threshold (1.301).
- **Clustering a null surface clusters noise.** See §5.
- **Sample size.** 116 subjects across six classes, ~19 per class, against a
  1,476-test correction family per biomarker.
- **Stage imbalance.** MCI has 16 subjects, not 20; it was not padded with repeat
  visits.
- **Row positions are not comparable between panels** of the combined figure —
  each is clustered independently. The colour scale *is* comparable.
- **FC's dendrogram is weakly supported** (cophenetic r = 0.403).
- **One-vs-rest ignores ordinality.** Pooling the other five stages discards the
  CN → AD ordering; it asks "is this stage separable", not "do the stages
  progress".
- **Cross-sectional design.** No progression claim is supported.
- **Confounds not modelled.** Residual motion correlates with 129/246 (mALFF) and
  131/246 (mReHo) ROIs after nuisance regression; one site contributes 40% of AD
  acquisitions. Neither enters these tests as a covariate.
- **A null result is not proof of absence**, and it does not establish that the
  features lack predictive information. **No model has been trained.**

---

## 8. Files

```
pscore_hierarchical/
├── pscore_hierarchical_mALFF.{png,pdf}
├── pscore_hierarchical_mReHo.{png,pdf}
├── pscore_hierarchical_DC_z.{png,pdf}
├── pscore_hierarchical_FC.{png,pdf}
├── pscore_hierarchical_combined.{png,pdf}          all four, shared fixed scale
├── effectsize_hierarchical_alternative.{png,pdf}   the complement of §5
└── PSCORE_HIERARCHICAL_README.md
../csv/
├── pscore_matrix_{mALFF,mReHo,DC_z,FC}.csv         246 x 6 P-scores each
└── pscore_clustering_order.csv                     984 rows: ROI order + leaf order
../logs/pscore_hierarchical.log
```

Each per-biomarker figure carries: the ROI dendrogram, the 246 × 6 heatmap with
all 246 rows (every 10th labelled with ROI ID and Brainnetome name), the fixed
colourbar with the q = 0.05 threshold marked, the observed P-score histogram, and
the explicit verdict.

`pscore_clustering_order.csv` maps, for each biomarker, every displayed row to its
dendrogram leaf index, original atlas index, ROI ID and ROI name.

---

## 9. Reproducing

```bash
python scripts/audit/pscore_hierarchical_heatmaps.py    # all 6 figures + CSVs
python scripts/audit/validate_pscore_hierarchical.py    # 70 checks
```

Single worker, low memory, a few minutes. Fully deterministic: hierarchical
clustering has no random initialisation, and the q-values are read from a stored,
seeded analysis rather than regenerated.

Validation confirms that no pre-existing file in `v2_cohort116`, and no
earlier-cohort output in the parent directory, was modified: `HEATMAP_README.md`,
`HEATMAP_README.v1_uncorrected.md`, `FINAL_STATISTICAL_AUDIT.md`,
`one_vs_rest_statistics.csv` and `one_vs_rest_checkpoint.jsonl` are all intact,
and the 108-check validation of the earlier Z-score and p-score outputs still
passes.
