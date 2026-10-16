# Dendrogram Branch Annotations — exact definitions

Every internal merge of the top dendrogram carries **two** labels. They are
different kinds of quantity and must not be read as one.

```
d = 0.898      merge DISTANCE  -- geometric, NOT a p-value
q = 0.171*     permutation q    -- a real test, but SELECTIVE (see below)
```

---

## 1. `d` — merge distance

The height at which two clusters join in the linkage tree: **correlation
distance between their 246-dimensional P-score profiles**, average linkage.

**This is a distance, not a significance value.** A small `d` means two groups
have similar *patterns of statistical evidence* across regions. It carries no
probability interpretation whatsoever. Labelling a merge height as a "P-score"
would be a category error, which is why every label is prefixed `d =`.

## 2. `q*` — permutation q-value for that contrast

Each merge joins cluster **A** and cluster **B**, which poses a well-defined
question: *do the subjects in A differ from the subjects in B?* That is a real
two-group comparison, and it is answered with the machinery already validated
elsewhere in this project:

| | |
|---|---|
| Statistic | distance-based **pseudo-F** (two-group PERMANOVA) |
| Data | subject-level 246-D biomarker vectors, z-scored across subjects |
| Inference | **9,999 permutations** of the cluster labels, seed 20261016 |
| Correction | Benjamini–Hochberg across the **5 merges** of that biomarker |
| Smallest attainable p | 1 × 10⁻⁴ |

This is **not** an average of ROI P-scores. Averaging ROI significance values
and calling the result a branch score would have no inferential meaning, and
was explicitly avoided.

### Why the asterisk — the limitation that governs interpretation

**The clusters were derived from the data.** The dendrogram was built by
looking at the P-score profiles, and the merge being tested is the split the
algorithm chose *because* it separated the groups well. Testing that split on
the same data is **selective inference** — commonly called double dipping.

The permutation null assumes the partition was fixed before seeing the data.
It was not. The resulting p-values are therefore **anti-conservative**: too
small, by an amount that is not quantified here.

A valid unconditional test would require either:

- **sample splitting** — build the tree on one half, test on the other. Not
  feasible at n = 116 across six classes, where a half-sample gives ~10
  subjects per stage; or
- a **selective-inference correction for hierarchical clustering**, which
  conditions on the selection event.

Neither is implemented. Consequently:

> **No branch is claimed to be statistically significant.** The `q*` values
> are descriptive, for ranking which merges separate their subjects more
> strongly than others, and nothing more.

Across all four biomarkers, 1 of 20 merges has `q* < 0.05` (mReHo,
{EMCI} + {SMC,AD}, q\* = 0.0465). Given the selective inference above, that is
**not** reported as a finding.

---

## 3. What the heatmap shows — unchanged

The cells are `P-score = -log10(FDR-adjusted q)` from the stored ROI × stage
analysis: two-sided Mann–Whitney U per ROI per stage (stage vs pooled rest),
Benjamini–Hochberg over 246 × 6 = 1,476 tests per biomarker.

**0 of 5,904 cells reach q < 0.05. The largest P-score anywhere is 0.9278,
below the 1.3010 threshold.** Verified against source before plotting;
unchanged from the previous audit. Colour scale fixed at 0.0–1.40 and never
rescaled.

**The heatmap q and the branch q\* are different tests on different units** —
the heatmap tests one ROI against one stage; the branch tests one cluster of
subjects against another across all 246 ROIs. They are not comparable and are
corrected within separate families.

---

## 4. Merge table

`csv/dendrogram_merge_statistics.csv`, 20 rows (4 biomarkers × 5 merges):

| column | meaning |
|---|---|
| `cluster_A`, `cluster_B` | the diagnostic groups merged |
| `n_subjects_A`, `n_subjects_B` | subjects contributing to the test |
| `merge_distance` | the `d` label — geometric |
| `pseudo_F` | observed test statistic |
| `p_permutation` | permutation p for that contrast |
| `q_BH_over_5_merges` | the `q*` label |
| `q_below_0.05` | boolean, **not** a significance claim |
| `test_and_correction` | the method |
| `validity_caveat` | the selective-inference warning, on every row |

---

## 5. Preserved

Six groups CN, SMC, EMCI, MCI, LMCI, AD; the P-score heatmap; the top
dendrogram; ROI labels on the left with **no ROI dendrogram**; the colourbar
with the 1.301 threshold marked; and the clinical progression order
`CN → SMC → EMCI → MCI → LMCI → AD` printed on every figure. Dendrogram
column order differs from clinical order, as expected.

Original figures (`pscore_grouptree_<bm>.*`) were **not overwritten**; the
annotated versions are new files `pscore_grouptree_annotated_<bm>.*`.

## 6. Reproducing

```bash
python scripts/audit/pscore_group_tree_annotated.py
```

Deterministic given the seed. Reads stored q-values and subject-level feature
arrays; modifies no dataset, checkpoint or existing statistical output.
