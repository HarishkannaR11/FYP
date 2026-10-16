# Results and Interpretation

All results below are **statistical analyses of the feature data**. No
predictive model has been trained, so **no accuracy, sensitivity,
specificity, AUC or any other performance metric appears in this document** —
none exists.

Cohort: **116 subjects, 116 acquisitions, one per subject**
(CN 20, SMC 20, EMCI 20, MCI 16, LMCI 20, AD 20).
Fingerprint `6ab68ae6e80f61c2`.

---

## 1. Overall one-vs-rest summary (4 biomarkers × 6 stages)

Each cell: that stage's subjects against the other five **pooled**.
Distance-based pseudo-F (two-group PERMANOVA), 9,999 subject-level
permutations, seed `20261013`, Benjamini–Hochberg across all 24 tests.

Figure — [overall_significance_heatmap.png](../derivatives/INTRA_INTER_CLASS_HEATMAPS/v2_cohort116/figures/overall_significance_heatmap.png) · [.pdf](../derivatives/INTRA_INTER_CLASS_HEATMAPS/v2_cohort116/figures/overall_significance_heatmap.pdf)
Source — [overall_significance_heatmap.csv](../derivatives/INTRA_INTER_CLASS_HEATMAPS/v2_cohort116/overall_significance_heatmap.csv) · [one_vs_rest_statistics.csv](../derivatives/INTRA_INTER_CLASS_HEATMAPS/v2_cohort116/one_vs_rest_statistics.csv)

| Biomarker | Stage | n | pseudo-F | Raw p | Adjusted q | q < 0.05 |
|---|---|---:|---:|---:|---:|:--:|
| mALFF | CN | 20 | 1.058 | 0.3506 | 0.7012 | no |
| mALFF | SMC | 20 | 1.249 | 0.1648 | 0.3955 | no |
| mALFF | EMCI | 20 | 0.594 | 0.9720 | 0.9720 | no |
| mALFF | MCI | 16 | 1.527 | 0.0533 | 0.1827 | no |
| mALFF | LMCI | 20 | 0.892 | 0.5986 | 0.7597 | no |
| mALFF | AD | 20 | 1.550 | 0.0469 | 0.1827 | no |
| mReHo | CN | 20 | 1.146 | 0.2100 | 0.4582 | no |
| mReHo | SMC | 20 | 0.867 | 0.7185 | 0.7597 | no |
| **mReHo** | **EMCI** | 20 | 1.721 | **0.0058** | **0.1392** | no |
| mReHo | MCI | 16 | 1.285 | 0.0920 | 0.2760 | no |
| mReHo | LMCI | 20 | 0.935 | 0.5790 | 0.7597 | no |
| mReHo | AD | 20 | 1.524 | 0.0216 | 0.1827 | no |
| DC_z | CN | 20 | 1.329 | 0.1095 | 0.2920 | no |
| DC_z | SMC | 20 | 0.848 | 0.6807 | 0.7597 | no |
| DC_z | EMCI | 20 | 0.829 | 0.7280 | 0.7597 | no |
| DC_z | MCI | 16 | 1.519 | 0.0437 | 0.1827 | no |
| DC_z | LMCI | 20 | 0.850 | 0.6923 | 0.7597 | no |
| DC_z | AD | 20 | 1.026 | 0.4033 | 0.7446 | no |
| FC | CN | 20 | 0.986 | 0.4667 | 0.7597 | no |
| FC | SMC | 20 | 0.908 | 0.7269 | 0.7597 | no |
| FC | EMCI | 20 | 0.981 | 0.4836 | 0.7597 | no |
| FC | MCI | 16 | 1.282 | 0.0522 | 0.1827 | no |
| FC | LMCI | 20 | 0.939 | 0.6297 | 0.7597 | no |
| FC | AD | 20 | 1.274 | 0.0453 | 0.1827 | no |

**0 of 24 cells reach q < 0.05.** Smallest q = **0.1392** (mReHo at EMCI,
raw p = 0.0058).

Five cells have raw p < 0.05 (mALFF/AD, mReHo/EMCI, mReHo/AD, DC_z/MCI,
FC/AD) and FC/MCI and mALFF/MCI sit just above. None survives correction
across 24 tests.

Null-distribution means sit at ≈ 1.000 throughout, which is what a pseudo-F
should average under the null — a sanity check that the permutation scheme
behaved correctly.

---

## 2. ROI × stage one-vs-rest (246 ROIs × 6 stages)

Mann-Whitney *U*, two-sided; BH across 246 × 6 = **1,476 tests per
biomarker**.

Figures — [mALFF](../derivatives/INTRA_INTER_CLASS_HEATMAPS/v2_cohort116/figures/mALFF_roi_stage_significance_heatmap.png) · [mReHo](../derivatives/INTRA_INTER_CLASS_HEATMAPS/v2_cohort116/figures/mReHo_roi_stage_significance_heatmap.png) · [DC_z](../derivatives/INTRA_INTER_CLASS_HEATMAPS/v2_cohort116/figures/DC_z_roi_stage_significance_heatmap.png) · [FC](../derivatives/INTRA_INTER_CLASS_HEATMAPS/v2_cohort116/figures/FC_roi_stage_significance_heatmap.png)
Sources — `<biomarker>_roi_stage_{pvalues,qvalues,effectsizes,significance}.csv`

| Biomarker | Tests | raw p < 0.05 | Expected by chance | **q < 0.05** | min q | max \|g\| |
|---|---:|---:|---:|---:|---:|---:|
| mALFF | 1,476 | 121 | 74 | **0** | 0.2227 | 0.963 |
| mReHo | 1,476 | 109 | 74 | **0** | 0.1181 | 1.063 |
| DC_z | 1,476 | 77 | 74 | **0** | 0.7297 | 1.099 |
| FC | 1,476 | 108 | 74 | **0** | 0.3858 | 0.797 |

**No ROI survives FDR in any biomarker.** mALFF and mReHo run modestly above
the chance count; DC_z is essentially at chance (77 vs 74).

---

## 3. Pairwise stage comparisons (246 ROIs × 15 comparisons)

Mann-Whitney *U*; BH across 246 × 15 = **3,690 tests per biomarker**. FC
edges form a separate family per comparison.

Figures — [pairwise_mALFF](../derivatives/INTRA_INTER_CLASS_HEATMAPS/v2_cohort116/pairwise/figures/pairwise_mALFF.png) · [pairwise_mReHo](../derivatives/INTRA_INTER_CLASS_HEATMAPS/v2_cohort116/pairwise/figures/pairwise_mReHo.png) · [pairwise_DC_z](../derivatives/INTRA_INTER_CLASS_HEATMAPS/v2_cohort116/pairwise/figures/pairwise_DC_z.png) · [consensus](../derivatives/INTRA_INTER_CLASS_HEATMAPS/v2_cohort116/pairwise/figures/cross_biomarker_consensus.png)

| Biomarker | Tests | raw p < 0.05 | Expected | **q < 0.05** | min q | max \|g\| |
|---|---:|---:|---:|---:|---:|---:|
| mALFF | 3,690 | 277 | 185 | **0** | 0.0667 | 1.223 |
| mReHo | 3,690 | 270 | 185 | **0** | 0.4839 | 1.368 |
| DC_z | 3,690 | 203 | 185 | **0** | 0.7842 | 1.286 |
| **FC edges** | **452,025** | — | — | **0** | 0.2011 | — |

**Cross-biomarker consensus is 0 in every cell** — no ROI × comparison has
even one biomarker significant, so `consensus_contributors.csv` is a header
only.

---

## 4. Interpretation

### What the results establish

Across **five independent analyses** on this cohort — pairwise ROI, pairwise
FC edges, intra- versus inter-class permutation, ROI × stage one-vs-rest, and
stage-level multivariate one-vs-rest — **nothing survives multiple-comparison
correction**.

That consistency matters. It is not one underpowered test; it is a coherent
absence of detectable **univariate** stage signal at this sample size.

### What the results do NOT establish

- **They do not prove that no biological difference exists.** Failure to
  reject the null is not evidence for it.
- **They do not prove the features carry no predictive information.**
  Statistical significance and predictive performance are different
  questions. A classifier weights many weak, correlated features jointly;
  per-feature testing asks whether any single one separates groups on its
  own. The second can fail while the first succeeds.
- **They say nothing about causality.**

### Why the null arises — effect size, not p-value resolution

An earlier interpretation attributed the null to a floor on attainable
p-values. **That was wrong and has been withdrawn**
([FINAL_STATISTICAL_AUDIT.md](../derivatives/INTRA_INTER_CLASS_HEATMAPS/v2_cohort116/FINAL_STATISTICAL_AUDIT.md) §2). The exact two-sided minimum for
20 vs 20 is 1.45 × 10⁻¹¹, and SciPy's asymptotic default returns 6.80 × 10⁻⁸
— orders of magnitude below any threshold used.

The real picture:

| Biomarker | min raw p (ROI × stage) | BH rank-1 threshold | ratio |
|---|---:|---:|---:|
| mReHo | 9.65 × 10⁻⁵ | 3.39 × 10⁻⁵ | 2.8× too large |
| mALFF | 1.51 × 10⁻⁴ | 3.39 × 10⁻⁵ | 4.5× |
| FC | 6.17 × 10⁻⁴ | 3.39 × 10⁻⁵ | 18× |
| DC_z | 1.07 × 10⁻³ | 3.39 × 10⁻⁵ | 32× |

mReHo came within a factor of three. The tests had ample resolution; the
observed effects were not large enough relative to between-subject variance
at n = 16–20 per stage.

Supporting measurement: between-stage variance is roughly **0.23×**
within-stage variance per ROI for mALFF — the stage effect is about four
times smaller than individual differences.

### Effect sizes are not zero

|*g*| reaches **1.37** in pairwise comparisons and **1.10** in ROI × stage.
Effects of that magnitude are large by convention. They do not survive
correction for 1,476 or 3,690 simultaneous tests at this sample size — which
is exactly what multiple-comparison correction is for. Without it, mALFF's
277 raw hits would be reported as findings when ~185 are expected from noise.

### Confounds that remain

- **Residual motion**: after the 27-regressor design, 129/246 (mALFF) and
  131/246 (mReHo) ROIs still correlate with mean FD at p < 0.05, against
  ~12 expected. Motion must enter any model as a covariate.
- **Site concentration**: one site contributes 40% of AD acquisitions, two
  contribute 68%, while no site exceeds 19% of CN. Any CN–AD contrast is
  partly a site contrast. No harmonisation has been applied.
- **MCI n = 16** makes its comparisons the least powered.
- **FC edge-wise inference is severely underpowered**: 30,135 edges against
  16–20 subjects per group. Minimum q-values near 1.0 reflect that ratio.
  Zero significant edges is **not** evidence that connectivity differences
  are absent; the design cannot detect them.

### Bearing on the proposed model

These results motivate a multivariate approach but **do not validate one**.
A high-capacity model is not justified merely because univariate tests are
non-significant — if anything, 116 subjects and a weak signal raise the
overfitting risk. Any future model must be judged on held-out predictive
performance under subject-grouped cross-validation, against simple baselines,
and not on the argument that something more complex ought to work better.

---

## 5. Provenance

| Item | Value |
|---|---|
| Cohort fingerprint | `6ab68ae6e80f61c2` |
| Subjects / acquisitions | 116 / 116, one per subject |
| Permutations | 9,999 |
| Random seed | `20261013` |
| Tests completed | 24 / 24, all `COMPLETED` |
| Missing results zero-filled | none — the plotter refuses to render on a gap |
| 114-subject results mixed in | no — fingerprint check rejects them |

Audit: [FINAL_STATISTICAL_AUDIT.md](../derivatives/INTRA_INTER_CLASS_HEATMAPS/v2_cohort116/FINAL_STATISTICAL_AUDIT.md) ·
README: [HEATMAP_README.md](../derivatives/INTRA_INTER_CLASS_HEATMAPS/v2_cohort116/HEATMAP_README.md) ·
Superseded: [HEATMAP_README.v1_uncorrected.md](../derivatives/INTRA_INTER_CLASS_HEATMAPS/v2_cohort116/HEATMAP_README.v1_uncorrected.md)
