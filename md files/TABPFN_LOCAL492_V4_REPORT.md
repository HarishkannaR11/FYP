# TabPFN on All 492 mALFF + mReHo Features, No Feature Selection (v4)

Written 2026-10-21. This is the single experiment proposed in `IMPROVEMENT_STUDY_V3_REPORT.md`, section 9. It was
pre-registered in `derivatives/experiments/tabpfn_local492_v4/PLAN.md` before it was run. It is an exploratory
cross-validation analysis on the fixed 116-subject cohort, not independent confirmation; the 35 "locked" subjects
are part of the CV.

## Answer

**The change does not improve on the baseline, and it makes intermediate-stage classification worse.**

| | Result |
|---|---|
| Balanced accuracy | 23.0%, against 24.6% for the baseline (−1.6 points, 95% CI −5.4 to +2.0) |
| Seeds | Better in 1 of 5 |
| Outer splits | Better in 8 of 25, worse in 14, tied in 3 |
| Corrected test | p = 0.67 |
| Macro-F1 | 2.9 points lower |
| Intermediate stages | Clearly worse: MCI recall falls from 9% to 0%, EMCI from 32% to 21% |

Under the pre-registered rules this is **not reliably different and not a consistent improvement.** The SVM-RFE k = 50
baseline on all 984 features is kept.

## What was run

- **Model:** TabPFN v2 with the baseline's settings.
- **Features:** all 246 mALFF and 246 mReHo values, raw. There was no SVM-RFE and no other learned step outside
  TabPFN.
- **Fitting:** TabPFN's internal preprocessing used only the outer training subjects. Outer-test labels were hidden
  from the fitting code.
- **Splits:** the existing baseline's 25 outer splits, checked index by index. The baseline records are byte-identical
  to the original TabPFN baseline run.
- **Selection and tuning:** none, so there is no inner loop.
- **Compute:** 24 minutes.

## Results (25 identical outer splits; 95% CI = class-stratified subject bootstrap)

| Model | Balanced acc. [CI] | Fold mean ± sd | Macro-F1 [CI] | Accuracy [CI] | Recall CN / SMC / EMCI / MCI / LMCI / AD (%) |
|---|---|---|---|---|---|
| **Baseline:** SVM-RFE k = 50 from 984 features | **24.6** [20.5, 29.0] | 24.6 ± 8.4 | **23.5** [19.2, 27.5] | **25.2** [21.0, 29.7] | 12 / 21 / 32 / 9 / 20 / 54 |
| **New:** all 492 mALFF + mReHo, no selection | 23.0 [18.8, 27.2] | 23.0 ± 6.0 | 20.6 [16.5, 24.2] | 23.8 [19.5, 28.1] | 14 / 27 / 21 / **0** / 16 / 60 |
| Baseline + class balancing | 26.1 [21.6, 30.9] | 26.1 ± 9.1 | 25.4 [20.8, 29.8] | 26.2 [21.7, 30.9] | 12 / 20 / 31 / 24 / 20 / 50 |
| New + class balancing | 22.6 [18.4, 26.9] | 22.4 ± 6.3 | 21.2 [16.8, 25.2] | 22.9 [18.6, 27.2] | 13 / 23 / 16 / 12 / 12 / 59 |

## Paired comparison (statistical procedure from v3)

| Comparison | Balanced acc. Δ [CI] | Per-seed Δ | Seeds better | Splits up / down / tied | Corrected p | Macro-F1 Δ [CI] |
|---|---|---|---|---|---|---|
| **New vs baseline (primary)** | **−1.6 [−5.4, +2.0]** | −2.9, −1.0, −1.5, −3.5, +0.8 | 1/5 | 8 / 14 / 3 | 0.67 | −2.9 [−6.3, +0.7] |
| New + balancing vs baseline | −2.0 [−6.1, +1.8] | −2.5, −0.4, 0.0, −6.0, −1.2 | 0/5 | 7 / 13 / 5 | 0.59 | −2.2 [−6.1, +1.5] |
| New + balancing vs baseline + balancing | −3.5 [−7.9, +0.7] | −2.7, −2.7, −3.3, −5.4, −3.5 | 0/5 | 6 / 18 / 1 | 0.42 | −4.2 [−8.4, −0.1] |

**Is it consistent across folds? No.**
- The new setting is worse in 14 of 25 outer splits and better in 8 (sign test p = 0.29).
- It is worse in 4 of 5 seeds.
- Per-split differences range from −12.5 to +12.5 points (`analysis/fig_paired_and_recall.png`).

## Intermediate stages

Intermediate stages are pre-declared as SMC, EMCI, MCI and LMCI. Differences are the new model minus the baseline.

| Measure | Baseline | New | Δ [95% CI] |
|---|---|---|---|
| Mean recall, SMC + EMCI + MCI + LMCI | 20.4% | 16.0% | **−4.4 [−8.6, −0.4]** |
| Mean recall, EMCI + MCI + LMCI (MCI spectrum) | 20.2% | 12.3% | **−7.9 [−12.7, −3.3]** |
| SMC recall | 21% | 27% | +6.0 [−2.0, +15.0] |
| EMCI recall | 32% | 21% | **−11.0 [−22.0, −1.0]** |
| MCI recall | 9% | 0% | **−8.8 [−16.2, −2.5]** |
| LMCI recall | 20% | 16% | −4.0 [−10.0, +2.0] |
| CN recall | 12% | 14% | +2.0 [−8.0, +12.0] |
| AD recall | 54% | 60% | +6.0 [−5.0, +17.0] |

- **Intermediate stages get worse, and the intervals exclude 0.**
- **MCI is almost never predicted:** 19 of 580 pooled predictions, none correct. The baseline predicted MCI 53
  times, 7 of them correctly.
- **Predictions shift to the extremes:** more subjects are labelled SMC and AD.

**Confusion matrices** (counts, 5 seeds pooled; rows = true stage; each row totals 100, or 80 for MCI):

| True \ predicted | CN | SMC | EMCI | MCI | LMCI | AD |
|---|---|---|---|---|---|---|
| **Baseline** CN | 12 | 21 | 28 | 8 | 17 | 14 |
| SMC | 23 | 21 | 6 | 10 | 20 | 20 |
| EMCI | 23 | 9 | 32 | 10 | 16 | 10 |
| MCI | 17 | 10 | 11 | 7 | 22 | 13 |
| LMCI | 11 | 17 | 23 | 9 | 20 | 20 |
| AD | 9 | 5 | 7 | 9 | 16 | 54 |
| **New** CN | 14 | 28 | 24 | 4 | 14 | 16 |
| SMC | 17 | 27 | 16 | 9 | 11 | 20 |
| EMCI | 21 | 20 | 21 | 3 | 12 | 23 |
| MCI | 15 | 20 | 16 | 0 | 13 | 16 |
| LMCI | 20 | 14 | 20 | 2 | 16 | 28 |
| AD | 6 | 8 | 16 | 1 | 9 | 60 |

## Interpretation and limitations

- **Removing supervised selection did not help.** It reduced fold-to-fold variability (sd 6.0 vs 8.4) but lowered
  accuracy overall.
  - With 492 mostly uninformative features, TabPFN falls back towards the extremes (SMC, AD) and drops the
    intermediate stages.
  - Supervised selection, though unstable, keeps more of the weak intermediate-stage signal.
- **The pool was not chosen blind.** It was chosen from v3 inner-CV evidence, but v3's held-out result for the same
  pool with SVM-RFE had been seen. This negative result does not depend on that.
- **The comparison cannot detect small effects.** With 116 subjects, the corrected test only detects differences of
  roughly 8–10 points.
- **Site confounding is unchanged** (see the v3 report).

## Files and reproducibility

All outputs are in `derivatives/experiments/tabpfn_local492_v4/`. This is a new folder; nothing earlier was changed.

| What | Where |
|---|---|
| Pre-registration | `PLAN.md` |
| Run log and pinned script hash | `run.log`, `manifest.json` |
| Per-split records (probabilities, predictions, training class counts) | `grid/` |
| Metrics, fold-level results, paired comparisons | `analysis/metrics.csv`, `fold_level.csv`, `paired_comparisons.csv` |
| Per-class recall differences, per-split differences, confusion matrices | `analysis/per_class_recall_differences.csv`, `per_split_differences.csv`, `confusion_matrices.csv` |
| Summary and figures | `analysis/summary.txt`, `fig_confusion.png`, `fig_paired_and_recall.png` |

```
python scripts/experiments/tabpfn_local492_v4/run_local492.py
python scripts/experiments/tabpfn_local492_v4/analyze_local492.py
```

No further experiments were started. The gated locked-test script was not run, and nothing was committed.
