# Pre-registered plan: TabPFN on all 492 mALFF + mReHo features, no supervised selection (v4)

Written 2026-10-21, **before the experiment was run**. This is the single experiment proposed in
`md files/IMPROVEMENT_STUDY_V3_REPORT.md`, section 9. It is an exploratory cross-validation analysis on the fixed
116-subject cohort, not independent confirmation.

**What was already known.** The proposal was justified by v3's inner-CV evidence: mALFF + mReHo had the highest
mean inner-CV score of the 13 settings. However, v3's held-out result for the same pool with SVM-RFE k = 50
(25.6%, +1.0 vs baseline, not reliable) had also been seen. This experiment is therefore not blind to that pool.

## Configuration (fixed)

| Item | Setting |
|---|---|
| Model | TabPFN v2, the baseline's settings: `ModelVersion.V2` defaults, CPU, `random_state` = seed + fold |
| Inputs | All 492 mALFF + mReHo columns of the 984-feature matrix (246 ROIs × 2), raw values, as in the baseline |
| Selection | None. No SVM-RFE, no other supervised or learned step outside TabPFN |
| Fitting | TabPFN's internal preprocessing uses only its training context (the outer training subjects). Outer-test labels are hidden from the fitting code (−1), as in v2 and v3 |
| Splits | The 25 outer splits of the existing TabPFN baseline (fingerprints `3edbea19…` to `8b7f4cd9…`), checked index by index |
| Inner CV | None. Nothing is selected or tuned, so there is no inner loop |
| Secondary (pre-declared) | TabPFN's `balance_probabilities`, applied afterwards from the saved probabilities and each training set's class counts |

## Comparisons and decision rules (fixed)

- **Baseline:** TabPFN, SVM-RFE k = 50 from all 984 features. The records come from `biomarker_ablation_v3/grid/A_all4`,
  which reproduce the original baseline in 25 of 25 splits.
- **Primary comparison:** the new configuration vs the baseline, both without balancing.
- **Secondary comparisons:**
  - new + balancing vs baseline;
  - new + balancing vs baseline + balancing.
- **Statistical procedure** (unchanged from v3):
  - a paired class-stratified subject bootstrap (2,000 resamples) of the seed-averaged difference;
  - seeds agreeing;
  - outer splits up, down and tied, with a sign test;
  - the corrected resampled t-test over the 25 splits (Nadeau–Bengio).
- **"Reliably different"** requires all three:
  - the bootstrap CI excludes 0;
  - at least 4 of 5 seeds agree;
  - corrected p < 0.05.
- **Consistent across folds:** "reliably different" plus improvement in a majority of the non-tied splits.
- **Intermediate stages:**
  - mean recall of SMC, EMCI, MCI and LMCI (all stages between CN and AD);
  - also mean recall of EMCI, MCI and LMCI (the MCI spectrum).

  Each is reported with a paired subject-bootstrap CI, as are the per-class recall differences.
- **Metrics:**
  - balanced accuracy (primary), macro-F1 and accuracy, each with a 95% subject-bootstrap CI;
  - per-class recall;
  - the confusion matrix, 5 seeds pooled;
  - fold-level mean ± sd.

## Stopping criteria

- **One run of 25 splits,** resumable. A crash gets one retry; if it crashes again, the run is reported as failed.
- **No further configuration or tuning** after the results are seen, and no further experiments in this phase.
- **The gated locked-test script is not run.** The 35 "locked" subjects are part of the CV and are not an untouched
  test set.

## Cost

About 45–65 s per split (fit and predict at 492 features), so about 20–30 minutes in total. Peak memory is about
0.9 GB.
