# Pre-registered plan: controlled improvement study (v3)

Written 2026-10-21, after the audit (`audit/audit.log`) and **before any experiment in this folder was run**.
It was revised once, also before any run, to follow the user's updated instruction ("test only these justified
changes"). No held-out result from this experiment had been seen when either version was written.

All analyses are **exploratory cross-validation analyses on the fixed 116-subject cohort**, not independent
confirmation. Earlier project decisions were made after seeing results on these same subjects, including the
35 "locked" subjects (see `md files/TRANSFORMER_NESTED_V2_REPORT.md`, section 2).

## Fixed elements

- **Cohort:** the 116 subjects (one scan each), their labels and the six-class target are unchanged.
- **Outer splits:** the project's 25 outer splits (5 folds × 5 seeds; fingerprints `3edbea19c1ee3c08`,
  `d53ad26a06b7d5c5`, `2a8e70dd0eb80cae`, `d54bf96735be41dd`, `8b7f4cd9e512b26a`).
- **Locked-test script:** the gated `run_test.py` is not touched, and the 35 subjects are not used for tuning.
- **Models:** TabPFN v2 (the baseline settings) and the corrected transformer (the `transformer_nested_v2`
  procedure). No other model family.
- **Inputs:** imaging only in the primary result. The comorbidity experiment is reported separately. MMSE and
  CDR-SB are not used anywhere in this phase.
- **Fitting:** every learned step is fitted on the training part of its own level only:
  - SVM-RFE and its scalers;
  - the transformer scaler;
  - the clinical median fill and scaling;
  - TabPFN's context;
  - the class counts used for weighting or balancing.

  Outer-test labels are hidden from all fitting code, and inner fits also hide their validation labels.

**References for every comparison:**
- **TabPFN baseline:** A_all4, SVM-RFE k = 50; balanced accuracy 24.6%.
- **Transformer baseline:** v2, fixed epoch by inner-CV loss; balanced accuracy 19.1%.

## Experiment matrix

| ID | Change tested | Model | Evaluation | Role |
|---|---|---|---|---|
| A_all4 | None. Reproduces the baseline (all four biomarkers, k = 50) | TabPFN | Outer + 3-fold inner CV | Reference |
| A_all4_k25, A_all4_k100 | Number of selected features | TabPFN | Outer + inner | Feature-selection setting |
| A_mALFF, A_mReHo, A_DCz, A_FC | One biomarker as the candidate pool (k = 50) | TabPFN | Outer + inner | Feature-selection setting |
| A_local, A_conn | mALFF + mReHo (local activity); DC_z + FC (connectivity) (k = 50) | TabPFN | Outer + inner | Feature-selection setting |
| NESTED | In each split, the setting with the best pooled inner-CV balanced accuracy among the 9 above. Ties go to the baseline, then to table order | TabPFN | Outer | **Primary test of feature-selection settings** |
| BAL | `balance_probabilities` (training class shares only), applied to every A configuration's saved probabilities | TabPFN | Outer + inner | Class imbalance |
| unweighted | Plain cross-entropy instead of inverse-frequency class weights | Transformer | Outer | Class imbalance |
| small | d_model 8 (was 16), dropout 0.5 (was 0.3), weight decay 0.1 (was 0.01): 2,197 parameters instead of 7,197 | Transformer | Outer | Regularisation |
| chosen | The per-split NESTED feature setting, applied to the transformer | Transformer | Outer | Feature-selection setting |
| comorb8 | Imaging + 8 comorbidity variables (age, sex, APOE4 count, GDS total, cardiovascular, endocrine-metabolic, psychiatric, smoking). Transformer with the existing gated fusion; TabPFN from the existing `cv116_tabpfn_svmrfe50_comorb8` and `cv116_tabpfn_comorb8` runs on the same folds | Both | Outer | **Separate experiment, not imaging-only** |
| P_site1–5, P_glob1–3 | Baseline TabPFN with training labels shuffled within site, or overall | TabPFN | Outer | Confound diagnostic |
| L_frac50, L_frac75 | Baseline TabPFN on 50% or 75% of each training set (stratified) | TabPFN | Outer | Learning curve |

**Why the regularisation test is justified:** in v2, the mean inner-validation loss was lowest at epoch 12
(1.794) and rose to 2.299 by epoch 60, while training loss fell to 1.33. That is a growing training–validation
gap, which meets the plan's overfitting condition.

**Not run, with reasons:**
- **Balanced mini-batches:** not in the existing implementations.
- **No-selection TabPFN runs:** not on the approved list, and about 2 hours at 246–492 features with a risk of
  running out of memory.
- **ComBat:** existing `leakage_safe_v1` results (15 splits) are reported instead.
- **Broader searches:** none.

## Decision rules (fixed now)

- **"Reliably different"** requires all three:
  - the paired subject-bootstrap 95% CI of the balanced-accuracy difference excludes 0;
  - at least 4 of 5 seeds agree;
  - the corrected resampled t-test over the 25 splits (Nadeau–Bengio; test/train ratio about 23/93) gives
    p < 0.05.
- **R1, feature-selection settings:**
  - NESTED replaces the TabPFN baseline only if it is reliably better.
  - The individual settings are exploratory (Holm-adjusted p-values) and are never chosen by their held-out
    results.
- **R2, class imbalance:**
  - Balancing (TabPFN) is adopted only if reliably better on the baseline.
  - The transformer keeps inverse-frequency weighting unless "unweighted" is reliably better.
  - Per-class recall changes are reported whatever the outcome.
- **R3, regularisation:** if "small" and the v2 transformer are not reliably different, the simpler model
  ("small") is retained, as instructed. If one is reliably better, that one is retained.
- **R4, site confound:** the baseline counts as exceeding what site structure allows only if both hold:
  - its balanced accuracy is above all 5 within-site-permuted runs;
  - it is above their mean by more than 2 points.
- **R5, comorbidities:** reported beside the imaging-only results, never merged into the primary imaging result.
- **Retention:** whatever does not pass these rules is reported as a negative result, and the current reference
  is retained.

## Stopping criteria

- **The matrix above is final.** Nothing is added after any result is seen.
- **Stop everything** if:
  - any leakage check fails;
  - A_all4 does not reproduce the existing baseline in 25 of 25 splits (automatic);
  - the transformer variant code does not reproduce v2 (automatic gate).
- **Crashes:** a configuration that crashes twice is dropped and reported as failed.
- **Budget:** about 2.5 hours of compute. Anything still running after 5 hours is stopped and reported as
  incomplete.
- **After this matrix, no further experiments in this phase.**

## Order and estimated cost

Measured on one split with 4 threads.

| Order | Run | Estimated time |
|---|---|---|
| 1 | TabPFN A configurations (baseline first, then gated) | about 40–60 min |
| 2 | P runs | about 17 min |
| 3 | L runs | about 3 min |
| 4 | Transformer variants small, unweighted, comorb8, chosen (2 threads), run after the TabPFN runs | about 60–70 min |

## Addendum 1: feature-count sweep (2026-10-21)

Added at the user's request while the main matrix was running. It was written **before any held-out result of
this experiment was inspected**: only inner-CV scores had been printed to the run log. The main matrix above is
unchanged.

- **New configurations:** TabPFN v2 on all four biomarkers with SVM-RFE k = 10, 20, 30 and 75. They are run as
  K_all4_k10, K_all4_k20, K_all4_k30 and K_all4_k75 by `run_k_sweep.py`, with outer folds plus 3-fold inner CV.
  k = 50 is A_all4 (the baseline).
- **Selection:** SVM-RFE is selected inside each training partition, exactly as for the A configurations.
- **NESTED_K:** in each outer split, the k among {10, 20, 30, 50, 75} with the best pooled inner-CV balanced
  accuracy. Ties go to the baseline (k = 50), then to the smaller k.
- **Decision rule:** NESTED_K replaces the baseline only if it is reliably better, by the same three criteria as
  R1. With two primary nested procedures (NESTED and NESTED_K), their corrected-t p-values are Holm-adjusted
  together. The individual k values are exploratory.
- **When it runs:** only after the main TabPFN and transformer job has finished, never in parallel. The estimated
  runtime is about 35 minutes.
- **Afterwards:** no further runs in this phase. The "single most promising next experiment" is proposed in the
  report, not run.
