# Ordinal Transformer for Alzheimer's Stage Prediction (v1)

Written 2026-10-21. This is a controlled comparison with the corrected six-class transformer (v2) on the same 25
outer splits of the fixed 116-subject cohort. It is an exploratory cross-validation analysis, not independent
confirmation; the 35 "locked" subjects are part of the CV, and the gated locked-test script was not run. No MMSE or
CDR-SB is used.

## Recommendation: REJECT

The pre-registered rules reject the ordinal transformer, so it is not adopted. The six-class transformer (v2) stays
the transformer reference.

- **Intermediate stages collapse.** Under the primary argmax rule, the model predicts only CN and AD. Recall for
  SMC, EMCI, MCI and LMCI drops from 16.4% on average to **0%**: −16.4 points (95% CI −19.6 to −13.3), worse in 0 of
  5 seeds and in 23 of 25 splits (corrected p = 0.003).
- **The small balanced-accuracy gain is an artefact.** Balanced accuracy is +1.3 points (CI −1.9 to +4.4, p = 0.76),
  produced entirely by high CN and AD recall. Macro-F1 falls from 17.7% to 10.5%.
- **Ordinal error does not improve.** With argmax, mean absolute error is *worse* (2.25 vs 1.87 stages). The median
  rule has lower error (1.47), but only because it always predicts a middle stage: an "always EMCI" rule gives
  1.52 stages and the same within-one share (48.3%).
- **No bug was found.** All correctness, gate, label-order and split checks pass. The model simply learned no usable
  ordering: its severity score is nearly constant across subjects (correlation with the true stage −0.06).

## 1. Hypothesis

An ordinal objective improves held-out balanced accuracy and/or macro-F1 over the six-class transformer under
matched conditions, while improving or preserving intermediate-stage recall. Ordinal error is measured separately:
fewer distant errors alone would not show better six-class classification.

**Outcome:** not supported.

## 2. Implementation

- **Label order:** verified in the data files and code, and confirmed by you: **CN = 0 < SMC = 1 < EMCI = 2 < MCI =
  3 < LMCI = 4 < AD = 5**. The brief's alternative (LMCI before MCI) conflicted with the repository encoder and was
  not used. The same order is used in all labels, predictions, confusion matrices and metrics.
- **Ordinal head:** `CumulativeLinkHead` (`scripts/experiments/ordinal_transformer_v1/ordinal_head.py`).
  - It reuses the repository's tested cumulative-link parameterisation (`heads.OrdinalHead`): a severity score
    g(H) = `Linear(16→1)`, and five thresholds kept in strictly increasing order by softplus increments.
  - The model is P(y ≤ k) = sigmoid(τ_k − g), so the six class probabilities are differences of a monotone CDF:
    they are valid and sum to 1.
- **Loss:** the exact cumulative-link negative log-likelihood, −log P(y = k), computed stably as
  logsigmoid(b) + logsigmoid(−a) + log(−expm1(a − b)).
  - Each sample is weighted by inverse class frequency from its inner-training set only, normalised like the
    reference's weighted cross-entropy.
  - The old auxiliary multiclass term (`ordinal_fixed`) was **not** used, so the head and loss are the only change.
- **Initialisation:** thresholds at logit(k/6), giving uniform initial predictions like the reference. No labels
  were used.
- **Parameters:** 7,117, against 7,197 for the reference. The encoder is identical.
- **Decision rules:** argmax of the six probabilities (primary, as the reference) and the median stage (secondary).

**Verification before training**

| Check | Result |
|---|---|
| Correctness tests (`tests/test_results.log`) | 11 of 11 pass: label order; valid probabilities and monotone CDF and thresholds, including extreme scores; stable log-probabilities, finite where the naive form gives −inf; exact weighted loss; gradient sign for every stage; finite-difference `gradcheck`; uniform initialisation; edge cases; median rule; recovery of a known ordering from synthetic data (Spearman 0.93, accuracy 0.63 against an oracle's 0.64) |
| Errors found while testing | 3, all in the test code (an inverted sign check, a missing `.detach()`, a wrong expected value), fixed before the passing run. None was in the head |
| Smoke test (one split, 3 epochs) | output 24 × 6, rows sum to 1, thresholds increasing, loss falling from about 1.80, memory 0.44 GB |
| Reproduction gate | the training code with the reference head reproduces v2's curves and early-stopping epoch exactly |
| Other checks | split identity (25 of 25 identical to v2) and outer labels identical |

## 3. Evaluation protocol

- **Splits:** the same 25 outer splits as transformer v2 (5 folds × 5 seeds), with identical subject assignments.
  There is one scan per subject.
- **Inside each outer split:** the same 4 inner folds and seeds as v2. Each inner model's scaler and SVM-RFE (k = 50,
  all four biomarkers) are fitted on its inner-training subjects only, and outer-test labels are hidden (−1) from all
  fitting and selection code.
- **Epoch choice:** the epoch with the lowest mean inner-validation loss, which is v2's primary rule. No epoch,
  threshold or variant was chosen with held-out labels.
- **Statistics:** the project's procedure.
  - Seed-pooled out-of-fold metrics are averaged over the 5 seeds.
  - 95% CIs come from 2,000 class-stratified subject bootstraps.
  - Paired comparisons also report the per-seed differences, the number of splits better, worse and tied, and the
    corrected resampled t-test (Nadeau–Bengio).
  - "Reliably" better or worse needs all three: the CI excludes 0, at least 4 of 5 seeds agree, and corrected
    p < 0.05.
- **Compute:** 17.6 minutes.

## 4. Results (25 identical outer splits)

| Model | Balanced acc. [CI] | Fold mean ± sd | Macro-F1 [CI] | Accuracy [CI] | Mean abs. error, stages [CI] | Within one stage [CI] | Recall CN / SMC / EMCI / MCI / LMCI / AD (%) |
|---|---|---|---|---|---|---|---|
| **Transformer v2** (reference) | 19.1 [16.2, 22.1] | 19.2 ± 6.3 | **17.7** [14.8, 20.3] | 19.5 [16.6, 22.4] | 1.87 [1.74, 2.00] | 47.4 [43.3, 51.7] | 11 / 19 / 23 / 8 / 16 / 38 |
| **Ordinal, argmax** (primary) | 20.3 [17.7, 22.8] | 20.3 ± 5.9 | 10.5 [9.0, 11.9] | 21.0 [18.3, 23.6] | 2.25 [2.08, 2.42] | 38.8 [34.8, 43.1] | 56 / **0 / 0 / 0 / 0** / 66 |
| Ordinal, median rule (secondary) | 18.9 [15.7, 22.0] | 19.1 ± 7.2 | 9.0 [7.4, 10.6] | 17.8 [15.0, 20.5] | 1.47 [1.42, 1.53] | 48.3 [45.2, 51.4] | **0 / 0** / 61 / 52 / **0 / 0** |
| TabPFN baseline (context only; different protocol) | 24.6 [20.5, 29.0] | 24.6 ± 8.4 | 23.5 [19.2, 27.5] | 25.2 [20.9, 29.5] | 1.68 [1.53, 1.86] | 50.5 [44.5, 56.0] | 12 / 21 / 32 / 9 / 20 / 54 |

**Per-class precision / recall / F1** (%, 5 seeds pooled):

| Model | CN | SMC | EMCI | MCI | LMCI | AD |
|---|---|---|---|---|---|---|
| Transformer v2 | 12 / 11 / 12 | 17 / 19 / 18 | 18 / 23 / 20 | 29 / 8 / 12 | 15 / 16 / 15 | 30 / 38 / 34 |
| Ordinal, argmax | 19 / 56 / 28 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 24 / 66 / 35 |
| Ordinal, median rule | 0 / 0 / 0 | 0 / 0 / 0 | 20 / 61 / 30 | 15 / 52 / 24 | 0 / 0 / 0 | 0 / 0 / 0 |

**Confusion matrices** (row %, 5 seeds pooled, verified stage order): `analysis/fig_confusion.png`; counts are in
`analysis/confusion_matrices.csv`.
- **Ordinal, argmax:** every true stage is predicted as CN or AD: CN 56% / AD 41% of CN subjects, down to CN 32% /
  AD 66% of AD subjects.
- **Median rule:** every true stage is predicted as EMCI or MCI.

## 5. Paired comparison: ordinal (argmax) minus transformer v2

| Metric | Δ [95% CI] | Per-seed Δ | Seeds better | Splits better / worse / tied | Corrected p | Verdict |
|---|---|---|---|---|---|---|
| Balanced accuracy | +1.3 [−1.9, +4.4] | +1.9, +5.0, −2.3, +1.7, 0.0 | 3/5 | 11 / 8 / 6 | 0.76 | not reliable. Without the 3 most favourable splits the mean is −0.4: **driven by a few folds** |
| Macro-F1 | −7.3 [−9.8, −4.5] | all 5 negative | 0/5 | 11 / 14 / 0 | 0.40 | lower; CI excludes 0, but the corrected test does not confirm it |
| Accuracy | +1.6 [−1.6, +4.7] | | 3/5 | 11 / 8 / 6 | 0.69 | not reliable |
| Mean abs. error (lower is better) | +0.39 [+0.22, +0.55] | all 5 worse | 0/5 | 5 / 19 / 1 | 0.14 | worse; CI excludes 0, but the corrected test does not confirm it |
| Within one stage | −8.6 [−13.5, −3.6] | all 5 negative | 0/5 | 5 / 20 / 0 | 0.15 | lower; not confirmed by the corrected test |
| **Intermediate recall** (SMC, EMCI, MCI, LMCI) | **−16.4 [−19.6, −13.3]** | all 5 negative | 0/5 | 0 / 23 / 2 | **0.003** | **reliably worse** |

**Per-class recall change** (ordinal − v2): CN +45 [+34, +57], AD +28 [+18, +37], SMC −19, EMCI −23, MCI −7.5, LMCI −16
(all four CIs below 0).

**Median rule (secondary):**
- Ordinal error is lower: −0.39 stages [−0.51, −0.28], better in 5 of 5 seeds and 23 of 25 splits, corrected
  p = 0.039.
- Intermediate recall is higher (+12.0, not confirmed by the corrected test). Balanced accuracy is unchanged
  (−0.2) and macro-F1 is lower (−8.7).
- These gains equal what any constant middle-stage prediction gets. "Always EMCI" scores 1.52 stages and 48.3%
  within one stage. So they are not evidence of learned ordering.

## 6. Training behaviour

| | Transformer v2 | Ordinal |
|---|---|---|
| Lowest mean inner-validation loss (epoch) | 1.794 (12) | 1.788 (11) |
| Uniform-guess loss, ln 6 | 1.792 | 1.792 |
| Inner-validation loss at epoch 60 | 2.299 | 2.003 |
| Training loss at epoch 60 | 1.32 | 1.52 |
| Training balanced accuracy at epoch 60 | 46.0% | 31.6% |
| Selected epoch: median (IQR, range) | 11 (7–12, 4–28) | 12 (9–18, 1–41) |
| Overfits after the minimum | yes | yes, more slowly |

**What this shows** (`analysis/fig_curves_epochs.png`):
- Neither model's validation loss gets meaningfully below the uniform-guess level.
- The ordinal head overfits more slowly, but it starts from, and stays near, a nearly flat prediction.

## 7. Why it fails: not a bug

`analysis/collapse_diagnosis.json` (from `diagnose_collapse.py`):

| Quantity | Value |
|---|---|
| Thresholds at the start | −1.61, −0.69, 0.00, 0.69, 1.61 |
| Thresholds at epoch 60 (mean of 100 inner models) | −1.61, −0.67, 0.06, 0.78, 1.73, with sd ≤ 0.06 (barely moved) |
| Average top probability of the bagged predictions | 0.187 (a uniform guess is 0.167) |
| Highest probability any intermediate stage reaches | about 0.17–0.22 |
| Highest probability CN and AD reach | 0.36 and 0.32 |
| Spread of the implied severity score across subjects (sd of logit P(CN)) | 0.18 |
| Correlation of that severity score with the true stage | −0.06 |

With an almost constant severity score, a cumulative-link model's interior stages cannot win an argmax, so argmax
returns CN or AD; the median rule returns the middle stages. This is the expected behaviour of the model class when
the imaging signal gives it nothing to order. The head was correct on synthetic ordered data (section 2), so the
failure reflects the data's weak stage signal, not the implementation.

## 8. Decision (PLAN.md rules)

| Pre-registered criterion | Met? |
|---|---|
| All tests, gate, label-order and split checks pass | Yes |
| Balanced accuracy or macro-F1 reliably better | **No** (+1.3, p = 0.76; macro-F1 −7.3) |
| Not driven by a few folds | **No** (the balanced-accuracy gain becomes −0.4 without the 3 best splits) |
| Intermediate recall falls by no more than 3 points, with CI upper bound ≥ 0 | **No** (−16.4, CI upper bound −13.3) |
| Ordinal-error benefit, if claimed, reliably better | Not claimed. Argmax is worse; the median rule's gain matches a constant prediction |

**Decision: REJECT.** Intermediate-stage recall is reliably worse. The higher balanced accuracy (not significant) is
produced by predicting only the two extreme stages, so it does not show better six-class classification. **Overall
classification accuracy did not improve**, and no claim is made that it did.

## 9. Limitations

- **One ordinal approach** (the cumulative-link model with its exact likelihood) was tested, as planned. Other
  ordinal formulations were not, so this rejects this implementation under this data, not ordinal learning in
  general.
- **The data has very little signal:** no transformer variant's validation loss gets below a uniform guess. In that
  regime the experiment cannot separate "ordinal structure does not help" from "there is no stage signal to order".
- **MCI vs LMCI order:** in ADNI the MCI label and LMCI use similar memory cut-offs, so their true order is uncertain.
  The repository order was used, as you confirmed.
- **Small sample:** 116 subjects; the corrected test can confirm only large differences (about 5–12 points).
- **TabPFN is context only:** its tuning and evaluation protocol differ, so this is not a controlled architecture
  comparison.

## 10. Files and reproducibility

All results are in `derivatives/experiments/ordinal_transformer_v1/`. This is a new folder; existing results are
unchanged.

| What | Where |
|---|---|
| Pre-registration, configuration and exact label mapping | `PLAN.md` |
| Manifest (script hash, settings, label order) | `manifest.json` |
| Run log | `run.log` |
| Tests | `tests/test_results.json`, `tests/test_results.log` |
| Smoke test | `smoke/smoke_check.json` |
| Per-split records: out-of-fold predictions and probabilities for every rule, selected epochs, per-epoch curves, final thresholds | `grid/cv5_seed*_fold*.json` |
| Metrics, per-class precision/recall/F1, fold-level results | `analysis/metrics.csv`, `per_class_precision_recall_f1.csv`, `fold_level.csv` |
| Confusion matrices, paired comparisons, recall differences | `analysis/confusion_matrices.csv`, `paired_comparisons.csv`, `per_class_recall_differences.csv` |
| Selected epochs, curves, fit diagnostics, collapse diagnosis | `analysis/selected_epochs.csv`, `curves_*.csv`, `fit_diagnostics.csv`, `collapse_diagnosis.json` |
| Summary and figures | `analysis/summary.txt`, `fig_curves_epochs.png`, `fig_confusion.png`, `fig_recall_error.png` |

```
python scripts/experiments/ordinal_transformer_v1/test_ordinal.py
python scripts/experiments/ordinal_transformer_v1/run_ordinal.py --smoke
python scripts/experiments/ordinal_transformer_v1/run_ordinal.py
python scripts/experiments/ordinal_transformer_v1/analyze_ordinal.py
python scripts/experiments/ordinal_transformer_v1/diagnose_collapse.py
```

No further experiments were started, and nothing was committed.
