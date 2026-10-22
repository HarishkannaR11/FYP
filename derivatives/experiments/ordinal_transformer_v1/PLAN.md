# Pre-registered plan: ordinal transformer v1

Written 2026-10-21, after the head's correctness tests passed (`tests/test_results.log`, 11 of 11) and **before any
training run**. It is an exploratory cross-validation analysis on the fixed 116-subject cohort, not independent
confirmation. The 35 "locked" subjects are part of the CV, and the gated `run_test.py` is not run.

## Hypothesis

An ordinal objective improves held-out balanced accuracy and/or macro-F1 over the six-class transformer under matched
conditions, while improving or preserving intermediate-stage recall. Ordinal error (stage steps) is measured
separately: fewer distant errors alone does not show better six-class classification.

## Label mapping

The repository's encoder was verified in data files and code, and the user confirmed this order:

| Stage | CN | SMC | EMCI | MCI | LMCI | AD |
|---|---|---|---|---|---|---|
| Index | 0 | 1 | 2 | 3 | 4 | 5 |

The ordinal model, the ordinal-error metrics and all confusion matrices use this order.

## The one change

Only the output head and its loss change. Everything else is the reference: transformer v2
(`transformer_nested_v2`), C0 settings.

| | Reference (v2) | Ordinal v1 |
|---|---|---|
| Head | `Linear(16→6)`, softmax | `CumulativeLinkHead`: `Linear(16→1)` severity score + 5 monotone thresholds (`heads.OrdinalHead` parameterisation) |
| Loss | weighted cross-entropy (NLL of the softmax) | weighted exact cumulative-link NLL, −log P(y = k), computed in a numerically stable way |
| Class weights | inverse frequency of each inner-training set | same |
| Initial output | uniform (zero-bias softmax) | uniform (thresholds at logit(k/6); no labels used) |
| Head parameters | 102 | 22 (the encoder is identical) |
| Decision rule | argmax | **argmax of the six probabilities (primary)**; median stage (secondary) |

**Identical to the reference:**
- the 25 outer splits;
- the 4 inner folds per split and all seeds (`bag_seed = seed·100 + fold`; torch seed `bag_seed·10 + j`; SVM-RFE
  seed `seed + fold`);
- per-inner-model scaler and SVM-RFE (k = 50, all four biomarkers) fitted on inner-training subjects only;
- outer-test labels hidden (−1) from all fitting code;
- the encoder (d16, 2 heads, 1 layer, dropout 0.3);
- AdamW (lr 1e-3, weight decay 1e-2), batch 8, gradient clip 1.0, 60 epochs logged.

**Epoch selection (primary, as in v2):** per outer split, the epoch minimising the mean inner-validation loss of the 4
inner models, i.e. early stopping on inner-validation loss. The other v2 rules (per-model early stopping on
inner-validation balanced accuracy, the pooled-balanced-accuracy epoch, epoch 60) are recorded for diagnosis only.

**Gate:** the training code, run with the reference head, must reproduce the v2 curves exactly before any ordinal
run. A one-split smoke test (3 epochs) checks output dimensions, losses, probabilities and memory first.

## Metrics

Each is reported for both models, primary rule:
- balanced accuracy (primary);
- macro-F1;
- accuracy;
- per-class precision, recall and F1;
- confusion matrices;
- mean absolute error in stage steps;
- the share of predictions within one stage of the true stage;
- fold mean ± sd;
- 95% CIs from the class-stratified subject bootstrap (2,000 resamples) of seed-averaged metrics;
- paired per-split differences and the project's corrected resampled t-test;
- training and validation loss curves;
- the distribution of selected epochs.

TabPFN (24.6% balanced accuracy) is shown as context only, because its protocol differs.

## Decision rules (fixed now)

- **"Reliably better"** on a metric means all three:
  - the paired bootstrap CI excludes 0 in the favourable direction;
  - at least 4 of 5 seeds agree;
  - corrected-t p < 0.05.
- **ADOPT** only if all of these hold:
  1. All tests, the gate, the label-order check and the split-identity check pass.
  2. Balanced accuracy or macro-F1 is reliably better.
  3. The result is not driven by a few folds: improved in a majority of non-tied splits, and the mean per-split
     difference stays positive after removing the 3 most favourable splits.
  4. Intermediate-stage recall (mean of SMC, EMCI, MCI, LMCI) falls by no more than 3 points, and its CI upper
     bound is ≥ 0.
  5. If a benefit in ordinal error is claimed, the reduction in mean absolute error is reliably better.
- **REJECT** if any of these holds:
  - balanced accuracy or macro-F1 is reliably worse;
  - intermediate-stage recall is reliably worse (CI entirely below 0);
  - a correctness, label or leakage check fails.
- **INCONCLUSIVE** otherwise, including favourable but non-significant results. A non-significant result is not
  evidence of equivalence.
- **Ordinal error is judged separately.** If it improves but balanced accuracy does not, the report says so and does
  not claim better classification.

## Stopping criteria

- **One full run of 25 splits,** resumable; a crash gets one retry.
- **No tuning against held-out folds,** no second variant, and no further experiments.
- **If a code bug, label-order problem or leakage issue is found during or after the run,** it is documented and the
  affected metrics are not interpreted.

## Cost

About 40 s per split, about 17–20 minutes in total, about 0.5 GB of memory. The run stops and reports if memory is
insufficient.
