# Pre-declaration: three-group task (coarse_tasks_v1)

Written on 2026-10-22 BEFORE any three-group model was trained or scored. Nothing in this file may be
changed after results exist; if something must change, add a dated amendment at the bottom.

## Why
Six-class imaging-only accuracy is low (TabPFN v2 24.6 %, chance 16.7 %). Re-scoring the stored six-class predictions
after the fact gave 48.7 % for a three-group merge, but (a) that was chosen after seeing data, (b) a rule that uses only
the scanner site is said to score about as well, and (c) the grouping was not declared. This experiment tests the
three-group task properly, with controls, on the same 116 subjects and the same outer splits. No new imaging data.

## Task (declared)
Three groups, built from the six stages (0 CN, 1 SMC, 2 EMCI, 3 MCI, 4 LMCI, 5 AD):

| group | stages | n subjects |
|---|---|---|
| G0 | CN, SMC | 40 |
| G1 | EMCI, MCI, LMCI | 56 |
| G2 | AD | 20 |

Rationale (clinical, fixed in advance): SMC subjects are cognitively normal; EMCI/MCI/LMCI are the mild-cognitive-
impairment family; AD is dementia. No other grouping will be run or reported as a result.
Labels are collapsed AFTER the outer folds are built, so fold membership is identical to the six-class run
(same fingerprints as `cv116_tabpfn_svmrfe50.json`).

## Models and controls (declared; nothing else will be added)
| id | what | needs |
|---|---|---|
| M1 | TabPFN v2, 50 SVM-RFE features, SVM-RFE refitted on the THREE-group labels inside each outer training set | TabPFN v2 weights |
| C1 | Site-only rule (no brain data): for a test subject, the training-fold class distribution at the same site, divided by the training class prior (Laplace +1); unseen site -> random tie-break (fixed seed) | none |
| C2 | M1 with training labels shuffled GLOBALLY (null model; expected = chance) | TabPFN v2 weights |
| C3 | M1 with training labels shuffled WITHIN SITE (keeps the site-label association, destroys the brain-label association) | TabPFN v2 weights |
| R  | The existing six-class TabPFN result, collapsed after the fact: reported ONLY as a labelled post-hoc reference, never as a headline | stored predictions |

Everything is fitted on training subjects only (selection, TabPFN context). Outer-test labels are never used before
prediction. The gated locked-test script is not touched; the 35 earlier-designated test subjects remain inside the CV
and are not independent.

## Metrics (declared)
Primary: balanced accuracy over the three groups (chance 33.3 %). Secondary: macro-F1, accuracy, per-class recall,
confusion matrix, 95 % stratified subject bootstrap interval (same method as `scripts/report/report_metrics.py`,
2000 resamples, seed-averaged), seed SD, fold SD.

## Decision rules (declared)
* The three-group imaging result counts as imaging information beyond scanner site ONLY IF all hold:
  1. M1 balanced accuracy exceeds C1 (site-only) with a paired bootstrap interval that excludes zero and at least
     4 of 5 seeds agreeing;
  2. M1 exceeds C3 (within-site shuffle) by the same rule;
  3. M1 interval excludes chance (33.3 %).
* If any condition fails, the report states that the three-group result is not shown to exceed what site alone
  explains. No grouping, model, feature set or threshold will be changed afterwards to rescue the result.
* Intervals describe this fixed 116-subject cohort; they are not independent-test intervals.

## Where it will be reported
Report Chapter 4 (new subsection, "Pre-declared three-group task"), a one-line pointer in the abstract and Chapter 6,
and the poster results panel as a separate small table. The six-class result remains the primary result.

## Amendments
(none)
