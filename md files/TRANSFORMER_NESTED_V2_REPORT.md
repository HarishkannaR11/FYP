# Transformer with Nested Feature Selection (v2): Report

Written 2026-10-21. Imaging only (rs-fMRI, SVM-RFE all4 k = 50). No MMSE, CDR or comorbidity variable is used
anywhere in this report. Nothing has been committed or published.

## 1. Summary

- **The leak is fixed and verified.** Each inner model now fits its own scaler and SVM-RFE on its inner-training
  subjects. As a result, inner-validation accuracy now matches held-out accuracy when no epoch is selected (gap
  +0.4 points, 95% CI −0.9 to +1.6). Before the fix the gap was +4.3 points (CI +2.7 to +6.2).
- **The corrected transformer does not improve reliably.** With the primary checkpoint rule (a fixed epoch chosen by
  inner-CV loss), it scores **19.1%** balanced accuracy against 19.6% for the old C0 (difference −0.5, CI −4.5 to
  +3.5). With nested early stopping it scores 17.9%.
- **TabPFN v2 (k = 50) remains better on identical folds:** 24.6% balanced accuracy and 23.5% macro-F1. It beats
  both corrected transformer variants: by +5.5 points (CI +0.6 to +10.7) and +6.8 (CI +1.8 to +11.8). TabPFN
  wins in 5 of 5 seeds in both comparisons.
- **The fix also exposes why the transformer fails.** On truly unseen validation subjects, its mean validation loss
  never falls below the uniform-guess level (minimum 1.794 vs ln 6 = 1.792). In C0, the apparent improvement down
  to 1.766, and the "best epoch ≈ 30" it suggested, came from the leak. The selected epoch is now a median of 11,
  and only 1 of 25 splits chose an epoch between 25 and 35.

## 2. The 35 "locked-test" subjects: status before any result

**This section answers your question, and the answer is that the locked test set is not an untouched hold-out.**
The audit flag is not harmless.

### Two different protocols

| | 116-subject cross-validation | Locked-test protocol (70/30) |
|---|---|---|
| Subjects | All 116 | 81 training pool / 35 test (split fingerprint `6f478fbcf56364e2`) |
| How it is used | 5 folds × 5 seeds. Every subject is tested once per seed and used for training in the other 4 folds | Select and train on the 81 only, then evaluate once on the 35 (`leakage_safe_v1/run_test.py`) |
| Role in this report | **All results here** (v1, v2, TabPFN) | Not used. `run_test.py` has never been run (no results folder exists) and stays gated behind `--confirm-final-test` |

**The 35 locked subjects are inside the 116-subject CV.** In every seed, each of them is an outer-test subject once
and a training subject 4 times. Every 116-CV comparison in this project has therefore used them.

### What the 35 subjects have already been used for

This was found by scanning every saved result file (inventory in section 8).

| When | What | Evaluated on the 35? |
|---|---|---|
| 2026-10-15 | Early ordinal transformer (`results_ordinal.json`, faulty decoder run) | Yes, once. The file records it as "NOT pristine" |
| 2026-10-15 | Frozen DC-only MLP (`final_heldout/`) | Yes, once: 20.0% balanced accuracy |
| 2026-10-20 | Feature-selection study, step 9: PCA k10, L1 k10, QUBO k8 | Yes, 3 methods, once. SVM-RFE was **not** among them |
| 2026-10-20 to 21 | 26 `split70_*` runs: transformer, logistic regression, TabPFN, Reptile meta-learning and MMSE/CDR ceilings | Yes: 4 imaging-only and 22 with clinical inputs. Results were reported in `TRANSFORMER_AND_COMORBIDITY_RESULTS.md` |
| 2026-10-20 | 80/20 split (seed 20261020), 3 runs | 7 of its 23 test subjects are locked subjects |
| 2026-10-20 to 21 | All 116-CV runs (TabPFN, transformer, meta-learning, comorbidity, learning curve, leakage-safe v1, diagnostics v1, this v2) | Inside the folds, as above |

**In total, the 35 subjects have had 31 separate model evaluations, and their results have been seen.**

### Were they used for model selection or tuning?

- **The settings this report relies on were chosen without them:**
  - The transformer's settings come from the 2026-10-15 capacity and learning-rate study, which used CV on the
    81-subject pool only (its test fields are empty and `final_test` is false). The 2-head variant was fixed later,
    in the SVM-RFE plan.
  - SVM-RFE (all4) was chosen from the feature-selection study's CV on the 81-subject pool, and SVM-RFE was never
    evaluated on the 35.
  - TabPFN k = 50 was fixed in advance to match the transformer.
  - v2's checkpoint rules were declared in the script before the run, and the script's hash is pinned in the
    manifest.
- **Project-level decisions were made after their results were visible:**
  - The 70/30 results above.
  - The 116-CV comparisons, which include the 35, motivated the v1 diagnosis, this v2 design, the choice of clinical
    feature sets, and the conclusion that TabPFN is the strongest imaging model.
  - The feature-selection study also showed 3 test results before SVM-RFE was chosen. An indirect influence cannot
    be ruled out from the files.
- **A correction to earlier documentation:** `TRANSFORMER_AND_COMORBIDITY_RESULTS.md` (line 53) says the 35
  subjects "were never in the feature-selection study". They were excluded from its CV, but they were its step 9
  held-out test.

**Consequence:** running `run_test.py` now would not give an independent confirmation and should not be reported as
one. Deciding the protocol is yours or your supervisor's. Three options:

1. **Make nested 116-subject CV the primary and only estimate (recommended).** Report the 70/30 history openly and
   drop the "locked test" claim.
2. **Use a fresh hold-out.** This needs subjects never used before. Only 13 such processed subjects exist (6 CN,
   6 EMCI, 1 SMC), which is too few and too unbalanced for a 6-stage test, so new data would be needed.
3. **Keep the 70/30 evaluation,** but label it as a previously viewed subset, with this history.

## 3. What changed in v2, and leakage checks

| | C0 (v1) | v2 |
|---|---|---|
| Scaler and SVM-RFE | Fitted once on the whole outer training set (about 93 subjects, including each inner-validation fold) | Fitted separately for each inner model, on its inner-training subjects only (about 70) |
| Outer-test labels | Not used for training | Removed from the label array (set to −1) before any training or selection code runs |
| Everything else | C0 settings | Same: same outer folds, inner folds, seeds and settings (d16, 2 heads, 1 layer, dropout 0.3, lr 1e-3, wd 1e-2, 60 epochs) |

**Checkpoint rules,** all declared before the run:

| Rule | How the epoch is chosen | Status |
|---|---|---|
| Nested early stopping (ES) | Each inner model keeps its best inner-validation epoch (patience 15) | Compared |
| Fixed epoch (FIXED) | One epoch per split: the minimum of mean inner-validation loss | **Primary** |
| FIXED_BAL | One epoch: the best pooled inner balanced accuracy | Sensitivity check |
| Epoch 60 | No selection | Diagnostic |

In every rule, the prediction is the average of the 4 inner models.

**Leakage audit** (`audit/audit.log`): 17 checks, 16 pass. The one flag is the locked-test overlap (section 2).
- **Outer folds:** identical to the TabPFN baseline in 25 of 25 splits. Inner folds partition each training set, and
  every inner-validation fold contains at least 3 subjects of each stage.
- **Inputs are per-subject:**
  - mALFF and mReHo have an ROI mean of exactly 1 for every subject;
  - DC_z has mean 0 and sd 1;
  - FC strength equals the subject's own mean |FC|, and FC is symmetric with a zero diagonal;
  - the attention mask is the atlas centroid 15-nearest-neighbour graph exactly.

  No statistic is computed across subjects before the model.
- **Perturbation tests** (one split):
  - The inner scaler and SVM-RFE do not change when the inner-validation and outer-test rows and labels are changed.
  - Training does not change when inner-validation labels change. They affect only the checkpoint choice, as
    intended.
  - Selection and training do not change when outer-test features change.
  - Training refuses to start if outer-test labels are not hidden.
  - Given C0's inputs, the training loop reproduces C0 exactly (same best epoch and curves), so where selection is
    fitted is the only change.
- **TabPFN baseline:** it fits SVM-RFE and the model on the outer training subjects only (`train_tabpfn.py` lines
  83 and 93). It needs no inner validation. Its predictions match the separately perturbation-tested leakage-safe
  v1 code exactly in 15 of 15 splits.

## 4. Results on identical outer folds (25 splits, 5 seeds; held-out = outer test fold)

| Model | Accuracy | Balanced acc. [95% CI] | Macro-F1 | Seed sd | Fold sd (range) | Recall CN / SMC / EMCI / MCI / LMCI / AD (%) | Predicted AD |
|---|---|---|---|---|---|---|---|
| **TabPFN v2, k = 50** | **25.2%** | **24.6%** [20.5, 29.0] | **23.5%** | 1.1 | 8.2 (8–42) | 12 / 21 / 32 / 9 / 20 / 54 | 23% |
| v2 fixed epoch (primary) | 19.5% | 19.1% [16.2, 22.1] | 17.7% | 3.2 | 6.2 (4–33) | 11 / 19 / 23 / 8 / 16 / 38 | 22% |
| v2 nested early stopping | 18.1% | 17.9% [15.2, 20.8] | 15.8% | 2.2 | 4.9 (4–29) | 9 / 19 / 17 / 11 / 21 / 30 | 22% |
| v2 fixed epoch by balanced acc. (sensitivity) | 20.2% | 19.9% [16.6, 23.3] | 18.1% | 1.9 | 8.0 (4–38) | 8 / 11 / 23 / 12 / 18 / 47 | 28% |
| v2 epoch 60 (no selection) | 16.9% | 17.0% [14.1, 20.0] | 16.1% | 2.0 | 7.5 (4–31) | 6 / 12 / 24 / 20 / 14 / 26 | 21% |
| C0 (v1, leaky inner selection) | 19.8% | 19.6% [16.5, 22.9] | 16.7% | 1.2 | 5.4 (8–31) | 9 / 22 / 21 / 14 / 3 / 49 | 33% |

Chance is 16.7%. Seed sd and fold sd refer to balanced accuracy, in points. Each subject's 5 out-of-fold predictions
per seed are pooled, then averaged over seeds. Fold-level results are in the appendix.

**Paired comparisons** (same subjects and folds; bootstrap CI on balanced accuracy):

| A vs B | Balanced acc. difference [95% CI] | Seeds A better | Splits up / down (sign p) | Macro-F1 difference | Verdict |
|---|---|---|---|---|---|
| v2 fixed vs C0 | −0.5 [−4.5, +3.5] | 1/5 | 10 / 11 (1.00) | +1.0 | no difference |
| v2 ES vs C0 | −1.8 [−5.8, +2.0] | 2/5 | 10 / 13 (0.68) | −0.9 | no difference |
| v2 fixed vs v2 ES | +1.2 [−1.9, +4.3] | 3/5 | 12 / 8 (0.50) | +1.9 | no difference |
| v2 fixed vs TabPFN | −5.5 [−10.7, −0.6] | 0/5 | 6 / 15 (0.08) | −5.8 | **TabPFN better** |
| v2 ES vs TabPFN | −6.8 [−11.8, −1.8] | 0/5 | 5 / 19 (0.01) | −7.7 | **TabPFN better** |
| C0 vs TabPFN | −5.0 [−10.3, +0.0] | 0/5 | 4 / 17 (0.01) | −6.8 | interval touches 0 |
| fixed by balanced acc. vs fixed by loss | +0.8 [−2.5, +4.3] | 4/5 | 9 / 11 (0.82) | +0.4 | no difference |
| epoch 60 vs fixed by loss | −2.1 [−5.6, +1.8] | 1/5 | 10 / 13 (0.68) | −1.6 | no difference |

The verdict rule, declared in advance: the 95% CI excludes 0 and at least 4 of 5 seeds agree. Macro-F1 differences
are seed-pooled. No correction is made for the 8 comparisons.

## 5. Effects of nested SVM-RFE and of epoch selection

**Nested SVM-RFE** (C0 vs v2; all values are inner-validation minus held-out):

| | C0 | v2 |
|---|---|---|
| Balanced accuracy, epochs 10–60, no epoch selected | **+4.3** [2.7, 6.2]; 23/25 splits positive | **+0.4** [−0.9, 1.6]; 16/25 splits |
| At each model's early-stopping epoch | +13.9 | +10.2 |
| Mean inner-validation loss, minimum (epoch) | 1.766 (25) | 1.794 (12); ln 6 = 1.792 |

- **The fix removed the leak:** the no-selection gap fell by 4.0 points (CI −5.8 to −2.2), lower in 21 of 25
  splits.
- **Held-out accuracy did not change.** The predicted classes shifted: the AD share of predictions fell from 33% to
  22%, LMCI recall rose from 3% to 16–21%, and AD recall fell from 49% to 30–38%.
- **Feature selection is very unstable.** The 4 inner SVM-RFE fits in a split share only 12% of their features, and
  23% with C0's outer fit. This is the likely reason seed-to-seed variability rose from 1.2 to 2.2–3.2 points.

**Epoch selection** (inner = pooled inner-validation estimate, made before the held-out fold is seen):

| Rule | Selected epoch | Inner estimate | Held-out | Optimism [95% CI] |
|---|---|---|---|---|
| Nested ES | median 11 (IQR 5–19); 30% at ≤ 5 | 28.2% | 18.0% | **+10.2** [7.9, 12.5] |
| Fixed, by loss (primary) | median 11 (IQR 7–12; range 4–28) | 17.8% | 19.2% | −1.4 [−3.6, +0.9] |
| Fixed, by balanced acc. | median 29 (IQR 23–48) | 24.2% | 20.0% | +4.2 [1.0, 7.1] |
| Epoch 60 | 60 | 17.4% | 17.0% | +0.4 [−2.5, +3.4] |

- **Epoch 30 is not supported.** It reflected C0's leaked validation signal. Inner-CV loss now picks epochs 4–28.
- **Picking the best epoch on about 23 subjects** still adds about 10 points of optimism, even without leakage. Of
  C0's 13.9 points of best-epoch optimism, about 4 came from the leak and about 10 from selection noise.
- **The fixed-epoch-by-loss rule is the only one** whose internal estimate is honest. It is still no better on the
  held-out folds.

## 6. Does the corrected transformer improve reliably?

**No.** Correcting the leakage made its internal estimates trustworthy but did not raise held-out performance. None
of the four checkpoint rules differs reliably from C0 or from each other. TabPFN v2 (k = 50) beats every
transformer variant on identical folds.

The clearest evidence is the loss curve. On validation subjects that took no part in its feature selection, the
transformer's mean validation loss never beats a uniform guess. With these 50 unstable features and about 70
training subjects, it learns no stage signal that it can detect itself.

## 7. Limitations

- **Small numbers:** 116 subjects, about 23 per outer fold and about 23 per inner-validation fold. CIs span about
  ±3 points per model, and the 8 paired comparisons are not corrected for multiplicity.
- **C0 settings were not re-tuned.** This was deliberate: no hyperparameter was chosen from held-out folds, and v1
  found no configuration that was consistently better. Properly tuning them would need another nested loop.
- **The fixed-epoch rule averages the 4 inner models,** each trained on about 70 subjects. It does not refit on all
  ~93 training subjects.
- **The analysis code was test-run on the first 5 finished splits** (written to a temporary folder) while the grid
  was running. The runner was pinned by its hash beforehand and not changed, and no rule or setting was altered
  after that.
- **Scanner site, scan year and TR are associated with stage** and were not harmonised here.
- **All comparisons share the 116-subject project history** described in section 2.

## 8. Next step (not started)

1. **Agree the evaluation protocol for the 35 subjects with your supervisor** (section 2, options 1–3). Keep
   `run_test.py` gated until then.
2. **Report TabPFN v2 (k = 50) with nested 116-subject CV as the imaging-only result:** 24.6% balanced accuracy,
   23.5% macro-F1.
3. **Stop tuning the transformer's training settings on this feature set.** Its validation loss does not beat
   chance, so changes to checkpointing or the optimiser cannot help. Any further transformer work would need a
   different input: features that are stable across folds rather than a fresh 50 per model, or more subjects. No
   gain is promised.

## Files

Results are under `derivatives/experiments/transformer_nested_v2/` (new; earlier results are untouched):

| What | Where |
|---|---|
| Design, settings and script hash | `manifest.json` |
| Per-split results: inner curves, selected epochs, predictions, SVM-RFE support | `grid/cv5_seed*_fold*.json` |
| Leakage audit | `audit/audit.log`, `audit/audit.json` |
| Model comparison | `analysis/model_comparison.csv` |
| Fold-level results | `analysis/fold_level.csv` (25 splits × 6 models, with per-class recall) |
| Paired comparisons | `analysis/paired_comparisons.csv` |
| Selected epochs | `analysis/selected_epochs.csv` |
| Inner vs held-out gap | `analysis/inner_vs_heldout_gap.csv` |
| Feature stability | `analysis/feature_stability.csv` |
| Confusion matrices | `analysis/confusion_matrices.csv` |
| Mean curves | `analysis/curves_*.csv` |
| Text summary | `analysis/summary.txt` |
| Figures | `analysis/fig_heldout_and_leakage.png`, `analysis/fig_epoch_selection.png` |

Scripts are in `scripts/experiments/transformer_nested_v2/`: `run_nested.py`, `audit_nested.py` and
`analyze_nested.py`.

## Appendix: fold-level balanced accuracy (%)

**Per seed** (pooled out-of-fold, balanced accuracy / macro-F1):

| Seed | C0 | v2 ES | v2 fixed | TabPFN |
|---|---|---|---|---|
| 20261015 | 18.8 / 16.5 | 15.2 / 14.8 | 18.1 / 18.5 | 25.4 / 25.5 |
| 20261016 | 18.3 / 14.0 | 17.5 / 15.2 | 15.0 / 12.8 | 24.4 / 23.3 |
| 20261017 | 20.0 / 15.9 | 20.6 / 18.7 | 24.8 / 23.0 | 25.6 / 24.2 |
| 20261018 | 21.7 / 20.8 | 15.8 / 12.8 | 19.2 / 17.3 | 25.2 / 23.5 |
| 20261019 | 19.4 / 16.3 | 20.2 / 17.7 | 18.3 / 17.1 | 22.5 / 21.0 |

**Per split** (outer test fold of about 23 subjects):

| Seed | Fold | C0 | v2 ES | v2 fixed | TabPFN |
|---|---|---|---|---|---|
| 20261015 | 0 | 8.3 | 20.8 | 16.7 | 25.0 |
| | 1 | 23.6 | 22.2 | 25.0 | 41.7 |
| | 2 | 16.7 | 4.2 | 16.7 | 16.7 |
| | 3 | 20.8 | 12.5 | 16.7 | 16.7 |
| | 4 | 25.0 | 16.7 | 16.7 | 26.4 |
| 20261016 | 0 | 16.7 | 20.8 | 4.2 | 33.3 |
| | 1 | 20.8 | 13.9 | 8.3 | 8.3 |
| | 2 | 16.7 | 22.2 | 25.0 | 22.2 |
| | 3 | 16.7 | 18.1 | 16.7 | 37.5 |
| | 4 | 20.8 | 13.9 | 20.8 | 20.8 |
| 20261017 | 0 | 20.8 | 16.7 | 16.7 | 33.3 |
| | 1 | 25.0 | 29.2 | 26.4 | 30.6 |
| | 2 | 29.2 | 15.3 | 26.4 | 22.2 |
| | 3 | 12.5 | 18.1 | 22.2 | 25.0 |
| | 4 | 12.5 | 25.0 | 33.3 | 16.7 |
| 20261018 | 0 | 25.0 | 16.7 | 20.8 | 12.5 |
| | 1 | 18.1 | 16.7 | 20.8 | 25.0 |
| | 2 | 20.8 | 12.5 | 20.8 | 29.2 |
| | 3 | 12.5 | 16.7 | 16.7 | 20.8 |
| | 4 | 30.6 | 16.7 | 16.7 | 38.9 |
| 20261019 | 0 | 16.7 | 20.8 | 25.0 | 16.7 |
| | 1 | 16.7 | 16.7 | 8.3 | 20.8 |
| | 2 | 16.7 | 20.8 | 20.8 | 16.7 |
| | 3 | 27.8 | 25.0 | 20.8 | 29.2 |
| | 4 | 18.1 | 18.1 | 16.7 | 29.2 |
