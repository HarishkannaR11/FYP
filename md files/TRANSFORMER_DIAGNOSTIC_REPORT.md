# Transformer Diagnostic Report

Written 2026-10-21. Scope: the existing Multi-Attention Transformer and TabPFN v2 only, on the 116-subject cohort
(6 stages, chance 16.7%), using rs-fMRI features only (SVM-RFE, 50 of 984). No MMSE, CDR or comorbidity
variables are used anywhere in this report.

**Rules followed:**
- No configuration was selected using held-out-fold results.
- The locked 70/30 test set was not accessed; `run_test.py` is gated behind an explicit flag.
- No other experiments were started.

---

## 1. Conclusion

- **No transformer configuration gives a consistent held-out improvement over the baseline (C0).** Every change
  (learning rate, dropout, weight decay, model size) moves held-out balanced accuracy by −2.5 to +2.3 points. All
  95% intervals include 0, and none improves in more than 6 of 10 splits.
- **TabPFN stays ahead of every transformer configuration on identical folds.** On the 25 shared splits it leads
  C0 by 5.0 points (24.6% vs 19.6% balanced; C0 is better in 0 of 5 seeds; 95% interval −10.3 to 0.0). At
  k = 25, 50, 150 or 246, TabPFN beats C0 by 6.1–7.4 points with intervals above 0 (3 of 3 seeds). At k = 100 the
  lead is +3.3 (95% interval −2.4 to +9.0).
- **The bottleneck is not the optimiser settings. It is model selection on tiny, biased validation sets, on top of
  a weak imaging signal:**
  - The early-stopping sets have about 23 subjects, so balanced accuracy jumps about 4.5 points between epochs.
  - Even with no selection, inner-validation accuracy runs 4–8 points above held-out.
  - Picking the best epoch then reports 32.8% inner-validation balanced accuracy for models that score 18.9% on
    the held-out fold.

## 2. What was run

| Item | Detail |
|---|---|
| Model | Multi-Attention Transformer, about 47 ROI tokens from SVM-RFE k = 50, 4 inner models per split (bagged), early stopping (patience 15) on inner-validation balanced accuracy |
| Logging | Every epoch (60) of every inner model: loss, accuracy, balanced accuracy and macro-F1 on its training subjects, its inner-validation subjects (the early-stopping set) and the held-out fold (diagnostic only) |
| C0 baseline | lr 1e-3, dropout 0.3, weight decay 1e-2, width 16, 1 layer: **25 outer splits** (5 folds × 5 seeds) |
| C1–C5 | lr 3e-4 / lr 3e-3 / dropout 0.1 / weight decay 0.1 / larger (width 32, 4 heads, 2 layers): **10 outer splits each** (seeds 20261015–20261016) |
| Per-split choice | Configuration with the best inner-validation score in each split (training subjects only) |
| TabPFN v2 | Existing saved predictions on the same folds: k = 50 (25 splits); k = 25 / 50 / 100 / 150 / 246 (15 splits each, from the leakage-safe grid) |

**Reproducibility:**
- The logged C0 rerun reproduces the earlier saved transformer predictions in **25 of 25 splits**, with the same
  best epochs.
- TabPFN k = 50 from the grid matches the earlier run in **15 of 15 splits**.
- Metrics are pooled out-of-fold per seed, then averaged over seeds. Intervals come from a stratified subject
  bootstrap (2,000 resamples); paired differences use the same resamples.

## 3. Data checks (from the Phase 1 audit, 38 checks)

- **Splits are subject-level:** 116 subjects with 1 scan each, and every split is disjoint by subject ID.
- **No duplicate scans:** the most similar pair of subjects correlates at r = 0.53.
- **No missing or non-finite imaging values.**
- **Classes:** CN 20, SMC 20, EMCI 20, MCI 16, LMCI 20, AD 20. Every 5-fold test fold contains at least 3 of each
  stage.
- **No feature leaks the label on its own:** the strongest single feature separates a stage with AUC 0.78. Only 4
  of 984 features differ by stage after correction for multiple testing.
- **Acquisition confounds:**
  - scanner site is strongly associated with stage (χ² p = 0.00004)
  - all 20 SMC scans are from 2013
  - TR differs by stage (p = 0.044): only EMCI includes scans at 2250 ms; every other scan is 3000 ms
  - all 112 scans matched to scanner metadata (of 116) are Philips
- **Feature selection:** fitted inside each outer training set; verified by perturbing test rows.
- **The leak found in this study:** the same selection is reused for the inner early-stopping folds, so it has seen
  the inner-validation labels (section 6).

## 4. Results per configuration

Held-out results are pooled over the 10 splits every configuration shares (seeds 20261015–20261016); identical
folds for all rows.

| Model | Held-out balanced acc. [95% CI] | Held-out macro-F1 | Fold sd (range) | Recall CN / SMC / EMCI / MCI / LMCI / AD (%) |
|---|---|---|---|---|
| TabPFN v2, k = 50 | **24.9%** [19.1, 31.0] | **24.4%** | 9.7 (8–42) | 18 / 28 / 25 / 9 / 15 / 55 |
| Transformer C0 (baseline) | 18.5% [14.2, 22.9] | 15.2% | 4.5 (8–25) | 15 / 18 / 15 / 6 / **0** / 57 |
| C1: lr 3e-4 | 16.0% [11.8, 20.7] | 13.9% | 6.8 (4–25) | 25 / 28 / 5 / 6 / 10 / 22 |
| C2: lr 3e-3 | 19.5% [14.9, 24.3] | 17.0% | 6.7 (8–33) | 2 / 20 / 22 / 9 / 8 / 55 |
| C3: dropout 0.1 | 20.3% [14.8, 26.3] | 18.5% | 7.3 (8–29) | 5 / 15 / 25 / 22 / 10 / 45 |
| C4: weight decay 0.1 | 18.5% [14.1, 23.2] | 15.8% | 5.2 (12–29) | 12 / 15 / 25 / 6 / 2 / 50 |
| C5: larger model | 20.8% [16.3, 25.9] | 18.0% | 7.8 (8–33) | 0 / 10 / 30 / 12 / 18 / 55 |
| Configuration chosen per split by inner CV | 21.7% [17.4, 26.3] | 19.4% | 7.1 (8–33) | 2 / 20 / 22 / 12 / 15 / 57 |
| C0 without early stopping (epoch 60) | 17.4% [13.2, 22.2] | 15.6% | 4.6 (8–26) | 2 / 12 / 20 / 9 / 12 / 48 |

On all **25** splits: C0 scores 19.6% balanced (seed sd 1.2, fold sd 5.4) with macro-F1 16.7%; TabPFN k = 50 scores
24.6% (seed sd 1.1, fold sd 8.2) with macro-F1 23.5%. The transformer predicts AD for 33% of subjects (the true
share is 17%) and LMCI for only 6%.

## 5. Training behaviour (inner models; "inner validation" = early-stopping set, "held-out" = outer test fold)

| Config | Best epoch, median (middle 50%) | Training bal. at best epoch | Inner-val bal. at best epoch | Held-out bal. at best epoch (single model) | Loss gap at best → epoch 60 | Runs overfitting later |
|---|---|---|---|---|---|---|
| C0 | 14 (7–24) | 27.2% | 32.8% | 18.9% | +0.05 → **+0.59** | 14% |
| C1 lr 3e-4 | 12 (4–31) | 22.3% | 29.2% | 16.9% | +0.01 → +0.06 | 0% |
| C2 lr 3e-3 | 14 (10–19) | 33.5% | 36.8% | 18.1% | +0.17 → **+1.49** | 70% |
| C3 dropout 0.1 | 16 (10–26) | 36.3% | 36.9% | 18.6% | +0.11 → **+1.12** | 78% |
| C4 wd 0.1 | 18 (9–27) | 28.1% | 35.2% | 18.5% | +0.04 → +0.52 | 12% |
| C5 larger | 14 (10–20) | 33.5% | 35.8% | 19.3% | +0.15 → **+1.33** | 68% |

The loss gap is validation loss minus training loss. A run counts as overfitting later if, at epoch 60, training
balanced accuracy is more than 30 points above validation and validation loss is more than 5% above its minimum.

**Readings:**
- **C0 overfits after about 30 epochs.**
  - Training loss falls steadily, from 1.82 to about 1.38.
  - Inner-validation loss bottoms out at about epoch 30 (1.71) and rises to 1.97 by epoch 60.
  - Held-out loss rises above the uniform-guess level (ln 6 = 1.79) and keeps climbing.
  - At the early-stopped epoch, the model has fitted almost nothing: training balanced accuracy is only 27%.
- **More capacity or less regularisation (C2, C3, C5)** raises training accuracy (+4.5 to +7.3 points) and
  inner-validation accuracy at the best epoch (+0.2 to +1.3). It also triples the loss gap and makes 68–78% of
  runs overfit. Held-out accuracy moves only +0.8 to +1.9 points.
- **C1 (lower learning rate) underfits.** Validation loss is still falling at epoch 60, and held-out accuracy drops
  to 16.0%.
- **C4 (more weight decay) behaves almost exactly like C0.**
- **Early stopping helps only slightly:** without it (epoch 60), C0 drops by 2.4 points (95% interval −5.9 to +1.2).

## 6. Consistency against C0 (same 10 splits; held-out results used to evaluate, never to choose)

| Config | Held-out bal. change | Splits better | Sign test p | Held-out macro-F1 change | Training bal. change | Consistent gain? |
|---|---|---|---|---|---|---|
| C1 lr 3e-4 | −2.5 | 1/10 | 0.38 | −2.3 | −6.7 | No |
| C2 lr 3e-3 | +0.8 | 5/10 | 0.73 | +2.8 | +4.5 | No |
| C3 dropout 0.1 | +1.7 | 5/10 | 0.73 | +4.1 | +7.3 | No |
| C4 wd 0.1 | +0.0 | 2/10 | 1.00 | +0.6 | −0.8 | No |
| C5 larger | +1.9 | 6/10 | 0.75 | +3.5 | +4.5 | No |
| Chosen per split by inner CV | +3.1 [−1.8, +8.1] | 2/2 seeds | – | +4.2 | – | No |

Changes that raise training accuracy do not carry over to the held-out fold in a consistent way. The per-split
inner-CV choice picked C2 four times, C5 three times and C3 three times, and never C0. Its +3.1 points are not
significant, and it still trails TabPFN by 3.2 points (95% interval −9.3 to +3.0).

## 7. The performance bottleneck, and the evidence for it

1. **The early-stopping signal is too noisy to select on.**
   - Inner-validation sets have about 23 subjects, so one subject moves balanced accuracy by about 4 points; the
     measured epoch-to-epoch jump is 3–6 points.
   - Taking the best epoch therefore captures luck: inner-validation accuracy at the best epoch is 29–37% for
     models that score 16–19% on the held-out fold.
2. **The early-stopping signal is also biased upward.**
   - SVM-RFE is fitted on the whole outer training set, including the inner-validation subjects, so those subjects
     helped choose the features.
   - Averaged over epochs 10–60 with no selection at all, inner-validation balanced accuracy is 4.3–7.6 points
     above the held-out fold in every configuration.
   - This does not affect the held-out numbers. It does weaken every choice made from inner validation: the stopping
     epoch, and the per-split configuration choice.
3. **There is little generalisable signal to learn.** Only 4 of 984 features pass correction. Selected features
   change from fold to fold (18% overlap). Learning curves were still rising at about 93 training subjects. Site is
   confounded with stage.
4. **The transformer turns extra capacity into memorisation, not accuracy.** Every configuration that fits training
   data better widens the loss gap with no consistent held-out gain. The model collapses towards predicting AD
   (33% of predictions) and almost never predicts LMCI (0–18% recall).
5. **TabPFN does better on the same inputs and folds.** It has no stopping epoch to choose and uses a strong
   pretrained prior, which suits this regime better.

## 8. Recommended next step

1. **For the project's main imaging result, use TabPFN v2 (SVM-RFE, k = 25–50).** It is the most accurate model on
   identical folds, and the comparison is paired.
2. **For the transformer, fix the selection procedure before tuning anything else.** In priority order:
   1. **Refit SVM-RFE inside each inner fold,** so the early-stopping subjects are truly unseen. This removes the
      measured 4–8 point bias.
   2. **Replace per-model "best epoch on 23 subjects" with a fixed number of epochs chosen by inner CV,** for
      example the median epoch of minimum inner-validation loss (C0: about epoch 30). Loss is smoother than
      balanced accuracy. Alternatively, average the weights or predictions over a window of epochs.
   3. **Compare against C0 and TabPFN on the same 25 outer folds,** without looking at held-out results to make any
      choice.

   This targets the failure mode measured above. It may make the transformer's estimate more reliable without
   raising accuracy, and that should be reported either way.
3. **Stop tuning optimiser settings and model size.** C1–C5 show none gives a consistent held-out gain.
4. **The larger levers lie outside the model:** more subjects (the learning curve is still rising) and harmonising
   scanner sites. Also confirm with the supervisor whether the target is 6 stages or the 3-stage EMCI / LMCI / AD
   output in the project flow.

## 9. Files

All under `derivatives/experiments/transformer_diagnostics_v1/`:

| What | Where |
|---|---|
| Epoch logs and predictions, every split and configuration | `grid/C0` … `grid/C5` (one JSON per split: 4 inner models × 60 epochs) |
| Model comparison (all seeds) and on the shared seeds | `analysis/model_comparison.csv`, `analysis/model_comparison_shared_seeds.csv` |
| Paired comparisons and consistency against C0 | `analysis/paired_comparisons.csv`, `analysis/config_consistency_vs_C0.csv` |
| Per-split validation metrics | `analysis/per_split_validation.csv` |
| Best epoch and fit diagnosis per inner model | `analysis/best_epochs_and_fit.csv` |
| Mean curves per configuration | `analysis/curves_C*.csv` |
| Confusion matrices | `analysis/confusion_matrices.csv` |
| Configuration chosen in each split | `analysis/config_choice_per_split.csv` |
| Figures | `analysis/fig_curves_C0.png`, `fig_gap_bestepoch_configs.png`, `fig_best_epoch_and_fixed_epoch.png`, `fig_confusion_recall.png` |
| Text summary | `analysis/summary.txt` |

**Scripts:**
- `scripts/experiments/transformer_diagnostics_v1/run_transformer_curves.py`
- `scripts/experiments/transformer_diagnostics_v1/analyze.py`

**Reproduce:**
```
python scripts/experiments/transformer_diagnostics_v1/run_transformer_curves.py --configs C0 --seeds all
python scripts/experiments/transformer_diagnostics_v1/run_transformer_curves.py --configs C1 C2 C3 C4 C5 --seeds 20261015 20261016
python scripts/experiments/transformer_diagnostics_v1/analyze.py
```
