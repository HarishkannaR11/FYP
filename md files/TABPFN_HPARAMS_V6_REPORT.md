# TabPFN v2 Controlled Hyperparameter Study (v6)

Written 2026-10-22.

- **Setup:** the fixed 116-subject cohort, imaging only (SVM-RFE k = 50 of 984 rs-fMRI features), and the 25
  matched outer splits (fingerprints `3edbea19c1ee3c08`, `d53ad26a06b7d5c5`, `2a8e70dd0eb80cae`,
  `d54bf96735be41dd`, `8b7f4cd9e512b26a`).
- **Status:** an exploratory CV analysis. The 35 "locked" subjects are part of the CV and are not untouched. The
  gated test script was not run, nothing was downloaded, and nothing was committed.
- **Plan and outputs:** pre-registered in `derivatives/experiments/tabpfn_hparams_v6/PLAN.md`; outputs are in the
  same folder.

## Decision: both candidates REJECTED; keep the baseline

Neither supported TabPFN setting changes the result. The baseline (TabPFN v2 defaults, SVM-RFE k = 50) stays at
**24.6% balanced accuracy**.

| Configuration | Balanced acc. [95% CI] | Macro-F1 [CI] | Accuracy [CI] | Δ balanced acc. vs C0 [CI] | Seeds better | Splits better / worse / tied | Corrected p (Holm) | Decision |
|---|---|---|---|---|---|---|---|---|
| **C0** stored baseline (defaults) | **24.6** [20.5, 29.0] | 23.5 [19.2, 27.5] | 25.2 [20.9, 29.5] | – | – | – | – | reference |
| C1 defaults passed explicitly | 24.6 [20.5, 29.0] | 23.5 | 25.2 | 0.0 (identical) | – | 0 / 0 / 25 | – | reproduction passed |
| C2 `n_estimators=32` | 24.1 [19.8, 28.6] | 23.2 [18.6, 27.7] | 24.5 [20.2, 29.0] | −0.5 [−2.8, +1.7] | 1/5 | 8 / 11 / 6 | 0.87 (1.00) | **REJECT** |
| C3 `average_before_softmax=True` | 24.7 [20.5, 29.2] | 23.6 [19.2, 27.7] | 25.2 [20.9, 29.5] | +0.0 [−0.5, +0.6] | 1/5 | 1 / 1 / 23 | 1.00 (1.00) | **REJECT** |

**Why both are rejected:** neither meets the pre-registered "promising" threshold (balanced accuracy +3 points or
more, macro-F1 not lower, intermediate recall not more than 3 points lower), and neither difference is reliable.

**C3 is practically a no-op here:** its predictions are identical to the baseline in 23 of 25 splits.

## 1. Inspection (read-only, before any run)

- **Installed version:** `tabpfn` 9.2.0.
- **Baseline call:** `TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu",
  random_state=seed+fold)` in `scripts/model/train_tabpfn.py` (line 91) and in v3's `run_tabpfn_configs.py` (line
  127). No other argument is set anywhere in the project. Earlier experiments (v1–v5) never changed an inference
  setting.
- **Effective defaults.** The V2 checkpoint declares no inference settings, so:

  | Setting | Value |
  |---|---|
  | `n_estimators` | `"auto"` → 8 (50 features is under the 500-per-member cap, so no scaling) |
  | `softmax_temperature` | `"auto"` → 0.9 |
  | `average_before_softmax` | False |
  | `balance_probabilities` | False |
  | metric tuning | none |

  The fitted models confirmed this in every split (`n_estimators_` = 8, `softmax_temperature_` = 0.9, 8 ensemble
  configurations).

### Supported parameters and decisions

| Parameter | Decision | Reason |
|---|---|---|
| `n_estimators` | tested (C2 = 32) | Ensemble of preprocessing/feature-shuffle prompts; 32 is the library's own maximum for automatic ensemble sizing |
| `average_before_softmax` | tested (C3) | Documented as possibly helping with many classes |
| `eval_metric` + `tuning_config` | excluded | The library warns against tuning below 500 training samples. Here it would tune 6-class thresholds on about 9 held-out subjects (10% of about 92) |
| `softmax_temperature` | excluded | Calibration only; choosing a value would need selection |
| `balance_probabilities` | excluded | Already tested; the brief requires the baseline's behaviour |
| `inference_config` knobs | excluded | Would be a broad search |
| Other model versions | excluded | Different models, and licence-gated |
| Compute flags | not relevant | Compute only |

## 2. Verification (`analysis/verification.json`: 14 of 14 pass)

- **Class counts:** 20 / 20 / 20 / 16 / 20 / 20. Labels are CN = 0, SMC = 1, EMCI = 2, MCI = 3, LMCI = 4, AD = 5.
- **Splits, labels and selected features:** for C1–C3, the outer test sets, stored labels and SVM-RFE features are
  identical to C0 in 25 of 25 splits.
- **Resolved settings:** exactly as pre-registered in every split:

  | Configuration | Estimators | Temperature | Averaging |
  |---|---|---|---|
  | C1 | 8 | 0.9 | probabilities |
  | C2 | 32 | 0.9 | probabilities |
  | C3 | 8 | 0.9 | logits |

- **No warnings** were raised by TabPFN in any run.
- **C1 reproduces C0:** identical predictions in 25 of 25 splits, probability difference 0.0. This confirms both the
  new code path and the inspection's account of the defaults.

## 3. Newly run results vs stored results

- **Stored (not rerun):** C0, from `model_training/tabpfn/cv116_tabpfn_svmrfe50.json`.
- **Newly run** (`grid/C1`, `grid/C2`, `grid/C3`, 25 of 25 splits each, logged in `run.log`, exit 0): C1 (2.8 min),
  C2 (14.2 min) and C3 (2.9 min).
- **Unchanged in every configuration:**
  - inputs, SVM-RFE (k = 50, fitted on the outer training subjects), raw features;
  - `random_state` = seed + fold;
  - no class balancing;
  - imaging only;
  - outer-test labels hidden from fitting.

  No inner loop was needed, because nothing was selected.

## 4. Per-stage recall (%, seed-pooled)

| Configuration | CN | SMC | EMCI | MCI | LMCI | AD | Mean of the 4 intermediate stages |
|---|---|---|---|---|---|---|---|
| C0 baseline | 12 | 21 | 32 | 9 | 20 | 54 | 20.4 |
| C2 32 estimators | 11 | 20 | 28 | 14 | 17 | 55 | 19.7 (−0.8 [−3.9, +2.1]) |
| C3 average before softmax | 12 | 21 | 32 | 10 | 20 | 53 | 20.8 (+0.3 [0.0, +0.9]) |

**Other outputs:**
- Confusion matrices (counts, 5 seeds pooled, this stage order) are in `analysis/confusion_matrices.csv`.
- Fold-level results are in `analysis/fold_level.csv`.
- Paired comparisons for balanced accuracy, macro-F1, accuracy and intermediate recall are in
  `analysis/paired_comparisons.csv`.
- The figure is `analysis/fig_hparams.png`.

**Statistics used:**
- Seed-pooled metrics, averaged over 5 seeds.
- Class-stratified subject bootstrap (2,000) for CIs and paired differences.
- Seeds agreeing, and splits better / worse / tied.
- The corrected resampled t-test (Nadeau–Bengio) for repeated-CV dependence, Holm-adjusted across the 2 planned
  candidates.

## 5. Interpretation

- **The settings are not the bottleneck.** TabPFN's defaults are already stable for this problem:
  - **Bigger ensemble:** quadrupling the ensemble changes predictions in only some splits, with no net gain.
  - **Different averaging:** averaging logits instead of probabilities changes almost nothing.
- **Consistent with earlier results:** v3–v5 showed that the limit is the weak, site-entangled imaging signal and
  the sample size, not the classifier.
- **No further TabPFN tuning is justified.**

## Reproducibility

```
python scripts/experiments/tabpfn_hparams_v6/run_hparams.py --configs C1 C3 C2
python scripts/experiments/tabpfn_hparams_v6/analyze_v6.py
```

**Outputs** (new; no existing output was changed):

| What | Where |
|---|---|
| Pre-registration | `PLAN.md` |
| Manifest (script hash, settings, version) | `manifest.json` |
| Run log | `run.log` |
| Per-split records (predictions, probabilities, resolved settings, warnings) | `grid/C1–C3/` |
| Analysis | `analysis/` |
