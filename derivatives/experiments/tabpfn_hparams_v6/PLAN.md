# Pre-registered plan: TabPFN v2 controlled hyperparameter study (v6)

Written 2026-10-22, after a read-only inspection and before any run. This is an exploratory CV analysis on the
fixed 116-subject cohort; the 35 "locked" subjects are part of the CV and are not untouched. The gated test
script is not run, nothing is downloaded, and nothing is committed.

## Inspection findings (read-only)

**Installed version and baseline:**
- `tabpfn` 9.2.0.
- The baseline (`scripts/model/train_tabpfn.py`, line 91, and v3 `run_tabpfn_configs.py`, line 127) is
  `TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu", random_state=seed+fold)`. No other
  argument is set anywhere in the project.
- The input is SVM-RFE k = 50 from the 984 rs-fMRI features, fitted on the outer training subjects.
- All 25 stored records were checked: candidate `tabpfn_svmrfe50`, 50 features, support indices 1–980, fold
  fingerprints `3edbea19c1ee3c08 d53ad26a06b7d5c5 2a8e70dd0eb80cae d54bf96735be41dd 8b7f4cd9e512b26a`, classes
  20 / 20 / 20 / 16 / 20 / 20, and every subject predicted once per seed.

**Effective defaults:** the V2 checkpoint's stored config declares no inference settings, so the library defaults
apply.

| Setting | Effective value |
|---|---|
| `n_estimators="auto"` | `DEFAULT_N_ESTIMATORS` = **8**. Coverage scaling does not trigger: 50 features ≤ 500 per member |
| `softmax_temperature="auto"` | **0.9** |
| `average_before_softmax` | False (average probabilities) |
| `balance_probabilities` | False |
| `eval_metric` / `tuning_config` | none |

**Earlier experiments did not touch these settings.** v1–v5 varied the feature count, biomarker pool, selection
on or off, class balancing, and clinical inputs.

### Supported parameters considered

| Parameter (documented in 9.2.0) | Decision | Reason |
|---|---|---|
| `n_estimators` | **C2** | Size of TabPFN's ensemble of preprocessing/feature-shuffle "prompts". More members average out prompt-to-prompt variance. 32 is the library's own upper value for automatic ensemble sizing (`MAX_AUTO_SCALED_N_ESTIMATORS`) |
| `average_before_softmax` | **C3** | Documented as possibly improving predictive performance with many classes (6 here): averages logits instead of probabilities |
| `eval_metric` + `tuning_config` (threshold/temperature tuning) | excluded | The library warns against tuning below 500 training samples (`MIN_NUM_SAMPLES_RECOMMENDED_FOR_TUNING`). With about 92 training subjects its defaults hold out 10% per tuning fold, about 9 subjects to tune 6-class thresholds. It would also be a second, internal selection layer |
| `softmax_temperature` | excluded | Documented as affecting predicted probabilities (calibration), not the decision; choosing a value would need its own selection loop |
| `balance_probabilities` | excluded | Class balancing was already tested (v3/v5), and the brief requires baseline class-balancing behaviour |
| `inference_config` (preprocessing transforms, feature shift, polynomial features, outlier removal, subsampling) | excluded | Advanced knobs; trying them would be a broad search |
| `model_path` / other versions (V2.5 and later) | excluded | Different models (and licence-gated) |
| `fit_mode`, `inference_precision`, `memory_saving_mode`, `device`, `n_jobs`, compile flags | not relevant | Compute only |
| `categorical_features_indices` | not relevant | No categorical inputs |

## Configurations (maximum 4)

| ID | Setting | Role |
|---|---|---|
| C0 | stored baseline (`model_training/tabpfn/cv116_tabpfn_svmrfe50.json`, identical to `biomarker_ablation_v3/grid/A_all4`) | Reference. Not rerun, not modified |
| C1 | the effective defaults passed explicitly: `n_estimators=8`, `softmax_temperature=0.9`, `average_before_softmax=False` | Reproduction check of the new code path and of the inspection's claims |
| C2 | `n_estimators=32` (everything else default) | Candidate |
| C3 | `average_before_softmax=True` (everything else default) | Candidate |

**Identical in all configurations:**
- the 25 outer splits;
- SVM-RFE k = 50 on the 984 features, fitted on each outer training set (seed + fold), using v3's verified
  `select()`;
- raw selected features passed to TabPFN;
- `random_state` = seed + fold;
- `balance_probabilities=False`;
- imaging only (no MMSE/CDR or other clinical data);
- outer-test labels hidden (−1) from fitting.

Each configuration is fixed in advance and nothing is chosen from data, so no inner loop is used.

**Reproduction rule:**
- **Expected outcome:** C1 predictions identical to C0 in 25 of 25 splits.
- **If C1 differs:** the difference is reported, and C2/C3 are also compared with C1 (same code path).

## Metrics, comparisons and decisions (fixed now)

- **Metrics:**
  - balanced accuracy (primary);
  - macro-F1 and accuracy;
  - recall per stage;
  - confusion matrices;
  - 95% CIs from a class-stratified subject bootstrap (2,000).
- **Paired comparisons:**
  - against C0 on the same resamples;
  - seeds agreeing;
  - splits better / worse / tied;
  - the corrected resampled t-test (Nadeau–Bengio), Holm-adjusted across the 2 planned candidate comparisons (C2
    and C3 vs C0).
- **Promising** requires all three:
  - balanced accuracy +3.0 points or more;
  - macro-F1 change ≥ 0;
  - mean intermediate recall (SMC, EMCI, MCI, LMCI) no more than 3 points lower.
- **Reliable:** promising, plus a bootstrap CI excluding 0, at least 4 of 5 seeds agreeing, and Holm-adjusted
  corrected p < 0.05.
- **Accept** a candidate only if it is reliable. Otherwise **reject**: the baseline is kept.
- **Stop after C1–C3** regardless of the outcome; there is no expansion.

## Cost

| Configuration | Per split | Total |
|---|---|---|
| C1, C3 | about 7 s | about 3 minutes each |
| C2 (4× the ensemble) | about 25 s | about 10 minutes |

Outputs go to `derivatives/experiments/tabpfn_hparams_v6/`.
