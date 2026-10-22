# Controlled Comparison v5: Imaging-Only, Non-Circular Multimodal and Clinical Reference

Written 2026-10-22. This uses the fixed 116-subject cohort and the 25 matched outer splits (fingerprints
`3edbea19c1ee3c08`, `d53ad26a06b7d5c5`, `2a8e70dd0eb80cae`, `d54bf96735be41dd`, `8b7f4cd9e512b26a`).

- **Status of the results:** all are exploratory cross-validation analyses. The 35 "locked" subjects are part of the
  CV and were evaluated before, so they are not untouched. The gated test script was not run, and no data was
  downloaded.
- **Plan and outputs:** pre-registered in `derivatives/experiments/controlled_comparison_v5/PLAN.md`; outputs are in
  the same folder.

## Recommendation

**No new experiment reliably improves on the baseline.** No candidate reached the pre-registered "promising"
threshold (+3 points balanced accuracy, macro-F1 not lower, intermediate recall not more than 3 points lower). The
stopping rule therefore applies: no further experiment and no confirmation run.

| Role | Model | Balanced acc. [95% CI] | Macro-F1 | Accuracy |
|---|---|---|---|---|
| **Best defensible imaging-only model** | TabPFN v2, SVM-RFE k = 50 of 984 rs-fMRI features | **24.6%** [20.5, 29.0] | 23.5% | 25.2% |
| **Best non-circular multimodal model** | TabPFN v2, the same imaging + 8 comorbidities (age, sex, APOE4, GDS, 4 medical-history flags) | **27.5%** [22.8, 32.3] | 25.8% | 28.1% |
| **Clinical reference, not an fMRI result** | TabPFN v2, MMSE + CDR-SB only | **47.1%** [40.2, 53.9] | 43.9% | 48.1% |

- **TabPFN with class balancing** (26.1%) is the most consistent imaging-only variant. It is not adopted, because
  it misses the +3-point threshold and is not confirmed by the corrected test (section 3). It can be reported as a
  sensitivity analysis.
- **Do not combine MMSE/CDR with imaging** (section 5): it gives no gain over the clinical scores alone.

## 1. Verification (newly executed, read-only; `verification/verification.log`: 15 of 15 pass)

| Check | Result |
|---|---|
| Label mapping | CN = 0, SMC = 1, EMCI = 2, MCI = 3, LMCI = 4, AD = 5, consistent in metadata, `y.npy` and `stages.npy` |
| Class counts | 20 / 20 / 20 / 16 / 20 / 20 |
| Scans | one per subject (116 subjects, 116 acquisitions) |
| Outer test sets, fold fingerprints, stored labels | identical to the plan in all 25 splits, for all 8 compared sources |
| Baseline records | equal to the original stored TabPFN baseline in 25 of 25 splits |
| Metrics | balanced accuracy and macro-F1 from the project code equal scikit-learn exactly; recomputed per-split metrics equal every stored value exactly |

**Corrections to the brief's reference numbers** (from the stored files):
- The "43.4 / 42.6" for imaging + MMSE/CDR-SB are **accuracy / balanced accuracy**; macro-F1 is 42.3.
- The "46.4 / 45.9" MMSE/CDR reference was a **logistic regression on MMSE + CDR global + CDR-SB**, so a model family
  outside this brief. It was replaced by a TabPFN MMSE + CDR-SB run (section 5).

## 2. Methods

- **Splits and processing:**
  - The same 25 outer splits for every model.
  - Every learned step is fitted on the outer training subjects: SVM-RFE and its scalers, TabPFN's internal
    preprocessing (including its handling of missing values), and the class counts used for balancing.
  - Outer-test labels are hidden from fitting.
  - No hyperparameter is selected, so no inner loop is needed.
- **Statistics:**
  - Seed-pooled out-of-fold metrics, averaged over 5 seeds.
  - 95% CIs from a class-stratified subject bootstrap (2,000 resamples). They may differ from earlier reports in
    the first decimal because the bootstrap draws differ.
  - Paired differences on the same resamples, with seeds agreeing and splits better / worse / tied.
  - The corrected resampled t-test (Nadeau–Bengio), for repeated-CV dependence, Holm-adjusted across the 3 planned
    candidate comparisons.
- **New run vs earlier runs:** only the TabPFN MMSE + CDR-SB reference was newly run (`clinical_reference/`, 25 of 25
  splits, logged). Every other model is a **previous run**, re-analysed from its stored predictions.

## 3. Imaging-only candidates (previous runs, re-analysed)

**Hypotheses:**
- **492 features:** removing the unstable SVM-RFE step (all 492 mALFF + mReHo features) improves balanced accuracy.
- **Class balancing:** TabPFN's training-prior balancing improves balanced accuracy, mainly through MCI recall.

| Comparison with the baseline | Δ balanced acc. [CI] | Seeds better | Splits better / worse / tied | Corrected p (Holm) | Δ macro-F1 [CI] | Δ intermediate recall [CI] | Decision |
|---|---|---|---|---|---|---|---|
| 492 mALFF + mReHo, no selection (`tabpfn_local492_v4`) | −1.6 [−5.3, +2.2] | 1/5 | 8 / 14 / 3 | 0.67 (1.00) | −2.9 [−6.3, +0.7] | −4.4 [−8.6, −0.4] | not promising; worse |
| Class balancing (`biomarker_ablation_v3`) | +1.5 [−0.3, +3.3] | 4/5 | 8 / 4 / 13 | 0.54 (1.00) | +1.9 [+0.3, +3.5] (5/5 seeds) | +3.2 [+0.9, +5.8] (5/5 seeds) | not promising (below +3); consistent direction, not confirmed |

## 4. Non-circular multimodal (previous runs, re-analysed)

**Hypothesis:** adding the 8 non-circular comorbidity variables to imaging improves balanced accuracy.

| Comparison | Δ balanced acc. [CI] | Seeds better | Splits better / worse / tied | Corrected p (Holm) | Δ macro-F1 [CI] | Δ intermediate recall | Decision |
|---|---|---|---|---|---|---|---|
| TabPFN imaging + comorbidities vs TabPFN imaging (`cv116_tabpfn_svmrfe50_comorb8`) | +2.8 [−0.4, +5.8] | 4/5 | 14 / 6 / 5 | 0.37 (1.00) | +2.3 [−0.8, +5.2] | +0.0 [−3.1, +2.9] | not promising (+2.8 is below +3); not confirmed |
| Context: transformer + comorbidities vs transformer imaging (`biomarker_ablation_v3/transformer/comorb8`) | +6.6 [+1.5, +11.5] | 4/5 | 18 / 4 / 3 | 0.32 | +6.7 [+1.6, +11.6] | +1.6 | the transformer gains more, but stays below TabPFN + comorbidities (25.7% vs 27.5%) |

**The stopping rule applies:** both justified experiments (sections 3 and 4) failed the pre-registered threshold.

## 5. Clinical references (MMSE and CDR are part of ADNI's diagnosis, so these are circular)

| Model | Balanced acc. [CI] | Macro-F1 [CI] | Accuracy [CI] | Run |
|---|---|---|---|---|
| TabPFN, MMSE + CDR-SB only | 47.1 [40.2, 53.9] | 43.9 [36.7, 50.7] | 48.1 [41.2, 55.0] | **new** |
| TabPFN, imaging + comorbidities + MMSE + CDR-SB (`cv116_tabpfn_svmrfe50_clin10`) | 42.6 [36.8, 49.0] | 42.3 [36.2, 48.1] | 43.4 [37.2, 49.5] | previous |

**Comparisons:**
- **MMSE + CDR-SB alone beats imaging-only TabPFN:** +22.5 points [+14.4, +30.7], better in 25 of 25 splits.
- **Adding imaging and comorbidities to MMSE/CDR does not help:**
  - balanced accuracy −4.5 [−10.9, +1.9], worse in 0 of 5 seeds;
  - intermediate recall −10.3 [−18.9, −1.8].

So the accuracy of these models comes from the clinical scores, not from rs-fMRI.

## 6. Per-stage recall (%, seed-pooled) and confusion matrices

| Model | CN | SMC | EMCI | MCI | LMCI | AD |
|---|---|---|---|---|---|---|
| TabPFN imaging (baseline) | 12 | 21 | 32 | 9 | 20 | 54 |
| TabPFN 492 mALFF + mReHo | 14 | 27 | 21 | 0 | 16 | 60 |
| TabPFN + balancing | 12 | 20 | 31 | 24 | 20 | 50 |
| Transformer imaging (v2) | 11 | 19 | 23 | 8 | 16 | 38 |
| TabPFN + comorbidities | 16 | 21 | 32 | 9 | 20 | 67 |
| Transformer + comorbidities | 35 | 28 | 18 | 5 | 21 | 47 |
| *Clinical reference:* TabPFN MMSE + CDR-SB | 11 | 77 | 47 | 19 | 43 | 86 |
| *Clinical reference:* imaging + comorb. + MMSE/CDR | 35 | 41 | 44 | 19 | 41 | 76 |

- Confusion matrices for every model (counts, 5 seeds pooled, in this stage order) are in
  `analysis/confusion_matrices.csv`.
- Fold-level results are in `analysis/fold_level.csv`.
- The overview figure is `analysis/fig_models.png`.

## 7. Confirmation step

No confirmation run was done. No candidate was promising, and no untouched subjects exist: only 13 unused subjects
(6 CN, 6 EMCI, 1 SMC), too few for six classes. An independent cohort would need your authorisation.

## Reproducibility

```
python scripts/experiments/controlled_comparison_v5/run_mmse_cdr_tabpfn.py    # the only new model run
python scripts/experiments/controlled_comparison_v5/analyze_v5.py             # verification + comparisons
```

**Outputs** (all new, in `derivatives/experiments/controlled_comparison_v5/`; no existing output was changed):

| What | Where |
|---|---|
| Pre-registration | `PLAN.md` |
| Run log | `run_mmse_cdr.log` |
| New per-split records | `clinical_reference/grid/` |
| Verification | `verification/verification.{json,log}` |
| Results | `analysis/metrics.csv`, `paired_comparisons.csv`, `confusion_matrices.csv`, `fold_level.csv`, `summary.txt`, `manifest.json`, `fig_models.png` |

The figure was redrawn once from the saved `metrics.csv` to fix overlapping axis labels; no number changed.
