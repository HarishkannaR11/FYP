# Improvement Study v3: Feature Selection, Biomarkers, Class Balance and Transformer Variants

Written 2026-10-21. Fixed cohort of 116 subjects, 6 stages (chance balanced accuracy 16.7%), rs-fMRI features
(246 Brainnetome ROIs × mALFF, mReHo, DC_z, FC strength). Only TabPFN v2 and the corrected transformer are used.
MMSE and CDR-SB are not used anywhere. The comorbidity experiment is reported separately in section 7.

**All results are exploratory cross-validation analyses on these 116 subjects, not independent confirmation.**
Earlier project decisions were made after seeing results on the same subjects.

## 1. Summary

- **No change reliably improves on the corrected baselines.** That covers 12 feature-selection settings, two nested
  (inner-CV) selection procedures, TabPFN class balancing, and three transformer variants.
  - Under the pre-registered rules every comparison came out "not reliably different" (section 4). The
    reference models therefore stay: TabPFN v2 with SVM-RFE k = 50 on all four biomarkers (**24.6%** balanced
    accuracy, 23.5% macro-F1), and the corrected transformer.
- **Strongest imaging-only improvement:** TabPFN with training-prior class balancing.
  - Balanced accuracy is **26.1% against 24.6%** (+1.5 points, 95% CI −0.1 to +3.2), and macro-F1 is 25.4%.
  - It is better in 4 of 5 seeds, and in 8 outer splits against 4 worse, with 13 unchanged.
  - The gain comes from MCI recall (9% → 24%), at the cost of some AD recall (54% → 50%).
  - It is **not consistent enough to adopt**: the CI touches 0, and the corrected repeated-CV test gives p = 0.54.
- **The transformer stays 5–7 points below TabPFN** in every variant (17.0–19.1%).
  - The direction is the same in 5 of 5 seeds, but the stricter corrected test used here does not confirm the gap
    (p = 0.24–0.35).
  - By the pre-registered parsimony rule, the smaller regularised transformer replaces v2 as the transformer
    reference (18.0% vs 19.1%, not different). It does not perform better.
- **Main limits, in order:** too little imaging signal for 116 subjects; unstable feature selection;
  confusion between neighbouring stages, especially MCI; a transformer that cannot use the signal at this sample
  size; and scanner site entangled with stage, which affects validity (section 5).

## 2. Protocol (final)

### Cohort and splits
- **Cohort:** 116 subjects with one scan each, so every split separates subjects, not scans. Classes: CN 20,
  SMC 20, EMCI 20, MCI 16, LMCI 20, AD 20. The model-ready inputs are byte-identical to those of the v2 run.
- **Splits:** 25 outer splits (5 folds × 5 seeds, fingerprints `3edbea19…` to `8b7f4cd9…`). They are identical for
  every model in this report and for all earlier TabPFN and transformer results.

### The 35 "locked-test" subjects
- They are inside every CV split and have been evaluated 31 times before (see `TRANSFORMER_NESTED_V2_REPORT.md`,
  section 2), so they are **not an independent test set**.
- `run_test.py` stays gated and was not run.
- The defensible protocol is to report **nested 116-subject CV as the only performance estimate**, together with
  the history of these subjects.

### What is fitted, and where
Every learned step is fitted inside the training partition of its own level:
- the SVM-RFE scalers and selection;
- the transformer scaler;
- the clinical median fill and scaling;
- the TabPFN context;
- class weights and balancing counts.

Inner-CV fits additionally hide their validation labels, and outer-test labels are removed before any fitting code
runs. The audit tested this by perturbation (`audit/audit.log`, 21 checks):
- Changing the outer-test or inner-validation rows or labels leaves every selection unchanged.
- Upstream biomarkers are computed within each subject.
- The local attention mask is atlas geometry.

### Metrics and uncertainty
- **Point estimates:** for each seed, the 5 outer test folds are pooled into one prediction per subject and scored;
  the result is averaged over the 5 seeds. Fold-level mean ± sd over the 25 outer folds is also reported.
- **The 95% CI:** a class-stratified subject bootstrap (2,000 resamples) of the seed-averaged score. It reflects
  test-subject sampling given the trained models, not training-set variability.
- **Paired comparisons** add a corrected resampled t-test over the 25 splits (Nadeau–Bengio). It accounts for the
  overlap between training sets in repeated CV.
- **"Reliably different"** requires all three: the bootstrap CI excludes 0, at least 4 of 5 seeds agree, and the
  corrected p < 0.05.
- **Sensitivity of the test:** with 116 subjects, the corrected test can detect differences of only about **5–12
  points** (median about 9; computed from the fold-level spread of each comparison).

### Pre-registration and revisions
The plan is in `derivatives/experiments/biomarker_ablation_v3/PLAN.md`. It was written before any run and revised
once before any run to follow the updated instructions. Addendum 1, the k sweep, was added before any held-out
result was inspected. All selection used inner-CV scores only.

## 3. Corrected baselines (Phase 1)

| Model | Balanced acc. [95% CI] | Fold mean ± sd | Macro-F1 | Accuracy | Recall CN / SMC / EMCI / MCI / LMCI / AD (%) |
|---|---|---|---|---|---|
| **TabPFN v2, all four, k = 50** | **24.6%** [20.5, 29.0] | 24.6 ± 8.4 | 23.5% | 25.2% | 12 / 21 / 32 / 9 / 20 / 54 |
| Transformer v2 (fixed epoch by inner-CV loss) | 19.1% [16.2, 22.1] | 19.2 ± 6.3 | 17.7% | 19.5% | 11 / 19 / 23 / 8 / 16 / 38 |

**Transformer v2 against TabPFN:** −5.5 points [−10.7, −0.6]. TabPFN is better in 5 of 5 seeds and in 15 of 21
non-tied splits, but the corrected test gives p = 0.35.

The v2 report called this "TabPFN better" using the bootstrap-and-seeds rule. **Under this phase's stricter rule
the gap is consistent in direction but not confirmed.**

## 4. Results of the controlled study

All rows use the 25 identical outer splits. "Δ" is the difference from the TabPFN baseline (rows 1–17) or from
transformer v2 (rows 18–21).

| Change | Balanced acc. [95% CI] | Fold mean ± sd | Macro-F1 | Recall CN / SMC / EMCI / MCI / LMCI / AD | Δ [95% CI] | Seeds better | Splits up / down | Corrected p |
|---|---|---|---|---|---|---|---|---|
| **TabPFN baseline** (all four, k = 50) | 24.6 [20.5, 29.0] | 24.6 ± 8.4 | 23.5 | 12 / 21 / 32 / 9 / 20 / 54 | – | – | – | – |
| **Nested choice of setting** (9 settings, inner CV) | 24.5 [20.2, 29.0] | 24.5 ± 7.3 | 23.5 | 14 / 29 / 34 / 10 / 17 / 43 | −0.1 [−3.3, +3.0] | 2/5 | 9 / 8 | 0.98 |
| **Nested choice of k** (10–75, inner CV) | 24.0 [19.6, 28.3] | 23.9 ± 8.4 | 22.8 | 12 / 22 / 35 / 4 / 23 / 48 | −0.7 [−3.4, +2.2] | 2/5 | 8 / 9 | 0.85 |
| k = 10 | 21.6 [18.0, 25.3] | 21.6 ± 9.1 | 21.0 | 16 / 24 / 26 / 8 / 22 / 34 | −3.0 [−6.4, +0.1] | 0/5 | 9 / 13 | 0.52 |
| k = 20 | 23.4 [19.1, 27.5] | 23.3 ± 11.4 | 22.9 | 17 / 23 / 29 / 11 / 23 / 37 | −1.3 [−5.2, +2.3] | 2/5 | 9 / 13 | 0.76 |
| k = 25 | 23.7 [19.7, 27.7] | 23.7 ± 9.2 | 22.8 | 17 / 19 / 36 / 6 / 19 / 45 | −0.9 [−4.4, +2.3] | 3/5 | 8 / 11 | 0.79 |
| k = 30 | 24.7 [20.2, 29.1] | 24.6 ± 9.2 | 23.8 | 12 / 21 / 32 / 15 / 20 / 48 | +0.0 [−3.4, +3.3] | 2/5 | 12 / 11 | 0.99 |
| k = 75 | 24.5 [20.1, 29.2] | 24.3 ± 7.1 | 23.4 | 11 / 19 / 29 / 14 / 18 / 56 | −0.2 [−3.2, +2.9] | 2/5 | 11 / 13 | 0.95 |
| k = 100 | 24.3 [20.1, 28.6] | 24.2 ± 8.3 | 23.4 | 14 / 20 / 24 / 14 / 18 / 56 | −0.3 [−3.0, +2.2] | 1/5 | 7 / 12 | 0.93 |
| mALFF only | 22.0 [16.9, 27.5] | 21.9 ± 7.6 | 20.3 | 29 / 14 / 5 / 10 / 25 / 49 | −2.6 [−8.7, +3.2] | 1/5 | 6 / 13 | 0.55 |
| mReHo only | 23.2 [18.6, 28.3] | 23.3 ± 8.1 | 21.8 | 13 / 27 / 40 / 8 / 12 / 40 | −1.4 [−6.0, +3.2] | 1/5 | 9 / 12 | 0.80 |
| DC_z only | 18.2 [14.2, 22.7] | 18.3 ± 7.2 | 17.4 | 21 / 17 / 5 / 21 / 22 / 23 | −6.4 [−11.3, −1.2] | 0/5 | 5 / 16 | 0.27 |
| FC strength only | 21.8 [17.2, 26.5] | 21.7 ± 8.6 | 20.9 | 14 / 25 / 27 / 8 / 20 / 37 | −2.9 [−8.3, +2.6] | 1/5 | 9 / 13 | 0.63 |
| mALFF + mReHo (local activity) | 25.6 [20.8, 30.5] | 25.7 ± 8.6 | 24.2 | 15 / 24 / 35 / 8 / 19 / 53 | +1.0 [−2.9, +4.8] | 3/5 | 13 / 10 | 0.83 |
| DC_z + FC (connectivity) | 21.3 [16.9, 26.1] | 21.3 ± 7.5 | 20.5 | 16 / 18 / 15 / 14 / 30 / 35 | −3.3 [−8.0, +1.6] | 2/5 | 8 / 15 | 0.56 |
| **Baseline + class balancing** | **26.1** [21.6, 30.9] | 26.1 ± 9.1 | **25.4** | 12 / 20 / 31 / **24** / 20 / 50 | **+1.5** [−0.1, +3.2] | **4/5** | 8 / 4 | 0.54 |
| Learning curve: 75% of training (about 70 subjects) | 22.7 [18.5, 27.1] | 22.7 ± 10.1 | 22.4 | 18 / 29 / 28 / 11 / 15 / 35 | −1.9 | 2/5 | 9 / 15 | 0.77 |
| Learning curve: 50% of training (about 46 subjects) | 19.9 [16.5, 23.3] | 19.8 ± 9.0 | 19.4 | 15 / 21 / 16 / 16 / 19 / 32 | −4.8 | 0/5 | 8 / 15 | 0.44 |
| **Transformer v2** (reference) | 19.1 [16.2, 22.1] | 19.2 ± 6.3 | 17.7 | 11 / 19 / 23 / 8 / 16 / 38 | – | – | – | – |
| Transformer, smaller and regularised | 18.0 [15.0, 21.0] | 17.9 ± 5.3 | 15.4 | 24 / 4 / 18 / 4 / 21 / 37 | −1.1 [−4.8, +2.4] | 2/5 | 9 / 15 | 0.72 |
| Transformer, unweighted loss | 17.0 [14.0, 20.2] | 17.0 ± 7.0 | 15.1 | 12 / 13 / 20 / 0 / 24 / 33 | −2.1 [−4.6, +0.4] | 1/5 | 5 / 13 | 0.49 |
| Transformer on the nested-chosen setting | 18.6 [15.4, 22.0] | 18.7 ± 5.1 | 15.9 | 9 / 23 / 10 / 8 / 19 / 43 | −0.5 [−4.3, +3.3] | 2/5 | 8 / 11 | 0.90 |

**Notes on the table:**
- The individual settings are exploratory. After Holm adjustment every p-value is 1.00.
- Accuracy, the full confusion matrices and every fold-level value are in `analysis/`.

**Pre-registered decisions** (`analysis/manifest.json`):

| Decision | Result |
|---|---|
| Nested choice of setting replaces the baseline | **No** |
| Nested choice of k replaces the baseline | **No** |
| TabPFN class balancing adopted | **No** |
| Transformer drops class weighting | **No** |
| Transformer reference | **Smaller model,** by the parsimony rule because it is not different from v2 |
| TabPFN baseline exceeds the within-site null | **Yes** (section 6) |

**Selection behaviour:**
- **Which setting inner CV chose:** inner CV picked 7 different settings across the 25 splits: FC 6, mReHo 5,
  baseline 5, k = 100 3, local pair 3, k = 25 2, mALFF 1.
- **Which k inner CV chose:** k = 20 and k = 50 7 times each, k = 30 and k = 75 5 times each, k = 10 once.
- **Inner vs held-out scores:** TabPFN's inner-CV scores, fitted on about 62 subjects, were 1–4 points *below* its
  held-out scores (on about 93), in every setting. This matches the learning curve.

## 5. Why validation performance is low (ranked by strength of evidence)

**1. Too little imaging signal for 116 subjects** (strongest evidence).
- No single feature differs by stage after FDR correction (0 of 984); the best single-feature AUC is 0.78.
- The TabPFN learning curve is still rising: 19.9% → 22.7% → 24.6% at 46, 70 and 93 training subjects.
- Every inner CV, with fewer subjects, scores 1–4 points lower.
- No transformer variant's mean validation loss gets below the uniform-guess level (minimum 1.784–1.802 against
  ln 6 = 1.792).

**2. Unstable supervised feature selection.**
- SVM-RFE choices barely repeat across folds: Jaccard 0.10–0.24 on all four biomarkers. At k = 50, only 19 features
  are chosen in at least half of the 25 fits.
- Inner CV cannot settle on a setting: it picked 7 different ones across the 25 splits.
- Between k = 25 and 100, accuracy is flat (23.7–24.7%). Fewer features (k = 10) is worse (21.6%).
- One-biomarker pools are steadier (Jaccard 0.29–0.38) but less accurate.

**3. Confusion between neighbouring stages, especially MCI.**
- MCI recall is the lowest or near-lowest class in most imaging models: TabPFN baseline 9%, transformer v2 8%,
  range 0–24% across all variants.
- AD is recognised best: 54% for the TabPFN baseline, 33–56% across most variants.
- Errors spread across the neighbouring stages (see `analysis/fig_confusion.png`).
- Class imbalance is mild (MCI 16 vs 20). Balancing raises MCI recall to 24% but barely moves the overall score.

**4. The transformer cannot use the signal at this sample size.**
- At baseline size it overfits after about 12 epochs: training loss falls to 1.32 while validation loss rises to
  2.30.
- The smaller model no longer overfits (validation loss 1.80, training loss 1.75 at epoch 60). But it then learns
  almost nothing: training balanced accuracy is 24% at epoch 60.
- Removing class weights and changing features do not help.
- **Training-versus-validation gap for TabPFN:** not informative. It uses the training subjects as context, so
  training accuracy is 100% by design.

**5. Scanner site is entangled with stage** (mainly a threat to validity, see section 6).

## 6. Scanner site and acquisition

- **Site predicts stage strongly.** Site and stage are strongly associated (13 sites, χ² p = 3.9 × 10⁻⁵, Cramér's
  V = 0.44), and 7 sites lack at least half of the stages.
- **Site alone beats every imaging model.** A site-only lookup rule reaches **30.8%** balanced accuracy under the
  same CV. It is a confound diagnostic, not a model.
- **The imaging features carry site information.** 31 of 984 features differ by site after FDR, against 0 by stage.
- **TabPFN is not just learning site:**
  - Retrained with labels shuffled *within* each site, which keeps the site–stage link, the baseline pipeline
    scores 17.2–20.1% (mean 18.3%).
  - With labels shuffled across all subjects it scores 13.9–18.2%.
  - The real baseline (24.6%) is above every within-site null run, by 6.3 points on average, so its accuracy is
    not explained by site-label structure alone.
  - Acquisition differences within sites are not ruled out: scan year differs by stage (all SMC scans are from
    2013), and TR differs by stage.
- **Harmonising site lowered accuracy.** ComBat fitted inside training folds (existing `leakage_safe_v1` result, 15
  splits) gave −4.0 points [−9.2, +0.9]. Because site and stage are entangled, harmonising also removes stage
  information.
- **No transparent correction preserves both the cohort and the full signal.** The defensible step is to report the
  site-only rule and the within-site null next to every model, as done here.

## 7. Separate experiment: imaging plus 8 comorbidity variables (not the primary imaging result)

The 8 variables are age, sex, APOE4 count, GDS total, and four medical-history flags: cardiovascular,
endocrine-metabolic, psychiatric and smoking. MMSE and CDR-SB are not included.

| Model | Balanced acc. [95% CI] | Macro-F1 | Recall CN / SMC / EMCI / MCI / LMCI / AD | Δ vs imaging-only | Seeds better | Corrected p |
|---|---|---|---|---|---|---|
| TabPFN, imaging + comorbidities | 27.5 [22.8, 32.3] | 25.8 | 16 / 21 / 32 / 9 / 20 / 67 | +2.8 [−0.3, +5.8] vs TabPFN | 4/5 | 0.37 |
| Transformer, imaging + comorbidities | 25.7 [21.5, 29.8] | 24.4 | 35 / 28 / 18 / 5 / 21 / 47 | +6.6 [+2.2, +11.5] vs transformer v2 | 4/5 | 0.32 |
| TabPFN, comorbidities only | 27.0 [22.8, 31.0] | 18.8 | 50 / 17 / 6 / 0 / 1 / 88 | +2.4 [−3.8, +8.1] vs TabPFN | 4/5 | 0.60 |

**What this shows:**
- Comorbidities mainly separate the extremes. On their own they find CN and AD but almost never MCI or LMCI.
- Adding them helps the transformer most, but none of these gains passes the corrected test.
- The TabPFN rows reuse the existing runs on the same folds (`model_training/tabpfn/cv116_*`).

## 8. Negative results, failed runs and incidents

**Negative results:**
- Every feature count (k = 10–100) and every biomarker pool.
- Both nested selection procedures, which match the baseline.
- The unweighted transformer.
- The transformer on the nested-chosen setting.
- ComBat (existing results).
- Each single biomarker is no better than all four combined; DC_z alone is close to chance (18.2%).

**Incidents** (all logged in `derivatives/experiments/biomarker_ablation_v3/INCIDENTS.md`). None affected a reported
result:

| What happened | Cause | Effect on results |
|---|---|---|
| A crash during the FC-only configuration | Out of memory (segmentation fault) | 20 splits saved; resumed |
| A failed restart | Out of memory (`MemoryError`) at start-up | Resumed after other programs were closed |
| A progress-message bug for the outer-only runs | Code bug in a print statement | None (the record had already been written) |
| A bug that applied the chosen feature setting to every transformer variant | Code bug | 5 invalid "small" records moved to a quarantine folder and excluded; the variant was rerun |

**Other disclosures:**
- The pre-run audit ran on an earlier revision of the TabPFN runner. The functions it tested are unchanged.
- The pinned runner script was never edited after its first result. The outer-only configurations ran through a
  separate driver that calls its unchanged functions.

## 9. Single most promising next experiment (proposed, not run)

**TabPFN on all 492 mALFF + mReHo features with no supervised feature selection,** pre-registered against the
baseline on the same 25 splits with inner CV. Class balancing would be a pre-declared secondary.

**Why this one:**
- It removes the least stable learned step. Selection is the most variable part of the pipeline (cause 2), and the
  flat k-curve shows the number of selected features adds nothing.
- The pool comes from inner-CV evidence, not held-out results. Of all 13 settings, mALFF + mReHo had the highest
  mean inner-CV score (22.0% vs the baseline's 21.4%).
- TabPFN v2 accepts up to 500 features.

**Cost and expectation:**
- About 65 s per fit, roughly 2 hours with inner CV, and about 0.9 GB of memory per fit.
- Any gain is likely to be smaller than the roughly 8–10 points this design can detect with 116 subjects. A realistic
  outcome is "no reliable difference", and it should be reported that way.

## 10. Limitations

- **Small sample:** 116 subjects, with about 23 per test fold. Changes of less than about 5–12 points cannot be
  confirmed.
- **Multiple comparisons:** many comparisons were made. Individual settings are Holm-adjusted, and only the two
  pre-declared nested procedures were primary.
- **The confidence interval** does not include training-set variability. The corrected t-test does, and it is
  conservative.
- **Site and acquisition** cannot be separated from stage in this cohort.
- **The 35 "locked" subjects** are not an independent test set (section 2).
- **TabPFN settings** are the package defaults and were not tuned. The transformer settings were fixed before this
  phase.

## 11. Reproducibility

```
python scripts/experiments/biomarker_ablation_v3/audit_v3.py
python scripts/experiments/biomarker_ablation_v3/run_tabpfn_configs.py --configs A_all4 A_all4_k25 A_all4_k100 A_mALFF A_mReHo A_DCz A_FC A_local A_conn
python scripts/experiments/biomarker_ablation_v3/run_tabpfn_outer_only.py --configs P_site1 P_site2 P_site3 P_site4 P_site5 P_glob1 P_glob2 P_glob3 L_frac50 L_frac75
python scripts/experiments/biomarker_ablation_v3/run_transformer_variants.py --variants small unweighted comorb8 chosen
python scripts/experiments/biomarker_ablation_v3/run_k_sweep.py
python scripts/experiments/biomarker_ablation_v3/analyze_v3.py
```

- **Resuming and duplicates:** every runner is resumable, writes each split once, and refuses to run twice at the
  same time.
- **Guards:** the TabPFN runner stops unless the baseline reproduces exactly; it reproduced in 25 of 25 splits. The
  transformer runner stops unless it reproduces v2 exactly, which it did.
- **Hashes:** the manifests pin the scripts' hashes.
- **Compute:** about 213 minutes.

## Files

All results are in `derivatives/experiments/biomarker_ablation_v3/`. This is a new folder; earlier results are
untouched.

| What | Where |
|---|---|
| Pre-registration and addendum | `PLAN.md` |
| Audit | `audit/audit.log`, `audit/audit.json` |
| Incidents | `INCIDENTS.md` |
| TabPFN records | `grid/<configuration>/` |
| Transformer records | `transformer/<variant>/` (invalid records in `transformer/_invalid_…/`) |
| Comparison table and all models | `analysis/comparison_table.csv`, `analysis/all_models.csv` |
| Fold-level results | `analysis/fold_level.csv` |
| Paired comparisons | `analysis/paired_comparisons.csv` |
| Confusion matrices | `analysis/confusion_matrices.csv` |
| Transformer epoch curves | `analysis/epoch_curves_*.csv` |
| Nested choices per split | `analysis/nested_choice_per_split.csv` |
| Selection stability | `analysis/settings_inner_vs_heldout_and_stability.csv` |
| Transformer fit diagnostics | `analysis/transformer_fit_diagnostics.csv` |
| Text summary | `analysis/summary.txt` |
| Figures | `analysis/fig_settings.png`, `fig_k_curve.png`, `fig_learning_curves.png`, `fig_confusion.png` |
