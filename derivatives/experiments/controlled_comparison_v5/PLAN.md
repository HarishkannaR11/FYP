# Pre-registered plan: controlled comparison v5

Written 2026-10-22, before any new run and before this re-analysis. All results are exploratory cross-validation
analyses on the fixed 116-subject cohort. The 35 "locked" subjects are part of the CV and have been evaluated
before, so they are not untouched. The gated locked-test script is not run.

## Corrections to the brief's reference numbers (from the stored files)

| Reference | As stated in the brief | Stored values |
|---|---|---|
| TabPFN imaging + MMSE/CDR-SB (`tabpfn_svmrfe50_clin10`) | "balanced accuracy 43.4, macro-F1 42.6" | **accuracy 43.4, balanced accuracy 42.6** |
| "MMSE + CDR-SB only" | "46.4 / 45.9" | 46.4 / 45.9 are accuracy / balanced accuracy of `ceiling_cdr_mmse`: **logistic regression on MMSE + CDR global + CDR-SB**. The MMSE + CDR-SB-only logistic regression (`ceiling_mmse_cdrsb`) is 43.6 / 42.5 |

The brief allows only TabPFN and the transformer, so those logistic-regression references are not used. The
MMSE + CDR-SB reference is rerun with TabPFN (step 4).

## Steps

1. **Verification** (read-only):
   - the label mapping (CN = 0, SMC = 1, EMCI = 2, MCI = 3, LMCI = 4, AD = 5);
   - class counts;
   - one scan per subject;
   - the 5 fold fingerprints;
   - that every compared source has identical test indices for all 25 splits;
   - that stored labels equal `y`;
   - that balanced accuracy, macro-F1 and accuracy recomputed by the project code match scikit-learn and each
     record's stored metrics.

   Nothing is regenerated.
2. **Stored imaging-only candidates vs the stored baseline** (no new runs):
   - baseline: TabPFN, SVM-RFE k = 50 of 984 features;
   - candidate: the 492 mALFF + mReHo features, no selection (v4);
   - candidate: the baseline with class balancing (training-prior `balance_probabilities`, v3).
3. **Non-circular multimodal** (no new run): the stored TabPFN imaging + 8 comorbidities (age, sex, APOE4, GDS and
   four medical-history flags), compared with the baseline.
4. **Clinical references,** clearly labelled and never an fMRI result:
   - **new run:** TabPFN on MMSE + CDR-SB only (2 features; baseline TabPFN settings; `random_state` seed + fold;
     missing CDR-SB, 2 subjects, handled inside TabPFN using its training context only; outer-test labels hidden);
   - **stored:** TabPFN imaging + 8 comorbidities + MMSE + CDR-SB.

   Comparisons: each against the baseline, and the stored combined model against MMSE + CDR-SB alone (does imaging
   add anything to the clinical scores?).
5. **Confirmation:** only if a candidate is promising. Untouched data would be needed. Only 13 never-used subjects
   exist (6 CN, 6 EMCI, 1 SMC), too few for a six-class test, and any independent cohort would need authorisation.
   No confirmation run is performed in this plan.

No hyperparameter is selected anywhere, so no inner loop is needed. The stored v3 baseline records include a
3-fold inner CV, which is not used.

## Statistics and decisions (fixed now)

- **Statistics:**
  - seed-pooled metrics averaged over 5 seeds;
  - 95% CIs from a class-stratified subject bootstrap (2,000);
  - paired differences on the same resamples;
  - seeds agreeing;
  - splits better, worse and tied;
  - the corrected resampled t-test (Nadeau–Bengio) for repeated-CV dependence;
  - Holm adjustment across the 3 planned candidate comparisons of steps 2–3.
- **Promising** (practical threshold, not significance):
  - the balanced-accuracy gain is at least 3.0 points;
  - the macro-F1 change is at least 0;
  - mean intermediate-stage recall (SMC, EMCI, MCI, LMCI) does not fall by more than 3 points.
- **Reliable:** the bootstrap CI excludes 0, at least 4 of 5 seeds agree, and the Holm-adjusted corrected p < 0.05.
- **Stopping rule:** steps 2–3 are the justified candidate experiments. If none is promising, stop: no further
  experiment and no confirmation.
- **Clinical references (step 4)** are never candidates for the fMRI claim.

## Outputs

| What | Where |
|---|---|
| New run (MMSE + CDR-SB TabPFN) | `clinical_reference/grid/` |
| Verification | `verification/` |
| Analysis | `analysis/` |
