# Limitations and Future Work

Written to be read before any claim is made about this project. Every figure
cited is verified against the repository.

---

## 1. Cohort size and imbalance

**116 subjects across six ordinal classes — roughly 19 per class.**

| Stage | Subjects |
|---|---:|
| CN | 20 |
| SMC | 20 |
| EMCI | 20 |
| **MCI** | **16** |
| LMCI | 20 |
| AD | 20 |

MCI falls short because only 16 MCI subjects have valid preprocessing,
biomarkers, normalisation and QC. It was **not** padded to 20 with repeat
visits — doing so would reintroduce within-subject dependence and break the
one-acquisition-per-subject property that makes subject-grouped validation
trivial here.

The deeper constraint is that **ADNI is longitudinal**. MCI has 34 valid
*acquisitions* but only 16 *people*. Searching ADNI by diagnostic group
returns more visits, not more subjects: of 12 additional MCI scans
downloaded, 10 were repeat visits of subjects already in the cohort and only
2 were new people. Expanding the cohort requires filtering by **Subject ID**
and excluding those already held.

---

## 2. Missing and incomplete records

### Clinical and comorbidity data

Verified coverage **within the 116-subject cohort**:

| Variable | Coverage | Gap |
|---|---:|---|
| MMSE, CDR global | **116/116** | — |
| GDS | 115/116 | `130_S_4990` (AD) |
| APOE ε4 count | 115/116 | `019_S_5180` (SMC) |
| CDR sum-of-boxes | 114/116 | the two MCI subjects below |
| Medical history (19 flags) | 114/116 | the two MCI subjects below |
| **All of the above** | **113/116** | |

The earlier gap — LMCI at 0/20 — was an acquisition gap, not a data
limitation: `Comorbidity/Tables_12Oct2026_LMCI.zip` supplies the LMCI
assessment tables and LMCI is now complete. `derivatives/clinical_cleaned/`
predates that download and still shows LMCI blank; use
`derivatives/comorbidity_features/model_clinical_116.csv` instead.

Two subjects (`007_S_2394`, `100_S_0296`) entered through a separate download
and appear in no ADNI assessment table in the project. Their MMSE, CDR and
GDS are recoverable from the undated `idaSearch` export — so those rows are
**not scan-matched** — but they have no CDR sum-of-boxes and **no medical
history at all**.

Usable *n* therefore depends on which fields a model takes: MMSE + CDR gives
116, adding GDS 115, adding APOE 114, adding the medical-history flags 113,
adding CDR-SB 112.

### The comorbidity variables are weaker than their count suggests

Of the 19 medical-history flags, five are too rare at this *n* to carry
signal (drug abuse 2, alcohol abuse 4, hematopoietic 10, hepatic 11,
malignancy 25) and three are near-constant at ~75% (major surgical
procedures; head/eyes/ears/nose/throat; musculoskeletal). Roughly 6–8 have
usable variance. They are also coarse by construction — a single yes/no box
per body system at screening, so "cardiovascular = 1" spans treated
hypertension to prior infarction with no distinction.

Medical history is also the loosest-matched block: median 209 days from the
MRI against 0 days for the cognitive and mood scores, because it is collected
at screening rather than at each visit.

### MMSE and CDR cannot be used as model features

They are inputs to ADNI's diagnostic criteria, which assign the stage being
predicted. Global CDR alone reaches **0.386** balanced accuracy on this
cohort (5-fold CV, chance 0.167) against the rs-fMRI model's 0.307 — a
measurement of the enrolment rule, not of the biology. CDR sum-of-boxes is
the finest-grained form of the same criterion and is worse in this respect.
`GDS_total`, the `MH_*` flags and `APOE4_count` do not enter the criteria and
are the usable covariates.

No imputation has been performed anywhere. `-4` is ADNI's missing code and is
stored as missing, never as 0; a blank is never read as "no disease". The 116
conflicting duplicate records found by the earlier builder remain flagged for
review rather than resolved.

### Imaging

- **No T1-weighted scan for any subject**, forcing direct EPI→MNI
  registration, template-level tissue priors, and CNR that is not computable.
- **CSF regressor absent in all 167 acquisitions** — the design is 27
  regressors, not the intended 28.
- **8 acquisitions unrecoverable** at the configured band-pass (TR ≈ 6.02 s,
  Nyquist 0.083 Hz < 0.10 Hz). Recovering them would require changing the
  filter, which would break comparability with the other 167.
- **2 acquisitions pre-excluded**: one 7-volume truncation, one byte-identical
  duplicate.

---

## 3. No statistically significant findings

Across five analyses, **nothing survives multiple-comparison correction**:

| Analysis | Family | Discoveries |
|---|---|---:|
| Pairwise ROI × 15 comparisons | 3,690 per biomarker | **0** |
| Pairwise FC edges | 30,135 per comparison | **0** of 452,025 |
| Intra- vs inter-class permutation | 4 | **0** (all q ≥ 0.092) |
| ROI × stage one-vs-rest | 1,476 per biomarker | **0** |
| Stage-level multivariate one-vs-rest | 24 | **0** |

The cause is **effect size relative to between-subject variance**, not a
p-value resolution limit — an earlier claim to the contrary was wrong and has
been withdrawn ([FINAL_STATISTICAL_AUDIT.md](../derivatives/INTRA_INTER_CLASS_HEATMAPS/v2_cohort116/FINAL_STATISTICAL_AUDIT.md) §2). mReHo's smallest raw
p (9.65 × 10⁻⁵) is only 2.8× above its BH rank-1 threshold. Between-stage
variance is about **0.23×** within-stage variance per ROI.

**This does not prove absence of biological difference**, and it does not
mean the features lack predictive information — significance and prediction
are different questions.

---

## 4. Confounds not yet addressed

### Residual motion

After the 27-regressor nuisance design, **129/246 (mALFF) and 131/246
(mReHo) ROIs still correlate with mean FD at p < 0.05**, against ~12 expected
by chance — roughly 10× chance. DC_z is less affected (39/246).

Motion must enter any model as a covariate. It has not been.

### Scanner site

14 sites, unevenly distributed. **One site contributes 40% of AD acquisitions
and two contribute 68%**, while no single site exceeds 19% of CN. Any CN–AD
contrast is partly a site contrast. No harmonisation (ComBat or equivalent)
has been applied.

A χ² test of site × stage independence returns p < 0.0001, but **96% of cells
have expected counts below 5**, so that p-value is unreliable and is not
quoted as a result. The descriptive imbalance above is the defensible
statement.

### Protocol heterogeneity

The cohort mixes 140-volume `Resting State fMRI` with 200-volume `Extended
Resting State fMRI` and `Axial rsfMRI (Eyes Open)`. All share TR ≈ 3 s so the
band-pass is valid throughout, but scan length affects estimator precision
for FC and ReHo. One acquisition is from a **GE** scanner in an otherwise
Philips/Siemens cohort.

---

## 5. Overfitting risk

At the smoke-test configuration (`d_model=64, heads=4, layers=2`) the
implemented encoder already has **196,374 parameters** for **116 subjects**.
Realistic configurations are larger.

**A high-capacity model is not justified by the absence of univariate
significance.** That reasoning is backwards: weak signal and small *n*
together are precisely the regime where a flexible model fits noise. Any
benefit must be demonstrated on held-out data against simple baselines
(majority class, ordinal logistic regression on a handful of features), not
asserted from architecture.

---

## 6. Cross-sectional, not longitudinal

Despite ADNI being longitudinal, this analysis is **cross-sectional**. Labels
are the diagnostic group at the scanned visit; the design predicts *stage at
time of scan*, not *future progression*.

A genuine progression model would need each subject's trajectory across
visits — data that exists in ADNI and partly in this download (167
acquisitions from 129 subjects) but is deliberately discarded by the
one-acquisition-per-subject rule. Reinstating it would require a model
handling repeated measures and within-subject dependence.

**No claim of progression prediction is supported by the current design.**

---

## 7. Leakage and validation

### Handled

- **One acquisition per subject**, so no subject straddles a fold.
- **Within-acquisition normalisation** uses only that acquisition's own 246
  values — no cross-subject statistic, no label. Leakage-free by
  construction.
- **Cohort selection is label-blind**: ordering by `(subject, session, run)`
  only; no QC metric, biomarker value or stage label enters it.
- **Cohort fingerprinting** prevents statistics from one cohort being reused
  for another.

### Not yet handled

- **No cross-validation has been run**, nested or otherwise.
- **No cross-subject scaler has been fitted.** If one is introduced it must
  be fitted on **training folds only**.
- **Hyperparameter selection** would need an inner loop to avoid optimistic
  bias; with ~19 subjects per class, nested CV will have high variance.
- **No external validation set.** Everything comes from ADNI, so
  generalisation to other cohorts, scanners and populations is untested.

---

## 8. Future work

Ordered by what most constrains the project now.

1. **Acquire the missing clinical tables**, especially LMCI. Without them the
   comorbidity-aware component — the project's main claimed novelty — cannot
   be evaluated on all six stages.
2. **Expand the cohort by subject, not by scan.** Filter ADNI by Subject ID,
   excluding the 129 already held. Four more valid MCI subjects would bring
   MCI to 20.
3. **Build the local attention mask from the atlas** — ROI centroid distances
   or voxel-boundary adjacency computed from `BN_Atlas_246_2mm.nii.gz`. One
   of the three attention branches cannot be used as designed without it, and
   deriving it from the atlas is measurement rather than assertion.
4. **Add motion and site as covariates**, or harmonise across sites, before
   any modelling.
5. **Establish baselines first**: majority class, ordinal logistic regression
   on a small feature set, and a linear model on the 246-ROI vectors. Report
   them before any transformer result.
6. **Implement subject-grouped nested cross-validation** with
   training-fold-only scaling.
7. **Run ablations** that test the specific claims: remove each attention
   branch; replace the learned fusion with a fixed average; remove the
   comorbidity block; replace the ordinal head with plain multi-class. Each
   isolates one design decision.
8. **Report calibration, not only accuracy.** The Bayesian motivation is that
   borderline subjects get wide uncertainty; that is a calibration claim and
   needs calibration metrics.
9. **Consider longitudinal modelling** if repeat visits are reinstated — a
   different and arguably more clinically useful problem.
10. **Seek external validation** on a non-ADNI cohort before any claim of
    generalisation.

---

## 9. What must not be claimed

- That any result here is **statistically significant** — nothing reaches
  q < 0.05.
- That a model has been **trained**, or any accuracy, sensitivity,
  specificity, AUC or F1 — none exists.
- That the null results **prove** no biological difference exists.
- That a complex model is **justified** because simple tests were
  non-significant.
- That the pipeline **predicts progression** — the design is cross-sectional.
- That all 246 ROIs are **individually disease-relevant** — they are retained
  as a complete extraction space, with importance to be determined by a
  model that has not yet been built.
- That the Mann-Whitney test has a **p-value floor near 10⁻⁵** — this earlier
  claim was false and is withdrawn.
