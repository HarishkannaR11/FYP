# Project Status + Plan

Written 2026-10-17 for Dhananchezhiyan M.
Sources: `md files/INSTRUCTIONS.md`, `README.md`, `docs/`, `derivatives/model_training/*`, `scripts/model/*`,
git history, `Review1_presention_fyp.pdf`, `r2_presentation (4).pdf`.

Companion file: `md files/FEATURE_SELECTION_PROCESS_AND_QUANTUM_METHODS.md` (the detailed process, before/after
feature counts, quantum methods, and the chosen one).

---

## 1. Short answer

* **Data side is finished.** Raw ADNI scans are preprocessed and turned into Brainnetome-246 features. A clean
  116-subject dataset is ready in `derivatives/model_ready/`.
* **Model side has been started but does not work well yet.** Best honest result: about **31 % balanced accuracy
  on 6 stages** (chance 16.7 %) and **20 %** on the untouched test set.
* **My task (from mam):** feature selection. PCA and LDA/LRA are "normal" methods. She expects an **advanced**
  method, including **quantum feature selection**. I must **list the advanced methods, choose the suitable one,
  justify it, compare it fairly with the others, and report the best accuracy.**
* **Her 96 % target is very unlikely to be reachable honestly** with 116 subjects and 6 stages. A number that
  high usually means a leak. Ask her what exactly it refers to (section 8).

---

## 2. What the three documents say

| Document | Where the project was |
|---|---|
| **Review 1** (30 slides) | Proposed 6-stage staging (CN, SMC, EMCI, MCI, LMCI, AD) from rs-fMRI using ALFF, ReHo, DC, FC, then Multi-Attention Transformer, meta-learning, comorbidity fusion, Bayesian ordinal output, dashboard. Done: DICOM to NIfTI only (about 30 %). |
| **Review 2** (36 slides) | Modules 1 to 3 complete: 165 scans preprocessed (127 subjects), Brainnetome-246 features. Findings: FC_Strength = DC/245 (removed), no univariate stage signal, biomarkers share only 6 to 11 % variance. Modules 4 to 6 not started, nothing trained. |
| **Repository now** | Went beyond Review 2: cohort re-cut to **116 subjects**, transformer built and trained, baselines, ablations and a held-out test done. README still says "no model trained" (out of date). |

Roles on the slides: Harishkanna = preprocessing + transformer; **Dhananchezhiyan (me) = Meta-Bayesian module +
dashboard** (Review 2) / Modules 3 and 6 (Review 1); Nagindhar = dataset and clinical cleaning.
Feature selection is a new assignment on top of that.

---

## 3. What is completed

| # | Module | Status | Where |
|---|---|---|---|
| 1 | Data audit, DICOM to NIfTI, BIDS | Done | `audit/`, `scripts/data_prep/`, `NIfTI_/` |
| 2 | Preprocessing (motion correction, EPI to MNI without T1, 4 mm, 6 mm smoothing, 27 regressors, band-pass 0.01 to 0.10 Hz, QC) | Done, 167 acquisitions (8 rejected: TR 6.02 s) | `pipeline/`, `derivatives/fsfast/` |
| 3 | Brainnetome-246: mALFF, mReHo, DC_z, 246x246 Fisher-z FC, within-scan normalisation | Done | `derivatives/biomarkers_normalized/` |
| 4 | Clinical / comorbidity cleaning | Done with gaps (LMCI has no MMSE, CDR, GDS or age) | `derivatives/clinical_cleaned/`, `Comorbidity/` |
| 5 | Statistics and heatmaps | Done. **Nothing reaches q < 0.05** (smallest q = 0.139) | `derivatives/INTRA_INTER_CLASS_HEATMAPS/` |
| 6 | Model-ready dataset, 116 subjects, one scan each | Done | `derivatives/model_ready/` |
| 7 | Leak-free split: 81 training pool, 35 locked test, 5-fold CV, seed 20261015 | Done | `derivatives/model_ready/splits/` |
| 8 | Multi-Attention Transformer, 24 unit tests | Built **and trained** | `scripts/model/multi_attention_transformer.py` |
| 9 | Baselines, MLP control, ablations, ROI importance, repeated CV (5 seeds x 5 folds), held-out test | Done | `derivatives/model_training/` |
| 10 | Clinical feasibility check | Done: **CDR-SB must NOT be used** (rho = 0.85 with the label); APOE4 is the safe covariate | `clinical_feasibility.log` |

**Not done:** meta-learning (MAML), comorbidity fusion, Bayesian ordinal head and uncertainty (ordinal head
collapses), dashboard, **any feature selection or PCA**, LMCI clinical data, README update.

---

## 4. Current model results

Chance for 6 balanced classes is 0.167. Source: `derivatives/model_training/`.

| Experiment | Balanced accuracy | Note |
|---|---:|---|
| Majority-class baseline | 0.167 | floor |
| Logistic regression on ROI features | 0.18 to 0.24 | |
| Transformer (best config) | 0.311 +/- 0.030 | 5-fold CV |
| Transformer, ordinal head | 0.228 | collapses to 1 to 2 classes |
| **MLP on DC_z only (chosen "frozen" model)** | **0.307 +/- 0.029** | 25 folds, 5 seeds |
| MLP on all four biomarkers | 0.278 +/- 0.012 | worse than DC only |
| **Final held-out test (35 subjects, used once)** | **0.200** | MAE 1.83 stages, QWK 0.15 |

Reading:
* A simple MLP matches the transformer. With 81 training subjects the big model has no advantage.
* The features carry **very little usable stage signal**: ROI importance is no better than a label-shuffled
  control, and top-ROI lists do not repeat across folds.
* One test subject moves balanced accuracy by about 3 points, so 0.20 vs 0.31 is within noise. Both are weak.

**Core problem: too many features for too few subjects** (246 to 30,135 features vs 81 training subjects).

---

## 5. Data available for my task

There are no images to handle, only **feature tables**, one row per subject, in `derivatives/model_ready/`:

| File | Content |
|---|---|
| `X_regional.npy` | 116 x 246 regions x 3 biomarkers (mALFF, mReHo, DC_z) |
| `X_fc.npy` | 116 x 246 x 246 Fisher-z connectivity matrices |
| `X_fc_roi_unsigned.npy`, `X_fc_roi_dispersion.npy` | 116 x 246 per-region FC summaries |
| `y.npy`, `stages.npy`, `subjects.npy` | labels (0 to 5 = CN, SMC, EMCI, MCI, LMCI, AD) and subject IDs |
| `splits/split_assignment.csv` | fixed split: 81 training pool, 35 test; plus fold information |

Per stage: CN 20, SMC 20, EMCI 20, **MCI 16**, LMCI 20, AD 20 = 116. **Everything needed is on disk.**

**Environment check (Windows Python 3.13.9):** scikit-learn 1.9.1, numpy, scipy, pandas, xgboost, lightgbm are
installed. **No quantum library (qiskit, pennylane, dwave-ocean) and no torch is installed** in this Python, so
quantum work needs `pip install` first. Versions to be verified before use, per INSTRUCTIONS.md.

---

## 6. What mam is asking, in plain words

**Feature selection / dimensionality reduction** = shrink the many columns to a small set of informative ones
before the classifier sees them.

| Level | Methods | Mam's view |
|---|---|---|
| "Normal" | PCA, LDA, LRA (low-rank approximation) | baseline, not enough |
| **"Advanced"** | see section 7 | **list these, pick the suitable one** |
| **Quantum** | quantum feature selection (QUBO with QAOA / annealing, quantum-inspired) | wants it listed and considered |

Her requests, as I understand them:
1. List the advanced methods (section 7).
2. From that list, **select the suitable one** and give reasons (done in the companion file).
3. Analyse all of them and report the one with the highest accuracy.
4. Reach about 96 % (see sections 8 and 9 for why this needs a conversation).

### Honest caveats
1. **PCA/LDA/LRA are not stage-aware** (PCA, LRA) or very data-hungry (LDA with 81 subjects). They may keep
   person, scanner or motion variation instead of stage signal.
2. **Reduction must be fitted inside each CV fold, on training subjects only.** Fitting on all 116 subjects first
   leaks test information and inflates accuracy.
3. **The ceiling comes from the data.** Five independent analyses found no stage signal that survives correction.
4. **"Quantum" gives no accuracy advantage by itself.** On a laptop it is simulated; at this problem size a
   classical solver finds the same answer. Its value here is as a principled *formulation* of feature selection
   (maximise relevance, minimise redundancy), not as a speed or accuracy miracle.

---

## 7. Advanced methods: the full list

### A. Normal (baselines, already known to mam)
PCA, LDA, LRA / truncated SVD.

### B. Advanced classical: feature extraction (new combined features)
| Method | Idea | Fit for our data |
|---|---|---|
| Kernel PCA | PCA after a non-linear kernel | possible, tuning heavy |
| Sparse PCA | PCA whose components use few original features, so it is readable | good: gives named ROIs |
| ICA | independent components | common in fMRI, medium fit |
| PLS / PLS-DA | components chosen to predict the label | good: supervised |
| NMF | non-negative parts | poor: our features can be negative |
| Autoencoder / concrete autoencoder | neural compression / learned feature picking | poor: 81 subjects too few |
| UMAP / Isomap | manifold embedding | exploratory only, unstable at this size |

### C. Advanced classical: feature selection (keeps original features)
| Method | Idea | Fit |
|---|---|---|
| ANOVA F-test / mutual information | rank by relevance | good fast filter |
| **mRMR** | relevant to the label, not redundant with each other | **very good** |
| ReliefF | rank by neighbour-based relevance | good for small n |
| L1 / elastic net | sparse logistic regression | good |
| SVM-RFE | recursively drop weakest features | good |
| Random-forest / Boruta / stability selection | importance with repeated resampling | good, tests stability |
| SHAP-based selection | importance from a fitted model | medium |
| Metaheuristics: genetic algorithm, particle swarm, grey wolf, simulated annealing | search the best subset | good but overfits easily at small n |
| Network-block averaging (domain-specific) | average FC edges into brain-network blocks | good for the 30,135-edge case |

### D. Quantum and quantum-inspired
| Method | Idea | Realistic here? |
|---|---|---|
| **QUBO feature selection (mRMR-style) solved by QAOA** | encode "pick k features, maximise relevance, minimise redundancy" as a quantum optimisation problem | **Yes, on a simulator, with a pre-filtered pool of about 20 to 30 features** |
| QUBO solved by quantum annealing (D-Wave) / hybrid solvers | same problem on an annealer | cloud account needed, availability to check |
| QUBO solved by simulated annealing (quantum-inspired, classical) | same formulation, classical solver | Yes, free, and the fair comparison for QAOA |
| VQE-based selection | variational quantum eigensolver on the same cost function | Yes, similar scale limits |
| Grover adaptive search | quantum search over subsets | theoretical here, not practical |
| Quantum-inspired evolutionary / swarm algorithms | classical algorithms borrowing quantum ideas | Yes, but classical |
| qPCA (quantum PCA) | quantum speed-up for PCA | **No**: needs QRAM and loading classical data removes the speed-up; also still PCA |
| Quantum kernel SVM | quantum feature map as kernel for classification | Possible *after* selection (needs few features) |

The detailed explanation, the choice and the reasons are in
`md files/FEATURE_SELECTION_PROCESS_AND_QUANTUM_METHODS.md`.

---

## 8. Why 96 % should be treated with suspicion

| Leak | How it would happen here |
|---|---|
| PCA / selection fitted on all subjects before CV | Test subjects shape the components: the classic fake 90 %+. |
| Choosing features with labels on the whole dataset | Same effect, stronger. |
| Repeat scans of one person across train and test | Cohort is one scan per subject; do not go back to 165 scans. |
| Using CDR-SB or MMSE as inputs | CDR-SB correlates 0.85 with stage: it is the label in disguise. |
| Plain accuracy on an easy subset (CN vs AD only) | Legitimate, but a different task and must be labelled. |
| Tuning on the 35 test subjects | Test set is for one use only (already used once for the MLP). |

Rule of thumb: **a leak-free 6-class result of 35 to 45 % would already be a clear improvement.**

---

## 9. Questions to ask mam

1. **"96 % of what?"** Accuracy, balanced accuracy, or AUC? On which classes?
2. **All 6 stages, or a simpler task** (CN vs AD, or CN / MCI / AD)? Binary tasks can score much higher.
3. Is a **quantum simulator** acceptable for the quantum method, or must it run on real hardware?
4. Is 96 % a hard requirement, or a hope? Is an honest result with a clear method comparison acceptable?

Suggested wording: *"Ma'am, with 116 subjects and 6 stages our best leak-free accuracy is about 31 %. I will list the
advanced and quantum feature-selection methods, choose a suitable one, and compare it with PCA, LDA and the others
inside cross-validation, then report the best honestly. Could you confirm which metric and which classification task
the 96 % refers to, and whether a quantum simulator is acceptable?"*

---

## 10. The plan (in order)

1. Read the companion file, decide the chosen method with mam (questions in section 9).
2. Install and verify tools (`qiskit` or `pennylane`, and `dimod`/`neal` for the classical annealer); record versions.
3. Write one leak-free script: `scale -> reduce/select -> classifier` inside each CV fold, same 5 seeds x 5 folds
   as `scripts/model/repeated_cv.py`, so results compare with 0.307.
4. Run the normal methods (PCA, LDA, LRA), then the advanced classical ones, then the quantum one.
5. Leaderboard table, then **shuffled-label control** on the winner (must fall to about 0.17).
6. **Test the winner once** on the 35 held-out subjects and report the number, whatever it is.
7. Update this file, `README.md`; scripts to `scripts/model/`, outputs to
   `derivatives/model_training/feature_selection/`.

---

## 11. Git state (as of 2026-10-17)

* Last pull: **2026-10-16 15:09 (+0530)**, fast-forward, bringing Harishkanna's 5 commits (up to `f364c580`
  "Fix tools ignore rule"). Earlier pull: 2026-10-12 10:17.
* `git pull origin main` on 2026-10-17: **Already up to date**; GitHub `main` equals local `HEAD`.
* The 16 Oct pull changed 6,432 files: 5,583 added, 20 modified, **829 deleted** (per-scan `.nii.gz` maps and masks
  under `derivatives/fsfast/`; `.npy` and `.csv` features remain, so training is unaffected).

---

## 12. Open items in the project

| Priority | Item | Owner on slides |
|---|---|---|
| 1 | **Feature-selection comparison incl. quantum (this task)**: done 2026-10-18, see `FEATURE_SELECTION_RESULTS_SUMMARY.md` | me |
| 2 | Confirm with mam what "96 %" means and which task | me |
| 3 | Finish new LMCI clinical tables (zip dated 12 Oct) | Nagindhar |
| 4 | Expand dataset by subject, not by scan | Nagindhar |
| 5 | Fix or drop the ordinal head (collapse diagnosed) | Harishkanna |
| 6 | Meta-learning, comorbidity fusion, Bayesian uncertainty (design only) | me / Harishkanna |
| 7 | Streamlit dashboard (Module 6) | me |
| 8 | Update `README.md` (says no model trained) | anyone |
| 9 | Review 3 story: honest results plus what feature selection changed | all |

---

## 13. Live progress log (update as work proceeds)

| Date | Step | Outcome |
|---|---|---|
| 2026-10-17 | Read INSTRUCTIONS.md, README, docs, logs, both review PDFs; checked git; wrote this file and the companion process file | No experiment code written yet. |
| 2026-10-17 | Feature-selection steps 1 to 4 (tools, harness, baselines) | Honest baseline 0.17 to 0.18; earlier 0.307 likely optimistic. |
| 2026-10-18 | Feature-selection study complete: 26 methods, 285 configurations, incl. quantum QUBO + QAOA | Best CV: PCA 738 -> 10, 0.253. Held-out test 0.172 (L1 0.228, QAOA 0.233), none significantly above chance. Summary: `md files/FEATURE_SELECTION_RESULTS_SUMMARY.md`. |
