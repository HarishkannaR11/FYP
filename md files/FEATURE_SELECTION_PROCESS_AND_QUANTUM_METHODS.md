# Feature Selection: Full Process, Feature Counts, and Quantum Methods

Written 2026-10-17. Companion to `md files/PROJECT_STATUS_PLUS_PLAN.md`.
All "after" numbers below are **planned targets**, not results. No experiment has been run yet.

---

## 1. The goal in one sentence

Take the large set of brain features per subject, reduce it to a **small, informative set** without leaking test
information, and show which reduction method gives the best accuracy on the 6 stages (CN, SMC, EMCI, MCI, LMCI, AD).

Fixed facts: 116 subjects (CN 20, SMC 20, EMCI 20, MCI 16, LMCI 20, AD 20); training pool 81, locked test 35;
5-fold CV on the pool, so each fold trains on about **64** subjects and validates on about **17**.

---

## 2. Features BEFORE selection

| Feature set | Built from | Count per subject | Count vs ~64 training subjects |
|---|---|---:|---|
| **DC_z** | degree centrality, z-scored, 246 regions | **246** | about 4 features per subject |
| **mALFF** | amplitude of low-frequency fluctuation / region mean | 246 | same |
| **mReHo** | regional homogeneity / region mean | 246 | same |
| **All regional** | mALFF + mReHo + DC_z | **738** | about 11 per subject |
| **All four (older run)** | the 3 above + FC unsigned mean strength per region | **984** | about 15 per subject |
| **FC edges** | upper triangle of the 246x246 Fisher-z matrix | **30,135** | about 470 per subject |

Source arrays: `X_regional.npy` (116 x 246 x 3), `X_fc.npy` (116 x 246 x 246), per-region FC summaries
`X_fc_roi_unsigned.npy`, `X_fc_roi_dispersion.npy`.
Clinical covariates are separate: **APOE4 and age only** are safe to add; **CDR-SB must not** be used
(rho = 0.85 with the label); MMSE is risky; LMCI has no age/MMSE/CDR.

**Why this is a problem:** a model with more inputs than subjects can memorise noise. Earlier tests confirm it:
about 31 % balanced accuracy in CV and 20 % on the held-out test, barely above the 16.7 % chance level.

---

## 3. Features AFTER selection (planned targets)

| Method family | Output features per subject (planned) | Reduction |
|---|---|---|
| PCA / Sparse PCA / PLS / Kernel PCA / ICA | k = 5, 10, 20, 30, 40 components (k must stay below about 64) | 738 to about 10-40; 30,135 to about 10-40 |
| LDA | at most 5 (classes minus 1) | 738 to 5 |
| ANOVA / MI / ReliefF / L1 / SVM-RFE / RF | top 10, 20, 50, 100 original features | 738 to about 10-100 |
| **QUBO (quantum / annealing)** | a pool of 20 pre-filtered features, then the best **k = 6, 8, 10** of them | 738 to 20 to about 6-10 |
| Network-block averaging (FC only) | 246 regions grouped into brain networks (for example 7 to 24 blocks), edges averaged within and between blocks | 30,135 to a few dozen |

The exact counts are tuned **inside the training folds**, never on the test set.
For selection methods the **selected ROI names** are reported (for example "A9/46v_L, DC_z"), which is the
readable result mam can discuss; for PCA-type methods only component loadings are available.

---

## 4. The process, step by step (every step happens inside each CV fold)

```
raw features (116 x F)
  -> split: 81 training pool | 35 locked test (not touched until the very end)
  -> for each seed (5) and each fold (5) of the training pool:
        1. fit StandardScaler on the fold's training subjects only
        2. fit the reducer / selector on those same subjects only
              (for supervised ones it uses their labels, never the validation labels)
        3. transform training AND validation subjects with the fitted reducer
        4. fit classifier on reduced training features
        5. predict validation subjects, record the metrics
  -> average over 25 folds (mean +/- sd), per-class recall, chance level
  -> leaderboard of all methods
  -> shuffled-label control on the winner
  -> freeze winner, run ONCE on the 35 test subjects
```

### Stage details

| Stage | What happens | Output |
|---|---|---|
| 0. Load | `X_regional.npy`, `X_fc.npy`, `y.npy`, `splits/split_assignment.csv`; verify the split fingerprint `6f478fbcf56364e2` | arrays and folds |
| 1. Build input views | (a) DC_z 246, (b) all regional 738, (c) FC edges 30,135, (d) FC network blocks | 4 input versions |
| 2. Scale | z-score per feature, fitted on fold-train only | scaled X |
| 3a. Normal baselines | PCA, LDA, LRA / truncated SVD | reduced X |
| 3b. Advanced classical | Sparse PCA, Kernel PCA, ICA, PLS; mRMR, ReliefF, L1, SVM-RFE, RF/Boruta/stability selection | reduced X |
| 3c. Quantum | QUBO feature selection (section 6) | selected feature subset |
| 4. Classify | logistic regression, linear SVM, RBF-SVM, random forest, gradient boosting, small MLP | predictions |
| 5. Evaluate | balanced accuracy, accuracy, macro-F1, MAE in stages, QWK, per-class recall | one row per method |
| 6. Controls | shuffled labels (expect about 0.17); compare with the current DC-only MLP 0.307 on identical folds; check no class collapse | leak check |
| 7. Final test | winner frozen from CV only, evaluated once on 35 held-out subjects | final number |
| 8. Report | before/after feature counts, selected ROIs, stability across folds, leaderboard | results section here |

Rules from `md files/INSTRUCTIONS.md` that apply: record tool versions first, keep a structured log
(JSON lines) per run, never overwrite existing outputs, keep the progress table in the `.md` live.

---

## 5. Quantum feature selection: what exists

### 5.1 The core idea (why feature selection fits quantum optimisation)

Choosing the best subset of features is a **combinatorial** problem: with n features there are 2^n subsets.
It can be written as a **QUBO** (Quadratic Unconstrained Binary Optimisation): one binary variable x_i per
feature (1 = keep, 0 = drop), and the cost

```
cost(x) =  - alpha * sum_i  relevance_i * x_i                 (reward features that relate to the stage label)
           + (1 - alpha) * sum_{i<j} redundancy_ij * x_i * x_j (punish pairs that carry the same information)
           + lambda * ( sum_i x_i - k )^2                      (keep exactly k features)
```

* `relevance_i` = mutual information or ANOVA F between feature i and the label (from training subjects only).
* `redundancy_ij` = absolute correlation (or mutual information) between features i and j.
* The result is the same idea as the classical **mRMR** method, written in a form quantum optimisers accept.

This QUBO maps to an **Ising Hamiltonian**, and quantum algorithms look for its lowest-energy state.

### 5.2 Methods, from most to least practical for this project

| Method | How it works | Hardware / tools | Size limit | Verdict here |
|---|---|---|---|---|
| **QAOA** (Quantum Approximate Optimisation Algorithm) | alternates a cost layer and a mixer layer, a classical optimiser tunes the angles | gate-model simulator (Qiskit, PennyLane); real IBM devices possible | about 20 to 25 qubits on a laptop simulator (n features = n qubits) | **Best fit** |
| **VQE** on the same Hamiltonian | variational circuit minimises energy | same | same | good alternative, similar cost |
| **Quantum annealing** (D-Wave) | hardware or hybrid solver finds low-energy states of the QUBO | cloud account, usage limits (to be checked) | thousands of variables on hybrid solvers | possible, needs account |
| **Simulated annealing on the QUBO** | classical "quantum-inspired" solver, same formulation | `dimod` / `neal`, free, local | hundreds to thousands of variables | **needed as the fair comparator, and as the solver for bigger pools** |
| **Exact brute force** on the QUBO | tries all 2^n subsets | numpy | n up to about 24 | gives ground truth to check QAOA |
| Grover adaptive search | quantum search over subsets with a quadratic speed-up | theory / small simulation | tiny | not practical here |
| Quantum-inspired evolutionary / swarm | classical algorithms borrowing quantum ideas | any | any | works, but is just a metaheuristic |
| **qPCA** | quantum principal-component analysis | needs QRAM | n/a | **Not suitable**: loading classical data cancels the speed-up, and it is still unsupervised PCA |
| Quantum kernel SVM | quantum feature map as kernel | simulator | few features (about 8 to 20) | optional **after** selection |
| Quantum autoencoder / quantum Boltzmann machine | quantum circuits learn compressed features | simulator | tiny | too data-hungry for 81 subjects |

### 5.3 Honest limits (tell mam)

* On a simulator, **there is no quantum speed-up and no accuracy advantage** at this size. A classical annealer
  or brute force finds the same optimum. The contribution is a **principled quantum-ready formulation** and a
  tested comparison, not "quantum beats classical".
* A statevector simulator needs one qubit per feature, so **246 or 738 features cannot go straight into QAOA**.
  The pool must be pre-reduced to about 20 to 24 features (section 6, stage A).
* QAOA is an approximate method; its answer can be below the optimum, which is why brute force is run as a check.
* Real quantum hardware is noisy and queue-limited; a simulator is the realistic route for 25 folds x 5 seeds.
* Packages are not installed yet (qiskit or pennylane, dimod, neal); install and record versions first.

---

## 6. The choice: QUBO (mRMR-style) feature selection with QAOA, compared with simulated annealing

### 6.1 What exactly we will run

**Stage A: classical pre-filter (inside each fold).** Rank all features with ANOVA F (or mutual information) on the
fold-train subjects and keep the top **20** as the candidate pool (changed from 24 after the timing test in 6.5).
This is what keeps the qubit count simulable.

**Stage B: quantum selection (inside each fold).**
1. Build the QUBO for the 20 candidates (relevance, redundancy, penalty for exactly k picks), k in {6, 8, 10}.
2. Solve it with **QAOA** (p = 1 to 3 layers) on a simulator, 20 qubits.
3. In parallel solve the same QUBO by **exact brute force** (2^20 is about 1 million subsets, under 1 second) to get
   the true optimum, and by **simulated annealing** as the classical quantum-inspired comparator.
4. Take the chosen subset into the classifier (logistic regression / SVM / small MLP).

**Stage C (optional, if time permits): quantum kernel SVM** on the 8 to 16 selected features, as a
quantum-classification extension.

### 6.2 Exactly why QUBO

**The problem we have.** About 64 training subjects, 246 to 738 candidate features, and weak stage signal
(honest baseline about 0.17). Two things go wrong at once: (1) most features are noise, and (2) the features that
carry signal overlap with each other (DC_z, mReHo and mALFF are partly correlated). A good selector must pick
**few features that are individually relevant AND not redundant with each other**, and it must pick a **fixed small
number k** so the model cannot overfit.

**What QUBO does about it.** Feature selection is a subset-choice problem: each feature is in or out, so there are
2^n possible subsets. QUBO writes exactly that trade-off as one equation (relevance reward, redundancy penalty,
"exactly k" penalty) over in/out variables. Solving it means finding the **best subset as a whole**, not feature by
feature. That is the same objective as the classical method mRMR, but written in the form quantum optimisers
(QAOA, VQE, annealers) accept. This is why QUBO is **the** standard way to do "quantum feature selection": it is
the only natural bridge between the feature-selection problem and quantum hardware/algorithms.

| Reason | Detail |
|---|---|
| **The problem is a subset search, which quantum optimisers target** | in/out choices map one-to-one onto qubits (1 feature = 1 qubit) |
| **It models relevance and redundancy together** | PCA, ANOVA and L1 each handle only part of this |
| **It keeps original, named features** | output is "these 8 regions/biomarkers", readable for the clinic and the dashboard |
| **It is what mam's phrase "quantum feature selection" means in practice** | QUBO + QAOA/annealing is the established formulation |
| **It is leak-free by construction** | relevance and redundancy are computed from fold-train subjects only |
| **It is checkable** | at 20 features brute force gives the exact optimum, so we can state how close QAOA gets and whether quantum adds anything over simulated annealing |

### 6.2b Why QUBO over each alternative

| Alternative | Why not chosen as the quantum method |
|---|---|
| **qPCA** | needs QRAM; loading classical data removes the speed-up; output is still unsupervised PCA, which ignores the stage labels |
| **Quantum kernel SVM** | it is a classifier, not a selector; it can only take a few features (about 8 to 20 qubits), so it needs selection first. Kept as optional Stage C **after** QUBO |
| **Quantum autoencoder / quantum Boltzmann machine** | need far more subjects than 81; compress rather than select |
| **Grover adaptive search** | gives only a quadratic speed-up and needs far more qubits and circuit depth than a simulator can run here |
| **VQE** | solves the same QUBO with a different circuit; no extra benefit for us, so QAOA stays primary and VQE is a spare |
| **D-Wave quantum annealing** | needs a cloud account, usage limits and cost; the QUBO built here can be sent to it later unchanged |
| **Quantum-inspired evolutionary / swarm** | these are classical metaheuristics with no clear quantum link; they overfit easily with 64 subjects |
| **Classical-only selectors (mRMR, L1, SVM-RFE, ANOVA)** | they are compared against, not replaced; they are the benchmark that tells us whether quantum helps |

### 6.2c Suitability check (what I tested and what I found)

| Check | Result | Verdict |
|---|---|---|
| Does the QUBO fit the problem? | yes: subset selection with relevance, redundancy and a fixed k is exactly a QUBO | Suitable |
| **Can QAOA run on this laptop?** (timed, dense 20 to 24-qubit QAOA circuits with qiskit-aer) | one circuit evaluation: 12 qubits 0.00 s, 16 qubits 0.01 s, **20 qubits 0.08 s**, 22 qubits 0.32 s, **24 qubits 1.81 s** | 20 qubits feasible; 24 qubits would take about 5 hours per run set (about 150 optimiser steps x 25 folds x 3 values of k), so **pool reduced to 20** |
| Can the exact answer be computed as a check? | brute force over all subsets: 20 features 0.7 s, 24 features 13.5 s | Yes, so QAOA's quality can be measured |
| Are 246 or 738 features directly usable? | no: one qubit per feature, far beyond simulation | A classical pre-filter to 20 is required (part of the design) |
| Does the data give the QUBO good inputs? | **weak**: relevance is estimated from about 64 subjects with weak stage signal (baseline about 0.17), so relevance scores are noisy and the QUBO may optimise noise | **Main risk** |
| Will quantum beat classical? | no: QUBO + QAOA solves the same problem as mRMR; at 20 variables brute force and annealing find the same optimum | No accuracy advantage expected |

**Verdict.** QUBO is the suitable choice **as the quantum feature-selection method**: it is the only candidate that
(a) is genuinely feature selection, (b) is runnable and verifiable here, and (c) matches what mam asked for.
It is **not** expected to beat the classical selectors, and I will not claim it does. Whether it is the best method
overall is decided by the leaderboard, not by this argument. If the noisy-relevance risk shows up (QUBO picks
different features in every fold, or does no better than random 20-feature subsets), I will report that and
recommend the best classical method for the final model, keeping QUBO as the quantum study.

Tests I will add to the experiment to settle the risk: selected-feature **stability across folds**; comparison with
**random subsets of the same size**; comparison with **greedy mRMR**; and the shuffled-label control.

### 6.3 What we compare it against (so "we selected it" is justified by results, not claims)

PCA, LDA, LRA; Sparse PCA, PLS, Kernel PCA, ICA; ANOVA, mutual information, mRMR, L1, SVM-RFE, RF/Boruta;
simulated annealing on the same QUBO; greedy mRMR; random subsets of equal size; no-reduction baselines (about
0.17 to 0.18 from step 4; the earlier 0.307 is optimistic). The winner by cross-validated balanced
accuracy, confirmed by the shuffled-label control, is reported, even if it is not the quantum one.

### 6.4 Expected outcome, stated plainly

* The quantum method is expected to perform **about the same as mRMR and simulated annealing**, because they
  solve the same problem.
* All methods are expected to land only modestly above the clean baseline (0.17 to 0.18), perhaps in the
  **0.20 to 0.35 range** for 6 stages. This is a revised, lower expectation than my first estimate, because step 4
  showed the earlier 0.307 was optimistic. It is **not 96 %**.
* If any method reports far higher, treat it as a leak until the shuffled-label control proves otherwise.

---

## 7. Execution plan and status tracker

Status values: **Not started** / **In process** / **Done**. Updated after every step (per INSTRUCTIONS.md).

### Phase 1: set up and prove the pipeline is leak-free
| # | Step | Status | Updated | Outcome |
|---|---|---|---|---|
| 1 | Install and verify tools (`qiskit`, `qiskit-aer`, `dimod`, `neal`); record exact versions | Done | 2026-10-17 | Python 3.13.9; qiskit 2.6.0, qiskit-aer 0.17.2, dimod 0.12.22, dwave-neal 0.6.0, scikit-learn 1.9.1, numpy 2.5.3, scipy 1.16.3. Smoke tests pass: Aer Bell circuit gives 00/11 only; neal and exact solver agree on a toy QUBO. |
| 2 | Load data and check it (split fingerprint `6f478fbcf56364e2`, 81 train / 35 test, no overlap, test locked) | Done | 2026-10-17 | All checks PASS: pool 81, test 35, fingerprint matches, class counts match (CN14 SMC14 EMCI14 MCI11 LMCI14 AD14). Test-subject feature rows are never kept in memory. |
| 3 | Write the shared CV harness `scripts/model/feature_selection_cv.py` (scale, reduce/select, classify, score, all inside each fold; 5 seeds x 5 folds) | Done | 2026-10-17 | `scripts/model/feature_selection_cv.py` (torch-free, scikit-learn Pipelines). All 5 seeds' fold fingerprints MATCH the earlier experiments, so folds are identical. |
| 4 | Reproduce old baseline (DC_z, no reduction, expect about 0.307) and shuffled-label control (expect about 0.17) | Done | 2026-10-17 | Shuffled-label controls all near chance (0.160 to 0.162): no leak. Cross-check: logistic regression on 738 regional features gives 0.176 +/- 0.008, identical to the earlier `logreg_roi3` result. Honest no-reduction baselines: logreg DC_z 0.168, sklearn MLP DC_z 0.181, majority 0.167. The earlier torch MLP 0.307 was not reproduced (torch not installed) and is probably optimistic, see note below. Records: `derivatives/model_training/feature_selection/step4_baseline_*`. |

**Important note on the 0.307 reference.** In `scripts/model/repeated_cv.py`, `run_mlp` picks the best training epoch by balanced accuracy on the *validation fold* itself. That is selection on the data it is then scored on, so 0.307 is likely optimistic. Consistent with this, the same model scored only 0.20 on the untouched test set, and a clean sklearn MLP gets 0.181 here. **The realistic starting point for this study is about 0.17 to 0.18, not 0.31.** All new methods will be judged on a clean footing (no epoch picking on validation data), against these baselines.

### Phase 2: run the methods, easiest first
| # | Step | Status | Updated | Outcome |
|---|---|---|---|---|
| 5 | Normal methods: PCA, LDA, LRA at several component counts | Done | 2026-10-18 | 29 configurations. Best **PCA k=10 on regional (738 -> 10): 0.253 +/- 0.037**. LDA = chance (0.167). Details in section 10.1. |
| 5b-7 all4 | All methods of steps 5, 6a, 6b, 6c, 7 re-run on all four biomarkers combined (984 features) | Done | 2026-10-18 | Best on all4: LRA k=20 0.248, Boruta 0.236, NMF k=10 0.226, Sparse PCA k=10 0.222, MI k=20 0.220; QUBO 0.15 to 0.20. Stability selection not run on all4: about 1 h per configuration, and it ranks low on every other view. |
| 8 | Leaderboard frozen; winner chosen from CV only | Done | 2026-10-18 | 276 configurations. Winner **PCA k=10 on regional (738 -> 10), CV 0.253**. Pre-declared secondary reports: L1 k=10 regional, QUBO-QAOA k=8 regional. |
| 6a | Advanced classical extraction: Sparse PCA, PLS, Kernel PCA, ICA | Done | 2026-10-18 | 39 configurations. Best ICA k=10 regional 0.259 (a rotation of PCA); Sparse PCA k=20 regional 0.239. Section 10.2. |
| 6b | Advanced classical selection: ANOVA, MI, mRMR, ReliefF, L1, SVM-RFE, RF, stability selection | Done except stability re-run | 2026-10-18 | 66 configurations. Best L1 k=10 regional 0.239; mRMR k=50 edges 0.234. Section 10.3. |
| 6c | Remaining methods from the list: Isomap, NMF, Boruta, genetic algorithm, particle swarm | Done | 2026-10-18 | 27 configurations. Best NMF 0.217; GA/PSO/Boruta 0.20 to 0.21. Section 10.4. |
| 7a | Quantum: build QUBO (relevance, redundancy, exactly-k penalty) on 20-feature pool | Done | 2026-10-18 | Built. Penalty weight now calibrated (see 10.4): the first version trapped both solvers. |
| 7b | Quantum: QAOA on simulator, with brute force and simulated annealing comparators | Done for dc, regional, fc_blocks; all4 in progress | 2026-10-18 | 45 configurations. QUBO 0.14 to 0.20 whichever solver; no better than random subsets of the same pool. Section 10.5. |
| 7c | (Optional) quantum kernel SVM on the selected 8 to 16 features | Not started | 2026-10-17 | |

### Phase 3: pick the winner
| # | Step | Status | Updated | Outcome |
|---|---|---|---|---|
| 8 | Leaderboard of all methods; shuffled-label check on the winner | Not started | 2026-10-17 | |
| 9 | Freeze winner from CV only; evaluate once on the 35 held-out subjects | Done | 2026-10-18 | Shuffled controls 0.161 to 0.170 (pass). Test: PCA k=10 **0.172** (6/35), L1 k=10 0.228, QUBO-QAOA k=8 0.233; none significantly above chance (p > 0.2). `final_test.log`. |
| 10 | Write results (before/after feature counts, selected ROIs, stability) here and in `PROJECT_STATUS_PLUS_PLAN.md`; update `README.md` | Done except README | 2026-10-18 | Separate summary: `md files/FEATURE_SELECTION_RESULTS_SUMMARY.md`. README not yet updated. |

### Waiting on mam (does not block steps 1 to 4)
| Question | Status |
|---|---|
| What does 96 % mean (metric), and for which task (6 stages or simpler)? | Not asked yet |
| Is a quantum simulator acceptable? | Not asked yet |

---

## 8. Deliverables

| Item | Location |
|---|---|
| Script | `scripts/model/feature_selection_cv.py` (and `qubo_feature_selection.py`) |
| Outputs | `derivatives/model_training/feature_selection/` (results JSON, logs, leaderboard CSV, selected-ROI tables) |
| Report | results section appended to `md files/PROJECT_STATUS_PLUS_PLAN.md` |
| Per-run log | JSON lines with versions, seeds, fold fingerprints, metrics, selected features |

---

## 9. Progress log (update as work proceeds)

| Date | Step | Outcome |
|---|---|---|
| 2026-10-17 | Methods listed, feature counts before/after defined, quantum method chosen on paper | Awaiting mam's answers (metric, task, simulator OK). |
| 2026-10-17 | Step 1: installed qiskit 2.6.0, qiskit-aer 0.17.2, dimod 0.12.22, dwave-neal 0.6.0; smoke tests pass | Done |
| 2026-10-17 | Steps 2 to 4: harness written and run; data/split checks pass; folds match earlier fingerprints; shuffled controls at chance | Done. Honest baseline about 0.17 to 0.18; earlier 0.307 likely optimistic (epoch picked on validation fold). |
| 2026-10-18 | Script `scripts/model/feature_selection_methods.py` written (steps 5, 6a, 6b, 7) | Same folds (fingerprints re-verified), same classifier for every method, test set never loaded. |
| 2026-10-18 | Step 5 run | Done. Best PCA k=10 regional 0.253. Output `derivatives/model_training/feature_selection/step5*`. |

---

## 9b. Method-by-method status (every method from the status-file list, section 7)

Status: **Done** / **In progress** / **Not started** / **Excluded** (with reason). Best = highest CV balanced
accuracy over the k values and input views tried for that method (chance 0.167). Updated 2026-10-18.

### A. Normal methods
| Method | Status | Best result | Notes |
|---|---|---|---|
| PCA | Done | **0.253 +/- 0.037** (k=10, regional) | |
| LDA (shrinkage) | Done | 0.194 (PCA-40 + LDA, dc); plain LDA 0.167 | too few subjects to estimate class covariance |
| LRA / truncated SVD | Done | 0.229 (k=20 regional; k=40 fc_edges) | |

### B. Advanced classical: feature extraction
| Method | Status | Best result | Notes |
|---|---|---|---|
| Kernel PCA (RBF) | Done | 0.213 (k=10, regional) | |
| Sparse PCA | Done | 0.239 +/- 0.006 (k=20, regional) | most consistent across seeds |
| ICA | Done | **0.259 +/- 0.040** (k=10, regional) | ICA is a rotation of whitened PCA, so with this classifier it nearly copies PCA (identical on dc) |
| PLS / PLS-DA | Done | 0.222 (k=5 or 10, fc_edges); 0.217 (k=5, regional) | the supervised one, yet not better |
| NMF | Done | 0.217 +/- 0.009 (k=10, regional) | step 6c; features shifted by the fold-train minimum because NMF needs non-negative input |
| Isomap | Done | 0.204 (k=10, fc_blocks) | step 6c; mostly near chance |
| UMAP | Excluded | | `umap-learn` not installed; Isomap run as the manifold method instead. Unstable at n = 64 |
| Autoencoder / concrete autoencoder | Excluded | | torch not installed; about 64 training subjects is far too few to train one |

### C. Advanced classical: feature selection
| Method | Status | Best result | Notes |
|---|---|---|---|
| ANOVA F-test | Done | 0.231 (k=50, fc_edges); 0.226 (k=20, regional) | step 6b |
| Mutual information | Done | 0.220 (k=20 or 50, all4); 0.201 (k=10, regional) | step 6b |
| mRMR | Done | 0.234 +/- 0.044 (k=50, fc_edges); 0.222 (k=50, regional) | step 6b, edges pre-filtered to 1,000 by ANOVA |
| ReliefF | Done | 0.224 (k=50, regional) | step 6b |
| L1 logistic regression | Done | **0.239 +/- 0.037** (k=10, regional) | best selector in 6b |
| SVM-RFE | Done | 0.212 (k=20, regional) | step 6b |
| Random-forest importance | Done | 0.213 (k=20, regional) | step 6b |
| Stability selection | Done | 0.221 (k=10, regional) | first run had a bug (L1 too strong, kept no feature, ties fell to column order); those numbers are discarded, these are from the fixed re-run. Not run on all4 (about 1 h per configuration) |
| Boruta | Done | 0.236 +/- 0.013 (all4); 0.212 (regional) | step 6c; own implementation. Few or no features beat the shadows, so it mostly fell back to the 10 highest-hit features |
| Genetic algorithm | Done | 0.204 (regional, about 29 features) | step 6c; inner-CV fitness overfits: lowest stability (0.09) |
| Particle swarm (binary PSO) | Done | 0.199 (regional); 0.190 (dc) | step 6c; same behaviour as GA |
| Grey wolf optimiser | Excluded | | same family as GA/PSO (wrapper metaheuristic); two representatives are enough, and all overfit easily at n = 64 |
| Simulated annealing (as wrapper) | Covered in D | | run on the QUBO instead |
| SHAP-based selection | Excluded | | `shap` not installed; it ranks by a fitted model's importance, which RF importance and Boruta already cover |
| Network-block averaging | Done (QUBO part in progress) | 0.216 (NMF k=5 on fc_blocks); most methods 0.13 to 0.20 | the `fc_blocks` view (24 gyri, 300 features); weaker than the regional view |

### D. Quantum and quantum-inspired
| Method | Status | Best result | Notes |
|---|---|---|---|
| QUBO solved by QAOA (qiskit, Aer simulator, 20 qubits, p = 2) | Done (all4 in progress) | 0.200 (k=8, regional); 0.192 (k=8, dc) | reaches the exact optimum in only 0 to 16 % of folds; about 140 s per solve |
| QUBO solved exactly (brute force) | Done (all4 in progress) | 0.196 (k=8, dc) | ground truth; 0.01 s per solve |
| QUBO solved by simulated annealing (dimod/neal) | Done (all4 in progress) | 0.194 (k=8, dc); 0.191 (k=10, regional) | reaches the optimum in 20 to 96 % of folds; 0.4 s per solve |
| Random subsets / greedy mRMR on the same 20-feature pool | Done (all4 in progress) | random 0.210 (k=8, regional); greedy 0.189 | **random subsets score as well as the QUBO optimum** |
| VQE on the same QUBO | Excluded | | solves the same cost function as QAOA with a different circuit; kept as a spare (section 6.2b) |
| D-Wave quantum annealing | Excluded | | needs a cloud account; the QUBO built here can be sent unchanged later |
| Grover adaptive search | Excluded | | far beyond simulable depth; theoretical here |
| Quantum-inspired evolutionary | Excluded | | classical metaheuristic; GA and PSO in C cover this family |
| qPCA | Excluded | | needs QRAM; still unsupervised PCA |
| Quantum kernel SVM | Not started | | optional step 7c, after the leaderboard |

---

## 10. Results (live, filled in as each step finishes)

Common set-up for every row: 81 training-pool subjects, 5 seeds x 5 folds (fold fingerprints verified),
scaling + reduction/selection fitted on fold-train only, classifier = logistic regression (C = 0.05,
class-balanced). bal.acc = mean +/- sd over the 5 seed means. Chance = 0.167. Clean no-reduction baselines:
logreg DC_z 0.168, logreg regional 0.176, MLP DC_z 0.181.

Input views: **dc** = DC_z 246; **regional** = mALFF + mReHo + DC_z 738; **fc_edges** = 30,135 FC edges;
**fc_blocks** = FC edges averaged within/between the 24 Brainnetome gyri = 300 features (new, step 6);
**all4** = all four biomarkers together: mALFF + mReHo + DC_z (738) + FC per region (mean |FC| of each region,
`X_fc_roi_unsigned.npy`, 246) = **984** features.

Which biomarker is in which view:

| View | mALFF | mReHo | DC_z | FC |
|---|:-:|:-:|:-:|:-:|
| dc (246) | | | yes | |
| regional (738) | yes | yes | yes | |
| fc_blocks (300) / fc_edges (30,135) | | | | yes |
| **all4 (984)** | yes | yes | yes | yes |

Note (2026-10-18): the first runs of steps 5 to 6c tested FC only on its own, never combined with the other
three. The all4 view was added afterwards and every method is re-run on it (results marked "all4" below).

### 10.1 Step 5: normal methods (PCA, LDA, LRA)

LRA = truncated SVD of the raw (uncentred, unscaled) matrix; PCA = SVD after z-scoring.
LDA = shrinkage LDA projected to 5 discriminant axes (on 30,135 edges only after PCA-40, memory).

| Method | dc 246 | regional 738 | fc_edges 30,135 |
|---|---:|---:|---:|
| PCA k=5 | 0.131 | 0.204 | 0.188 |
| PCA k=10 | 0.177 | **0.253** | 0.192 |
| PCA k=20 | 0.193 | 0.229 | 0.160 |
| PCA k=40 | 0.191 | 0.199 | 0.206 |
| LRA k=5 | 0.156 | 0.160 | 0.216 |
| LRA k=10 | 0.186 | 0.210 | 0.212 |
| LRA k=20 | 0.206 | 0.229 | 0.189 |
| LRA k=40 | 0.212 | 0.186 | 0.229 |
| LDA (5 axes) | 0.167 | 0.167 | n/a |
| PCA-40 + LDA | 0.194 | 0.163 | 0.181 |

Fold-to-fold sd is about 0.02 to 0.05 for every row.

**All four biomarkers (all4, 984 features), step 5:**

| Method | k=5 | k=10 | k=20 | k=40 |
|---|---:|---:|---:|---:|
| PCA | 0.191 | 0.204 | 0.190 | 0.181 |
| LRA | 0.157 | 0.214 | **0.248 +/- 0.015** | 0.200 |
| LDA (5 axes) | 0.162 | | | |
| PCA-40 + LDA | 0.193 | | | |

Adding FC to the three regional biomarkers does **not** raise the best score: LRA k=20 on all4 gives 0.248,
the same as PCA k=10 on regional (0.253). PCA itself gets worse on all4 (0.204 vs 0.253), so the extra 246
FC features add more variance than stage signal. This matches the older project result (MLP on all four 0.278
was worse than DC only 0.307).

Reading:
* **Best: PCA 10 components on the 738 regional features, 0.253 +/- 0.037** (macro-F1 0.224, MAE 1.70 stages).
  That is about 8 points above chance and 8 above the no-reduction logreg on the same features (0.176).
* **LDA does nothing (0.167 = chance).** With about 64 subjects it cannot estimate the class covariance, as
  predicted in section 6 of the status file.
* The ranking moves with k by several points, and neighbouring k values differ by more than the sd. So any one
  "best k" is partly luck; the honest summary is "PCA/LRA give about 0.20 to 0.25".

### 10.2 Step 6a: advanced extraction (Sparse PCA, PLS, Kernel PCA, ICA)

| Method | dc 246 | regional 738 | fc_blocks 300 | fc_edges 30,135 |
|---|---:|---:|---:|---:|
| Sparse PCA k=5 | 0.157 | 0.200 | 0.186 | |
| Sparse PCA k=10 | 0.176 | 0.218 | 0.170 | |
| Sparse PCA k=20 | 0.164 | 0.239 | 0.177 | |
| PLS-DA k=5 | 0.184 | 0.217 | 0.204 | 0.222 |
| PLS-DA k=10 | 0.186 | 0.179 | 0.182 | 0.222 |
| PLS-DA k=20 | 0.170 | 0.152 | 0.183 | 0.180 |
| Kernel PCA (RBF) k=5 | 0.168 | 0.191 | 0.130 | |
| Kernel PCA (RBF) k=10 | 0.178 | 0.213 | 0.170 | |
| Kernel PCA (RBF) k=20 | 0.194 | 0.192 | 0.171 | |
| ICA k=5 | 0.131 | 0.204 | 0.188 | |
| ICA k=10 | 0.177 | **0.259** | 0.178 | |
| ICA k=20 | 0.193 | 0.218 | 0.177 | |

Reading:
* **Best ICA k=10 on regional, 0.259 +/- 0.040**, but ICA only rotates the PCA components, so this is
  essentially the PCA k=10 result (0.253) again, not a separate finding. On dc the two give identical numbers.
* **Sparse PCA k=20 on regional, 0.239 +/- 0.006**, is a little lower but by far the most stable across seeds,
  and its components use few named regions.
* **PLS-DA, the only supervised extractor, does not beat PCA.** Its components chase the training labels and
  overfit at 64 subjects (it gets worse as k grows: 0.217 -> 0.179 -> 0.152 on regional).
* The new **fc_blocks** view (24-gyrus FC averages) stays at 0.13 to 0.20, close to chance.

### 10.3 Step 6b: advanced selection (keeps original, named features)

bal.acc for k = 10 / 20 / 50 selected features. stab = mean Jaccard overlap of the selected sets across the 25
folds (1 = same features every fold).

| Method | dc 246 | regional 738 | fc_blocks 300 | fc_edges 30,135 |
|---|---|---|---|---|
| ANOVA F | 0.202 / 0.194 / 0.181 | 0.177 / 0.226 / 0.219 | 0.153 / 0.174 / 0.166 | 0.160 / 0.176 / 0.231 |
| Mutual information | 0.174 / 0.178 / 0.182 | 0.201 / 0.171 / 0.191 | 0.151 / 0.146 / 0.131 | |
| mRMR | 0.178 / 0.183 / 0.180 | 0.164 / 0.208 / 0.222 | 0.139 / 0.166 / 0.159 | 0.181 / 0.177 / **0.234** |
| ReliefF | 0.178 / 0.176 / 0.188 | 0.191 / 0.216 / 0.224 | 0.182 / 0.174 / 0.171 | |
| L1 logistic | 0.193 / 0.172 / 0.154 | **0.239** / 0.223 / 0.203 | 0.203 / 0.184 / 0.158 | |
| SVM-RFE | 0.144 / 0.171 / 0.159 | 0.192 / 0.212 / 0.210 | 0.171 / 0.168 / 0.151 | |
| RF importance | 0.169 / 0.170 / 0.177 | 0.207 / 0.213 / 0.206 | 0.156 / 0.146 / 0.170 | |

Reading:
* **Best: L1 logistic, 10 features from regional, 0.239 +/- 0.037.** The features it picks most often across
  25 folds: A4tl_R mALFF (21/25), vmPu_L mReHo (17), A37vl_L mReHo (17), mPFtha_R mReHo (15), A40rd_R DC_z (11).
* **Selection is unstable**: stability 0.05 to 0.40, mostly about 0.2. Different folds pick mostly different
  features, which is what weak signal looks like.
* On edges, mRMR and ANOVA reach 0.23 only at k = 50, and pick different edges in almost every fold (stab 0.1).
* Selectors do **not** beat PCA (0.253) here.

### 10.4 Step 6c: Isomap, NMF, Boruta, genetic algorithm, particle swarm

| Method | dc 246 | regional 738 | fc_blocks 300 |
|---|---:|---:|---:|
| Isomap k=5 / 10 / 20 | 0.137 / 0.123 / 0.188 | 0.174 / 0.173 / 0.191 | 0.153 / 0.204 / 0.198 |
| NMF k=5 / 10 / 20 | 0.123 / 0.182 / 0.193 | 0.206 / **0.217** / 0.217 | 0.216 / 0.150 / 0.161 |
| Boruta (-> about 10) | 0.159 | 0.212 | 0.168 |
| Genetic algorithm (-> 22 to 29) | 0.162 | 0.204 | 0.197 |
| Particle swarm (-> 39 to 45) | 0.190 | 0.199 | 0.162 |

Reading:
* **Wrapper metaheuristics (GA, PSO) do not help.** Their inner-CV fitness on about 64 subjects is noisy, so they
  fit noise. They have the lowest stability of all (0.09 to 0.15) and take about 10 minutes per configuration.
* **Boruta** finds almost no feature that beats its shuffled shadows, an independent sign that single features
  carry little stage signal.
* Isomap is near chance; NMF is similar to the other extractors (about 0.21).
* Features that keep recurring across methods (L1, Boruta, GA, PSO): **A4tl_R mALFF, vmPu_L mReHo,
  A37vl_L mReHo, A12/47o_R DC_z, A40rd_R DC_z**. These are the candidate regions to discuss with mam.

### 10.4a Leaderboard so far (every method, best configuration each)

Updated 2026-10-18 from all result files: **231 configurations, 25 methods**. Stability selection (re-run) and
the all4 runs of steps 6b, 6c and 7 are still running and will be added.

| # | Method | Type | Best bal.acc | View, features kept |
|---:|---|---|---:|---|
| 1 | ICA | extraction | 0.259 +/- 0.040 | regional, 10 |
| 2 | PCA | extraction (normal) | 0.253 +/- 0.037 | regional, 10 |
| 3 | LRA | extraction (normal) | 0.248 +/- 0.015 | all4, 20 |
| 4 | Sparse PCA | extraction | 0.239 +/- 0.006 | regional, 20 |
| 5 | L1 logistic | selection | 0.239 +/- 0.037 | regional, 10 |
| 6 | mRMR | selection | 0.234 +/- 0.044 | fc_edges, 50 |
| 7 | ANOVA | selection | 0.231 +/- 0.040 | fc_edges, 50 |
| 8 | ReliefF | selection | 0.224 +/- 0.031 | regional, 50 |
| 9 | PLS-DA | extraction | 0.222 +/- 0.038 | fc_edges, 5 |
| 10 | NMF | extraction | 0.217 +/- 0.009 | regional, 10 |
| 11 | Kernel PCA | extraction | 0.213 +/- 0.037 | regional, 10 |
| 12 | RF importance | selection | 0.213 +/- 0.027 | regional, 20 |
| 13 | SVM-RFE | selection | 0.212 +/- 0.040 | regional, 20 |
| 14 | Boruta | selection | 0.212 +/- 0.023 | regional, 10 |
| 15 | QUBO random-subset control | control | 0.210 +/- 0.065 | regional, 8 |
| 16 | Isomap | extraction | 0.204 +/- 0.028 | fc_blocks, 10 |
| 17 | Genetic algorithm | selection | 0.204 +/- 0.017 | regional, 29 |
| 18 | Mutual information | selection | 0.201 +/- 0.041 | regional, 10 |
| 19 | **QUBO + QAOA (quantum)** | selection | 0.200 +/- 0.061 | regional, 8 |
| 20 | Particle swarm | selection | 0.199 +/- 0.017 | regional, 45 |
| 21 | QUBO exact | selection | 0.196 +/- 0.024 | dc, 8 |
| 22 | QUBO simulated annealing | selection | 0.194 +/- 0.021 | dc, 8 |
| 23 | PCA-40 + LDA | extraction (normal) | 0.194 +/- 0.025 | dc, 5 |
| 24 | Greedy mRMR (20-pool) | control | 0.189 +/- 0.035 | dc, 8 |
| 25 | LDA | extraction (normal) | 0.167 +/- 0.021 | dc, 5 |

Chance 0.167; no reduction 0.176. The top 5 (0.239 to 0.259) are within one sd of each other, so they are
statistically tied; positions 6 to 14 are only slightly lower.

### 10.4b Are we throwing away useful features? (k sweep, regional 738)

Question raised 2026-10-18: reducing 738/984 features to 10 to 50 may drop useful ones. Tested directly by
keeping from 2 up to all features (`ksweep_regional.log`).

| Features kept | PCA | variance kept by PCA | ANOVA | L1 |
|---:|---:|---:|---:|---:|
| 2 | 0.177 | 15 % | | |
| 5 | 0.204 | 31 % | 0.172 | 0.222 |
| **10** | **0.253** | 47 % | 0.177 | **0.239** |
| 15 | 0.217 | | | |
| 20 | 0.229 | 66 % | 0.226 | 0.223 |
| 30 | 0.183 | | | |
| 40 | 0.199 | 87 % | | |
| 50 | 0.148 | | 0.219 | 0.203 |
| 60 | 0.167 | 99 % | | |
| 100 | (max about 63) | | 0.197 | 0.206 |
| 200 | | | 0.203 | 0.157 |
| 400 | | | 0.186 | 0.142 |
| 700 | | | 0.179 | 0.174 |
| all 738 (no reduction) | 0.176 | 100 % | 0.176 | 0.176 |

Reading:
* **Keeping more features makes the score worse, not better.** All three methods peak at about 10 to 20
  features and fall back to about 0.17 to 0.18 (chance level) as more are kept.
* PCA with 10 components keeps only 47 % of the total variance, yet beats keeping 99 % (0.167). The variance it
  drops is mostly noise (person, scanner, motion), not stage signal.
* Reason: each fold trains on about 64 subjects. With hundreds of features the classifier fits noise in those 64
  and fails on new subjects (overfitting). Reduction is what makes any learning possible here.
* So reduction is **not** removing useful information that the classifier could otherwise use; at this sample
  size, the extra features hurt.

### 10.5 Step 7: quantum QUBO feature selection

Per fold: ANOVA pre-filter to 20 candidates (fold-train only) -> mRMR QUBO (relevance, redundancy, exactly k)
-> solved four ways -> logistic regression. Controls: random k of the same 20; classical greedy mRMR on the same 20.

**Accuracy (bal.acc), k = 6 / 8 / 10:**

| Solver | dc | regional | fc_blocks |
|---|---|---|---|
| Exact (brute force, true optimum) | 0.191 / 0.196 / 0.193 | 0.166 / 0.163 / 0.166 | 0.153 / 0.160 / 0.143 |
| Simulated annealing | 0.182 / 0.194 / 0.189 | 0.172 / 0.178 / 0.191 | 0.142 / 0.170 / 0.141 |
| **QAOA (quantum, simulated)** | 0.148 / 0.192 / 0.188 | 0.189 / **0.200** / 0.164 | 0.158 / 0.172 / 0.146 |
| Random subset (control) | 0.184 / 0.188 / 0.193 | 0.177 / 0.210 / 0.181 | 0.157 / 0.139 / 0.149 |
| Greedy mRMR (classical) | 0.173 / 0.189 / 0.178 | 0.166 / 0.164 / 0.164 | 0.144 / 0.160 / 0.148 |

**Solver quality (how often each reached the true QUBO optimum, over 25 folds):**

| Solver | Hit optimum | Mean time per solve |
|---|---|---|
| Exact | 100 % (by definition) | 0.01 s |
| Simulated annealing | 20 to 96 % (mean cost gap 0.0003 to 0.018) | 0.4 s |
| QAOA, p = 2, 20 qubits | **0 to 16 %** (mean cost gap 0.04 to 0.09); 17 to 30 % of shots had exactly k features | about 140 s |
| Random | 0 % (cost gap about 0.45) | 0 s |

Reading:
* **The QUBO approach works as designed but does not help accuracy.** Even the exact optimum gives
  0.14 to 0.20, below PCA (0.253), and **random subsets of the same 20 candidates score the same**. The QUBO
  objective (relevance minus redundancy on about 64 subjects) is too noisy to find better features: the main risk
  named in section 6.2c happened.
* **QAOA is the weakest optimiser**: shallow p = 2 circuits rarely reach the optimum, and it is about 350x slower
  than simulated annealing and 10,000x slower than brute force at this size. Its accuracy is not worse only
  because being optimal does not matter here (previous point).
* As predicted in 6.4: **no quantum advantage**. QUBO is reported as the quantum study, not as the winner.
