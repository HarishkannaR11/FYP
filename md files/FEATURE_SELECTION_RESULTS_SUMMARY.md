# Feature Selection: Results Summary

Written 2026-10-18 for Dhananchezhiyan M. Final results of the feature-selection / dimensionality-reduction
study. Full process, every intermediate table and the method-by-method status are in
`md files/FEATURE_SELECTION_PROCESS_AND_QUANTUM_METHODS.md` (sections 9b and 10).

---

## 1. Short answer

| Question | Answer |
|---|---|
| How many methods were tried? | **26 methods, 285 configurations** (method x input view x number of features), plus 10 more methods listed and excluded with reasons |
| Which one performed best in cross-validation? | **PCA, 10 components, on the 738 regional features (mALFF + mReHo + DC_z): balanced accuracy 0.253 +/- 0.037** (chance 0.167) |
| Features reduced from | **738 -> 10** (98.6 % fewer). On all four biomarkers the best was 984 -> 20 (LRA, 0.248) |
| Final selected feature count | **10** |
| Leak check (shuffled labels) | passed: 0.167 (chance) |
| Held-out test (35 subjects, used once) | **0.172** for PCA; 0.228 for L1 (10 named features); 0.233 for the quantum QUBO + QAOA (8 features). All three are inside the range chance alone gives on 35 subjects |
| Quantum method | runs correctly, but gives no accuracy advantage (CV 0.200, test 0.233, not significant) |
| 96 % target | not reachable honestly with this data; best honest 6-stage result is about 0.25 in CV |

---

## 2. Input: features before reduction

Source: `derivatives/model_ready/`. 116 subjects: 81 training pool (used for everything below) and 35 locked
test subjects (used once, section 8). Stages CN, SMC, EMCI, MCI, LMCI, AD; chance = 1/6 = 0.167.

| Input view | Biomarkers | Features before |
|---|---|---:|
| dc | DC_z | 246 |
| **regional** | mALFF + mReHo + DC_z | **738** |
| **all4** | mALFF + mReHo + DC_z + FC per region (mean \|FC\| of each region) | **984** |
| fc_blocks | FC averaged within/between the 24 Brainnetome gyri | 300 |
| fc_edges | every FC edge (upper triangle of 246 x 246) | 30,135 |

Each cross-validation fold trains on only about **64 subjects**, so 738 or 984 features is 11 to 15 features
per subject. That is the core problem the reduction has to solve.

---

## 3. How every method was scored (cross-validation)

```
for each of 5 seeds (20261015 .. 20261019):
    split the 81 training subjects into 5 stratified folds (same folds as all earlier experiments)
    for each fold:
        on ~64 training subjects:  z-score  ->  reduce/select features  ->  logistic regression
        on ~16 validation subjects: apply the same fitted steps -> predict stage -> balanced accuracy
    seed score = mean of its 5 folds
CV result = mean +/- sd of the 5 seed scores   (25 train/validate runs per configuration)
```

* The classifier is **the same for every method** (logistic regression, C = 0.05, class-balanced), so
  differences come only from the feature step.
* Every learned step, including use of the labels by supervised selectors, is fitted on the fold's training
  subjects only. The 35 test subjects were not loaded until section 8.
* Main metric: **balanced accuracy** (average recall of the 6 stages). Also recorded: accuracy, macro-F1, mean
  absolute error in stages, quadratic weighted kappa, and selection stability across folds.
* Baselines on the same folds: majority class 0.167; **no reduction 0.176** (regional) / 0.168 (dc).

---

## 4. All models tried, with cross-validation results

Best configuration of each method (out of all views and feature counts tried). "Features" = before -> after.

### 4.1 Normal methods

| Method | Views tried | Feature counts tried | Best CV bal.acc | Best view | Features |
|---|---|---|---:|---|---|
| **PCA** | dc, regional, fc_edges, all4 | 2, 5, 10, 15, 20, 30, 40, 50, 60 | **0.253 +/- 0.037** | regional | 738 -> 10 |
| LRA (truncated SVD) | dc, regional, fc_edges, all4 | 5, 10, 20, 40 | 0.248 +/- 0.015 | all4 | 984 -> 20 |
| LDA (shrinkage) | dc, regional, all4 | 5 | 0.167 +/- 0.021 | dc | 246 -> 5 |
| PCA-40 + LDA | dc, regional, fc_edges, all4 | 5 | 0.194 +/- 0.025 | dc | 246 -> 5 |

### 4.2 Advanced classical: feature extraction (new combined features)

| Method | Views tried | Feature counts tried | Best CV bal.acc | Best view | Features |
|---|---|---|---:|---|---|
| ICA | dc, regional, fc_blocks, all4 | 5, 10, 20 | 0.259 +/- 0.040 | regional | 738 -> 10 |
| Sparse PCA | dc, regional, fc_blocks, all4 | 5, 10, 20 | 0.239 +/- 0.006 | regional | 738 -> 20 |
| NMF | dc, regional, fc_blocks, all4 | 5, 10, 20 | 0.226 +/- 0.027 | all4 | 984 -> 10 |
| PLS-DA | dc, regional, fc_blocks, fc_edges, all4 | 5, 10, 20 | 0.222 +/- 0.038 | fc_edges | 30,135 -> 5 |
| Kernel PCA (RBF) | dc, regional, fc_blocks, all4 | 5, 10, 20 | 0.213 +/- 0.037 | regional | 738 -> 10 |
| Isomap | dc, regional, fc_blocks, all4 | 5, 10, 20 | 0.204 +/- 0.028 | fc_blocks | 300 -> 10 |

### 4.3 Advanced classical: feature selection (keeps original, named features)

| Method | Views tried | Feature counts tried | Best CV bal.acc | Best view | Features |
|---|---|---|---:|---|---|
| **L1 logistic regression** | dc, regional, fc_blocks, all4 | 5 to 700 | **0.239 +/- 0.037** | regional | 738 -> 10 |
| Boruta | dc, regional, fc_blocks, all4 | automatic (about 10) | 0.236 +/- 0.013 | all4 | 984 -> 10 |
| mRMR | dc, regional, fc_blocks, fc_edges, all4 | 10, 20, 50 | 0.234 +/- 0.044 | fc_edges | 30,135 -> 50 |
| ANOVA F-test | dc, regional, fc_blocks, fc_edges, all4 | 5 to 700 | 0.231 +/- 0.040 | fc_edges | 30,135 -> 50 |
| ReliefF | dc, regional, fc_blocks, all4 | 10, 20, 50 | 0.224 +/- 0.031 | regional | 738 -> 50 |
| Stability selection | dc, regional, fc_blocks | 10, 20, 50 | 0.221 +/- 0.041 | regional | 738 -> 10 |
| Mutual information | dc, regional, fc_blocks, all4 | 10, 20, 50 | 0.220 +/- 0.049 | all4 | 984 -> 20 |
| Random-forest importance | dc, regional, fc_blocks, all4 | 10, 20, 50 | 0.213 +/- 0.027 | regional | 738 -> 20 |
| SVM-RFE | dc, regional, fc_blocks, all4 | 10, 20, 50 | 0.212 +/- 0.040 | regional | 738 -> 20 |
| Genetic algorithm | dc, regional, fc_blocks, all4 | automatic (22 to 30) | 0.204 +/- 0.017 | regional | 738 -> 29 |
| Particle swarm (binary PSO) | dc, regional, fc_blocks, all4 | automatic (39 to 46) | 0.199 +/- 0.017 | regional | 738 -> 45 |

### 4.4 Quantum and quantum-inspired (QUBO: relevance - redundancy, exactly k features)

Each fold: ANOVA pre-filter to 20 candidates -> QUBO -> solved four ways. 20 candidates = 20 qubits.

| Method | Views tried | Feature counts tried | Best CV bal.acc | Best view | Features | Reached true optimum |
|---|---|---|---:|---|---|---|
| **QUBO + QAOA** (qiskit 2.6.0, Aer simulator, p = 2) | dc, regional, fc_blocks, all4 | 6, 8, 10 | 0.200 +/- 0.061 | regional | 738 -> 8 | 0 to 16 % of folds |
| QUBO exact (brute force) | same | same | 0.196 +/- 0.024 | dc | 246 -> 8 | 100 % |
| QUBO simulated annealing (dwave-neal) | same | same | 0.194 +/- 0.021 | dc | 246 -> 8 | 20 to 96 % |
| Control: random 8 of the same 20 | same | same | 0.210 +/- 0.065 | regional | 738 -> 8 | 0 % |
| Control: greedy mRMR on the same 20 | same | same | 0.189 +/- 0.035 | dc | 246 -> 8 | n/a |

### 4.5 Listed but not run (reasons)

| Method | Reason |
|---|---|
| UMAP | package not installed; Isomap run as the manifold method |
| Autoencoder / concrete autoencoder | torch not installed; about 64 subjects is far too few to train one |
| SHAP-based selection | package not installed; covered by RF importance and Boruta |
| Grey wolf optimiser | same family as GA and PSO, which were both run |
| VQE | same QUBO as QAOA with a different circuit; kept as a spare |
| D-Wave quantum annealing | needs a cloud account; the QUBO can be sent to it unchanged |
| Grover adaptive search | far beyond simulable circuit depth |
| Quantum-inspired evolutionary | classical metaheuristic; GA/PSO cover the family |
| qPCA | needs QRAM; still unsupervised PCA |
| Quantum kernel SVM | optional extension; not started |

---

## 5. Leaderboard (best 10 configurations overall)

| # | Method | View | Features | CV bal.acc | macro-F1 | MAE (stages) |
|---:|---|---|---|---:|---:|---:|
| 1 | ICA | regional | 738 -> 10 | 0.259 +/- 0.040 | 0.227 | 1.70 |
| 2 | **PCA** | regional | 738 -> 10 | **0.253 +/- 0.037** | 0.224 | 1.70 |
| 3 | LRA | all4 | 984 -> 20 | 0.248 +/- 0.015 | 0.213 | 1.73 |
| 4 | Sparse PCA | regional | 738 -> 20 | 0.239 +/- 0.006 | 0.208 | 1.69 |
| 5 | L1 logistic | regional | 738 -> 10 | 0.239 +/- 0.037 | 0.222 | 1.72 |
| 6 | Boruta | all4 | 984 -> 10 | 0.236 +/- 0.013 | 0.205 | 1.82 |
| 7 | mRMR | fc_edges | 30,135 -> 50 | 0.234 +/- 0.044 | 0.205 | 1.82 |
| 8 | ANOVA | fc_edges | 30,135 -> 50 | 0.231 +/- 0.040 | 0.194 | 1.84 |
| 9 | PCA | regional | 738 -> 20 | 0.229 +/- 0.026 | 0.213 | 1.76 |
| 10 | LRA | regional | 738 -> 20 | 0.229 +/- 0.020 | 0.199 | 1.76 |

Best per input view: regional 0.259 (ICA) / 0.253 (PCA); all4 0.248 (LRA); fc_edges 0.234 (mRMR);
fc_blocks 0.216 (NMF); dc 0.212 (LRA). **Adding FC to the three regional biomarkers (all4) did not help.**

---

## 5b. Same input for every method: all four biomarkers (984 features)

Section 4 lists each method's best over all five input views, so the "features before" differ between rows.
This table compares every method on **one fixed input, all4 = 984 features** (mALFF + mReHo + DC_z + FC per
region). Best k per method shown; all k values tried in the last column.

| # | Method | Type | CV bal.acc | 984 -> | macro-F1 | All k tried (bal.acc) |
|---:|---|---|---:|---:|---:|---|
| 1 | **LRA** | normal | **0.248 +/- 0.015** | **20** | 0.213 | 5: 0.157, 10: 0.214, 20: 0.248, 40: 0.200 |
| 2 | Boruta | selection | 0.236 +/- 0.013 | 10 | 0.205 | auto: 0.236 |
| 3 | NMF | extraction | 0.226 +/- 0.027 | 10 | 0.193 | 5: 0.204, 10: 0.226, 20: 0.210 |
| 4 | Sparse PCA | extraction | 0.222 +/- 0.026 | 10 | 0.192 | 5: 0.198, 10: 0.222, 20: 0.213 |
| 5 | Mutual information | selection | 0.220 +/- 0.049 | 20 | 0.198 | 10: 0.187, 20: 0.220, 50: 0.220 |
| 6 | ReliefF | selection | 0.216 +/- 0.029 | 20 | 0.193 | 10: 0.196, 20: 0.216, 50: 0.202 |
| 7 | RF importance | selection | 0.212 +/- 0.038 | 20 | 0.192 | 10: 0.199, 20: 0.212, 50: 0.197 |
| 8 | L1 logistic | selection | 0.210 +/- 0.041 | 20 | 0.192 | 10: 0.178, 20: 0.210, 50: 0.186 |
| 9 | Kernel PCA (RBF) | extraction | 0.209 +/- 0.027 | 5 | 0.175 | 5: 0.209, 10: 0.201, 20: 0.171 |
| 10 | PCA | normal | 0.204 +/- 0.026 | 10 | 0.173 | 5: 0.191, 10: 0.204, 20: 0.190, 40: 0.181 |
| 11 | ICA | extraction | 0.201 +/- 0.025 | 10 | 0.173 | 5: 0.188, 10: 0.201, 20: 0.190 |
| 12 | ANOVA | selection | 0.201 +/- 0.033 | 50 | 0.188 | 10: 0.162, 20: 0.198, 50: 0.201 |
| 13 | QUBO random-subset control | control | 0.196 +/- 0.029 | 8 | 0.172 | 6: 0.159, 8: 0.196, 10: 0.176 |
| 14 | PCA-40 + LDA | normal | 0.193 +/- 0.035 | 5 | 0.169 | 5: 0.193 |
| 15 | PLS-DA | extraction | 0.191 +/- 0.038 | 5 | 0.168 | 5: 0.191, 10: 0.177, 20: 0.163 |
| 16 | SVM-RFE | selection | 0.189 +/- 0.036 | 50 | 0.174 | 10: 0.187, 20: 0.176, 50: 0.189 |
| 17 | mRMR | selection | 0.188 +/- 0.053 | 20 | 0.171 | 10: 0.161, 20: 0.188, 50: 0.186 |
| 18 | Particle swarm | selection | 0.188 +/- 0.069 | 46 | 0.172 | auto: 0.188 |
| 19 | Isomap | extraction | 0.187 +/- 0.018 | 5 | 0.150 | 5: 0.187, 10: 0.172, 20: 0.171 |
| 20 | **QUBO + QAOA (quantum)** | selection | 0.181 +/- 0.025 | 6 | 0.154 | 6: 0.181, 8: 0.148, 10: 0.160 |
| 21 | Genetic algorithm | selection | 0.173 +/- 0.042 | 30 | 0.161 | auto: 0.173 |
| 22 | QUBO exact | selection | 0.164 +/- 0.042 | 10 | 0.147 | 6: 0.147, 8: 0.158, 10: 0.164 |
| 23 | LDA | normal | 0.162 +/- 0.018 | 5 | 0.146 | 5: 0.162 |
| 24 | Greedy mRMR (20-pool) | control | 0.161 +/- 0.040 | 10 | 0.145 | 6: 0.147, 8: 0.161, 10: 0.161 |
| 25 | QUBO simulated annealing | selection | 0.151 +/- 0.051 | 6 | 0.124 | 6: 0.151, 8: 0.149, 10: 0.150 |

Stability selection was not run on all4 (about 1 h per configuration). No-reduction baseline on 984: not run
separately; on 738 it is 0.176.

Reading: on the fixed 984 input the best is **LRA, 984 -> 20, 0.248**, then Boruta 984 -> 10 (0.236). PCA drops
from 0.253 (on 738) to 0.204 (on 984). For almost every method the 738-feature regional input scores higher than
the 984-feature all4 input: **the 246 FC-per-region features add noise rather than stage signal**, so the
three-biomarker input is the better choice.

---

## 5c. Training vs validation accuracy (overfitting check)

Computed 2026-10-18 on the same 25 folds: training = balanced accuracy on the ~64 subjects the model was fitted
on; validation = on the ~16 held-back subjects of the same fold.

| Configuration | Features | Training bal.acc | Validation bal.acc | Gap |
|---|---|---:|---:|---:|
| No reduction | 738 | **1.000** | 0.176 | 0.82 |
| No reduction | 984 | **1.000** | 0.180 | 0.82 |
| **PCA k=10 (chosen)** | 738 -> 10 | **0.483 +/- 0.032** | **0.253 +/- 0.037** | **0.23** |
| LRA k=20 | 984 -> 20 | 0.668 +/- 0.011 | 0.248 +/- 0.015 | 0.42 |
| L1 k=10 | 738 -> 10 | 0.709 +/- 0.006 | 0.239 +/- 0.037 | 0.47 |
| Sparse PCA k=20 | 738 -> 20 | 0.640 +/- 0.013 | 0.239 +/- 0.006 | 0.40 |
| Boruta | 984 -> 10 | 0.576 +/- 0.054 | 0.236 +/- 0.013 | 0.34 |
| QUBO exact k=8 | 738 -> 8 | 0.654 +/- 0.014 | 0.163 +/- 0.045 | 0.49 |

Reading:
* **Without reduction the model memorises the training subjects (100 %) and fails on new ones (18 %):**
  classic overfitting, from 738 to 984 features on about 64 subjects.
* **Reduction shrinks the gap** (PCA: 48 % train vs 25 % validation), which is why it helps. PCA k=10 has the
  smallest gap of all, another reason it was chosen.
* Even on its own training data the PCA model reaches only 48 %: the stage pattern is not clearly present
  even in the subjects it learns from. Selection methods (L1, QUBO) fit training better (65 to 71 %) but
  generalise no better, so their extra training accuracy is fitted noise.

---

## 6. Which one is better, and why it was chosen

Chosen from cross-validation only, before any test subject was read:
**PCA with 10 components on the 738 regional features.**

1. **Highest CV score in practice.** ICA (0.259) is listed first, but ICA only rotates the same 10 PCA
   components; with this classifier it is the PCA result again (on DC_z the two give identical numbers).
   The 0.006 gap is noise, so the simpler method is taken.
2. **The top five (0.239 to 0.259) are tied within one sd**, so the choice among them goes to the simplest:
   PCA has one setting (k) and no randomness, and is standard and easy to defend.
3. **Passed the shuffled-label leak check**: 0.167, exactly chance.
4. **10 is the right number of features.** The sweep from 2 to all 738 features (section 7) peaks at 10.

Two more were reported alongside it, declared **before** the test (not used to choose):
* **L1 logistic, 10 features**: best method that keeps **named** brain regions (CV 0.239, tied with PCA).
* **QUBO + QAOA, 8 features**: the quantum method mam asked about (CV 0.200).

---

## 7. How many features to keep (738 -> k)

| Features kept | PCA | PCA variance kept | L1 | ANOVA |
|---:|---:|---:|---:|---:|
| 5 | 0.204 | 31 % | 0.222 | 0.172 |
| **10** | **0.253** | 47 % | **0.239** | 0.177 |
| 20 | 0.229 | 66 % | 0.223 | 0.226 |
| 50 | 0.148 | | 0.203 | 0.219 |
| 200 | | | 0.157 | 0.203 |
| 700 | | | 0.174 | 0.179 |
| all 738 | 0.176 | 100 % | 0.176 | 0.176 |

Keeping more features makes the result **worse**: with about 64 training subjects the classifier fits noise.
The best number is about 10, so the **final selected feature count is 10**.

---

## 8. Held-out test (35 locked subjects, used once)

Fitted on all 81 training-pool subjects, then evaluated once on the 35 test subjects
(CN 6, SMC 6, EMCI 6, MCI 5, LMCI 6, AD 6). Shuffled-label controls on the same configurations first:

| Configuration | Features | Shuffled-label CV | CV bal.acc | **Test bal.acc** | Test correct | 95 % CI of test accuracy | p vs chance |
|---|---|---:|---:|---:|---:|---|---:|
| **PCA k=10, regional (chosen)** | 738 -> 10 | 0.167 | 0.253 | **0.172** | 6 / 35 | 0.07 to 0.34 | 0.54 |
| L1 k=10, regional | 738 -> 10 | 0.170 | 0.239 | 0.228 | 8 / 35 | 0.10 to 0.40 | 0.22 |
| QUBO + QAOA k=8, regional | 738 -> 8 | 0.161 | 0.200 | 0.233 | 8 / 35 | 0.10 to 0.40 | 0.22 |

Other test metrics: PCA macro-F1 0.178, MAE 1.86 stages, QWK +0.08; L1 macro-F1 0.210, MAE 1.74, QWK +0.17;
QAOA macro-F1 0.212, MAE 1.77, QWK +0.07. Confusion matrices are in `final_test.log`.

**Honest reading:**
* **The chosen PCA model scores at chance on the test set (0.172).** Its CV score (0.253) did not carry over.
  Part of the CV score was luck: picking the best of 285 configurations makes the top CV numbers optimistic.
* L1 and QAOA score a little higher on the test (8 vs 6 correct), but **two subjects is within noise**: none of
  the three is significantly above chance (all p > 0.2), and their confidence intervals overlap almost fully.
  The test set must **not** be used to switch the winner afterwards; that would turn it into a tuning set.
* This agrees with the earlier project result (MLP 0.307 in CV, 0.20 on the same test set) and with the
  statistics (nothing survives correction): **with 116 subjects the rs-fMRI features carry very little
  usable 6-stage signal, whatever reduction method is used.**

---

## 9. Final selected features

**Chosen model (PCA k=10):** 10 components, each a weighted mix of all 738 regional features
(mALFF, mReHo, DC_z of the 246 regions). They keep 47 % of the variance. No original feature is dropped, but
components are not single brain regions.

**Named alternative (L1 k=10), selected when fitted on all 81 training subjects:**

| # | Region (Brainnetome) | Biomarker |
|---:|---|---|
| 1 | A45r_R | DC_z |
| 2 | A12/47l_R | mReHo |
| 3 | A4tl_R | mALFF |
| 4 | A20il_R | DC_z |
| 5 | A37vl_L | mReHo |
| 6 | A37mv_L | mALFF |
| 7 | A40rd_R | DC_z |
| 8 | A40rv_L | mALFF |
| 9 | vmPu_L | mReHo |
| 10 | mPFtha_R | mReHo |

**Quantum QUBO + QAOA (k=8):** mOccG_L DC_z, A40rd_R DC_z, A4tl_R mALFF, A39rd_L mALFF, A37vl_L mReHo,
A37dl_R DC_z, mOccG_R DC_z, NAC_L mReHo.

Regions picked repeatedly across methods and folds: **A4tl_R (mALFF), A37vl_L (mReHo), A40rd_R (DC_z),
vmPu_L (mReHo)**. Treat them as candidates to look at, not as established markers: selection stability
across folds was low (about 0.1 to 0.3).

---

## 10. Quantum method: what to tell mam

* QUBO feature selection was built, solved on a quantum simulator with QAOA (20 qubits), and checked against the
  exact optimum and simulated annealing, inside the same leak-free cross-validation.
* QAOA rarely found the true optimum (0 to 16 % of folds) and took about 140 to 280 s per solve, compared with
  0.4 s for simulated annealing and 0.01 s for brute force.
* Even the **exact** QUBO optimum was no better than **random** subsets of the same 20 candidates, so the
  limit is the data, not the solver. **No quantum advantage** at this size, as expected.
* Its test result (0.233) is not significantly above chance and not significantly different from L1.

---

## 11. Conclusion and recommendation

1. **Report:** 26 methods compared fairly; best CV result PCA 738 -> 10 at 0.253 (chance 0.167, no reduction
   0.176); top methods tied at about 0.24 to 0.26; feature reduction to about 10 is clearly better than keeping
   all features.
2. **Report the test result honestly:** 0.17 to 0.23 on the 35 held-out subjects, not significantly above chance.
3. **For the pipeline and dashboard**, use the 10-feature reduction. If named regions are needed, use **L1 with
   10 features**: it is tied with PCA in CV and gives the region list above.
4. **The 96 % target is not reachable** for 6 stages with this cohort; ask mam which task and metric she meant
   (a 2-class task such as CN vs AD may score much higher). Improving the real result needs **more subjects**,
   not a different reduction method.

---

## 12. Files

| Item | Location |
|---|---|
| Method scripts | `scripts/model/feature_selection_cv.py`, `scripts/model/feature_selection_methods.py` |
| Final test script | `scripts/model/feature_selection_final.py` |
| Per-step results | `derivatives/model_training/feature_selection/step5*`, `step6a*`, `step6b*`, `step6c*`, `step7*`, `ksweep_regional*` (`.log` = readable table, `_records.json` = every fold, selected features) |
| Final test | `derivatives/model_training/feature_selection/final_test.log`, `final_test_records.json` |
| Full process document | `md files/FEATURE_SELECTION_PROCESS_AND_QUANTUM_METHODS.md` |

---

## 13. Training vs validation accuracy: all models

Computed 2026-10-18 with `scripts/model/feature_selection_train_val.py` (`train_vs_validation.log`). Same 5 seeds x 5 folds. **Training** = balanced accuracy on the ~64 subjects the model was fitted on; **validation** = on the ~16 held-back subjects of the same fold (the CV result used everywhere above). Mean +/- sd over the 5 seeds. Chance 0.167. For selectors, the features stored for each fold were reused; the recomputed validation score matched the stored one exactly in every fold.

Baseline, no reduction: 738 features train 1.000 / validation 0.176; 984 features train 1.000 / validation 0.180.

### 13.1 Each method at its best configuration (any input)

| # | Method | Input | Features | Training bal.acc | Validation bal.acc | Gap |
|---:|---|---|---|---:|---:|---:|
| 1 | ICA | regional | 738 -> 10 | 0.486 +/- 0.032 | 0.259 +/- 0.040 | 0.23 |
| 2 | PCA | regional | 738 -> 10 | 0.483 +/- 0.032 | 0.253 +/- 0.037 | 0.23 |
| 3 | LRA | all4 | 984 -> 20 | 0.668 +/- 0.011 | 0.248 +/- 0.015 | 0.42 |
| 4 | Sparse PCA | regional | 738 -> 20 | 0.640 +/- 0.013 | 0.239 +/- 0.006 | 0.40 |
| 5 | L1 logistic | regional | 738 -> 10 | 0.709 +/- 0.006 | 0.239 +/- 0.037 | 0.47 |
| 6 | Boruta | all4 | 984 -> 10 | 0.576 +/- 0.054 | 0.236 +/- 0.013 | 0.34 |
| 7 | mRMR | fc_edges | 30,135 -> 50 | 1.000 +/- 0.000 | 0.234 +/- 0.044 | 0.77 |
| 8 | ANOVA | fc_edges | 30,135 -> 50 | 0.998 +/- 0.002 | 0.231 +/- 0.040 | 0.77 |
| 9 | NMF | all4 | 984 -> 10 | 0.429 +/- 0.011 | 0.226 +/- 0.027 | 0.20 |
| 10 | ReliefF | regional | 738 -> 50 | 0.963 +/- 0.007 | 0.224 +/- 0.031 | 0.74 |
| 11 | PLS-DA | fc_edges | 30,135 -> 5 | 0.980 +/- 0.008 | 0.222 +/- 0.038 | 0.76 |
| 12 | Stability selection | regional | 738 -> 10 | 0.667 +/- 0.027 | 0.221 +/- 0.041 | 0.45 |
| 13 | Mutual information | all4 | 984 -> 20 | 0.718 +/- 0.020 | 0.220 +/- 0.049 | 0.50 |
| 14 | Kernel PCA (RBF) | regional | 738 -> 10 | 0.471 +/- 0.025 | 0.213 +/- 0.037 | 0.26 |
| 15 | RF importance | regional | 738 -> 20 | 0.778 +/- 0.017 | 0.213 +/- 0.027 | 0.56 |
| 16 | SVM-RFE | regional | 738 -> 20 | 0.874 +/- 0.009 | 0.212 +/- 0.040 | 0.66 |
| 17 | QUBO random-subset control | regional | 738 -> 8 | 0.597 +/- 0.017 | 0.210 +/- 0.065 | 0.39 |
| 18 | Isomap | fc_blocks | 300 -> 10 | 0.468 +/- 0.016 | 0.204 +/- 0.028 | 0.26 |
| 19 | Genetic algorithm | regional | 738 -> 29 | 0.884 +/- 0.014 | 0.204 +/- 0.017 | 0.68 |
| 20 | **QUBO + QAOA (quantum)** | regional | 738 -> 8 | 0.645 +/- 0.012 | 0.200 +/- 0.061 | 0.45 |
| 21 | Particle swarm | regional | 738 -> 45 | 0.942 +/- 0.008 | 0.199 +/- 0.017 | 0.74 |
| 22 | QUBO exact | dc | 246 -> 8 | 0.557 +/- 0.023 | 0.196 +/- 0.024 | 0.36 |
| 23 | QUBO simulated annealing | dc | 246 -> 8 | 0.555 +/- 0.018 | 0.194 +/- 0.021 | 0.36 |
| 24 | PCA-40 + LDA | dc | 246 -> 5 | 0.649 +/- 0.025 | 0.194 +/- 0.025 | 0.45 |
| 25 | Greedy mRMR (20-pool control) | dc | 246 -> 8 | 0.559 +/- 0.021 | 0.189 +/- 0.035 | 0.37 |
| 26 | LDA | dc | 246 -> 5 | 0.980 +/- 0.005 | 0.167 +/- 0.021 | 0.81 |

### 13.2 Every method on the same input: all four biomarkers (984 features)

| # | Method | Input | Features | Training bal.acc | Validation bal.acc | Gap |
|---:|---|---|---|---:|---:|---:|
| 1 | LRA | all4 | 984 -> 20 | 0.668 +/- 0.011 | 0.248 +/- 0.015 | 0.42 |
| 2 | Boruta | all4 | 984 -> 10 | 0.576 +/- 0.054 | 0.236 +/- 0.013 | 0.34 |
| 3 | NMF | all4 | 984 -> 10 | 0.429 +/- 0.011 | 0.226 +/- 0.027 | 0.20 |
| 4 | Sparse PCA | all4 | 984 -> 10 | 0.460 +/- 0.013 | 0.222 +/- 0.026 | 0.24 |
| 5 | Mutual information | all4 | 984 -> 20 | 0.718 +/- 0.020 | 0.220 +/- 0.049 | 0.50 |
| 6 | ReliefF | all4 | 984 -> 20 | 0.800 +/- 0.018 | 0.216 +/- 0.029 | 0.58 |
| 7 | RF importance | all4 | 984 -> 20 | 0.777 +/- 0.015 | 0.212 +/- 0.038 | 0.57 |
| 8 | L1 logistic | all4 | 984 -> 20 | 0.895 +/- 0.012 | 0.210 +/- 0.041 | 0.68 |
| 9 | Kernel PCA (RBF) | all4 | 984 -> 5 | 0.363 +/- 0.008 | 0.209 +/- 0.027 | 0.15 |
| 10 | PCA | all4 | 984 -> 10 | 0.455 +/- 0.015 | 0.204 +/- 0.026 | 0.25 |
| 11 | ICA | all4 | 984 -> 10 | 0.454 +/- 0.016 | 0.201 +/- 0.025 | 0.25 |
| 12 | ANOVA | all4 | 984 -> 50 | 0.963 +/- 0.005 | 0.201 +/- 0.033 | 0.76 |
| 13 | QUBO random-subset control | all4 | 984 -> 8 | 0.596 +/- 0.020 | 0.196 +/- 0.029 | 0.40 |
| 14 | PCA-40 + LDA | all4 | 984 -> 5 | 0.644 +/- 0.025 | 0.193 +/- 0.035 | 0.45 |
| 15 | PLS-DA | all4 | 984 -> 5 | 0.863 +/- 0.015 | 0.191 +/- 0.038 | 0.67 |
| 16 | SVM-RFE | all4 | 984 -> 50 | 0.997 +/- 0.002 | 0.189 +/- 0.036 | 0.81 |
| 17 | mRMR | all4 | 984 -> 20 | 0.848 +/- 0.010 | 0.188 +/- 0.053 | 0.66 |
| 18 | Particle swarm | all4 | 984 -> 46 | 0.957 +/- 0.009 | 0.188 +/- 0.069 | 0.77 |
| 19 | Isomap | all4 | 984 -> 5 | 0.358 +/- 0.020 | 0.187 +/- 0.018 | 0.17 |
| 20 | **QUBO + QAOA (quantum)** | all4 | 984 -> 6 | 0.588 +/- 0.015 | 0.181 +/- 0.025 | 0.41 |
| 21 | Genetic algorithm | all4 | 984 -> 30 | 0.920 +/- 0.019 | 0.173 +/- 0.042 | 0.75 |
| 22 | QUBO exact | all4 | 984 -> 10 | 0.703 +/- 0.014 | 0.164 +/- 0.042 | 0.54 |
| 23 | LDA | all4 | 984 -> 5 | 1.000 +/- 0.000 | 0.162 +/- 0.018 | 0.84 |
| 24 | Greedy mRMR (20-pool control) | all4 | 984 -> 10 | 0.703 +/- 0.015 | 0.161 +/- 0.040 | 0.54 |
| 25 | QUBO simulated annealing | all4 | 984 -> 6 | 0.599 +/- 0.015 | 0.151 +/- 0.051 | 0.45 |

Reading:
* **Every model fits its training subjects far better than new subjects.** Training accuracy ranges from 0.36 to
  1.00, validation only from 0.15 to 0.26.
* Methods that keep many features or search hard for them (ANOVA/mRMR on 30,135 edges, LDA, SVM-RFE, GA, PSO,
  ReliefF k=50) reach **0.87 to 1.00 on training** but only about 0.17 to 0.23 on validation: they memorise the
  training subjects (overfitting).
* The extraction methods with few components (PCA, ICA, NMF, Kernel PCA, Isomap) have the **smallest gaps**
  (0.15 to 0.26). **PCA k=10 on 738: training 0.483, validation 0.253**, the best validation score with one of
  the smallest gaps.
* The quantum QUBO + QAOA (738 -> 8): training 0.645, validation 0.200.
* Even the best models reach under 0.50 on their own training data at low feature counts, which again shows the
  stage signal in these features is weak.
