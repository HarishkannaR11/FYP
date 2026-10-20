"""
Feature-selection / dimensionality-reduction comparison, steps 5 to 7 of
md files/FEATURE_SELECTION_PROCESS_AND_QUANTUM_METHODS.md.

Every method is a scikit-learn Pipeline  scale -> reduce/select -> classify,
run through the SAME 5 seeds x 5 folds as feature_selection_cv.py (fold
fingerprints are re-verified). Every learned step, including the selectors'
use of labels, is fitted on the fold's training subjects only. The 35 test
subjects are never loaded.

The classifier is fixed for every method (logistic regression, C=0.05,
class-balanced, the same one as the step-4 baseline), so differences come
from the feature step alone.

Run:
  python scripts/model/feature_selection_methods.py step5    # PCA, LDA, LRA
  python scripts/model/feature_selection_methods.py step6a   # Sparse PCA, PLS, Kernel PCA, ICA
  python scripts/model/feature_selection_methods.py step6b   # ANOVA, MI, mRMR, ReliefF, L1, SVM-RFE, RF, stability
  python scripts/model/feature_selection_methods.py step7    # QUBO: brute force, simulated annealing, QAOA
"""
import itertools
import json
import os
import platform
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feature_selection_cv as H  # noqa: E402  (sets thread env vars first)

import numpy as np  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
from sklearn.base import BaseEstimator, TransformerMixin  # noqa: E402
from sklearn.decomposition import PCA, FastICA, KernelPCA, MiniBatchSparsePCA, TruncatedSVD  # noqa: E402
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.feature_selection import (RFE, SelectFromModel, SelectKBest,  # noqa: E402
                                       f_classif, mutual_info_classif)
from sklearn.cross_decomposition import PLSRegression  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

log = H.log

# Brainnetome-246 gyral hierarchy (Fan et al. 2016): first region of each of the
# 24 gyri, 1-based; checked against atlases/Brainnetome246/BN_Atlas_246_LUT.txt
GYRUS_STARTS = [1, 15, 29, 41, 53, 65, 69, 81, 89, 103, 109, 121, 125, 135, 147,
                155, 163, 175, 189, 199, 211, 215, 219, 231]
GYRUS_NAMES = ["SFG", "MFG", "IFG", "OrG", "PrG", "PCL", "STG", "MTG", "ITG", "FuG",
               "PhG", "pSTS", "SPL", "IPL", "Pcun", "PoG", "INS", "CG", "MVOcC",
               "LOcC", "Amyg", "Hipp", "BG", "Tha"]


def classifier():
    return LogisticRegression(C=0.05, max_iter=2000, class_weight="balanced")


def roi_names():
    names = {}
    lut = os.path.join(H.ROOT, "atlases", "Brainnetome246", "BN_Atlas_246_LUT.txt")
    for line in open(lut):
        p = line.split()
        if p and p[0].isdigit() and 1 <= int(p[0]) <= 246:
            names[int(p[0]) - 1] = p[1]
    return [names[i] for i in range(246)]


def feature_names(view):
    rn = roi_names()
    if view in ("alff", "reho", "dc"):
        return [f"{r}|{view}" for r in rn]
    if view == "regional":   # X_regional reshaped (246, 3): roi-major
        return [f"{r}|{b}" for r in rn for b in ("mALFF", "mReHo", "DC_z")]
    if view == "all4":       # regional 738 then per-region FC strength 246
        return ([f"{r}|{b}" for r in rn for b in ("mALFF", "mReHo", "DC_z")]
                + [f"{r}|FC" for r in rn])
    if view == "fc_blocks":
        g = len(GYRUS_NAMES)
        return [f"{GYRUS_NAMES[a]}-{GYRUS_NAMES[b]}" for a in range(g) for b in range(a, g)]
    return None


def fc_blocks(fc_edges):
    """Average the 30,135 edges within and between the 24 gyri -> 300 features."""
    gid = np.zeros(246, int)
    for g, s in enumerate(GYRUS_STARTS):
        gid[s - 1:] = g
    iu = np.triu_indices(246, k=1)
    ga, gb = np.minimum(gid[iu[0]], gid[iu[1]]), np.maximum(gid[iu[0]], gid[iu[1]])
    cols = []
    for a in range(24):
        for b in range(a, 24):
            m = (ga == a) & (gb == b)
            cols.append(fc_edges[:, m].mean(axis=1))
    return np.stack(cols, axis=1)


# ------------------------------------------------------------ transformers
class PLSTransformer(BaseEstimator, TransformerMixin):
    """PLS-DA: PLS regression on one-hot labels, used as a supervised reducer."""

    def __init__(self, n_components=5):
        self.n_components = n_components

    def fit(self, X, y):
        Y = np.eye(H.N_STAGES)[y]
        self.pls_ = PLSRegression(n_components=self.n_components, scale=False).fit(X, Y)
        return self

    def transform(self, X):
        return self.pls_.transform(X)


class RankSelector(BaseEstimator, TransformerMixin):
    """Keep the top-k features of a score computed on the training subjects."""

    def __init__(self, k=20, method="mrmr", prefilter=None, random_state=0):
        self.k = k
        self.method = method
        self.prefilter = prefilter
        self.random_state = random_state

    def fit(self, X, y):
        cand = np.arange(X.shape[1])
        if self.prefilter and X.shape[1] > self.prefilter:
            F = np.nan_to_num(f_classif(X, y)[0])
            cand = np.argsort(-F)[:self.prefilter]
        Xc = X[:, cand]
        if self.method == "mrmr":
            sel = mrmr(Xc, y, self.k)
        elif self.method == "relieff":
            sel = np.argsort(-relieff(Xc, y))[:self.k]
        elif self.method == "stability":
            # ties (e.g. many features never picked) broken by ANOVA F, not by column order
            F = np.nan_to_num(f_classif(Xc, y)[0])
            sel = np.lexsort((-F, -stability_scores(Xc, y, self.random_state)))[:self.k]
        else:
            raise ValueError(self.method)
        self.support_idx_ = cand[np.asarray(sel)]
        return self

    def transform(self, X):
        return X[:, self.support_idx_]


def mrmr(X, y, k):
    """Greedy mRMR (MID form): relevance = ANOVA F scaled to [0, 1],
    redundancy = mean |Pearson r| with the already-selected features."""
    F = np.nan_to_num(f_classif(X, y)[0])
    rel = F / F.max() if F.max() > 0 else F
    C = np.abs(np.nan_to_num(np.corrcoef(X, rowvar=False)))
    sel = [int(np.argmax(rel))]
    red_sum = C[sel[0]].copy()
    while len(sel) < k:
        score = rel - red_sum / len(sel)
        score[sel] = -np.inf
        j = int(np.argmax(score))
        sel.append(j)
        red_sum += C[j]
    return sel


def relieff(X, y, n_neighbors=10):
    """Multi-class ReliefF (Kononenko 1994), Manhattan distance, features in [0, 1]."""
    rng_ = X.max(0) - X.min(0)
    rng_[rng_ == 0] = 1
    Z = (X - X.min(0)) / rng_
    n, d = Z.shape
    prior = np.bincount(y, minlength=H.N_STAGES) / n
    D = np.abs(Z[:, None, :] - Z[None, :, :]).sum(-1)
    W = np.zeros(d)
    for i in range(n):
        for c in range(H.N_STAGES):
            idx = np.flatnonzero((y == c) & (np.arange(n) != i))
            if idx.size == 0:
                continue
            nn = idx[np.argsort(D[i, idx])[:n_neighbors]]
            diff = np.abs(Z[nn] - Z[i]).mean(0)
            if c == y[i]:
                W -= diff
            else:
                W += prior[c] / (1 - prior[y[i]]) * diff
    return W / n


def stability_scores(X, y, seed, n_boot=50, C=0.5):
    """Stability selection (Meinshausen & Buhlmann 2010): how often each feature
    gets a non-zero L1 coefficient over half-sample subsamples."""
    rng = np.random.default_rng(seed)
    cnt = np.zeros(X.shape[1])
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(np.flatnonzero(y == c), max(1, (y == c).sum() // 2),
                                         replace=False) for c in np.unique(y)])
        m = LogisticRegression(penalty="l1", C=C, solver="saga", max_iter=3000)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m.fit(X[idx], y[idx])
        cnt += (np.abs(m.coef_).max(0) > 1e-8)
    return cnt / n_boot


def inner_cv_score(X, y, cols, n_splits=3, seed=0):
    """Balanced accuracy of the fixed classifier by inner stratified CV on the
    fold-TRAINING subjects only (used as the fitness of the metaheuristics)."""
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import balanced_accuracy_score
    if len(cols) == 0:
        return 0.0
    s = []
    for a, b in StratifiedKFold(n_splits, shuffle=True, random_state=seed).split(X, y):
        m = Pipeline([("s", StandardScaler()), ("c", classifier())]).fit(X[a][:, cols], y[a])
        s.append(balanced_accuracy_score(y[b], m.predict(X[b][:, cols])))
    return float(np.mean(s))


class MetaheuristicSelector(BaseEstimator, TransformerMixin):
    """Wrapper selection by a genetic algorithm or binary particle swarm.
    Search space: the top `pool` ANOVA features of the fold-training subjects.
    Fitness: inner 3-fold CV balanced accuracy on fold-train minus 0.002 per
    feature kept (a small size penalty). The outer validation fold is never seen."""

    def __init__(self, method="ga", pool=100, pop=30, gens=25, random_state=0):
        self.method = method
        self.pool = pool
        self.pop = pop
        self.gens = gens
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        F = np.nan_to_num(f_classif(X, y)[0])
        cand = np.argsort(-F)[:self.pool]
        Xc = X[:, cand]
        n = cand.size
        cache = {}

        def fit_of(mask):
            key = mask.tobytes()
            if key not in cache:
                cols = np.flatnonzero(mask)
                cache[key] = inner_cv_score(Xc, y, cols, seed=self.random_state) - 0.002 * cols.size
            return cache[key]

        P = rng.random((self.pop, n)) < (20 / n)          # start around 20 features
        if self.method == "ga":
            fit = np.array([fit_of(p) for p in P])
            for _ in range(self.gens):
                new = [P[np.argmax(fit)].copy()]              # elitism
                while len(new) < self.pop:
                    a, b = (P[max(rng.choice(self.pop, 3), key=lambda i: fit[i])] for _ in range(2))
                    cut = rng.random(n) < 0.5                 # uniform crossover
                    child = np.where(cut, a, b)
                    child ^= rng.random(n) < (1.0 / n)        # bit-flip mutation
                    new.append(child)
                P = np.array(new)
                fit = np.array([fit_of(p) for p in P])
            best = P[np.argmax(fit)]
        elif self.method == "pso":                            # binary PSO, sigmoid transfer
            V = rng.normal(0, 1, (self.pop, n))
            fit = np.array([fit_of(p) for p in P])
            pbest, pfit = P.copy(), fit.copy()
            g = pbest[np.argmax(pfit)].copy()
            for _ in range(self.gens):
                r1, r2 = rng.random((2, self.pop, n))
                V = np.clip(0.7 * V + 1.5 * r1 * (pbest.astype(float) - P)
                            + 1.5 * r2 * (g.astype(float) - P), -4, 4)
                P = rng.random((self.pop, n)) < 1 / (1 + np.exp(-V))
                fit = np.array([fit_of(p) for p in P])
                up = fit > pfit
                pbest[up], pfit[up] = P[up], fit[up]
                g = pbest[np.argmax(pfit)].copy()
            best = g
        else:
            raise ValueError(self.method)
        if not best.any():
            best = np.zeros(n, bool)
            best[0] = True
        self.support_idx_ = cand[np.flatnonzero(best)]
        return self

    def transform(self, X):
        return X[:, self.support_idx_]


class BorutaSelector(BaseEstimator, TransformerMixin):
    """Boruta (Kursa & Rudnicki 2010), compact version: each iteration adds a
    shuffled 'shadow' copy of every feature, fits a random forest, and counts a
    hit when a real feature beats the best shadow. Features whose hit count is
    significantly above 50% (one-sided binomial p < 0.05) are confirmed. If none
    are confirmed, the `fallback_k` highest-hit features are kept (recorded)."""

    def __init__(self, n_iter=30, fallback_k=10, random_state=0):
        self.n_iter = n_iter
        self.fallback_k = fallback_k
        self.random_state = random_state

    def fit(self, X, y):
        from scipy.stats import binom
        rng = np.random.default_rng(self.random_state)
        n, d = X.shape
        hits = np.zeros(d)
        for it in range(self.n_iter):
            Sh = np.column_stack([rng.permutation(X[:, j]) for j in range(d)])
            rf = RandomForestClassifier(200, random_state=self.random_state + it, n_jobs=-1)
            rf.fit(np.hstack([X, Sh]), y)
            imp = rf.feature_importances_
            hits += imp[:d] > imp[d:].max()
        p = binom.sf(hits - 1, self.n_iter, 0.5)
        conf = np.flatnonzero(p < 0.05)
        self.n_confirmed_ = int(conf.size)
        self.support_idx_ = conf if conf.size else np.argsort(-hits)[:self.fallback_k]
        return self

    def transform(self, X):
        return X[:, self.support_idx_]


class ShiftedNMF(BaseEstimator, TransformerMixin):
    """NMF needs non-negative input: shift each feature by its fold-train minimum."""

    def __init__(self, n_components=10, random_state=0):
        self.n_components = n_components
        self.random_state = random_state

    def fit(self, X, y=None):
        from sklearn.decomposition import NMF
        self.min_ = X.min(0)
        self.nmf_ = NMF(self.n_components, init="nndsvda", max_iter=1000,
                        random_state=self.random_state).fit(X - self.min_)
        return self

    def transform(self, X):
        return self.nmf_.transform(np.maximum(X - self.min_, 0))


# ------------------------------------------------------------------ QUBO
def build_qubo(X, y, k, alpha=0.5):
    """mRMR-style QUBO over the candidate features (Mucke et al. 2023 form):
       cost(x) = -alpha * sum_i rel_i x_i
                 + (1 - alpha) / (k - 1) * sum_{i<j} red_ij x_i x_j
                 + lam * (sum_i x_i - k)^2
    rel = ANOVA F scaled to [0, 1]; red = |Pearson r|. Returns (Q, offset) with
    cost = x^T Q x + offset for binary x (Q upper-triangular, diagonal = linear)."""
    F = np.nan_to_num(f_classif(X, y)[0])
    rel = F / F.max() if F.max() > 0 else F
    red = np.abs(np.nan_to_num(np.corrcoef(X, rowvar=False)))
    n = X.shape[1]
    w_red = (1 - alpha) / (k - 1)
    B = np.diag(-alpha * rel) + np.triu(w_red * red, 1)        # mRMR part
    P = np.diag(np.full(n, 1.0 - 2 * k)) + np.triu(np.full((n, n), 2.0), 1)   # (sum x - k)^2 - k^2
    # Penalty weight: a large one makes every feasible subset a local minimum and
    # traps annealing and QAOA (checked: SA missed the optimum in every fold).
    # Use the smallest lambda (x1.25 steps) whose UNconstrained global minimum
    # already has exactly k features, then a 1.5x safety margin. At 20 variables
    # this is checked exactly over all 2^20 states.
    base, weight = cost_diagonal(B, 0.0)
    pen = (weight - k) ** 2 - k * k
    lam = 0.01
    while weight[np.argmin(base + lam * pen)] != k:
        lam *= 1.25
    lam *= 1.5
    return B + lam * P, lam * k * k


def qubo_cost(Q, offset, x):
    x = np.asarray(x, float)
    return float(x @ Q @ x + offset)


def cost_diagonal(Q, offset):
    """QUBO cost of all 2^n bitstrings; index b <-> x_i = bit i of b
    (the same little-endian order as a qiskit statevector)."""
    n = Q.shape[0]
    b = np.arange(2 ** n, dtype=np.int64)
    Xb = ((b[:, None] >> np.arange(n)) & 1).astype(np.float64)
    return ((Xb @ Q) * Xb).sum(1) + offset, Xb.sum(1).astype(int)


def bits_of(idx, n):
    return ((int(idx) >> np.arange(n)) & 1).astype(int)


def solve_exact(Q, offset, k, diag=None):
    """Exact optimum by brute force over all 2^n subsets, restricted to exactly k."""
    cost, weight = diag if diag is not None else cost_diagonal(Q, offset)
    c = np.where(weight == k, cost, np.inf)
    return bits_of(np.argmin(c), Q.shape[0])


def solve_sa(Q, offset, seed, num_reads=500):
    import dimod
    import neal
    n = Q.shape[0]
    bqm = dimod.BinaryQuadraticModel.from_qubo(
        {(i, j): Q[i, j] for i in range(n) for j in range(i, n) if Q[i, j] != 0}, offset=offset)
    ss = neal.SimulatedAnnealingSampler().sample(bqm, num_reads=num_reads, seed=int(seed % 2**31))
    s = ss.first.sample
    return np.array([s[i] for i in range(n)], int)


def ising_hamiltonian(Q):
    """QUBO -> Ising with x_i = (1 - z_i) / 2 (constant dropped)."""
    from qiskit.quantum_info import SparsePauliOp
    n = Q.shape[0]
    terms, h = [], np.zeros(n)
    for i in range(n):
        h[i] += -Q[i, i] / 2
        for j in range(i + 1, n):
            if Q[i, j] != 0:
                terms.append(("ZZ", [i, j], Q[i, j] / 4))
                h[i] -= Q[i, j] / 4
                h[j] -= Q[i, j] / 4
    terms += [("Z", [i], h[i]) for i in range(n) if h[i] != 0]
    H_cost = SparsePauliOp.from_sparse_list(terms, num_qubits=n)
    return H_cost / np.max(np.abs(H_cost.coeffs))   # scale so angles in [0, pi] are sensible


def solve_qaoa(Q, offset, k, seed, diag=None, reps=2, maxiter=100, shots=4096):
    """QAOA (Farhi et al. 2014): qiskit QAOAAnsatz circuit simulated on the Aer
    statevector simulator. Each optimiser step runs the circuit once; the
    energy <psi|C|psi> is read from the statevector against the QUBO cost
    diagonal (exact, same value an Estimator returns, far faster than
    evaluating 200+ Pauli terms one by one). Angles tuned by COBYLA; the final
    state is sampled `shots` times and the lowest-cost sampled bitstring with
    exactly k ones is returned."""
    from qiskit import transpile
    from qiskit.circuit.library import QAOAAnsatz
    from qiskit_aer import AerSimulator
    from scipy.optimize import minimize

    n = Q.shape[0]
    cost, weight = diag if diag is not None else cost_diagonal(Q, offset)
    sim = AerSimulator(method="statevector")
    ansatz = QAOAAnsatz(ising_hamiltonian(Q), reps=reps)
    circ = ansatz.copy()
    circ.save_statevector()
    circ = transpile(circ, sim)
    n_eval = [0]

    def probs(theta):
        bound = circ.assign_parameters(dict(zip(ansatz.parameters, theta)))
        sv = np.asarray(sim.run(bound).result().get_statevector())
        return np.abs(sv) ** 2

    def energy(theta):
        n_eval[0] += 1
        return float(probs(theta) @ cost)

    rng = np.random.default_rng(seed)
    # QAOAAnsatz parameters are ordered beta[0..p-1], gamma[0..p-1]
    x0 = np.concatenate([rng.uniform(0, np.pi / 2, reps), rng.uniform(0, np.pi, reps)])
    res = minimize(energy, x0, method="COBYLA", options={"maxiter": maxiter})

    p = probs(res.x)
    p /= p.sum()
    sample = rng.choice(p.size, size=shots, p=p)
    feas = sample[weight[sample] == k]
    info = {"n_eval": n_eval[0], "feasible_shot_frac": float(feas.size / shots),
            "p_optimum": float(p[np.argmin(np.where(weight == k, cost, np.inf))]),
            "final_energy": float(res.fun)}
    if feas.size:
        return bits_of(feas[np.argmin(cost[feas])], n), info
    marg = (p[:, None] * ((np.arange(p.size)[:, None] >> np.arange(n)) & 1)).sum(0)
    x = np.zeros(n, int)               # no feasible sample: report it, use top-k marginals
    x[np.argsort(-marg)[:k]] = 1
    info["fallback_marginals"] = True
    return x, info


class QUBOSelector(BaseEstimator, TransformerMixin):
    """Stage A: ANOVA pre-filter to `pool` features (training subjects only).
    Stage B: pick exactly k of them by solving the mRMR QUBO with `solver`."""

    def __init__(self, k=8, pool=20, solver="exact", alpha=0.5, random_state=0):
        self.k = k
        self.pool = pool
        self.solver = solver
        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y):
        F = np.nan_to_num(f_classif(X, y)[0])
        cand = np.argsort(-F)[:self.pool]
        Q, off = build_qubo(X[:, cand], y, self.k, self.alpha)
        diag = cost_diagonal(Q, off)
        t0 = time.time()
        self.info_ = {}
        if self.solver == "exact":
            x = solve_exact(Q, off, self.k, diag)
        elif self.solver == "sa":
            x = solve_sa(Q, off, self.random_state)
        elif self.solver == "qaoa":
            x, self.info_ = solve_qaoa(Q, off, self.k, self.random_state, diag)
        elif self.solver == "random":
            x = np.zeros(self.pool, int)
            x[np.random.default_rng(self.random_state).choice(self.pool, self.k, replace=False)] = 1
        else:
            raise ValueError(self.solver)
        solve_s = time.time() - t0
        x_opt = solve_exact(Q, off, self.k, diag)   # ground truth, to grade every solver
        c, c_opt = qubo_cost(Q, off, x), qubo_cost(Q, off, x_opt)
        self.info_.update({"solve_s": solve_s, "cost": c, "cost_opt": c_opt,
                           "hit_optimum": bool(np.array_equal(x, x_opt)),
                           "n_selected": int(x.sum())})
        self.support_idx_ = cand[np.flatnonzero(x)]
        return self

    def transform(self, X):
        return X[:, self.support_idx_]


# --------------------------------------------------------------- harness
def run_cv2(name, X, y, plan, make_pipeline, shuffle_labels=False):
    """Like H.run_cv, but also records which features a selector kept."""
    recs = []
    t0 = time.time()
    for seed in H.SEEDS:
        fp, folds = plan[seed]
        for f, (tr, va) in enumerate(folds):
            ytr = y[tr].copy()
            if shuffle_labels:
                np.random.default_rng(seed * 100 + f).shuffle(ytr)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe = make_pipeline(seed + f)
                pipe.fit(X[tr], ytr)
                pred = pipe.predict(X[va]).astype(int)
            m = H.evaluate(y[va], pred)
            rec = {"method": name, "seed": seed, "fold": f, "fold_fingerprint": fp,
                   "n_train": int(len(tr)), "n_val": int(len(va)),
                   "n_features_in": int(X.shape[1]), "shuffled_labels": shuffle_labels, **m}
            sel = pipe.named_steps.get("select")
            if sel is not None:
                if hasattr(sel, "support_idx_"):
                    rec["selected"] = sel.support_idx_.tolist()
                elif hasattr(sel, "get_support"):
                    rec["selected"] = np.flatnonzero(sel.get_support()).tolist()
                if hasattr(sel, "info_"):
                    rec["solver_info"] = sel.info_
            rec["n_features_out"] = int(np.atleast_2d(pipe[:-1].transform(X[va])).shape[1])
            recs.append(rec)
    return recs, time.time() - t0


def stability(recs):
    """Mean pairwise Jaccard overlap of the selected sets across the 25 folds."""
    sets = [set(r["selected"]) for r in recs if "selected" in r]
    if len(sets) < 2:
        return None
    j = [len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2) if a | b]
    return float(np.mean(j))


def top_features(recs, names, n=10):
    from collections import Counter
    cnt = Counter(i for r in recs for i in r.get("selected", []))
    return [(names[i] if names else str(i), c) for i, c in cnt.most_common(n)]


def pipe(reducer=None, selector=None, scale_first=True):
    steps = [("scale", StandardScaler())] if scale_first else []
    if reducer is not None:
        steps.append(("reduce", reducer))
    if selector is not None:
        steps.append(("select", selector))
    steps += [("post", StandardScaler()), ("clf", classifier())]
    return Pipeline(steps)


# ------------------------------------------------------------- job lists
def jobs_step5():
    J = []
    for view in ("dc", "regional", "fc_edges", "all4"):
        for k in (5, 10, 20, 40):
            J.append((f"PCA_k{k}", view, lambda s, k=k: pipe(PCA(k, random_state=s))))
            J.append((f"LRA_k{k}", view,       # truncated SVD of the UNcentred raw matrix
                      lambda s, k=k: pipe(TruncatedSVD(k, random_state=s), scale_first=False)))
        if view != "fc_edges":
            J.append(("LDA_k5", view, lambda s: pipe(
                LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto", n_components=5))))
        J.append(("PCA40+LDA_k5", view, lambda s: pipe(Pipeline([
            ("pca", PCA(40, random_state=s)),
            ("lda", LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto",
                                               n_components=5))]))))
    return J


def jobs_step6a():
    J = []
    for view in ("dc", "regional", "fc_blocks", "all4"):
        for k in (5, 10, 20):
            J.append((f"SparsePCA_k{k}", view, lambda s, k=k: pipe(
                MiniBatchSparsePCA(k, alpha=1.0, batch_size=16, max_iter=200, random_state=s))))
            J.append((f"PLS_k{k}", view, lambda s, k=k: pipe(PLSTransformer(k))))
            J.append((f"KernelPCA_rbf_k{k}", view, lambda s, k=k: pipe(
                KernelPCA(k, kernel="rbf", random_state=s))))
            J.append((f"ICA_k{k}", view, lambda s, k=k: pipe(
                FastICA(k, whiten="unit-variance", max_iter=1000, random_state=s))))
    for k in (5, 10, 20):
        J.append((f"PLS_k{k}", "fc_edges", lambda s, k=k: pipe(PLSTransformer(k))))
    return J


def jobs_step6b():
    J = []
    for view in ("dc", "regional", "fc_blocks", "all4"):
        for k in (10, 20, 50):
            J.append((f"ANOVA_k{k}", view, lambda s, k=k: pipe(selector=SelectKBest(f_classif, k=k))))
            J.append((f"MI_k{k}", view, lambda s, k=k: pipe(selector=SelectKBest(
                lambda X, y, s=s: mutual_info_classif(X, y, random_state=s), k=k))))
            J.append((f"mRMR_k{k}", view, lambda s, k=k: pipe(selector=RankSelector(k, "mrmr"))))
            J.append((f"ReliefF_k{k}", view, lambda s, k=k: pipe(selector=RankSelector(k, "relieff"))))
            J.append((f"L1_k{k}", view, lambda s, k=k: pipe(selector=SelectFromModel(
                LogisticRegression(penalty="l1", C=0.5, solver="saga", max_iter=3000),
                max_features=k, threshold=-np.inf))))
            J.append((f"SVM-RFE_k{k}", view, lambda s, k=k: pipe(selector=RFE(
                LinearSVC(C=0.01, max_iter=5000), n_features_to_select=k, step=0.1))))
            J.append((f"RF_k{k}", view, lambda s, k=k: pipe(selector=SelectFromModel(
                RandomForestClassifier(300, random_state=s, n_jobs=-1),
                max_features=k, threshold=-np.inf))))
            J.append((f"Stability_k{k}", view, lambda s, k=k: pipe(
                selector=RankSelector(k, "stability", random_state=s))))
    for k in (10, 20, 50):
        J.append((f"ANOVA_k{k}", "fc_edges", lambda s, k=k: pipe(selector=SelectKBest(f_classif, k=k))))
        J.append((f"mRMR_k{k}", "fc_edges", lambda s, k=k: pipe(
            selector=RankSelector(k, "mrmr", prefilter=1000))))
    return J


def jobs_step7():
    J = []
    for view in ("dc", "regional", "fc_blocks", "all4"):
        for k in (6, 8, 10):
            for solver in ("exact", "sa", "qaoa", "random"):
                J.append((f"QUBO-{solver}_k{k}", view, lambda s, k=k, solver=solver: pipe(
                    selector=QUBOSelector(k, 20, solver, random_state=s))))
            J.append((f"mRMR-greedy-pool20_k{k}", view, lambda s, k=k: pipe(
                selector=RankSelector(k, "mrmr", prefilter=20))))
    return J


def jobs_step6c():
    """Methods from the status-file list not covered by 6a/6b."""
    from sklearn.manifold import Isomap
    J = []
    for view in ("dc", "regional", "fc_blocks", "all4"):
        for k in (5, 10, 20):
            J.append((f"Isomap_k{k}", view, lambda s, k=k: pipe(Isomap(n_neighbors=10, n_components=k))))
            J.append((f"NMF_k{k}", view, lambda s, k=k: pipe(ShiftedNMF(k, random_state=s))))
        J.append(("Boruta", view, lambda s: pipe(selector=BorutaSelector(random_state=s))))
        J.append(("GA", view, lambda s: pipe(selector=MetaheuristicSelector("ga", random_state=s))))
        J.append(("PSO", view, lambda s: pipe(selector=MetaheuristicSelector("pso", random_state=s))))
    return J


def jobs_ksweep():
    """Does keeping MORE features help? PCA can give at most ~63 components (one
    fewer than the ~64 training subjects); selectors can keep any number."""
    J = []
    for view in ("regional", "all4"):
        J.append(("NoReduction", view, lambda s: pipe()))
        for k in (2, 5, 10, 15, 20, 30, 40, 50, 60):
            J.append((f"PCA_k{k}", view, lambda s, k=k: pipe(PCA(k, random_state=s))))
        for k in (5, 10, 20, 50, 100, 200, 400, 700):
            J.append((f"ANOVA_k{k}", view, lambda s, k=k: pipe(selector=SelectKBest(f_classif, k=k))))
            J.append((f"L1_k{k}", view, lambda s, k=k: pipe(selector=SelectFromModel(
                LogisticRegression(penalty="l1", C=0.5, solver="saga", max_iter=3000),
                max_features=k, threshold=-np.inf))))
    return J


def jobs_step6b_nostab():
    """Step 6b without stability selection, which takes about 1 h per
    configuration on the 984-feature all4 view (50 L1 fits per fold)."""
    return [j for j in jobs_step6b() if not j[0].startswith("Stability_")]


def jobs_step6b_stability_fix():
    """Re-run of stability selection only: the first step6b run used C=0.1, at which
    the L1 model kept no feature, so ties fell back to column order (bug)."""
    return [j for j in jobs_step6b() if j[0].startswith("Stability_")]


STEPS = {"step5": ("PCA, LDA, LRA (normal methods)", jobs_step5),
         "step6b_stability_fix": ("stability selection re-run (bug fix)", jobs_step6b_stability_fix),
         "step6c": ("Isomap, NMF, Boruta, genetic algorithm, particle swarm", jobs_step6c),
         "step6b_nostab": ("step 6b without stability selection", jobs_step6b_nostab),
         "ksweep":("how many features to keep: 2 to all", jobs_ksweep),
         "step6a": ("Sparse PCA, PLS, Kernel PCA, ICA (advanced extraction)", jobs_step6a),
         "step6b": ("ANOVA, MI, mRMR, ReliefF, L1, SVM-RFE, RF, stability selection", jobs_step6b),
         "step7": ("QUBO feature selection: exact, simulated annealing, QAOA, random", jobs_step7)}


def main():
    step = sys.argv[1] if len(sys.argv) > 1 else "step5"
    only_view = sys.argv[2] if len(sys.argv) > 2 else None
    only_k = sys.argv[3] if len(sys.argv) > 3 else None      # e.g. "k8", to run in parallel
    tag = step + (f"_{only_view}" if only_view else "") + (f"_{only_k}" if only_k else "")
    title, mk_jobs = STEPS[step]
    log("=" * 96)
    log(f"FEATURE-SELECTION STUDY -- {step}: {title}")
    log("=" * 96)
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "scikit-learn": sklearn.__version__}
    if step == "step7":
        import dimod
        import neal
        import qiskit
        import qiskit_aer
        versions.update({"qiskit": qiskit.__version__, "qiskit-aer": qiskit_aer.__version__,
                         "dimod": dimod.__version__, "dwave-neal": neal.__version__})
    log("versions: " + json.dumps(versions))

    data = H.PoolData()
    checks = data.verify()
    plan = H.make_fold_plan(data)
    checks["fold fingerprints match"] = all(plan[s][0] == H.EXPECTED_FINGERPRINTS[s] for s in H.SEEDS)
    for k_, v in checks.items():
        log(f"  {'PASS' if v else 'FAIL'}  {k_}")
    if not all(checks.values()):
        raise SystemExit("data / fold checks failed")
    data.views["fc_blocks"] = fc_blocks(data.views["fc_edges"])
    # all four biomarkers together: mALFF + mReHo + DC_z (738) + per-region mean |FC| (246)
    data.views["all4"] = np.hstack([data.views["regional"], data.views["fc_summary"]])
    log("  views: " + ", ".join(f"{k_}={v.shape[1]}" for k_, v in data.views.items()))
    log("  classifier for every method: LogisticRegression(C=0.05, class_weight=balanced)")
    log(f"  chance = {H.CHANCE:.3f}; clean no-reduction baselines: logreg DC 0.168, "
        f"logreg regional 0.176, MLP DC 0.181")

    jobs = [j for j in mk_jobs() if (only_view is None or j[1] == only_view)
            and (only_k is None or j[0].endswith("_" + only_k))]
    log(f"\n  {'method':26s} {'view':10s} {'feat in->out':>14s} {'bal.acc':>14s} {'macroF1':>8s} "
        f"{'MAE':>6s} {'QWK':>7s} {'stab':>5s} {'cls':>5s}  time")
    all_recs, summary = [], {}
    for name, view, mk in jobs:
        X = data.views[view]
        recs, el = run_cv2(name, X, data.y, plan, mk)
        s = H.summarise(recs)
        key = f"{name}|{view}"
        st = stability(recs)
        n_out = int(np.median([r["n_features_out"] for r in recs]))
        summary[key] = {"method": name, "view": view, "n_features_in": int(X.shape[1]),
                        "n_features_out": n_out, "stability_jaccard": st, "time_s": el, **s}
        if any("selected" in r for r in recs):
            summary[key]["top_features"] = top_features(recs, feature_names(view))
        if any("solver_info" in r for r in recs):
            inf = [r["solver_info"] for r in recs]
            summary[key]["solver"] = {
                "hit_optimum_frac": float(np.mean([i["hit_optimum"] for i in inf])),
                "mean_cost_gap": float(np.mean([i["cost"] - i["cost_opt"] for i in inf])),
                "mean_solve_s": float(np.mean([i["solve_s"] for i in inf])),
                **({"mean_feasible_shot_frac": float(np.mean([i["feasible_shot_frac"] for i in inf])),
                    "fallbacks": int(sum(i.get("fallback_marginals", False) for i in inf))}
                   if "feasible_shot_frac" in inf[0] else {})}
        all_recs += recs
        b = s["balanced_accuracy"]
        log(f"  {name:26s} {view:10s} {X.shape[1]:>6d}->{n_out:<6d} {b[0]:.3f}+/-{b[1]:.3f} "
            f"{s['macro_f1'][0]:8.3f} {s['mae_stages'][0]:6.3f} {s['qwk'][0]:+7.3f} "
            f"{'' if st is None else f'{st:.2f}':>5s} {s['classes_min']}-{s['classes_max']}  {el:.0f}s")
        if "solver" in summary[key]:
            log(f"      solver: {json.dumps(summary[key]['solver'])}")

    best = max(summary, key=lambda q: summary[q]["balanced_accuracy"][0])
    log(f"\n  best in {step}: {best}  bal.acc {summary[best]['balanced_accuracy'][0]:.3f}"
        f" +/- {summary[best]['balanced_accuracy'][1]:.3f}")
    H.save_outputs(tag, all_recs, {"versions": versions, "summary": summary,
                                   "seeds": H.SEEDS, "n_folds": H.N_FOLDS,
                                   "test_set_touched": False})


if __name__ == "__main__":
    main()
