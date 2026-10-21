"""
Shared code for the leakage-safe experiment set v1.

Everything that learns from data (ComBat, SVM-RFE, imputation, scaling,
hyperparameters, the classifier) is fitted inside a function that receives
the TRAINING indices only. The test rows of a split are passed only to
`transform` / `predict` calls. `audit.py` checks this programmatically by
perturbing test rows and confirming that every fitted quantity is unchanged.

Outputs go to derivatives/experiments/leakage_safe_v1/ and are never
overwritten: per-split result files are written once and skipped if present,
so an interrupted run resumes where it stopped.
"""
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "model"))
import train_transformer_comorbidity as C  # noqa: E402  (sets thread env vars first)

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import scipy  # noqa: E402
import sklearn  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from scipy.special import expit  # noqa: E402
from sklearn.feature_selection import RFE  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC, LinearSVC  # noqa: E402

S, H, M = C.S, C.H, C.M
ROOT = C.ROOT
EXP = os.path.join(ROOT, "derivatives", "experiments", "leakage_safe_v1")
STAGES = list(H.STAGES)
NS = len(STAGES)
SEEDS = list(H.SEEDS)

# ----------------------------------------------------------- the design
K_GRID = [25, 50, 100, 150, 246, 984]
TABPFN_K = [25, 50, 100, 150, 246]          # TabPFN v2 refuses > 500 features
TABPFN_NESTED_K = [25, 50, 100]             # inner-loop TabPFN grid (compute limit, fixed a priori)
HARMS = ["none", "combat"]
INPUTS = {"img": None, "imgcom": "comorbidity8"}
BIOMARKERS = {"mALFF": 0, "mReHo": 1, "DC_z": 2, "FC": 3}
N_INNER = 3
LR_C = [0.01, 0.1, 1.0]
ORD_C = [0.01, 0.1]
SVM_GRID = [(1.0, 0.25), (1.0, 1.0), (10.0, 0.25), (10.0, 1.0)]   # (C, g): gamma = g / n_features
LIGHT = ["lr", "svm", "ordinal"]


def hp_list(family):
    if family == "lr":
        return [f"C={c}" for c in LR_C]
    if family == "ordinal":
        return [f"C={c}" for c in ORD_C]
    if family == "svm":
        return [f"C={c},g={g}" for c, g in SVM_GRID]
    return ["-"]


def cid(family, inputs, view, harm, hp):
    """Config id: family|inputs|view|harm|hp, e.g. tabpfn|imgcom|k50|none|-"""
    return f"{family}|{inputs}|{view}|{harm}|{hp}"


# ------------------------------------------------------------------ data
def load():
    X4, flat, fc, y, subjects, mask = S.load_cohort()
    clin = C.load_clinical(y, subjects, C.FEATURE_SETS["comorbidity8"])[0]
    demo = pd.read_csv(os.path.join(ROOT, "derivatives", "comorbidity_features", "demographics_116.csv"))
    if demo["bids_id"].tolist() != list(subjects):
        raise SystemExit("demographics not in cohort order")
    site = np.array([s.replace("sub-", "")[:3] for s in subjects])
    covars = np.column_stack([demo["age_at_scan"].to_numpy(float), (demo["sex"] == "M").to_numpy(float)])
    return {"flat": flat.astype(np.float64), "y": y.astype(int), "subjects": subjects, "clin": clin,
            "site": site, "covars": covars}


def biomarker_columns(b):
    """Flat all4 columns of one biomarker: 738 regional (roi-major, 3 channels) then 246 FC."""
    return np.arange(246) * 3 + b if b < 3 else 738 + np.arange(246)


def outer_plans(n_folds=5):
    d = load()
    return S.cv_plan(d["y"], d["subjects"], n_folds)


def inner_folds(labels, tr, seed, fold):
    fo = H.stratified_folds(labels, tr, N_INNER, np.random.default_rng([seed, fold, 33]))
    return [(np.array([i for i in tr if fo[int(i)] != j]), np.array([i for i in tr if fo[int(i)] == j]))
            for j in range(N_INNER)]


# ---------------------------------------------------------------- ComBat
class ComBat:
    """ComBat harmonisation (Johnson et al. 2007, parametric empirical Bayes),
    written to the neuroCombat formulation, with a fit / transform split so
    it can be fitted on training subjects only.

    batch      scanner site of each subject
    covars     biological covariates to PRESERVE: age, sex. Diagnosis is NOT
               a covariate: test subjects' diagnoses are unknown, and using
               them would leak the label into preprocessing.

    Sites with fewer than `min_n` training subjects cannot be estimated; their
    subjects (training or test) and subjects from sites unseen in training are
    left without a site adjustment and are counted in `report()`.
    """

    def __init__(self, min_n=2, eb=True, conv=1e-4):
        self.min_n, self.eb, self.conv = min_n, eb, conv

    def fit(self, X, batch, covars):
        X = np.asarray(X, float)
        sites, counts = np.unique(batch, return_counts=True)
        self.sites_ = [s for s, c in zip(sites, counts) if c >= self.min_n]
        keep = np.isin(batch, self.sites_)
        Xk, bk, ck = X[keep], np.asarray(batch)[keep], np.asarray(covars, float)[keep]
        n, nb = Xk.shape[0], len(self.sites_)
        B = np.column_stack([(bk == s).astype(float) for s in self.sites_])
        D = np.column_stack([B, ck])
        beta = np.linalg.lstsq(D, Xk, rcond=None)[0]                     # [nb + c, p]
        n_b = B.sum(0)
        self.grand_mean_ = (n_b / n) @ beta[:nb]
        self.beta_cov_ = beta[nb:]
        resid = Xk - D @ beta
        self.var_pooled_ = np.maximum((resid ** 2).mean(0), 1e-12)
        Z = (Xk - self.grand_mean_ - ck @ self.beta_cov_) / np.sqrt(self.var_pooled_)
        g_hat = np.vstack([Z[bk == s].mean(0) for s in self.sites_])
        d_hat = np.vstack([Z[bk == s].var(0, ddof=1) for s in self.sites_])
        d_hat = np.maximum(d_hat, 1e-8)
        if not self.eb:
            self.gamma_, self.delta_ = g_hat, d_hat
            return self
        g_bar, t2 = g_hat.mean(1), g_hat.var(1, ddof=1)
        m, s2 = d_hat.mean(1), d_hat.var(1, ddof=1)
        a = (2 * s2 + m ** 2) / s2
        b = (m * s2 + m ** 3) / s2
        gs, ds = [], []
        for i, s in enumerate(self.sites_):
            Zb = Z[bk == s]
            nn = Zb.shape[0]
            g_old, d_old = g_hat[i].copy(), d_hat[i].copy()
            for _ in range(1000):
                g_new = (t2[i] * nn * g_hat[i] + d_old * g_bar[i]) / (t2[i] * nn + d_old)
                d_new = (0.5 * ((Zb - g_new) ** 2).sum(0) + b[i]) / (nn / 2.0 + a[i] - 1.0)
                change = max(np.max(np.abs(g_new - g_old) / np.maximum(np.abs(g_old), 1e-12)),
                             np.max(np.abs(d_new - d_old) / np.maximum(np.abs(d_old), 1e-12)))
                g_old, d_old = g_new, d_new
                if change < self.conv:
                    break
            gs.append(g_old)
            ds.append(np.maximum(d_old, 1e-8))
        self.gamma_, self.delta_ = np.vstack(gs), np.vstack(ds)
        return self

    def transform(self, X, batch, covars):
        X = np.asarray(X, float)
        stand = self.grand_mean_ + np.asarray(covars, float) @ self.beta_cov_
        sd = np.sqrt(self.var_pooled_)
        Z = (X - stand) / sd
        out = Z.copy()
        batch = np.asarray(batch)
        for i, s in enumerate(self.sites_):
            r = batch == s
            out[r] = (Z[r] - self.gamma_[i]) / np.sqrt(self.delta_[i])
        self.last_unadjusted_ = int((~np.isin(batch, self.sites_)).sum())
        return out * sd + stand

    def state(self):
        return np.concatenate([self.grand_mean_, self.var_pooled_, self.beta_cov_.ravel(),
                               self.gamma_.ravel(), self.delta_.ravel()])


# ------------------------------------------------ ordinal logistic regression
class OrdinalLogistic:
    """Proportional-odds (cumulative logit) model with an L2 penalty:
        P(y <= k | x) = sigmoid(theta_k - w.x),   theta_0 < ... < theta_{K-2}
    Thresholds are kept ordered by construction (theta_k = a_0 + sum exp(a_j)).
    Loss = C * sum_i v_i NLL_i + 0.5 ||w||^2, v_i = balanced class weights.
    Analytic gradient, L-BFGS. Prediction = argmax of the class probabilities."""

    def __init__(self, C=0.1, n_classes=NS, max_iter=2000):
        self.C, self.K, self.max_iter = C, n_classes, max_iter

    def _theta(self, a):
        return a[0] + np.concatenate([[0.0], np.cumsum(np.exp(a[1:]))])

    def _cdf(self, eta, theta):
        F = expit(theta[None, :] - eta[:, None])                         # [n, K-1]
        return np.column_stack([np.zeros(len(eta)), F, np.ones(len(eta))])

    def _objective(self, params, X, y, v):
        n, p = X.shape
        K = self.K
        w, a = params[:p], params[p:]
        theta = self._theta(a)
        F = self._cdf(X @ w, theta)
        hi, lo = F[np.arange(n), y + 1], F[np.arange(n), y]
        P = np.maximum(hi - lo, 1e-12)
        f_hi, f_lo = hi * (1 - hi), lo * (1 - lo)
        loss = self.C * np.sum(v * -np.log(P)) + 0.5 * w @ w
        d_eta = self.C * v * (f_hi - f_lo) / P
        gw = X.T @ d_eta + w
        gtheta = np.zeros(K - 1)
        m_hi, m_lo = y <= K - 2, y >= 1
        np.add.at(gtheta, y[m_hi], self.C * v[m_hi] * -f_hi[m_hi] / P[m_hi])
        np.add.at(gtheta, y[m_lo] - 1, self.C * v[m_lo] * f_lo[m_lo] / P[m_lo])
        ga = np.zeros(K - 1)
        ga[0] = gtheta.sum()
        ea = np.exp(a[1:])
        for j in range(1, K - 1):
            ga[j] = ea[j - 1] * gtheta[j:].sum()
        return loss, np.concatenate([gw, ga])

    def fit(self, X, y):
        X = np.asarray(X, float)
        y = np.asarray(y, int)
        n, p = X.shape
        cnt = np.bincount(y, minlength=self.K).astype(float)
        v = np.where(cnt[y] > 0, n / (self.K * np.maximum(cnt[y], 1)), 0.0)
        a0 = np.concatenate([[-1.0], np.zeros(self.K - 2)])
        res = minimize(self._objective, np.concatenate([np.zeros(p), a0]), args=(X, y, v), jac=True,
                       method="L-BFGS-B", options={"maxiter": self.max_iter})
        self.w_, self.a_ = res.x[:p], res.x[p:]
        self.classes_ = np.arange(self.K)
        return self

    def predict_proba(self, X):
        F = self._cdf(np.asarray(X, float) @ self.w_, self._theta(self.a_))
        return np.clip(np.diff(F, axis=1), 0, 1)

    def predict(self, X):
        return self.predict_proba(X).argmax(1)


# ------------------------------------------------------------ pipelines
def rfe_support(X, y, tr, k, rs):
    """The project's SVM-RFE (scale -> RFE(LinearSVC C=0.01, step 0.1)), fitted on tr only.
    k = 50 reproduces train_transformer_svmrfe.fit_selection exactly."""
    if k >= X.shape[1]:
        return np.arange(X.shape[1])
    pipe = M.pipe(selector=RFE(LinearSVC(C=0.01, max_iter=5000, random_state=rs),
                               n_features_to_select=k, step=0.1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe.fit(X[tr], y[tr])
    return np.flatnonzero(pipe.named_steps["select"].get_support())


def harmonise(data, harm, tr):
    """Imaging matrix for one training set: ComBat fitted on tr only (batch = site,
    covariates = age, sex), applied to every row with the fitted parameters."""
    X = data["flat"]
    if harm == "none":
        return X, None
    cb = ComBat().fit(X[tr], data["site"][tr], data["covars"][tr])
    Xh = cb.transform(X, data["site"], data["covars"])
    return Xh, cb


def design(Ximg, data, inputs, cols):
    parts = [Ximg[:, cols]]
    if INPUTS[inputs]:
        parts.append(data["clin"])
    return np.hstack(parts)


def _full_probs(p, classes):
    out = np.zeros((p.shape[0], NS))
    out[:, np.asarray(classes, int)] = p
    return out


def fit_predict(family, hp, Xtr, ytr, Xte, rs, train_score=False):
    """Fit on training rows only, predict test rows. Returns (pred, probs|None, train_bal|None)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if family == "tabpfn":
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion
            m = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu", random_state=rs)
            m.fit(Xtr.astype(np.float32), ytr)
            p = _full_probs(m.predict_proba(Xte.astype(np.float32)), m.classes_)
            return p.argmax(1), p, None
        steps = [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
        kv = dict(x.split("=") for x in hp.split(","))
        if family == "lr":
            clf = LogisticRegression(C=float(kv["C"]), class_weight="balanced", max_iter=5000)
        elif family == "ordinal":
            clf = OrdinalLogistic(C=float(kv["C"]))
        elif family == "svm":
            clf = SVC(C=float(kv["C"]), gamma=float(kv["g"]) / Xtr.shape[1], class_weight="balanced")
        else:
            raise ValueError(family)
        pipe = Pipeline(steps + [("clf", clf)]).fit(Xtr, ytr)
        if family == "svm":
            pred, p = pipe.predict(Xte).astype(int), None
        else:
            p = _full_probs(pipe.predict_proba(Xte), pipe.classes_)
            pred = p.argmax(1)
        tb = None
        if train_score:
            ptr = pipe.predict(Xtr) if family == "svm" else pipe.predict_proba(Xtr).argmax(1)
            tb = H.evaluate(ytr, np.asarray(pipe.classes_)[ptr] if family != "svm" else ptr)["balanced_accuracy"]
        return pred, p, tb


def bal_acc(y, pred):
    return H.evaluate(np.asarray(y), np.asarray(pred))["balanced_accuracy"]


# ------------------------------------------------------------ selection + uncertainty
FAMILY_RANK = {"lr": 0, "ordinal": 1, "svm": 2, "tabpfn": 3}     # ties go to the simpler model


def parse_cid(c):
    fam, inputs, view, harm, hp = c.split("|")
    return fam, inputs, view, harm, hp


def select_best(scores):
    """Config with the highest mean inner balanced accuracy. Ties (to 1e-9) go to the simpler
    family, then fewer features, then no harmonisation, then the earlier hyperparameter."""
    def key(c):
        fam, _, view, harm, hp = parse_cid(c)
        k = int(view[1:]) if view.startswith("k") else 0
        return (-round(scores[c], 9), FAMILY_RANK.get(fam, 9), k, harm != "none",
                hp_list(fam).index(hp) if hp in hp_list(fam) else 0)
    return min(scores, key=key)


def bootstrap_ci(y, preds_by_seed, n_boot=2000, seed=0, metric="balanced_accuracy"):
    """95% percentile CI of a metric averaged over seeds, resampling SUBJECTS (stratified by stage).
    preds_by_seed: list of length-116 prediction arrays aligned with y (one per seed)."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    by = [np.flatnonzero(y == g) for g in range(NS)]
    P = np.asarray(preds_by_seed)
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(b, len(b), replace=True) for b in by if len(b)])
        if metric == "balanced_accuracy":
            vals.append(np.mean([np.mean([np.mean(p[idx][y[idx] == g] == g) for g in range(NS) if (y[idx] == g).any()])
                                 for p in P]))
        else:
            vals.append(np.mean([np.mean(p[idx] == y[idx]) for p in P]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bootstrap_diff_ci(y, preds_a, preds_b, n_boot=2000, seed=0):
    """95% CI of (balanced accuracy A - B), both averaged over seeds, paired by subject."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    by = [np.flatnonzero(y == g) for g in range(NS)]
    A, B = np.asarray(preds_a), np.asarray(preds_b)

    def ba(P, idx):
        return np.mean([np.mean([np.mean(p[idx][y[idx] == g] == g) for g in range(NS) if (y[idx] == g).any()])
                        for p in P])
    vals = []
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(b, len(b), replace=True) for b in by if len(b)])
        vals.append(ba(A, idx) - ba(B, idx))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def wilson_ci(k, n, z=1.96):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return float(c - h), float(c + h)


# ------------------------------------------------------------ storage + manifest
def r6(a):
    return None if a is None else np.round(np.asarray(a, float), 6).tolist()


def split_path(job, seed, fold, tag="cv5"):
    return os.path.join(EXP, "grid", job, f"{tag}_seed{seed}_fold{fold}.json")


def write_once(path, obj):
    if os.path.exists(path):
        raise SystemExit(f"refusing to overwrite {path}")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"))
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


INPUT_FILES = ["derivatives/model_ready/X_regional.npy", "derivatives/model_ready/X_fc_roi_unsigned.npy",
               "derivatives/model_ready/X_fc.npy", "derivatives/model_ready/y.npy",
               "derivatives/model_ready/subjects.npy", "derivatives/model_ready/splits/split_assignment.csv",
               "derivatives/comorbidity_features/model_clinical_116.csv",
               "derivatives/comorbidity_features/demographics_116.csv"]


def manifest_entry(command, started, outputs, note=""):
    """Append one run to manifest.json (created on first use with inputs, versions, git commit)."""
    path = os.path.join(EXP, "manifest.json")
    if os.path.exists(path):
        man = json.load(open(path))
    else:
        try:
            commit = subprocess.check_output(["git", "-C", ROOT, "rev-parse", "HEAD"], text=True).strip()
            branch = subprocess.check_output(["git", "-C", ROOT, "branch", "--show-current"], text=True).strip()
        except Exception:  # noqa: BLE001
            commit, branch = "unknown", "unknown"
        import torch
        try:
            import tabpfn
            tv = getattr(tabpfn, "__version__", "9.2.0")
        except Exception:  # noqa: BLE001
            tv = "not installed"
        man = {"experiment": "leakage_safe_v1", "created": time.strftime("%Y-%m-%d %H:%M:%S"),
               "git": {"commit": commit, "branch": branch},
               "versions": {"python": platform.python_version(), "numpy": np.__version__,
                            "scipy": scipy.__version__, "scikit-learn": sklearn.__version__,
                            "torch": torch.__version__, "tabpfn": tv, "pandas": pd.__version__},
               "inputs": {f: sha256(os.path.join(ROOT, f)) for f in INPUT_FILES},
               "design": {"seeds": SEEDS, "outer_folds": 5, "inner_folds": N_INNER, "k_grid": K_GRID,
                          "tabpfn_k": TABPFN_K, "tabpfn_nested_k": TABPFN_NESTED_K, "harms": HARMS,
                          "lr_C": LR_C, "ordinal_C": ORD_C, "svm_grid": SVM_GRID,
                          "comorbidity8": C.FEATURE_SETS["comorbidity8"]},
               "runs": []}
    man["runs"].append({"command": command, "started": started, "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "note": note, "outputs": outputs})
    os.makedirs(EXP, exist_ok=True)
    json.dump(man, open(path, "w"), indent=1)
