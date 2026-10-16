"""
Baselines, on the identical splits used by the transformer.

A transformer result is uninterpretable on its own. With 81 training subjects
and six classes, the questions that matter are "is this better than always
guessing the majority class?" and "is this better than a linear model on the
same features?". Those have to be answered on the SAME partition, or the
comparison is meaningless.

Baselines
---------
  majority          always predict the most frequent training class
  stratified_prior  sample from the training class distribution (seeded)
  logreg_roi        multinomial logistic regression on the 738-D regional
                    features (246 ROIs x 3 biomarkers), fold-fitted scaler
  logreg_summary    multinomial logistic regression on 4 whole-brain summary
                    numbers: mean mALFF, mean mReHo, mean DC_z, mean FC
                    strength. The smallest sensible model.
  ordinal_logreg    cumulative-link ordinal regression on the same 4 summaries

Every baseline uses the same 70/30 split, the same 5 folds, the same
train-fold-only scaling and the same metrics as train_transformer.py.
"""
import csv
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit, softmax

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_transformer import (STAGES, N_STAGES, Scaler, evaluate,  # noqa: E402
                               load_all)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "derivatives", "model_training")
SEED = 20261015

_log = []


def log(m=""):
    print(m, flush=True)
    _log.append(str(m))


def summaries(reg, fcu):
    """Four whole-brain numbers per subject."""
    return np.column_stack([reg[:, :, 0].mean(1), reg[:, :, 1].mean(1),
                            reg[:, :, 2].mean(1), fcu.mean(1)])


def fit_multinomial(X, y, l2=1.0, iters=400):
    n, d = X.shape
    X1 = np.hstack([X, np.ones((n, 1))])
    w0 = np.zeros((d + 1) * N_STAGES)
    Y = np.eye(N_STAGES)[y]
    cw = len(y) / (N_STAGES * np.maximum(np.bincount(y, minlength=N_STAGES), 1))
    sw = cw[y]

    def obj(w):
        W = w.reshape(d + 1, N_STAGES)
        Z = X1 @ W
        P = softmax(Z, axis=1)
        ll = -(sw * (Y * np.log(np.clip(P, 1e-12, None))).sum(1)).sum() / sw.sum()
        reg_term = l2 * (W[:-1] ** 2).sum() / (2 * n)
        G = X1.T @ ((P - Y) * sw[:, None]) / sw.sum()
        G[:-1] += l2 * W[:-1] / n
        return ll + reg_term, G.ravel()

    r = minimize(obj, w0, jac=True, method="L-BFGS-B",
                 options={"maxiter": iters})
    return r.x.reshape(d + 1, N_STAGES)


def predict_multinomial(W, X):
    return softmax(np.hstack([X, np.ones((X.shape[0], 1))]) @ W, axis=1)


def fit_ordinal(X, y, l2=1.0, iters=400):
    """Cumulative link: P(y<=k) = sigmoid(tau_k - x.b), tau increasing."""
    n, d = X.shape
    w0 = np.concatenate([np.zeros(d), [0.0], np.zeros(N_STAGES - 2)])
    cw = len(y) / (N_STAGES * np.maximum(np.bincount(y, minlength=N_STAGES), 1))
    sw = cw[y]
    K = np.arange(N_STAGES - 1)
    T = (y[:, None] <= K[None, :]).astype(float)

    def taus(w):
        return w[d] + np.concatenate([[0.0], np.cumsum(np.exp(w[d + 1:]))])

    def obj(w):
        s = X @ w[:d]
        z = taus(w)[None, :] - s[:, None]
        p = expit(z)
        ll = -(sw[:, None] * (T * np.log(np.clip(p, 1e-12, None))
                              + (1 - T) * np.log(np.clip(1 - p, 1e-12, None)))
               ).sum() / sw.sum()
        return ll + l2 * (w[:d] ** 2).sum() / (2 * n)

    r = minimize(obj, w0, method="L-BFGS-B", options={"maxiter": iters})
    return r.x, d


def predict_ordinal(w, d, X):
    tau = w[d] + np.concatenate([[0.0], np.cumsum(np.exp(w[d + 1:]))])
    z = tau[None, :] - (X @ w[:d])[:, None]
    cum = expit(z)
    cum = np.hstack([np.zeros((X.shape[0], 1)), cum, np.ones((X.shape[0], 1))])
    return np.clip(cum[:, 1:] - cum[:, :-1], 0, None)


def main():
    os.makedirs(OUT, exist_ok=True)
    reg, fc, y, subjects, mask, pool, test, fold, smeta = load_all()
    fcu = np.load(os.path.join(ROOT, "derivatives", "model_ready",
                               "X_fc_roi_unsigned.npy")).astype(np.float32)
    n_folds = smeta["n_folds"]
    log("=" * 74)
    log("BASELINES -- identical splits, folds and metrics as the transformer")
    log("=" * 74)
    log(f"split fingerprint {smeta['split_fingerprint']}  "
        f"pool {pool.size}  test {test.size}")

    rng = np.random.default_rng(SEED)
    results = {}

    def cv(name, fn):
        per = []
        for f in range(n_folds):
            va = np.array([i for i in pool if fold[i] == f])
            tr = np.array([i for i in pool if fold[i] != f])
            p = fn(tr, va)
            per.append(evaluate(p, y[va]))
        summ = {k: {"mean": float(np.mean([m[k] for m in per])),
                    "sd": float(np.std([m[k] for m in per], ddof=1))}
                for k in ("accuracy", "balanced_accuracy", "mae_stages",
                          "qwk", "ece")}
        results[name] = {"cv": {"folds": per, "summary": summ}}
        log(f"  {name:18s} bal.acc {summ['balanced_accuracy']['mean']:.3f} "
            f"+/- {summ['balanced_accuracy']['sd']:.3f}   "
            f"acc {summ['accuracy']['mean']:.3f}   "
            f"MAE {summ['mae_stages']['mean']:.2f}   "
            f"QWK {summ['qwk']['mean']:+.3f}")
        return fn

    log("")
    log(f"  {n_folds}-fold CV inside the training pool:")

    def f_majority(tr, va):
        c = np.bincount(y[tr], minlength=N_STAGES)
        p = np.zeros((va.size, N_STAGES))
        p[:, int(c.argmax())] = 1.0
        return p
    cv("majority", f_majority)

    def f_prior(tr, va):
        c = np.bincount(y[tr], minlength=N_STAGES).astype(float)
        return np.tile(c / c.sum(), (va.size, 1))
    cv("stratified_prior", f_prior)

    def f_logreg_roi(tr, va):
        """3 regional biomarkers x 246 ROIs = 738 features. FC is NOT included,
        so this is deliberately not the full-evidence baseline."""
        sc = Scaler().fit(reg, tr)
        Xtr = sc.transform(reg[tr]).reshape(len(tr), -1)
        Xva = sc.transform(reg[va]).reshape(len(va), -1)
        W = fit_multinomial(Xtr, y[tr], l2=10.0)
        return predict_multinomial(W, Xva)
    cv("logreg_roi3", f_logreg_roi)

    def f_logreg_roi4(tr, va):
        """All FOUR biomarkers at ROI level: mALFF, mReHo, DC_z and the
        unsigned FC strength, 246 x 4 = 984 features. This is the fair
        comparison for the transformer, which also sees all four."""
        R4 = np.concatenate([reg, fcu[:, :, None]], axis=2)
        sc = Scaler().fit(R4, tr)
        Xtr = sc.transform(R4[tr]).reshape(len(tr), -1)
        Xva = sc.transform(R4[va]).reshape(len(va), -1)
        W = fit_multinomial(Xtr, y[tr], l2=10.0)
        return predict_multinomial(W, Xva)
    cv("logreg_roi4", f_logreg_roi4)

    iu = np.triu_indices(246, k=1)

    def f_logreg_fc_edges(tr, va):
        """The FC matrix itself: 30,135 unique edges. Far more features than
        the 81 training subjects, so it is heavily regularised and is included
        to show what the raw connectivity pattern alone supports."""
        E = fc[:, iu[0], iu[1]]
        mu, sd = E[tr].mean(0), E[tr].std(0) + 1e-8
        W = fit_multinomial((E[tr] - mu) / sd, y[tr], l2=1000.0, iters=200)
        return predict_multinomial(W, (E[va] - mu) / sd)
    cv("logreg_fc_edges", f_logreg_fc_edges)

    S = summaries(reg, fcu)

    def f_logreg_summary(tr, va):
        mu, sd = S[tr].mean(0), S[tr].std(0) + 1e-8
        W = fit_multinomial((S[tr] - mu) / sd, y[tr], l2=1.0)
        return predict_multinomial(W, (S[va] - mu) / sd)
    cv("logreg_summary", f_logreg_summary)

    def f_ordinal(tr, va):
        mu, sd = S[tr].mean(0), S[tr].std(0) + 1e-8
        w, d = fit_ordinal((S[tr] - mu) / sd, y[tr], l2=1.0)
        return predict_ordinal(w, d, (S[va] - mu) / sd)
    cv("ordinal_logreg", f_ordinal)

    log("")
    log("  held-out test (fitted on the whole training pool, evaluated once):")
    for name, fn in (("majority", f_majority), ("stratified_prior", f_prior),
                     ("logreg_roi3", f_logreg_roi),
                     ("logreg_roi4", f_logreg_roi4),
                     ("logreg_fc_edges", f_logreg_fc_edges),
                     ("logreg_summary", f_logreg_summary),
                     ("ordinal_logreg", f_ordinal)):
        p = fn(pool, test)
        m = evaluate(p, y[test])
        results[name]["test"] = m
        log(f"  {name:18s} bal.acc {m['balanced_accuracy']:.3f}   "
            f"acc {m['accuracy']:.3f}   MAE {m['mae_stages']:.2f}   "
            f"QWK {m['qwk']:+.3f}   ECE {m['ece']:.3f}")

    log("")
    log(f"  chance balanced accuracy = 1/6 = {1 / 6:.3f}")

    p = os.path.join(OUT, "results_baselines.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"split_fingerprint": smeta["split_fingerprint"],
                   "baselines": results}, f, indent=2)
    log(f"\nwrote {os.path.relpath(p, ROOT)}")
    with open(os.path.join(OUT, "train_baselines.log"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")


if __name__ == "__main__":
    main()
