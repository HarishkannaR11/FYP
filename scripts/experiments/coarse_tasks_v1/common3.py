"""
Shared helpers for the pre-declared three-group task (see docs/NEW_EXPERIMENT_PREDECLARATION.md).
numpy only: the outer folds are read from the stored TabPFN six-class run, so they are IDENTICAL to every other
experiment of this project (same test indices and fingerprints), and no model code is needed to build them.
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MR = os.path.join(ROOT, "derivatives", "model_ready")
FOLD_SOURCE = os.path.join(ROOT, "derivatives", "model_training", "tabpfn", "cv116_tabpfn_svmrfe50.json")
OUT = os.path.join(ROOT, "derivatives", "experiments", "coarse_tasks_v1")
SEEDS = [20261015, 20261016, 20261017, 20261018, 20261019]
GROUP_MAP = {0: 0, 1: 0, 2: 1, 3: 1, 4: 1, 5: 2}            # CN,SMC | EMCI,MCI,LMCI | AD   (declared)
GROUP_NAMES = ["CN+SMC", "EMCI+MCI+LMCI", "AD"]
NG = 3
B = 2000


def load_cohort():
    y6 = np.load(os.path.join(MR, "y.npy"))
    subjects = np.load(os.path.join(MR, "subjects.npy"), allow_pickle=True)
    y3 = np.array([GROUP_MAP[int(v)] for v in y6])
    site = np.array([str(s).replace("sub-", "")[:3] for s in subjects])
    assert np.bincount(y3).tolist() == [40, 56, 20]
    return y6, y3, subjects, site


def load_folds():
    """{(seed, fold): (train_idx, test_idx, fingerprint)} from the stored six-class run."""
    recs = json.load(open(FOLD_SOURCE))["records"]
    n = 116
    out = {}
    for r in recs:
        te = np.array(r["test_index"]); tr = np.setdiff1d(np.arange(n), te)
        out[(r["seed"], r["fold"])] = (tr, te, r["fold_fingerprint"])
    assert len(out) == 25
    return out


def bal(yt, yp, k=NG):
    return float(np.mean([np.mean(yp[yt == c] == c) for c in range(k) if (yt == c).any()]))


def macro_f1(yt, yp, k=NG):
    f = []
    for c in range(k):
        tp = np.sum((yt == c) & (yp == c)); fp = np.sum((yt != c) & (yp == c)); fn = np.sum((yt == c) & (yp != c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def pooled(rec, seed, n=116):
    p = np.full(n, -1)
    for f in range(5):
        idx, pr = rec[(seed, f)]
        p[idx] = pr
    assert (p >= 0).all()
    return p


def boot_ci(A, y, seed=0):
    """Stratified subject bootstrap of the seed-averaged balanced accuracy (same method as report_metrics.py)."""
    rng = np.random.default_rng(seed)
    C = (A == y[None, :]).astype(float); rec = []
    for g in range(NG):
        ids = np.flatnonzero(y == g)
        smp = rng.choice(ids, size=(B, len(ids)), replace=True)
        rec.append(C[:, smp].mean(2))
    bb = np.mean(rec, 0).mean(0)
    return float(np.percentile(bb, 2.5)), float(np.percentile(bb, 97.5))


def paired(A, Bm, y, seed=1):
    """Paired bootstrap of (balanced accuracy of A minus of B) on the same resampled subjects."""
    rng = np.random.default_rng(seed)
    CA = (A == y[None, :]).astype(float); CB = (Bm == y[None, :]).astype(float); rec = []
    for g in range(NG):
        ids = np.flatnonzero(y == g)
        smp = rng.choice(ids, size=(B, len(ids)), replace=True)
        rec.append(CA[:, smp].mean(2) - CB[:, smp].mean(2))
    d = np.mean(rec, 0).mean(0)
    ba = [bal(y, a) for a in A]; bb = [bal(y, b) for b in Bm]
    return dict(diff=float(np.mean(ba) - np.mean(bb)), lo=float(np.percentile(d, 2.5)), hi=float(np.percentile(d, 97.5)),
                seeds_first_better=int(sum(a > b for a, b in zip(ba, bb))), n_seeds=len(ba))
