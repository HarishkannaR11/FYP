"""
Self-tests for the two components written for this experiment set:
OrdinalLogistic (gradient + behaviour) and ComBat (synthetic site effects).
Prints PASS/FAIL per check and exits non-zero on any failure.

Run:  python scripts/experiments/leakage_safe_v1/selftest.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as K  # noqa: E402

import numpy as np  # noqa: E402
from scipy.optimize import approx_fprime  # noqa: E402

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")


rng = np.random.default_rng(0)

# ---- ordinal: analytic gradient vs finite differences
X = rng.normal(size=(40, 7))
y = rng.integers(0, 6, 40)
v = rng.uniform(0.5, 2, 40)
m = K.OrdinalLogistic(C=0.7)
params = rng.normal(size=7 + 5) * 0.3
g_an = m._objective(params, X, y, v)[1]
g_num = approx_fprime(params, lambda q: m._objective(q, X, y, v)[0], 1e-6)
err = np.max(np.abs(g_an - g_num)) / max(np.max(np.abs(g_num)), 1e-9)
check("ordinal gradient matches finite differences", err < 1e-4, f"relative error {err:.2e}")

# ---- ordinal: recovers an ordered latent variable
n = 600
Xo = rng.normal(size=(n, 5))
latent = Xo @ np.array([1.5, -1.0, 0.5, 0, 0]) + rng.logistic(size=n)
yo = np.digitize(latent, [-3, -1.5, 0, 1.5, 3])
mo = K.OrdinalLogistic(C=1.0).fit(Xo[:400], yo[:400])
acc = (mo.predict(Xo[400:]) == yo[400:]).mean()
theta = mo._theta(mo.a_)
# the noise is heavy: the true generating model itself reaches only ~0.40 here
from scipy.special import expit  # noqa: E402
cuts = np.array([-3, -1.5, 0, 1.5, 3])
Ft = expit(cuts[None, :] - (Xo[400:] @ np.array([1.5, -1.0, 0.5, 0, 0]))[:, None])
Ft = np.column_stack([np.zeros(200), Ft, np.ones(200)])
oracle = (np.diff(Ft, axis=1).argmax(1) == yo[400:]).mean()
check("ordinal accuracy close to the true model's", acc >= 0.8 * oracle, f"{acc:.3f} vs oracle {oracle:.3f}")
r = np.corrcoef(mo.w_, [1.5, -1.0, 0.5, 0, 0])[0, 1]
check("ordinal recovers the true coefficients", r > 0.95, f"r = {r:.3f}")
check("ordinal thresholds strictly increasing", bool(np.all(np.diff(theta) > 0)), str(np.round(theta, 2)))
check("ordinal probabilities sum to 1", np.allclose(mo.predict_proba(Xo[:5]).sum(1), 1))

# ---- ComBat: synthetic site location/scale effects with an age effect to preserve
n_per, p = 30, 50
sites = np.repeat(["A", "B", "C"], n_per)
age = rng.uniform(60, 85, 3 * n_per)
sex = rng.integers(0, 2, 3 * n_per).astype(float)
cov = np.column_stack([age, sex])
base = rng.normal(size=(3 * n_per, p)) + 0.05 * (age[:, None] - 72)
shift = {"A": 0.0, "B": 2.0, "C": -1.5}
scale = {"A": 1.0, "B": 2.5, "C": 0.6}
Xs = np.vstack([base[sites == s] * scale[s] + shift[s] for s in ("A", "B", "C")])
tr = np.concatenate([np.arange(0, 20), np.arange(30, 50), np.arange(60, 80)])
cb = K.ComBat().fit(Xs[tr], sites[tr], cov[tr])
Xh = cb.transform(Xs, sites, cov)


def site_spread(Z):
    means = np.vstack([Z[sites == s].mean(0) for s in ("A", "B", "C")])
    sds = np.vstack([Z[sites == s].std(0) for s in ("A", "B", "C")])
    return np.mean(means.std(0)), np.mean(sds.std(0))


m0, s0 = site_spread(Xs)
m1, s1 = site_spread(Xh)
check("ComBat removes site mean differences (incl. held-out rows)", m1 < 0.25 * m0, f"{m0:.2f} -> {m1:.2f}")
check("ComBat removes site scale differences", s1 < 0.25 * s0, f"{s0:.2f} -> {s1:.2f}")
r_before = np.mean([np.corrcoef(age, base[:, j])[0, 1] for j in range(p)])
r_after = np.mean([np.corrcoef(age, Xh[:, j])[0, 1] for j in range(p)])
check("ComBat preserves the age effect", r_after > 0.5 * r_before, f"mean r(age) {r_before:.2f} -> {r_after:.2f}")
Xu = cb.transform(Xs[:5], np.array(["Z"] * 5), cov[:5])
check("unseen site rows are left unadjusted", np.allclose(Xu, Xs[:5]), "")
cb2 = K.ComBat().fit(np.vstack([Xs[tr], Xs[[90 % len(Xs)]]]), np.concatenate([sites[tr], ["D"]]),
                     np.vstack([cov[tr], cov[[0]]]))
check("a site with one training subject is not estimated", "D" not in cb2.sites_, str(cb2.sites_))

# ---- SVM-RFE reproduces the project's 50-feature selection
d = K.load()
plan = K.S.cv_plan(d["y"], d["subjects"], 5)
tr0, te0 = plan[K.SEEDS[0]][1][0]
mine = K.rfe_support(d["flat"], d["y"], tr0, 50, K.SEEDS[0])
_, ref, _ = K.S.fit_selection(d["flat"], d["y"], tr0, K.SEEDS[0])
check("rfe_support(k=50) equals train_transformer_svmrfe.fit_selection", np.array_equal(mine, ref))

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)
