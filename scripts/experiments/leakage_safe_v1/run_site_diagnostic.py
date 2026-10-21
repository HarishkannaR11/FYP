"""
Does ComBat remove scanner-site information? Predict the SITE (not the stage)
from the 984 imaging features, before and after ComBat.

  * sites with >= 5 subjects (smaller sites cannot be both trained and tested)
  * 5 folds stratified by site x 5 seeds; ComBat is fitted on each training
    fold (batch = site, covariates = age and sex) and applied to the test fold
    with the fitted parameters, as in the stage models
  * classifier: standardise + L2 logistic regression (C = 1, balanced classes)
  * chance = 1 / number of sites

Caveat: ComBat uses each subject's site label to apply its adjustment (site is
known at scan time, unlike diagnosis), so a fall to chance is expected if
harmonisation works. Residual site predictability after ComBat means site
information survives in feature interactions.

Run:  python scripts/experiments/leakage_safe_v1/run_site_diagnostic.py
"""
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as K  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402


def site_folds(site, idx, k, rng):
    fold_of = {}
    for s in np.unique(site[idx]):
        members = rng.permutation(idx[site[idx] == s])
        for j, i in enumerate(members):
            fold_of[int(i)] = j % k
    return [(np.array([i for i in idx if fold_of[int(i)] != f]), np.array([i for i in idx if fold_of[int(i)] == f]))
            for f in range(k)]


def main():
    out_csv = os.path.join(K.EXP, "results", "site_prediction.csv")
    if os.path.exists(out_csv):
        raise SystemExit(f"{out_csv} exists; not overwritten")
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    d = K.load()
    site = d["site"]
    sites, counts = np.unique(site, return_counts=True)
    keep_sites = sites[counts >= 5]
    idx = np.flatnonzero(np.isin(site, keep_sites))
    rows = []
    for seed in K.SEEDS:
        for f, (tr, te) in enumerate(site_folds(site, idx, 5, np.random.default_rng(seed))):
            for harm in K.HARMS:
                X, cb = K.harmonise(d, harm, tr)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    clf = Pipeline([("scale", StandardScaler()),
                                    ("lr", LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000)))])
                    clf.fit(X[tr], site[tr])
                    pred = clf.predict(X[te])
                recall = [np.mean(pred[site[te] == s] == s) for s in np.unique(site[te])]
                rows.append({"seed": seed, "fold": f, "harmonisation": harm, "n_test": len(te),
                             "site_balanced_accuracy": float(np.mean(recall)),
                             "chance": 1 / len(keep_sites)})
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)
    summ = df.groupby(["harmonisation", "seed"])["site_balanced_accuracy"].mean().groupby("harmonisation").agg(["mean", "std"])
    print(f"sites with >= 5 subjects: {len(keep_sites)} ({len(idx)} subjects); chance {1 / len(keep_sites):.3f}")
    print(summ.round(3).to_string())
    K.manifest_entry("python scripts/experiments/leakage_safe_v1/run_site_diagnostic.py", started,
                     [os.path.relpath(out_csv, K.ROOT)])


if __name__ == "__main__":
    main()
