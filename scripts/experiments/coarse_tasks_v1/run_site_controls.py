"""
C1 (site-only rule) and the labelled post-hoc reference R (collapsed six-class TabPFN predictions).
No brain data and no model weights are used for C1. Writes derivatives/experiments/coarse_tasks_v1/site_only.json.

Run:  python scripts/experiments/coarse_tasks_v1/run_site_controls.py
"""
import json
import os

import numpy as np

import common3 as K

y6, y3, subjects, site = K.load_cohort()
folds = K.load_folds()
os.makedirs(K.OUT, exist_ok=True)
dest = os.path.join(K.OUT, "site_only.json")
if os.path.exists(dest):
    raise SystemExit("COLLISION: site_only.json exists; not overwritten")


def site_rule(tr, te, rng):
    prior = np.bincount(y3[tr], minlength=K.NG) / len(tr)
    pred = []
    for i in te:
        m = site[tr] == site[i]
        cnt = np.bincount(y3[tr][m], minlength=K.NG).astype(float)
        score = (cnt + 1.0) / (cnt.sum() + K.NG) / prior
        best = np.flatnonzero(score >= score.max() - 1e-12)
        pred.append(int(rng.choice(best)))            # unseen site -> all equal -> random tie-break (fixed seed)
    return np.array(pred)


records = []
for (seed, f), (tr, te, fp) in sorted(folds.items()):
    rng = np.random.default_rng(seed * 100 + f)
    pr = site_rule(tr, te, rng)
    records.append({"seed": seed, "fold": f, "fold_fingerprint": fp, "test_index": te.tolist(), "pred": pr.tolist(),
                    "true": y3[te].tolist(), "balanced_accuracy": K.bal(y3[te], pr)})

# post-hoc reference R: stored six-class TabPFN predictions, collapsed after the fact (NOT a headline)
tab = json.load(open(K.FOLD_SOURCE))["records"]
ref = [{"seed": r["seed"], "fold": r["fold"], "test_index": r["test_index"],
        "pred": [K.GROUP_MAP[p] for p in r["pred"]]} for r in tab]
json.dump({"declared": "docs/NEW_EXPERIMENT_PREDECLARATION.md", "rule": "C1 site-only", "records": records,
           "reference_R_posthoc_collapsed_six_class_tabpfn": ref}, open(dest, "w"))
print("wrote", dest, len(records), "splits")
