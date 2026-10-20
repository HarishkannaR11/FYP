"""
Training vs validation balanced accuracy for every method's best configuration
(over all views) and for every method on the fixed all4 (984) view.

Same 5 seeds x 5 folds. Selectors whose per-fold choice is stored in the
step records (QUBO/QAOA, GA, PSO, Boruta, stability, ...) are not re-run: the
stored feature indices are reused, and the validation score is recomputed and
checked against the stored one. Extraction methods (PCA etc.) are re-fitted.
The 35 test subjects are never loaded.

Run:  python scripts/model/feature_selection_train_val.py
"""
import glob
import json
import os
import re
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feature_selection_cv as H  # noqa: E402
import feature_selection_methods as M  # noqa: E402

import numpy as np  # noqa: E402
from sklearn.metrics import balanced_accuracy_score as bas  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import FunctionTransformer, StandardScaler  # noqa: E402

log = H.log
JOBS = (M.jobs_step5, M.jobs_step6a, M.jobs_step6b, M.jobs_step6c, M.jobs_step7)


def load_configs():
    """{(method, view): (summary, records)} from the step result files; the buggy
    first stability run is skipped (the fixed re-run is used)."""
    out = {}
    for f in sorted(glob.glob(os.path.join(H.OUT, "step*_records.json"))):
        if "step4" in f:
            continue
        d = json.load(open(f))
        for key, s in d["extra"]["summary"].items():
            if s["method"].startswith("Stability") and "stability_fix" not in f:
                continue
            recs = [r for r in d["records"] if r["method"] == s["method"]
                    and r.get("n_features_in") == s["n_features_in"]]
            out[(s["method"], s["view"])] = (s, recs)
    return out


def family(m):
    return re.sub(r"_k\d+$", "", m)


def find_job(name, view):
    for mk in JOBS:
        for n, v, f in mk():
            if n == name and v == view:
                return f
    raise KeyError((name, view))


def train_val(name, view, recs, X, y, plan):
    stored = {(r["seed"], r["fold"]): r for r in recs}
    tr_seed, va_seed, max_diff = [], [], 0.0
    for seed in H.SEEDS:
        a, b = [], []
        for f, (tr, va) in enumerate(plan[seed][1]):
            r = stored.get((seed, f))
            if r is not None and "selected" in r:
                idx = np.array(r["selected"])
                pipe = Pipeline([("scale", StandardScaler()),
                                 ("select", FunctionTransformer(lambda Z, idx=idx: Z[:, idx])),
                                 ("post", StandardScaler()), ("clf", M.classifier())])
            else:
                pipe = find_job(name, view)(seed + f)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pipe.fit(X[tr], y[tr])
                a.append(bas(y[tr], pipe.predict(X[tr])))
                v = bas(y[va], pipe.predict(X[va]))
            b.append(v)
            if r is not None:
                max_diff = max(max_diff, abs(v - r["balanced_accuracy"]))
        tr_seed.append(np.mean(a))
        va_seed.append(np.mean(b))
    return (float(np.mean(tr_seed)), float(np.std(tr_seed)),
            float(np.mean(va_seed)), float(np.std(va_seed)), max_diff)


def main():
    data = H.PoolData()
    plan = H.make_fold_plan(data)
    if not all(plan[s][0] == H.EXPECTED_FINGERPRINTS[s] for s in H.SEEDS):
        raise SystemExit("fold fingerprints do not match")
    data.views["fc_blocks"] = M.fc_blocks(data.views["fc_edges"])
    data.views["all4"] = np.hstack([data.views["regional"], data.views["fc_summary"]])
    cfgs = load_configs()

    best_any, best_all4 = {}, {}
    for (m, v), (s, recs) in cfgs.items():
        fam = family(m)
        b = s["balanced_accuracy"][0]
        if fam not in best_any or b > best_any[fam][2]["balanced_accuracy"][0]:
            best_any[fam] = (m, v, s, recs)
        if v == "all4" and (fam not in best_all4 or b > best_all4[fam][2]["balanced_accuracy"][0]):
            best_all4[fam] = (m, v, s, recs)

    result = {}
    for label, table in (("best_any_view", best_any), ("all4_984", best_all4)):
        log(f"\n[{label}]  {'method':22s} {'view':9s} {'features':>13s} {'train':>14s} {'validation':>14s} {'gap':>6s} {'check':>6s}")
        rows = []
        for fam, (m, v, s, recs) in sorted(table.items(), key=lambda kv: -kv[1][2]["balanced_accuracy"][0]):
            t, ts, va, vs, diff = train_val(m, v, recs, data.views[v], data.y, plan)
            row = {"family": fam, "method": m, "view": v, "n_in": s["n_features_in"],
                   "n_out": s["n_features_out"], "train": [t, ts], "validation": [va, vs],
                   "stored_validation": s["balanced_accuracy"], "max_fold_diff": diff}
            rows.append(row)
            log(f"  {fam:22s} {v:9s} {s['n_features_in']:>6d}->{s['n_features_out']:<5d} "
                f"{t:.3f}+/-{ts:.3f} {va:.3f}+/-{vs:.3f} {t - va:6.3f} {diff:6.3f}")
        result[label] = rows
    H.save_outputs("train_vs_validation", [], result)


if __name__ == "__main__":
    main()
