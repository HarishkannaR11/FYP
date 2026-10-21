"""
Learning curve: does accuracy keep rising as the number of training subjects
grows? If it does, more subjects (not a different model) is the way to a
better result.

Design
------
  * The 5-fold plans of the project seeds (train_transformer_svmrfe.cv_plan),
    so the test fold of every point is a real held-out fold of ~23 subjects.
  * Inside each training fold, nested stratified subsets of 30, 45, 60, 75 and
    all (~93) subjects: per stage the training subjects are shuffled once and
    the first share is taken, so every smaller subset sits inside the larger
    ones and the curve compares like with like.
  * Scaling, SVM-RFE selection, clinical fill/scaling and the 4 early-stopped
    inner models are all fitted on the subset only (the same pipeline as
    train_transformer_comorbidity.py / train_transformer_svmrfe.py).
  * Transformers: first 2 seeds (10 folds) per size, to keep the run near
    45 minutes. Logistic-regression references: all 5 seeds (25 folds).

Models
------
  transformer_svmrfe50_clin   model A: rs-fMRI SVM-RFE 50 + 8 comorbidities
  transformer_svmrfe50        rs-fMRI only
  logreg_svmrfe50_clin        model A features, logistic regression
  logreg_svmrfe50             rs-fMRI only, logistic regression

Run:  python scripts/model/learning_curve.py
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_transformer_comorbidity as C  # noqa: E402  (sets thread env vars first)

import numpy as np  # noqa: E402
import torch  # noqa: E402

S, H = C.S, C.H
ROOT = C.ROOT
OUT = os.path.join(ROOT, "derivatives", "model_training", "learning_curve")
SIZES = [30, 45, 60, 75, None]                     # None = the whole training fold
FEATURES = "comorbidity8"
MODELS = {"transformer_svmrfe50_clin": 2, "transformer_svmrfe50": 2,       # value = number of seeds
          "logreg_svmrfe50_clin": 5, "logreg_svmrfe50": 5}

_lines = []


def log(m=""):
    print(m, flush=True)
    _lines.append(str(m))


def nested_subset(y, tr, n, rng_key):
    """Stratified subset of `tr` with n subjects; nested across n for one key."""
    rng = np.random.default_rng(rng_key)
    by = {g: rng.permutation(tr[y[tr] == g]) for g in range(S.N_STAGES)}
    share = np.array([by[g].size for g in range(S.N_STAGES)], float) * n / tr.size
    k = np.floor(share).astype(int)
    for g in np.argsort(-(share - k))[: n - k.sum()]:              # largest remainders
        k[g] += 1
    sub = np.sort(np.concatenate([by[g][: k[g]] for g in range(S.N_STAGES)]))
    assert sub.size == n and set(sub.tolist()) <= set(tr.tolist())
    return sub


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    p_json = os.path.join(OUT, "learning_curve_records.json")
    if os.path.exists(p_json):
        raise SystemExit(f"COLLISION: {p_json} exists; not overwritten")
    os.makedirs(OUT, exist_ok=True)

    data = S.load_cohort()
    y, subjects = data[3], data[4]
    cols = C.FEATURE_SETS[FEATURES]
    clin, ceil = C.load_clinical(y, subjects, cols)
    plan = S.cv_plan(y, subjects)
    log("=" * 100)
    log(f"LEARNING CURVE -- training-set sizes {[s or 'all' for s in SIZES]}, 5-fold plans, "
        f"comorbidities '{FEATURES}' {cols}")
    log("=" * 100)
    log("fold fingerprints " + json.dumps({s: plan[s][0] for s in H.SEEDS}))

    records = []
    t0 = time.time()
    for cand, n_seeds in MODELS.items():
        log(f"\n{cand}  ({n_seeds} seeds x 5 folds per size)")
        for seed in H.SEEDS[:n_seeds]:
            fp, folds = plan[seed]
            for f, (tr, te) in enumerate(folds):
                for n in SIZES:
                    sub = tr if n is None else nested_subset(y, tr, n, [seed, f, 99])
                    if cand in C.IMAGING_ONLY:
                        r = S.run_split(cand, data, sub, te, seed, f, fp, S.TRAIN["max_epochs"], None,
                                        explain_outputs=False)
                    else:
                        r = C.run_split(cand, data, clin, ceil, cols, sub, te, seed, f, fp,
                                        S.TRAIN["max_epochs"], explain_outputs=False)
                    keep = ("seed", "fold", "fold_fingerprint", "n_train", "n_test", "balanced_accuracy",
                            "accuracy", "macro_f1", "mae_stages", "qwk", "train_balanced_accuracy",
                            "test_index", "true", "pred", "probs")
                    records.append({"model": cand, "size": n or "all", **{k: r[k] for k in keep}})
        sub = [r for r in records if r["model"] == cand]
        for n in SIZES:
            rr = [r for r in sub if r["size"] == (n or "all")]
            log(f"  n_train {np.mean([r['n_train'] for r in rr]):5.1f}  "
                f"test bal.acc {np.mean([r['balanced_accuracy'] for r in rr]):.3f}  "
                f"train {np.mean([r['train_balanced_accuracy'] for r in rr]):.3f}  ({len(rr)} folds)")

    json.dump({"extra": {"sizes": [s or "all" for s in SIZES], "models": MODELS, "features": FEATURES,
                         "clinical_columns": cols, "seconds": round(time.time() - t0, 1)},
               "records": records}, open(p_json, "w"))
    open(os.path.join(OUT, "learning_curve.log"), "w", encoding="utf-8").write("\n".join(_lines) + "\n")
    log(f"\nwrote {os.path.relpath(p_json, ROOT)}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
