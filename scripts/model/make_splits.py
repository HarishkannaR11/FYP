"""
Subject-level splits: a locked 70/30 train/test split, with K-fold
cross-validation INSIDE the 70% training portion.

Design
------
    116 subjects
      |-- 70%  training pool   -> StratifiedKFold(K) for model selection
      |-- 30%  held-out test   -> touched ONCE, at the very end

"70/30 with cross-fold" is implemented as a held-out test set plus
cross-validation within the training pool, rather than K repetitions of a
70/30 split. The reason is that cross-validation inside the training pool is
what selects the model, while the 30% stays untouched so the final number is
not the one the model was tuned on. Reporting a figure that was used for
selection is the most common way small-sample work overstates performance.

Leakage control
---------------
  * Splits are over SUBJECTS. Each subject contributes exactly one
    acquisition here, so a subject cannot straddle a fold by construction;
    the split is still written in terms of subject IDs so this stays true if
    repeat visits are ever reinstated.
  * Stratified by diagnostic stage, so every fold keeps the class mix.
  * No feature value, biomarker or QC metric enters the split -- only the
    label, which is required for stratification.
  * Any cross-subject scaling is fitted on training folds ONLY; that happens
    in the training script, not here.

Everything is seeded and written to CSV so the exact split is auditable and
reusable across runs and across baselines.
"""
import csv
import hashlib
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "derivatives", "model_ready")
OUT = os.path.join(SRC, "splits")

STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
TEST_FRACTION = 0.30
N_FOLDS = 5
SEED = 20261015


def stratified_holdout(y, frac, rng):
    """Return (train_pool_idx, test_idx), stratified, subject-level."""
    test = []
    for g in range(len(STAGES)):
        idx = np.flatnonzero(y == g)
        rng.shuffle(idx)
        n_test = int(round(frac * idx.size))
        n_test = max(1, min(n_test, idx.size - 1))     # never empty either side
        test.extend(idx[:n_test].tolist())
    test = np.array(sorted(test))
    train = np.array(sorted(set(range(y.size)) - set(test.tolist())))
    return train, test


def stratified_folds(y, pool, k, rng):
    """Assign each index in `pool` to one of k folds, stratified by label.

    The round-robin carries its offset from one class to the next. Restarting
    at fold 0 for every class would pile every class's remainder onto the same
    low-numbered folds, leaving the last fold noticeably smaller."""
    fold_of = {}
    start = 0
    for g in range(len(STAGES)):
        idx = pool[y[pool] == g].copy()
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            fold_of[int(i)] = (start + j) % k
        start = (start + idx.size) % k
    return fold_of


def main():
    os.makedirs(OUT, exist_ok=True)
    y = np.load(os.path.join(SRC, "y.npy"))
    subjects = np.load(os.path.join(SRC, "subjects.npy")).astype(str)
    meta = json.load(open(os.path.join(SRC, "metadata.json")))
    if meta["stage_order"] != STAGES:
        raise SystemExit(f"stage order mismatch: {meta['stage_order']}")
    if len(subjects) != len(set(subjects.tolist())):
        raise SystemExit("subjects are not unique")
    n = y.size
    print(f"cohort: {n} subjects, {len(set(subjects.tolist()))} unique")
    print(f"class counts: {{" + ", ".join(f'{s}: {int((y == i).sum())}'
                                          for i, s in enumerate(STAGES)) + "}")

    rng = np.random.default_rng(SEED)
    pool, test = stratified_holdout(y, TEST_FRACTION, rng)
    if set(pool.tolist()) & set(test.tolist()):
        raise SystemExit("train pool and test set overlap")
    if pool.size + test.size != n:
        raise SystemExit("split does not cover the cohort")

    print(f"\nheld-out test : {test.size} subjects ({test.size / n:.1%})")
    print(f"training pool : {pool.size} subjects ({pool.size / n:.1%})")
    print(f"{'stage':6s} {'pool':>5s} {'test':>5s}")
    for g, s in enumerate(STAGES):
        print(f"{s:6s} {int((y[pool] == g).sum()):5d} {int((y[test] == g).sum()):5d}")

    fold_of = stratified_folds(y, pool, N_FOLDS, rng)
    print(f"\n{N_FOLDS}-fold CV inside the training pool:")
    print(f"{'fold':>4s} {'train':>6s} {'val':>4s}   per-class val")
    for f in range(N_FOLDS):
        val = np.array([i for i in pool if fold_of[int(i)] == f])
        tr = np.array([i for i in pool if fold_of[int(i)] != f])
        per = " ".join(f"{s}:{int((y[val] == g).sum())}"
                       for g, s in enumerate(STAGES))
        print(f"{f:4d} {tr.size:6d} {val.size:4d}   {per}")
        if set(tr.tolist()) & set(val.tolist()):
            raise SystemExit(f"fold {f}: train and val overlap")

    rows = []
    for i in range(n):
        in_test = i in set(test.tolist())
        rows.append([subjects[i], STAGES[y[i]], int(y[i]), i,
                     "test" if in_test else "train_pool",
                     "" if in_test else fold_of[int(i)]])
    p = os.path.join(OUT, "split_assignment.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["subject_id", "stage", "stage_index", "row_index",
                    "partition", "cv_fold"])
        w.writerows(rows)
    print(f"\nwrote {os.path.relpath(p, ROOT)}")

    # fingerprint so a training run can prove which split it used
    h = hashlib.sha256()
    for r in rows:
        h.update(f"{r[0]}|{r[4]}|{r[5]}".encode())
    fp = h.hexdigest()[:16]
    meta_out = {
        "seed": SEED, "test_fraction": TEST_FRACTION, "n_folds": N_FOLDS,
        "n_subjects": int(n), "n_train_pool": int(pool.size),
        "n_test": int(test.size),
        "stage_order": STAGES,
        "split_fingerprint": fp,
        "stratified_by": "diagnostic stage",
        "split_unit": "subject",
        "test_set_policy": "evaluated once, after model selection is complete",
        "scaling_policy": "any cross-subject scaler is fitted on training folds only",
        "pool_class_counts": {s: int((y[pool] == g).sum())
                              for g, s in enumerate(STAGES)},
        "test_class_counts": {s: int((y[test] == g).sum())
                              for g, s in enumerate(STAGES)},
    }
    p = os.path.join(OUT, "split_metadata.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(meta_out, f, indent=2)
    print(f"wrote {os.path.relpath(p, ROOT)}")
    print(f"split fingerprint: {fp}")

    print("\nNOTE: the test set holds "
          f"{test.size} subjects, about {test.size / len(STAGES):.0f} per class. "
          "Metrics from a set this small carry wide uncertainty and must be "
          "reported with that caveat.")


if __name__ == "__main__":
    main()
