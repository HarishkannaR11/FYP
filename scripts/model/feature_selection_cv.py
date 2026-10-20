"""
Leak-free repeated cross-validation harness for the feature-selection study.

Steps 2-4 of md files/FEATURE_SELECTION_PROCESS_AND_QUANTUM_METHODS.md:

  2. load the data, verify the split, keep the 35 held-out subjects LOCKED
  3. a shared harness: scale -> reduce/select -> classify -> score, with every
     learned step fitted on a fold's training subjects only
  4. reproduce the baseline and a shuffled-label control

Design (identical to scripts/model/repeated_cv.py, but torch-free):
  * Only the 81 training-pool subjects are ever kept in memory. The feature
    rows of the 35 test subjects are dropped immediately after loading.
  * Same 5 seeds and same stratified fold rule, so folds are IDENTICAL to the
    earlier experiments. This is verified against the fold fingerprints
    recorded in derivatives/model_training/biomarker_ablation/ablation.log.
  * A method is a scikit-learn Pipeline, so scaling/reduction/selection are fitted
    on each fold's training subjects only by construction.
  * Shuffled-label control permutes the TRAINING labels of each fold and
    scores against the true validation labels. It should fall to chance (0.167).

Run:  python scripts/model/feature_selection_cv.py
"""
import csv
import hashlib
import json
import os
import platform
import sys
import time
import warnings

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import scipy
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "derivatives", "model_ready")
SPLITS = os.path.join(SRC, "splits")
OUT = os.path.join(ROOT, "derivatives", "model_training", "feature_selection")

STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
N_STAGES = 6
SEEDS = [20261015, 20261016, 20261017, 20261018, 20261019]
N_FOLDS = 5
CHANCE = 1.0 / N_STAGES

# fold fingerprints recorded by the earlier ablation run (ablation.log)
EXPECTED_FINGERPRINTS = {
    20261015: "9d88243dd033155e",
    20261016: "cb9c36eccb7d4496",
    20261017: "09525a6309ccdf34",
    20261018: "ca297c0daac768d3",
    20261019: "bc396eacad3893f2",
}
EXPECTED_SPLIT_FINGERPRINT = "6f478fbcf56364e2"

_log = []


def log(m=""):
    print(m, flush=True)
    _log.append(str(m))


# ------------------------------------------------------------------ data
class PoolData:
    """Training-pool data only. Test-subject feature rows are never retained."""

    def __init__(self):
        meta = json.load(open(os.path.join(SRC, "metadata.json")))
        if meta["stage_order"] != STAGES:
            raise SystemExit("stage order mismatch")
        smeta = json.load(open(os.path.join(SPLITS, "split_metadata.json")))
        y_all = np.load(os.path.join(SRC, "y.npy")).astype(np.int64)
        subj_all = np.load(os.path.join(SRC, "subjects.npy")).astype(str)

        rows = list(csv.DictReader(open(os.path.join(SPLITS, "split_assignment.csv"))))
        if [r["subject_id"] for r in rows] != list(subj_all):
            raise SystemExit("split file does not match the cohort order")
        part = np.array([r["partition"] for r in rows])
        pool = np.flatnonzero(part == "train_pool")
        test = np.flatnonzero(part == "test")
        if set(pool.tolist()) & set(test.tolist()):
            raise SystemExit("pool and test overlap")

        self.split_meta = smeta
        self.n_pool, self.n_test = int(pool.size), int(test.size)
        self.pool_idx = pool
        # global-index labels are kept ONLY because the fold rule shuffles by
        # global index; features of test rows are dropped below
        self._y_global = y_all
        self.y = y_all[pool]
        self.subjects = subj_all[pool]

        reg = np.load(os.path.join(SRC, "X_regional.npy"))[pool].astype(np.float64)
        fcu = np.load(os.path.join(SRC, "X_fc_roi_unsigned.npy"))[pool].astype(np.float64)
        fc = np.load(os.path.join(SRC, "X_fc.npy"), mmap_mode="r")
        iu = np.triu_indices(246, k=1)
        fc_edges = np.stack([np.asarray(fc[i])[iu] for i in pool]).astype(np.float64)
        for nm, a in (("regional", reg), ("fc_roi", fcu), ("fc_edges", fc_edges)):
            if not np.isfinite(a).all():
                raise SystemExit(f"non-finite values in {nm}")

        self.views = {
            "alff": reg[:, :, 0],
            "reho": reg[:, :, 1],
            "dc": reg[:, :, 2],
            "regional": reg.reshape(len(pool), -1),
            "fc_summary": fcu,
            "fc_edges": fc_edges,
        }

    def verify(self):
        smeta = self.split_meta
        checks = {
            "pool size 81": self.n_pool == smeta["n_train_pool"] == 81,
            "test size 35": self.n_test == smeta["n_test"] == 35,
            "split fingerprint": smeta["split_fingerprint"] == EXPECTED_SPLIT_FINGERPRINT,
            "pool class counts match metadata": {
                STAGES[g]: int((self.y == g).sum()) for g in range(N_STAGES)
            } == smeta["pool_class_counts"],
            "no test rows in memory": all(v.shape[0] == self.n_pool
                                          for v in self.views.values()),
        }
        return checks


# ----------------------------------------------------------------- folds
def stratified_folds(y_global, pool, k, rng):
    """Identical rule to make_splits.py / repeated_cv.py (round-robin per class,
    carrying the offset between classes)."""
    fold_of = {}
    start = 0
    for g in range(N_STAGES):
        idx = pool[y_global[pool] == g].copy()
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            fold_of[int(i)] = (start + j) % k
        start = (start + idx.size) % k
    return fold_of


def fold_fingerprint(subjects_global, fold_of, pool):
    h = hashlib.sha256()
    for i in sorted(pool):
        h.update(f"{subjects_global[i]}|{fold_of[int(i)]}".encode())
    return h.hexdigest()[:16]


def make_fold_plan(data):
    """{seed: (fingerprint, [(train_pos, val_pos) x 5])} in POOL coordinates."""
    subj_all = np.load(os.path.join(SRC, "subjects.npy")).astype(str)
    pos_of = {int(g): p for p, g in enumerate(data.pool_idx)}
    plan = {}
    for seed in SEEDS:
        fo = stratified_folds(data._y_global, data.pool_idx, N_FOLDS,
                              np.random.default_rng(seed))
        fp = fold_fingerprint(subj_all, fo, data.pool_idx)
        folds = []
        for f in range(N_FOLDS):
            va = np.array([pos_of[int(i)] for i in data.pool_idx if fo[int(i)] == f])
            tr = np.array([pos_of[int(i)] for i in data.pool_idx if fo[int(i)] != f])
            assert not (set(tr.tolist()) & set(va.tolist()))
            folds.append((tr, va))
        plan[seed] = (fp, folds)
    return plan


# --------------------------------------------------------------- metrics
def qwk(y, pred, k=N_STAGES):
    O = np.zeros((k, k))
    for a, b in zip(y, pred):
        O[a, b] += 1
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], float)
    w /= (k - 1) ** 2
    E = np.outer(np.bincount(y, minlength=k), np.bincount(pred, minlength=k)).astype(float)
    E = E / E.sum() * O.sum()
    den = (w * E).sum()
    return float(1 - (w * O).sum() / den) if den > 0 else float("nan")


def macro_f1(y, pred, k=N_STAGES):
    f1s = []
    for g in range(k):
        tp = float(((pred == g) & (y == g)).sum())
        fp = float(((pred == g) & (y != g)).sum())
        fn = float(((pred != g) & (y == g)).sum())
        d = 2 * tp + fp + fn
        f1s.append(2 * tp / d if d > 0 else 0.0)
    return float(np.mean(f1s))


def evaluate(y, pred):
    recalls = []
    for g in range(N_STAGES):
        m = y == g
        recalls.append(float((pred[m] == g).mean()) if m.sum() else float("nan"))
    return {
        "accuracy": float((pred == y).mean()),
        "balanced_accuracy": float(np.nanmean(recalls)),
        "macro_f1": macro_f1(y, pred),
        "mae_stages": float(np.abs(pred - y).mean()),
        "qwk": qwk(y, pred),
        "n_predicted_classes": int(len(np.unique(pred))),
        "per_class_recall": {STAGES[g]: recalls[g] for g in range(N_STAGES)},
    }


# --------------------------------------------------------------- harness
def run_cv(name, X, y, plan, make_pipeline, shuffle_labels=False):
    """Run one method over 5 seeds x 5 folds. `make_pipeline(seed)` must return a
    fresh sklearn Pipeline; every learned step is fitted on the fold's training
    subjects only. With shuffle_labels the training labels are permuted."""
    recs = []
    t0 = time.time()
    for seed in SEEDS:
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
            m = evaluate(y[va], pred)
            recs.append({"method": name, "seed": seed, "fold": f,
                         "fold_fingerprint": fp, "n_train": int(len(tr)),
                         "n_val": int(len(va)), "n_features_in": int(X.shape[1]),
                         "shuffled_labels": shuffle_labels, **m})
    return recs, time.time() - t0


def summarise(recs):
    """Seed-level mean +/- sd of the 5 seed means (same convention as repeated_cv)."""
    out = {}
    for key in ("balanced_accuracy", "accuracy", "macro_f1", "mae_stages", "qwk"):
        per_seed = [np.mean([r[key] for r in recs if r["seed"] == s]) for s in SEEDS]
        out[key] = (float(np.mean(per_seed)), float(np.std(per_seed, ddof=0)))
    out["fold_sd_bal_acc"] = float(np.std([r["balanced_accuracy"] for r in recs]))
    out["classes_min"] = min(r["n_predicted_classes"] for r in recs)
    out["classes_max"] = max(r["n_predicted_classes"] for r in recs)
    return out


# ------------------------------------------------------- baseline methods
def pipe_dummy(_):
    return Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])


def pipe_logreg(seed):
    return Pipeline([("scale", StandardScaler()),
                     ("clf", LogisticRegression(C=0.05, max_iter=2000,
                                                class_weight="balanced"))])


def pipe_mlp(seed):
    return Pipeline([("scale", StandardScaler()),
                     ("clf", MLPClassifier(hidden_layer_sizes=(64, 64), alpha=1e-2,
                                           max_iter=300, random_state=seed))])


def save_outputs(tag, records, extra):
    os.makedirs(OUT, exist_ok=True)
    jp = os.path.join(OUT, f"{tag}_records.json")
    lp = os.path.join(OUT, f"{tag}.log")
    for p in (jp, lp):
        if os.path.exists(p):               # INSTRUCTIONS.md: never overwrite
            log(f"COLLISION: {os.path.relpath(p, ROOT)} exists; not overwritten")
            return
    json.dump({"extra": extra, "records": records}, open(jp, "w"), indent=1)
    open(lp, "w", encoding="utf-8").write("\n".join(_log) + "\n")
    log(f"wrote {os.path.relpath(jp, ROOT)}")


def main():
    log("=" * 86)
    log("FEATURE-SELECTION HARNESS -- steps 2, 3 and 4")
    log("=" * 86)
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "scipy": scipy.__version__, "scikit-learn": sklearn.__version__}
    log("versions: " + json.dumps(versions))

    # ---- step 2: data and split verification
    data = PoolData()
    log("\n[STEP 2] DATA AND SPLIT CHECK")
    ok = True
    for k, v in data.verify().items():
        log(f"  {'PASS' if v else 'FAIL'}  {k}")
        ok &= bool(v)
    log("  views (rows = 81 pool subjects only): "
        + ", ".join(f"{k}={v.shape[1]}" for k, v in data.views.items()))
    log("  pool class counts: "
        + ", ".join(f"{STAGES[g]}={int((data.y == g).sum())}" for g in range(N_STAGES)))
    log("  test set: 35 subjects, feature rows NOT loaded into memory")
    if not ok:
        raise SystemExit("step 2 failed")

    # ---- step 3: folds identical to earlier experiments
    log("\n[STEP 3] FOLD PLAN vs FINGERPRINTS FROM THE EARLIER ABLATION RUN")
    plan = make_fold_plan(data)
    for seed in SEEDS:
        fp = plan[seed][0]
        same = fp == EXPECTED_FINGERPRINTS[seed]
        sizes = [len(v) for _, v in plan[seed][1]]
        log(f"  seed {seed}  {fp}  expected {EXPECTED_FINGERPRINTS[seed]}  "
            f"{'MATCH' if same else 'MISMATCH'}  val sizes {sizes}")
        ok &= same
    if not ok:
        raise SystemExit("fold plan does not reproduce earlier folds")

    # ---- step 4: baselines and shuffled-label controls
    log("\n[STEP 4] BASELINES AND SHUFFLED-LABEL CONTROLS (25 folds each, no reduction)")
    log(f"  chance for 6 balanced classes = {CHANCE:.3f}")
    log("  reference (earlier torch MLP, DC_z only, no reduction): bal.acc 0.307 +/- 0.029")
    log("  note: torch is not installed here, so the reference is compared, not re-run")
    jobs = [("majority", "dc", pipe_dummy, False),
            ("logreg_dc", "dc", pipe_logreg, False),
            ("logreg_dc_SHUFFLED", "dc", pipe_logreg, True),
            ("mlp_dc", "dc", pipe_mlp, False),
            ("mlp_dc_SHUFFLED", "dc", pipe_mlp, True),
            ("logreg_regional", "regional", pipe_logreg, False),
            ("logreg_regional_SHUFFLED", "regional", pipe_logreg, True)]
    log(f"\n  {'method':26s} {'bal.acc':>14s} {'accuracy':>8s} {'macroF1':>8s} "
        f"{'MAE':>6s} {'QWK':>7s} {'cls':>5s}  time")
    all_recs, summary = [], {}
    for name, view, mk, shuf in jobs:
        recs, el = run_cv(name, data.views[view], data.y, plan, mk, shuf)
        s = summarise(recs)
        summary[name] = {"view": view, **s}
        all_recs += recs
        b = s["balanced_accuracy"]
        log(f"  {name:26s} {b[0]:.3f}+/-{b[1]:.3f} {s['accuracy'][0]:8.3f} "
            f"{s['macro_f1'][0]:8.3f} {s['mae_stages'][0]:6.3f} {s['qwk'][0]:+7.3f} "
            f"{s['classes_min']}-{s['classes_max']}  {el:.0f}s")

    log("\n  leak check: every SHUFFLED row must sit near chance (0.167)")
    for name in summary:
        if name.endswith("SHUFFLED"):
            b = summary[name]["balanced_accuracy"][0]
            log(f"    {name:26s} {b:.3f}  "
                f"{'OK (near chance)' if abs(b - CHANCE) < 0.06 else 'CHECK: away from chance'}")

    save_outputs("step4_baseline", all_recs,
                 {"versions": versions, "summary": summary,
                  "seeds": SEEDS, "n_folds": N_FOLDS,
                  "test_set_touched": False})


if __name__ == "__main__":
    main()
