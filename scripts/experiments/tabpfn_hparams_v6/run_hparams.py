"""
TabPFN v2 controlled hyperparameter study (v6; pre-registered in derivatives/experiments/tabpfn_hparams_v6/PLAN.md).

  C1  effective defaults passed explicitly: n_estimators=8, softmax_temperature=0.9, average_before_softmax=False
  C2  n_estimators=32
  C3  average_before_softmax=True

Everything else is the baseline's: SVM-RFE k = 50 on the 984 features fitted on the outer training subjects
(biomarker_ablation_v3 select(), verified to reproduce the baseline), raw selected features, V2 weights on CPU,
random_state = seed + fold, balance_probabilities False. Outer-test labels are hidden (-1) from fitting and read only
when scoring. Each record stores the fitted model's resolved settings (n_estimators_, softmax_temperature_,
number of ensemble configurations) and any warnings raised.

Outputs: derivatives/experiments/tabpfn_hparams_v6/grid/<config>/cv5_seed{s}_fold{f}.json (written once; resumable).
Run:  python scripts/experiments/tabpfn_hparams_v6/run_hparams.py --configs C1 C3 C2
"""
import argparse
import gc
import hashlib
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "biomarker_ablation_v3"))
import run_tabpfn_configs as T  # noqa: E402  (sets thread env vars first)

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import tabpfn  # noqa: E402
import torch  # noqa: E402
from tabpfn import TabPFNClassifier  # noqa: E402
from tabpfn.constants import ModelVersion  # noqa: E402

EXP = os.path.join(T.ROOT, "derivatives", "experiments", "tabpfn_hparams_v6")
CONFIGS = {"C1": {"n_estimators": 8, "softmax_temperature": 0.9, "average_before_softmax": False},
           "C2": {"n_estimators": 32},
           "C3": {"average_before_softmax": True}}


def script_hash():
    return hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]


def fit_predict(X, y_fit, rows, pred_rows, rs, overrides):
    """T.tabpfn_probs with the configuration's overrides; also returns the fitted model's resolved settings."""
    assert (y_fit[rows] >= 0).all()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        m = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu", random_state=rs, **overrides)
        m.fit(X[rows], y_fit[rows])
        p = np.asarray(m.predict_proba(X[pred_rows]), dtype=np.float64)
    assert list(m.classes_) == list(range(T.NS))
    resolved = {"n_estimators_": int(m.n_estimators_), "softmax_temperature_": float(m.softmax_temperature_),
                "n_ensemble_configs": len(m.ensemble_configs_), "average_before_softmax": bool(m.average_before_softmax),
                "balance_probabilities": bool(m.balance_probabilities)}
    msgs = sorted({str(w.message)[:200] for w in caught})
    del m
    gc.collect()
    return p, resolved, msgs


def run_split(d, y, cname, seed, fold, fp, tr, te):
    y_sel = T.hide(y, te)
    sup = T.select(d[1], y_sel, tr, T.pool_columns("all4"), seed + fold, 50)
    p, resolved, msgs = fit_predict(d[1][:, sup].astype(np.float32), y_sel, tr, te, seed + fold, CONFIGS[cname])
    frozen = hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest()
    heldout, _ = T.score_record(y, te, {"probs": p, "train_counts": np.bincount(y_sel[tr], minlength=T.NS)})
    assert hashlib.sha256(np.ascontiguousarray(p).tobytes()).hexdigest() == frozen
    return {"config": cname, "overrides": CONFIGS[cname], "resolved": resolved, "warnings": msgs, "seed": seed,
            "fold": fold, "fold_fingerprint": fp, "script_sha256": script_hash(), "tabpfn": tabpfn.__version__,
            "support": sup.tolist(), "test_idx": te.tolist(), "true": y[te].tolist(), "pred": p.argmax(1).tolist(),
            "probs": T.S.r6(p), "heldout": heldout["raw"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True, choices=list(CONFIGS))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    if not os.path.exists(os.path.join(EXP, "PLAN.md")):
        raise SystemExit("PLAN.md (pre-registration) missing")
    lock = os.path.join(EXP, "run.lock")
    if os.path.exists(lock) and psutil.pid_exists(int(open(lock).read().strip() or 0)):
        raise SystemExit("another run is active; not starting a duplicate")
    mpath = os.path.join(EXP, "manifest.json")
    if os.path.exists(mpath) and json.load(open(mpath))["script_sha256"] != script_hash():
        raise SystemExit("run_hparams.py changed since the run started; refusing to mix versions")
    json.dump({"experiment": "tabpfn_hparams_v6", "script_sha256": script_hash(), "configs": CONFIGS,
               "tabpfn": tabpfn.__version__, "baseline": "model_training/tabpfn/cv116_tabpfn_svmrfe50.json",
               "created": time.strftime("%Y-%m-%d %H:%M:%S")}, open(mpath, "w"), indent=2)
    open(lock, "w").write(str(os.getpid()))
    torch.set_num_threads(args.threads)
    try:
        d = T.S.load_cohort()
        y = d[3]
        plan = T.S.cv_plan(y, d[4], 5)
        base = {(r["seed"], r["fold"]): r["test_index"] for r in json.load(open(T.TABPFN_BASELINE))["records"]}
        for cname in args.configs:
            t_cfg = time.time()
            for seed in T.H.SEEDS:
                fp, folds = plan[seed]
                for f, (tr, te) in enumerate(folds):
                    assert te.tolist() == base[(seed, f)], "outer split differs from the baseline"
                    path = os.path.join(EXP, "grid", cname, f"cv5_seed{seed}_fold{f}.json")
                    if os.path.exists(path):
                        continue
                    t0 = time.time()
                    rec = run_split(d, y, cname, seed, f, fp, tr, te)
                    rec["seconds"] = round(time.time() - t0, 1)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    json.dump(rec, open(path + ".tmp", "w"))
                    os.replace(path + ".tmp", path)
                    print(f"{cname} seed {seed} fold {f}: n_estimators_ {rec['resolved']['n_estimators_']}, "
                          f"warnings {len(rec['warnings'])}, {rec['seconds']:.0f}s", flush=True)
            print(f"{cname} done in {(time.time() - t_cfg) / 60:.1f} min", flush=True)
    finally:
        os.remove(lock)


if __name__ == "__main__":
    main()
