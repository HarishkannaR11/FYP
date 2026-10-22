"""
Experiment v4 (pre-registered in derivatives/experiments/tabpfn_local492_v4/PLAN.md): TabPFN v2 on all 492
mALFF + mReHo features with no supervised feature selection, on the existing baseline's 25 outer splits.

Reuses biomarker_ablation_v3/run_tabpfn_configs.py functions unchanged (column map, TabPFN call, balancing,
scoring). Outer-test labels are hidden (-1) from the fitting code and read only when scoring after the
probabilities are final. Nothing is selected or tuned, so there is no inner loop.

Outputs: derivatives/experiments/tabpfn_local492_v4/grid/cv5_seed{s}_fold{f}.json, written once (resumable).
Run:  python scripts/experiments/tabpfn_local492_v4/run_local492.py
"""
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "biomarker_ablation_v3"))
import run_tabpfn_configs as T  # noqa: E402  (sets thread env vars first)

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402

EXP = os.path.join(T.ROOT, "derivatives", "experiments", "tabpfn_local492_v4")
COLS = T.pool_columns("local")                       # 246 mALFF + 246 mReHo columns of the 984-feature matrix


def script_hash():
    return hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]


def fit(d, y_sel, seed, fold, tr, te):
    """Everything that learns. Never sees an outer-test label."""
    assert (y_sel[te] == T.HIDDEN).all() and (y_sel[tr] >= 0).all()
    p = T.tabpfn_probs(d[1][:, COLS].astype(np.float32), y_sel, tr, te, seed + fold)
    return {"probs": p, "train_counts": np.bincount(y_sel[tr], minlength=T.NS).tolist()}


def run_split(d, y, seed, fold, fp, tr, te):
    out = fit(d, T.hide(y, te), seed, fold, tr, te)
    frozen = hashlib.sha256(np.ascontiguousarray(out["probs"]).tobytes()).hexdigest()
    heldout, pb = T.score_record(y, te, out)
    assert hashlib.sha256(np.ascontiguousarray(out["probs"]).tobytes()).hexdigest() == frozen
    return {"config": "L492_mALFF_mReHo_noselect", "seed": seed, "fold": fold, "fold_fingerprint": fp,
            "script_sha256": script_hash(), "uses_run_tabpfn_configs_sha256": T.script_hash(),
            "n_features": int(COLS.size), "columns": COLS.tolist(), "test_idx": te.tolist(), "true": y[te].tolist(),
            "train_counts": out["train_counts"], "pred": out["probs"].argmax(1).tolist(),
            "pred_balanced": pb.argmax(1).tolist(), "probs": T.S.r6(out["probs"]), "heldout": heldout}


def main():
    if not os.path.exists(os.path.join(EXP, "PLAN.md")):
        raise SystemExit("PLAN.md (pre-registration) missing")
    lock = os.path.join(EXP, "run.lock")
    if os.path.exists(lock) and psutil.pid_exists(int(open(lock).read().strip() or 0)):
        raise SystemExit("another run is active; not starting a duplicate")
    mpath = os.path.join(EXP, "manifest.json")
    if os.path.exists(mpath) and json.load(open(mpath))["script_sha256"] != script_hash():
        raise SystemExit("run_local492.py changed since the run started; refusing to mix versions")
    json.dump({"experiment": "tabpfn_local492_v4", "script_sha256": script_hash(),
               "uses_run_tabpfn_configs_sha256": T.script_hash(), "n_features": int(COLS.size),
               "baseline": "derivatives/experiments/biomarker_ablation_v3/grid/A_all4",
               "tabpfn": getattr(T.tabpfn, "__version__", "?"), "created": time.strftime("%Y-%m-%d %H:%M:%S")},
              open(mpath, "w"), indent=2)
    open(lock, "w").write(str(os.getpid()))
    torch.set_num_threads(4)
    try:
        assert COLS.size == 492 and np.array_equal(COLS, np.sort(np.r_[np.arange(0, 738, 3), np.arange(1, 738, 3)]))
        d = T.S.load_cohort()
        y = d[3]
        plan = T.S.cv_plan(y, d[4], 5)
        base = {(r["seed"], r["fold"]): r["test_index"] for r in json.load(open(T.TABPFN_BASELINE))["records"]}
        for seed in T.H.SEEDS:
            fp, folds = plan[seed]
            for f, (tr, te) in enumerate(folds):
                assert te.tolist() == base[(seed, f)], "outer split differs from the TabPFN baseline"
                path = os.path.join(EXP, "grid", f"cv5_seed{seed}_fold{f}.json")
                if os.path.exists(path):
                    continue
                t0 = time.time()
                rec = run_split(d, y, seed, f, fp, tr, te)
                rec["seconds"] = round(time.time() - t0, 1)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                json.dump(rec, open(path + ".tmp", "w"))
                os.replace(path + ".tmp", path)
                print(f"seed {seed} fold {f}: {rec['n_features']} features, {rec['seconds']:.0f}s", flush=True)
        print("done", flush=True)
    finally:
        os.remove(lock)


if __name__ == "__main__":
    main()
