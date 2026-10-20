"""
Steps 8 and 9 of md files/FEATURE_SELECTION_PROCESS_AND_QUANTUM_METHODS.md.

  8. Shuffled-label control on the frozen winner(s): same 5 seeds x 5 folds,
     training labels permuted; must fall to chance (0.167).
  9. ONE evaluation on the 35 locked test subjects: fit on all 81 training-pool
     subjects, predict the test subjects, report whatever comes out.

The configurations are frozen from cross-validation alone and passed on the
command line BEFORE any test row is read, e.g.

  python scripts/model/feature_selection_final.py PCA_k10:regional L1_k10:regional

The first one is the primary winner; any others are pre-declared secondary
reports (they are not used to choose between methods after seeing the test).
"""
import csv
import json
import os
import sys
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feature_selection_cv as H  # noqa: E402
import feature_selection_methods as M  # noqa: E402

import numpy as np  # noqa: E402

log = H.log


def build_view(view, X_regional, X_fc_unsigned, X_fc=None):
    """Same view definitions as feature_selection_methods.main()."""
    reg = X_regional.reshape(len(X_regional), -1)
    if view == "regional":
        return reg
    if view == "dc":
        return X_regional[:, :, 2]
    if view == "all4":
        return np.hstack([reg, X_fc_unsigned])
    iu = np.triu_indices(246, k=1)
    edges = np.stack([np.asarray(m)[iu] for m in X_fc])
    if view == "fc_edges":
        return edges
    if view == "fc_blocks":
        return M.fc_blocks(edges)
    raise ValueError(view)


def find_job(name, view):
    for mk in (M.jobs_step5, M.jobs_step6a, M.jobs_step6b, M.jobs_step6c, M.jobs_step7):
        for n, v, f in mk():
            if n == name and v == view:
                return f
    raise SystemExit(f"unknown configuration {name}:{view}")


def main():
    specs = [a.split(":") for a in sys.argv[1:]]
    if not specs:
        raise SystemExit(__doc__)
    log("=" * 96)
    log("FEATURE-SELECTION STUDY -- steps 8 (shuffled control) and 9 (held-out test, used once)")
    log("=" * 96)
    log("frozen from CV before reading test rows: " + ", ".join(f"{n} on {v}" for n, v in specs)
        + f"  (primary = {specs[0][0]} on {specs[0][1]})")

    data = H.PoolData()
    plan = H.make_fold_plan(data)
    data.views["fc_blocks"] = M.fc_blocks(data.views["fc_edges"])
    data.views["all4"] = np.hstack([data.views["regional"], data.views["fc_summary"]])

    out = {"frozen": specs, "control": {}, "test": {}}
    log("\n[STEP 8] SHUFFLED-LABEL CONTROL (25 folds, training labels permuted)")
    for name, view in specs:
        mk = find_job(name, view)
        recs, _ = M.run_cv2(name, data.views[view], data.y, plan, mk, shuffle_labels=True)
        b = H.summarise(recs)["balanced_accuracy"]
        ok = abs(b[0] - H.CHANCE) < 0.06
        out["control"][f"{name}|{view}"] = {"bal_acc": b, "near_chance": ok}
        log(f"  {name:20s} {view:10s} shuffled bal.acc {b[0]:.3f} +/- {b[1]:.3f}  "
            f"{'OK (near chance)' if ok else 'CHECK: away from chance'}")

    # ---- step 9: the ONLY place test rows are read
    log("\n[STEP 9] HELD-OUT TEST: fit on 81 pool subjects, predict 35 test subjects, once")
    src = H.SRC
    rows = list(csv.DictReader(open(os.path.join(H.SPLITS, "split_assignment.csv"))))
    part = np.array([r["partition"] for r in rows])
    pool, test = np.flatnonzero(part == "train_pool"), np.flatnonzero(part == "test")
    y_all = np.load(os.path.join(src, "y.npy")).astype(np.int64)
    reg = np.load(os.path.join(src, "X_regional.npy")).astype(np.float64)
    fcu = np.load(os.path.join(src, "X_fc_roi_unsigned.npy")).astype(np.float64)
    need_fc = any(v in ("fc_edges", "fc_blocks") for _, v in specs)
    fc = np.load(os.path.join(src, "X_fc.npy"), mmap_mode="r") if need_fc else None
    log(f"  pool {pool.size} subjects, test {test.size} subjects, test class counts "
        + ", ".join(f"{H.STAGES[g]}={int((y_all[test] == g).sum())}" for g in range(H.N_STAGES)))

    for name, view in specs:
        X = build_view(view, reg, fcu, fc)
        mk = find_job(name, view)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipe = mk(H.SEEDS[0])
            pipe.fit(X[pool], y_all[pool])
            pred = pipe.predict(X[test]).astype(int)
        m = H.evaluate(y_all[test], pred)
        n_out = int(np.atleast_2d(pipe[:-1].transform(X[test])).shape[1])
        res = {**m, "n_features_in": int(X.shape[1]), "n_features_out": n_out}
        sel = pipe.named_steps.get("select")
        if sel is not None:
            idx = (sel.support_idx_ if hasattr(sel, "support_idx_")
                   else np.flatnonzero(sel.get_support()))
            names = M.feature_names(view)
            res["selected_features"] = [names[i] if names else int(i) for i in idx]
        cm = np.zeros((H.N_STAGES, H.N_STAGES), int)
        for a, b in zip(y_all[test], pred):
            cm[a, b] += 1
        res["confusion_matrix"] = cm.tolist()
        out["test"][f"{name}|{view}"] = res
        log(f"\n  {name} on {view}: {X.shape[1]} -> {n_out} features")
        log(f"    balanced accuracy {m['balanced_accuracy']:.3f}  accuracy {m['accuracy']:.3f}  "
            f"macro-F1 {m['macro_f1']:.3f}  MAE {m['mae_stages']:.2f} stages  QWK {m['qwk']:+.3f}")
        log("    per-class recall: " + ", ".join(f"{k} {v:.2f}" for k, v in m["per_class_recall"].items()))
        log("    confusion matrix (rows true CN..AD, cols predicted):")
        for g in range(H.N_STAGES):
            log(f"      {H.STAGES[g]:5s} {cm[g].tolist()}")
        if "selected_features" in res:
            log("    selected: " + ", ".join(map(str, res["selected_features"])))

    H.save_outputs("final_test", [], {**out, "test_set_touched": True})


if __name__ == "__main__":
    main()
