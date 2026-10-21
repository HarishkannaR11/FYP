"""
Locked evaluation on the project's 70/30 test set (81 train / 35 test).

The locked "configuration" is the whole nested-selection PROCEDURE, fixed in
common.py before this script is run:
  candidate grid  LR / RBF-SVM / ordinal LR at every k in K_GRID, both
                  harmonisation settings, every hyperparameter; TabPFN at
                  k in TABPFN_NESTED_K without ComBat and k = 50 with ComBat
  selection       3-fold inner CV on the 81 training subjects only, highest
                  mean balanced accuracy (common.select_best tie-break)
  final fit       ComBat (if chosen) and SVM-RFE refitted on all 81, then the
                  chosen model; predict the 35 test subjects once

Run for the pre-declared primary input set (imaging + 8 comorbidities) and the
secondary set (imaging only). No other test evaluation is made. The 35 test
subjects were scored in earlier, unrelated experiments of this project, so
this test set is not pristine; nothing in this procedure was chosen from those
scores.

Run:  python scripts/experiments/leakage_safe_v1/run_test.py
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as K  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from train_transformer import load_all  # noqa: E402


def nested_scores(d, labels, tr, inputs, seed_tag):
    inner = K.inner_folds(labels, tr, seed_tag, 0)
    scores = {}
    for harm in K.HARMS:
        inn = []
        for itr, iva in inner:
            Xi, _ = K.harmonise(d, harm, itr)
            sup = {k: K.rfe_support(Xi, labels, itr, k, seed_tag) for k in K.K_GRID if k < 984}
            sup[984] = np.arange(984)
            inn.append((itr, iva, Xi, sup))
        for k in K.K_GRID:
            for fam in K.LIGHT:
                for hp in K.hp_list(fam):
                    s = []
                    for itr, iva, Xi, sup in inn:
                        Xd = K.design(Xi, d, inputs, sup[k])
                        p, _, _ = K.fit_predict(fam, hp, Xd[itr], labels[itr], Xd[iva], 0)
                        s.append(K.bal_acc(labels[iva], p))
                    scores[K.cid(fam, inputs, f"k{k}", harm, hp)] = float(np.mean(s))
            if (harm == "none" and k in K.TABPFN_NESTED_K) or (harm == "combat" and k == 50):
                s = []
                for j, (itr, iva, Xi, sup) in enumerate(inn):
                    Xd = K.design(Xi, d, inputs, sup[k])
                    p, _, _ = K.fit_predict("tabpfn", "-", Xd[itr], labels[itr], Xd[iva], seed_tag + 100 * (j + 1))
                    s.append(K.bal_acc(labels[iva], p))
                scores[K.cid("tabpfn", inputs, f"k{k}", harm, "-")] = float(np.mean(s))
    return scores


def main():
    # Guard added 2026-10-21: the user asked to preserve the locked test set and evaluate it at most once,
    # with the final model chosen by cross-validation. Without the explicit flag, nothing is read or scored.
    if "--confirm-final-test" not in sys.argv:
        print("run_test.py: test evaluation is GATED (pass --confirm-final-test to run it). "
              "No test subject was read or scored.")
        return
    torch.set_num_threads(4)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    d = K.load()
    y = d["y"]
    _, _, _, _, _, pool, test, _, smeta = load_all()
    assert smeta["split_fingerprint"] == K.H.EXPECTED_SPLIT_FINGERPRINT
    seed_tag = K.C.SPLIT70_SEED
    written = []
    for inputs, role in (("imgcom", "primary"), ("img", "secondary")):
        path = os.path.join(K.EXP, "results", f"test_locked_{inputs}.json")
        if os.path.exists(path):
            print(f"{path} exists; skipped")
            continue
        t0 = time.time()
        scores = nested_scores(d, y, pool, inputs, seed_tag)
        best = K.select_best(scores)
        fam, _, view, harm, hp = K.parse_cid(best)
        k = int(view[1:])
        X, _ = K.harmonise(d, harm, pool)
        sup = K.rfe_support(X, y, pool, k, seed_tag) if k < 984 else np.arange(984)
        Xd = K.design(X, d, inputs, sup)
        pred, probs, _ = K.fit_predict(fam, hp, Xd[pool], y[pool], Xd[test], seed_tag)
        m = K.H.evaluate(y[test], pred)
        n_ok = int((pred == y[test]).sum())
        lo, hi = K.bootstrap_ci(y[test], [pred])
        rec = {"inputs": inputs, "role": role, "split_fingerprint": smeta["split_fingerprint"],
               "selected": best, "inner_scores": dict(sorted(scores.items(), key=lambda kv: -kv[1])),
               "test_subjects": [str(d["subjects"][i]) for i in test], "true": y[test].tolist(),
               "pred": pred.tolist(), "probs": K.r6(probs), "metrics": m,
               "accuracy_correct": n_ok, "accuracy_wilson95": K.wilson_ci(n_ok, len(test)),
               "balanced_accuracy_boot95": [lo, hi], "seconds": round(time.time() - t0, 1)}
        K.write_once(path, rec)
        written.append(os.path.relpath(path, K.ROOT))
        print(f"{inputs} ({role}): selected {best} (inner {scores[best]:.3f}); test accuracy {m['accuracy']:.3f} "
              f"({n_ok}/{len(test)}), balanced {m['balanced_accuracy']:.3f} [{lo:.3f}, {hi:.3f}]")
    K.manifest_entry("python scripts/experiments/leakage_safe_v1/run_test.py", started, written,
                     note="locked nested procedure, one evaluation per input set")


if __name__ == "__main__":
    main()
