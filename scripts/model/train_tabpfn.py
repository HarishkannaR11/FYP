"""
TabPFN (v2) on the same inputs, splits and folds as the transformer.

TabPFN is a transformer pretrained on millions of synthetic tables to classify
small tabular datasets in one forward pass ("in-context learning"): the
training subjects are its context, nothing is fitted by gradient descent on
this cohort. It is built for datasets of this size (116 subjects, <= 60
features, 6 classes). The v2 weights are used because they are openly
downloadable; v2.5 and later need an interactive license login.

Inputs (all chosen inside each training set, exactly as for the transformer)
---------------------------------------------------------------------------
  tabpfn_svmrfe50                 rs-fMRI only: the 50 SVM-RFE features
  tabpfn_svmrfe50_comorb8         + the 8 comorbidities (model A inputs)
  tabpfn_svmrfe50_clin10          + 8 comorbidities + MMSE + CDR-SB (model B inputs)
  tabpfn_comorb8                  the 8 comorbidities only
  tabpfn_svmrfe50_comorb8_SHUFFLED  training labels permuted (cv designs only)

SVM-RFE is train_transformer_svmrfe.fit_selection with the same seed per fold,
so every fold uses the same 50 features as the transformer and the logistic
regression. Raw feature values are passed: TabPFN does its own
preprocessing and handles missing values, and it only ever sees the training
subjects as context.

Designs: split70 (81/35), cv116 (5 folds x 5 seeds), cv116k10 (10 folds x 5 seeds).

Run:  python scripts/model/train_tabpfn.py --smoke
      python scripts/model/train_tabpfn.py --designs split70 cv5 cv10
"""
import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_transformer_comorbidity as C  # noqa: E402  (sets thread env vars first)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tabpfn import TabPFNClassifier  # noqa: E402
from tabpfn.constants import ModelVersion  # noqa: E402

from train_transformer import evaluate, load_all  # noqa: E402

S, H = C.S, C.H
ROOT = C.ROOT
OUT = os.path.join(ROOT, "derivatives", "model_training", "tabpfn")
INPUTS = {"tabpfn_svmrfe50": (True, None),
          "tabpfn_svmrfe50_comorb8": (True, "comorbidity8"),
          "tabpfn_svmrfe50_clin10": (True, "clinical10"),
          "tabpfn_comorb8": (False, "comorbidity8"),
          "tabpfn_custom5": (False, "custom5"),                 # MMSE, CDR-SB, GDS, APOE4, cardiovascular
          "tabpfn_svmrfe50_custom5": (True, "custom5"),
          "tabpfn_svmrfe50_comorb8_SHUFFLED": (True, "comorbidity8")}

_lines = []


def log(m=""):
    print(m, flush=True)
    _lines.append(str(m))


def run_split(cand, data, clin_sets, tr, te, seed, fold, fp, train_acc=True):
    X4, flat, fc, y, subjects, mask = data
    assert not set(tr.tolist()) & set(te.tolist())
    use_brain, clin_key = INPUTS[cand]
    shuffled = cand.endswith("_SHUFFLED")
    y_used = y.copy()
    if shuffled:
        ytr = y[tr].copy()
        np.random.default_rng(seed * 100 + fold).shuffle(ytr)
        y_used[tr] = ytr
    t0 = time.time()
    rec = {"candidate": cand, "seed": seed, "fold": fold, "fold_fingerprint": fp,
           "n_train": int(tr.size), "n_test": int(te.size), "shuffled_labels": shuffled,
           "test_index": [int(i) for i in te], "subjects": [str(subjects[i]) for i in te],
           "true": [int(v) for v in y[te]]}
    parts = []
    if use_brain:
        _, support, _ = S.fit_selection(flat, y_used, tr, seed + fold)    # same 50 as the transformer
        parts.append(flat[:, support])
        rec["support"] = support.tolist()
    if clin_key:
        parts.append(clin_sets[clin_key])
    X = np.hstack(parts).astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu",
                                                            random_state=seed + fold)
        model.fit(X[tr], y_used[tr])                                     # TRAINING SUBJECTS ONLY
        p_te = model.predict_proba(X[te])
        # training subjects are TabPFN's own context, so this is near 100% by design
        p_tr = model.predict_proba(X[tr]) if train_acc else None
    classes = list(model.classes_)
    if classes != list(range(S.N_STAGES)):                              # a stage absent from training
        full = np.zeros((len(te), S.N_STAGES))
        full[:, classes] = p_te
        p_te = full
        if p_tr is not None:
            full = np.zeros((len(tr), S.N_STAGES))
            full[:, classes] = p_tr
            p_tr = full
    pred = p_te.argmax(1)
    rec["train_balanced_accuracy"] = (H.evaluate(y_used[tr], p_tr.argmax(1))["balanced_accuracy"]
                                      if p_tr is not None else None)
    rec["n_features"] = int(X.shape[1])
    m = evaluate(np.asarray(p_te), y[te], pred=pred)
    rec.update({k: m[k] for k in S.METRIC_KEYS})
    rec["pred"] = [int(v) for v in pred]
    rec["probs"] = S.r6(p_te)
    rec["seconds"] = round(time.time() - t0, 1)
    log(f"    seed {seed} fold {fold}  features {X.shape[1]}  test bal.acc {m['balanced_accuracy']:.3f}  "
        f"acc {m['accuracy']:.3f}  train {rec['train_balanced_accuracy'] if p_tr is None else round(rec['train_balanced_accuracy'], 3)}  "
        f"{rec['seconds']:.0f}s")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", nargs="+", choices=["split70", "cv5", "cv10"], default=["split70", "cv5", "cv10"])
    ap.add_argument("--candidates", nargs="+", choices=list(INPUTS), default=list(INPUTS))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="one CV fold per input, nothing saved")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(OUT, exist_ok=True)

    data = S.load_cohort()
    y, subjects = data[3], data[4]
    clin_sets = {k: C.load_clinical(y, subjects, C.FEATURE_SETS[k])[0]
                 for k in ("comorbidity8", "clinical10", "custom5")}
    _, _, y2, _, _, pool, test, _, smeta = load_all()
    assert np.array_equal(y2, y) and smeta["split_fingerprint"] == H.EXPECTED_SPLIT_FINGERPRINT
    plans = {"cv5": S.cv_plan(y, subjects, 5), "cv10": S.cv_plan(y, subjects, 10)}

    if args.smoke:
        fp, folds = plans["cv5"][H.SEEDS[0]]
        tr, te = folds[0]
        for cand in args.candidates:
            run_split(cand, data, clin_sets, tr, te, H.SEEDS[0], 0, fp)
        log("SMOKE OK")
        return

    for design in args.designs:
        dname = {"split70": "split70", "cv5": "cv116", "cv10": "cv116k10"}[design]
        for cand in args.candidates:
            if design == "split70" and cand.endswith("_SHUFFLED"):
                continue
            if design == "cv10" and cand.endswith("_SHUFFLED"):
                continue
            stem = os.path.join(OUT, f"{dname}_{cand}")
            if os.path.exists(stem + ".json"):
                print(f"COLLISION: {dname}_{cand}.json exists; skipped, not overwritten")
                continue
            _lines.clear()
            log("=" * 100)
            log(f"TabPFN v2 -- design {dname}, candidate {cand}")
            log("=" * 100)
            t0 = time.time()
            records = []
            if design == "split70":
                log(f"split fingerprint {smeta['split_fingerprint']}: train {pool.size} / test {test.size}")
                records.append(run_split(cand, data, clin_sets, pool, test, C.SPLIT70_SEED, 0,
                                         smeta["split_fingerprint"]))
                extra = {"design": dname, "split_fingerprint": smeta["split_fingerprint"]}
            else:
                plan = plans[design]
                log("fold fingerprints " + json.dumps({s: plan[s][0] for s in H.SEEDS}))
                for seed in H.SEEDS:
                    fp, folds = plan[seed]
                    for f, (tr, te) in enumerate(folds):
                        records.append(run_split(cand, data, clin_sets, tr, te, seed, f, fp,
                                                 train_acc=design == "cv5"))
                s = H.summarise(records)
                log(f"  {cand}: seed-level bal.acc {s['balanced_accuracy'][0]:.3f} +/- "
                    f"{s['balanced_accuracy'][1]:.3f}  acc {s['accuracy'][0]:.3f}")
                extra = {"design": dname, "n_folds": 5 if design == "cv5" else 10,
                         "fold_fingerprints": {str(s_): plan[s_][0] for s_ in H.SEEDS}}
            extra.update({"candidate": cand, "model": "TabPFN v2 (tabpfn 9.2.0, ModelVersion.V2)",
                          "seconds": round(time.time() - t0, 1)})
            log(f"  elapsed {time.time() - t0:.0f}s")
            json.dump({"extra": extra, "records": records}, open(stem + ".json", "w"))
            open(stem + ".log", "w", encoding="utf-8").write("\n".join(_lines) + "\n")
            print(f"  wrote {os.path.relpath(stem, ROOT)}.json")


if __name__ == "__main__":
    main()
