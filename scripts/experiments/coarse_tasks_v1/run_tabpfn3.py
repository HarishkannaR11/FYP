"""
M1, C2, C3 of the pre-declared three-group task (docs/NEW_EXPERIMENT_PREDECLARATION.md).
Needs torch, tabpfn 9.2.0 and the TabPFN v2 weights (downloaded from Hugging Face on first use), i.e. the
authors' PC. Reuses the project's own selection and cohort loader so the pipeline is identical to
scripts/model/train_tabpfn.py, except that SVM-RFE is refitted on the THREE-group labels inside each outer
training set. The outer folds are asserted identical (test indices and fingerprints) to the stored six-class run.

  M1  tabpfn3_M1          true three-group labels
  C2  tabpfn3_C2_GLOBAL   training labels shuffled globally
  C3  tabpfn3_C3_WITHIN_SITE  training labels shuffled within scanner site

Run:  python scripts/experiments/coarse_tasks_v1/run_tabpfn3.py --smoke
      python scripts/experiments/coarse_tasks_v1/run_tabpfn3.py
It checkpoints after every outer split (resumable) and never overwrites a finished result.
"""
import argparse
import json
import os
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "model"))
import common3 as K  # noqa: E402
import train_transformer_comorbidity as C  # noqa: E402  (sets thread env vars first)

import numpy as np  # noqa: E402
import torch  # noqa: E402
from tabpfn import TabPFNClassifier  # noqa: E402
from tabpfn.constants import ModelVersion  # noqa: E402

S = C.S
CANDIDATES = ["M1", "C2_GLOBAL", "C3_WITHIN_SITE"]


def shuffled_labels(cand, y3, site, tr, seed, fold):
    ytr = y3[tr].copy()
    rng = np.random.default_rng(seed * 100 + fold)
    if cand == "C2_GLOBAL":
        rng.shuffle(ytr)
    elif cand == "C3_WITHIN_SITE":
        for s in np.unique(site[tr]):
            m = site[tr] == s
            sub = ytr[m].copy(); rng.shuffle(sub); ytr[m] = sub
    y_used = y3.copy(); y_used[tr] = ytr
    return y_used


def run_split(cand, flat, y3, site, subjects, tr, te, seed, fold, fp):
    assert not set(tr.tolist()) & set(te.tolist())
    y_used = y3 if cand == "M1" else shuffled_labels(cand, y3, site, tr, seed, fold)
    t0 = time.time()
    _, support, _ = S.fit_selection(flat, y_used, tr, seed + fold)      # SVM-RFE on the 3-group labels, training only
    X = flat[:, support].astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu", random_state=seed + fold)
        model.fit(X[tr], y_used[tr])                                    # TRAINING SUBJECTS ONLY
        p = model.predict_proba(X[te])
    classes = list(model.classes_)
    if classes != list(range(K.NG)):
        full = np.zeros((len(te), K.NG)); full[:, classes] = p; p = full
    pred = p.argmax(1)
    return {"candidate": cand, "seed": seed, "fold": fold, "fold_fingerprint": fp, "test_index": [int(i) for i in te],
            "subjects": [str(subjects[i]) for i in te], "true": [int(v) for v in y3[te]], "pred": [int(v) for v in pred],
            "probs": np.round(p, 5).tolist(), "support": [int(i) for i in support],
            "balanced_accuracy": K.bal(y3[te], pred), "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", nargs="+", choices=CANDIDATES, default=CANDIDATES)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="one split per candidate, nothing saved")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(K.OUT, exist_ok=True)

    data = S.load_cohort()
    _, flat, _, y6, subjects, _ = data
    _, y3, subj_chk, site = K.load_cohort()
    assert np.array_equal(subjects, subj_chk)
    stored = K.load_folds()
    plan = S.cv_plan(y6, subjects, 5)
    for seed in K.SEEDS:                                                  # folds identical to every other experiment
        fp, folds = plan[seed]
        for f, (tr, te) in enumerate(folds):
            s_tr, s_te, s_fp = stored[(seed, f)]
            assert np.array_equal(np.sort(te), np.sort(s_te)) and fp == s_fp, "fold mismatch with stored run"

    for cand in args.candidates:
        stem = os.path.join(K.OUT, f"tabpfn3_{cand}")
        if os.path.exists(stem + ".json"):
            print(f"COLLISION: tabpfn3_{cand}.json exists; skipped, not overwritten"); continue
        part = stem + ".partial.json"
        records = json.load(open(part)) if os.path.exists(part) and not args.smoke else []
        done = {(r["seed"], r["fold"]) for r in records}
        t0 = time.time()
        for seed in K.SEEDS:
            fp, folds = plan[seed]
            for f, (tr, te) in enumerate(folds):
                if (seed, f) in done:
                    continue
                rec = run_split(cand, flat, y3, site, subjects, tr, te, seed, f, fp)
                print(f"  {cand} seed {seed} fold {f}  3-group bal.acc {rec['balanced_accuracy']:.3f}  {rec['seconds']:.0f}s", flush=True)
                if args.smoke:
                    print("SMOKE OK"); return
                records.append(rec); json.dump(records, open(part, "w"))
        json.dump({"extra": {"candidate": cand, "declared": "docs/NEW_EXPERIMENT_PREDECLARATION.md",
                             "model": "TabPFN v2 (tabpfn 9.2.0, ModelVersion.V2)", "groups": K.GROUP_NAMES,
                             "seconds": round(time.time() - t0, 1)}, "records": records}, open(stem + ".json", "w"))
        os.remove(part)
        print("wrote", os.path.relpath(stem, K.ROOT) + ".json")


if __name__ == "__main__":
    main()
