"""
Clinical reference for v5 (PLAN.md, step 4): TabPFN v2 on MMSE + CDR-SB only, on the project's 25 outer splits.

This is NOT an fMRI result: MMSE and CDR are part of ADNI's diagnostic criteria. TabPFN settings are the baseline's
(V2 defaults, CPU, random_state = seed + fold). Missing CDR-SB values (2 subjects) are passed as NaN and handled by
TabPFN's own preprocessing, which uses only its training context. Outer-test labels are hidden (-1) from the fit.

Output: derivatives/experiments/controlled_comparison_v5/clinical_reference/grid/cv5_seed{s}_fold{f}.json (written once)
Run:  python scripts/experiments/controlled_comparison_v5/run_mmse_cdr_tabpfn.py
"""
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "biomarker_ablation_v3"))
import run_tabpfn_configs as T  # noqa: E402  (sets thread env vars first)
import train_transformer_comorbidity as CM  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

EXP = os.path.join(T.ROOT, "derivatives", "experiments", "controlled_comparison_v5", "clinical_reference")
COLS = ["MMSE_total", "CDR_SOB"]


def main():
    torch.set_num_threads(4)
    own = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]
    d = T.S.load_cohort()
    y, subjects = d[3], d[4]
    C = CM.load_clinical(y, subjects, COLS)[0].astype(np.float32)
    plan = T.S.cv_plan(y, subjects, 5)
    base = {(r["seed"], r["fold"]): r["test_index"] for r in json.load(open(T.TABPFN_BASELINE))["records"]}
    for seed in T.H.SEEDS:
        fp, folds = plan[seed]
        for f, (tr, te) in enumerate(folds):
            assert te.tolist() == base[(seed, f)], "outer split differs from the TabPFN baseline"
            path = os.path.join(EXP, "grid", f"cv5_seed{seed}_fold{f}.json")
            if os.path.exists(path):
                continue
            t0 = time.time()
            y_sel = T.hide(y, te)
            p = T.tabpfn_probs(C, y_sel, tr, te, seed + f)            # never reads y_sel[te]
            heldout, pb = T.score_record(y, te, {"probs": p, "train_counts": np.bincount(y_sel[tr], minlength=T.NS)})
            rec = {"config": "TabPFN_MMSE_CDRSB_only", "columns": COLS, "seed": seed, "fold": f, "fold_fingerprint": fp,
                   "script_sha256": own, "test_idx": te.tolist(), "true": y[te].tolist(), "pred": p.argmax(1).tolist(),
                   "probs": T.S.r6(p), "heldout": heldout, "n_missing_train": int(np.isnan(C[tr]).any(1).sum()),
                   "seconds": round(time.time() - t0, 1)}
            os.makedirs(os.path.dirname(path), exist_ok=True)
            json.dump(rec, open(path + ".tmp", "w"))
            os.replace(path + ".tmp", path)
            print(f"seed {seed} fold {f}: {rec['seconds']:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
