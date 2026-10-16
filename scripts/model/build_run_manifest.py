"""
Step 4: run manifest.

Collects everything needed to reproduce or audit a training run: dataset
identity, the exact subject IDs in each split, stage mapping, preprocessing,
model configuration, seeds, package versions, output paths, metrics and
failure reasons. Written from the artifacts on disk, not from memory.
"""
import csv
import hashlib
import json
import os
import platform
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "derivatives", "model_ready")
SPLITS = os.path.join(SRC, "splits")
TRAIN = os.path.join(ROOT, "derivatives", "model_training")
STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]


def sha256(path, chunk=1 << 16):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()[:16]


def versions():
    out = {"python": sys.version.split()[0], "platform": platform.platform()}
    for mod in ("numpy", "scipy", "torch", "nibabel", "matplotlib"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:                                   # noqa: BLE001
            out[mod] = "not installed"
    return out


def main():
    y = np.load(os.path.join(SRC, "y.npy"))
    subjects = np.load(os.path.join(SRC, "subjects.npy")).astype(str)
    meta = json.load(open(os.path.join(SRC, "metadata.json")))
    smeta = json.load(open(os.path.join(SPLITS, "split_metadata.json")))
    srows = list(csv.DictReader(open(os.path.join(SPLITS, "split_assignment.csv"))))

    pool = [r["subject_id"] for r in srows if r["partition"] == "train_pool"]
    test = [r["subject_id"] for r in srows if r["partition"] == "test"]
    folds = {}
    for f in range(smeta["n_folds"]):
        folds[f"fold_{f}"] = {
            "validation_subjects": [r["subject_id"] for r in srows
                                    if r["partition"] == "train_pool"
                                    and r["cv_fold"] == str(f)],
        }

    arrays = {}
    for name in ("X_regional.npy", "X_fc.npy", "X_fc_roi_unsigned.npy",
                 "y.npy", "subjects.npy", "local_attention_mask.npy"):
        p = os.path.join(SRC, name)
        if os.path.exists(p):
            a = np.load(p, mmap_mode="r")
            arrays[name] = {"shape": list(a.shape), "dtype": str(a.dtype),
                            "sha256_16": sha256(p),
                            "bytes": os.path.getsize(p)}

    runs = {}
    for fn in sorted(os.listdir(TRAIN)) if os.path.isdir(TRAIN) else []:
        if not (fn.startswith("results_") and fn.endswith(".json")):
            continue
        tag = fn[len("results_"):-len(".json")]
        try:
            r = json.load(open(os.path.join(TRAIN, fn)))
        except Exception as e:                              # noqa: BLE001
            runs[tag] = {"status": "unreadable", "error": str(e)}
            continue
        if tag == "baselines":
            runs[tag] = {"status": "complete", "kind": "baselines",
                         "models": sorted(r["baselines"].keys()),
                         "split_fingerprint": r.get("split_fingerprint")}
            continue
        cv = r.get("cv", {})
        summ = cv.get("summary", {})
        folds_done = len(cv.get("folds", []))
        complete = folds_done == smeta["n_folds"]
        runs[tag] = {
            "status": "complete" if complete else f"incomplete ({folds_done} folds)",
            "kind": "transformer",
            "config": r.get("config"),
            "split_fingerprint": r.get("split_fingerprint"),
            "seeds_per_fold": [m.get("seed") for m in cv.get("folds", [])],
            "stop_reasons": [m.get("stop_reason") for m in cv.get("folds", [])],
            "best_epochs": cv.get("best_epochs"),
            "degenerate": summ.get("degenerate"),
            "n_predicted_classes_per_fold": summ.get("n_predicted_classes_per_fold"),
            "cv_balanced_accuracy": summ.get("balanced_accuracy"),
            "cv_macro_f1": summ.get("macro_f1"),
            "cv_qwk": summ.get("qwk"),
            "cv_mae_stages": summ.get("mae_stages"),
            "test_evaluated": r.get("test") is not None,
            "artifacts": {
                "results_json": os.path.relpath(os.path.join(TRAIN, fn), ROOT),
                "log": os.path.relpath(os.path.join(TRAIN, f"train_{tag}.log"), ROOT)
                if os.path.exists(os.path.join(TRAIN, f"train_{tag}.log")) else None,
                "test_predictions": os.path.relpath(
                    os.path.join(TRAIN, f"test_predictions_{tag}.csv"), ROOT)
                if os.path.exists(os.path.join(TRAIN,
                                               f"test_predictions_{tag}.csv")) else None,
            },
        }

    manifest = {
        "generated_from": "artifacts on disk",
        "dataset": {
            "source_directory": meta.get("source"),
            "n_subjects": int(len(subjects)),
            "n_unique_subjects": int(len(set(subjects.tolist()))),
            "acquisitions_per_subject": 1,
            "class_counts": {s: int((y == i).sum()) for i, s in enumerate(STAGES)},
            "atlas": "Brainnetome 246",
            "biomarkers": ["mALFF", "mReHo", "DC_z", "FC"],
            "arrays": arrays,
        },
        "stage_mapping": {
            "stage_to_index": meta["stage_to_index"],
            "stage_order": meta["stage_order"],
            "note": "preserved from the repository; NOT modified. The order "
                    "asserted in one task brief (LMCI before MCI) does not "
                    "match this encoding and was not applied.",
        },
        "split": {
            "fingerprint": smeta["split_fingerprint"],
            "seed": smeta["seed"],
            "test_fraction": smeta["test_fraction"],
            "n_folds": smeta["n_folds"],
            "unit": "subject",
            "stratified_by": "diagnostic stage",
            "train_pool_subjects": pool,
            "held_out_test_subjects": test,
            "folds": folds,
            "test_set_policy": smeta["test_set_policy"],
        },
        "preprocessing": {
            "within_acquisition": {
                "mALFF": "ratio to that acquisition's own 246-ROI mean",
                "mReHo": "ratio to that acquisition's own 246-ROI mean",
                "DC_z": "z-score across the 246 ROIs of that acquisition",
                "FC": "Fisher-z, signed, unthresholded, diagonal zeroed",
                "leakage": "none by construction -- no cross-subject statistic, "
                           "no label",
            },
            "cross_subject": {
                "scaler": "per (ROI, channel) standardisation of the regional "
                          "features",
                "fitted_on": "training fold only (CV) / training pool only (final)",
                "fc_rescaled": False,
            },
            "class_weighting": "inverse class frequency, computed on the "
                               "training fold",
        },
        "local_attention_mask": json.load(
            open(os.path.join(SRC, "local_attention_mask.json")))
        if os.path.exists(os.path.join(SRC, "local_attention_mask.json")) else None,
        "runs": runs,
        "software": versions(),
        "scripts": {
            "splits": "scripts/model/make_splits.py",
            "mask": "scripts/model/build_local_attention_mask.py",
            "training": "scripts/model/train_transformer.py",
            "baselines": "scripts/model/baselines.py",
            "heads": "scripts/model/heads.py",
            "encoder": "scripts/model/multi_attention_transformer.py",
            "audit": "scripts/model/audit_pipeline.py",
            "ordinal_diagnosis": "scripts/model/diagnose_ordinal.py",
            "protocol_validation": "scripts/model/validate_training_protocol.py",
            "tests": ["scripts/model/test_multi_attention_transformer.py",
                      "scripts/model/test_heads.py"],
        },
        "claims_not_supported": [
            "no meta-learning is implemented",
            "no comorbidity fusion is implemented",
            "uncertainty is MC-dropout, not a full Bayesian posterior",
            "LMCI has no MMSE/CDR/GDS/age/medical-history data (0/20)",
            "no clinical readiness or progression-prediction claim is made",
        ],
    }

    p = os.path.join(TRAIN, "run_manifest.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {os.path.relpath(p, ROOT)}")
    print(f"  dataset  : {manifest['dataset']['n_subjects']} subjects, "
          f"{len(arrays)} arrays hashed")
    print(f"  split    : fingerprint {smeta['split_fingerprint']}, "
          f"{len(pool)} pool / {len(test)} test")
    complete = [k for k, v in runs.items() if v.get("status") == "complete"]
    incomplete = [k for k, v in runs.items() if v.get("status") != "complete"]
    print(f"  runs     : {len(complete)} complete, {len(incomplete)} incomplete")
    if incomplete:
        print(f"             incomplete: {incomplete}")
    tested = [k for k, v in runs.items() if v.get("test_evaluated")]
    print(f"  test set : evaluated by {len(tested)} run(s) {tested}")


if __name__ == "__main__":
    main()
