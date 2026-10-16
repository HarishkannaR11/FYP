"""
Phase 6A: biomarker ablation on the frozen reference architecture.

Feature mapping, verified in Phase A rather than assumed:

    the 984-D vector is 246 ROIs x 4 channels, INTERLEAVED:
        index = roi * 4 + channel
        channel 0 mALFF   246 values, one per ROI
        channel 1 mReHo   246 values, one per ROI
        channel 2 DC_z    246 values, one per ROI
        channel 3 FCstr   246 values, one per ROI

Channel 3 is NOT the connectivity matrix. It is mean_{j!=i} |z_ij|, a 246-D
SUMMARY that compresses 30,135 unique edges into 246 numbers. "FC only"
therefore has two defensible meanings and both are evaluated:

    fc_summary   the 246-D strength vector the frozen model actually uses
    fc_edges     the 30,135-D upper triangle of the connectivity matrix

Architecture is held fixed at the frozen reference (MLP, two hidden layers of
width 64, LayerNorm+GELU+dropout 0.3, multiclass head). Only the input width
changes, which is unavoidable when the number of input features changes and is
recorded per configuration.

Same five split seeds, same five folds per seed, same fold assignments, same
training-fold-only preprocessing as the repeated-CV experiment, so every
comparison is paired. The held-out test set is never loaded.
"""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repeated_cv import SEEDS, N_FOLDS, stratified_folds, fold_fingerprint, run_mlp
from train_transformer import N_STAGES, STAGES, evaluate, load_all

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "derivatives", "model_ready")
OUT = os.path.join(ROOT, "derivatives", "model_training", "biomarker_ablation")
torch.set_num_threads(1)

_log = []


def log(m=""):
    print(m, flush=True)
    _log.append(str(m))


def main():
    os.makedirs(OUT, exist_ok=True)
    reg, fc, y, subjects, mask, pool, test, fold0, smeta = load_all()
    fcu = np.load(os.path.join(SRC, "X_fc_roi_unsigned.npy")).astype(np.float64)
    reg = reg.astype(np.float64)
    iu = np.triu_indices(246, k=1)

    CONFIGS = {
        "alff_only":    reg[:, :, 0],
        "reho_only":    reg[:, :, 1],
        "dc_only":      reg[:, :, 2],
        "fc_summary":   fcu,
        "fc_edges":     np.asarray(fc[:, iu[0], iu[1]], dtype=np.float64),
        "all_four":     np.concatenate([reg, fcu[:, :, None]], axis=2
                                       ).reshape(len(y), -1),
    }

    log("=" * 86)
    log("PHASE 6B -- BIOMARKER ABLATION")
    log("=" * 86)
    log(f"training pool {pool.size}; held-out test ({test.size}) NOT loaded")
    log(f"architecture fixed: MLP h64 multiclass; only the input width varies")
    log("")
    log(f"{'configuration':16s} {'input dim':>10s}  meaning")
    MEANING = {
        "alff_only": "246 ROIs, mALFF (ratio to within-acquisition mean)",
        "reho_only": "246 ROIs, mReHo (ratio to within-acquisition mean)",
        "dc_only": "246 ROIs, DC_z (signed degree centrality, z across ROIs)",
        "fc_summary": "246 ROIs, mean_{j!=i} |Fisher-z| (SUMMARY of the matrix)",
        "fc_edges": "30,135 unique edges, upper triangle of the Fisher-z matrix",
        "all_four": "246 x 4 interleaved (mALFF, mReHo, DC_z, FCstr)",
    }
    for k, v in CONFIGS.items():
        log(f"{k:16s} {v.shape[1]:10d}  {MEANING[k]}")
        assert np.isfinite(v).all(), f"{k} has non-finite values"

    records = []
    for seed in SEEDS:
        fo = stratified_folds(y, pool, N_FOLDS, np.random.default_rng(seed))
        fp = fold_fingerprint(subjects, fo, pool)
        log(f"\nseed {seed}  fold fingerprint {fp}")
        for name, X in CONFIGS.items():
            t0 = time.time()
            for f in range(N_FOLDS):
                va = np.array([i for i in pool if fo[int(i)] == f])
                tr = np.array([i for i in pool if fo[int(i)] != f])
                assert not (set(tr.tolist()) & set(va.tolist()))
                p, d, be, npar = run_mlp("multiclass", X, y, tr, va,
                                         seed + f, hidden=64)
                m = evaluate(p, y[va], pred=d)
                records.append({
                    "config": name, "input_dim": int(X.shape[1]),
                    "seed": seed, "fold": f, "fold_fingerprint": fp,
                    "best_epoch": be, "n_parameters": npar,
                    **{k: m[k] for k in ("accuracy", "balanced_accuracy",
                                         "macro_f1", "mae_stages", "qwk",
                                         "n_predicted_classes")},
                    "per_class_recall": m["per_class_recall"],
                    "per_class_f1": m["per_class_f1"],
                    "confusion_matrix": m["confusion_matrix"],
                    "predicted_distribution": m["predicted_distribution"],
                    "subjects": [subjects[i] for i in va],
                    "true": [int(v) for v in y[va]],
                    "pred": [int(v) for v in d],
                    "probs": [[round(float(x), 6) for x in r] for r in p],
                })
            sub = [r for r in records if r["config"] == name and r["seed"] == seed]
            log(f"  {name:14s} bal.acc {np.mean([r['balanced_accuracy'] for r in sub]):.3f}"
                f"  macroF1 {np.mean([r['macro_f1'] for r in sub]):.3f}"
                f"  MAE {np.mean([r['mae_stages'] for r in sub]):.3f}"
                f"  QWK {np.mean([r['qwk'] for r in sub]):+.3f}"
                f"  cls {[r['n_predicted_classes'] for r in sub]}"
                f"  {time.time() - t0:.0f}s")

    p = os.path.join(OUT, "ablation_records.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"feature_mapping": {
            "layout": "index = roi * 4 + channel (interleaved, C order)",
            "channels": {0: "mALFF", 1: "mReHo", 2: "DC_z",
                         3: "FC unsigned mean strength"},
            "fc_caveat": "channel 3 is a 246-D SUMMARY of the 246x246 matrix "
                         "(mean_{j!=i} |z_ij|); 30,135 unique edges are "
                         "compressed to 246 values",
            "meanings": MEANING},
            "design": {"seeds": SEEDS, "n_folds": N_FOLDS,
                       "architecture": "MLP h64 multiclass (frozen reference)",
                       "test_set_touched": False},
            "records": records}, f, indent=2)
    log(f"\nwrote {os.path.relpath(p, ROOT)}  ({len(records)} fold evaluations)")
    with open(os.path.join(OUT, "ablation.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")


if __name__ == "__main__":
    main()
