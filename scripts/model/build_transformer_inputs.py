"""
Assemble the selected dataset into model-ready tensors for the
Multi-Attention Transformer.

Reads derivatives/phase1_selected_20/ (read-only) and writes a single
bundle to derivatives/model_ready/. No training, no normalisation, no
feature engineering -- the arrays are the ones the pipeline already
produced, stacked in a fixed, documented order.

Produced arrays
    X_regional  [N, 246, 3]    float32   channels = (mALFF, mReHo, DC_z)
    X_fc        [N, 246, 246]  float32   Fisher-z, signed, diagonal 0
    y           [N]            int64     ordinal stage, CN=0 .. AD=5
    subjects    [N]            str       for subject-grouped CV
    stages      [N]            str
    acq_ids     [N]            str       subject|session|run

NOT produced: the local attention mask. The transformer takes it as an
input and this script will not invent anatomical neighbourhoods. See
--build-mask in scripts/model/build_local_mask.py if you want one derived
from the atlas itself.
"""
import csv
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "derivatives", "phase1_selected_20")
OUT = os.path.join(ROOT, "derivatives", "model_ready")

# FIXED progression order. Never reorder: the ordinal head depends on it.
STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
STAGE_IDX = {s: i for i, s in enumerate(STAGES)}
N_ROI = 246
CHANNELS = ("mALFF", "mReHo", "DC_z")
FILES = ("malff.npy", "mreho.npy", "dc_z.npy")


def log(m=""):
    print(m, flush=True)


def discover():
    """Every acquisition in the selected dataset, in deterministic order."""
    rows = []
    for st in STAGES:
        base = os.path.join(SRC, st)
        if not os.path.isdir(base):
            continue
        for sub in sorted(os.listdir(base)):
            sp = os.path.join(base, sub)
            if not os.path.isdir(sp):
                continue
            for ses in sorted(os.listdir(sp)):
                for run in sorted(os.listdir(os.path.join(sp, ses))):
                    d = os.path.join(sp, ses, run)
                    if os.path.isdir(os.path.join(d, "normalized")):
                        rows.append((st, sub, ses, run, d))
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = discover()
    n = len(rows)
    log(f"acquisitions found: {n}")
    if not n:
        raise SystemExit("nothing to assemble")

    X_reg = np.zeros((n, N_ROI, len(FILES)), dtype=np.float32)
    X_fc = np.zeros((n, N_ROI, N_ROI), dtype=np.float32)
    y = np.zeros(n, dtype=np.int64)
    subjects, stages, acq_ids, problems = [], [], [], []

    for i, (st, sub, ses, run, d) in enumerate(rows):
        nd = os.path.join(d, "normalized")
        try:
            for c, f in enumerate(FILES):
                a = np.load(os.path.join(nd, f))
                if a.shape != (N_ROI,):
                    problems.append(f"{sub}/{ses}/{run}: {f} shape {a.shape}")
                X_reg[i, :, c] = a.astype(np.float32)
                del a
            fc = np.load(os.path.join(nd, "fc_fisher_z.npy"))
            if fc.shape != (N_ROI, N_ROI):
                problems.append(f"{sub}/{ses}/{run}: fc shape {fc.shape}")
            X_fc[i] = fc.astype(np.float32)
            del fc
        except Exception as e:                               # noqa: BLE001
            problems.append(f"{sub}/{ses}/{run}: {type(e).__name__}: {e}")
            continue
        y[i] = STAGE_IDX[st]
        subjects.append(sub)
        stages.append(st)
        acq_ids.append(f"{sub}|{ses}|{run}")
        if (i + 1) % 25 == 0 or i + 1 == n:
            log(f"   loaded {i + 1}/{n}")

    # ---- verification before anything is written -----------------------
    log("\nverification")
    checks = {
        "X_regional finite": bool(np.isfinite(X_reg).all()),
        "X_fc finite": bool(np.isfinite(X_fc).all()),
        "X_fc symmetric": bool(np.allclose(X_fc, X_fc.transpose(0, 2, 1),
                                           atol=1e-5)),
        "X_fc diagonal zero": bool(np.abs(
            X_fc[:, np.arange(N_ROI), np.arange(N_ROI)]).max() < 1e-6),
        "no zero-variance acquisition": bool(
            (X_reg.std(axis=1) > 0).all()),
        "labels within range": bool(y.min() >= 0 and y.max() < len(STAGES)),
        "one row per acquisition": len(subjects) == n,
    }
    for k, v in checks.items():
        log(f"   {'PASS' if v else 'FAIL'}  {k}")
    if problems:
        log(f"\n{len(problems)} problem(s):")
        for p in problems[:15]:
            log(f"   {p}")
    if not all(checks.values()) or problems:
        raise SystemExit("verification failed -- nothing written")

    uniq, counts = np.unique(y, return_counts=True)
    log("\nclass balance")
    for u, c in zip(uniq, counts):
        log(f"   {STAGES[u]:5s} (y={u})  {c:3d}")
    log(f"\ndistinct subjects: {len(set(subjects))} of {n} acquisitions "
        f"({'one per subject' if len(set(subjects)) == n else 'REPEATS PRESENT'})")

    np.save(os.path.join(OUT, "X_regional.npy"), X_reg)
    np.save(os.path.join(OUT, "X_fc.npy"), X_fc)
    np.save(os.path.join(OUT, "y.npy"), y)
    np.save(os.path.join(OUT, "subjects.npy"), np.array(subjects))
    np.save(os.path.join(OUT, "stages.npy"), np.array(stages))
    np.save(os.path.join(OUT, "acq_ids.npy"), np.array(acq_ids))

    with open(os.path.join(OUT, "index.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["row", "acquisition_id", "subject", "stage", "y",
                    "source_dir"])
        for i, (st, sub, ses, run, d) in enumerate(rows):
            w.writerow([i, f"{sub}|{ses}|{run}", sub, st, STAGE_IDX[st],
                        os.path.relpath(d, ROOT)])

    meta = {
        "n_acquisitions": n,
        "n_subjects": len(set(subjects)),
        "n_roi": N_ROI,
        "regional_channels": list(CHANNELS),
        "stage_order": STAGES,
        "stage_to_index": STAGE_IDX,
        "class_counts": {STAGES[int(u)]: int(c) for u, c in zip(uniq, counts)},
        "fc": "Fisher-z, signed, unthresholded, diagonal zeroed",
        "excluded_feature": "FC_Strength (= DC/245 exactly, not independent)",
        "source": os.path.relpath(SRC, ROOT),
        "local_attention_mask": "NOT PROVIDED -- must be supplied to the model",
        "dtypes": {"X_regional": "float32", "X_fc": "float32", "y": "int64"},
        "verification": checks,
    }
    with open(os.path.join(OUT, "metadata.json"), "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=2)

    log(f"\nwritten to {os.path.relpath(OUT, ROOT)}")
    log(f"   X_regional.npy  {X_reg.shape}  {X_reg.nbytes/1e6:.1f} MB")
    log(f"   X_fc.npy        {X_fc.shape}  {X_fc.nbytes/1e6:.1f} MB")
    log(f"   y.npy           {y.shape}")
    log("\nMISSING for a forward pass: local_mask [246, 246]. "
        "It is a model input and is deliberately not invented here.")


if __name__ == "__main__":
    main()
