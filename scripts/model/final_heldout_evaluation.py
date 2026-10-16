"""
FINAL HELD-OUT EVALUATION of the frozen dc_only_mlp_h64 model.

Run exactly once, under explicit user authorisation.

Procedure, taken from run_manifest.json["model_freeze"]:
  * input: 246 DC_z values per subject (channel index 2 of X_regional.npy)
  * MLP, two hidden layers of width 64, LayerNorm+GELU+dropout 0.3
  * multiclass 6-way softmax, argmax decode
  * AdamW lr=1e-3, weight_decay=1e-2, batch 16, grad clip 1.0
  * class weighting by inverse training-pool class frequency
  * per-feature standardisation FITTED ON THE TRAINING POOL ONLY

Epoch count comes from cross-validation alone, using the rule already
documented in train_transformer.py: the mean best epoch over the folds that
beat chance, plus one, floored at MIN_FINAL_EPOCHS. The test labels play no
part in it.

LIMITATION, stated rather than improvised around: no weights file was ever
saved for the frozen model. The frozen artifact is a documented procedure, so
the model is reproduced deterministically from the recorded seed and that
procedure. This is reproducible but is not the same as loading a stored
checkpoint, and the run manifest records it.

Nothing here is fitted, selected or tuned using test labels.
"""
import csv
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mlp_control import MLP
from train_transformer import MIN_FINAL_EPOCHS, N_STAGES, STAGES, evaluate

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "derivatives", "model_ready")
TRAIN = os.path.join(ROOT, "derivatives", "model_training")
OUT = os.path.join(TRAIN, "final_heldout")
SEED = 20261015
torch.set_num_threads(1)

_log = []


def log(m=""):
    print(m, flush=True)
    _log.append(str(m))


def main():
    os.makedirs(OUT, exist_ok=True)
    log("=" * 84)
    log("FINAL HELD-OUT EVALUATION -- frozen dc_only_mlp_h64")
    log("=" * 84)

    man = json.load(open(os.path.join(TRAIN, "run_manifest.json")))
    fz = man["model_freeze"]
    if fz["selected"] != "dc_only_mlp_h64" or fz["status"] != "FROZEN":
        log(f"ABORT: manifest does not show a frozen dc_only_mlp_h64 "
            f"({fz.get('status')} / {fz.get('selected')})")
        sys.exit(1)
    log(f"frozen model confirmed: {fz['selected']}")

    # ---- data and split
    reg = np.load(os.path.join(SRC, "X_regional.npy")).astype(np.float64)
    y = np.load(os.path.join(SRC, "y.npy"))
    subjects = np.load(os.path.join(SRC, "subjects.npy")).astype(str)
    meta = json.load(open(os.path.join(SRC, "metadata.json")))
    if meta["regional_channels"][2] != "DC_z":
        log("ABORT: channel 2 is not DC_z")
        sys.exit(1)
    if meta["stage_to_index"] != {s: i for i, s in enumerate(STAGES)}:
        log("ABORT: stage mapping changed")
        sys.exit(1)
    X = reg[:, :, 2]                      # 246 DC_z values per subject
    log(f"input: {X.shape} DC_z, channel index 2 verified")
    log(f"stage mapping: {meta['stage_to_index']}")

    rows = list(csv.DictReader(open(os.path.join(SRC, "splits",
                                                 "split_assignment.csv"))))
    smeta = json.load(open(os.path.join(SRC, "splits", "split_metadata.json")))
    pool = np.array([i for i, r in enumerate(rows)
                     if r["partition"] == "train_pool"])
    test = np.array([i for i, r in enumerate(rows) if r["partition"] == "test"])
    if set(pool.tolist()) & set(test.tolist()):
        log("ABORT: pool and test overlap")
        sys.exit(1)
    if sorted(subjects[test]) != sorted(man["split"]["held_out_test_subjects"]):
        log("ABORT: held-out subject IDs do not match the manifest")
        sys.exit(1)
    log(f"split fingerprint {smeta['split_fingerprint']}; "
        f"pool {pool.size}, held-out {test.size}; no overlap")

    # ---- epoch count, from cross-validation only
    ab = json.load(open(os.path.join(TRAIN, "biomarker_ablation",
                                     "ablation_records.json")))
    dc = [r for r in ab["records"] if r["config"] == "dc_only"]
    useful = [r["best_epoch"] for r in dc
              if r["balanced_accuracy"] > 1.0 / N_STAGES]
    epochs = max(int(round(float(np.mean(useful)))) + 1, MIN_FINAL_EPOCHS) \
        if useful else MIN_FINAL_EPOCHS
    log(f"epochs = {epochs}  (mean best epoch over the {len(useful)} of "
        f"{len(dc)} CV folds that beat chance, +1, floored at "
        f"{MIN_FINAL_EPOCHS}); test labels played no part")

    # ---- preprocessing fitted on the pool ONLY
    mu, sd = X[pool].mean(0), X[pool].std(0) + 1e-8
    if np.allclose(mu, X.mean(0), atol=1e-9):
        log("ABORT: scaler statistics equal whole-cohort statistics")
        sys.exit(1)
    log("scaler fitted on the training pool only (verified != cohort statistics)")
    Xtr = torch.from_numpy(((X[pool] - mu) / sd).astype(np.float32))
    ytr = torch.from_numpy(y[pool])
    counts = np.bincount(y[pool], minlength=N_STAGES).astype(np.float32)
    w = torch.from_numpy((counts.sum() / (N_STAGES * np.maximum(counts, 1))
                          ).astype(np.float32))

    # ---- train once on the pool
    t0 = time.time()
    torch.manual_seed(SEED)
    model = MLP(X.shape[1], 64, "multiclass", 0.3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    g = torch.Generator().manual_seed(SEED)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(pool.size, generator=g)
        for s in range(0, pool.size, 16):
            b = perm[s:s + 16]
            opt.zero_grad()
            model.head.loss(model(Xtr[b]), ytr[b], weight=w).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    npar = sum(p.numel() for p in model.parameters())
    log(f"trained on {pool.size} pool subjects, {epochs} epochs, "
        f"{npar:,} parameters, {time.time() - t0:.1f}s")

    ckpt = os.path.join(OUT, "frozen_dc_only_mlp_h64.pt")
    torch.save({"state_dict": model.state_dict(), "seed": SEED,
                "epochs": epochs, "scaler_mean": mu, "scaler_sd": sd,
                "input": "DC_z 246", "config": fz["configuration"]}, ckpt)
    log(f"saved checkpoint {os.path.relpath(ckpt, ROOT)}")

    # ---- evaluate ONCE
    model.eval()
    with torch.no_grad():
        out = model(torch.from_numpy(((X[test] - mu) / sd).astype(np.float32)))
        probs = model.head.probabilities(out).numpy()
        pred = model.head.decode(out).numpy()
    m = evaluate(probs, y[test], pred=pred)

    p = os.path.join(OUT, "test_predictions.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["subject_id", "true_stage", "true_index",
                     "predicted_stage", "predicted_index", "correct"]
                    + [f"p_{s}" for s in STAGES])
        for j, i in enumerate(test):
            wr.writerow([subjects[i], STAGES[y[i]], int(y[i]),
                         STAGES[pred[j]], int(pred[j]),
                         int(pred[j] == y[i])]
                        + [f"{v:.6f}" for v in probs[j]])
    log(f"saved {os.path.relpath(p, ROOT)}")

    # ---- recompute every metric from the saved file
    saved = list(csv.DictReader(open(p)))
    ys = np.array([int(r["true_index"]) for r in saved])
    ps = np.array([int(r["predicted_index"]) for r in saved])
    pr = np.array([[float(r[f"p_{s}"]) for s in STAGES] for r in saved])
    m2 = evaluate(pr, ys, pred=ps)
    agree = all(np.isclose(m[k], m2[k], atol=1e-6) for k in
                ("accuracy", "balanced_accuracy", "macro_f1", "mae_stages", "qwk"))
    log(f"metrics recomputed from the saved CSV agree: {agree}")
    if not agree:
        log("ABORT: metrics not reproducible from saved predictions")
        sys.exit(1)

    # ---- report
    C = np.array(m["confusion_matrix"])
    log("")
    log("=" * 84)
    log("HELD-OUT RESULTS  (35 subjects, evaluated once)")
    log("=" * 84)
    log(f"  balanced accuracy {m['balanced_accuracy']:.4f}   "
        f"(chance {1 / N_STAGES:.4f})")
    log(f"  accuracy          {m['accuracy']:.4f}")
    log(f"  macro-F1          {m['macro_f1']:.4f}")
    log(f"  MAE (stages)      {m['mae_stages']:.4f}")
    log(f"  QWK               {m['qwk']:+.4f}")
    log(f"  distinct classes predicted {m['n_predicted_classes']} of 6")
    log("")
    log(f"  {'stage':6s} {'supp':>5s} {'pred':>5s} {'prec':>6s} {'rec':>6s} {'F1':>6s}")
    per = {}
    for gi in range(N_STAGES):
        supp = int((ys == gi).sum())
        npred = int((ps == gi).sum())
        tp = int(C[gi, gi])
        prec = tp / npred if npred else float("nan")
        rec = tp / supp if supp else float("nan")
        f1 = (2 * prec * rec / (prec + rec)
              if npred and supp and (prec + rec) > 0 else 0.0)
        per[STAGES[gi]] = {"support": supp, "n_predicted": npred,
                           "precision": prec, "recall": rec, "f1": f1}
        log(f"  {STAGES[gi]:6s} {supp:5d} {npred:5d} {prec:6.3f} {rec:6.3f} {f1:6.3f}")
    log("")
    log("  confusion (rows = true, cols = predicted)")
    log("          " + "".join(f"{s:>6s}" for s in STAGES))
    for gi in range(N_STAGES):
        log(f"    {STAGES[gi]:6s}" + "".join(f"{int(v):6d}" for v in C[gi]))

    # ---- comparison
    maj = int(np.bincount(y[pool], minlength=N_STAGES).argmax())
    maj_pred = np.full(ys.shape, maj)
    maj_bal = float(np.nanmean([(maj_pred[ys == gi] == gi).mean()
                                if (ys == gi).any() else np.nan
                                for gi in range(N_STAGES)]))
    maj_mae = float(np.abs(maj_pred - ys).mean())
    cv = fz["evidence"]["dc_only"]
    log("")
    log("  comparison (descriptive; a single 35-subject estimate is NOT "
        "comparable in precision to a 25-fold CV mean)")
    log(f"    repeated-CV balanced accuracy (25 folds) : {cv['bal_acc']}")
    log(f"    held-out balanced accuracy (1 estimate)  : "
        f"{m['balanced_accuracy']:.3f}")
    log(f"    majority-class on the same 35 subjects   : {maj_bal:.3f} "
        f"(MAE {maj_mae:.3f})")
    log(f"    one subject changes balanced accuracy by about "
        f"{1 / 6 / (35 / 6):.3f}")

    res = {"model": "dc_only_mlp_h64", "evaluated_once": True,
           "n_test": int(test.size), "epochs": epochs, "seed": SEED,
           "n_parameters": npar,
           "split_fingerprint": smeta["split_fingerprint"],
           "metrics": {k: m[k] for k in
                       ("accuracy", "balanced_accuracy", "macro_f1",
                        "mae_stages", "qwk", "ece", "n_predicted_classes")},
           "per_class": per, "confusion_matrix": C.tolist(),
           "class_support": m["class_support"],
           "predicted_distribution": m["predicted_distribution"],
           "majority_baseline": {"balanced_accuracy": maj_bal, "mae": maj_mae},
           "repeated_cv_reference": cv,
           "metrics_reproduced_from_saved_predictions": True,
           "checkpoint": os.path.relpath(ckpt, ROOT),
           "predictions": os.path.relpath(p, ROOT),
           "test_set_history": fz["test_set_history"],
           "limitation": "no weights file existed for the frozen model; it was "
                         "reproduced deterministically from the documented "
                         "procedure and recorded seed"}
    rp = os.path.join(OUT, "final_heldout_results.json")
    with open(rp, "w", encoding="utf-8") as f:
        json.dump(res, f, indent=2)
    log(f"\nwrote {os.path.relpath(rp, ROOT)}")
    with open(os.path.join(OUT, "final_heldout.log"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")


if __name__ == "__main__":
    main()
