"""
Train and evaluate the Multi-Attention Transformer for stage prediction.

Protocol
--------
  70% training pool, 30% held-out test, stratified and subject-level
  (scripts/model/make_splits.py). Inside the pool, 5-fold cross-validation
  selects the model. The test set is evaluated ONCE, after selection.

Leakage control, enforced here
------------------------------
  * The cross-subject scaler is fitted on the TRAINING FOLD ONLY and applied
    unchanged to the validation fold; for the final model it is fitted on the
    whole training pool and applied unchanged to the test set. Statistics from
    validation or test data never reach the fit.
  * Early stopping uses the validation fold, never the test set.
  * The test set is loaded only in the final stage, after every
    hyperparameter and the epoch count are fixed.
  * The split fingerprint is recorded in every output, so a result can always
    be traced to the exact partition that produced it.

The within-acquisition normalisation already applied to these features
(mALFF/mReHo as a ratio to the acquisition's own mean, DC_z z-scored across
that acquisition's ROIs, FC Fisher-z) uses no cross-subject statistic and no
label, so it is leakage-free by construction and is left in place.

Class imbalance
---------------
MCI has 16 subjects where the others have 20. Loss is weighted by inverse
class frequency computed on the training fold.

What this script does NOT do
----------------------------
It does not implement meta-learning, comorbidity fusion, or a full Bayesian
posterior. Uncertainty here is MC-dropout, which is an approximation and is
reported as such. No result is written as a project claim until it has
actually been produced by a run of this script.
"""
import argparse
import csv
import json
import os
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import gc

import numpy as np
import torch
import torch.nn as nn

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heads import StagePredictor, build_head           # noqa: E402
from multi_attention_transformer import build_model    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "derivatives", "model_ready")
SPLITS = os.path.join(SRC, "splits")
OUT = os.path.join(ROOT, "derivatives", "model_training")

STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
N_STAGES = 6
MIN_FINAL_EPOCHS = 10          # floor for the final retrain, declared up front
torch.set_num_threads(1)


# ----------------------------------------------------------------- data
def load_all():
    reg = np.load(os.path.join(SRC, "X_regional.npy")).astype(np.float32)
    fc = np.load(os.path.join(SRC, "X_fc.npy")).astype(np.float32)
    y = np.load(os.path.join(SRC, "y.npy")).astype(np.int64)
    subjects = np.load(os.path.join(SRC, "subjects.npy")).astype(str)
    mask = np.load(os.path.join(SRC, "local_attention_mask.npy"))
    meta = json.load(open(os.path.join(SRC, "metadata.json")))
    if meta["stage_order"] != STAGES:
        raise SystemExit("stage order mismatch")
    if not (np.isfinite(reg).all() and np.isfinite(fc).all()):
        raise SystemExit("non-finite features")

    rows = list(csv.DictReader(open(os.path.join(SPLITS, "split_assignment.csv"))))
    smeta = json.load(open(os.path.join(SPLITS, "split_metadata.json")))
    if [r["subject_id"] for r in rows] != list(subjects):
        raise SystemExit("split file does not match the cohort order")
    part = np.array([r["partition"] for r in rows])
    fold = np.array([int(r["cv_fold"]) if r["cv_fold"] != "" else -1 for r in rows])
    pool = np.flatnonzero(part == "train_pool")
    test = np.flatnonzero(part == "test")
    if set(pool.tolist()) & set(test.tolist()):
        raise SystemExit("pool and test overlap")
    return reg, fc, y, subjects, mask, pool, test, fold, smeta


class Scaler:
    """Per-(ROI, channel) standardisation of the regional features.

    Fitted on training indices only. FC is left untouched: it is already
    Fisher-z within each acquisition and rescaling it across subjects would
    distort the connectivity bias the network branch depends on."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, reg, idx):
        self.mean = reg[idx].mean(axis=0, keepdims=True)
        self.std = reg[idx].std(axis=0, ddof=0, keepdims=True)
        self.std = np.where(self.std < 1e-8, 1.0, self.std)
        return self

    def transform(self, reg):
        return (reg - self.mean) / self.std


# -------------------------------------------------------------- metrics
def expected_calibration_error(probs, y, n_bins=10):
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == y).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1])
        if m.sum() == 0:
            continue
        ece += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return float(ece)


def quadratic_weighted_kappa(y, pred, k=N_STAGES):
    O = np.zeros((k, k))
    for a, b in zip(y, pred):
        O[a, b] += 1
    w = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)], float)
    w /= (k - 1) ** 2
    hy = np.bincount(y, minlength=k).astype(float)
    hp = np.bincount(pred, minlength=k).astype(float)
    E = np.outer(hy, hp)
    E = E / E.sum() * O.sum()
    den = (w * E).sum()
    return float(1 - (w * O).sum() / den) if den > 0 else float("nan")


def macro_f1(y, pred, k=N_STAGES):
    f1s = []
    for g in range(k):
        tp = float(((pred == g) & (y == g)).sum())
        fp = float(((pred == g) & (y != g)).sum())
        fn = float(((pred != g) & (y == g)).sum())
        denom = 2 * tp + fp + fn
        f1s.append(2 * tp / denom if denom > 0 else 0.0)
    return float(np.mean(f1s)), f1s


def confusion(y, pred, k=N_STAGES):
    C = np.zeros((k, k), dtype=int)
    for a, b in zip(y, pred):
        C[a, b] += 1
    return C


def evaluate(probs, y, pred=None):
    """`pred` lets a head supply its own decision rule. Cumulative-link
    ordinal models must not be decoded with argmax."""
    pred = probs.argmax(axis=1) if pred is None else np.asarray(pred)
    acc = float((pred == y).mean())
    recalls = []
    for g in range(N_STAGES):
        m = y == g
        recalls.append(float((pred[m] == g).mean()) if m.sum() else float("nan"))
    bal = float(np.nanmean(recalls))
    mf1, per_f1 = macro_f1(y, pred)
    C = confusion(y, pred)
    return {
        "accuracy": acc,
        "balanced_accuracy": bal,
        "macro_f1": mf1,
        "mae_stages": float(np.abs(pred - y).mean()),
        "qwk": quadratic_weighted_kappa(y, pred),
        "ece": expected_calibration_error(probs, y),
        "per_class_recall": {STAGES[g]: recalls[g] for g in range(N_STAGES)},
        "per_class_f1": {STAGES[g]: per_f1[g] for g in range(N_STAGES)},
        "confusion_matrix": C.tolist(),
        "class_support": {STAGES[g]: int((y == g).sum()) for g in range(N_STAGES)},
        "predicted_distribution": {STAGES[g]: int((pred == g).sum())
                                   for g in range(N_STAGES)},
        "n_predicted_classes": int(len(np.unique(pred))),
        "n": int(y.size),
    }


# ------------------------------------------------------------- training
def predict_chunked_with_decode(model, X, F, mask_t, chunk=4):
    """Returns (probabilities, head-decoded point predictions)."""
    ps, ds = [], []
    with torch.no_grad():
        for s in range(0, X.shape[0], chunk):
            out = model(X[s:s + chunk], F[s:s + chunk], mask_t)
            ps.append(model.head.probabilities(out))
            ds.append(model.head.decode(out))
    return torch.cat(ps).numpy(), torch.cat(ds).numpy()


def predict_chunked(model, X, F, mask_t, chunk=4):
    """Attention is [B, H, 246, 246]; at batch 16 that is 15.5 MB per tensor and
    autograd keeps one per branch per layer. On a memory-constrained machine the
    whole fold must not be pushed through in one call."""
    outs = []
    with torch.no_grad():
        for s in range(0, X.shape[0], chunk):
            outs.append(model.head.probabilities(
                model(X[s:s + chunk], F[s:s + chunk], mask_t)))
    return torch.cat(outs).numpy()


def make_model(cfg, seed, class_counts=None):
    torch.manual_seed(seed)
    enc = build_model(d_model=cfg["d_model"], num_heads=cfg["num_heads"],
                      num_layers=cfg["num_layers"], dropout=cfg["dropout"])
    head = build_head(cfg["head"], cfg["d_model"], dropout=cfg["dropout"])
    # thresholds are placed at the TRAINING-fold class CDF; no validation or
    # test information is used
    if class_counts is not None and hasattr(head, "init_from_prior"):
        head.init_from_prior(class_counts)
    return StagePredictor(enc, head)


def run_fold(reg, fc, y, mask_t, tr_idx, va_idx, cfg, seed, log):
    sc = Scaler().fit(reg, tr_idx)                 # TRAIN FOLD ONLY
    Xtr = torch.from_numpy(sc.transform(reg[tr_idx]))
    Xva = torch.from_numpy(sc.transform(reg[va_idx]))
    Ftr = torch.from_numpy(fc[tr_idx])
    Fva = torch.from_numpy(fc[va_idx])
    ytr = torch.from_numpy(y[tr_idx])
    yva = torch.from_numpy(y[va_idx])

    counts = np.bincount(y[tr_idx], minlength=N_STAGES).astype(np.float32)
    w = torch.from_numpy(np.where(counts > 0, counts.sum() / (N_STAGES * np.maximum(counts, 1)), 0.0).astype(np.float32))

    model = make_model(cfg, seed, class_counts=counts)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"],
                            weight_decay=cfg["weight_decay"])
    bs = cfg["batch_size"]
    best = {"score": -np.inf, "epoch": -1, "state": None}
    g = torch.Generator().manual_seed(seed)
    history = []
    stop_reason = "max_epochs"

    for epoch in range(cfg["max_epochs"]):
        model.train()
        gn_enc, gn_head = [], []
        perm = torch.randperm(len(tr_idx), generator=g)
        tot = 0.0
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            opt.zero_grad()
            out = model(Xtr[b], Ftr[b], mask_t)
            loss = model.head.loss(out, ytr[b], weight=w)
            loss.backward()
            enc_sq = sum(float(p.grad.pow(2).sum()) for p in model.encoder.parameters()
                         if p.grad is not None)
            head_sq = sum(float(p.grad.pow(2).sum()) for p in model.head.parameters()
                          if p.grad is not None)
            gn_enc.append(enc_sq ** 0.5)
            gn_head.append(head_sq ** 0.5)
            nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            opt.step()
            tot += loss.detach().item() * len(b)
        model.eval()
        pv, dv = predict_chunked_with_decode(model, Xva, Fva, mask_t,
                                             cfg["eval_chunk"])
        with torch.no_grad():
            val_loss = float(model.head.loss(
                model(Xva[:cfg["eval_chunk"]], Fva[:cfg["eval_chunk"]], mask_t),
                yva[:cfg["eval_chunk"]]))
        m = evaluate(pv, y[va_idx], pred=dv)
        history.append({"epoch": epoch, "train_loss": tot / len(perm),
                        "val_loss": float(val_loss),
                        "val_balanced_accuracy": m["balanced_accuracy"],
                        "val_macro_f1": m["macro_f1"],
                        "val_mae": m["mae_stages"],
                        "grad_norm_encoder": float(np.mean(gn_enc)),
                        "grad_norm_head": float(np.mean(gn_head)),
                        "n_predicted_classes": m["n_predicted_classes"]})
        if not np.isfinite(tot):
            stop_reason = "non_finite_loss"
            break
        if m["balanced_accuracy"] > best["score"]:
            best = {"score": m["balanced_accuracy"], "epoch": epoch,
                    "state": {k: v.detach().clone()
                              for k, v in model.state_dict().items()}}
        if epoch - best["epoch"] >= cfg["patience"]:
            stop_reason = "early_stopping_patience"
            break

    model.load_state_dict(best["state"])
    model.eval()
    pv, dv = predict_chunked_with_decode(model, Xva, Fva, mask_t,
                                         cfg["eval_chunk"])
    metrics = evaluate(pv, y[va_idx], pred=dv)
    metrics["beat_chance"] = bool(metrics["balanced_accuracy"] > 1.0 / N_STAGES)
    metrics["best_epoch"] = best["epoch"]
    metrics["epochs_run"] = len(history)
    metrics["stop_reason"] = stop_reason
    metrics["seed"] = seed
    metrics["grad_norm_encoder_mean"] = float(np.mean(
        [h["grad_norm_encoder"] for h in history]))
    metrics["grad_norm_head_mean"] = float(np.mean(
        [h["grad_norm_head"] for h in history]))
    metrics["val_predictions"] = {
        "subject_index": [int(i) for i in va_idx],
        "true": [int(v) for v in y[va_idx]],
        "probs": [[round(float(x), 6) for x in row] for row in pv]}
    del Xtr, Xva, Ftr, Fva, model, opt, best
    gc.collect()
    log(f"      best epoch {metrics['best_epoch']:3d}/{metrics['epochs_run']:3d}  "
        f"bal.acc {metrics['balanced_accuracy']:.3f}  "
        f"acc {metrics['accuracy']:.3f}  MAE {metrics['mae_stages']:.2f}  "
        f"QWK {metrics['qwk']:+.3f}")
    return metrics, history, metrics["best_epoch"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", choices=["multiclass", "ordinal", "ordinal_fixed"],
                    required=True)
    ap.add_argument("--mask", choices=["local", "global"], default="local",
                    help="'global' replaces the atlas mask with all-True, "
                         "making the local branch equivalent to global "
                         "attention -- the ablation control")
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--num-heads", type=int, default=4)
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-chunk", type=int, default=4,
                    help="subjects per forward pass at inference; lower uses "
                         "less memory")
    ap.add_argument("--max-epochs", type=int, default=120)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=20261015)
    ap.add_argument("--mc-samples", type=int, default=30)
    ap.add_argument("--final-test", action="store_true",
                    help="after CV, retrain on the whole pool and evaluate the "
                         "held-out test set ONCE")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    cfg = {k: v for k, v in vars(args).items()}

    os.makedirs(OUT, exist_ok=True)
    tag = args.tag or args.head
    lines = []

    def log(m=""):
        print(m, flush=True)
        lines.append(str(m))

    reg, fc, y, subjects, mask, pool, test, fold, smeta = load_all()
    if args.mask == "global":
        mask = np.ones_like(mask, dtype=bool)
    mask_t = torch.from_numpy(mask)
    log("=" * 74)
    log(f"TRAINING  head={args.head}  d_model={args.d_model} "
        f"heads={args.num_heads} layers={args.num_layers}")
    log("=" * 74)
    log(f"split fingerprint {smeta['split_fingerprint']}  "
        f"pool {pool.size}  test {test.size} (held out)")
    log(f"attention mask: {args.mask}, density "
        f"{mask.sum() / mask.size * 100:.2f}%"
        + (", derived from the atlas" if args.mask == "local"
           else " (all-True: local branch == global attention)"))
    n_folds = smeta["n_folds"]

    t0 = time.time()
    fold_metrics, best_epochs = [], []
    for f in range(n_folds):
        va = np.array([i for i in pool if fold[i] == f])
        tr = np.array([i for i in pool if fold[i] != f])
        log(f"  fold {f}: train {tr.size}  val {va.size}")
        m, hist, be = run_fold(reg, fc, y, mask_t, tr, va, cfg,
                               args.seed + f, log)
        m["fold"] = f
        fold_metrics.append(m)
        best_epochs.append(be)

    def agg(key):
        v = [m[key] for m in fold_metrics]
        return float(np.mean(v)), float(np.std(v, ddof=1))

    log("")
    log(f"  {n_folds}-fold CV summary (mean +/- sd across folds)")
    summary = {}
    for key in ("accuracy", "balanced_accuracy", "macro_f1", "mae_stages",
                "qwk", "ece"):
        mu, sd = agg(key)
        summary[key] = {"mean": mu, "sd": sd}
        log(f"    {key:20s} {mu:+.4f} +/- {sd:.4f}")
    Csum = np.sum([np.array(m["confusion_matrix"]) for m in fold_metrics], axis=0)
    summary["confusion_matrix_summed"] = Csum.tolist()
    npc = [m["n_predicted_classes"] for m in fold_metrics]
    summary["n_predicted_classes_per_fold"] = npc
    summary["degenerate"] = bool(max(npc) <= 2)
    log(f"    best epochs per fold  {best_epochs}")
    log(f"    stop reasons          {[m['stop_reason'] for m in fold_metrics]}")
    log(f"    distinct classes predicted per fold {npc}"
        + ("   <-- DEGENERATE" if max(npc) <= 2 else ""))
    log(f"    grad norm encoder/head {np.mean([m['grad_norm_encoder_mean'] for m in fold_metrics]):.4f}"
        f" / {np.mean([m['grad_norm_head_mean'] for m in fold_metrics]):.4f}")
    log("    confusion (rows=true, cols=pred, summed over folds)")
    log("              " + "".join(f"{s:>6s}" for s in STAGES))
    for gi, srow in enumerate(Csum):
        log(f"      {STAGES[gi]:6s}" + "".join(f"{int(v):6d}" for v in srow))
    log(f"    elapsed               {time.time() - t0:.1f}s")

    result = {
        "config": cfg,
        "split_fingerprint": smeta["split_fingerprint"],
        "n_train_pool": int(pool.size),
        "n_test": int(test.size),
        "cv": {"folds": fold_metrics, "summary": summary,
               "best_epochs": best_epochs},
        "test": None,
        "trained": True,
        "notes": "MC-dropout uncertainty is an approximation, not a full "
                 "Bayesian posterior. Meta-learning and comorbidity fusion are "
                 "not implemented.",
    }

    if args.final_test:
        log("")
        log("  FINAL: retraining on the full training pool, then evaluating the "
            "held-out test set once")
        # A fold whose best epoch is 0 never beat its starting point, so it
        # carries no information about how long to train. Averaging those zeros
        # in produces a final model trained for almost no epochs, which is how
        # an earlier run ended up predicting a single class for every subject.
        useful = [e for e, m in zip(best_epochs, fold_metrics) if m["beat_chance"]]
        n_flat = len(best_epochs) - len(useful)
        if useful:
            epochs = int(round(float(np.mean(useful)))) + 1
            basis = (f"mean best epoch over the {len(useful)} fold(s) that beat "
                     f"chance, + 1")
        else:
            epochs = MIN_FINAL_EPOCHS
            basis = (f"no fold beat chance, so CV gives no usable epoch count; "
                     f"falling back to the declared floor of {MIN_FINAL_EPOCHS}")
        epochs = max(epochs, MIN_FINAL_EPOCHS)
        log(f"    epochs = {epochs}  ({basis}; chosen from CV only, the test "
            f"set played no part)")
        if n_flat:
            log(f"    WARNING: {n_flat} of {len(best_epochs)} folds never beat "
                f"chance balanced accuracy ({1 / N_STAGES:.3f}).")
        if not useful:
            log("    WARNING: cross-validation did not find a configuration that "
                "beats chance. The test numbers below describe a model that CV "
                "gives no reason to trust, and must be reported that way.")
        result["final_epoch_selection"] = {
            "epochs": epochs, "basis": basis,
            "folds_beating_chance": len(useful), "folds_flat": n_flat}
        sc = Scaler().fit(reg, pool)                   # POOL ONLY
        Xtr = torch.from_numpy(sc.transform(reg[pool]))
        Ftr = torch.from_numpy(fc[pool])
        ytr = torch.from_numpy(y[pool])
        counts = np.bincount(y[pool], minlength=N_STAGES).astype(np.float32)
        w = torch.from_numpy((counts.sum() / (N_STAGES * np.maximum(counts, 1))
                              ).astype(np.float32))
        model = make_model(cfg, args.seed, class_counts=counts)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay)
        g = torch.Generator().manual_seed(args.seed)
        for epoch in range(epochs):
            model.train()
            perm = torch.randperm(pool.size, generator=g)
            for s in range(0, pool.size, args.batch_size):
                b = perm[s:s + args.batch_size]
                opt.zero_grad()
                loss = model.head.loss(model(Xtr[b], Ftr[b], mask_t), ytr[b],
                                       weight=w)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()

        Xte = torch.from_numpy(sc.transform(reg[test]))
        Fte = torch.from_numpy(fc[test])
        model.eval()
        pt, dt = predict_chunked_with_decode(model, Xte, Fte, mask_t,
                                             args.eval_chunk)
        tm = evaluate(pt, y[test], pred=dt)
        pm, ent = model.predict_mc(Xte, Fte, mask_t, n_samples=args.mc_samples)
        tm_mc = evaluate(pm.numpy(), y[test])
        tm_mc["mean_predictive_entropy"] = float(ent.mean())
        tm_mc["max_entropy_possible"] = float(np.log(N_STAGES))
        log(f"    test accuracy          {tm['accuracy']:.4f}")
        log(f"    test balanced accuracy {tm['balanced_accuracy']:.4f}  "
            f"(chance {1 / N_STAGES:.4f})")
        log(f"    test MAE (stages)      {tm['mae_stages']:.3f}")
        log(f"    test QWK               {tm['qwk']:+.4f}")
        log(f"    test ECE               {tm['ece']:.4f}")
        log(f"    MC-dropout bal.acc     {tm_mc['balanced_accuracy']:.4f}  "
            f"mean entropy {tm_mc['mean_predictive_entropy']:.3f} "
            f"(max {np.log(N_STAGES):.3f})")
        log(f"    per-class recall       " +
            "  ".join(f"{k} {v:.2f}" for k, v in tm["per_class_recall"].items()))
        result["test"] = {"deterministic": tm, "mc_dropout": tm_mc,
                          "epochs": epochs}

        p = os.path.join(OUT, f"test_predictions_{tag}.csv")
        with open(p, "w", newline="", encoding="utf-8") as f:
            wr = csv.writer(f)
            wr.writerow(["subject_id", "true_stage", "predicted_stage",
                         "predictive_entropy"]
                        + [f"p_{s}" for s in STAGES])
            for j, i in enumerate(test):
                wr.writerow([subjects[i], STAGES[y[i]],
                             STAGES[int(pm[j].argmax())], f"{float(ent[j]):.4f}"]
                            + [f"{float(v):.4f}" for v in pm[j]])
        log(f"    wrote {os.path.relpath(p, ROOT)}")

    p = os.path.join(OUT, f"results_{tag}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    log(f"\nwrote {os.path.relpath(p, ROOT)}")
    with open(os.path.join(OUT, f"train_{tag}.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
