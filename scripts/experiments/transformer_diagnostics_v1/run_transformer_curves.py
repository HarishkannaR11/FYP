"""
Transformer training diagnostics: epoch-wise curves and a small configuration study.

Same data, folds, SVM-RFE (k = 50, rs-fMRI only), ROI tokens and bagging as
scripts/model/train_transformer_svmrfe.py. For each outer split, 4 inner
models are trained (inner 4-fold split of the training subjects, as before).
Each inner model now trains for the full max_epochs and every epoch is logged:

  train      the inner model's own training subjects
  val        its inner-validation subjects (the early-stopping set)
  outer      the outer test fold -- DIAGNOSTIC ONLY, never used for any decision

Logged per epoch: loss (unweighted cross-entropy), accuracy, balanced accuracy,
macro-F1. Early stopping (patience 15 on inner-validation balanced accuracy) is
applied exactly as in the original online loop: the training trajectory is
identical, the stop point and best epoch are found the same way, and the
reported prediction is the bag average at each model's best epoch. Continuing
past the stop point only adds curve points; it does not change the prediction.

Configurations (one change at a time from C0, the existing setting):
  C0  d16, 2 heads, 1 layer, dropout 0.3, lr 1e-3, weight decay 1e-2   (baseline)
  C1  lr 3e-4                 C2  lr 3e-3
  C3  dropout 0.1             C4  weight decay 1e-1
  C5  larger: d32, 4 heads, 2 layers

Selection: per outer split, the configuration with the highest mean inner-
validation balanced accuracy (training subjects only) is the "optimised"
transformer for that split (analysed by analyze.py).

Run:  python scripts/experiments/transformer_diagnostics_v1/run_transformer_curves.py --configs C0 --seeds all
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "model"))
import train_transformer_svmrfe as S  # noqa: E402  (sets thread env vars first)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

from heads import StagePredictor, build_head  # noqa: E402
from multi_attention_transformer import build_model  # noqa: E402
from train_transformer import Scaler  # noqa: E402

H = S.H
ROOT = S.ROOT
EXP = os.path.join(ROOT, "derivatives", "experiments", "transformer_diagnostics_v1")
BASE = {"d_model": 16, "num_heads": 2, "num_layers": 1, "dropout": 0.3, "lr": 1e-3, "weight_decay": 1e-2}
CONFIGS = {"C0": {}, "C1": {"lr": 3e-4}, "C2": {"lr": 3e-3}, "C3": {"dropout": 0.1},
           "C4": {"weight_decay": 1e-1}, "C5": {"d_model": 32, "num_heads": 4, "num_layers": 2}}
MAX_EPOCHS, PATIENCE, BS, CLIP = S.TRAIN["max_epochs"], S.TRAIN["patience"], S.TRAIN["batch_size"], S.TRAIN["grad_clip"]


@torch.no_grad()
def evaluate_set(model, X, F, mask_t, idx, y):
    model.eval()
    logits = torch.cat([model(X[idx[s:s + S.CHUNK]], F[idx[s:s + S.CHUNK]], mask_t)
                        for s in range(0, len(idx), S.CHUNK)])
    probs = torch.softmax(logits, 1).numpy()
    yy = torch.from_numpy(y[idx])
    m = H.evaluate(y[idx], probs.argmax(1))
    return probs, {"loss": float(Fn.cross_entropy(logits, yy)), "acc": m["accuracy"],
                   "bal": m["balanced_accuracy"], "f1": m["macro_f1"]}


def train_logged(cfg, X, F, mask_t, y, itr, iva, te, seed):
    """S.train_model's exact loop (same seeding, batching, clipping, criterion), run to MAX_EPOCHS with logging."""
    torch.manual_seed(seed)
    enc = build_model(n_roi=X.shape[1], n_features=X.shape[2], d_model=cfg["d_model"], num_heads=cfg["num_heads"],
                      num_layers=cfg["num_layers"], dropout=cfg["dropout"])
    model = StagePredictor(enc, build_head("multiclass", cfg["d_model"], dropout=cfg["dropout"]))
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    w = S.class_weights(y[itr])
    Xtr, Ftr, ytr = X[itr], F[itr], torch.from_numpy(y[itr])
    g = torch.Generator().manual_seed(seed)
    curves = {s: {k: [] for k in ("loss", "acc", "bal", "f1")} for s in ("train", "val", "outer")}
    outer_probs, best, best_ep, stop_ep, grad = [], -np.inf, -1, None, []
    for ep in range(MAX_EPOCHS):
        model.train()
        perm = torch.randperm(len(itr), generator=g)
        gn = []
        for s in range(0, len(perm), BS):
            b = perm[s:s + BS]
            opt.zero_grad()
            loss = model.head.loss(model(Xtr[b], Ftr[b], mask_t), ytr[b], weight=w)
            loss.backward()
            gn.append(float(nn.utils.clip_grad_norm_(model.parameters(), CLIP)))
            opt.step()
        grad.append(float(np.mean(gn)))
        _, mv = evaluate_set(model, X, F, mask_t, iva, y)
        _, mt = evaluate_set(model, X, F, mask_t, itr, y)
        po, mo = evaluate_set(model, X, F, mask_t, te, y)
        for k in mv:
            curves["val"][k].append(mv[k])
            curves["train"][k].append(mt[k])
            curves["outer"][k].append(mo[k])
        outer_probs.append(po)
        if stop_ep is None:                         # online early-stopping bookkeeping, unchanged
            if mv["bal"] > best:
                best, best_ep = mv["bal"], ep
            if ep - best_ep >= PATIENCE:
                stop_ep = ep
    n_params = sum(p.numel() for p in model.parameters())
    return curves, np.array(outer_probs), best_ep, (stop_ep if stop_ep is not None else MAX_EPOCHS - 1), grad, n_params


def run_split(cname, d, seed, fold, fp, tr, te):
    cfg = {**BASE, **CONFIGS[cname]}
    X4, flat, fc, y, subjects, mask = d
    Xs = Scaler().fit(X4, tr).transform(X4).astype(np.float32)               # training subjects only
    _, support, chmask = S.fit_selection(flat, y, tr, seed + fold)            # same 50 features as before
    tok, rois = S.make_tokens(Xs, chmask)
    X = torch.from_numpy(np.ascontiguousarray(tok))
    F = torch.from_numpy(np.ascontiguousarray(fc[:, rois][:, :, rois], dtype=np.float32))
    mask_t = torch.from_numpy(np.ascontiguousarray(mask[np.ix_(rois, rois)]))
    bag_seed = seed * 100 + fold
    fo = H.stratified_folds(y, tr, S.N_INNER, np.random.default_rng([bag_seed, 7]))
    models = []
    for j in range(S.N_INNER):
        itr = np.array([i for i in tr if fo[int(i)] != j])
        iva = np.array([i for i in tr if fo[int(i)] == j])
        curves, oprobs, best_ep, stop_ep, grad, n_params = train_logged(cfg, X, F, mask_t, y, itr, iva, te,
                                                                        bag_seed * 10 + j)
        models.append({"curves": curves, "best_epoch": best_ep, "stop_epoch": stop_ep, "grad_norm": grad,
                       "best_val_bal": curves["val"]["bal"][best_ep], "oprobs": oprobs})
    p_es = np.mean([m["oprobs"][m["best_epoch"]] for m in models], 0)            # the reported prediction
    p_last = np.mean([m["oprobs"][-1] for m in models], 0)                       # no early stopping
    bag_curve = {k: [] for k in ("acc", "bal", "f1")}
    for ep in range(MAX_EPOCHS):                                                 # bag at a fixed epoch (diagnostic)
        m = H.evaluate(y[te], np.mean([mm["oprobs"][ep] for mm in models], 0).argmax(1))
        bag_curve["acc"].append(m["accuracy"])
        bag_curve["bal"].append(m["balanced_accuracy"])
        bag_curve["f1"].append(m["macro_f1"])
    for mm in models:
        del mm["oprobs"]
    return {"config": cname, "settings": cfg, "seed": seed, "fold": fold, "fold_fingerprint": fp,
            "test_idx": te.tolist(), "true": y[te].tolist(), "K": int(rois.size), "n_parameters": n_params,
            "support": support.tolist(), "pred_es": p_es.argmax(1).tolist(), "probs_es": S.r6(p_es),
            "pred_last": p_last.argmax(1).tolist(), "inner_score": float(np.mean([m["best_val_bal"] for m in models])),
            "bag_outer_curve": bag_curve, "models": models}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", default=["C0"], choices=list(CONFIGS))
    ap.add_argument("--seeds", nargs="+", default=["all"])
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    seeds = H.SEEDS if args.seeds == ["all"] else [int(s) for s in args.seeds]
    d = S.load_cohort()
    plan = S.cv_plan(d[3], d[4], 5)
    n = 0
    for cname in args.configs:
        for seed in seeds:
            fp, folds = plan[seed]
            for f, (tr, te) in enumerate(folds):
                path = os.path.join(EXP, "grid", cname, f"cv5_seed{seed}_fold{f}.json")
                if os.path.exists(path):
                    continue
                t0 = time.time()
                rec = run_split(cname, d, seed, f, fp, tr, te)
                rec["seconds"] = round(time.time() - t0, 1)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                json.dump(rec, open(path + ".tmp", "w"))
                os.replace(path + ".tmp", path)
                print(f"{cname} seed {seed} fold {f}: best epochs {[m['best_epoch'] for m in rec['models']]}  "
                      f"{rec['seconds']:.0f}s", flush=True)
                n += 1
                if args.limit and n >= args.limit:
                    return


if __name__ == "__main__":
    main()
