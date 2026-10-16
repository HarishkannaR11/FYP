"""
Repeated stratified subject-level cross-validation: 5 folds x 5 split seeds.

The single 5-fold comparison was inconclusive -- candidate standard deviations
overlapped every rival. This repeats the whole cross-validation under five
different partitions of the same 81-subject training pool, so a difference can
be judged against the variation caused by partitioning alone.

Design
------
  * Training pool only (81 subjects). The held-out 35 are never loaded.
  * For each seed, ONE fold assignment is computed and used by EVERY
    candidate, so comparisons are paired at the fold level.
  * All learned preprocessing is fitted on each fold's training subjects only.
  * Every fold's subject IDs, true labels, probabilities and decoded
    predictions are saved, so all metrics are recomputable without retraining.
  * Candidates run sequentially; no concurrent jobs.

What repeated CV does NOT do: it does not guarantee better generalisation, and
it does not validate anything clinically. It only measures whether an observed
difference survives re-partitioning.
"""
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn
from scipy.special import softmax

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from baselines import fit_multinomial, predict_multinomial
from heads import build_head
from mlp_control import MLP
from multi_attention_transformer import build_model
from heads import StagePredictor
from train_transformer import (N_STAGES, STAGES, Scaler, evaluate,
                               load_all, predict_chunked_with_decode)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "derivatives", "model_training")
REP = os.path.join(OUT, "repeated_cv")
SEEDS = [20261015, 20261016, 20261017, 20261018, 20261019]
N_FOLDS = 5
torch.set_num_threads(1)

_log = []


def log(m=""):
    print(m, flush=True)
    _log.append(str(m))


def stratified_folds(y, pool, k, rng):
    """Identical rule to make_splits.py: round-robin per class, carrying the
    offset between classes so remainders do not pile onto low fold indices."""
    fold_of = {}
    start = 0
    for g in range(N_STAGES):
        idx = pool[y[pool] == g].copy()
        rng.shuffle(idx)
        for j, i in enumerate(idx):
            fold_of[int(i)] = (start + j) % k
        start = (start + idx.size) % k
    return fold_of


def fold_fingerprint(subjects, fold_of, pool):
    h = hashlib.sha256()
    for i in sorted(pool):
        h.update(f"{subjects[i]}|{fold_of[int(i)]}".encode())
    return h.hexdigest()[:16]


# ------------------------------------------------------------- candidates
def run_transformer(head_kind, Xreg, Xfc, y, mask_t, tr, va, seed,
                    d_model=16, layers=1, dropout=0.3, lr=1e-3,
                    epochs=60, patience=15):
    sc = Scaler().fit(Xreg, tr)
    Xtr = torch.from_numpy(sc.transform(Xreg[tr]).astype(np.float32))
    Xva = torch.from_numpy(sc.transform(Xreg[va]).astype(np.float32))
    Ftr = torch.from_numpy(Xfc[tr])
    Fva = torch.from_numpy(Xfc[va])
    ytr = torch.from_numpy(y[tr])
    counts = np.bincount(y[tr], minlength=N_STAGES).astype(np.float32)
    w = torch.from_numpy((counts.sum() / (N_STAGES * np.maximum(counts, 1))
                          ).astype(np.float32))
    torch.manual_seed(seed)
    enc = build_model(d_model=d_model, num_heads=4 if d_model >= 32 else 2,
                      num_layers=layers, dropout=dropout)
    head = build_head(head_kind, d_model, dropout=dropout)
    if hasattr(head, "init_from_prior"):
        head.init_from_prior(counts)
    model = StagePredictor(enc, head)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    g = torch.Generator().manual_seed(seed)
    best = {"score": -np.inf, "epoch": -1, "state": None}
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr), generator=g)
        for s in range(0, len(perm), 8):
            b = perm[s:s + 8]
            opt.zero_grad()
            model.head.loss(model(Xtr[b], Ftr[b], mask_t), ytr[b],
                            weight=w).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        p, d = predict_chunked_with_decode(model, Xva, Fva, mask_t, 4)
        m = evaluate(p, y[va], pred=d)
        if m["balanced_accuracy"] > best["score"]:
            best = {"score": m["balanced_accuracy"], "epoch": ep,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
        if ep - best["epoch"] >= patience:
            break
    model.load_state_dict(best["state"])
    model.eval()
    p, d = predict_chunked_with_decode(model, Xva, Fva, mask_t, 4)
    npar = sum(q.numel() for q in model.parameters())
    del model, opt
    return p, d, best["epoch"], npar


def run_mlp(head_kind, Xflat, y, tr, va, seed, hidden, lr=1e-3,
            epochs=120, patience=20, dropout=0.3):
    mu, sd = Xflat[tr].mean(0), Xflat[tr].std(0) + 1e-8
    Xtr = torch.from_numpy(((Xflat[tr] - mu) / sd).astype(np.float32))
    Xva = torch.from_numpy(((Xflat[va] - mu) / sd).astype(np.float32))
    ytr = torch.from_numpy(y[tr])
    counts = np.bincount(y[tr], minlength=N_STAGES).astype(np.float32)
    w = torch.from_numpy((counts.sum() / (N_STAGES * np.maximum(counts, 1))
                          ).astype(np.float32))
    torch.manual_seed(seed)
    model = MLP(Xflat.shape[1], hidden, head_kind, dropout)
    if hasattr(model.head, "init_from_prior"):
        model.head.init_from_prior(counts)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    g = torch.Generator().manual_seed(seed)
    best = {"score": -np.inf, "epoch": -1, "state": None}
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr), generator=g)
        for s in range(0, len(perm), 16):
            b = perm[s:s + 16]
            opt.zero_grad()
            model.head.loss(model(Xtr[b]), ytr[b], weight=w).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            out = model(Xva)
            m = evaluate(model.head.probabilities(out).numpy(), y[va],
                         pred=model.head.decode(out).numpy())
        if m["balanced_accuracy"] > best["score"]:
            best = {"score": m["balanced_accuracy"], "epoch": ep,
                    "state": {k: v.clone() for k, v in model.state_dict().items()}}
        if ep - best["epoch"] >= patience:
            break
    model.load_state_dict(best["state"])
    model.eval()
    with torch.no_grad():
        out = model(Xva)
        p = model.head.probabilities(out).numpy()
        d = model.head.decode(out).numpy()
    npar = sum(q.numel() for q in model.parameters())
    del model, opt
    return p, d, best["epoch"], npar


def run_logreg(Xreg, y, tr, va, l2=10.0):
    sc = Scaler().fit(Xreg, tr)
    Xtr = sc.transform(Xreg[tr]).reshape(len(tr), -1)
    Xva = sc.transform(Xreg[va]).reshape(len(va), -1)
    W = fit_multinomial(Xtr, y[tr], l2=l2)
    p = predict_multinomial(W, Xva)
    return p, p.argmax(1), None, int(W.size)


def main():
    os.makedirs(REP, exist_ok=True)
    reg, fc, y, subjects, mask, pool, test, fold0, smeta = load_all()
    mask_t = torch.from_numpy(mask)
    fcu = np.load(os.path.join(ROOT, "derivatives", "model_ready",
                               "X_fc_roi_unsigned.npy")).astype(np.float64)
    Xflat = np.concatenate([reg.astype(np.float64), fcu[:, :, None]],
                           axis=2).reshape(len(y), -1)
    reg64 = reg.astype(np.float64)

    log("=" * 78)
    log("REPEATED STRATIFIED CROSS-VALIDATION -- 5 folds x 5 split seeds")
    log("=" * 78)
    log(f"training pool {pool.size} subjects; held-out test ({test.size}) NOT loaded")
    log(f"base split fingerprint {smeta['split_fingerprint']}")
    log(f"seeds {SEEDS}")

    candidates = {
        "mlp_ordinal_h32":
            lambda tr, va, sd: run_mlp("ordinal_fixed", Xflat, y, tr, va, sd, 32),
        "mlp_multiclass_h64":
            lambda tr, va, sd: run_mlp("multiclass", Xflat, y, tr, va, sd, 64),
        "transformer_mc_d16":
            lambda tr, va, sd: run_transformer("multiclass", reg, fc, y, mask_t,
                                               tr, va, sd),
        "transformer_ordfix_d16":
            lambda tr, va, sd: run_transformer("ordinal_fixed", reg, fc, y,
                                               mask_t, tr, va, sd),
        "logreg_roi3":
            lambda tr, va, sd: run_logreg(reg64, y, tr, va),
    }

    records = []
    fold_defs = {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        fold_of = stratified_folds(y, pool, N_FOLDS, rng)
        fp = fold_fingerprint(subjects, fold_of, pool)
        fold_defs[str(seed)] = {"fingerprint": fp, "assignment":
                                {subjects[i]: fold_of[int(i)] for i in pool}}
        sizes, cover = [], []
        for f in range(N_FOLDS):
            va = np.array([i for i in pool if fold_of[int(i)] == f])
            sizes.append(len(va))
            cover.append(int(len(set(y[va].tolist()))))
        log(f"\nseed {seed}  fold fingerprint {fp}  val sizes {sizes}  "
            f"classes present per fold {cover}")
        if min(cover) < N_STAGES:
            log(f"  LIMITATION: some folds do not contain all {N_STAGES} "
                f"classes; per-class recall is undefined for absent classes "
                f"and balanced accuracy averages over present classes only")

        for name, fn in candidates.items():
            t0 = time.time()
            for f in range(N_FOLDS):
                va = np.array([i for i in pool if fold_of[int(i)] == f])
                tr = np.array([i for i in pool if fold_of[int(i)] != f])
                assert not (set(tr.tolist()) & set(va.tolist()))
                p, d, be, npar = fn(tr, va, seed + f)
                m = evaluate(p, y[va], pred=d)
                records.append({
                    "candidate": name, "seed": seed, "fold": f,
                    "fold_fingerprint": fp, "n_val": int(len(va)),
                    "best_epoch": be, "n_parameters": npar,
                    **{k: m[k] for k in ("accuracy", "balanced_accuracy",
                                         "macro_f1", "mae_stages", "qwk", "ece",
                                         "n_predicted_classes")},
                    "per_class_recall": m["per_class_recall"],
                    "per_class_f1": m["per_class_f1"],
                    "confusion_matrix": m["confusion_matrix"],
                    "predicted_distribution": m["predicted_distribution"],
                    "subjects": [subjects[i] for i in va],
                    "true": [int(v) for v in y[va]],
                    "pred": [int(v) for v in d],
                    "probs": [[round(float(x), 6) for x in row] for row in p],
                })
            el = time.time() - t0
            sub = [r for r in records if r["candidate"] == name
                   and r["seed"] == seed]
            log(f"  {name:24s} bal.acc "
                f"{np.mean([r['balanced_accuracy'] for r in sub]):.3f}  "
                f"macroF1 {np.mean([r['macro_f1'] for r in sub]):.3f}  "
                f"MAE {np.mean([r['mae_stages'] for r in sub]):.3f}  "
                f"QWK {np.mean([r['qwk'] for r in sub]):+.3f}  "
                f"cls {[r['n_predicted_classes'] for r in sub]}  "
                f"{el:.0f}s")

    out = {"design": {"n_folds": N_FOLDS, "seeds": SEEDS,
                      "evaluations_per_candidate": N_FOLDS * len(SEEDS),
                      "pool_size": int(pool.size),
                      "test_set_touched": False,
                      "fold_definitions": fold_defs},
           "records": records}
    p = os.path.join(REP, "repeated_cv_records.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log(f"\nwrote {os.path.relpath(p, ROOT)}  ({len(records)} fold evaluations)")
    with open(os.path.join(REP, "repeated_cv.log"), "w", encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")


if __name__ == "__main__":
    main()
