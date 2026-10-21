"""
Transformer with properly nested selection (experiment v2).

Fixes the leak found by transformer_diagnostics_v1: there the scaler and
SVM-RFE were fitted once on the whole outer training set, so the inner-
validation subjects used for early stopping had already helped choose the
features (inner-validation balanced accuracy ran 4-8 points above the
held-out fold even with no epoch selection). Here every inner model gets its
own scaler and its own SVM-RFE, fitted on its inner-training subjects only.

Per outer split (5 folds x 5 seeds on all 116, identical to the TabPFN k=50
baseline in derivatives/model_training/tabpfn/cv116_tabpfn_svmrfe50.json):
  inner folds   the same 4 stratified inner folds and torch seeds as C0
  per inner j   Scaler + SVM-RFE (all4, k = 50) fitted on itr_j only, tokens built
                from them; C0 settings unchanged (d16, 2 heads, 1 layer, dropout 0.3,
                AdamW lr 1e-3, wd 1e-2, batch 8, clip 1.0, class weights from itr_j);
                trained for 60 epochs, logged every epoch

Label access, enforced in code (and tested by audit_nested.py):
  * training and selection receive y_sel, a copy of y with every outer-test
    label set to -1;
  * scaling and SVM-RFE receive only itr_j rows, from a copy in which the
    iva_j labels are also -1;
  * inner-validation labels are read only to choose the checkpoint;
  * outer-test labels are read only in score_split(), after every checkpoint
    choice has been frozen.

Checkpoint rules, declared before any result was seen:
  ES         each inner model keeps its best inner-validation balanced-accuracy
             epoch (patience 15, exactly as C0)
  FIXED      PRIMARY. One epoch E* per outer split: the epoch minimising the
             mean inner-validation cross-entropy of the 4 inner models (ties ->
             earliest). All 4 models are read at E*.
  FIXED_BAL  sensitivity. E* maximising the balanced accuracy of the pooled
             inner out-of-fold predictions (all training subjects; ties -> earliest).
  LAST       epoch 60, no selection (diagnostic)
The prediction of every rule is the mean probability of the 4 inner models
(CV-bagging, as in C0). No rule is chosen using outer-fold results.

Outputs: derivatives/experiments/transformer_nested_v2/grid/cv5_seed{s}_fold{f}.json,
written once and skipped if present (resumable). A lock file stops a second
copy from running, and the manifest pins this script's hash so a resumed run
cannot mix code versions.

Run:  python scripts/experiments/transformer_nested_v2/run_nested.py --seeds all
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "model"))
import train_transformer_svmrfe as S  # noqa: E402  (sets thread env vars first)

import numpy as np  # noqa: E402
import sklearn  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

from heads import StagePredictor, build_head  # noqa: E402
from multi_attention_transformer import build_model  # noqa: E402
from train_transformer import Scaler  # noqa: E402

H = S.H
ROOT = S.ROOT
EXP = os.path.join(ROOT, "derivatives", "experiments", "transformer_nested_v2")
GRID = os.path.join(EXP, "grid")
CFG = {**S.ARCH, "lr": S.TRAIN["lr"], "weight_decay": S.TRAIN["weight_decay"]}      # C0, unchanged
MAX_EPOCHS, PATIENCE, BS, CLIP = S.TRAIN["max_epochs"], S.TRAIN["patience"], S.TRAIN["batch_size"], S.TRAIN["grad_clip"]
RULES = ("ES", "FIXED", "FIXED_BAL", "LAST")
PRIMARY = "FIXED"
HIDDEN = -1


def script_hash():
    return hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]


def hide(y, idx):
    """Copy of y with the labels at idx replaced by HIDDEN."""
    out = y.copy()
    out[idx] = HIDDEN
    return out


# ------------------------------------------------------------- inputs
def inner_inputs(d, y_sel, itr, iva, seed_tag):
    """Scaler + SVM-RFE fitted on the inner-training subjects only -> tokens for every subject."""
    X4, flat, fc, _, _, mask = d
    y_fit = hide(y_sel, iva)
    assert (y_fit[itr] >= 0).all()
    sc = Scaler().fit(X4, itr)
    Xs = sc.transform(X4).astype(np.float32)
    _, support, chmask = S.fit_selection(flat, y_fit, itr, seed_tag)
    tok, rois = S.make_tokens(Xs, chmask)
    X = torch.from_numpy(np.ascontiguousarray(tok))
    F = torch.from_numpy(np.ascontiguousarray(fc[:, rois][:, :, rois], dtype=np.float32))
    mask_t = torch.from_numpy(np.ascontiguousarray(mask[np.ix_(rois, rois)]))
    return X, F, mask_t, support, rois, sc


# ------------------------------------------------------------- training
@torch.no_grad()
def logits_on(model, X, F, mask_t, idx):
    model.eval()
    return torch.cat([model(X[idx[s:s + S.CHUNK]], F[idx[s:s + S.CHUNK]], mask_t)
                      for s in range(0, len(idx), S.CHUNK)])


def scores(logits, yy):
    probs = torch.softmax(logits, 1).numpy()
    m = H.evaluate(yy, probs.argmax(1))
    return probs, {"loss": float(Fn.cross_entropy(logits, torch.from_numpy(yy))), "acc": m["accuracy"],
                   "bal": m["balanced_accuracy"], "f1": m["macro_f1"]}


def train_inner(X, F, mask_t, y_sel, itr, iva, te, seed, max_epochs=MAX_EPOCHS):
    """C0's loop (same seeding, batching, clipping, weighted criterion), run to max_epochs.
    Sees y_sel only. Outer-test probabilities are stored per epoch without reading any label."""
    assert (y_sel[itr] >= 0).all() and (y_sel[iva] >= 0).all() and (y_sel[te] == HIDDEN).all()
    torch.manual_seed(seed)
    enc = build_model(n_roi=X.shape[1], n_features=X.shape[2], d_model=CFG["d_model"], num_heads=CFG["num_heads"],
                      num_layers=CFG["num_layers"], dropout=CFG["dropout"])
    model = StagePredictor(enc, build_head("multiclass", CFG["d_model"], dropout=CFG["dropout"]))
    opt = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    w = S.class_weights(y_sel[itr])
    Xtr, Ftr, ytr = X[itr], F[itr], torch.from_numpy(y_sel[itr])
    g = torch.Generator().manual_seed(seed)
    curves = {s: {k: [] for k in ("loss", "acc", "bal", "f1")} for s in ("train", "val")}
    val_probs, outer_logits = [], []
    best, best_ep, stop_ep = -np.inf, -1, None
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(itr), generator=g)
        for s in range(0, len(perm), BS):
            b = perm[s:s + BS]
            opt.zero_grad()
            loss = model.head.loss(model(Xtr[b], Ftr[b], mask_t), ytr[b], weight=w)
            if not torch.isfinite(loss):
                raise SystemExit("non-finite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CLIP)
            opt.step()
        pv, mv = scores(logits_on(model, X, F, mask_t, iva), y_sel[iva])
        _, mt = scores(logits_on(model, X, F, mask_t, itr), y_sel[itr])
        outer_logits.append(logits_on(model, X, F, mask_t, te))
        for k in mv:
            curves["val"][k].append(mv[k])
            curves["train"][k].append(mt[k])
        val_probs.append(pv)
        if stop_ep is None:                          # online early stopping, as in S.train_model
            if mv["bal"] > best:
                best, best_ep = mv["bal"], ep
            if ep - best_ep >= PATIENCE:
                stop_ep = ep
    return {"curves": curves, "val_probs": np.array(val_probs), "outer_logits": torch.stack(outer_logits),
            "best_epoch": best_ep, "stop_epoch": stop_ep if stop_ep is not None else max_epochs - 1,
            "n_parameters": sum(p.numel() for p in model.parameters())}


# ------------------------------------------------------------- selection
def select_epochs(inner, y_sel):
    """Every checkpoint rule, from inner-validation results only. Returns the epoch of each inner model per rule."""
    n_ep = len(inner[0]["curves"]["val"]["loss"])
    vloss = np.mean([m["curves"]["val"]["loss"] for m in inner], 0)
    yv = np.concatenate([y_sel[m["iva"]] for m in inner])
    assert (yv >= 0).all()

    def pooled_bal(epochs):
        return H.evaluate(yv, np.concatenate([m["val_probs"][e].argmax(1) for m, e in zip(inner, epochs)]))[
            "balanced_accuracy"]

    pooled = np.array([pooled_bal([e] * len(inner)) for e in range(n_ep)])
    fixed, fixed_bal = int(np.argmin(vloss)), int(np.argmax(pooled))
    epochs = {"ES": [int(m["best_epoch"]) for m in inner], "FIXED": [fixed] * len(inner),
              "FIXED_BAL": [fixed_bal] * len(inner), "LAST": [n_ep - 1] * len(inner)}
    inner_bal = {r: float(pooled_bal(e)) for r, e in epochs.items()}           # pooled inner out-of-fold, per rule
    return epochs, inner_bal, vloss, pooled


def bag_probs(inner, epochs):
    return np.mean([torch.softmax(m["outer_logits"][e], 1).numpy() for m, e in zip(inner, epochs)], 0)


# ------------------------------------------------------------- one split
def fit_split(d, y_sel, seed, fold, tr, te, max_epochs=MAX_EPOCHS):
    """Everything that learns or selects. Never sees an outer-test label."""
    assert (y_sel[te] == HIDDEN).all() and (y_sel[tr] >= 0).all()
    bag_seed = seed * 100 + fold
    fo = H.stratified_folds(y_sel, tr, S.N_INNER, np.random.default_rng([bag_seed, 7]))
    inner = []
    for j in range(S.N_INNER):
        itr = np.array([i for i in tr if fo[int(i)] != j])
        iva = np.array([i for i in tr if fo[int(i)] == j])
        X, F, mask_t, support, rois, sc = inner_inputs(d, y_sel, itr, iva, seed + fold)
        m = train_inner(X, F, mask_t, y_sel, itr, iva, te, bag_seed * 10 + j, max_epochs)
        m.update({"itr": itr, "iva": iva, "support": support, "rois": rois,
                  "scaler_hash": hashlib.sha256(np.concatenate([sc.mean.ravel(), sc.std.ravel()]).tobytes()).hexdigest()[:16]})
        inner.append(m)
    epochs, inner_bal, vloss, pooled = select_epochs(inner, y_sel)
    probs = {r: bag_probs(inner, e) for r, e in epochs.items()}
    return inner, epochs, inner_bal, vloss, pooled, probs


def score_split(y, te, inner, probs):
    """Reads outer-test labels. Called only after fit_split has returned (all choices frozen)."""
    yt = y[te]
    out = {}
    for r, p in probs.items():
        m = H.evaluate(yt, p.argmax(1))
        out[r] = {"accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"], "macro_f1": m["macro_f1"],
                  "per_class_recall": m["per_class_recall"]}
    outer_curves = []                                # diagnostic only: each inner model on the outer fold, every epoch
    for m in inner:
        c = {k: [] for k in ("loss", "acc", "bal", "f1")}
        for e in range(m["outer_logits"].shape[0]):
            _, s = scores(m["outer_logits"][e], yt)
            for k in c:
                c[k].append(s[k])
        outer_curves.append(c)
    return out, outer_curves


def run_split(d, y, seed, fold, fp, tr, te):
    y_sel = hide(y, te)
    inner, epochs, inner_bal, vloss, pooled, probs = fit_split(d, y_sel, seed, fold, tr, te)
    frozen = json.dumps(epochs, sort_keys=True)
    metrics, outer_curves = score_split(y, te, inner, probs)
    assert json.dumps(epochs, sort_keys=True) == frozen
    return {"seed": seed, "fold": fold, "fold_fingerprint": fp, "settings": CFG, "script_sha256": script_hash(),
            "test_idx": te.tolist(), "true": y[te].tolist(),
            "epochs": epochs, "inner_oof_bal": inner_bal,
            "mean_inner_val_loss": S.r6(vloss), "pooled_inner_val_bal": S.r6(pooled),
            "pred": {r: p.argmax(1).tolist() for r, p in probs.items()},
            "probs": {r: S.r6(p) for r, p in probs.items()},
            "heldout": metrics,
            "inner_models": [{"n_train": int(m["itr"].size), "n_val": int(m["iva"].size), "support": m["support"].tolist(),
                              "K": int(m["rois"].size), "n_parameters": int(m["n_parameters"]),
                              "scaler_hash": m["scaler_hash"], "best_epoch": int(m["best_epoch"]),
                              "stop_epoch": int(m["stop_epoch"]), "curves": m["curves"], "outer_curves": oc}
                             for m, oc in zip(inner, outer_curves)]}


# ------------------------------------------------------------- driver
def manifest(seeds):
    path = os.path.join(EXP, "manifest.json")
    info = {"experiment": "transformer_nested_v2", "script": os.path.relpath(os.path.abspath(__file__), ROOT),
            "script_sha256": script_hash(), "settings": CFG, "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "rules": list(RULES), "primary_rule": PRIMARY, "seeds": seeds, "n_folds": 5, "n_inner": S.N_INNER,
            "n_select": S.N_SELECT, "inputs": "rs-fMRI only (all4, 984 features); no clinical variables",
            "tabpfn_baseline": "derivatives/model_training/tabpfn/cv116_tabpfn_svmrfe50.json",
            "python": platform.python_version(), "torch": torch.__version__, "sklearn": sklearn.__version__,
            "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    if os.path.exists(path):
        old = json.load(open(path))
        if old["script_sha256"] != info["script_sha256"]:
            raise SystemExit("run_nested.py changed since this experiment started; refusing to mix code versions")
        return
    os.makedirs(EXP, exist_ok=True)
    json.dump(info, open(path, "w"), indent=2)


def acquire_lock():
    import psutil
    path = os.path.join(EXP, "run.lock")
    if os.path.exists(path):
        pid = int(open(path).read().strip() or 0)
        if pid and psutil.pid_exists(pid):
            raise SystemExit(f"another run_nested.py is running (pid {pid}); not starting a duplicate")
    os.makedirs(EXP, exist_ok=True)
    open(path, "w").write(str(os.getpid()))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", default=["all"])
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    seeds = H.SEEDS if args.seeds == ["all"] else [int(s) for s in args.seeds]
    manifest(seeds)
    lock = acquire_lock()
    try:
        d = S.load_cohort()
        y = d[3]
        plan = S.cv_plan(y, d[4], 5)
        for seed in seeds:
            fp, folds = plan[seed]
            for f, (tr, te) in enumerate(folds):
                path = os.path.join(GRID, f"cv5_seed{seed}_fold{f}.json")
                if os.path.exists(path):
                    continue
                t0 = time.time()
                rec = run_split(d, y, seed, f, fp, tr, te)
                rec["seconds"] = round(time.time() - t0, 1)
                os.makedirs(GRID, exist_ok=True)
                json.dump(rec, open(path + ".tmp", "w"))
                os.replace(path + ".tmp", path)
                print(f"seed {seed} fold {f}: E*(loss) {rec['epochs']['FIXED'][0] + 1}, "
                      f"E*(bal) {rec['epochs']['FIXED_BAL'][0] + 1}, ES {[e + 1 for e in rec['epochs']['ES']]}, "
                      f"K {[m['K'] for m in rec['inner_models']]}  {rec['seconds']:.0f}s", flush=True)
    finally:
        os.remove(lock)


if __name__ == "__main__":
    main()
