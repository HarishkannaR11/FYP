"""
Multi-Attention Transformer on SVM-RFE features (all4, 984 -> 50), 116-subject cohort.

Input
-----
all4 = mALFF, mReHo, DC_z and FC strength (mean |Fisher-z| of each region) for
the 246 Brainnetome regions, 984 features per subject. SVM-RFE, with the exact
settings of the feature-selection study (feature_selection_methods.jobs_step6b:
LinearSVC C=0.01, step 0.1), keeps 50 of them. Every region that owns at least
one kept feature becomes a transformer token:

    token [8] = 4 biomarker slots (only the kept ones filled, others 0, which is
                the training mean after scaling) + 4 flags marking real slots
    network attention gets the FC submatrix between the kept regions
    local attention gets the atlas neighbourhood mask between the kept regions

so the model sees exactly the 50 selected features plus the connectivity
between their regions.

Designs (all 116 subjects are both training and test data)
-----------------------------------------------------------
  split : one stratified 80/20 split (seed 20261020), 93 train / 23 test, once
  cv    : 5 folds x the 5 project seeds on all 116, every subject tested once
          per seed

Leakage control
---------------
  * Scaling and SVM-RFE are fitted on the training subjects of each split only.
  * No test subject is used for any training decision. Epochs are chosen by
    early stopping on INNER folds of the training subjects; the test
    prediction is the mean probability of the 4 inner-fold models
    (CV-bagging). The earlier torch scripts picked the best epoch on the fold
    they reported, which made their CV optimistic.
  * Shuffled-label control: training labels permuted (selection included).

Candidates
----------
  transformer_svmrfe50           primary
  logreg_svmrfe50                same 50 features, logistic regression (study setup)
  transformer_all246             transformer on all 246 regions x 4 biomarkers
  transformer_svmrfe50_SHUFFLED  training labels permuted (cv only)

For the primary model every test subject also gets its stage-pattern outputs
(MC-dropout entropy, embedding-axis score, pooling weights, occlusion
importance, branch weights, attention maps), analysed by stage_patterns.py.

Run:
  python scripts/model/train_transformer_svmrfe.py --smoke
  python scripts/model/train_transformer_svmrfe.py --eval both --candidates transformer_svmrfe50 logreg_svmrfe50
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feature_selection_cv as H  # noqa: E402  (sets thread env vars first)
import feature_selection_methods as M  # noqa: E402

import numpy as np  # noqa: E402
import sklearn  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.feature_selection import RFE  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402

from heads import StagePredictor, build_head  # noqa: E402
from make_splits import stratified_holdout  # noqa: E402
from multi_attention_transformer import build_model  # noqa: E402
from train_transformer import Scaler, evaluate, load_all  # noqa: E402

ROOT = H.ROOT
OUT = os.path.join(ROOT, "derivatives", "model_training", "transformer_svmrfe")
STAGES = H.STAGES
N_STAGES = H.N_STAGES
N_ROI = 246
CHANNELS = ("mALFF", "mReHo", "DC_z", "FC")

SPLIT_SEED = 20261020
TEST_FRACTION = 0.20
N_SELECT = 50
N_INNER = 4
ARCH = {"d_model": 16, "num_heads": 2, "num_layers": 1, "dropout": 0.3}
TRAIN = {"lr": 1e-3, "weight_decay": 1e-2, "batch_size": 8, "grad_clip": 1.0,
         "max_epochs": 60, "patience": 15}
MC_SAMPLES = 30
CHUNK = 8
CANDIDATES = ["transformer_svmrfe50", "logreg_svmrfe50",
              "transformer_svmrfe50_SHUFFLED", "transformer_all246"]
PRIMARY = "transformer_svmrfe50"
METRIC_KEYS = ("accuracy", "balanced_accuracy", "macro_f1", "mae_stages", "qwk",
               "ece", "n_predicted_classes", "per_class_recall",
               "confusion_matrix", "predicted_distribution")

_lines = []


def log(m=""):
    print(m, flush=True)
    _lines.append(str(m))


def r6(a):
    return np.round(np.asarray(a, dtype=float), 6).tolist()


# ------------------------------------------------------------------ data
def load_cohort():
    reg, fc, y, subjects, mask, _, _, _, _ = load_all()
    fcu = np.load(os.path.join(H.SRC, "X_fc_roi_unsigned.npy")).astype(np.float32)
    if not np.isfinite(fcu).all():
        raise SystemExit("non-finite FC strength")
    X4 = np.concatenate([reg, fcu[:, :, None]], axis=2)          # [116, 246, 4]
    # flat all4 in the study's order: 738 regional (roi-major) then 246 FC
    flat = np.hstack([reg.reshape(len(y), -1), fcu]).astype(np.float64)
    names = M.feature_names("all4")
    assert len(names) == flat.shape[1] == 984
    return X4, flat, fc, y, subjects, mask.astype(bool)


def feature_slot(i):
    """Flat all4 index -> (roi, channel)."""
    return (i // 3, i % 3) if i < 738 else (i - 738, 3)


def split_80_20(y, subjects):
    tr, te = stratified_holdout(y, TEST_FRACTION, np.random.default_rng(SPLIT_SEED))
    assert not set(tr.tolist()) & set(te.tolist())
    assert tr.size + te.size == y.size
    rows = [[subjects[i], STAGES[y[i]], i, "test" if i in set(te.tolist()) else "train"]
            for i in range(y.size)]
    h = hashlib.sha256()
    for r in rows:
        h.update(f"{r[0]}|{r[3]}".encode())
    fp = h.hexdigest()[:16]
    meta = {"seed": SPLIT_SEED, "test_fraction": TEST_FRACTION, "split_fingerprint": fp,
            "n_train": int(tr.size), "n_test": int(te.size),
            "train_class_counts": {s: int((y[tr] == g).sum()) for g, s in enumerate(STAGES)},
            "test_class_counts": {s: int((y[te] == g).sum()) for g, s in enumerate(STAGES)}}
    pj = os.path.join(OUT, "split_80_20.json")
    if os.path.exists(pj):
        old = json.load(open(pj))
        if old["split_fingerprint"] != fp:
            raise SystemExit("split_80_20.json exists with a different fingerprint")
    else:
        with open(os.path.join(OUT, "split_80_20.csv"), "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["subject_id", "stage", "row_index", "partition"])
            w.writerows(rows)
        json.dump(meta, open(pj, "w"), indent=2)
    return tr, te, meta


def cv_plan(y, subjects, n_folds=H.N_FOLDS):
    pool = np.arange(y.size)
    plan = {}
    for seed in H.SEEDS:
        fo = H.stratified_folds(y, pool, n_folds, np.random.default_rng(seed))
        fp = H.fold_fingerprint(subjects, fo, pool)
        folds = []
        for f in range(n_folds):
            te = np.array([i for i in pool if fo[int(i)] == f])
            tr = np.array([i for i in pool if fo[int(i)] != f])
            assert not set(tr.tolist()) & set(te.tolist())
            folds.append((tr, te))
        tested = np.sort(np.concatenate([te for _, te in folds]))
        assert np.array_equal(tested, pool), "every subject must be tested once per seed"
        plan[seed] = (fp, folds)
    return plan


# ------------------------------------------------------------- selection
def svmrfe_pipeline(seed):
    # random_state only fixes liblinear's dual-CD shuffling, for reproducibility
    return M.pipe(selector=RFE(LinearSVC(C=0.01, max_iter=5000, random_state=seed),
                               n_features_to_select=N_SELECT, step=0.1))


def fit_selection(flat, y_used, tr, seed):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = svmrfe_pipeline(seed).fit(flat[tr], y_used[tr])
    support = np.flatnonzero(pipe.named_steps["select"].get_support())
    chmask = np.zeros((N_ROI, 4), dtype=bool)
    for i in support:
        chmask[feature_slot(int(i))] = True
    assert chmask.sum() == N_SELECT, "exactly 50 real slots"
    return pipe, support, chmask


def make_tokens(Xs, chmask):
    """[N, 246, 4] scaled -> [N, K, 8]: kept slots + presence flags."""
    rois = np.flatnonzero(chmask.any(axis=1))
    cm = chmask[rois].astype(np.float32)                         # [K, 4]
    vals = Xs[:, rois, :] * cm[None]
    flags = np.broadcast_to(cm[None], vals.shape)
    return np.concatenate([vals, flags], axis=2).astype(np.float32), rois


# ------------------------------------------------------------- training
def class_weights(y):
    c = np.bincount(y, minlength=N_STAGES).astype(np.float32)
    return torch.from_numpy(np.where(c > 0, c.sum() / (N_STAGES * np.maximum(c, 1)), 0.0)
                            .astype(np.float32))


def new_model(n_roi, n_feat, seed):
    torch.manual_seed(seed)
    enc = build_model(n_roi=n_roi, n_features=n_feat, **ARCH)
    return StagePredictor(enc, build_head("multiclass", ARCH["d_model"], dropout=ARCH["dropout"]))


@torch.no_grad()
def probs_of(model, X, F, mask_t):
    model.eval()
    out = [model.head.probabilities(model(X[s:s + CHUNK], F[s:s + CHUNK], mask_t))
           for s in range(0, X.shape[0], CHUNK)]
    return torch.cat(out).numpy()


def train_model(X, F, mask_t, y_used, itr, iva, seed, max_epochs):
    """One inner model, early-stopped on its INNER validation subjects."""
    model = new_model(X.shape[1], X.shape[2], seed)
    opt = torch.optim.AdamW(model.parameters(), lr=TRAIN["lr"],
                            weight_decay=TRAIN["weight_decay"])
    w = class_weights(y_used[itr])
    Xtr, Ftr = X[itr], F[itr]
    ytr = torch.from_numpy(y_used[itr])
    Xva, Fva = X[iva], F[iva]
    g = torch.Generator().manual_seed(seed)
    best = {"score": -np.inf, "epoch": -1, "state": None}
    bs = TRAIN["batch_size"]
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(itr), generator=g)
        for s in range(0, len(perm), bs):
            b = perm[s:s + bs]
            opt.zero_grad()
            loss = model.head.loss(model(Xtr[b], Ftr[b], mask_t), ytr[b], weight=w)
            if not torch.isfinite(loss):
                raise SystemExit("non-finite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), TRAIN["grad_clip"])
            opt.step()
        pv = probs_of(model, Xva, Fva, mask_t)
        score = H.evaluate(y_used[iva], pv.argmax(1))["balanced_accuracy"]
        if score > best["score"]:
            best = {"score": score, "epoch": ep,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        if ep - best["epoch"] >= TRAIN["patience"]:
            break
    model.load_state_dict(best["state"])
    model.eval()
    return model, best["epoch"]


def fit_bag(X, F, mask_t, y_used, tr, seed, max_epochs):
    """4 inner folds of the training subjects -> 4 early-stopped models."""
    fo = H.stratified_folds(y_used, tr, N_INNER, np.random.default_rng([seed, 7]))
    models, epochs = [], []
    for j in range(N_INNER):
        itr = np.array([i for i in tr if fo[int(i)] != j])
        iva = np.array([i for i in tr if fo[int(i)] == j])
        m, ep = train_model(X, F, mask_t, y_used, itr, iva, seed * 10 + j, max_epochs)
        models.append(m)
        epochs.append(ep)
    return models, epochs


def bag_probs(models, X, F, mask_t):
    return np.mean([probs_of(m, X, F, mask_t) for m in models], axis=0)


# ------------------------------------------------------- stage patterns
@torch.no_grad()
def explain(models, X, F, mask_t, y_used, tr, te, y_true_te, p_te):
    """Per test subject: entropy, embedding-axis score, pooling weights,
    occlusion importance, branch weights; and mean attention maps."""
    Xte, Fte = X[te], F[te]
    n, K = len(te), X.shape[1]

    # MC-dropout, pooled over the 4 models
    pm = torch.stack([m.predict_mc(Xte, Fte, mask_t, MC_SAMPLES)[0] for m in models]).mean(0)
    pm = pm / pm.sum(1, keepdim=True)
    entropy = -(pm * torch.log(pm.clamp_min(1e-12))).sum(1).numpy()

    axis, pool, branch, att_g, att_n = [], [], [], [], []
    for m in models:
        m.eval()
        # first principal axis of the TRAINING embeddings; sign oriented with the
        # training labels, scaled by the training projection
        e_tr = torch.cat([m.encoder(X[tr][s:s + CHUNK], F[tr][s:s + CHUNK], mask_t)
                          ["brain_embedding"] for s in range(0, len(tr), CHUNK)]).numpy()
        mu = e_tr.mean(0)
        v = np.linalg.svd(e_tr - mu, full_matrices=False)[2][0]
        p_tr = (e_tr - mu) @ v
        if np.corrcoef(p_tr, y_used[tr])[0, 1] < 0:
            v, p_tr = -v, -p_tr
        sd = p_tr.std() if p_tr.std() > 0 else 1.0
        o = m.encoder(Xte, Fte, mask_t, return_attention=True)
        axis.append(((o["brain_embedding"].numpy() - mu) @ v) / sd)
        pool.append(o["pool_weights"].numpy())                         # [n, K]
        branch.append(o["branch_weights"].numpy())                     # [n, K, 3]
        att_g.append(o["attention"][0]["global"].mean(1).numpy())      # [n, K, K]
        att_n.append(o["attention"][0]["network"].mean(1).numpy())

    # occlusion: blank one token (values and flags), drop in P(true stage)
    rows = np.arange(n)
    occl = np.zeros((n, K))
    for k in range(K):
        Xo = Xte.clone()
        Xo[:, k, :] = 0.0
        po = bag_probs(models, Xo, Fte, mask_t)
        occl[:, k] = p_te[rows, y_true_te] - po[rows, y_true_te]

    return {"entropy": entropy,
            "emb_axis": np.mean(axis, axis=0),
            "pool_w": np.mean(pool, axis=0),
            "occlusion": occl,
            "branch_w": np.mean(branch, axis=0).mean(axis=1),          # [n, 3] global, local, network
            "att_global": np.mean(att_g, axis=0),
            "att_network": np.mean(att_n, axis=0)}


# ------------------------------------------------------------ one split
def run_split(cand, data, tr, te, seed, fold, fp, max_epochs, acc, smoke=False, explain_outputs=True):
    X4, flat, fc, y, subjects, mask = data
    assert not set(tr.tolist()) & set(te.tolist())
    shuffled = cand.endswith("_SHUFFLED")
    y_used = y.copy()
    if shuffled:
        ytr = y[tr].copy()
        np.random.default_rng(seed * 100 + fold).shuffle(ytr)
        y_used[tr] = ytr
    t0 = time.time()
    rec = {"candidate": cand, "seed": seed, "fold": fold, "fold_fingerprint": fp,
           "n_train": int(tr.size), "n_test": int(te.size), "shuffled_labels": shuffled,
           "test_index": [int(i) for i in te], "subjects": [str(subjects[i]) for i in te],
           "true": [int(v) for v in y[te]]}

    if cand == "logreg_svmrfe50":
        pipe, support, _ = fit_selection(flat, y_used, tr, seed + fold)
        p_te = pipe.predict_proba(flat[te])
        pred = pipe.predict(flat[te]).astype(int)
        p_tr_pred = pipe.predict(flat[tr]).astype(int)
        rec["support"] = support.tolist()
        rec["train_balanced_accuracy"] = H.evaluate(y_used[tr], p_tr_pred)["balanced_accuracy"]
    else:
        sc = Scaler().fit(X4, tr)                                   # TRAINING SUBJECTS ONLY
        Xs = sc.transform(X4).astype(np.float32)
        if cand == "transformer_all246":
            tok, rois = Xs, np.arange(N_ROI)
            F, mask_k = fc, mask
        else:
            _, support, chmask = fit_selection(flat, y_used, tr, seed + fold)
            tok, rois = make_tokens(Xs, chmask)
            F = fc[:, rois][:, :, rois]
            mask_k = mask[np.ix_(rois, rois)]
            rec["support"] = support.tolist()
            assert tok.shape[1:] == (rois.size, 8)
            assert np.allclose(tok[:, :, 4:].sum(axis=(1, 2)), N_SELECT)
        X = torch.from_numpy(np.ascontiguousarray(tok))
        Ft = torch.from_numpy(np.ascontiguousarray(F, dtype=np.float32))
        mask_t = torch.from_numpy(np.ascontiguousarray(mask_k))
        if smoke:
            log(f"    tokens {tuple(X.shape)}  fc {tuple(Ft.shape)}  mask {tuple(mask_t.shape)}  "
                f"K={rois.size}")
        models, epochs = fit_bag(X, Ft, mask_t, y_used, tr, seed * 100 + fold, max_epochs)
        p_te = bag_probs(models, X[te], Ft[te], mask_t)
        pred = p_te.argmax(1)
        p_tr = bag_probs(models, X[tr], Ft[tr], mask_t)
        rec["train_balanced_accuracy"] = H.evaluate(y_used[tr], p_tr.argmax(1))["balanced_accuracy"]
        rec["rois"] = rois.tolist()
        rec["inner_best_epochs"] = epochs
        rec["n_parameters"] = int(sum(q.numel() for q in models[0].parameters()))
        if cand == PRIMARY and explain_outputs:
            ex = explain(models, X, Ft, mask_t, y_used, tr, te, y[te], p_te)
            for k in ("entropy", "emb_axis", "pool_w", "occlusion", "branch_w"):
                rec[k] = r6(ex[k])
            if not (np.isfinite(ex["att_global"]).all() and np.isfinite(ex["att_network"]).all()):
                raise SystemExit("non-finite attention")
            if acc is not None:                       # stage-averaged attention, 246 x 246
                ix = np.ix_(rois, rois)
                for j, g in enumerate(y[te]):
                    acc["global"][g][ix] += ex["att_global"][j]
                    acc["network"][g][ix] += ex["att_network"][j]
                    acc["count"][g][ix] += 1
            if smoke:
                log(f"    explain: entropy {ex['entropy'].shape} pool {ex['pool_w'].shape} "
                    f"occlusion {ex['occlusion'].shape} attention {ex['att_global'].shape}")

    if not np.isfinite(p_te).all():
        raise SystemExit("non-finite probabilities")
    m = evaluate(np.asarray(p_te), y[te], pred=pred)
    rec.update({k: m[k] for k in METRIC_KEYS})
    rec["pred"] = [int(v) for v in pred]
    rec["probs"] = r6(p_te)
    rec["seconds"] = round(time.time() - t0, 1)
    log(f"    seed {seed} fold {fold}  K={len(rec.get('rois', [])) or '-':>3}  "
        f"test bal.acc {m['balanced_accuracy']:.3f}  train {rec['train_balanced_accuracy']:.3f}  "
        f"MAE {m['mae_stages']:.2f}  QWK {m['qwk']:+.3f}  cls {m['n_predicted_classes']}  "
        f"epochs {rec.get('inner_best_epochs', '-')}  {rec['seconds']:.0f}s")
    return rec


def new_acc():
    return {k: np.zeros((N_STAGES, N_ROI, N_ROI)) for k in ("global", "network", "count")}


def save(design, cand, records, extra, acc):
    stem = os.path.join(OUT, f"{design}_{cand}")
    json.dump({"extra": extra, "records": records}, open(stem + ".json", "w"))
    open(stem + ".log", "w", encoding="utf-8").write("\n".join(_lines) + "\n")
    if acc is not None:
        np.savez_compressed(stem + "_attention.npz", **acc)
    log(f"  wrote {os.path.relpath(stem, ROOT)}.json")


def header(design, cand, extra_line):
    _lines.clear()
    log("=" * 100)
    log(f"TRANSFORMER ON SVM-RFE FEATURES -- design {design}, candidate {cand}")
    log("=" * 100)
    log(extra_line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", choices=["split", "cv", "both"], default="both")
    ap.add_argument("--candidates", nargs="+", default=CANDIDATES, choices=CANDIDATES)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--smoke", action="store_true",
                    help="one CV fold of the primary, 3 epochs, nothing saved")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(OUT, exist_ok=True)

    data = load_cohort()
    X4, flat, fc, y, subjects, mask = data
    versions = {"python": platform.python_version(), "numpy": np.__version__,
                "scikit-learn": sklearn.__version__, "torch": torch.__version__}
    counts = {s: int((y == g).sum()) for g, s in enumerate(STAGES)}
    common = (f"cohort {y.size} subjects {counts}; versions {json.dumps(versions)}\n"
              f"architecture {ARCH}; training {TRAIN}; inner folds {N_INNER} (bagged)")
    plan = cv_plan(y, subjects)

    if args.smoke:
        _lines.clear()
        log("SMOKE: primary, CV seed 1 fold 0, 3 epochs, nothing saved")
        fp, folds = plan[H.SEEDS[0]]
        tr, te = folds[0]
        log(f"  train {tr.size} test {te.size}  fold fingerprint {fp}")
        run_split(PRIMARY, data, tr, te, H.SEEDS[0], 0, fp, 3, new_acc(), smoke=True)
        log("SMOKE OK")
        return

    designs = ["split", "cv"] if args.eval == "both" else [args.eval]
    for design in designs:
        for cand in args.candidates:
            if design == "split" and cand.endswith("_SHUFFLED"):
                continue
            dname = "split_80_20" if design == "split" else "cv116"
            if os.path.exists(os.path.join(OUT, f"{dname}_{cand}.json")):
                print(f"COLLISION: {dname}_{cand}.json exists; skipped, not overwritten")
                continue
            acc = new_acc() if cand == PRIMARY else None
            records = []
            t0 = time.time()
            if design == "split":
                tr, te, meta = split_80_20(y, subjects)
                header(dname, cand, common + f"\nsplit {meta}")
                records.append(run_split(cand, data, tr, te, SPLIT_SEED, 0,
                                         meta["split_fingerprint"], TRAIN["max_epochs"], acc))
                extra = {"design": dname, "split": meta}
            else:
                header(dname, cand, common + "\nfold fingerprints "
                       + json.dumps({s: plan[s][0] for s in H.SEEDS}))
                for seed in H.SEEDS:
                    fp, folds = plan[seed]
                    for f, (tr, te) in enumerate(folds):
                        records.append(run_split(cand, data, tr, te, seed, f, fp,
                                                 TRAIN["max_epochs"], acc))
                    sub = [r for r in records if r["seed"] == seed]
                    log(f"  seed {seed}: bal.acc {np.mean([r['balanced_accuracy'] for r in sub]):.3f}")
                s = H.summarise(records)
                log(f"  {cand}: seed-level bal.acc {s['balanced_accuracy'][0]:.3f} +/- "
                    f"{s['balanced_accuracy'][1]:.3f}  macroF1 {s['macro_f1'][0]:.3f}  "
                    f"MAE {s['mae_stages'][0]:.3f}  QWK {s['qwk'][0]:+.3f}  "
                    f"train {np.mean([r['train_balanced_accuracy'] for r in records]):.3f}")
                extra = {"design": dname,
                         "fold_fingerprints": {str(s_): plan[s_][0] for s_ in H.SEEDS}}
            extra.update({"candidate": cand, "versions": versions, "arch": ARCH,
                          "train": TRAIN, "n_inner": N_INNER, "n_select": N_SELECT,
                          "seconds": round(time.time() - t0, 1)})
            log(f"  elapsed {time.time() - t0:.0f}s")
            save(dname, cand, records, extra, acc)


if __name__ == "__main__":
    main()
