"""
Episodic meta-learning (Block 4): Reptile across biomarker tasks, then
fine-tuning of the Multi-Attention Transformer. 116-subject cohort.

Why Reptile
-----------
Reptile (Nichol et al. 2018) is the first-order member of the MAML family: it
learns an initialisation from which a few gradient steps solve each task well.
Full second-order MAML is costly and unstable at this sample size; Reptile
keeps the same idea with first-order updates.

Tasks = biomarkers (cross-biomarker transfer)
--------------------------------------------
Inside each training set, SVM-RFE picks 50 features as before; their ~47
regions are the tokens. A biomarker TASK shows the transformer ONE biomarker
on all those regions: task b = mALFF, mReHo, DC_z or FC strength, its slot
filled for every token and its flag set, the other slots 0. Each Reptile
iteration samples one task and a batch of training subjects, runs a few inner
Adam steps from the shared weights theta, and moves theta toward the adapted
weights. The resulting theta is one starting point that adapts to every
biomarker: the "shared, biomarker-invariant representation" of Block 4.

Fine-tuning
-----------
theta initialises the encoder (and head) of the same model used everywhere
else, trained on the NORMAL input (the 50 selected features as in
train_transformer_svmrfe.py), with the same inner 4-fold early stopping and
bagging, the same seeds and the same folds. Only the starting weights differ,
so the comparison with the from-scratch transformer is paired fold by fold.

  meta_transformer_svmrfe50               rs-fMRI only        vs transformer_svmrfe50
  meta_transformer_svmrfe50_comorb8       + 8 comorbidities   vs model A (fusion, gate)
  meta_transformer_svmrfe50_comorb8_SHUFFLED  training labels permuted (meta AND fine-tune; cv only)

Leakage: scaling, SVM-RFE, Reptile and fine-tuning see the training subjects
of the split only. Reptile uses all of them, including the inner-validation
subjects of the later bagging, which can only make inner early stopping
slightly optimistic, never the reported test fold.

Run:  python scripts/model/train_meta_reptile.py --smoke
      python scripts/model/train_meta_reptile.py --designs split70 cv5
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_transformer_comorbidity as C  # noqa: E402  (sets thread env vars first)

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from fusion_heads import FusionPredictor  # noqa: E402
from heads import StagePredictor, build_head  # noqa: E402
from multi_attention_transformer import build_model  # noqa: E402
from train_transformer import Scaler, evaluate, load_all  # noqa: E402

S, H = C.S, C.H
ROOT = C.ROOT
OUT = os.path.join(ROOT, "derivatives", "model_training", "meta_reptile")
META = {"iterations": 200, "inner_steps": 5, "inner_lr": 1e-3, "batch": 16,
        "eps_start": 0.5, "eps_end": 0.05}
TASKS = ("mALFF", "mReHo", "DC_z", "FC")
CANDIDATES = {"meta_transformer_svmrfe50": None,
              "meta_transformer_svmrfe50_comorb8": "comorbidity8",
              "meta_transformer_svmrfe50_comorb8_SHUFFLED": "comorbidity8"}

_lines = []


def log(m=""):
    print(m, flush=True)
    _lines.append(str(m))


def task_views(Xs, rois):
    """One [N, K, 8] tensor per biomarker: that biomarker on every token, its flag set."""
    views = []
    for b in range(4):
        v = np.zeros((Xs.shape[0], rois.size, 8), np.float32)
        v[:, :, b] = Xs[:, rois, b]
        v[:, :, 4 + b] = 1.0
        views.append(torch.from_numpy(v))
    return views


def new_stage_model(K, seed):
    torch.manual_seed(seed)
    return StagePredictor(build_model(n_roi=K, n_features=8, **S.ARCH),
                          build_head("multiclass", S.ARCH["d_model"], dropout=S.ARCH["dropout"]))


def reptile(views, F, mask_t, y_used, tr, seed, iterations):
    """Meta-learn theta over biomarker tasks on the training subjects only."""
    K = views[0].shape[1]
    model = new_stage_model(K, seed)
    theta = {k: v.detach().clone() for k, v in model.state_dict().items()}
    w = S.class_weights(y_used[tr])
    rng = np.random.default_rng([seed, 11])
    losses = {t: [] for t in TASKS}
    for it in range(iterations):
        b = int(rng.integers(4))
        X = views[b]
        model.load_state_dict(theta)
        model.train()
        opt = torch.optim.Adam(model.parameters(), lr=META["inner_lr"])
        for _ in range(META["inner_steps"]):
            idx = torch.from_numpy(rng.choice(tr, size=min(META["batch"], tr.size), replace=False))
            opt.zero_grad()
            loss = model.head.loss(model(X[idx], F[idx], mask_t), torch.from_numpy(y_used[idx.numpy()]), weight=w)
            if not torch.isfinite(loss):
                raise SystemExit("non-finite loss in Reptile")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), S.TRAIN["grad_clip"])
            opt.step()
        losses[TASKS[b]].append(float(loss))
        eps = META["eps_start"] + (META["eps_end"] - META["eps_start"]) * it / max(iterations - 1, 1)
        phi = model.state_dict()
        theta = {k: (theta[k] + eps * (phi[k].detach() - theta[k])) if theta[k].is_floating_point()
                 else phi[k].detach().clone() for k in theta}
    return theta, {t: (float(np.mean(v[-10:])) if v else None) for t, v in losses.items()}


def make_finetune_model(theta, K, n_clin, seed):
    torch.manual_seed(seed)
    enc = build_model(n_roi=K, n_features=8, **S.ARCH)
    head = build_head("multiclass", S.ARCH["d_model"], dropout=S.ARCH["dropout"])
    if n_clin:
        m = FusionPredictor(enc, head, n_clin, S.ARCH["d_model"], S.ARCH["dropout"])
        missing, unexpected = m.load_state_dict(theta, strict=False)   # encoder + head from theta
        assert not unexpected and all(k.startswith(("clinical.", "fusion.")) for k in missing)
    else:
        m = StagePredictor(enc, head)
        m.load_state_dict(theta)
    return m


@torch.no_grad()
def probs_of(model, X, F, mask_t, Cl):
    model.eval()
    out = []
    for s in range(0, X.shape[0], S.CHUNK):
        args = (X[s:s + S.CHUNK], F[s:s + S.CHUNK], mask_t) + ((Cl[s:s + S.CHUNK],) if Cl is not None else ())
        out.append(model.head.probabilities(model(*args)))
    return torch.cat(out).numpy()


def finetune(theta, X, F, mask_t, Cl, y_used, itr, iva, seed, max_epochs):
    """As train_transformer_comorbidity.train_model, starting from theta."""
    model = make_finetune_model(theta, X.shape[1], None if Cl is None else Cl.shape[1], seed)
    opt = torch.optim.AdamW(model.parameters(), lr=S.TRAIN["lr"], weight_decay=S.TRAIN["weight_decay"])
    w = S.class_weights(y_used[itr])
    ytr = torch.from_numpy(y_used[itr])
    g = torch.Generator().manual_seed(seed)
    best = {"score": -np.inf, "epoch": -1, "state": None}
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(itr), generator=g)
        for s in range(0, len(perm), S.TRAIN["batch_size"]):
            b = torch.from_numpy(itr[perm[s:s + S.TRAIN["batch_size"]].numpy()])
            opt.zero_grad()
            args = (X[b], F[b], mask_t) + ((Cl[b],) if Cl is not None else ())
            loss = model.head.loss(model(*args), ytr[perm[s:s + S.TRAIN["batch_size"]]], weight=w)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), S.TRAIN["grad_clip"])
            opt.step()
        pv = probs_of(model, X[iva], F[iva], mask_t, None if Cl is None else Cl[iva])
        score = H.evaluate(y_used[iva], pv.argmax(1))["balanced_accuracy"]
        if score > best["score"]:
            best = {"score": score, "epoch": ep, "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        if ep - best["epoch"] >= S.TRAIN["patience"]:
            break
    model.load_state_dict(best["state"])
    model.eval()
    return model, best["epoch"]


def run_split(cand, data, clin_sets, tr, te, seed, fold, fp, max_epochs, iterations, smoke=False):
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
    Xs = Scaler().fit(X4, tr).transform(X4).astype(np.float32)          # TRAINING SUBJECTS ONLY
    _, support, chmask = S.fit_selection(flat, y_used, tr, seed + fold)  # same 50 as the transformer
    tok, rois = S.make_tokens(Xs, chmask)
    X = torch.from_numpy(np.ascontiguousarray(tok))
    F = torch.from_numpy(np.ascontiguousarray(fc[:, rois][:, :, rois], dtype=np.float32))
    mask_t = torch.from_numpy(np.ascontiguousarray(mask[np.ix_(rois, rois)]))
    key = CANDIDATES[cand]
    Cl = torch.from_numpy(C.ClinPrep().fit(clin_sets[key], tr).transform(clin_sets[key])) if key else None

    bag_seed = seed * 100 + fold                                         # same seeds as the scratch runs
    t_meta = time.time()
    theta, task_loss = reptile(task_views(Xs, rois), F, mask_t, y_used, tr, bag_seed, iterations)
    rec["meta_seconds"] = round(time.time() - t_meta, 1)
    rec["meta_task_loss_last10"] = task_loss
    fo = H.stratified_folds(y_used, tr, S.N_INNER, np.random.default_rng([bag_seed, 7]))
    models, epochs = [], []
    for j in range(S.N_INNER):
        itr = np.array([i for i in tr if fo[int(i)] != j])
        iva = np.array([i for i in tr if fo[int(i)] == j])
        m, ep = finetune(theta, X, F, mask_t, Cl, y_used, itr, iva, bag_seed * 10 + j, max_epochs)
        models.append(m)
        epochs.append(ep)
    p_te = np.mean([probs_of(m, X[te], F[te], mask_t, None if Cl is None else Cl[te]) for m in models], axis=0)
    p_tr = np.mean([probs_of(m, X[tr], F[tr], mask_t, None if Cl is None else Cl[tr]) for m in models], axis=0)
    if not np.isfinite(p_te).all():
        raise SystemExit("non-finite probabilities")
    pred = p_te.argmax(1)
    m = evaluate(np.asarray(p_te), y[te], pred=pred)
    rec.update({k: m[k] for k in S.METRIC_KEYS})
    rec.update({"train_balanced_accuracy": H.evaluate(y_used[tr], p_tr.argmax(1))["balanced_accuracy"],
                "support": support.tolist(), "rois": rois.tolist(), "inner_best_epochs": epochs,
                "pred": [int(v) for v in pred], "probs": S.r6(p_te), "seconds": round(time.time() - t0, 1)})
    log(f"    seed {seed} fold {fold}  K={rois.size}  test bal.acc {m['balanced_accuracy']:.3f}  "
        f"acc {m['accuracy']:.3f}  train {rec['train_balanced_accuracy']:.3f}  epochs {epochs}  "
        f"meta {rec['meta_seconds']:.0f}s  total {rec['seconds']:.0f}s"
        + (f"  task loss {json.dumps({k: round(v, 3) for k, v in task_loss.items() if v is not None})}" if smoke else ""))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", nargs="+", choices=["split70", "cv5"], default=["split70", "cv5"])
    ap.add_argument("--candidates", nargs="+", choices=list(CANDIDATES), default=list(CANDIDATES))
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="one CV fold, 10 Reptile iterations, 3 epochs, nothing saved")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    os.makedirs(OUT, exist_ok=True)

    data = S.load_cohort()
    y, subjects = data[3], data[4]
    clin_sets = {"comorbidity8": C.load_clinical(y, subjects, C.FEATURE_SETS["comorbidity8"])[0]}
    _, _, y2, _, _, pool, test, _, smeta = load_all()
    assert np.array_equal(y2, y) and smeta["split_fingerprint"] == H.EXPECTED_SPLIT_FINGERPRINT
    plan = S.cv_plan(y, subjects, 5)

    if args.smoke:
        fp, folds = plan[H.SEEDS[0]]
        tr, te = folds[0]
        for cand in args.candidates:
            run_split(cand, data, clin_sets, tr, te, H.SEEDS[0], 0, fp, 3, 10, smoke=True)
        log("SMOKE OK")
        return

    for design in args.designs:
        dname = "split70" if design == "split70" else "cv116"
        for cand in args.candidates:
            if design == "split70" and cand.endswith("_SHUFFLED"):
                continue
            stem = os.path.join(OUT, f"{dname}_{cand}")
            if os.path.exists(stem + ".json"):
                print(f"COLLISION: {dname}_{cand}.json exists; skipped, not overwritten")
                continue
            _lines.clear()
            log("=" * 100)
            log(f"REPTILE META-LEARNING (biomarker tasks) + TRANSFORMER FINE-TUNE -- design {dname}, candidate {cand}")
            log("=" * 100)
            log(f"meta settings {META}; tasks {TASKS}; architecture {S.ARCH}; fine-tune {S.TRAIN}")
            t0 = time.time()
            records = []
            if design == "split70":
                records.append(run_split(cand, data, clin_sets, pool, test, C.SPLIT70_SEED, 0,
                                         smeta["split_fingerprint"], S.TRAIN["max_epochs"], META["iterations"]))
                extra = {"design": dname, "split_fingerprint": smeta["split_fingerprint"]}
            else:
                log("fold fingerprints " + json.dumps({s: plan[s][0] for s in H.SEEDS}))
                for seed in H.SEEDS:
                    fp, folds = plan[seed]
                    for f, (tr, te) in enumerate(folds):
                        records.append(run_split(cand, data, clin_sets, tr, te, seed, f, fp,
                                                 S.TRAIN["max_epochs"], META["iterations"]))
                s = H.summarise(records)
                log(f"  {cand}: seed-level bal.acc {s['balanced_accuracy'][0]:.3f} +/- "
                    f"{s['balanced_accuracy'][1]:.3f}  acc {s['accuracy'][0]:.3f}  "
                    f"train {np.mean([r['train_balanced_accuracy'] for r in records]):.3f}")
                extra = {"design": dname, "n_folds": 5, "fold_fingerprints": {str(s_): plan[s_][0] for s_ in H.SEEDS}}
            extra.update({"candidate": cand, "meta": META, "tasks": TASKS, "arch": S.ARCH, "train": S.TRAIN,
                          "seconds": round(time.time() - t0, 1)})
            log(f"  elapsed {time.time() - t0:.0f}s")
            json.dump({"extra": extra, "records": records}, open(stem + ".json", "w"))
            open(stem + ".log", "w", encoding="utf-8").write("\n".join(_lines) + "\n")
            print(f"  wrote {os.path.relpath(stem, ROOT)}.json")


if __name__ == "__main__":
    main()
