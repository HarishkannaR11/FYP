"""
Corrected-transformer variants for experiment v3 (pre-registered in PLAN.md).

Every variant uses the transformer_nested_v2 procedure: the same outer folds, the same 4 inner folds and torch
seeds, the scaler and SVM-RFE fitted on each inner model's inner-training subjects only, outer-test labels hidden
from all fitting code, 60 logged epochs, and checkpoint rules ES / FIXED (primary) / FIXED_BAL / LAST.
Exactly one thing changes per variant:

  small       smaller and more regularised: d_model 8 (was 16), dropout 0.5 (0.3), weight decay 0.1 (0.01)
  unweighted  plain cross-entropy instead of inverse-frequency class weights (the baseline is weighted)
  comorb8     SEPARATE EXPERIMENT, not imaging-only: the 8 comorbidity variables (age, sex, APOE4 count, GDS
              total, cardiovascular, endocrine-metabolic, psychiatric, smoking) fused by the existing gated
              FusionPredictor; their median fill and scaling are fitted on each inner-training set only
  chosen      the feature-selection setting (pool and k) chosen in each outer split by TabPFN's nested inner-CV
              rule (run_tabpfn_configs.nested_choice); uses that split's training subjects only

Gate: before any variant runs, the baseline settings are re-run for one inner model and must reproduce the v2
curves exactly; otherwise the script stops.

Outputs: derivatives/experiments/biomarker_ablation_v3/transformer/<variant>/cv5_seed{s}_fold{f}.json (written once).
Run:  python scripts/experiments/biomarker_ablation_v3/run_transformer_variants.py --variants small unweighted comorb8 chosen
"""
import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "transformer_nested_v2"))
import run_tabpfn_configs as T  # noqa: E402  (sets thread env vars first)
import run_nested as N  # noqa: E402
import train_transformer_comorbidity as CM  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

from fusion_heads import FusionPredictor  # noqa: E402
from heads import StagePredictor, build_head  # noqa: E402
from multi_attention_transformer import build_model  # noqa: E402

S, H = N.S, N.H
OUT = os.path.join(T.EXP, "transformer")
V2_GRID = os.path.join(T.ROOT, "derivatives", "experiments", "transformer_nested_v2", "grid")
VARIANTS = {"small": {"cfg": {"d_model": 8, "dropout": 0.5, "weight_decay": 0.1}},
            "unweighted": {"weighted": False},
            "comorb8": {"clinical": "comorbidity8"},
            "chosen": {"chosen": True}}


def script_hash():
    return hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]


@torch.no_grad()
def logits_on(model, X, F, mask_t, C, idx):
    model.eval()
    out = []
    for s in range(0, len(idx), S.CHUNK):
        b = idx[s:s + S.CHUNK]
        out.append(model(X[b], F[b], mask_t) if C is None else model(X[b], F[b], mask_t, C[b]))
    return torch.cat(out)


def train_inner_v(X, F, mask_t, C, y_sel, itr, iva, te, seed, cfg, weighted=True, max_epochs=N.MAX_EPOCHS):
    """N.train_inner with the variant's settings, class weighting switch and optional clinical input."""
    assert (y_sel[itr] >= 0).all() and (y_sel[iva] >= 0).all() and (y_sel[te] == N.HIDDEN).all()
    torch.manual_seed(seed)
    enc = build_model(n_roi=X.shape[1], n_features=X.shape[2], d_model=cfg["d_model"], num_heads=cfg["num_heads"],
                      num_layers=cfg["num_layers"], dropout=cfg["dropout"])
    head = build_head("multiclass", cfg["d_model"], dropout=cfg["dropout"])
    model = StagePredictor(enc, head) if C is None else FusionPredictor(enc, head, C.shape[1], cfg["d_model"], cfg["dropout"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    w = S.class_weights(y_sel[itr]) if weighted else None
    ytr = torch.from_numpy(y_sel[itr])
    g = torch.Generator().manual_seed(seed)
    curves = {s: {k: [] for k in ("loss", "acc", "bal", "f1")} for s in ("train", "val")}
    val_probs, outer_logits = [], []
    best, best_ep, stop_ep = -np.inf, -1, None
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(itr), generator=g)
        for s in range(0, len(perm), N.BS):
            b = perm[s:s + N.BS]
            rows = itr[b.numpy()]
            opt.zero_grad()
            out = model(X[rows], F[rows], mask_t) if C is None else model(X[rows], F[rows], mask_t, C[rows])
            loss = model.head.loss(out, ytr[b], weight=w)
            if not torch.isfinite(loss):
                raise SystemExit("non-finite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), N.CLIP)
            opt.step()
        pv, mv = N.scores(logits_on(model, X, F, mask_t, C, iva), y_sel[iva])
        _, mt = N.scores(logits_on(model, X, F, mask_t, C, itr), y_sel[itr])
        outer_logits.append(logits_on(model, X, F, mask_t, C, te))
        for k in mv:
            curves["val"][k].append(mv[k])
            curves["train"][k].append(mt[k])
        val_probs.append(pv)
        if stop_ep is None:
            if mv["bal"] > best:
                best, best_ep = mv["bal"], ep
            if ep - best_ep >= N.PATIENCE:
                stop_ep = ep
    return {"curves": curves, "val_probs": np.array(val_probs), "outer_logits": torch.stack(outer_logits),
            "best_epoch": best_ep, "stop_epoch": stop_ep if stop_ep is not None else max_epochs - 1,
            "n_parameters": sum(p.numel() for p in model.parameters())}


def inner_inputs_pool(d, y_sel, itr, iva, seed_tag, cols, k):
    """N.inner_inputs with SVM-RFE restricted to a pool and k (identical to it for all four biomarkers, k = 50)."""
    X4, flat, fc, _, _, mask = d
    y_fit = N.hide(y_sel, iva)
    sc = N.Scaler().fit(X4, itr)
    Xs = sc.transform(X4).astype(np.float32)
    support = T.select(flat, y_fit, itr, cols, seed_tag, k)
    chmask = np.zeros((S.N_ROI, 4), dtype=bool)
    for i in support:
        chmask[S.feature_slot(int(i))] = True
    tok, rois = S.make_tokens(Xs, chmask)
    X = torch.from_numpy(np.ascontiguousarray(tok))
    F = torch.from_numpy(np.ascontiguousarray(d[2][:, rois][:, :, rois], dtype=np.float32))
    mask_t = torch.from_numpy(np.ascontiguousarray(mask[np.ix_(rois, rois)]))
    return X, F, mask_t, support, rois


def fit_split_v(d, clin, y_sel, seed, fold, tr, te, var, pool="all4", k=50):
    """Everything that learns or selects for one variant and outer split. Never sees an outer-test label."""
    assert (y_sel[te] == N.HIDDEN).all() and (y_sel[tr] >= 0).all()
    spec = VARIANTS[var]
    cfg = {**N.CFG, **spec.get("cfg", {})}
    bag_seed = seed * 100 + fold
    fo = H.stratified_folds(y_sel, tr, S.N_INNER, np.random.default_rng([bag_seed, 7]))
    cols = T.pool_columns(pool)
    inner = []
    for j in range(S.N_INNER):
        itr = np.array([i for i in tr if fo[int(i)] != j])
        iva = np.array([i for i in tr if fo[int(i)] == j])
        X, F, mask_t, support, rois = inner_inputs_pool(d, y_sel, itr, iva, seed + fold, cols, k)
        C = torch.from_numpy(CM.ClinPrep().fit(clin, itr).transform(clin)) if spec.get("clinical") else None
        m = train_inner_v(X, F, mask_t, C, y_sel, itr, iva, te, bag_seed * 10 + j, cfg, spec.get("weighted", True))
        m.update({"itr": itr, "iva": iva, "support": support, "rois": rois})
        inner.append(m)
    epochs, inner_bal, vloss, pooled = N.select_epochs(inner, y_sel)
    probs = {r: N.bag_probs(inner, e) for r, e in epochs.items()}
    return inner, epochs, inner_bal, vloss, pooled, probs, cfg


def chosen_settings():
    """{(seed, fold): (config, {config: TabPFN inner score})} from the Stage 1 TabPFN records."""
    out = {}
    for s in H.SEEDS:
        for f in range(5):
            sc = {c: json.load(open(os.path.join(T.EXP, "grid", c, f"cv5_seed{s}_fold{f}.json")))["inner"]["score"]
                  for c in T.NESTED_ORDER}
            out[(s, f)] = (T.nested_choice(sc), sc)
    return out


def gate(d):
    """Baseline settings must reproduce the v2 curves of seed 20261015 fold 0, inner model 0, exactly."""
    y = d[3]
    plan = S.cv_plan(y, d[4], 5)
    tr, te = plan[20261015][1][0]
    y_sel = N.hide(y, te)
    fo = H.stratified_folds(y_sel, tr, S.N_INNER, np.random.default_rng([20261015 * 100, 7]))
    itr = np.array([i for i in tr if fo[int(i)] != 0])
    iva = np.array([i for i in tr if fo[int(i)] == 0])
    X, F, mask_t, _, _ = inner_inputs_pool(d, y_sel, itr, iva, 20261015, T.pool_columns("all4"), 50)
    m = train_inner_v(X, F, mask_t, None, y_sel, itr, iva, te, 20261015 * 100 * 10, N.CFG, True)
    ref = json.load(open(os.path.join(V2_GRID, "cv5_seed20261015_fold0.json")))["inner_models"][0]["curves"]
    ok = m["curves"]["val"]["bal"] == ref["val"]["bal"] and np.allclose(m["curves"]["train"]["loss"], ref["train"]["loss"])
    print(f"gate: baseline settings reproduce transformer v2 exactly: {ok}", flush=True)
    if not ok:
        raise SystemExit("STOP: the variant code does not reproduce transformer v2")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", required=True, choices=list(VARIANTS))
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    lock = T.acquire_lock("transformer")
    try:
        d = S.load_cohort()
        y = d[3]
        clin = CM.load_clinical(y, d[4], CM.FEATURE_SETS["comorbidity8"])[0]
        plan = S.cv_plan(y, d[4], 5)
        gate(d)
        choice = chosen_settings() if "chosen" in args.variants else {}
        for var in args.variants:
            for seed in H.SEEDS:
                fp, folds = plan[seed]
                for f, (tr, te) in enumerate(folds):
                    path = os.path.join(OUT, var, f"cv5_seed{seed}_fold{f}.json")
                    if os.path.exists(path):
                        continue
                    t0 = time.time()
                    # only the "chosen" variant changes the feature setting; every other variant keeps all four, k = 50
                    cname, scores = choice[(seed, f)] if var == "chosen" else ("A_all4", None)
                    pool, k = T.CONFIGS[cname]["pool"], T.CONFIGS[cname]["k"]
                    inner, epochs, inner_bal, vloss, pooled, probs, cfg = fit_split_v(
                        d, clin, N.hide(y, te), seed, f, tr, te, var, pool, k)
                    frozen = json.dumps(epochs, sort_keys=True)
                    metrics, outer_curves = N.score_split(y, te, inner, probs)
                    assert json.dumps(epochs, sort_keys=True) == frozen
                    rec = {"variant": var, "seed": seed, "fold": f, "fold_fingerprint": fp, "settings": cfg,
                           "weighted": VARIANTS[var].get("weighted", True), "clinical": VARIANTS[var].get("clinical"),
                           "config": cname, "pool": pool, "k": k, "tabpfn_inner_scores": scores,
                           "script_sha256": script_hash(), "test_idx": te.tolist(), "true": y[te].tolist(),
                           "epochs": epochs, "inner_oof_bal": inner_bal, "mean_inner_val_loss": S.r6(vloss),
                           "pooled_inner_val_bal": S.r6(pooled),
                           "pred": {r: p.argmax(1).tolist() for r, p in probs.items()},
                           "probs": {r: S.r6(p) for r, p in probs.items()}, "heldout": metrics,
                           "inner_models": [{"n_train": int(m["itr"].size), "n_val": int(m["iva"].size),
                                             "support": m["support"].tolist(), "K": int(m["rois"].size),
                                             "n_parameters": int(m["n_parameters"]), "best_epoch": int(m["best_epoch"]),
                                             "stop_epoch": int(m["stop_epoch"]), "curves": m["curves"],
                                             "outer_curves": oc} for m, oc in zip(inner, outer_curves)],
                           "seconds": round(time.time() - t0, 1)}
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    json.dump(rec, open(path + ".tmp", "w"))
                    os.replace(path + ".tmp", path)
                    print(f"{var} seed {seed} fold {f}: {cname}, E*(loss) {epochs['FIXED'][0] + 1}  {rec['seconds']:.0f}s",
                          flush=True)
    finally:
        os.remove(lock)


if __name__ == "__main__":
    main()
