"""
Ordinal transformer v1 (pre-registered in derivatives/experiments/ordinal_transformer_v1/PLAN.md).

transformer_nested_v2's procedure with one change: the output head and its loss (CumulativeLinkHead, exact
cumulative-link NLL) instead of the 6-way softmax with cross-entropy. Everything else is v2's code or an exact copy
of it: N.inner_inputs (scaler + SVM-RFE per inner-training set), the same folds and seeds, N.select_epochs. Because
an ordinal head does not output softmax logits, scoring is done from each head's own probabilities and NLL (for the
reference head this is identical to v2, which the gate checks).

Outer-test labels are hidden (-1) from all fitting and selection code and read only in score_split(), after every
choice is frozen.

Run:
  python scripts/experiments/ordinal_transformer_v1/run_ordinal.py --smoke     # one split, 3 epochs, into smoke/
  python scripts/experiments/ordinal_transformer_v1/run_ordinal.py             # gate, then all 25 splits
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
import run_nested as N  # noqa: E402  (sets thread env vars first)

import numpy as np  # noqa: E402
import psutil  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as Fn  # noqa: E402

from heads import StagePredictor, build_head  # noqa: E402
from multi_attention_transformer import build_model  # noqa: E402
from ordinal_head import CumulativeLinkHead, median_decode  # noqa: E402

S, H = N.S, N.H
EXP = os.path.join(N.ROOT, "derivatives", "experiments", "ordinal_transformer_v1")
LABEL_ORDER = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
V2_GRID = os.path.join(N.ROOT, "derivatives", "experiments", "transformer_nested_v2", "grid")


def script_hash():
    return hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]


def new_model(n_roi, n_feat, seed, head_kind):
    torch.manual_seed(seed)                        # as N.train_inner: seed, then encoder, then head
    enc = build_model(n_roi=n_roi, n_features=n_feat, d_model=N.CFG["d_model"], num_heads=N.CFG["num_heads"],
                      num_layers=N.CFG["num_layers"], dropout=N.CFG["dropout"])
    head = (build_head("multiclass", N.CFG["d_model"], dropout=N.CFG["dropout"]) if head_kind == "multiclass"
            else CumulativeLinkHead(N.CFG["d_model"], dropout=N.CFG["dropout"]))
    return StagePredictor(enc, head)


def head_scores(model, out, yy):
    """Probabilities, NLL and metrics from the head's own output (softmax/CE for the reference head)."""
    if model.head.kind == "multiclass":
        probs = torch.softmax(out, 1)
        loss = Fn.cross_entropy(out, torch.from_numpy(yy))
    else:
        probs = model.head.probabilities(out)
        loss = -model.head.log_probs(out).gather(1, torch.from_numpy(yy)[:, None]).mean()
    p = probs.numpy()
    pred = p.argmax(1)
    m = H.evaluate(yy, pred)
    return p, {"loss": float(loss), "acc": m["accuracy"], "bal": m["balanced_accuracy"], "f1": m["macro_f1"],
               "mae": float(np.abs(pred - yy).mean())}


def train_inner_h(X, F, mask_t, y_sel, itr, iva, te, seed, head_kind, max_epochs=N.MAX_EPOCHS):
    """N.train_inner with a choice of head; stores per-epoch probabilities (not logits) for every set."""
    assert (y_sel[itr] >= 0).all() and (y_sel[iva] >= 0).all() and (y_sel[te] == N.HIDDEN).all()
    model = new_model(X.shape[1], X.shape[2], seed, head_kind)
    opt = torch.optim.AdamW(model.parameters(), lr=N.CFG["lr"], weight_decay=N.CFG["weight_decay"])
    w = S.class_weights(y_sel[itr])
    Xtr, Ftr, ytr = X[itr], F[itr], torch.from_numpy(y_sel[itr])
    g = torch.Generator().manual_seed(seed)
    curves = {s: {k: [] for k in ("loss", "acc", "bal", "f1", "mae")} for s in ("train", "val")}
    val_probs, outer_probs = [], []
    best, best_ep, stop_ep = -np.inf, -1, None
    for ep in range(max_epochs):
        model.train()
        perm = torch.randperm(len(itr), generator=g)
        for s in range(0, len(perm), N.BS):
            b = perm[s:s + N.BS]
            opt.zero_grad()
            loss = model.head.loss(model(Xtr[b], Ftr[b], mask_t), ytr[b], weight=w)
            if not torch.isfinite(loss):
                raise SystemExit("non-finite loss")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), N.CLIP)
            opt.step()
        pv, mv = head_scores(model, N.logits_on(model, X, F, mask_t, iva), y_sel[iva])
        _, mt = head_scores(model, N.logits_on(model, X, F, mask_t, itr), y_sel[itr])
        with torch.no_grad():
            o = N.logits_on(model, X, F, mask_t, te)
            outer_probs.append((torch.softmax(o, 1) if head_kind == "multiclass" else model.head.probabilities(o)).numpy())
        for k in mv:
            curves["val"][k].append(mv[k])
            curves["train"][k].append(mt[k])
        val_probs.append(pv)
        if stop_ep is None:
            if mv["bal"] > best:
                best, best_ep = mv["bal"], ep
            if ep - best_ep >= N.PATIENCE:
                stop_ep = ep
    th = model.head.thresholds().detach().numpy().tolist() if head_kind != "multiclass" else None
    return {"curves": curves, "val_probs": np.array(val_probs), "outer_probs": np.array(outer_probs),
            "best_epoch": best_ep, "stop_epoch": stop_ep if stop_ep is not None else max_epochs - 1,
            "n_parameters": sum(p.numel() for p in model.parameters()), "final_thresholds": th}


def fit_split(d, y_sel, seed, fold, tr, te, head_kind, max_epochs=N.MAX_EPOCHS):
    """Everything that learns or selects. Never sees an outer-test label."""
    assert (y_sel[te] == N.HIDDEN).all() and (y_sel[tr] >= 0).all()
    bag_seed = seed * 100 + fold
    fo = H.stratified_folds(y_sel, tr, S.N_INNER, np.random.default_rng([bag_seed, 7]))
    inner = []
    for j in range(S.N_INNER):
        itr = np.array([i for i in tr if fo[int(i)] != j])
        iva = np.array([i for i in tr if fo[int(i)] == j])
        X, F, mask_t, support, rois, _ = N.inner_inputs(d, y_sel, itr, iva, seed + fold)
        m = train_inner_h(X, F, mask_t, y_sel, itr, iva, te, bag_seed * 10 + j, head_kind, max_epochs)
        m.update({"itr": itr, "iva": iva, "support": support, "rois": rois})
        inner.append(m)
    epochs, inner_bal, vloss, pooled = N.select_epochs(inner, y_sel)
    probs = {r: np.mean([m["outer_probs"][e] for m, e in zip(inner, ep)], 0) for r, ep in epochs.items()}
    return inner, epochs, inner_bal, vloss, pooled, probs


def score_split(y, te, inner, probs):
    """Reads outer-test labels; called only after fit_split has returned."""
    yt = y[te]
    out = {}
    for r, p in probs.items():
        for dec, pred in (("argmax", p.argmax(1)), ("median", median_decode(torch.from_numpy(p)).numpy())):
            m = H.evaluate(yt, pred)
            out[f"{r}_{dec}"] = {"accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"],
                                 "macro_f1": m["macro_f1"], "mae": float(np.abs(pred - yt).mean()),
                                 "per_class_recall": m["per_class_recall"]}
    outer_curves = []                               # diagnostic only
    for m in inner:
        c = {k: [] for k in ("loss", "bal", "mae")}
        for e in range(m["outer_probs"].shape[0]):
            pe = m["outer_probs"][e]
            pred = pe.argmax(1)
            c["loss"].append(float(-np.log(np.clip(pe[np.arange(yt.size), yt], 1e-12, None)).mean()))
            c["bal"].append(H.evaluate(yt, pred)["balanced_accuracy"])
            c["mae"].append(float(np.abs(pred - yt).mean()))
        outer_curves.append(c)
    return out, outer_curves


def run_split(d, y, seed, fold, fp, tr, te, head_kind="ordinal", max_epochs=N.MAX_EPOCHS):
    inner, epochs, inner_bal, vloss, pooled, probs = fit_split(d, N.hide(y, te), seed, fold, tr, te, head_kind, max_epochs)
    frozen = json.dumps(epochs, sort_keys=True)
    metrics, outer_curves = score_split(y, te, inner, probs)
    assert json.dumps(epochs, sort_keys=True) == frozen
    return {"seed": seed, "fold": fold, "fold_fingerprint": fp, "head": head_kind, "label_order": LABEL_ORDER,
            "settings": N.CFG, "script_sha256": script_hash(), "test_idx": te.tolist(), "true": y[te].tolist(),
            "epochs": epochs, "inner_oof_bal": inner_bal, "mean_inner_val_loss": S.r6(vloss),
            "pooled_inner_val_bal": S.r6(pooled),
            "pred": {r: p.argmax(1).tolist() for r, p in probs.items()},
            "pred_median": {r: median_decode(torch.from_numpy(p)).tolist() for r, p in probs.items()},
            "probs": {r: S.r6(p) for r, p in probs.items()}, "heldout": metrics,
            "inner_models": [{"n_train": int(m["itr"].size), "n_val": int(m["iva"].size), "support": m["support"].tolist(),
                              "K": int(m["rois"].size), "n_parameters": int(m["n_parameters"]),
                              "best_epoch": int(m["best_epoch"]), "stop_epoch": int(m["stop_epoch"]),
                              "final_thresholds": m["final_thresholds"], "curves": m["curves"], "outer_curves": oc}
                             for m, oc in zip(inner, outer_curves)]}


def gate(d):
    """The reference head through this code must reproduce v2's curves exactly (seed 20261015, fold 0, model 0)."""
    y = d[3]
    tr, te = S.cv_plan(y, d[4], 5)[20261015][1][0]
    y_sel = N.hide(y, te)
    fo = H.stratified_folds(y_sel, tr, S.N_INNER, np.random.default_rng([20261015 * 100, 7]))
    itr = np.array([i for i in tr if fo[int(i)] != 0])
    iva = np.array([i for i in tr if fo[int(i)] == 0])
    X, F, mask_t, _, _, _ = N.inner_inputs(d, y_sel, itr, iva, 20261015)
    m = train_inner_h(X, F, mask_t, y_sel, itr, iva, te, 20261015 * 100 * 10, "multiclass")
    ref = json.load(open(os.path.join(V2_GRID, "cv5_seed20261015_fold0.json")))["inner_models"][0]
    ok = (m["curves"]["val"]["bal"] == ref["curves"]["val"]["bal"]
          and np.allclose(m["curves"]["train"]["loss"], ref["curves"]["train"]["loss"])
          and np.allclose(m["curves"]["val"]["loss"], ref["curves"]["val"]["loss"]) and m["best_epoch"] == ref["best_epoch"])
    print(f"gate: reference head reproduces transformer v2 exactly: {ok}", flush=True)
    if not ok:
        raise SystemExit("STOP: training code does not reproduce transformer v2")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--threads", type=int, default=2)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if not os.path.exists(os.path.join(EXP, "PLAN.md")):
        raise SystemExit("PLAN.md (pre-registration) missing")
    d = S.load_cohort()
    y = d[3]
    assert list(H.STAGES) == LABEL_ORDER
    plan = S.cv_plan(y, d[4], 5)
    if args.smoke:
        t0 = time.time()
        tr, te = plan[20261015][1][0]
        rec = run_split(d, y, 20261015, 0, plan[20261015][0], tr, te, "ordinal", max_epochs=3)
        p = np.array(rec["probs"]["FIXED"])
        im = rec["inner_models"][0]
        info = {"probs_shape": list(p.shape), "probs_row_sums_min_max": [float(p.sum(1).min()), float(p.sum(1).max())],
                "probs_min": float(p.min()), "n_parameters": im["n_parameters"], "K_tokens": im["K"],
                "train_loss": im["curves"]["train"]["loss"], "val_loss": im["curves"]["val"]["loss"],
                "thresholds_increasing": bool(np.all(np.diff(im["final_thresholds"]) > 0)),
                "final_thresholds": im["final_thresholds"], "rss_gb": psutil.Process().memory_info().rss / 1e9,
                "seconds": round(time.time() - t0, 1)}
        os.makedirs(os.path.join(EXP, "smoke"), exist_ok=True)
        json.dump(info, open(os.path.join(EXP, "smoke", "smoke_check.json"), "w"), indent=2)
        print(json.dumps(info, indent=1))
        return
    lock = os.path.join(EXP, "run.lock")
    if os.path.exists(lock) and psutil.pid_exists(int(open(lock).read().strip() or 0)):
        raise SystemExit("another run is active; not starting a duplicate")
    mpath = os.path.join(EXP, "manifest.json")
    if os.path.exists(mpath) and json.load(open(mpath))["script_sha256"] != script_hash():
        raise SystemExit("run_ordinal.py changed since the run started; refusing to mix versions")
    json.dump({"experiment": "ordinal_transformer_v1", "script_sha256": script_hash(), "label_order": LABEL_ORDER,
               "settings": N.CFG, "max_epochs": N.MAX_EPOCHS, "primary_rule": "FIXED", "primary_decision": "argmax",
               "reference": "derivatives/experiments/transformer_nested_v2/grid", "torch": torch.__version__,
               "created": time.strftime("%Y-%m-%d %H:%M:%S")}, open(mpath, "w"), indent=2)
    open(lock, "w").write(str(os.getpid()))
    try:
        gate(d)
        for seed in H.SEEDS:
            fp, folds = plan[seed]
            for f, (tr, te) in enumerate(folds):
                ref = json.load(open(os.path.join(V2_GRID, f"cv5_seed{seed}_fold{f}.json")))
                assert te.tolist() == ref["test_idx"], "outer split differs from the reference"
                path = os.path.join(EXP, "grid", f"cv5_seed{seed}_fold{f}.json")
                if os.path.exists(path):
                    continue
                t0 = time.time()
                rec = run_split(d, y, seed, f, fp, tr, te, "ordinal")
                rec["seconds"] = round(time.time() - t0, 1)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                json.dump(rec, open(path + ".tmp", "w"))
                os.replace(path + ".tmp", path)
                print(f"seed {seed} fold {f}: E*(loss) {rec['epochs']['FIXED'][0] + 1}  {rec['seconds']:.0f}s", flush=True)
        print("done", flush=True)
    finally:
        os.remove(lock)


if __name__ == "__main__":
    main()
