"""
Extra model families on the leakage-safe v1 outer splits (added 2026-10-21).

  rf       Random forest, 200 trees, sqrt features, balanced class weights;
           min_samples_leaf in {1, 5} chosen by inner 3-fold CV
  xgb      XGBoost (hist), 150 trees, learning rate 0.05, subsample 0.8,
           colsample 0.5, balanced sample weights; max_depth in {2, 3} by inner CV
  ordmlp   ordinal MLP: one hidden layer (32, GELU, dropout 0.3) + the project's
           cumulative-link OrdinalHead (heads.py, ordered thresholds, decoded by
           counting thresholds passed); AdamW 1e-3 / 1e-2, batch 8, <= 150 epochs.
           Fixed a priori (no tuning). Each of the 3 inner models is early-stopped
           on its own inner-validation subjects; the outer prediction averages
           their cumulative probabilities.

Views: SVM-RFE k in {25, 50, 100, 150} x inputs {img, imgcom} (no
harmonisation), using the exact selections and inner folds saved by the main
job; plus each biomarker alone (246 features, imaging only). Imputation and
scaling are fitted on training rows only. CatBoost is not installed.

Run:  python scripts/experiments/leakage_safe_v1/run_extra_models.py [--limit 1]
"""
import argparse
import json
import os
import sys
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as K  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from xgboost import XGBClassifier  # noqa: E402

from heads import OrdinalHead  # noqa: E402

KS = [25, 50, 100, 150]
HP = {"rf": ["leaf=1", "leaf=5"], "xgb": ["depth=2", "depth=3"], "ordmlp": ["-"]}


def balanced_weights(y):
    c = np.bincount(y, minlength=K.NS).astype(float)
    return (len(y) / (K.NS * np.maximum(c, 1)))[y]


def fit_tree(family, hp, Xtr, ytr, Xte, rs):
    v = int(hp.split("=")[1])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if family == "rf":
            m = Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("rf", RandomForestClassifier(n_estimators=200, min_samples_leaf=v, max_features="sqrt",
                                                        class_weight="balanced_subsample", n_jobs=4, random_state=rs))])
            m.fit(Xtr, ytr)
        else:
            m = XGBClassifier(n_estimators=150, max_depth=v, learning_rate=0.05, subsample=0.8, colsample_bytree=0.5,
                              tree_method="hist", n_jobs=4, random_state=rs, objective="multi:softprob",
                              num_class=K.NS, eval_metric="mlogloss")
            m.fit(Xtr, ytr, sample_weight=balanced_weights(ytr))
        p = m.predict_proba(Xte)
        classes = m.classes_ if family == "rf" else np.arange(K.NS)
        P = np.zeros((len(Xte), K.NS))
        P[:, np.asarray(classes, int)] = p
        ptr = m.predict_proba(Xtr)
        Ptr = np.zeros((len(Xtr), K.NS))
        Ptr[:, np.asarray(classes, int)] = ptr
    return P.argmax(1), P, K.bal_acc(ytr, Ptr.argmax(1))


class OrdMLP(nn.Module):
    def __init__(self, n_in, hidden=32, dropout=0.3):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_in, hidden), nn.GELU(), nn.Dropout(dropout))
        self.head = OrdinalHead(hidden, dropout=dropout, mean_reduce=True)

    def forward(self, x):
        return self.head(self.body(x))


def fit_ordmlp(Xd, y, inner, te, seed):
    """3 inner models, each early-stopped on its own inner-validation subjects; outer = mean cumulative prob."""
    cums, train_cums, tr_all = [], [], np.sort(np.concatenate([inner[0][0], inner[0][1]]))
    for j, (itr, iva) in enumerate(inner):
        prep = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]).fit(Xd[itr])
        Xt = torch.tensor(prep.transform(Xd), dtype=torch.float32)
        torch.manual_seed(seed * 10 + j)
        m = OrdMLP(Xd.shape[1])
        m.head.init_from_prior(np.bincount(y[itr], minlength=K.NS))
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
        w = torch.tensor(len(itr) / (K.NS * np.maximum(np.bincount(y[itr], minlength=K.NS), 1)), dtype=torch.float32)
        g = torch.Generator().manual_seed(seed * 10 + j)
        yt = torch.tensor(y)
        best, best_ep, state = -1.0, -1, None
        for ep in range(150):
            m.train()
            perm = torch.randperm(len(itr), generator=g)
            for s in range(0, len(itr), 8):
                b = torch.tensor(itr[perm[s:s + 8].numpy()])
                opt.zero_grad()
                m.head.loss(m(Xt[b]), yt[b], weight=w).backward()
                opt.step()
            m.eval()
            with torch.no_grad():
                sc = K.bal_acc(y[iva], m.head.decode(m(Xt[iva])).numpy())
            if sc > best:
                best, best_ep, state = sc, ep, {k: v.clone() for k, v in m.state_dict().items()}
            if ep - best_ep >= 20:
                break
        m.load_state_dict(state)
        m.eval()
        with torch.no_grad():
            cums.append(torch.sigmoid(m(Xt[te])).numpy())
            train_cums.append(torch.sigmoid(m(Xt[tr_all])).numpy())

    def decode(c):
        return (c < 0.5).sum(1)

    def probs(c):
        cc = np.column_stack([np.zeros(len(c)), c, np.ones(len(c))])
        return np.clip(np.diff(cc, axis=1), 0, 1)
    cm = np.mean(cums, 0)
    return decode(cm), probs(cm), K.bal_acc(y[tr_all], decode(np.mean(train_cums, 0)))


def run_split(d, mrec):
    seed, fold = mrec["seed"], mrec["fold"]
    tr, te = np.array(mrec["train_idx"]), np.array(mrec["test_idx"])
    inner = [(np.array(a), np.array(b)) for a, b in mrec["inner_folds"]]
    y, X = d["y"], d["flat"]
    rec = {"seed": seed, "fold": fold, "fold_fingerprint": mrec["fold_fingerprint"], "train_idx": tr.tolist(),
           "test_idx": te.tolist(), "true": y[te].tolist(), "configs": {}}
    views = [(inp, f"k{k}", np.array(mrec["supports"]["none"]["outer"][str(k)]),
              [np.array(mrec["supports"]["none"]["inner"][j][str(k)]) for j in range(K.N_INNER)])
             for inp in K.INPUTS for k in KS]
    views += [("img", f"bio_{b}", K.biomarker_columns(i), [K.biomarker_columns(i)] * K.N_INNER)
              for b, i in K.BIOMARKERS.items()]
    for inputs, view, cols, icols in views:
        Xd = K.design(X, d, inputs, cols)
        for fam in ("rf", "xgb"):
            for hp in HP[fam]:
                pred, p, tb = fit_tree(fam, hp, Xd[tr], y[tr], Xd[te], seed + fold)
                sc = []
                for j, (itr, iva) in enumerate(inner):
                    Xi = K.design(X, d, inputs, icols[j])
                    pi, _, _ = fit_tree(fam, hp, Xi[itr], y[itr], Xi[iva], seed + fold)
                    sc.append(K.bal_acc(y[iva], pi))
                rec["configs"][K.cid(fam, inputs, view, "none", hp)] = {"pred": pred.tolist(), "probs": K.r6(p),
                                                                       "inner": sc, "train_bal": tb}
        # the ordinal MLP uses the OUTER selection for all inner models (selection was fitted on the outer
        # training set only, so no outer-test information enters)
        pred, p, tb = fit_ordmlp(Xd, y, inner, te, seed * 100 + fold)
        rec["configs"][K.cid("ordmlp", inputs, view, "none", "-")] = {"pred": pred.tolist(), "probs": K.r6(p),
                                                                     "inner": None, "train_bal": tb}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(4)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    d = K.load()
    written = []
    for seed in K.SEEDS:
        for f in range(5):
            path = K.split_path("extra", seed, f)
            mp = K.split_path("main", seed, f)
            if os.path.exists(path):
                continue
            if not os.path.exists(mp):
                print(f"seed {seed} fold {f}: main record missing, skipped", flush=True)
                continue
            t0 = time.time()
            rec = run_split(d, json.load(open(mp)))
            rec["seconds"] = round(time.time() - t0, 1)
            K.write_once(path, rec)
            written.append(os.path.relpath(path, K.ROOT))
            print(f"extra: seed {seed} fold {f} done in {rec['seconds']:.0f}s ({len(rec['configs'])} configs)", flush=True)
            if args.limit and len(written) >= args.limit:
                break
        if args.limit and len(written) >= args.limit:
            break
    K.manifest_entry(" ".join(["python", "scripts/experiments/leakage_safe_v1/run_extra_models.py"] + sys.argv[1:]),
                     started, written, note=f"extra families: {len(written)} new split files")


if __name__ == "__main__":
    main()
