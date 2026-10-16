"""
MLP architecture control.

The transformer's claim is that attention over 246 ROI tokens buys something.
The control for that claim is a plain feed-forward network on the SAME
features, the SAME folds and the SAME preprocessing -- no attention, no ROI
tokens, just a flattened vector. If the MLP matches the transformer, the
attention machinery is not earning its complexity at this sample size.

Features: 246 ROIs x 4 biomarkers (mALFF, mReHo, DC_z, unsigned FC strength)
= 984 inputs, which is what the transformer sees per ROI, flattened.

Training pool only. The held-out test set is never loaded.
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heads import build_head
from train_transformer import (MIN_FINAL_EPOCHS, N_STAGES, STAGES, Scaler,
                               evaluate, load_all)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "derivatives", "model_training")
torch.set_num_threads(1)

_log = []


def log(m=""):
    print(m, flush=True)
    _log.append(str(m))


class MLP(nn.Module):
    def __init__(self, d_in, hidden, head_kind, dropout=0.3):
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(d_in, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.GELU(),
        )
        self.head = build_head(head_kind, hidden, dropout=dropout)

    def forward(self, x):
        return self.head(self.body(x))


def run_fold(X, y, tr, va, head_kind, hidden, lr, wd, dropout, epochs,
             patience, seed):
    sc_mu, sc_sd = X[tr].mean(0), X[tr].std(0) + 1e-8   # TRAIN FOLD ONLY
    Xtr = torch.from_numpy(((X[tr] - sc_mu) / sc_sd).astype(np.float32))
    Xva = torch.from_numpy(((X[va] - sc_mu) / sc_sd).astype(np.float32))
    ytr, yva = torch.from_numpy(y[tr]), torch.from_numpy(y[va])
    counts = np.bincount(y[tr], minlength=N_STAGES).astype(np.float32)
    w = torch.from_numpy((counts.sum() / (N_STAGES * np.maximum(counts, 1))
                          ).astype(np.float32))

    torch.manual_seed(seed)
    model = MLP(X.shape[1], hidden, head_kind, dropout)
    if hasattr(model.head, "init_from_prior"):
        model.head.init_from_prior(counts)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
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
        m = evaluate(model.head.probabilities(out).numpy(), y[va],
                     pred=model.head.decode(out).numpy())
    m["best_epoch"] = best["epoch"]
    return m, sum(p.numel() for p in model.parameters())


def main():
    reg, fc, y, subjects, mask, pool, test, fold, smeta = load_all()
    fcu = np.load(os.path.join(ROOT, "derivatives", "model_ready",
                               "X_fc_roi_unsigned.npy")).astype(np.float64)
    X = np.concatenate([reg.astype(np.float64), fcu[:, :, None]],
                       axis=2).reshape(len(y), -1)
    log("=" * 78)
    log("MLP ARCHITECTURE CONTROL -- no attention, same folds, same features")
    log("=" * 78)
    log(f"features {X.shape} (246 ROIs x 4 biomarkers, flattened)")
    log(f"split fingerprint {smeta['split_fingerprint']}; test set NOT loaded")

    results = {}
    for head_kind in ("multiclass", "ordinal_fixed"):
        for hidden in (32, 64):
            per, npar = [], 0
            for f in range(smeta["n_folds"]):
                va = np.array([i for i in pool if fold[i] == f])
                tr = np.array([i for i in pool if fold[i] != f])
                m, npar = run_fold(X, y, tr, va, head_kind, hidden,
                                   lr=1e-3, wd=1e-2, dropout=0.3,
                                   epochs=120, patience=20,
                                   seed=20261015 + f)
                per.append(m)
            summ = {k: {"mean": float(np.mean([p[k] for p in per])),
                        "sd": float(np.std([p[k] for p in per], ddof=1))}
                    for k in ("accuracy", "balanced_accuracy", "macro_f1",
                              "mae_stages", "qwk", "ece")}
            C = np.sum([np.array(p["confusion_matrix"]) for p in per], axis=0)
            name = f"mlp_{head_kind}_h{hidden}"
            results[name] = {"summary": summ, "folds": per,
                             "confusion_matrix_summed": C.tolist(),
                             "n_parameters": int(npar),
                             "n_predicted_classes_per_fold":
                                 [p["n_predicted_classes"] for p in per]}
            log(f"\n  {name}  ({npar:,} parameters)")
            log(f"    balanced_accuracy {summ['balanced_accuracy']['mean']:.4f} "
                f"+/- {summ['balanced_accuracy']['sd']:.4f}")
            log(f"    macro_f1          {summ['macro_f1']['mean']:.4f} "
                f"+/- {summ['macro_f1']['sd']:.4f}")
            log(f"    mae_stages        {summ['mae_stages']['mean']:.4f} "
                f"+/- {summ['mae_stages']['sd']:.4f}")
            log(f"    qwk               {summ['qwk']['mean']:+.4f} "
                f"+/- {summ['qwk']['sd']:.4f}")
            log(f"    distinct classes predicted per fold "
                f"{results[name]['n_predicted_classes_per_fold']}")

    p = os.path.join(OUT, "results_mlp_control.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"split_fingerprint": smeta["split_fingerprint"],
                   "feature_shape": list(X.shape),
                   "results": results, "test_set_touched": False}, f, indent=2)
    log(f"\nwrote {os.path.relpath(p, ROOT)}")
    with open(os.path.join(OUT, "train_mlp_control.log"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")


if __name__ == "__main__":
    main()
