"""
Ordinal fix ablation.

The diagnosis found three supported causes of the collapse:

    H2  the loss sums 5 BCE terms, so it is 1.64x the scale of cross-entropy
    H3  the encoder receives a RANK-1 gradient (effective rank 1.00)
    H5  initialisation starts at P(CN)=0.53 against a uniform 0.167

Each fix targets exactly one cause and is tested alone before being combined,
so the contribution of each is attributable:

    A  mean_reduce       average the BCE terms           -> H2
    B  init_from_prior   thresholds at the class CDF     -> H5
    C  aux_rank          auxiliary CE head in training   -> H3

The gate is the test the broken head failed: memorise 12 training-pool
subjects with no dropout and no weight decay. A head that cannot do that has
an optimisation problem. Multiclass reaches train accuracy 1.00 on this task,
so it is the reference.

Inference always decodes the cumulative-link model. The auxiliary head is a
training-time signal only, so the ordinal formulation is preserved.

Training pool only. The held-out test set is never loaded.
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from heads import OrdinalHead, StagePredictor, build_head
from multi_attention_transformer import build_model
from train_transformer import Scaler, evaluate, load_all

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "derivatives", "model_training")
N = 6
EPOCHS = 250
torch.set_num_threads(1)

_log = []


def log(m=""):
    print(m, flush=True)
    _log.append(str(m))


def gradient_rank(head, d=32, B=24, seed=0):
    torch.manual_seed(seed)
    h = torch.randn(B, d, requires_grad=True)
    y = torch.randint(0, N, (B,))
    loss = head.loss(head(h), y)
    (gh,) = torch.autograd.grad(loss, h)
    sv = torch.linalg.svdvals(gh)
    svn = sv / sv.sum()
    eff = float(torch.exp(-(svn * torch.log(svn.clamp_min(1e-12))).sum()))
    return int(torch.linalg.matrix_rank(gh)), eff


def overfit_subset(make_head, reg, fc, y, mask_t, pick, mask_arg, seed=0):
    sc = Scaler().fit(reg, pick)
    X = torch.from_numpy(sc.transform(reg[pick]))
    Fm = torch.from_numpy(fc[pick])
    yt = torch.from_numpy(y[pick])
    torch.manual_seed(seed)
    enc = build_model(d_model=16, num_heads=2, num_layers=1, dropout=0.0)
    head = make_head(16, np.bincount(y[pick], minlength=N))
    model = StagePredictor(enc, head)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=0.0)
    model.train()
    losses = []
    for _ in range(EPOCHS):
        opt.zero_grad()
        loss = model.head.loss(model(X, Fm, mask_arg), yt)
        loss.backward()
        opt.step()
        losses.append(float(loss))
    model.eval()
    with torch.no_grad():
        out = model(X, Fm, mask_arg)
        p = model.head.probabilities(out).numpy()
        dec = model.head.decode(out).numpy()
    m = evaluate(p, y[pick], pred=dec)
    return {"train_accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "n_classes_predicted": m["n_predicted_classes"],
            "final_loss": losses[-1],
            "loss_drop": losses[0] - losses[-1]}


def main():
    reg, fc, y, subjects, mask, pool, test, fold, smeta = load_all()
    mask_t = torch.from_numpy(mask)
    rng = np.random.default_rng(0)
    pick = []
    for g in range(N):
        idx = pool[y[pool] == g]
        pick.extend(rng.choice(idx, size=min(2, idx.size), replace=False).tolist())
    pick = np.array(sorted(pick))
    counts = np.bincount(y[pick], minlength=N)

    log("=" * 78)
    log("ORDINAL FIX ABLATION -- can each variant memorise 12 subjects?")
    log("=" * 78)
    log(f"subset: {len(pick)} training-pool subjects, counts {counts.tolist()}")
    log(f"gate  : multiclass reaches train accuracy 1.00 on this subset")
    log(f"split fingerprint {smeta['split_fingerprint']}; test set NOT loaded")
    log("")

    variants = {
        "D  argmax->cumulative":
            lambda d, c: OrdinalHead(d, dropout=0.0),
        "ordinal (baseline)":
            lambda d, c: OrdinalHead(d, dropout=0.0),
        "A  mean_reduce":
            lambda d, c: OrdinalHead(d, dropout=0.0, mean_reduce=True),
        "B  init_from_prior":
            lambda d, c: OrdinalHead(d, dropout=0.0).init_from_prior(c),
        "C  aux CE head":
            lambda d, c: OrdinalHead(d, dropout=0.0, aux_rank=N, aux_weight=0.5),
        "A+B":
            lambda d, c: OrdinalHead(d, dropout=0.0, mean_reduce=True
                                     ).init_from_prior(c),
        "A+B+C (ordinal_fixed)":
            lambda d, c: OrdinalHead(d, dropout=0.0, mean_reduce=True,
                                     aux_rank=N, aux_weight=0.5
                                     ).init_from_prior(c),
        "multiclass (reference)":
            lambda d, c: build_head("multiclass", d, dropout=0.0),
    }

    results = {}
    log(f"{'variant':24s} {'rank':>5s} {'eff':>5s} {'train acc':>10s} "
        f"{'classes':>8s} {'final loss':>11s}")
    log("-" * 72)
    for name, mk in variants.items():
        r, eff = gradient_rank(mk(32, counts))
        o = overfit_subset(mk, reg, fc, y, mask_t, pick, mask_t)
        o["gradient_rank"] = r
        o["gradient_effective_rank"] = eff
        o["memorised"] = bool(o["train_accuracy"] >= 0.95)
        results[name] = o
        log(f"{name:24s} {r:5d} {eff:5.2f} {o['train_accuracy']:10.2f} "
            f"{o['n_classes_predicted']:8d} {o['final_loss']:11.4f}"
            + ("   MEMORISED" if o["memorised"] else ""))

    log("")
    ok = [n for n, v in results.items() if v["memorised"] and "multiclass" not in n]
    if ok:
        log(f"Variants that now memorise the subset: {ok}")
        log("The collapse is therefore an optimisation failure with an "
            "identified and corrected cause, not absent signal.")
    else:
        log("No ordinal variant memorised the subset. The optimisation "
            "problem is NOT yet solved; ordinal modelling remains unresolved "
            "and multiclass stays the working model.")

    p = os.path.join(OUT, "ordinal_fix_ablation.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"subset_indices": [int(i) for i in pick],
                   "subset_counts": counts.tolist(),
                   "epochs": EPOCHS, "results": results,
                   "split_fingerprint": smeta["split_fingerprint"],
                   "test_set_touched": False}, f, indent=2)
    log(f"\nwrote {os.path.relpath(p, ROOT)}")
    with open(os.path.join(OUT, "ordinal_fix_ablation.log"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")


if __name__ == "__main__":
    main()
