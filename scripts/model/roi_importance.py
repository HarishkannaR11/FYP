"""
Phase 6D: ROI importance and its stability.

Method: permutation importance on the frozen reference MLP. For each fold a
model is trained on the training subjects, then a feature group is shuffled
ACROSS THE VALIDATION SUBJECTS and the drop in validation balanced accuracy is
recorded. Shuffling breaks the association between that group and the label
while preserving its marginal distribution.

Grouping follows the verified feature map (index = roi*4 + channel):

    per ROI        all four channels of one ROI permuted together (246 groups)
    per biomarker  all 246 ROIs of one channel permuted together (4 groups)

FC is not treated as an ordinary ROI measurement: channel 3 is already an
ROI-level aggregation of the connectivity matrix (mean_{j!=i} |z_ij|), so
permuting it permutes that ROI's total connectivity magnitude, not an edge.

Controls, because 246 importance values will always have a maximum
-------------------------------------------------------------------
    label-shuffled null   the whole procedure repeated with permuted training
                          labels. Any apparent importance there is an artefact
                          of the estimator, not signal.
    rank stability        Spearman correlation between the ROI importance
                          rankings of different folds and seeds, plus top-20
                          overlap. An ROI set that does not recur across
                          partitions is not a finding.

Importance is computed only on validation subjects, with models trained
without them. No importance value feeds back into training or feature
selection. The held-out test set is never loaded.
"""
import json
import os
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from repeated_cv import SEEDS, N_FOLDS, stratified_folds, run_mlp
from train_transformer import N_STAGES, STAGES, evaluate, load_all
from mlp_control import MLP
from heads import build_head

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "derivatives", "model_ready")
OUT = os.path.join(ROOT, "derivatives", "model_training", "roi_importance")
LUT = os.path.join(ROOT, "derivatives", "fsfast", "FINAL_6GROUP_AUDIT",
                   "dataset_brainnetome_summary.csv")
N_ROI, N_CH, N_REPEATS = 246, 4, 5
CHANNELS = ["mALFF", "mReHo", "DC_z", "FCstr"]
torch.set_num_threads(1)

_log = []


def log(m=""):
    print(m, flush=True)
    _log.append(str(m))


def roi_labels():
    import csv
    rows = list(csv.DictReader(open(LUT)))
    return [int(r["ROI_ID"]) for r in rows], [r["ROI_NAME"] for r in rows]


def train_fold(X, y, tr, seed, hidden=64):
    """Train the frozen reference configuration on one fold."""
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-8
    Xtr = torch.from_numpy(((X[tr] - mu) / sd).astype(np.float32))
    ytr = torch.from_numpy(y[tr])
    counts = np.bincount(y[tr], minlength=N_STAGES).astype(np.float32)
    w = torch.from_numpy((counts.sum() / (N_STAGES * np.maximum(counts, 1))
                          ).astype(np.float32))
    torch.manual_seed(seed)
    model = MLP(X.shape[1], hidden, "multiclass", 0.3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    g = torch.Generator().manual_seed(seed)
    for _ in range(60):
        model.train()
        perm = torch.randperm(len(tr), generator=g)
        for s in range(0, len(perm), 16):
            b = perm[s:s + 16]
            opt.zero_grad()
            model.head.loss(model(Xtr[b]), ytr[b], weight=w).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    model.eval()
    return model, mu, sd


def score(model, Xv, yv):
    with torch.no_grad():
        out = model(torch.from_numpy(Xv.astype(np.float32)))
        p = model.head.probabilities(out).numpy()
        d = model.head.decode(out).numpy()
    return evaluate(p, yv, pred=d)["balanced_accuracy"]


def permutation_importance(model, Xv, yv, groups, rng, repeats=N_REPEATS):
    base = score(model, Xv, yv)
    imp = np.zeros(len(groups))
    for gi, cols in enumerate(groups):
        drops = []
        for _ in range(repeats):
            Xp = Xv.copy()
            order = rng.permutation(Xp.shape[0])
            Xp[:, cols] = Xp[np.ix_(order, cols)]
            drops.append(base - score(model, Xp, yv))
        imp[gi] = float(np.mean(drops))
    return base, imp


def main():
    os.makedirs(OUT, exist_ok=True)
    reg, fc, y, subjects, mask, pool, test, fold0, smeta = load_all()
    fcu = np.load(os.path.join(SRC, "X_fc_roi_unsigned.npy")).astype(np.float64)
    X = np.concatenate([reg.astype(np.float64), fcu[:, :, None]],
                       axis=2).reshape(len(y), -1)
    ids, names = roi_labels()

    roi_groups = [list(range(r * N_CH, r * N_CH + N_CH)) for r in range(N_ROI)]
    bio_groups = [[r * N_CH + c for r in range(N_ROI)] for c in range(N_CH)]

    log("=" * 86)
    log("PHASE 6D -- ROI IMPORTANCE AND STABILITY")
    log("=" * 86)
    log(f"features {X.shape}; {N_ROI} ROI groups of {N_CH}; "
        f"{N_CH} biomarker groups of {N_ROI}")
    log(f"{N_REPEATS} permutation repeats; importance on validation subjects only")
    log(f"held-out test ({test.size}) NOT loaded")

    roi_imp, bio_imp, null_imp, bases = [], [], [], []
    t0 = time.time()
    for seed in SEEDS:
        fo = stratified_folds(y, pool, N_FOLDS, np.random.default_rng(seed))
        for f in range(N_FOLDS):
            va = np.array([i for i in pool if fo[int(i)] == f])
            tr = np.array([i for i in pool if fo[int(i)] != f])
            model, mu, sd = train_fold(X, y, tr, seed + f)
            Xv = (X[va] - mu) / sd
            rng = np.random.default_rng(seed * 1000 + f)
            base, ri = permutation_importance(model, Xv, y[va], roi_groups, rng)
            _, bi = permutation_importance(model, Xv, y[va], bio_groups, rng)
            roi_imp.append(ri)
            bio_imp.append(bi)
            bases.append(base)

            # control: same procedure, labels shuffled before training
            yperm = y.copy()
            rp = np.random.default_rng(seed * 7919 + f)
            yperm[tr] = rp.permutation(yperm[tr])
            mdl_n, mun, sdn = train_fold(X, yperm, tr, seed + f)
            Xvn = (X[va] - mun) / sdn
            _, ni = permutation_importance(mdl_n, Xvn, y[va], roi_groups, rng,
                                           repeats=2)
            null_imp.append(ni)
        log(f"  seed {seed} done  ({time.time() - t0:.0f}s elapsed)")

    R = np.vstack(roi_imp)       # [25, 246]
    B = np.vstack(bio_imp)       # [25, 4]
    Nl = np.vstack(null_imp)

    log("")
    log("=" * 86)
    log("BIOMARKER-LEVEL IMPORTANCE (drop in validation balanced accuracy)")
    log("=" * 86)
    log(f"mean validation balanced accuracy before permutation: "
        f"{np.mean(bases):.4f}")
    for c in range(N_CH):
        log(f"  {CHANNELS[c]:8s} {B[:, c].mean():+.4f} +/- {B[:, c].std(ddof=1):.4f}"
            f"   folds with a positive drop: {int((B[:, c] > 0).sum())}/25")

    log("")
    log("=" * 86)
    log("ROI IMPORTANCE vs LABEL-SHUFFLED CONTROL")
    log("=" * 86)
    log(f"  real   mean {R.mean():+.5f}  sd {R.std():.5f}  max {R.max():+.4f}")
    log(f"  null   mean {Nl.mean():+.5f}  sd {Nl.std():.5f}  max {Nl.max():+.4f}")
    thr = np.percentile(Nl, 95)
    log(f"  95th percentile of the null: {thr:+.5f}")
    log(f"  ROI-folds exceeding it in the real run: "
        f"{int((R > thr).sum())}/{R.size} ({(R > thr).mean():.1%}; "
        f"5% expected by construction)")

    log("")
    log("=" * 86)
    log("RANK STABILITY ACROSS THE 25 FOLDS")
    log("=" * 86)
    from scipy import stats
    rhos = []
    for i in range(len(R)):
        for j in range(i + 1, len(R)):
            rhos.append(stats.spearmanr(R[i], R[j]).statistic)
    rhos = np.array(rhos)
    log(f"  pairwise Spearman between fold rankings: mean {rhos.mean():+.3f}, "
        f"median {np.median(rhos):+.3f}, range [{rhos.min():+.3f}, {rhos.max():+.3f}]")
    tops = [set(np.argsort(-R[i])[:20]) for i in range(len(R))]
    ov = [len(tops[i] & tops[j]) for i in range(len(tops))
          for j in range(i + 1, len(tops))]
    log(f"  top-20 ROI overlap between folds: mean {np.mean(ov):.1f}/20 "
        f"(chance {20 * 20 / N_ROI:.1f}/20)")

    mean_imp = R.mean(0)
    freq = np.array([sum(1 for i in range(len(R))
                         if r in np.argsort(-R[i])[:20]) for r in range(N_ROI)])
    log("")
    log("  top 15 ROIs by mean importance")
    log(f"  {'ROI':>4s} {'name':14s} {'mean imp':>10s} {'sd':>8s} "
        f"{'in top-20':>10s}")
    for r in np.argsort(-mean_imp)[:15]:
        log(f"  {ids[r]:4d} {names[r]:14.14s} {mean_imp[r]:+10.5f} "
            f"{R[:, r].std(ddof=1):8.5f} {freq[r]:7d}/25")

    out = {"design": {"seeds": SEEDS, "n_folds": N_FOLDS,
                      "permutation_repeats": N_REPEATS,
                      "grouping": "index = roi*4 + channel",
                      "test_set_touched": False},
           "mean_validation_balanced_accuracy": float(np.mean(bases)),
           "biomarker_importance": {CHANNELS[c]: {
               "mean": float(B[:, c].mean()), "sd": float(B[:, c].std(ddof=1)),
               "folds_positive": int((B[:, c] > 0).sum())} for c in range(N_CH)},
           "null_control": {"mean": float(Nl.mean()), "sd": float(Nl.std()),
                            "p95": float(thr),
                            "fraction_real_exceeding_p95": float((R > thr).mean())},
           "rank_stability": {"mean_pairwise_spearman": float(rhos.mean()),
                              "median": float(np.median(rhos)),
                              "top20_overlap_mean": float(np.mean(ov)),
                              "top20_overlap_chance": 20 * 20 / N_ROI},
           "roi_importance": [{"roi_id": ids[r], "roi_name": names[r],
                               "mean": float(mean_imp[r]),
                               "sd": float(R[:, r].std(ddof=1)),
                               "times_in_top20": int(freq[r])}
                              for r in np.argsort(-mean_imp)]}
    p = os.path.join(OUT, "roi_importance.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log(f"\nwrote {os.path.relpath(p, ROOT)}")
    with open(os.path.join(OUT, "roi_importance.log"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")


if __name__ == "__main__":
    main()
