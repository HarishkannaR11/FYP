"""
Final model-selection check: DC-only vs all-four biomarkers.

Both candidates were already evaluated in the Phase 6A ablation on identical
seeds, identical fold assignments and identical subject IDs, with per-fold
predictions saved. This script therefore VERIFIES and ANALYSES those saved
outputs rather than retraining, which would consume compute without adding
evidence and would risk a different result from a different random draw.

Checks performed before any comparison:
  * the DC channel really is DC_z, and the feature mapping is as documented
  * both candidates used the same fold fingerprints, subject IDs and folds
  * every recorded metric recomputes from the saved predictions (abort if not)
"""
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "derivatives", "model_ready")
ABL = os.path.join(ROOT, "derivatives", "model_training", "biomarker_ablation")
OUT = os.path.join(ROOT, "derivatives", "model_training")
STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
N = 6
A, B = "dc_only", "all_four"
KEYS = ("balanced_accuracy", "accuracy", "macro_f1", "mae_stages", "qwk")

_log = []


def log(m=""):
    print(m, flush=True)
    _log.append(str(m))


def recompute(rec):
    y, pred = np.array(rec["true"]), np.array(rec["pred"])
    rec_, f1s = [], []
    for g in range(N):
        m = y == g
        tp = int(((pred == g) & m).sum())
        supp, npred = int(m.sum()), int((pred == g).sum())
        rec_.append(tp / supp if supp else np.nan)
        den = 2 * tp + (npred - tp) + (supp - tp)
        f1s.append(2 * tp / den if den > 0 else 0.0)
    C = np.zeros((N, N))
    for a, b in zip(y, pred):
        C[a, b] += 1
    w = np.array([[(i - j) ** 2 for j in range(N)] for i in range(N)], float)
    w /= (N - 1) ** 2
    E = np.outer(np.bincount(y, minlength=N), np.bincount(pred, minlength=N)).astype(float)
    E = E / E.sum() * C.sum()
    den = (w * E).sum()
    return {"accuracy": float((pred == y).mean()),
            "balanced_accuracy": float(np.nanmean(rec_)),
            "macro_f1": float(np.mean(f1s)),
            "mae_stages": float(np.abs(pred - y).mean()),
            "qwk": float(1 - (w * C).sum() / den) if den > 0 else np.nan}


def main():
    log("=" * 88)
    log("FINAL SELECTION CHECK -- DC-only vs all-four")
    log("=" * 88)

    # ---- feature mapping
    meta = json.load(open(os.path.join(SRC, "metadata.json")))
    reg = np.load(os.path.join(SRC, "X_regional.npy"))
    ch = meta["regional_channels"]
    log(f"\n[1] FEATURE MAPPING")
    log(f"    regional_channels = {ch}")
    ok_dc = ch[2] == "DC_z"
    log(f"    channel index 2 is DC_z: {ok_dc}")
    if not ok_dc:
        log("    ABORT: DC channel is not where the ablation assumed")
        sys.exit(1)
    d = json.load(open(os.path.join(ABL, "ablation_records.json")))
    fm = d["feature_mapping"]
    log(f"    layout: {fm['layout']}")
    log(f"    dc_only input = channel 2 only, {reg.shape[1]} ROIs")
    log(f"    all_four input = {fm['channels']}")
    log(f"    FC caveat: {fm['fc_caveat'][:90]}...")

    recs = d["records"]
    ra = [r for r in recs if r["config"] == A]
    rb = [r for r in recs if r["config"] == B]
    log(f"\n[2] MATCHED DESIGN")
    log(f"    {A}: {len(ra)} evaluations | {B}: {len(rb)} evaluations")
    seeds = sorted({r["seed"] for r in ra})
    mismatch = 0
    for s in seeds:
        for f in range(5):
            x = next(r for r in ra if r["seed"] == s and r["fold"] == f)
            yy = next(r for r in rb if r["seed"] == s and r["fold"] == f)
            if (x["fold_fingerprint"] != yy["fold_fingerprint"]
                    or x["subjects"] != yy["subjects"] or x["true"] != yy["true"]):
                mismatch += 1
    log(f"    identical fold fingerprints, subject IDs and labels in all 25 "
        f"pairs: {mismatch == 0}")
    if mismatch:
        log("    ABORT: candidates were not evaluated on identical folds")
        sys.exit(1)
    log(f"    parameters: {A} {ra[0]['n_parameters']:,} | "
        f"{B} {rb[0]['n_parameters']:,}")
    log(f"    input dim : {A} {ra[0]['input_dim']} | {B} {rb[0]['input_dim']}")

    # ---- reproducibility gate
    log(f"\n[3] METRIC REPRODUCIBILITY")
    bad = []
    for r in ra + rb:
        rc = recompute(r)
        for k in ("balanced_accuracy", "macro_f1", "mae_stages", "qwk"):
            if not np.isclose(rc[k], r[k], atol=1e-6, equal_nan=True):
                bad.append((r["config"], r["seed"], r["fold"], k))
    log(f"    {len(ra + rb) * 4} checks, {len(bad)} mismatches")
    if bad:
        for x in bad[:5]:
            log(f"    MISMATCH {x}")
        log("    ABORT: recorded metrics are not reproducible")
        sys.exit(1)
    log("    all metrics reproduce from saved predictions")

    # ---- verify historical numbers
    log(f"\n[4] VERIFY HISTORICAL FIGURES")
    HIST = {A: {"balanced_accuracy": 0.307, "macro_f1": 0.264,
                "mae_stages": 1.642, "qwk": 0.172},
            B: {"balanced_accuracy": 0.278, "macro_f1": 0.246,
                "mae_stages": 1.638, "qwk": 0.171}}
    sm = {}
    for c, rr in ((A, ra), (B, rb)):
        sm[c] = {k: [float(np.mean([r[k] for r in rr if r["seed"] == s]))
                     for s in seeds] for k in KEYS}
        for k, v in HIST[c].items():
            got = float(np.mean(sm[c][k]))
            flag = "OK" if abs(got - v) < 0.002 else "DIFFERS"
            log(f"    {c:10s} {k:20s} recorded {v:.3f}  recomputed {got:.3f}  {flag}")

    # ---- summary
    log(f"\n[5] SEED-LEVEL SUMMARY  (+/- = partition sensitivity, not a "
        f"standard error)")
    log(f"    {'candidate':10s} {'bal.acc':>15s} {'accuracy':>15s} "
        f"{'macro-F1':>15s} {'MAE':>15s} {'QWK':>15s} {'cls':>5s}")
    for c, rr in ((A, ra), (B, rb)):
        cls = [r["n_predicted_classes"] for r in rr]
        f = lambda k: (f"{np.mean(sm[c][k]):.3f}+/-"
                       f"{np.std(sm[c][k], ddof=1):.3f}")
        log(f"    {c:10s} {f('balanced_accuracy'):>15s} {f('accuracy'):>15s} "
            f"{f('macro_f1'):>15s} {f('mae_stages'):>15s} {f('qwk'):>15s} "
            f"{min(cls)}-{max(cls)}")

    # ---- paired
    log(f"\n[6] PAIRED DIFFERENCES ({A} minus {B}, identical folds)")
    diffs = {k: [] for k in KEYS}
    for s in seeds:
        for f in range(5):
            x = next(r for r in ra if r["seed"] == s and r["fold"] == f)
            yy = next(r for r in rb if r["seed"] == s and r["fold"] == f)
            for k in KEYS:
                diffs[k].append(x[k] - yy[k])
    paired = {}
    for k in KEYS:
        per_seed = [float(np.mean(diffs[k][i * 5:(i + 1) * 5]))
                    for i in range(len(seeds))]
        better = sum(1 for v in per_seed if (v < 0 if k == "mae_stages" else v > 0))
        folds_better = sum(1 for v in diffs[k]
                           if (v < 0 if k == "mae_stages" else v > 0))
        paired[k] = {"fold_mean": float(np.mean(diffs[k])),
                     "fold_sd": float(np.std(diffs[k], ddof=1)),
                     "per_seed": per_seed, "seeds_favouring_dc": better,
                     "folds_favouring_dc": folds_better}
        log(f"    {k:20s} {np.mean(diffs[k]):+.4f} (fold sd "
            f"{np.std(diffs[k], ddof=1):.4f})  seeds {better}/5  "
            f"folds {folds_better}/25")
        log(f"    {'':20s} per seed: {np.round(per_seed, 3).tolist()}")

    # ---- per class + confusion
    log(f"\n[7] PER-CLASS RECALL and COVERAGE (pooled over 25 folds)")
    log(f"    {'candidate':10s}" + "".join(f"{s:>8s}" for s in STAGES)
        + "   never predicted")
    perclass = {}
    conf = {}
    for c, rr in ((A, ra), (B, rb)):
        y = np.concatenate([r["true"] for r in rr])
        p = np.concatenate([r["pred"] for r in rr])
        rec_ = [float((p[y == g] == g).mean()) for g in range(N)]
        never = [STAGES[g] for g in range(N) if (p == g).sum() == 0]
        C = np.zeros((N, N), int)
        for a_, b_ in zip(y, p):
            C[a_, b_] += 1
        perclass[c] = {STAGES[g]: rec_[g] for g in range(N)}
        conf[c] = C.tolist()
        log(f"    {c:10s}" + "".join(f"{v:8.2f}" for v in rec_)
            + f"   {','.join(never) if never else '-'}")
    for c in (A, B):
        log(f"\n    confusion, {c} (rows true, cols predicted)")
        log("            " + "".join(f"{s:>6s}" for s in STAGES))
        for g in range(N):
            log(f"      {STAGES[g]:6s}" + "".join(f"{v:6d}" for v in conf[c][g]))

    # ---- decision
    log(f"\n[8] DECISION")
    ba = paired["balanced_accuracy"]
    consistent = ba["seeds_favouring_dc"] >= 4
    worst_dc = min(perclass[A].values())
    worst_af = min(perclass[B].values())
    acceptable = worst_dc >= worst_af - 0.10
    log(f"    advantage consistent across seeds: {ba['seeds_favouring_dc']}/5 "
        f"-> {consistent}")
    log(f"    fold-level: {ba['folds_favouring_dc']}/25 folds favour {A}")
    log(f"    magnitude {ba['fold_mean']:+.4f} vs fold sd {ba['fold_sd']:.4f} "
        f"(ratio {abs(ba['fold_mean']) / ba['fold_sd']:.2f})")
    log(f"    worst per-class recall: {A} {worst_dc:.2f} vs {B} {worst_af:.2f} "
        f"-> acceptable {acceptable}")
    decision = ("ADOPT dc_only" if consistent and acceptable
                else f"RETAIN {B} provisionally")
    log(f"    => {decision}")

    out = {"candidates": {A: {"input_dim": ra[0]["input_dim"],
                              "n_parameters": ra[0]["n_parameters"]},
                          B: {"input_dim": rb[0]["input_dim"],
                              "n_parameters": rb[0]["n_parameters"]}},
           "seed_level": {c: {k: {"mean": float(np.mean(sm[c][k])),
                                  "sd": float(np.std(sm[c][k], ddof=1)),
                                  "per_seed": sm[c][k]} for k in KEYS}
                          for c in (A, B)},
           "paired": paired, "per_class_recall": perclass,
           "confusion": conf, "decision": decision,
           "metrics_reproduced": True, "test_set_touched": False}
    p = os.path.join(OUT, "dc_vs_allfour_decision.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log(f"\nwrote {os.path.relpath(p, ROOT)}")
    with open(os.path.join(OUT, "dc_vs_allfour_decision.log"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")


if __name__ == "__main__":
    main()
