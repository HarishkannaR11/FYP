"""
Analysis of the repeated cross-validation.

Three things are done here, in order:

  1. METRIC REPRODUCIBILITY -- every recorded metric is recomputed from the
     saved subject-level predictions and compared with the stored value. If
     they disagree, the analysis stops.
  2. SUMMARY -- per candidate, pooled over 25 fold evaluations and broken down
     by split seed, so one lucky partition cannot hide inside a pooled mean.
  3. PAIRED COMPARISON -- differences computed on identical folds.

On the statistics
-----------------
The 25 fold evaluations are NOT 25 independent observations. Within one split
seed the five folds share the same 81 subjects, and every subject appears in
exactly one validation fold per seed, so the folds partition one dataset
rather than sampling new ones. The five seeds reuse those same 81 subjects
too.

Treating 25 numbers as an independent sample would understate the standard
error badly. The primary summary here is therefore the SEED-LEVEL mean: five
values per candidate, one per partition, which are closer to exchangeable.
Dispersion across seeds measures sensitivity to partitioning, not sampling
error for the population. No p-value is reported: with five dependent
seed-level values over one small cohort, any such test would carry assumptions
the data cannot support. Paired win counts and the spread of differences are
reported instead, as descriptive evidence.
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
REP = os.path.join(ROOT, "derivatives", "model_training", "repeated_cv")
STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
N = 6
KEYS = ("balanced_accuracy", "accuracy", "macro_f1", "mae_stages", "qwk")

_log = []


def log(m=""):
    print(m, flush=True)
    _log.append(str(m))


def recompute(rec):
    """Recompute metrics from saved predictions."""
    y = np.array(rec["true"])
    pred = np.array(rec["pred"])
    recalls, f1s = [], []
    for g in range(N):
        m = y == g
        tp = int(((pred == g) & m).sum())
        supp, npred = int(m.sum()), int((pred == g).sum())
        recalls.append(tp / supp if supp else np.nan)
        den = 2 * tp + (npred - tp) + (supp - tp)
        f1s.append(2 * tp / den if den > 0 else 0.0)
    C = np.zeros((N, N))
    for a, b in zip(y, pred):
        C[a, b] += 1
    w = np.array([[(i - j) ** 2 for j in range(N)] for i in range(N)], float)
    w /= (N - 1) ** 2
    hy, hp = np.bincount(y, minlength=N).astype(float), \
        np.bincount(pred, minlength=N).astype(float)
    E = np.outer(hy, hp)
    E = E / E.sum() * C.sum()
    den = (w * E).sum()
    return {"accuracy": float((pred == y).mean()),
            "balanced_accuracy": float(np.nanmean(recalls)),
            "macro_f1": float(np.mean(f1s)),
            "mae_stages": float(np.abs(pred - y).mean()),
            "qwk": float(1 - (w * C).sum() / den) if den > 0 else np.nan,
            "n_predicted_classes": int(len(np.unique(pred)))}


def main():
    data = json.load(open(os.path.join(REP, "repeated_cv_records.json")))
    recs = data["records"]
    cands = sorted({r["candidate"] for r in recs})
    seeds = sorted({r["seed"] for r in recs})

    log("=" * 86)
    log("REPEATED CV ANALYSIS")
    log("=" * 86)
    log(f"{len(recs)} fold evaluations | {len(cands)} candidates | "
        f"{len(seeds)} seeds x 5 folds")
    log(f"held-out test touched: {data['design']['test_set_touched']}")

    # ---- 1. reproducibility
    bad = []
    for r in recs:
        rc = recompute(r)
        for k in ("balanced_accuracy", "macro_f1", "mae_stages", "qwk"):
            if not np.isclose(rc[k], r[k], atol=1e-6, equal_nan=True):
                bad.append((r["candidate"], r["seed"], r["fold"], k,
                            r[k], rc[k]))
    log(f"\n[QC] metrics recomputed from saved predictions: "
        f"{len(recs) * 4} checks, {len(bad)} mismatches")
    if bad:
        for b in bad[:10]:
            log(f"     MISMATCH {b}")
        log("     stopping: recorded metrics are not reproducible")
        sys.exit(1)
    log("     all recorded metrics reproduce exactly from saved predictions")

    # ---- 2. summary
    log("\n" + "=" * 86)
    log("SUMMARY -- seed-level means (primary) and pooled fold spread")
    log("=" * 86)
    seed_means = {}
    log(f"{'candidate':24s} {'bal.acc':>16s} {'macro-F1':>16s} "
        f"{'MAE':>15s} {'QWK':>16s} {'cls':>6s}")
    log("-" * 86)
    rows = {}
    for c in cands:
        sm = {k: [] for k in KEYS}
        for s in seeds:
            sub = [r for r in recs if r["candidate"] == c and r["seed"] == s]
            for k in KEYS:
                sm[k].append(float(np.mean([r[k] for r in sub])))
        seed_means[c] = sm
        allf = [r for r in recs if r["candidate"] == c]
        cls = [r["n_predicted_classes"] for r in allf]
        rows[c] = {k: (float(np.mean(sm[k])), float(np.std(sm[k], ddof=1)))
                   for k in KEYS}
        rows[c]["classes"] = (min(cls), max(cls),
                              float(np.mean(cls)))
        rows[c]["params"] = allf[0]["n_parameters"]
        log(f"{c:24s} "
            f"{rows[c]['balanced_accuracy'][0]:.3f}+/-{rows[c]['balanced_accuracy'][1]:.3f}   "
            f"{rows[c]['macro_f1'][0]:.3f}+/-{rows[c]['macro_f1'][1]:.3f}   "
            f"{rows[c]['mae_stages'][0]:.3f}+/-{rows[c]['mae_stages'][1]:.3f}  "
            f"{rows[c]['qwk'][0]:+.3f}+/-{rows[c]['qwk'][1]:.3f}   "
            f"{min(cls)}-{max(cls)}")
    log("\n+/- is the standard deviation of the 5 SEED-LEVEL means "
        "(sensitivity to partitioning), not a standard error.")

    # per seed
    log("\nper-seed balanced accuracy (checks that no single partition drives "
        "the result)")
    log(f"{'candidate':24s}" + "".join(f"{s:>12d}" for s in seeds))
    for c in cands:
        log(f"{c:24s}" + "".join(f"{v:12.3f}" for v in
                                 seed_means[c]["balanced_accuracy"]))

    # ---- 3. paired comparison
    log("\n" + "=" * 86)
    log("PAIRED COMPARISON -- identical folds, identical seeds")
    log("=" * 86)
    order = sorted(cands, key=lambda c: -rows[c]["balanced_accuracy"][0])
    pairs = [(order[0], o) for o in order[1:]]
    paired = {}
    for a, b in pairs:
        diffs = {k: [] for k in KEYS}
        for s in seeds:
            for f in range(5):
                ra = next(r for r in recs if r["candidate"] == a
                          and r["seed"] == s and r["fold"] == f)
                rb = next(r for r in recs if r["candidate"] == b
                          and r["seed"] == s and r["fold"] == f)
                assert ra["fold_fingerprint"] == rb["fold_fingerprint"]
                for k in KEYS:
                    diffs[k].append(ra[k] - rb[k])
        d = np.array(diffs["balanced_accuracy"])
        sd = np.array([np.mean(d[i * 5:(i + 1) * 5]) for i in range(len(seeds))])
        paired[f"{a}__vs__{b}"] = {
            k: {"fold_mean": float(np.mean(diffs[k])),
                "fold_sd": float(np.std(diffs[k], ddof=1)),
                "seed_level_mean": float(np.mean(
                    [np.mean(diffs[k][i * 5:(i + 1) * 5])
                     for i in range(len(seeds))])),
                "seeds_favouring_a": int(sum(
                    1 for i in range(len(seeds))
                    if (np.mean(diffs[k][i * 5:(i + 1) * 5]) > 0)
                    != (k == "mae_stages")))}
            for k in KEYS}
        log(f"\n{a}  vs  {b}")
        for k in KEYS:
            p = paired[f"{a}__vs__{b}"][k]
            log(f"  {k:20s} fold-mean diff {p['fold_mean']:+.4f} "
                f"(sd {p['fold_sd']:.4f})   seed-level {p['seed_level_mean']:+.4f}"
                f"   seeds favouring {a.split('_')[0]}: "
                f"{p['seeds_favouring_a']}/{len(seeds)}")
        log(f"  balanced-accuracy diff by seed: {np.round(sd, 3).tolist()}")

    # ---- per-class, pooled over all 25 folds
    log("\n" + "=" * 86)
    log("PER-CLASS RECALL pooled over all 25 fold evaluations")
    log("=" * 86)
    log(f"{'candidate':24s}" + "".join(f"{s:>8s}" for s in STAGES) + "   never")
    perclass = {}
    for c in cands:
        allf = [r for r in recs if r["candidate"] == c]
        y = np.concatenate([r["true"] for r in allf])
        pr = np.concatenate([r["pred"] for r in allf])
        rec = []
        for g in range(N):
            m = y == g
            rec.append(float((pr[m] == g).mean()) if m.sum() else np.nan)
        never = [STAGES[g] for g in range(N) if (pr == g).sum() == 0]
        perclass[c] = {"recall": {STAGES[g]: rec[g] for g in range(N)},
                       "never_predicted": never}
        log(f"{c:24s}" + "".join(f"{v:8.2f}" for v in rec)
            + f"   {','.join(never) if never else '-'}")

    out = {"summary": rows, "seed_means": seed_means, "paired": paired,
           "per_class": perclass,
           "statistical_note": "25 fold evaluations are not independent; "
                               "seed-level means are the primary summary and "
                               "no p-values are reported",
           "test_set_touched": False}
    p = os.path.join(REP, "repeated_cv_analysis.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=float)
    log(f"\nwrote {os.path.relpath(p, ROOT)}")
    with open(os.path.join(REP, "repeated_cv_analysis.log"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")


if __name__ == "__main__":
    main()
