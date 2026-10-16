"""
Per-class review (Task 3A) and collapse taxonomy (Task 3D).

Everything is recomputed from the per-fold validation predictions stored in
results_*.json, so every number here is reproducible from saved artifacts
without retraining.

Collapse taxonomy, applied per class per model:

    NEVER PREDICTED   the model never emits this class in any fold
    ZERO RECALL       the class is emitted somewhere, but no true member of
                      it is ever caught
    NEIGHBOUR CONFUSION  recall > 0, and most errors land on an adjacent stage
    SCATTERED         recall > 0, and errors are spread across distant stages

and per model:

    COLLAPSED         predictions concentrate on <= 2 distinct classes
    PARTIAL COLLAPSE  3 to 4 distinct classes
    FULL COVERAGE     5 to 6 distinct classes
"""
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN = os.path.join(ROOT, "derivatives", "model_training")
STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
N = 6
INVALID = {"ordinal", "ordinal_lr1e-3", "ordinal_lr3e-3"}

_log = []


def log(m=""):
    print(m, flush=True)
    _log.append(str(m))


def gather(tag, r):
    """Pool the per-fold validation predictions into one array."""
    yy, pp = [], []
    for m in r["cv"]["folds"]:
        vp = m.get("val_predictions")
        if not vp:
            return None
        yy.extend(vp["true"])
        pp.append(np.array(vp["probs"]))
    return np.array(yy), np.vstack(pp)


def decode_for(head_kind, probs):
    """Apply the decision rule that belongs to the head.

    For a cumulative-link ordinal model the point prediction is the number of
    thresholds the severity score exceeds, i.e. sum_k 1[P(y<=k) < 0.5].
    P(y<=k) is just the cumulative sum of the saved class probabilities, so
    this is recoverable from stored predictions without retraining.
    argmax is NOT valid here -- it can only ever reach the extreme stages."""
    if head_kind and head_kind.startswith("ordinal"):
        return (np.cumsum(probs, axis=1)[:, :N - 1] < 0.5).sum(axis=1)
    return probs.argmax(1)


def analyse(name, y, probs, decoded=None):
    pred = probs.argmax(1) if decoded is None else np.asarray(decoded)
    C = np.zeros((N, N), dtype=int)
    for a, b in zip(y, pred):
        C[a, b] += 1

    log(f"\n{'=' * 74}")
    log(f"{name}")
    log(f"{'=' * 74}")
    log(f"pooled over folds: {len(y)} validation predictions")

    emitted = sorted(set(int(v) for v in pred))
    log(f"classes ever predicted: {[STAGES[i] for i in emitted]} "
        f"({len(emitted)} of {N})")
    never = [STAGES[g] for g in range(N) if g not in emitted]
    if never:
        log(f"NEVER PREDICTED      : {never}")

    log("")
    log(f"{'stage':6s} {'supp':>5s} {'pred':>5s} {'prec':>6s} {'rec':>6s} "
        f"{'F1':>6s}   verdict")
    rows = []
    for g in range(N):
        supp = int((y == g).sum())
        npred = int((pred == g).sum())
        tp = int(C[g, g])
        prec = tp / npred if npred else float("nan")
        rec = tp / supp if supp else float("nan")
        f1 = (2 * prec * rec / (prec + rec)
              if npred and supp and (prec + rec) > 0 else 0.0)
        if g not in emitted:
            verdict = "NEVER PREDICTED"
        elif tp == 0:
            verdict = "ZERO RECALL"
        else:
            err = C[g].copy()
            err[g] = 0
            if err.sum() == 0:
                verdict = "no errors"
            else:
                dist = np.abs(np.arange(N) - g)
                near = err[dist == 1].sum()
                verdict = ("NEIGHBOUR CONFUSION"
                           if near >= err.sum() / 2 else "SCATTERED")
        rows.append({"stage": STAGES[g], "support": supp, "n_predicted": npred,
                     "precision": prec, "recall": rec, "f1": f1,
                     "verdict": verdict})
        log(f"{STAGES[g]:6s} {supp:5d} {npred:5d} {prec:6.3f} {rec:6.3f} "
            f"{f1:6.3f}   {verdict}")

    log("")
    log("confusion (rows = true, cols = predicted)")
    log("        " + "".join(f"{s:>6s}" for s in STAGES))
    for g in range(N):
        log(f"  {STAGES[g]:6s}" + "".join(f"{int(v):6d}" for v in C[g]))

    status = ("COLLAPSED" if len(emitted) <= 2
              else "PARTIAL COLLAPSE" if len(emitted) <= 4 else "FULL COVERAGE")
    mae = float(np.abs(pred - y).mean())
    adj = float((np.abs(pred - y) <= 1).mean())
    log("")
    log(f"status: {status}   MAE {mae:.3f}   within-one-stage {adj:.1%}")
    return {"classes_emitted": [STAGES[i] for i in emitted],
            "never_predicted": never, "per_class": rows,
            "confusion_matrix": C.tolist(), "status": status,
            "mae": mae, "within_one_stage": adj}


def main():
    out = {}
    for fn in sorted(os.listdir(TRAIN)):
        if not (fn.startswith("results_") and fn.endswith(".json")):
            continue
        tag = fn[len("results_"):-len(".json")]
        if tag in INVALID or tag in ("baselines", "mlp_control"):
            continue
        r = json.load(open(os.path.join(TRAIN, fn)))
        if "cv" not in r:
            continue
        g = gather(tag, r)
        if g is None:
            log(f"[skip] {tag}: no saved per-fold predictions "
                f"(predates prediction saving)")
            continue
        y, probs = g
        cfg = r.get("config", {})
        head_kind = cfg.get("head")
        name = (f"{tag}   head={head_kind} mask={cfg.get('mask')} "
                f"d={cfg.get('d_model')} L={cfg.get('num_layers')}")
        out[tag] = analyse(name, y, probs,
                           decoded=decode_for(head_kind, probs))
        out[tag]["decode_rule"] = ("cumulative-threshold"
                                   if head_kind and head_kind.startswith("ordinal")
                                   else "argmax")

    if out:
        p = os.path.join(TRAIN, "per_class_analysis.json")
        with open(p, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2)
        log(f"\nwrote {os.path.relpath(p, ROOT)}")
    else:
        log("\nno runs with saved per-fold predictions yet")
    with open(os.path.join(TRAIN, "per_class_analysis.log"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")


if __name__ == "__main__":
    main()
