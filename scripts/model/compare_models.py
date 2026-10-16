"""
Assemble the cross-validation comparison table from saved artifacts.

Reads every results_*.json produced by the transformer runs, the MLP control
and the baselines, checks they share one split fingerprint, and prints a
single table. Nothing is recomputed from scratch and nothing is taken from a
log; the numbers come from the stored per-fold metrics.
"""
import csv
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAIN = os.path.join(ROOT, "derivatives", "model_training")
SPLITS = os.path.join(ROOT, "derivatives", "model_ready", "splits")
STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
KEYS = ("balanced_accuracy", "macro_f1", "mae_stages", "qwk", "ece")

# runs produced before the ordinal decoder was corrected
INVALID_ORDINAL = {"ordinal", "ordinal_lr1e-3", "ordinal_lr3e-3"}


def collect():
    rows, fps, excluded = {}, set(), {}
    smeta = json.load(open(os.path.join(SPLITS, "split_metadata.json")))

    for fn in sorted(os.listdir(TRAIN)):
        if not (fn.startswith("results_") and fn.endswith(".json")):
            continue
        tag = fn[len("results_"):-len(".json")]
        r = json.load(open(os.path.join(TRAIN, fn)))

        if tag == "baselines":
            for name, v in r["baselines"].items():
                s = v["cv"]["summary"]
                rows[f"baseline: {name}"] = {
                    "kind": "baseline",
                    **{k: (s[k]["mean"], s[k]["sd"]) for k in KEYS if k in s},
                    "folds": len(v["cv"]["folds"]),
                    "classes": None, "params": None, "seconds": None,
                }
            fps.add(r.get("split_fingerprint"))
            continue

        if tag == "mlp_control":
            for name, v in r["results"].items():
                s = v["summary"]
                rows[f"control: {name}"] = {
                    "kind": "mlp",
                    **{k: (s[k]["mean"], s[k]["sd"]) for k in KEYS},
                    "folds": len(v["folds"]),
                    "classes": v["n_predicted_classes_per_fold"],
                    "params": v["n_parameters"], "seconds": None,
                }
            fps.add(r.get("split_fingerprint"))
            continue

        if tag in INVALID_ORDINAL:
            excluded[tag] = ("produced with the argmax decoder, under which "
                             "four of six stages were unreachable")
            continue

        cv = r.get("cv", {})
        s = cv.get("summary", {})
        if not s or len(cv.get("folds", [])) != smeta["n_folds"]:
            excluded[tag] = f"incomplete ({len(cv.get('folds', []))} folds)"
            continue
        cfg = r.get("config", {})
        rows[f"transformer: {tag}"] = {
            "kind": "transformer",
            **{k: (s[k]["mean"], s[k]["sd"]) for k in KEYS if k in s},
            "folds": len(cv["folds"]),
            "classes": s.get("n_predicted_classes_per_fold"),
            "params": None,
            "seconds": None,
            "config": {k: cfg.get(k) for k in
                       ("head", "mask", "d_model", "num_layers", "dropout", "lr")},
            "test_evaluated": r.get("test") is not None,
        }
        fps.add(r.get("split_fingerprint"))
    return rows, fps, excluded, smeta


def main():
    rows, fps, excluded, smeta = collect()
    fps.discard(None)
    print("=" * 100)
    print("CROSS-VALIDATION COMPARISON  (training pool only; held-out test not used)")
    print("=" * 100)
    print(f"split fingerprint(s): {sorted(fps)}")
    if len(fps) > 1:
        print("  WARNING: results come from different splits and are NOT comparable")
    else:
        print(f"  all results share one split: {smeta['n_train_pool']} pool / "
              f"{smeta['n_test']} test, {smeta['n_folds']} folds, stratified by stage")
    print()

    hdr = (f"{'model':40s} {'bal.acc':>15s} {'macro-F1':>15s} "
           f"{'MAE':>14s} {'QWK':>15s} {'cls':>5s}")
    print(hdr)
    print("-" * len(hdr))

    def fmt(v):
        return f"{v[0]:.3f}+/-{v[1]:.3f}" if v else " " * 13

    order = sorted(rows, key=lambda k: -(rows[k].get("balanced_accuracy") or (0, 0))[0])
    for name in order:
        r = rows[name]
        cls = ("-" if r["classes"] is None
               else f"{min(r['classes'])}-{max(r['classes'])}")
        print(f"{name:40s} {fmt(r.get('balanced_accuracy')):>15s} "
              f"{fmt(r.get('macro_f1')):>15s} {fmt(r.get('mae_stages')):>14s} "
              f"{fmt(r.get('qwk')):>15s} {cls:>5s}")

    print()
    print("cls = range of distinct stages predicted across folds (6 = all stages)")
    print(f"chance balanced accuracy = 1/6 = {1 / 6:.3f}")

    if excluded:
        print("\nEXCLUDED from the table:")
        for t, why in excluded.items():
            print(f"  {t}: {why}")

    tested = [n for n, r in rows.items() if r.get("test_evaluated")]
    print(f"\nruns that have touched the held-out test set: {tested or 'none'}")

    out = os.path.join(TRAIN, "cv_comparison.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "kind"] + [f"{k}_{s}" for k in KEYS
                                        for s in ("mean", "sd")]
                   + ["folds", "classes_min", "classes_max", "config"])
        for name in order:
            r = rows[name]
            vals = []
            for k in KEYS:
                v = r.get(k)
                vals += [f"{v[0]:.6f}", f"{v[1]:.6f}"] if v else ["", ""]
            w.writerow([name, r["kind"]] + vals + [
                r["folds"],
                min(r["classes"]) if r["classes"] else "",
                max(r["classes"]) if r["classes"] else "",
                json.dumps(r.get("config", {}))])
    print(f"\nwrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
