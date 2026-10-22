"""
Was the ordinal transformer's extreme-only (argmax) / middle-only (median) behaviour a code bug or a property of a
cumulative-link model with an uninformative severity score? Reads saved records only; writes
derivatives/experiments/ordinal_transformer_v1/analysis/collapse_diagnosis.json.

Run:  python scripts/experiments/ordinal_transformer_v1/diagnose_collapse.py
"""
import glob
import json
import os

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
EXP = os.path.join(ROOT, "derivatives", "experiments", "ordinal_transformer_v1")


def main():
    recs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(EXP, "grid", "*.json")))]
    th = np.array([m["final_thresholds"] for r in recs for m in r["inner_models"]])
    init = np.log((np.arange(1, 6) / 6) / (1 - np.arange(1, 6) / 6))
    P = np.vstack([np.array(r["probs"]["FIXED"]) for r in recs])
    y = np.concatenate([r["true"] for r in recs])
    z = np.log(P[:, 0] / (1 - P[:, 0]))                       # logit P(y <= CN) = tau_0 - g, per subject
    counts, stages = np.array([20, 20, 20, 16, 20, 20]), np.arange(6)
    out = {"initial_thresholds": init.round(3).tolist(),
           "final_thresholds_mean": th.mean(0).round(3).tolist(), "final_thresholds_sd": th.std(0).round(3).tolist(),
           "bagged_probability_mean_per_stage": P.mean(0).round(4).tolist(),
           "bagged_max_probability_mean": float(P.max(1).mean().round(4)),
           "probability_range_per_stage": [[float(P[:, k].min().round(3)), float(P[:, k].max().round(3))] for k in range(6)],
           "severity_logit_sd": float(z.std().round(4)), "corr_severity_logit_with_true_stage": float(np.corrcoef(z, y)[0, 1].round(4)),
           "constant_rule_reference": {f"always_stage_{c}": {"mae": float((counts * np.abs(stages - c)).sum() / 116),
                                                             "within_one": float(counts[np.abs(stages - c) <= 1].sum() / 116)}
                                       for c in (2, 3)}}
    os.makedirs(os.path.join(EXP, "analysis"), exist_ok=True)
    json.dump(out, open(os.path.join(EXP, "analysis", "collapse_diagnosis.json"), "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
