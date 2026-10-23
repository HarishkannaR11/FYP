"""
Analysis of the pre-declared three-group task. Read-only on stored predictions; writes
docs/ProjectReport/results/coarse_tasks_results.json and prints the decision rules of the pre-declaration.
Works with whatever is present (site-only control and post-hoc reference now; M1/C2/C3 after run_tabpfn3.py).

Run:  python scripts/experiments/coarse_tasks_v1/analyze_coarse.py
"""
import json
import os

import numpy as np

import common3 as K

y6, y3, subjects, site = K.load_cohort()


def load_json(name):
    p = os.path.join(K.OUT, name)
    return json.load(open(p)) if os.path.exists(p) else None


def to_rec(records):
    return {(r["seed"], r["fold"]): (np.array(r["test_index"]), np.array(r["pred"])) for r in records}


def complete(rec):
    return [s for s in K.SEEDS if all((s, f) in rec for f in range(5))]


def summarise(name, rec):
    seeds = complete(rec)
    if not seeds:
        return None, None
    A = np.vstack([K.pooled(rec, s) for s in seeds])
    cm = np.zeros((K.NG, K.NG), int)
    for a in A:
        for t, p in zip(y3, a):
            cm[t, p] += 1
    bals = [K.bal(y3, a) for a in A]
    lo, hi = K.boot_ci(A, y3)
    out = {"model": name, "seeds_used": len(seeds), "balanced_accuracy": float(np.mean(bals)), "ci95": [lo, hi],
           "seed_sd": float(np.std(bals)), "accuracy": float(np.mean([np.mean(a == y3) for a in A])),
           "macro_f1": float(np.mean([K.macro_f1(y3, a) for a in A])),
           "recall": {K.GROUP_NAMES[g]: float(cm[g, g] / cm[g].sum()) for g in range(K.NG)}, "confusion_matrix": cm.tolist()}
    return out, A


results, mats = {}, {}
site_only = load_json("site_only.json")
if site_only:
    r, A = summarise("C1 site-only rule", to_rec(site_only["records"])); results["C1"] = r; mats["C1"] = A
    r, A = summarise("R post-hoc: collapsed six-class TabPFN (reference only)", to_rec(site_only["reference_R_posthoc_collapsed_six_class_tabpfn"]))
    results["R"] = r; mats["R"] = A
for key, fname in (("M1", "tabpfn3_M1.json"), ("C2", "tabpfn3_C2_GLOBAL.json"), ("C3", "tabpfn3_C3_WITHIN_SITE.json")):
    d = load_json(fname)
    if d:
        r, A = summarise({"M1": "M1 TabPFN v2, three-group labels", "C2": "C2 global label shuffle", "C3": "C3 within-site label shuffle"}[key], to_rec(d["records"]))
        results[key] = r; mats[key] = A

paired = {}
if "M1" in mats:
    for other, label in (("C1", "M1 minus site-only"), ("C3", "M1 minus within-site shuffle"), ("C2", "M1 minus global shuffle")):
        if other in mats and mats[other].shape == mats["M1"].shape:
            paired[label] = K.paired(mats["M1"], mats[other], y3)
    m1 = results["M1"]
    rules = {"1_beats_site_only": bool(paired.get("M1 minus site-only") and paired["M1 minus site-only"]["lo"] > 0
                                      and paired["M1 minus site-only"]["seeds_first_better"] >= 4),
             "2_beats_within_site_shuffle": bool(paired.get("M1 minus within-site shuffle") and paired["M1 minus within-site shuffle"]["lo"] > 0
                                                 and paired["M1 minus within-site shuffle"]["seeds_first_better"] >= 4),
             "3_interval_excludes_chance_33.3": bool(m1["ci95"][0] > 1 / 3)}
    rules["imaging_information_beyond_site_shown"] = all(rules.values())
else:
    rules = {"note": "M1 not yet run (needs TabPFN v2 weights); decision rules cannot be evaluated"}

out = {"declared": "docs/NEW_EXPERIMENT_PREDECLARATION.md", "chance_balanced_accuracy": 1 / 3, "results": results,
       "paired": paired, "decision_rules": rules}
dest = os.path.join(K.ROOT, "docs", "ProjectReport", "results", "coarse_tasks_results.json")
json.dump(out, open(dest, "w"), indent=1)
for k, v in results.items():
    print(f"{k:3s} {v['model'][:58]:58s} BA {v['balanced_accuracy']*100:5.1f} [{v['ci95'][0]*100:4.1f}, {v['ci95'][1]*100:4.1f}]  "
          f"acc {v['accuracy']*100:5.1f}  F1 {v['macro_f1']*100:5.1f}  recall " + " / ".join(f"{x*100:.0f}" for x in v["recall"].values()))
for k, v in paired.items():
    print(f"{k}: {v['diff']*100:+.1f} [{v['lo']*100:+.1f}, {v['hi']*100:+.1f}]  seeds first better {v['seeds_first_better']}/{v['n_seeds']}")
print("decision rules:", rules)
