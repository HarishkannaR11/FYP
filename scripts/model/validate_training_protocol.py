"""
Audit of the training protocol.

Checks the properties that make a small-sample result trustworthy, by
recomputing them rather than trusting the training script's own claims.
"""
import csv
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "derivatives", "model_ready")
SPLITS = os.path.join(SRC, "splits")
TRAIN = os.path.join(ROOT, "derivatives", "model_training")
STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
N_STAGES = 6

results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok), detail))
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  --  {detail}" if detail else ""),
          flush=True)


# ------------------------------------------------------------ split integrity
rows = list(csv.DictReader(open(os.path.join(SPLITS, "split_assignment.csv"))))
smeta = json.load(open(os.path.join(SPLITS, "split_metadata.json")))
y = np.load(os.path.join(SRC, "y.npy"))
subjects = np.load(os.path.join(SRC, "subjects.npy")).astype(str)

check("split covers every subject exactly once", len(rows) == len(subjects)
      and len({r["subject_id"] for r in rows}) == len(subjects), f"{len(rows)} rows")
check("split row order matches the cohort arrays",
      [r["subject_id"] for r in rows] == list(subjects))

pool = [r for r in rows if r["partition"] == "train_pool"]
test = [r for r in rows if r["partition"] == "test"]
check("partitions are disjoint and exhaustive",
      len(pool) + len(test) == len(rows)
      and not ({r["subject_id"] for r in pool} & {r["subject_id"] for r in test}),
      f"pool {len(pool)}, test {len(test)}")
check("test fraction is approximately 30%",
      0.27 <= len(test) / len(rows) <= 0.33, f"{len(test) / len(rows):.3f}")
check("a subject never appears in both partitions",
      len({r["subject_id"] for r in pool} & {r["subject_id"] for r in test}) == 0)

folds = sorted({int(r["cv_fold"]) for r in pool})
check(f"training pool is divided into {smeta['n_folds']} folds",
      folds == list(range(smeta["n_folds"])), str(folds))
check("every test subject has no fold assigned",
      all(r["cv_fold"] == "" for r in test))
sizes = [sum(1 for r in pool if int(r["cv_fold"]) == f) for f in folds]
check("folds are of comparable size", max(sizes) - min(sizes) <= 2, str(sizes))
for f in folds:
    va = {r["subject_id"] for r in pool if int(r["cv_fold"]) == f}
    tr = {r["subject_id"] for r in pool if int(r["cv_fold"]) != f}
    if va & tr:
        check(f"fold {f}: train and validation are disjoint", False)
        break
else:
    check("every fold has disjoint train and validation subjects", True)

per = {s: sum(1 for r in pool if r["stage"] == s) for s in STAGES}
pet = {s: sum(1 for r in test if r["stage"] == s) for s in STAGES}
check("stratification keeps every stage in both partitions",
      all(per[s] > 0 and pet[s] > 0 for s in STAGES), f"pool {per}, test {pet}")
check("every fold contains every stage",
      all(sum(1 for r in pool if int(r["cv_fold"]) == f and r["stage"] == s) > 0
          for f in folds for s in STAGES))

# ------------------------------------------------------------ scaler leakage
reg = np.load(os.path.join(SRC, "X_regional.npy")).astype(np.float32)
pool_idx = np.array([i for i, r in enumerate(rows) if r["partition"] == "train_pool"])
test_idx = np.array([i for i, r in enumerate(rows) if r["partition"] == "test"])
mu_pool = reg[pool_idx].mean(axis=0)
mu_all = reg.mean(axis=0)
check("pool-fitted scaler statistics differ from whole-cohort statistics",
      not np.allclose(mu_pool, mu_all, atol=1e-7),
      f"max |difference| = {np.abs(mu_pool - mu_all).max():.3e}")

sys.path.insert(0, os.path.join(ROOT, "scripts", "model"))
from train_transformer import Scaler  # noqa: E402
sc = Scaler().fit(reg, pool_idx)
check("Scaler.fit uses only the indices it is given",
      np.allclose(sc.mean[0], mu_pool, atol=1e-6)
      and not np.allclose(sc.mean[0], mu_all, atol=1e-7))
zt = sc.transform(reg[test_idx])
check("test features are transformed, not re-fitted",
      not np.allclose(zt.mean(axis=0), 0, atol=1e-3),
      "a re-fitted scaler would force the test mean to 0")

# ----------------------------------------------------------- mask provenance
mask = np.load(os.path.join(SRC, "local_attention_mask.npy"))
mmeta = json.load(open(os.path.join(SRC, "local_attention_mask.json")))
check("local attention mask is 246 x 246 boolean",
      mask.shape == (246, 246) and mask.dtype == bool, str(mask.shape))
check("local attention mask is symmetric", bool((mask == mask.T).all()))
check("every ROI attends to itself", bool(mask.diagonal().all()))
check("no ROI is left without a neighbour",
      int(((mask.sum(1) - 1) == 0).sum()) == 0)
check("mask is derived from the atlas, not invented",
      "BN_Atlas" in mmeta.get("source_atlas", ""), mmeta.get("source_atlas", ""))
check("mask encodes spatial proximity only",
      "spatial proximity only" in mmeta.get("encodes", ""))

# ------------------------------------------------------------ results, if any
# scan whatever completed runs exist rather than hardcoding tag names
RUN_TAGS = sorted(
    f[len("results_"):-len(".json")] for f in os.listdir(TRAIN)
    if f.startswith("results_") and f.endswith(".json")
    and f[len("results_"):-len(".json")] not in ("baselines", "mlp_control"))
check("at least one completed transformer run exists", bool(RUN_TAGS),
      ", ".join(RUN_TAGS) if RUN_TAGS else "none")
# runs produced before the ordinal decoder was corrected are not valid evidence
PRE_FIX = {"ordinal", "ordinal_lr1e-3", "ordinal_lr3e-3"}
for tag in RUN_TAGS:
    p = os.path.join(TRAIN, f"results_{tag}.json")
    if tag in PRE_FIX:
        check(f"{tag}: excluded as a pre-fix ordinal run", True,
              "produced with the argmax decoder; four of six stages were "
              "unreachable")
        continue
    r = json.load(open(p))
    check(f"{tag}: result records the split fingerprint",
          r.get("split_fingerprint") == smeta["split_fingerprint"],
          r.get("split_fingerprint", ""))
    check(f"{tag}: cross-validation ran on every fold",
          len(r["cv"]["folds"]) == smeta["n_folds"])
    metric_entries = {k: v for k, v in r["cv"]["summary"].items()
                      if isinstance(v, dict) and "mean" in v}
    check(f"{tag}: every reported metric is finite",
          bool(metric_entries) and all(
              np.isfinite(v["mean"]) and np.isfinite(v["sd"])
              for v in metric_entries.values()),
          f"{len(metric_entries)} metrics checked")
    npc = r["cv"]["summary"].get("n_predicted_classes_per_fold")
    if npc is not None:
        check(f"{tag}: class coverage recorded per fold", len(npc) == 5,
              f"distinct classes per fold {npc}"
              + ("  <-- DEGENERATE" if max(npc) <= 2 else ""))
    if r.get("test"):
        t = r["test"]["deterministic"]
        check(f"{tag}: test metrics are on the held-out {len(test)} subjects",
              t["n"] == len(test), f"n = {t['n']}")
        check(f"{tag}: accuracy is a valid proportion", 0.0 <= t["accuracy"] <= 1.0)
        check(f"{tag}: MC-dropout entropy does not exceed the maximum",
              r["test"]["mc_dropout"]["mean_predictive_entropy"]
              <= np.log(N_STAGES) + 1e-9)
        pr = os.path.join(TRAIN, f"test_predictions_{tag}.csv")
        if os.path.exists(pr):
            prows = list(csv.DictReader(open(pr)))
            check(f"{tag}: one prediction per held-out subject",
                  len(prows) == len(test), f"{len(prows)} rows")
            tsubs = {r_["subject_id"] for r_ in test}
            check(f"{tag}: predictions cover exactly the test subjects",
                  {r_["subject_id"] for r_ in prows} == tsubs)
            check(f"{tag}: no training subject appears in the predictions",
                  not ({r_["subject_id"] for r_ in prows}
                       & {r_["subject_id"] for r_ in pool}))
            probs = np.array([[float(r_[f"p_{s}"]) for s in STAGES]
                              for r_ in prows])
            check(f"{tag}: predicted probabilities sum to 1",
                  np.allclose(probs.sum(1), 1.0, atol=1e-3),
                  f"max deviation {np.abs(probs.sum(1) - 1).max():.2e}")

# -------------------------------------------------------- baselines present
p = os.path.join(TRAIN, "results_baselines.json")
if os.path.exists(p):
    b = json.load(open(p))
    check("baselines used the same split", b["split_fingerprint"]
          == smeta["split_fingerprint"])
    check("majority and prior baselines are present",
          "majority" in b["baselines"] and "stratified_prior" in b["baselines"])
    check("at least one learned baseline is present",
          any(k.startswith("logreg") or k.startswith("ordinal")
              for k in b["baselines"]))
else:
    check("baseline results present", False, "baselines.py has not been run")

# ------------------------------------------------------------ no overclaiming
weights = [os.path.join(d, f) for d, _, fs in os.walk(ROOT) for f in fs
           if f.endswith((".pt", ".pth", ".ckpt", ".safetensors")) and ".git" not in d]
check("no model weights are committed to the repository", not weights,
      "; ".join(weights[:3]) if weights else "none")

n_fail = sum(1 for _, ok, _ in results if not ok)
print("\n" + "=" * 70)
print(f"{len(results) - n_fail}/{len(results)} checks passed")
print("=" * 70)
if n_fail:
    for name, ok, detail in results:
        if not ok:
            print(f"  FAILED: {name}  {detail}")
    sys.exit(1)
print("All protocol checks passed.")
