"""
Controlled TabPFN v2 study on the fixed 116-subject cohort (experiment v3).

The plan (configurations, decision rules, stopping criteria) is pre-registered in
derivatives/experiments/biomarker_ablation_v3/PLAN.md, written before any run.

Feature-selection settings (SVM-RFE with the study's settings, fitted inside each training partition), all
nested-eligible (outer folds + 3-fold inner CV):
    A_all4          all four biomarkers, 984 candidates, k = 50  = the existing baseline (must reproduce it)
    A_all4_k25      same, k = 25
    A_all4_k100     same, k = 100
    A_mALFF, A_mReHo, A_DCz, A_FC   one biomarker (246 candidates), k = 50
    A_local         mALFF + mReHo (local amplitude + local synchrony), 492 candidates, k = 50
    A_conn          DC_z + FC strength (whole-brain connectivity), 492 candidates, k = 50
Confound diagnostic (outer folds only): the baseline pipeline refitted with the outer-training labels permuted,
test labels untouched.
    P_site1..5      shuffled WITHIN each scanner site (keeps the site-stage association, breaks any
                    imaging-stage association beyond site)
    P_glob1..3      shuffled across all training subjects (the ordinary chance null)
Learning curve (outer folds only): the baseline pipeline on a class-stratified random subset of each outer
training set, outer test fold unchanged.
    L_frac50, L_frac75

Per outer split (identical to every earlier comparison: 5 folds x 5 seeds on all 116):
  outer  selection + TabPFN fitted on the outer training subjects; outer-test probabilities saved
  inner  A configurations only: 3 stratified inner folds of the outer training subjects; selection + TabPFN
         refitted on each inner-training part and scored on its inner-validation part. The pooled inner
         out-of-fold balanced accuracy is the configuration's inner score, the only quantity nested selection uses.

Label access (as in transformer_nested_v2): fitting code receives y with the outer-test labels set to -1, and inner
fits additionally hide the inner-validation labels; outer-test labels are read only in score_record(), after the
record's predictions are final.

Class balancing: TabPFN's balance_probabilities divides each probability by the class's share of the training
labels and renormalises. It is applied post hoc from the saved probabilities and the saved training class counts
(checked identical to the flag: max difference 6e-8, same predictions).

TabPFN settings are the existing baseline's: ModelVersion.V2 defaults, CPU, random_state = seed + fold.

Outputs: derivatives/experiments/biomarker_ablation_v3/grid/<config>/cv5_seed{s}_fold{f}.json, written once
(resumable); a lock file prevents duplicate runs; the manifest pins this script's hash. After A_all4 completes,
the run stops unless it reproduces the existing baseline exactly in 25/25 splits.

Run:  python scripts/experiments/biomarker_ablation_v3/run_tabpfn_configs.py --configs A_all4 A_all4_k25 ...
"""
import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "model"))
import train_transformer_svmrfe as S  # noqa: E402  (sets thread env vars first)

import numpy as np  # noqa: E402
import sklearn  # noqa: E402
import tabpfn  # noqa: E402
import torch  # noqa: E402
from sklearn.feature_selection import RFE  # noqa: E402
from sklearn.svm import LinearSVC  # noqa: E402
from tabpfn import TabPFNClassifier  # noqa: E402
from tabpfn.constants import ModelVersion  # noqa: E402

H, M = S.H, S.M
ROOT = S.ROOT
EXP = os.path.join(ROOT, "derivatives", "experiments", "biomarker_ablation_v3")
TABPFN_BASELINE = os.path.join(ROOT, "derivatives", "model_training", "tabpfn", "cv116_tabpfn_svmrfe50.json")
NS = S.N_STAGES
HIDDEN = -1
N_INNER = 3
CHANNEL = {"mALFF": 0, "mReHo": 1, "DC_z": 2, "FC": 3}
POOLS = {"mALFF": ["mALFF"], "mReHo": ["mReHo"], "DCz": ["DC_z"], "FC": ["FC"],
         "all4": ["mALFF", "mReHo", "DC_z", "FC"], "local": ["mALFF", "mReHo"], "conn": ["DC_z", "FC"]}
CONFIGS = {"A_all4": {"pool": "all4", "k": 50, "inner": True},
           "A_all4_k25": {"pool": "all4", "k": 25, "inner": True},
           "A_all4_k100": {"pool": "all4", "k": 100, "inner": True},
           **{f"A_{p}": {"pool": p, "k": 50, "inner": True} for p in ("mALFF", "mReHo", "DCz", "FC", "local", "conn")},
           **{f"P_site{r}": {"pool": "all4", "k": 50, "inner": False, "permute": "site", "rep": r} for r in range(1, 6)},
           **{f"P_glob{r}": {"pool": "all4", "k": 50, "inner": False, "permute": "glob", "rep": r} for r in range(1, 4)},
           **{f"L_frac{int(100 * q)}": {"pool": "all4", "k": 50, "inner": False, "frac": q} for q in (0.5, 0.75)}}
BASELINE = "A_all4"
NESTED_ORDER = ["A_all4", "A_all4_k25", "A_all4_k100", "A_mALFF", "A_mReHo", "A_DCz", "A_FC", "A_local", "A_conn"]


def script_hash():
    return hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()[:16]


def hide(y, idx):
    out = y.copy()
    out[idx] = HIDDEN
    return out


def pool_columns(pool):
    """Columns of flat all4 (738 regional, ROI-major with channels mALFF, mReHo, DC_z; then 246 FC) in a pool."""
    cols = [np.arange(CHANNEL[b], 738, 3) if CHANNEL[b] < 3 else np.arange(738, 984) for b in POOLS[pool]]
    return np.sort(np.concatenate(cols))


def rfe_pipeline(seed, k):
    """S.svmrfe_pipeline with a chosen k (identical to it at k = 50)."""
    return M.pipe(selector=RFE(LinearSVC(C=0.01, max_iter=5000, random_state=seed), n_features_to_select=k, step=0.1))


def select(flat, y_fit, rows, cols, seed_tag, k=50):
    """SVM-RFE (LinearSVC C=0.01, step 0.1, scalers inside the pipeline) fitted on `rows` only."""
    assert (y_fit[rows] >= 0).all()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pipe = rfe_pipeline(seed_tag, k).fit(flat[rows][:, cols], y_fit[rows])
    sup = cols[np.flatnonzero(pipe.named_steps["select"].get_support())]
    assert sup.size == k
    return sup


def tabpfn_probs(X, y_fit, rows, pred_rows, rs):
    """TabPFN v2 (baseline settings) with `rows` as its training context; probabilities for pred_rows."""
    assert (y_fit[rows] >= 0).all()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = TabPFNClassifier.create_default_for_version(ModelVersion.V2, device="cpu", random_state=rs)
        m.fit(X[rows], y_fit[rows])
        p = m.predict_proba(X[pred_rows])
        classes = list(m.classes_)
    if classes != list(range(NS)):
        full = np.zeros((len(pred_rows), NS))
        full[:, classes] = p
        p = full
    del m
    gc.collect()
    return np.asarray(p, dtype=np.float64)


def balance(p, counts):
    """TabPFN's balance_probabilities: p / training class share, renormalised."""
    share = np.asarray(counts, float) / np.sum(counts)
    q = np.asarray(p, float) / np.where(share > 0, share, 1.0)
    return q / q.sum(1, keepdims=True)


def permuted(y_sel, tr, site, kind, seed, fold, rep):
    """Confound null: outer-training labels shuffled within each site ('site') or across all ('glob')."""
    out = y_sel.copy()
    rng = np.random.default_rng([seed, fold, rep, 99])
    groups = [tr] if kind == "glob" else [tr[site[tr] == s] for s in np.unique(site[tr])]
    for g in groups:
        out[g] = rng.permutation(y_sel[g])
    return out


def subsample(y_sel, tr, frac, seed, fold):
    """Class-stratified random subset of the training subjects (learning curve)."""
    rng = np.random.default_rng([seed, fold, int(100 * frac), 77])
    keep = [rng.choice(tr[y_sel[tr] == g], int(round(frac * (y_sel[tr] == g).sum())), replace=False) for g in range(NS)]
    return np.sort(np.concatenate(keep))


def fit_config(d, y_sel, cname, seed, fold, tr, te):
    """Everything that learns or selects for one configuration and outer split. Never sees an outer-test label."""
    assert (y_sel[te] == HIDDEN).all() and (y_sel[tr] >= 0).all()
    cfg = CONFIGS[cname]
    flat = d[1]
    if cfg.get("permute"):
        site = np.array([s.replace("sub-", "")[:3] for s in d[4]])
        y_sel = permuted(y_sel, tr, site, cfg["permute"], seed, fold, cfg["rep"])
    if cfg.get("frac"):
        tr = subsample(y_sel, tr, cfg["frac"], seed, fold)
    cols = pool_columns(cfg["pool"])
    sup = select(flat, y_sel, tr, cols, seed + fold, cfg["k"])
    out = {"n_candidates": int(cols.size), "n_features": int(sup.size), "n_train": int(tr.size),
           "support": sup.tolist(), "train_counts": np.bincount(y_sel[tr], minlength=NS).tolist(),
           "probs": tabpfn_probs(flat[:, sup].astype(np.float32), y_sel, tr, te, seed + fold)}
    if cfg["inner"]:
        fo = H.stratified_folds(y_sel, tr, N_INNER, np.random.default_rng([seed, fold, 33]))
        pos = {int(i): k for k, i in enumerate(tr)}
        oof, oof_bal = np.zeros((tr.size, NS)), np.zeros((tr.size, NS))
        inner = []
        for j in range(N_INNER):
            itr = np.array([i for i in tr if fo[int(i)] != j])
            iva = np.array([i for i in tr if fo[int(i)] == j])
            y_fit = hide(y_sel, iva)
            s_j = select(flat, y_fit, itr, cols, seed + fold, cfg["k"])
            p_j = tabpfn_probs(flat[:, s_j].astype(np.float32), y_fit, itr, iva, seed + fold)
            cnt = np.bincount(y_fit[itr], minlength=NS)
            rows = [pos[int(i)] for i in iva]
            oof[rows], oof_bal[rows] = p_j, balance(p_j, cnt)
            inner.append({"val_idx": iva.tolist(), "support": s_j.tolist(), "train_counts": cnt.tolist()})
        yt = y_sel[tr]
        out["inner"] = {"folds": inner, "train_idx": tr.tolist(), "oof_probs": S.r6(oof),
                        "score": H.evaluate(yt, oof.argmax(1))["balanced_accuracy"],
                        "score_balanced": H.evaluate(yt, oof_bal.argmax(1))["balanced_accuracy"],
                        "macro_f1": H.evaluate(yt, oof.argmax(1))["macro_f1"]}
    return out


def score_record(y, te, out):
    """Reads outer-test labels; called only after fit_config has returned."""
    p = out["probs"]
    pb = balance(p, out["train_counts"])
    res = {}
    for tag, q in (("raw", p), ("balanced", pb)):
        m = H.evaluate(y[te], q.argmax(1))
        res[tag] = {"accuracy": m["accuracy"], "balanced_accuracy": m["balanced_accuracy"], "macro_f1": m["macro_f1"],
                    "per_class_recall": m["per_class_recall"]}
    return res, pb


def run_split(d, y, cname, seed, fold, fp, tr, te):
    out = fit_config(d, hide(y, te), cname, seed, fold, tr, te)
    frozen = hashlib.sha256(np.ascontiguousarray(out["probs"]).tobytes()).hexdigest()
    heldout, pb = score_record(y, te, out)
    assert hashlib.sha256(np.ascontiguousarray(out["probs"]).tobytes()).hexdigest() == frozen
    return {"config": cname, **CONFIGS[cname], "seed": seed, "fold": fold, "fold_fingerprint": fp,
            "script_sha256": script_hash(), "test_idx": te.tolist(), "true": y[te].tolist(),
            "pred": out["probs"].argmax(1).tolist(), "pred_balanced": pb.argmax(1).tolist(),
            "probs": S.r6(out["probs"]), "heldout": heldout, **{k: v for k, v in out.items() if k != "probs"}}


def nested_choice(inner_scores):
    """Pre-registered nested selection for one outer split: the configuration in NESTED_ORDER with the highest
    inner score; ties go to the earliest in NESTED_ORDER (the baseline first). inner_scores: {config: score}."""
    best = max(inner_scores.values())
    return next(c for c in NESTED_ORDER if c in inner_scores and inner_scores[c] == best)


# ------------------------------------------------------------------ driver
def verify_baseline():
    """Stopping rule: A_all4 must reproduce the existing TabPFN baseline exactly, or nothing else runs."""
    old = {(r["seed"], r["fold"]): r for r in json.load(open(TABPFN_BASELINE))["records"]}
    bad = []
    for (s, f), r in old.items():
        new = json.load(open(os.path.join(EXP, "grid", BASELINE, f"cv5_seed{s}_fold{f}.json")))
        if new["test_idx"] != r["test_index"] or new["pred"] != r["pred"] or new["support"] != r["support"] \
                or np.abs(np.array(new["probs"]) - np.array(r["probs"])).max() > 1e-5:
            bad.append((s, f))
    print(f"baseline reproduction: {len(old) - len(bad)}/{len(old)} splits identical", flush=True)
    if bad:
        raise SystemExit(f"STOP: A_all4 does not reproduce the existing baseline in splits {bad}")


def manifest():
    path = os.path.join(EXP, "manifest.json")
    info = {"experiment": "biomarker_ablation_v3", "script": os.path.relpath(os.path.abspath(__file__), ROOT),
            "script_sha256": script_hash(), "configs": CONFIGS, "baseline": BASELINE, "nested_order": NESTED_ORDER,
            "n_inner": N_INNER, "seeds": H.SEEDS, "n_folds": 5, "inputs": "rs-fMRI only; no clinical variables",
            "tabpfn": getattr(tabpfn, "__version__", "?"), "python": platform.python_version(),
            "torch": torch.__version__, "sklearn": sklearn.__version__, "created": time.strftime("%Y-%m-%d %H:%M:%S")}
    if os.path.exists(path):
        if json.load(open(path))["script_sha256"] != info["script_sha256"]:
            raise SystemExit("run_tabpfn_configs.py changed since this experiment started; refusing to mix versions")
        return
    os.makedirs(EXP, exist_ok=True)
    json.dump(info, open(path, "w"), indent=2)


def acquire_lock(name):
    import psutil
    path = os.path.join(EXP, f"{name}.lock")
    if os.path.exists(path):
        pid = int(open(path).read().strip() or 0)
        if pid and psutil.pid_exists(pid):
            raise SystemExit(f"another run is active (pid {pid}); not starting a duplicate")
    os.makedirs(EXP, exist_ok=True)
    open(path, "w").write(str(os.getpid()))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+", required=True, choices=list(CONFIGS))
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    if not os.path.exists(os.path.join(EXP, "PLAN.md")):
        raise SystemExit("PLAN.md (pre-registration) missing; write it before running")
    torch.set_num_threads(args.threads)
    manifest()
    lock = acquire_lock("tabpfn")
    try:
        d = S.load_cohort()
        y = d[3]
        plan = S.cv_plan(y, d[4], 5)
        for cname in args.configs:
            t_cfg = time.time()
            for seed in H.SEEDS:
                fp, folds = plan[seed]
                for f, (tr, te) in enumerate(folds):
                    path = os.path.join(EXP, "grid", cname, f"cv5_seed{seed}_fold{f}.json")
                    if os.path.exists(path):
                        continue
                    t0 = time.time()
                    rec = run_split(d, y, cname, seed, f, fp, tr, te)
                    rec["seconds"] = round(time.time() - t0, 1)
                    os.makedirs(os.path.dirname(path), exist_ok=True)
                    json.dump(rec, open(path + ".tmp", "w"))
                    os.replace(path + ".tmp", path)
                    inner = f"inner {rec['inner']['score']:.3f}" if "inner" in rec else "outer only"
                    print(f"{cname} seed {seed} fold {f}: {rec['n_features']} features, {inner}, {rec['seconds']:.0f}s",
                          flush=True)
            print(f"{cname} done in {(time.time() - t_cfg) / 60:.1f} min", flush=True)
            if cname == BASELINE:
                verify_baseline()
    finally:
        os.remove(lock)


if __name__ == "__main__":
    main()
