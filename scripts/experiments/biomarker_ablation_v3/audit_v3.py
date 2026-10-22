"""
Pre-run audit for biomarker_ablation_v3. Reads project data and earlier results; writes only
derivatives/experiments/biomarker_ablation_v3/audit/ (refuses to overwrite). Changes no subject or label.

  A  cohort: manifest, identities, labels, class counts, one scan per subject, inputs unchanged since v2
  B  splits: fingerprints, identical outer test folds across TabPFN baseline, v1 C0 and v2, coverage
  C  data quality: missing values, constant features, near-duplicate subjects, within-subject biomarker
     construction, how the four biomarkers relate within a subject
  D  fit scope: for every candidate pool, SVM-RFE ignores outer-test rows and labels, and inner fits ignore
     inner-validation rows and labels (perturbation tests on one split)
  E  stability of the existing SVM-RFE selections (TabPFN baseline, v2 inner fits)
  F  confounds: scanner site and acquisition vs stage, per-feature site vs stage association, and a
     site-only lookup rule under the same 25 outer splits (a confound diagnostic, not a candidate model)
  G  metrics: balanced accuracy, macro-F1 and recall from the project code agree with scikit-learn;
     the reported baseline numbers are reproduced from saved predictions; the CI definition

Run:  python scripts/experiments/biomarker_ablation_v3/audit_v3.py
"""
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_tabpfn_configs as T  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import chi2_contingency, kruskal  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score  # noqa: E402

S, H, ROOT = T.S, T.H, T.ROOT
OUT = os.path.join(T.EXP, "audit")
SRC = os.path.join(ROOT, "derivatives", "model_ready")
V2 = os.path.join(ROOT, "derivatives", "experiments", "transformer_nested_v2")
V1 = os.path.join(ROOT, "derivatives", "experiments", "transformer_diagnostics_v1")
TABPFN = os.path.join(ROOT, "derivatives", "model_training", "tabpfn", "cv116_tabpfn_svmrfe50.json")
FPS = ["3edbea19c1ee3c08", "d53ad26a06b7d5c5", "2a8e70dd0eb80cae", "d54bf96735be41dd", "8b7f4cd9e512b26a"]
lines, results, facts = [], [], {}


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def check(group, name, ok, detail="", flag="FLAG"):
    results.append({"group": group, "check": name, "status": "PASS" if ok else flag, "detail": detail})
    log(f"  {'PASS' if ok else flag:4s}  {name}" + (f"  -- {detail}" if detail else ""))


def bh(p):
    p = np.asarray(p)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def jacc(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def main():
    if os.path.exists(os.path.join(OUT, "audit.json")):
        raise SystemExit("audit already written; not overwritten")
    os.makedirs(OUT, exist_ok=True)
    d = S.load_cohort()
    X4, flat, fc, y, subjects, mask = d
    plan = S.cv_plan(y, subjects, 5)

    # ------------------------------------------------------------------ A
    log("[A] cohort")
    idx = pd.read_csv(os.path.join(SRC, "index.csv"))
    stages = np.load(os.path.join(SRC, "stages.npy")).astype(str)
    acq = np.load(os.path.join(SRC, "acq_ids.npy")).astype(str)
    demo = pd.read_csv(os.path.join(ROOT, "derivatives", "comorbidity_features", "demographics_116.csv"))
    counts = {s: int((y == g).sum()) for g, s in enumerate(S.STAGES)}
    check("A", "116 subjects; class counts CN 20, SMC 20, EMCI 20, MCI 16, LMCI 20, AD 20",
          len(y) == 116 and list(counts.values()) == [20, 20, 20, 16, 20, 20], json.dumps(counts))
    check("A", "subject IDs unique; one acquisition per subject",
          len(set(subjects)) == 116 and len(set(acq)) == 116 and len({a.split("|")[0] for a in acq}) == 116,
          f"sessions/runs used: {sorted({'|'.join(a.split('|')[1:]) for a in acq})}")
    same = (idx.subject.tolist() == list(subjects) and np.array_equal(idx.y.to_numpy(), y)
            and list(stages) == [S.STAGES[v] for v in y] and idx.stage.tolist() == list(stages)
            and demo.bids_id.tolist() == list(subjects) and demo.cohort_stage.tolist() == list(stages))
    check("A", "index.csv, subjects.npy, y.npy, stages.npy and demographics agree row by row", same)
    v2h = json.load(open(os.path.join(V2, "audit", "audit.json")))["input_hashes"]
    now = {f: hashlib.sha256(open(os.path.join(SRC, f), "rb").read()).hexdigest()[:16] for f in v2h}
    check("A", "model-ready inputs byte-identical to those used by the v2 run", now == v2h,
          ", ".join(f"{k} {v}" for k, v in now.items()))

    # ------------------------------------------------------------------ B
    log("\n[B] outer splits")
    fps = [plan[s][0] for s in H.SEEDS]
    check("B", "5 x 5 CV fold fingerprints match the project's", fps == FPS, ", ".join(fps))
    tab = {(r["seed"], r["fold"]): sorted(r["test_index"]) for r in json.load(open(TABPFN))["records"]}
    v2 = {(r["seed"], r["fold"]): sorted(r["test_idx"]) for r in
          (json.load(open(f)) for f in glob.glob(os.path.join(V2, "grid", "*.json")))}
    v1 = {(r["seed"], r["fold"]): sorted(r["test_idx"]) for r in
          (json.load(open(f)) for f in glob.glob(os.path.join(V1, "grid", "C0", "*.json")))}
    ok = all(sorted(te.tolist()) == tab[(s, f)] == v2[(s, f)] == v1[(s, f)]
             for s in H.SEEDS for f, (_, te) in enumerate(plan[s][1]))
    check("B", "outer test folds identical: this plan, TabPFN baseline, transformer C0 (v1), transformer v2", ok,
          "25/25 splits")
    cover = all(np.array_equal(np.sort(np.concatenate([te for _, te in plan[s][1]])), np.arange(116)) for s in H.SEEDS)
    mins = min(int((y[te] == g).sum()) for s in H.SEEDS for _, te in plan[s][1] for g in range(T.NS))
    check("B", "every subject tested exactly once per seed; every outer test fold holds every stage", cover and mins >= 3,
          f"smallest stage count in an outer test fold: {mins}")
    part = pd.read_csv(os.path.join(SRC, "splits", "split_assignment.csv"))
    locked = int((part.partition == "test").sum())
    refs = [os.path.basename(p) for p in glob.glob(os.path.join(HERE, "*.py"))
            if "run_test" in open(p, encoding="utf-8").read() and os.path.basename(p) != os.path.basename(__file__)]
    check("B", "no v3 script calls the gated locked-test evaluation", not refs, ", ".join(refs))
    log(f"  INFO  the {locked} subjects of the 70/30 'locked' partition are inside every CV split and have been "
        "evaluated before (transformer_nested_v2 report, section 2); they are not an untouched hold-out")

    # ------------------------------------------------------------------ C
    log("\n[C] data quality and biomarker construction")
    check("C", "no missing or non-finite values (regional, FC strength, FC matrices)",
          bool(np.isfinite(flat).all() and np.isfinite(fc).all()))
    sd = flat.std(0)
    check("C", "no constant feature across subjects", bool((sd > 1e-8).all()), f"min feature sd {sd.min():.3g}")
    Z = (flat - flat.mean(0)) / sd
    R = np.corrcoef(Z)
    np.fill_diagonal(R, -1)
    i, j = np.unravel_index(R.argmax(), R.shape)
    check("C", "no near-duplicate subjects (max between-subject correlation of the 984 standardised features < 0.9)",
          R.max() < 0.9, f"max r {R.max():.3f} ({subjects[i]} vs {subjects[j]})")
    mu = [X4[:, :, c].mean(1) for c in range(3)]
    ok = (np.allclose(mu[0], 1, atol=1e-4) and np.allclose(mu[1], 1, atol=1e-4) and np.allclose(mu[2], 0, atol=1e-4)
          and np.allclose(X4[:, :, 2].std(1), 1, atol=1e-4))
    off = ~np.eye(246, dtype=bool)
    st = np.abs(fc.astype(np.float64))[:, off].reshape(116, 246, 245).mean(2)
    check("C", "biomarkers built within each subject (mALFF, mReHo ROI mean 1; DC_z mean 0 sd 1; FC strength = mean |FC|)",
          ok and np.abs(st - X4[:, :, 3]).max() < 1e-5)
    names = ["mALFF", "mReHo", "DC_z", "FC"]
    rel = {}
    for a in range(4):
        for b in range(a + 1, 4):
            r = np.array([np.corrcoef(X4[k, :, a], X4[k, :, b])[0, 1] for k in range(116)])
            rel[f"{names[a]}~{names[b]}"] = round(float(np.median(r)), 3)
    facts["within_subject_biomarker_r_median"] = rel
    log(f"  INFO  median within-subject correlation across ROIs between biomarkers: {rel}")

    # ------------------------------------------------------------------ D
    log("\n[D] where learned steps are fitted (perturbation tests, seed 20261015 fold 0)")
    rng = np.random.default_rng(0)
    tr, te = plan[20261015][1][0]
    y_sel = T.hide(y, te)
    fo = H.stratified_folds(y_sel, tr, T.N_INNER, np.random.default_rng([20261015, 0, 33]))
    itr = np.array([i for i in tr if fo[int(i)] != 0])
    iva = np.array([i for i in tr if fo[int(i)] == 0])
    flat_te = flat.copy()                                   # outer-test rows perturbed
    flat_te[te] += rng.normal(0, 3, (te.size, flat.shape[1]))
    flat_p = flat_te.copy()                                 # outer-test and inner-validation rows perturbed
    flat_p[iva] += rng.normal(0, 3, (iva.size, flat.shape[1]))
    y_p = y_sel.copy()
    y_p[iva] = (y_p[iva] + 1 + rng.integers(0, T.NS - 1, iva.size)) % T.NS
    ok_outer, ok_inner = True, True
    for pool in T.POOLS:
        cols = T.pool_columns(pool)
        ok_outer &= np.array_equal(T.select(flat, y_sel, tr, cols, 7), T.select(flat_te, y_sel, tr, cols, 7))
        ok_inner &= np.array_equal(T.select(flat, T.hide(y_sel, iva), itr, cols, 7),
                                   T.select(flat_p, T.hide(y_p, iva), itr, cols, 7))
    check("D", "outer SVM-RFE ignores outer-test rows, for all 7 pools", bool(ok_outer))
    check("D", "inner SVM-RFE ignores inner-validation and outer-test rows and labels, for all 7 pools", bool(ok_inner))
    try:
        T.fit_config(d, y, "A_mALFF", 20261015, 0, tr, te)
        refused = False
    except AssertionError:
        refused = True
    check("D", "fit_config refuses to run when outer-test labels are not hidden", refused)
    full = T.pool_columns("all4")
    _, sup_ref, _ = S.fit_selection(flat, y, tr, 20261015)
    check("D", "the all-four pool reproduces the baseline's SVM-RFE selection exactly",
          np.array_equal(T.select(flat, y_sel, tr, full, 20261015), sup_ref) and np.array_equal(full, np.arange(984)))
    log("  INFO  scaling: SVM-RFE standardises inside its pipeline on the rows it is fitted on; TabPFN receives raw "
        "values and preprocesses with its training context only; there is no imputation or resampling step")

    # ------------------------------------------------------------------ E
    log("\n[E] stability of the existing SVM-RFE selections (all4, k = 50)")
    recs = json.load(open(TABPFN))["records"]
    sups = [r["support"] for r in recs]
    jj = [jacc(sups[a], sups[b]) for a in range(len(sups)) for b in range(a + 1, len(sups))]
    freq = np.bincount(np.concatenate(sups), minlength=984) / len(sups)
    comp = {n: float(np.mean([np.isin(np.array(s), T.pool_columns(p)).mean() for s in sups]))
            for n, p in zip(names, ["mALFF", "mReHo", "DCz", "FC"])}
    fs = pd.read_csv(os.path.join(V2, "analysis", "feature_stability.csv"))
    facts["baseline_selection"] = {"pairwise_jaccard_25_outer_fits": float(np.mean(jj)),
                                   "features_selected_in_half_or_more": int((freq >= 0.5).sum()),
                                   "biomarker_share": comp, "v2_inner_pairwise_jaccard": float(fs.inner_pairwise_jaccard.mean())}
    log(f"  INFO  baseline outer fits: mean pairwise Jaccard {np.mean(jj):.2f}; features chosen in >= 50% of the 25 fits: "
        f"{int((freq >= 0.5).sum())}; share of selected features by biomarker "
        + ", ".join(f"{k} {100 * v:.0f}%" for k, v in comp.items()) + f"; v2 inner fits Jaccard {fs.inner_pairwise_jaccard.mean():.2f}")

    # ------------------------------------------------------------------ F
    log("\n[F] scanner site and acquisition")
    site = np.array([s.replace("sub-", "")[:3] for s in subjects])
    tabct = pd.crosstab(site, [S.STAGES[v] for v in y])[S.STAGES]
    chi2, p, dof, _ = chi2_contingency(tabct.to_numpy())
    v = np.sqrt(chi2 / (116 * (min(tabct.shape) - 1)))
    facts["site_by_stage"] = tabct.to_dict(orient="index")
    check("F", "scanner site independent of stage", p > 0.05,
          f"{tabct.shape[0]} sites, chi2 p = {p:.1e}, Cramer's V = {v:.2f}; "
          f"sites missing at least 3 of the 6 stages: {int(((tabct > 0).sum(1) <= 3).sum())}")
    log("  INFO  site x stage counts (CN SMC EMCI MCI LMCI AD):")
    for s_, r in tabct.iterrows():
        log(f"        {s_}: {r.tolist()}")
    big = [s_ for s_, n in pd.Series(site).value_counts().items() if n >= 5]
    keep = np.isin(site, big)
    p_site = [kruskal(*[flat[keep & (site == s_), k] for s_ in big]).pvalue for k in range(984)]
    p_stage = [kruskal(*[flat[y == g, k] for g in range(T.NS)]).pvalue for k in range(984)]
    q_site, q_stage = bh(p_site), bh(p_stage)
    facts["features_fdr_site"] = int((q_site < 0.05).sum())
    facts["features_fdr_stage"] = int((q_stage < 0.05).sum())
    check("F", "imaging features carry little site information (few features differ by site after FDR)",
          facts["features_fdr_site"] < 50,
          f"{facts['features_fdr_site']} of 984 features differ by site (sites with >= 5 subjects, FDR 5%) vs "
          f"{facts['features_fdr_stage']} by stage")
    ba = []
    for s in H.SEEDS:
        pred = np.full(116, -1)
        for tr_, te_ in plan[s][1]:
            glob_major = np.bincount(y[tr_], minlength=T.NS).argmax()
            for i in te_:
                ys = y[tr_][site[tr_] == site[i]]
                if ys.size:
                    c = np.bincount(ys, minlength=T.NS)
                    tie = np.flatnonzero(c == c.max())
                    pred[i] = tie[np.argmax(np.bincount(y[tr_], minlength=T.NS)[tie])]
                else:
                    pred[i] = glob_major
        ba.append(balanced_accuracy_score(y, pred))
    facts["site_only_rule_balanced_accuracy"] = {"mean": float(np.mean(ba)), "per_seed": [float(b) for b in ba]}
    check("F", "site alone does not predict stage under the same CV (site-only lookup rule near chance)",
          np.mean(ba) < 0.2, f"balanced accuracy {100 * np.mean(ba):.1f}% (seeds {[round(100 * b, 1) for b in ba]}); "
          "the rule predicts the most common training stage of the test subject's site")
    log("  INFO  from the leakage-safe v1 audit: scan year differs by stage (Kruskal p = 3.8e-05; all SMC scans 2013); "
        "TR differs by stage (p = 0.044; only EMCI has 2250 ms scans); all 112 matched scans are Philips; age (p = 0.11) "
        "and sex (p = 0.31) do not differ by stage")

    # ------------------------------------------------------------------ G
    log("\n[G] metric consistency")
    worst = 0.0
    sources = {"TabPFN baseline": {(r["seed"], r["fold"]): (r["test_index"], r["pred"]) for r in recs},
               "transformer v2 FIXED": {(r["seed"], r["fold"]): (r["test_idx"], r["pred"]["FIXED"]) for r in
                                        (json.load(open(f)) for f in glob.glob(os.path.join(V2, "grid", "*.json")))}}
    repro = {}
    for name, src in sources.items():
        bals, f1s = [], []
        for s in H.SEEDS:
            pred = np.full(116, -1)
            for f in range(5):
                ti, pr = src[(s, f)]
                pred[np.array(ti)] = pr
            e = H.evaluate(y, pred)
            worst = max(worst, abs(e["balanced_accuracy"] - balanced_accuracy_score(y, pred)),
                        abs(e["macro_f1"] - f1_score(y, pred, labels=range(T.NS), average="macro", zero_division=0)),
                        max(abs(e["per_class_recall"][S.STAGES[g]] - r) for g, r in
                            enumerate(recall_score(y, pred, labels=range(T.NS), average=None, zero_division=0))))
            bals.append(e["balanced_accuracy"])
            f1s.append(e["macro_f1"])
        repro[name] = (100 * np.mean(bals), 100 * np.mean(f1s))
    check("G", "project balanced accuracy, macro-F1 and recall equal scikit-learn's on all saved predictions",
          worst < 1e-12, f"max difference {worst:.1e}")
    check("G", "reported numbers reproduced from saved predictions (TabPFN 24.6 / 23.5; transformer v2 19.1 / 17.7)",
          [round(v, 1) for v in repro["TabPFN baseline"]] == [24.6, 23.5]
          and [round(v, 1) for v in repro["transformer v2 FIXED"]] == [19.1, 17.7],
          "; ".join(f"{k}: balanced {v[0]:.2f}%, macro-F1 {v[1]:.2f}%" for k, v in repro.items()))
    log("  INFO  definition used for all point estimates: for each seed the 5 outer test folds are pooled into one "
        "prediction per subject, the metric is computed on the 116 subjects, then averaged over the 5 seeds")
    log("  INFO  95% CI used: percentile interval of 2000 class-stratified bootstraps of SUBJECTS; in each resample every "
        "seed's predictions for the drawn subjects are reused and the 5 seeds averaged (seeds are not treated as "
        "independent). It covers test-subject sampling given the trained models, not training-set variability; paired "
        "comparisons add a corrected resampled t-test (Nadeau-Bengio) over the 25 splits for that")

    n_flag = sum(r["status"] != "PASS" for r in results)
    log(f"\n{len(results)} checks, {n_flag} flagged")
    json.dump({"results": results, "facts": facts}, open(os.path.join(OUT, "audit.json"), "w"), indent=2, default=str)
    open(os.path.join(OUT, "audit.log"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
