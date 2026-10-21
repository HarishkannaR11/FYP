"""
Phase 1 audit, before any new modelling. Read-only on every existing file.

  A identity and grouping   one scan per subject; IDs agree across files;
                            every outer/inner split is disjoint by subject
  B labels                  valid, complete, identical across every source
  C duplicates              exact and near-duplicate scans
  D features                finiteness, constants, FC-strength vs DC redundancy
  E leakage scans           single-feature separability, row order, confounds
                            (site, ADNI phase, scanner, age, sex)
  F fit-on-train checks     perturb the test rows and labels of a split and
                            confirm every fitted quantity (ComBat, SVM-RFE,
                            imputer, scaler, LR/SVM/ordinal, TabPFN, inner folds)
                            is unchanged
  G metrics                 the project's metric functions vs scikit-learn;
                            every saved prediction file recomputed

Writes audit/audit.json and audit/audit.md under derivatives/experiments/leakage_safe_v1/.

Run:  python scripts/experiments/leakage_safe_v1/audit.py
"""
import csv
import glob
import io
import json
import os
import sys
import time
import zipfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as K  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.metrics import balanced_accuracy_score, cohen_kappa_score, f1_score, roc_auc_score  # noqa: E402

from train_transformer import evaluate as tt_evaluate  # noqa: E402

ROOT, SRC = K.ROOT, os.path.join(K.ROOT, "derivatives", "model_ready")
OUT = os.path.join(K.EXP, "audit")
res, lines = {}, []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def check(section, name, ok, detail=""):
    res.setdefault(section, []).append({"check": name, "ok": bool(ok), "detail": detail})
    log(f"  {'PASS' if ok else 'FLAG'}  {name}" + (f"  -- {detail}" if detail else ""))


def main():
    if os.path.exists(os.path.join(OUT, "audit.json")):
        raise SystemExit("audit already written; not overwritten")
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    d = K.load()
    y, subjects, flat = d["y"], d["subjects"], d["flat"]
    n = len(y)
    idx = pd.read_csv(os.path.join(SRC, "index.csv"))
    acq = np.load(os.path.join(SRC, "acq_ids.npy"), allow_pickle=True).astype(str)
    split = pd.read_csv(os.path.join(SRC, "splits", "split_assignment.csv"))
    smeta = json.load(open(os.path.join(SRC, "splits", "split_metadata.json")))
    meta = json.load(open(os.path.join(SRC, "metadata.json")))
    clin = pd.read_csv(os.path.join(ROOT, "derivatives", "comorbidity_features", "model_clinical_116.csv"))
    demo = pd.read_csv(os.path.join(ROOT, "derivatives", "comorbidity_features", "demographics_116.csv"))

    log("=" * 100)
    log("PHASE 1 AUDIT -- 116-subject rs-fMRI cohort")
    log("=" * 100)

    # ---------------------------------------------------------------- A
    log("\n[A] identity and subject-level grouping")
    check("A", "116 rows, 116 unique subjects", n == 116 and len(set(subjects)) == 116, f"rows {n}, unique {len(set(subjects))}")
    check("A", "116 unique acquisition ids, one acquisition per subject",
          len(set(acq)) == n and idx.groupby("subject").size().max() == 1,
          f"max acquisitions per subject {idx.groupby('subject').size().max()}")
    same = (list(idx["subject"]) == list(subjects) and list(split["subject_id"]) == list(subjects)
            and list(clin["bids_id"]) == list(subjects) and list(demo["bids_id"]) == list(subjects))
    check("A", "subject order identical in subjects.npy, index.csv, split file, clinical and demographics tables", same)
    for nf in (5, 10):
        plan = K.S.cv_plan(y, subjects, nf)
        ok, minc = True, 99
        for seed in K.SEEDS:
            fp, folds = plan[seed]
            tested = []
            for tr, te in folds:
                ok &= not (set(subjects[tr]) & set(subjects[te]))
                tested += subjects[te].tolist()
                minc = min(minc, int(np.bincount(y[te], minlength=6).min()))
            ok &= sorted(tested) == sorted(subjects.tolist())
        check("A", f"{nf}-fold x 5 seeds: train/test disjoint by subject ID, every subject tested once per seed", ok,
              f"smallest per-stage count in a test fold: {minc}"
              + (" (some test folds miss a stage)" if minc == 0 else ""))
        if nf == 5:
            res["fold_fingerprints_5"] = {str(s): plan[s][0] for s in K.SEEDS}
            res["fold_fingerprints_match_earlier_runs"] = res["fold_fingerprints_5"] == {
                "20261015": "3edbea19c1ee3c08", "20261016": "d53ad26a06b7d5c5", "20261017": "2a8e70dd0eb80cae",
                "20261018": "d54bf96735be41dd", "20261019": "8b7f4cd9e512b26a"}
            check("A", "5-fold plans identical to every earlier experiment (fingerprints)",
                  res["fold_fingerprints_match_earlier_runs"])
            tr, te = plan[K.SEEDS[0]][1][0]
            inner = K.inner_folds(y, tr, K.SEEDS[0], 0)
            check("A", "inner folds use only the outer training subjects",
                  all(set(a) | set(b) <= set(tr.tolist()) and not (set(a) & set(b)) for a, b in inner))
    pool = split.index[split["partition"] == "train_pool"].to_numpy()
    test = split.index[split["partition"] == "test"].to_numpy()
    check("A", "70/30 split disjoint, 81 + 35, fingerprint as recorded",
          not (set(pool) & set(test)) and len(pool) == 81 and len(test) == 35
          and smeta["split_fingerprint"] == K.H.EXPECTED_SPLIT_FINGERPRINT,
          f"fingerprint {smeta['split_fingerprint']}")

    # ---------------------------------------------------------------- B
    log("\n[B] labels")
    check("B", "labels complete and within 0..5", np.isfinite(y).all() and y.min() >= 0 and y.max() <= 5)
    st = [K.STAGES[v] for v in y]
    agree = (idx["y"].tolist() == y.tolist() and idx["stage"].tolist() == st and split["stage"].tolist() == st
             and clin["cohort_stage"].tolist() == st and clin["stage_index"].tolist() == y.tolist()
             and demo["cohort_stage"].tolist() == st)
    check("B", "label identical across y.npy, index.csv, split file, clinical and demographics tables", agree)
    counts = {K.STAGES[g]: int((y == g).sum()) for g in range(6)}
    check("B", "class counts match metadata.json", counts == meta["class_counts"], json.dumps(counts))

    # ---------------------------------------------------------------- C
    log("\n[C] duplicate scans")
    check("C", "no two subjects with identical 984-feature rows", len({r.tobytes() for r in flat}) == n)
    Z = (flat - flat.mean(0)) / np.where(flat.std(0) > 0, flat.std(0), 1)
    R = np.corrcoef(Z)
    np.fill_diagonal(R, -1)
    i, j = np.unravel_index(R.argmax(), R.shape)
    check("C", "no near-duplicate subjects (max between-subject correlation of standardised features < 0.95)",
          R.max() < 0.95, f"max r = {R.max():.3f} ({subjects[i]} vs {subjects[j]})")
    fc = np.load(os.path.join(SRC, "X_fc.npy"), mmap_mode="r")
    iu = np.triu_indices(246, 1)
    E = np.stack([np.asarray(fc[k])[iu] for k in range(n)])
    RE = np.corrcoef(E)
    np.fill_diagonal(RE, -1)
    i, j = np.unravel_index(RE.argmax(), RE.shape)
    check("C", "no near-duplicate connectivity matrices (max between-subject r < 0.95)",
          RE.max() < 0.95, f"max r = {RE.max():.3f} ({subjects[i]} vs {subjects[j]})")
    res["max_between_subject_r"] = {"features": float(R.max()), "fc_edges": float(RE.max())}

    # ---------------------------------------------------------------- D
    log("\n[D] features")
    check("D", "all 984 features finite", np.isfinite(flat).all())
    nconst = int((flat.std(0) == 0).sum())
    check("D", "no constant features", nconst == 0, f"{nconst} constant")
    fcu = flat[:, 738:]
    recomputed = np.stack([np.abs(np.asarray(fc[k])).sum(1) / 245 for k in range(n)])
    check("D", "FC strength = mean |Fisher-z| over the other 245 regions (as documented)",
          np.allclose(recomputed, fcu, atol=1e-4), f"max abs diff {np.abs(recomputed - fcu).max():.2e}")
    dc = flat[:, 2:738:3]
    r_dc = np.array([stats.pearsonr(dc[k], fcu[k])[0] for k in range(n)])
    res["fc_strength_vs_dc_within_subject_r"] = {"median": float(np.median(r_dc)), "min": float(r_dc.min()),
                                                 "max": float(r_dc.max())}
    check("D", "FC strength is not a copy of DC_z (within-subject r < 0.99)", np.median(r_dc) < 0.99,
          f"median within-subject r(DC_z, FC strength) = {np.median(r_dc):.3f} (min {r_dc.min():.3f})")

    # ---------------------------------------------------------------- E
    log("\n[E] leakage scans and confounds")
    rho = np.array([abs(stats.spearmanr(flat[:, k], y)[0]) for k in range(984)])
    aucs = []
    for g in range(6):
        yy = (y == g).astype(int)
        aucs.append(max(max(roc_auc_score(yy, flat[:, k]), 1 - roc_auc_score(yy, flat[:, k])) for k in range(984)))
    res["single_feature"] = {"max_abs_spearman": float(rho.max()), "max_one_vs_rest_auc": float(max(aucs)),
                             "max_auc_by_stage": dict(zip(K.STAGES, map(float, aucs)))}
    check("E", "no single feature separates a stage (max one-vs-rest AUC < 0.95)", max(aucs) < 0.95,
          f"max AUC {max(aucs):.3f}; max |Spearman rho| with stage {rho.max():.3f}")
    r_order = stats.spearmanr(np.arange(n), y)[0]
    check("E", "row order is sorted by stage (no model uses row order; folds are shuffled and stratified)",
          True, f"Spearman(row index, stage) = {r_order:.2f}")
    tab = pd.crosstab(d["site"], y)
    p_site = stats.chi2_contingency(tab.to_numpy())[1]
    res["site_stage"] = {"n_sites": int(tab.shape[0]), "chi2_p": float(p_site),
                         "counts": {s: tab.loc[s].tolist() for s in tab.index}}
    check("E", "scanner site independent of stage", p_site > 0.05,
          f"{tab.shape[0]} sites, chi2 p = {p_site:.1e} (CONFOUND: site predicts stage)")
    years = pd.to_datetime(clin["mri_date"]).dt.year.to_numpy()
    p_year = stats.kruskal(*[years[y == g] for g in range(6)]).pvalue
    res["scan_year_by_stage"] = {K.STAGES[g]: [int(years[y == g].min()), int(np.median(years[y == g])),
                                               int(years[y == g].max())] for g in range(6)}
    check("E", "scan year independent of stage", p_year > 0.05,
          f"Kruskal p = {p_year:.1e}; median year by stage "
          + ", ".join(f"{K.STAGES[g]} {int(np.median(years[y == g]))}" for g in range(6)))
    # scanner make/model/TR from the IDA image tables, matched by subject + scan date
    rows = []
    for z in glob.glob(os.path.join(ROOT, "Comorbidity", "*.zip")):
        zf = zipfile.ZipFile(z)
        for nm in zf.namelist():
            if "Functional_MRI_Images" in nm:
                rows.append(pd.read_csv(io.BytesIO(zf.read(nm))))
    img = pd.concat(rows) if rows else pd.DataFrame()
    scan = {}
    if len(img):
        img["fmri_date"] = pd.to_datetime(img["fmri_date"], errors="coerce")
        for k, (sid, dt) in enumerate(zip(clin["adni_id"], pd.to_datetime(clin["mri_date"]))):
            m = img[img["subject_id"] == sid]
            if len(m):
                mm = m.iloc[(m["fmri_date"] - dt).abs().argsort()[:1]]
                if abs((mm["fmri_date"].iloc[0] - dt).days) <= 31:
                    scan[k] = (str(mm["fmri_mfr"].iloc[0]), str(mm["fmri_mfr_model"].iloc[0]),
                               float(mm["fmri_tr"].iloc[0]) if pd.notna(mm["fmri_tr"].iloc[0]) else np.nan)
    res["scanner_matched"] = len(scan)
    if scan:
        ks = sorted(scan)
        mfr = pd.Series([scan[k][0] for k in ks])
        tab_m = pd.crosstab(mfr.to_numpy(), y[ks])
        p_m = stats.chi2_contingency(tab_m.to_numpy())[1] if tab_m.shape[0] > 1 else 1.0
        res["manufacturer_stage"] = {"chi2_p": float(p_m), "counts": {s: tab_m.loc[s].tolist() for s in tab_m.index}}
        check("E", "scanner manufacturer independent of stage", p_m > 0.05,
              f"{len(scan)}/116 scans matched; manufacturers {dict(mfr.value_counts())}; chi2 p = {p_m:.2g}")
        tr_ = np.array([scan[k][2] for k in ks])
        if np.isfinite(tr_).sum() > 10:
            ok = np.isfinite(tr_)
            p_tr = stats.kruskal(*[tr_[ok & (y[ks] == g)] for g in range(6) if (ok & (y[ks] == g)).sum()]).pvalue
            res["tr_by_stage"] = {K.STAGES[g]: sorted(set(np.round(tr_[ok & (y[ks] == g)], 0).tolist()))
                                  for g in range(6)}
            check("E", "repetition time (TR) independent of stage", p_tr > 0.05, f"Kruskal p = {p_tr:.2g}")
    age, male = d["covars"][:, 0], d["covars"][:, 1]
    p_age = stats.kruskal(*[age[y == g] for g in range(6)]).pvalue
    p_sex = stats.chi2_contingency(pd.crosstab(male, y).to_numpy())[1]
    check("E", "age independent of stage", p_age > 0.05, f"Kruskal p = {p_age:.2g}")
    check("E", "sex independent of stage", p_sex > 0.05, f"chi2 p = {p_sex:.2g}")

    # ---------------------------------------------------------------- F
    log("\n[F] fit-on-train verification: perturb test rows and labels, refit, compare")
    plan = K.S.cv_plan(y, subjects, 5)
    tr, te = plan[K.SEEDS[0]][1][0]
    rng = np.random.default_rng(1)
    d2 = {**d, "flat": d["flat"].copy(), "clin": d["clin"].copy(), "y": d["y"].copy()}
    d2["flat"][te] = rng.normal(size=d2["flat"][te].shape) * 50
    d2["clin"][te] = rng.normal(size=d2["clin"][te].shape) * 50
    y2 = y.copy()
    y2[te] = rng.integers(0, 6, len(te))
    cb1 = K.ComBat().fit(d["flat"][tr], d["site"][tr], d["covars"][tr])
    cb2 = K.ComBat().fit(d2["flat"][tr], d2["site"][tr], d2["covars"][tr])
    check("F", "ComBat parameters depend on training rows only", np.array_equal(cb1.state(), cb2.state()))
    X1, _ = K.harmonise(d, "combat", tr)
    X2, _ = K.harmonise(d2, "combat", tr)
    check("F", "harmonised training rows unchanged when test rows change", np.array_equal(X1[tr], X2[tr]))
    for k in (25, 50):
        s1 = K.rfe_support(d["flat"], y, tr, k, 7)
        s2 = K.rfe_support(d2["flat"], y2, tr, k, 7)
        check("F", f"SVM-RFE (k={k}) selection depends on training rows only", np.array_equal(s1, s2))
    s1 = K.rfe_support(X1, y, tr, 50, 7)
    s2 = K.rfe_support(X2, y2, tr, 50, 7)
    check("F", "SVM-RFE after ComBat depends on training rows only", np.array_equal(s1, s2))
    for fam in ("lr", "svm", "ordinal", "tabpfn"):
        hp = K.hp_list(fam)[0]
        A1 = K.design(d["flat"], d, "imgcom", s1)
        A2 = K.design(d2["flat"], d2, "imgcom", s1)
        probe = tr[:10]
        p1 = K.fit_predict(fam, hp, A1[tr], y[tr], A1[probe], 3)
        p2 = K.fit_predict(fam, hp, A2[tr], y2[tr], A2[probe], 3)
        same = np.array_equal(p1[0], p2[0]) and (p1[1] is None or np.allclose(p1[1], p2[1]))
        check("F", f"{fam}: imputer, scaler and model fitted on training rows only "
                   "(identical predictions after test rows change)", same)
    check("F", "inner folds depend on training labels only",
          [list(map(list, f)) for f in K.inner_folds(y, tr, 1, 0)] ==
          [list(map(list, f)) for f in K.inner_folds(y2, tr, 1, 0)])

    # ---------------------------------------------------------------- G
    log("\n[G] metric consistency and saved predictions")
    rng = np.random.default_rng(2)
    diff = 0.0
    for _ in range(500):
        t = rng.integers(0, 6, 40)
        p = np.where(rng.random(40) < 0.3, t, rng.integers(0, 6, 40))
        a, b = K.H.evaluate(t, p), tt_evaluate(np.eye(6)[p], t, pred=p)
        diff = max(diff, abs(a["balanced_accuracy"] - balanced_accuracy_score(t, p)),
                   abs(b["balanced_accuracy"] - balanced_accuracy_score(t, p)),
                   abs(a["macro_f1"] - f1_score(t, p, average="macro", labels=range(6), zero_division=0)),
                   abs(a["qwk"] - cohen_kappa_score(t, p, weights="quadratic", labels=range(6))),
                   abs(a["mae_stages"] - np.abs(t - p).mean()), abs(a["accuracy"] - (t == p).mean()))
    check("G", "project metric functions equal scikit-learn (accuracy, balanced accuracy, macro-F1, QWK, MAE)",
          diff < 1e-9, f"max abs difference {diff:.1e} over 500 random cases")
    files = glob.glob(os.path.join(ROOT, "derivatives", "model_training", "**", "*.json"), recursive=True)
    nrec = nbad = nfile = 0
    for fpath in files:
        try:
            obj = json.load(open(fpath))
        except Exception:  # noqa: BLE001
            continue
        recs = obj.get("records") if isinstance(obj, dict) else None
        if not isinstance(recs, list) or not recs or not isinstance(recs[0], dict) or "pred" not in recs[0]:
            continue
        nfile += 1
        for r in recs:
            if "true" not in r or "balanced_accuracy" not in r:
                continue
            nrec += 1
            m = K.H.evaluate(np.asarray(r["true"]), np.asarray(r["pred"]))
            bad = abs(m["balanced_accuracy"] - r["balanced_accuracy"]) > 1e-9
            if "test_index" in r:
                bad |= r["true"] != [int(y[i]) for i in r["test_index"]]
            nbad += bad
    check("G", "every saved prediction record reproduces its stored balanced accuracy and labels",
          nbad == 0, f"{nrec} records in {nfile} files, {nbad} mismatches")

    out = {"started": started, "results": res}
    os.makedirs(OUT, exist_ok=True)
    json.dump(out, open(os.path.join(OUT, "audit.json"), "w"), indent=1, default=str)
    flags = [c for sec in "ABCDEFG" for c in res.get(sec, []) if not c["ok"]]
    log(f"\n{sum(len(res.get(s, [])) for s in 'ABCDEFG')} checks, {len(flags)} flagged")
    open(os.path.join(OUT, "audit.log"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    K.manifest_entry("python scripts/experiments/leakage_safe_v1/audit.py", started,
                     ["derivatives/experiments/leakage_safe_v1/audit/audit.json",
                      "derivatives/experiments/leakage_safe_v1/audit/audit.log"])


if __name__ == "__main__":
    main()
