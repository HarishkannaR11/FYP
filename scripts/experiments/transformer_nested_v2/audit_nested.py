"""
Leakage audit for transformer_nested_v2 (run before the grid; touches no result file).

  A  folds and partitions: outer folds identical to the TabPFN k=50 baseline,
     inner folds partition the outer training set, locked 70/30 test set status
  B  inputs: every model-ready array is a per-subject quantity (no statistic
     across subjects), the local attention mask is the atlas geometry only
  C  fit-on-train perturbation tests on one outer split (seed 20261015, fold 0):
     scaler and SVM-RFE ignore inner-validation and outer-test rows and labels;
     training ignores inner-validation labels; selection and training ignore
     outer-test rows; fit_split refuses unhidden outer-test labels; the
     training loop reproduces C0 exactly when given C0's inputs
  D  static: the new scripts never call the locked-test evaluation

Writes derivatives/experiments/transformer_nested_v2/audit/audit.{json,log} (refuses to overwrite).
Run:  python scripts/experiments/transformer_nested_v2/audit_nested.py
"""
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_nested as N  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

S, H, ROOT = N.S, N.H, N.ROOT
OUT = os.path.join(N.EXP, "audit")
SRC = os.path.join(ROOT, "derivatives", "model_ready")
SEED, FOLD, J = 20261015, 0, 0
SHORT = 6                                        # epochs for the perturbation runs (same code path)
lines, results = [], []


def log(m=""):
    print(m, flush=True)
    lines.append(str(m))


def check(group, name, ok, detail="", flag_word="FLAG"):
    status = "PASS" if ok else flag_word
    results.append({"group": group, "check": name, "status": status, "detail": detail})
    log(f"  {status:4s}  {name}" + (f"  -- {detail}" if detail else ""))


def perturbed(d, rows, rng):
    X4, flat, fc, y, subjects, mask = d
    X4, flat, fc = X4.copy(), flat.copy(), fc.copy()
    X4[rows] += rng.normal(0, 3, X4[rows].shape).astype(X4.dtype)
    flat[rows] += rng.normal(0, 3, flat[rows].shape)
    noise = rng.normal(0, 0.5, fc[rows].shape).astype(fc.dtype)
    fc[rows] += noise + noise.transpose(0, 2, 1)
    return X4, flat, fc, y, subjects, mask


def main():
    if os.path.exists(os.path.join(OUT, "audit.json")):
        raise SystemExit("audit already written; not overwritten")
    os.makedirs(OUT, exist_ok=True)
    torch.set_num_threads(2)
    d = S.load_cohort()
    X4, flat, fc, y, subjects, mask = d
    plan = S.cv_plan(y, subjects, 5)
    rng = np.random.default_rng(0)

    # ------------------------------------------------------------ A
    log("[A] folds and partitions")
    tab = json.load(open(os.path.join(ROOT, "derivatives", "model_training", "tabpfn", "cv116_tabpfn_svmrfe50.json")))
    tab = {(r["seed"], r["fold"]): r for r in tab["records"]}
    same = [(sorted(te.tolist()) == sorted(tab[(s, f)]["test_index"])) and plan[s][0] == tab[(s, f)]["fold_fingerprint"]
            for s in H.SEEDS for f, (_, te) in enumerate(plan[s][1])]
    check("A", "outer folds identical to the TabPFN k=50 baseline (test indices and fingerprints)", all(same),
          f"{sum(same)}/{len(same)} splits; fingerprints {[plan[s][0] for s in H.SEEDS]}")
    ok_part, min_val = True, []
    for s in H.SEEDS:
        for f, (tr, te) in enumerate(plan[s][1]):
            ok_part &= not set(tr.tolist()) & set(te.tolist())
            fo = H.stratified_folds(N.hide(y, te), tr, S.N_INNER, np.random.default_rng([s * 100 + f, 7]))
            fo_true = H.stratified_folds(y, tr, S.N_INNER, np.random.default_rng([s * 100 + f, 7]))
            ok_part &= fo == fo_true                                  # hiding outer labels leaves inner folds unchanged
            for j in range(S.N_INNER):
                itr = {i for i in tr if fo[int(i)] != j}
                iva = {i for i in tr if fo[int(i)] == j}
                ok_part &= not itr & iva and itr | iva == set(tr.tolist()) and not (itr | iva) & set(te.tolist())
                min_val.append(min(int((y[list(iva)] == g).sum()) for g in range(S.N_STAGES)))
    check("A", "inner folds partition each outer training set; disjoint from the outer test fold; same as C0's", ok_part,
          f"25 splits x 4 inner folds; smallest per-stage count in an inner-validation fold {min(min_val)}")
    part = pd.read_csv(os.path.join(SRC, "splits", "split_assignment.csv"))
    locked = np.flatnonzero(part.partition.to_numpy() == "test")
    in_cv = [len(set(locked.tolist()) & set(np.concatenate([te for _, te in plan[s][1]]).tolist())) for s in H.SEEDS]
    check("A", "locked 70/30 test subjects are outside the 116-subject CV", False,
          f"{len(locked)} locked subjects; each CV seed tests {in_cv[0]} of them in its outer folds and trains on them in "
          "the other folds. This is the project's established design (all 116 in CV; identical folds to the TabPFN "
          "baseline). run_test.py is not run here, but the 35 subjects are not unseen by CV-based comparisons",
          flag_word="FLAG")

    # ------------------------------------------------------------ B
    log("\n[B] inputs are per-subject quantities")
    for c, n, target in ((0, "mALFF", "ROI mean = 1"), (1, "mReHo", "ROI mean = 1"), (2, "DC_z", "ROI mean 0, sd 1")):
        mu, sd = X4[:, :, c].mean(1), X4[:, :, c].std(1)
        ok = np.allclose(mu, 1, atol=1e-4) if c < 2 else (np.allclose(mu, 0, atol=1e-4) and np.allclose(sd, 1, atol=1e-4))
        check("B", f"{n} normalised within each subject ({target} for every subject)", ok,
              f"ROI mean range {mu.min():.5f}..{mu.max():.5f}, ROI sd range {sd.min():.4f}..{sd.max():.4f}")
    off = ~np.eye(246, dtype=bool)
    st = np.abs(fc.astype(np.float64))[:, off].reshape(len(y), 246, 245).mean(2)
    check("B", "FC strength = each subject's own mean |Fisher-z| per region", np.abs(st - X4[:, :, 3]).max() < 1e-5,
          f"max abs difference {np.abs(st - X4[:, :, 3]).max():.1e}")
    check("B", "FC matrices symmetric with zero diagonal (per-subject Fisher-z)",
          bool(np.allclose(fc, fc.transpose(0, 2, 1)) and np.abs(fc[:, np.arange(246), np.arange(246)]).max() == 0))
    cen = np.load(os.path.join(SRC, "roi_centroids_mm.npy"))
    D = np.linalg.norm(cen[:, None] - cen[None], axis=2)
    nn = np.argsort(D, 1)[:, 1:16]
    A = np.zeros((246, 246), bool)
    A[np.repeat(np.arange(246), 15), nn.ravel()] = True
    A = A | A.T
    A[np.arange(246), np.arange(246)] = True
    check("B", "local attention mask = atlas centroid 15-NN graph (geometry only, no subject data)",
          bool(np.array_equal(A, mask)), f"{int((A != mask).sum())} differing entries")
    hashes = {f: hashlib.sha256(open(os.path.join(SRC, f), "rb").read()).hexdigest()[:16]
              for f in ("X_regional.npy", "X_fc.npy", "X_fc_roi_unsigned.npy", "y.npy", "subjects.npy",
                        "local_attention_mask.npy")}
    log(f"  INFO  input hashes {hashes}")

    # ------------------------------------------------------------ C
    log(f"\n[C] perturbation tests (seed {SEED}, fold {FOLD}, inner model {J}; short runs use {SHORT} epochs)")
    fp, folds = plan[SEED]
    tr, te = folds[FOLD]
    y_sel = N.hide(y, te)
    fo = H.stratified_folds(y_sel, tr, S.N_INNER, np.random.default_rng([SEED * 100 + FOLD, 7]))
    itr = np.array([i for i in tr if fo[int(i)] != J])
    iva = np.array([i for i in tr if fo[int(i)] == J])
    tag = SEED + FOLD
    _, _, _, sup0, rois0, sc0 = N.inner_inputs(d, y_sel, itr, iva, tag)

    y_p = y_sel.copy()
    y_p[iva] =(y_p[iva] + 1 + rng.integers(0, S.N_STAGES - 1, len(iva))) % S.N_STAGES      # every iva label changed
    dp = perturbed(d, np.concatenate([iva, te]), rng)
    _, _, _, sup1, rois1, sc1 = N.inner_inputs(dp, y_p, itr, iva, tag)
    check("C", "inner scaler ignores inner-validation and outer-test rows",
          bool(np.array_equal(sc0.mean, sc1.mean) and np.array_equal(sc0.std, sc1.std)))
    check("C", "inner SVM-RFE ignores inner-validation and outer-test rows and labels", bool(np.array_equal(sup0, sup1)),
          f"{len(sup0)} features, {len(rois0)} regions")
    _, sup_outer, _ = S.fit_selection(flat, y, tr, tag)
    jac = len(set(sup0) & set(sup_outer)) / len(set(sup0) | set(sup_outer))
    check("C", "inner SVM-RFE is a separate fit from C0's outer-training fit", jac < 1,
          f"Jaccard overlap with the outer-training selection {jac:.2f} ({len(itr)} vs {len(tr)} subjects)")

    X, F, mt, _, _, _ = N.inner_inputs(d, y_sel, itr, iva, tag)
    a = N.train_inner(X, F, mt, y_sel, itr, iva, te, 7, SHORT)
    b = N.train_inner(X, F, mt, y_p, itr, iva, te, 7, SHORT)
    check("C", "training trajectory ignores inner-validation labels (labels used only to choose the checkpoint)",
          a["curves"]["train"] == b["curves"]["train"] and torch.equal(a["outer_logits"], b["outer_logits"]),
          "train curves and outer-fold outputs identical; validation curves differ as intended: "
          f"{a['curves']['val']['bal'] != b['curves']['val']['bal']}")

    try:
        N.fit_split(d, y, SEED, FOLD, tr, te, SHORT)
        refused = False
    except AssertionError:
        refused = True
    check("C", "fit_split refuses to run when outer-test labels are not hidden", refused)

    r0 = N.fit_split(d, y_sel, SEED, FOLD, tr, te, SHORT)
    r1 = N.fit_split(perturbed(d, te, rng), y_sel, SEED, FOLD, tr, te, SHORT)
    same_sel = (r0[1] == r1[1] and r0[2] == r1[2] and np.array_equal(r0[3], r1[3]) and np.array_equal(r0[4], r1[4])
                and all(m0["curves"] == m1["curves"] and np.array_equal(m0["support"], m1["support"])
                        for m0, m1 in zip(r0[0], r1[0])))
    changed = any(not np.allclose(r0[5][k], r1[5][k]) for k in N.RULES)
    check("C", "selection, inner curves and chosen epochs ignore outer-test rows", same_sel,
          f"outer-fold probabilities change with the perturbed rows (sanity): {changed}")

    v1 = json.load(open(os.path.join(ROOT, "derivatives", "experiments", "transformer_diagnostics_v1", "grid", "C0",
                                     f"cv5_seed{SEED}_fold{FOLD}.json")))
    Xs = N.Scaler().fit(X4, tr).transform(X4).astype(np.float32)
    _, _, chm = S.fit_selection(flat, y, tr, tag)
    tok, rois = S.make_tokens(Xs, chm)
    Xc = torch.from_numpy(np.ascontiguousarray(tok))
    Fc = torch.from_numpy(np.ascontiguousarray(fc[:, rois][:, :, rois], dtype=np.float32))
    Mc = torch.from_numpy(np.ascontiguousarray(mask[np.ix_(rois, rois)]))
    c0 = N.train_inner(Xc, Fc, Mc, y_sel, itr, iva, te, (SEED * 100 + FOLD) * 10 + J)
    ref = v1["models"][J]["curves"]
    same_c0 = (np.allclose(c0["curves"]["val"]["bal"], ref["val"]["bal"]) and
               np.allclose(c0["curves"]["train"]["loss"], ref["train"]["loss"], atol=1e-6) and
               c0["best_epoch"] == v1["models"][J]["best_epoch"])
    check("C", "training loop reproduces C0 exactly when given C0's outer-fitted inputs", same_c0,
          f"best epoch {c0['best_epoch'] + 1} vs C0 {v1['models'][J]['best_epoch'] + 1}; so the only change in v2 is "
          "where the scaler and SVM-RFE are fitted")

    # ------------------------------------------------------------ D
    log("\n[D] static")
    srcs = {os.path.basename(p): open(p, encoding="utf-8").read() for p in glob.glob(os.path.join(HERE, "*.py"))}
    calls = [n for n, s in srcs.items() if "run_test" in s and n != os.path.basename(__file__)]
    check("D", "no v2 script calls the locked-test evaluation (run_test.py)", not calls, ", ".join(calls))

    n_flag = sum(r["status"] != "PASS" for r in results)
    log(f"\n{len(results)} checks, {n_flag} flagged")
    json.dump({"results": results, "input_hashes": hashes, "split": {"seed": SEED, "fold": FOLD, "inner_model": J}},
              open(os.path.join(OUT, "audit.json"), "w"), indent=2)
    open(os.path.join(OUT, "audit.log"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
