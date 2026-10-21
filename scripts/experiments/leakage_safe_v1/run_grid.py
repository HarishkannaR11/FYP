"""
Leakage-safe experiment grid, one outer split at a time (5 folds x 5 seeds,
the project's existing fold plans, identical to every earlier experiment).

Jobs (each writes grid/<job>/cv5_seed<s>_fold<f>.json once; existing files are skipped):

  main       per split, for harmonisation in {none, ComBat} and inputs in
             {img, imgcom}:
               - LR, RBF-SVM, ordinal LR at every k in K_GRID and every
                 hyperparameter: outer prediction + inner 3-fold scores
               - TabPFN outer predictions at k in TABPFN_K (no ComBat) and k=50 (ComBat)
               - majority-class baseline
             ComBat and SVM-RFE are fitted on the outer training set for the
             outer prediction, and on each inner training set for inner scores.
             The selected feature indices are saved for reuse.
  nested     TabPFN inner 3-fold scores for the nested grid (k in
             TABPFN_NESTED_K without ComBat, k=50 with ComBat), reusing the
             saved selections and inner folds of `main`
  biomarker  each biomarker alone (246 features, no selection, no ComBat,
             imaging only): TabPFN outer; LR/SVM/ordinal outer + inner scores
  shuffled   training labels permuted (selection, ComBat, inner CV and fit all
             see the permuted labels): TabPFN k=50 on all 25 splits; the whole
             nested grid (imgcom) on the first seed

Run:  python scripts/experiments/leakage_safe_v1/run_grid.py --job main [--limit 1]
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as K  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


def log(m):
    print(m, flush=True)


def majority(labels, tr, te):
    c = np.bincount(labels[tr], minlength=K.NS).argmax()
    return np.full(len(te), c)


def prep(d, labels, harm, tr, seed, fold, ks, inner):
    """ComBat + SVM-RFE fitted on tr (and on each inner training set) only."""
    Xo, cb = K.harmonise(d, harm, tr)
    unadj = None
    if cb is not None:                       # test subjects whose site has no fitted adjustment
        te_mask = np.ones(len(labels), bool)
        te_mask[tr] = False
        unadj = int((~np.isin(d["site"][te_mask], cb.sites_)).sum())
    sup_o = {k: K.rfe_support(Xo, labels, tr, k, seed + fold) for k in ks}
    inn = []
    for itr, iva in inner:
        Xi, _ = K.harmonise(d, harm, itr)
        inn.append((itr, iva, Xi, {k: K.rfe_support(Xi, labels, itr, k, seed + fold) for k in ks}))
    return Xo, sup_o, inn, unadj


def light_configs(rec, d, labels, inputs, view, harm, Xo, cols_o, inn_cols, tr, te):
    """LR / SVM / ordinal: every hyperparameter, outer prediction + inner scores."""
    for fam in K.LIGHT:
        for hp in K.hp_list(fam):
            Xd = K.design(Xo, d, inputs, cols_o)
            pred, p, tb = K.fit_predict(fam, hp, Xd[tr], labels[tr], Xd[te], 0, train_score=True)
            scores = []
            for itr, iva, Xi, ci in inn_cols:
                Xdi = K.design(Xi, d, inputs, ci)
                pi, _, _ = K.fit_predict(fam, hp, Xdi[itr], labels[itr], Xdi[iva], 0)
                scores.append(K.bal_acc(labels[iva], pi))
            rec["configs"][K.cid(fam, inputs, view, harm, hp)] = {
                "pred": pred.tolist(), "probs": K.r6(p), "inner": scores, "train_bal": tb}


def tabpfn_outer(rec, d, labels, inputs, view, harm, Xo, cols, tr, te, seed, fold):
    Xd = K.design(Xo, d, inputs, cols)
    pred, p, _ = K.fit_predict("tabpfn", "-", Xd[tr], labels[tr], Xd[te], seed + fold)
    rec["configs"][K.cid("tabpfn", inputs, view, harm, "-")] = {
        "pred": pred.tolist(), "probs": K.r6(p), "inner": None, "train_bal": None}


def tabpfn_inner(rec, d, labels, inputs, view, harm, inn_cols, seed, fold):
    scores = []
    for j, (itr, iva, Xi, ci) in enumerate(inn_cols):
        Xdi = K.design(Xi, d, inputs, ci)
        pi, _, _ = K.fit_predict("tabpfn", "-", Xdi[itr], labels[itr], Xdi[iva], seed + fold + 100 * (j + 1))
        scores.append(K.bal_acc(labels[iva], pi))
    rec["configs"][K.cid("tabpfn", inputs, view, harm, "-")] = {"inner": scores}


def base_record(seed, fold, fp, tr, te, d, labels, shuffled):
    return {"seed": seed, "fold": fold, "fold_fingerprint": fp, "train_idx": tr.tolist(), "test_idx": te.tolist(),
            "subjects": [str(d["subjects"][i]) for i in te], "true": d["y"][te].tolist(),
            "shuffled_labels": shuffled, "configs": {}, "supports": {}, "combat_unadjusted_test": {}}


def run_main(d, seed, fold, fp, tr, te, labels, shuffled=False, inputs_list=("img", "imgcom"), full=True):
    rec = base_record(seed, fold, fp, tr, te, d, labels, shuffled)
    inner = K.inner_folds(labels, tr, seed, fold)
    rec["inner_folds"] = [[a.tolist(), b.tolist()] for a, b in inner]
    rec["configs"][K.cid("majority", "-", "-", "-", "-")] = {"pred": majority(labels, tr, te).tolist(),
                                                              "probs": None, "inner": None, "train_bal": None}
    for harm in K.HARMS:
        t0 = time.time()
        ks = [k for k in K.K_GRID if k < 984] if full else [50]
        Xo, sup_o, inn, unadj = prep(d, labels, harm, tr, seed, fold, ks, inner)
        if full:
            sup_o[984] = np.arange(984)
            for _, _, _, ci in inn:
                ci[984] = np.arange(984)
        rec["combat_unadjusted_test"][harm] = unadj
        rec["supports"][harm] = {"outer": {str(k): v.tolist() for k, v in sup_o.items() if k < 984},
                                 "inner": [{str(k): v.tolist() for k, v in ci.items() if k < 984} for _, _, _, ci in inn]}
        for inputs in inputs_list:
            for k in [k for k in K.K_GRID if k in sup_o]:
                if full:
                    light_configs(rec, d, labels, inputs, f"k{k}", harm, Xo, sup_o[k],
                                  [(a, b, X, ci[k]) for a, b, X, ci in inn], tr, te)
                if (harm == "none" and k in K.TABPFN_K) or (harm == "combat" and k == 50):
                    if full or k == 50:
                        tabpfn_outer(rec, d, labels, inputs, f"k{k}", harm, Xo, sup_o[k], tr, te, seed, fold)
        rec.setdefault("timing", {})[harm] = round(time.time() - t0, 1)
    return rec


def run_nested(d, seed, fold, main_rec, labels, inputs_list=("img", "imgcom")):
    tr, te = np.array(main_rec["train_idx"]), np.array(main_rec["test_idx"])
    inner = [(np.array(a), np.array(b)) for a, b in main_rec["inner_folds"]]
    rec = {"seed": seed, "fold": fold, "fold_fingerprint": main_rec["fold_fingerprint"], "configs": {}}
    for harm, ks in (("none", K.TABPFN_NESTED_K), ("combat", [50])):
        inn = []
        for j, (itr, iva) in enumerate(inner):
            Xi, _ = K.harmonise(d, harm, itr)
            ci = {k: np.array(main_rec["supports"][harm]["inner"][j][str(k)]) for k in ks}
            inn.append((itr, iva, Xi, ci))
        for inputs in inputs_list:
            for k in ks:
                tabpfn_inner(rec, d, labels, inputs, f"k{k}", harm, [(a, b, X, ci[k]) for a, b, X, ci in inn],
                             seed, fold)
    return rec


def run_biomarker(d, seed, fold, fp, tr, te, labels):
    rec = base_record(seed, fold, fp, tr, te, d, labels, False)
    inner = K.inner_folds(labels, tr, seed, fold)
    X = d["flat"]
    for name, b in K.BIOMARKERS.items():
        cols = K.biomarker_columns(b)
        view = f"bio_{name}"
        light_configs(rec, d, labels, "img", view, "none", X, cols, [(a, c, X, cols) for a, c in inner], tr, te)
        tabpfn_outer(rec, d, labels, "img", view, "none", X, cols, tr, te, seed, fold)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--job", required=True, choices=["main", "nested", "biomarker", "shuffled"])
    ap.add_argument("--limit", type=int, default=0, help="stop after this many new splits (0 = all)")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    d = K.load()
    plan = K.S.cv_plan(d["y"], d["subjects"], 5)
    done, written = 0, []
    for seed in K.SEEDS:
        fp, folds = plan[seed]
        for f, (tr, te) in enumerate(folds):
            path = K.split_path(args.job, seed, f)
            if os.path.exists(path):
                continue
            t0 = time.time()
            if args.job == "main":
                rec = run_main(d, seed, f, fp, tr, te, d["y"])
            elif args.job == "nested":
                mp = K.split_path("main", seed, f)
                if not os.path.exists(mp):
                    log(f"seed {seed} fold {f}: main record missing, skipped")
                    continue
                import json
                rec = run_nested(d, seed, f, json.load(open(mp)), d["y"])
            elif args.job == "biomarker":
                rec = run_biomarker(d, seed, f, fp, tr, te, d["y"])
            else:
                labels = d["y"].copy()
                yt = labels[tr].copy()
                np.random.default_rng(seed * 100 + f).shuffle(yt)
                labels[tr] = yt
                first = seed == K.SEEDS[0]
                rec = run_main(d, seed, f, fp, tr, te, labels, shuffled=True, full=first)
                if first:
                    rec["nested"] = run_nested(d, seed, f, rec, labels, inputs_list=("imgcom",))["configs"]
            rec["seconds"] = round(time.time() - t0, 1)
            K.write_once(path, rec)
            written.append(os.path.relpath(path, K.ROOT))
            log(f"{args.job}: seed {seed} fold {f} done in {rec['seconds']:.0f}s "
                f"({len([c for c in rec['configs']])} configs)")
            done += 1
            if args.limit and done >= args.limit:
                break
        if args.limit and done >= args.limit:
            break
    K.manifest_entry(" ".join(["python", "scripts/experiments/leakage_safe_v1/run_grid.py"] + sys.argv[1:]),
                     started, written, note=f"{args.job}: {len(written)} new split files")
    log(f"{args.job}: {len(written)} new split files")


if __name__ == "__main__":
    main()
