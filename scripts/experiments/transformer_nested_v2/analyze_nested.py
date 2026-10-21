"""
Analysis for transformer_nested_v2, from saved predictions and curves only (trains nothing).

Models compared on the same 25 outer splits (5 folds x 5 seeds, rs-fMRI only):
  v2 FIXED       corrected transformer, one epoch per split from inner-CV loss (primary, declared in advance)
  v2 ES          corrected transformer, nested early stopping
  v2 FIXED_BAL   sensitivity: epoch from pooled inner-CV balanced accuracy
  v2 LAST        diagnostic: epoch 60, no selection
  C0 (v1)        the earlier transformer: scaler + SVM-RFE fitted on the whole outer training set
  TabPFN k=50    the existing TabPFN v2 baseline

Held-out = the outer test fold. Inner = the inner-validation folds inside the outer training set.
Held-out results are reported, never used to choose anything.

Metric convention as in transformer_diagnostics_v1/analyze.py: per seed the 5 test folds are pooled
into one out-of-fold prediction per subject, metrics are computed on that, then mean +/- sd over
seeds; fold-level sd separately; CIs from a stratified subject bootstrap (2000), paired
differences on the same resamples; verdict = CI excludes 0 and at least 4 of 5 seeds agree.

Writes derivatives/experiments/transformer_nested_v2/analysis/ (refuses to overwrite).
Run:  python scripts/experiments/transformer_nested_v2/analyze_nested.py
"""
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "transformer_diagnostics_v1"))
import analyze as A1  # noqa: E402  (pooling, bootstrap, verdict; imports leakage_safe_v1/report)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402

R, K, P = A1.R, A1.K, A1.P
ROOT = K.ROOT
EXP = os.path.join(ROOT, "derivatives", "experiments", "transformer_nested_v2")
OUT = os.path.join(EXP, "analysis")
STAGES, NS = K.STAGES, K.NS
NAMES = {"FIXED": "Transformer v2, fixed epoch from inner CV (primary)",
         "ES": "Transformer v2, nested early stopping",
         "FIXED_BAL": "Transformer v2, fixed epoch from inner balanced acc. (sensitivity)",
         "LAST": "Transformer v2, epoch 60 (no selection)",
         "C0": "Transformer C0 (v1: selection fitted on whole outer training set)",
         "TABPFN": "TabPFN v2, k=50 (existing baseline)"}
ORDER = ["FIXED", "ES", "FIXED_BAL", "LAST", "C0", "TABPFN"]
N_SPLITS = 25


def load_v2():
    recs = {}
    for f in glob.glob(os.path.join(EXP, "grid", "cv5_*.json")):
        r = json.load(open(f))
        recs[(r["seed"], r["fold"])] = r
    return recs


def no_selection_gap(models, outer_of, lo=9):
    """Mean over inner models and epochs 10..60 of inner-val minus held-out balanced accuracy (no epoch chosen)."""
    return float(np.mean([np.mean(np.array(m["curves"]["val"]["bal"][lo:]) - np.array(outer_of(m)[lo:]))
                          for m in models]))


def jaccard(a, b):
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def main():
    if os.path.exists(os.path.join(OUT, "model_comparison.csv")):
        raise SystemExit("analysis already written; not overwritten")
    v2 = load_v2()
    if len(v2) < N_SPLITS:
        raise SystemExit(f"grid incomplete: {len(v2)}/{N_SPLITS} splits")
    os.makedirs(OUT, exist_ok=True)
    y = np.load(os.path.join(ROOT, "derivatives", "model_ready", "y.npy")).astype(int)
    c0_src, c0 = A1.from_grid("C0")
    srcs = {r: {k: (np.array(v["test_idx"]), np.array(v["pred"][r])) for k, v in v2.items()}
            for r in ("FIXED", "ES", "FIXED_BAL", "LAST")}
    srcs["C0"] = c0_src
    srcs["TABPFN"] = A1.from_records(os.path.join(ROOT, "derivatives", "model_training", "tabpfn",
                                                  "cv116_tabpfn_svmrfe50.json"))
    keys = sorted(v2)
    for name, s in srcs.items():                                        # identical outer folds everywhere
        assert set(keys) <= set(s), name
        assert all(np.array_equal(np.sort(s[k][0]), np.sort(v2[k]["test_idx"])) for k in keys), name
        srcs[name] = {k: s[k] for k in keys}
    seeds = A1.full_seeds(srcs["FIXED"])

    # ---------------------------------------------------------------- held-out comparison
    rows, cms, fold_rows = [], {}, []
    for m in ORDER:
        st, cm, _ = A1.summarize(srcs[m], y, seeds)
        cms[m] = cm
        fb, ff, fa = [], [], []
        for (s, f) in keys:
            te, pr = srcs[m][(s, f)]
            ev = K.H.evaluate(y[te], pr)
            fb.append(ev["balanced_accuracy"])
            ff.append(ev["macro_f1"])
            fa.append(ev["accuracy"])
            fold_rows.append({"model": m, "seed": s, "fold": f, "accuracy": ev["accuracy"],
                              "balanced_accuracy": ev["balanced_accuracy"], "macro_f1": ev["macro_f1"],
                              **{f"recall_{g}": v for g, v in ev["per_class_recall"].items()}})
        rows.append({"model": m, "description": NAMES[m], "seeds": st["seeds"], "predictions": st["predictions"],
                     "accuracy": st["accuracy"], "accuracy_seed_sd": st["acc_sd"],
                     "accuracy_ci": json.dumps(np.round(st["acc_ci"], 4).tolist()),
                     "balanced_accuracy": st["bal"], "bal_seed_sd": st["bal_sd"], "bal_fold_sd": st["bal_fold_sd"],
                     "bal_ci": json.dumps(np.round(st["bal_ci"], 4).tolist()),
                     "bal_fold_min": float(np.min(fb)), "bal_fold_max": float(np.max(fb)),
                     "macro_f1": st["f1"], "macro_f1_seed_sd": st["f1_sd"], "macro_f1_fold_sd": float(np.std(ff)),
                     "accuracy_fold_sd": float(np.std(fa)), "qwk": st["qwk"],
                     **{f"recall_{g}": v for g, v in st["recall"].items()},
                     **{f"predicted_share_{g}": v for g, v in st["pred_share"].items()}})
    comp = pd.DataFrame(rows)
    comp.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(os.path.join(OUT, "fold_level.csv"), index=False)
    cm_rows = [{"model": m, "true": STAGES[a], **{f"pred_{STAGES[b]}": int(cms[m][a, b]) for b in range(NS)}}
               for m in ORDER for a in range(NS)]
    pd.DataFrame(cm_rows).to_csv(os.path.join(OUT, "confusion_matrices.csv"), index=False)

    pairs = [("FIXED", "C0"), ("ES", "C0"), ("FIXED", "ES"), ("FIXED", "TABPFN"), ("ES", "TABPFN"),
             ("C0", "TABPFN"), ("FIXED_BAL", "FIXED"), ("LAST", "FIXED")]
    fl = folds.set_index(["model", "seed", "fold"])
    prs = []
    for a, b in pairs:
        p = A1.paired(srcs[a], srcs[b], y, seeds)
        row = {"A": a, "B": b, "bal_diff": p["diff"], "bal_diff_ci": json.dumps(np.round(p["ci"], 4).tolist()),
               "seeds_A_better": p["seeds_better"], "n_seeds": p["n_seeds"], "verdict": p["verdict"]}
        for mt in ("balanced_accuracy", "macro_f1", "accuracy"):
            dd = np.array([fl.loc[(a, s, f), mt] - fl.loc[(b, s, f), mt] for (s, f) in keys])
            row[f"{mt}_split_mean_diff"] = float(dd.mean())
            row[f"{mt}_splits_up"] = int((dd > 0).sum())
            row[f"{mt}_splits_down"] = int((dd < 0).sum())
        n_up, n_dn = row["balanced_accuracy_splits_up"], row["balanced_accuracy_splits_down"]
        row["bal_sign_test_p"] = float(binomtest(n_up, n_up + n_dn).pvalue) if n_up + n_dn else 1.0
        prs.append(row)
    pr = pd.DataFrame(prs)
    pr.to_csv(os.path.join(OUT, "paired_comparisons.csv"), index=False)

    # ---------------------------------------------------------------- selection, inner vs held-out, leakage check
    sel_rows, gap_rows, feat_rows = [], [], []
    for k in keys:
        r, r0 = v2[k], c0[k]
        sel_rows.append({"seed": k[0], "fold": k[1], "fixed_epoch": r["epochs"]["FIXED"][0] + 1,
                         "fixed_bal_epoch": r["epochs"]["FIXED_BAL"][0] + 1,
                         "es_epochs": json.dumps([e + 1 for e in r["epochs"]["ES"]]),
                         "es_epoch_median": float(np.median(r["epochs"]["ES"])) + 1,
                         "c0_es_epochs": json.dumps([m["best_epoch"] + 1 for m in r0["models"]]),
                         **{f"inner_oof_bal_{x}": r["inner_oof_bal"][x] for x in ("ES", "FIXED", "FIXED_BAL", "LAST")},
                         **{f"heldout_bal_{x}": r["heldout"][x]["balanced_accuracy"] for x in
                            ("ES", "FIXED", "FIXED_BAL", "LAST")}})
        g2 = no_selection_gap(r["inner_models"], lambda m: m["outer_curves"]["bal"])
        g1 = no_selection_gap(r0["models"], lambda m: m["curves"]["outer"]["bal"])
        es2 = float(np.mean([im["curves"]["val"]["bal"][im["best_epoch"]] - im["outer_curves"]["bal"][im["best_epoch"]]
                             for im in r["inner_models"]]))
        es1 = float(np.mean([m["curves"]["val"]["bal"][m["best_epoch"]] - m["curves"]["outer"]["bal"][m["best_epoch"]]
                             for m in r0["models"]]))
        gap_rows.append({"seed": k[0], "fold": k[1], "no_selection_gap_v2": g2, "no_selection_gap_c0": g1,
                         "best_epoch_gap_v2": es2, "best_epoch_gap_c0": es1})
        sups = [im["support"] for im in r["inner_models"]]
        feat_rows.append({"seed": k[0], "fold": k[1],
                          "inner_pairwise_jaccard": float(np.mean([jaccard(sups[i], sups[j])
                                                                   for i in range(4) for j in range(i + 1, 4)])),
                          "inner_vs_c0_outer_jaccard": float(np.mean([jaccard(s_, r0["support"]) for s_ in sups])),
                          "K_mean": float(np.mean([im["K"] for im in r["inner_models"]]))})
    sel = pd.DataFrame(sel_rows)
    sel.to_csv(os.path.join(OUT, "selected_epochs.csv"), index=False)
    gaps = pd.DataFrame(gap_rows)
    gaps.to_csv(os.path.join(OUT, "inner_vs_heldout_gap.csv"), index=False)
    feats = pd.DataFrame(feat_rows)
    feats.to_csv(os.path.join(OUT, "feature_stability.csv"), index=False)

    def split_ci(v, seed=0):
        v = np.asarray(v)
        bs = np.random.default_rng(seed).choice(v, size=(2000, v.size), replace=True).mean(1)
        return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))

    # ---------------------------------------------------------------- curves (mean over all inner models)
    cur = {}
    for tag, models, ok in (("v2", [im for r in v2.values() for im in r["inner_models"]], "outer_curves"),
                            ("C0", [m for r in c0.values() for m in r["models"]], None)):
        val = np.array([m["curves"]["val"]["bal"] for m in models])
        out = np.array([m[ok]["bal"] if ok else m["curves"]["outer"]["bal"] for m in models])
        vl = np.array([m["curves"]["val"]["loss"] for m in models])
        tl = np.array([m["curves"]["train"]["loss"] for m in models])
        cur[tag] = pd.DataFrame({"epoch": np.arange(1, val.shape[1] + 1), "inner_val_bal": val.mean(0),
                                 "heldout_bal": out.mean(0), "gap": (val - out).mean(0),
                                 "inner_val_loss": vl.mean(0), "train_loss": tl.mean(0)})
        cur[tag].to_csv(os.path.join(OUT, f"curves_{tag}.csv"), index=False)

    # ---------------------------------------------------------------- figures
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": P.AXIS, "axes.labelcolor": P.INK2, "xtick.color": P.INK2,
                         "ytick.color": P.INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "grid.color": P.GRID, "figure.facecolor": P.SURFACE, "axes.facecolor": P.SURFACE})
    show = ["C0", "ES", "FIXED", "TABPFN"]
    col = {"C0": P.MUTED, "ES": P.SERIES[1], "FIXED": P.SERIES[0], "TABPFN": P.SERIES[2]}
    lab = {"C0": "C0 (v1)", "ES": "v2 nested\nearly stopping", "FIXED": "v2 fixed epoch\n(primary)", "TABPFN": "TabPFN k=50"}
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1.3, 1]})
    for i, m in enumerate(show):
        v = folds[folds.model == m].balanced_accuracy.to_numpy()
        axs[0].scatter(i + np.random.default_rng(i).uniform(-0.14, 0.14, v.size), v, s=14, color=col[m], alpha=0.55,
                       edgecolor="none")
        rr = comp[comp.model == m].iloc[0]
        lo, hi = json.loads(rr.bal_ci)
        axs[0].errorbar([i + 0.3], [rr.balanced_accuracy], yerr=[[rr.balanced_accuracy - lo], [hi - rr.balanced_accuracy]],
                        fmt="D", ms=6, color=col[m], ecolor=col[m], elinewidth=2, capsize=0)
        axs[0].text(i + 0.38, rr.balanced_accuracy, f"{100 * rr.balanced_accuracy:.1f}%", fontsize=7.5, color=P.INK,
                    va="center")
    axs[0].axhline(1 / 6, color=P.MUTED, ls="--", lw=1)
    axs[0].text(len(show) - 0.5, 1 / 6, "chance", color=P.MUTED, fontsize=7, va="bottom", ha="right")
    axs[0].set_xticks(range(len(show)), [lab[m] for m in show])
    axs[0].set_ylabel("held-out balanced accuracy")
    axs[0].set_title(f"Held-out folds (dots: {len(keys)} splits; diamond: seed-pooled mean, 95% CI)", loc="left",
                     fontsize=9)
    axs[0].yaxis.grid(True)
    for tag, c, nm in (("C0", P.MUTED, "C0 (v1): selection fitted with the validation subjects"),
                       ("v2", P.SERIES[0], "v2: selection fitted inside each inner fold")):
        axs[1].plot(cur[tag].epoch, 100 * cur[tag].gap, color=c, lw=2, label=nm)
    axs[1].axhline(0, color=P.AXIS, lw=1)
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("inner-validation minus held-out\nbalanced accuracy (points)")
    axs[1].set_title("Optimism of the inner-validation signal, no epoch selected", loc="left", fontsize=9)
    axs[1].legend(fontsize=7, frameon=False)
    axs[1].yaxis.grid(True)
    fig.savefig(os.path.join(OUT, "fig_heldout_and_leakage.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 3.4))
    bins = np.arange(0.5, 61.5, 2)
    axs[0].hist(sel.fixed_epoch, bins=bins, color=P.SERIES[0], edgecolor=P.SURFACE, linewidth=1)
    axs[0].axvline(30, color=P.MUTED, ls="--", lw=1)
    axs[0].text(30.5, axs[0].get_ylim()[1] * 0.92, "epoch 30 (v1 candidate)", color=P.MUTED, fontsize=7)
    axs[0].set_xlabel("epoch chosen by inner-CV loss (one per outer split)")
    axs[0].set_ylabel("outer splits")
    axs[0].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axs[0].set_title(f"Fixed epoch selected in each of the {len(keys)} splits", loc="left", fontsize=9)
    axs[0].yaxis.grid(True)
    axs[1].plot(cur["v2"].epoch, cur["v2"].inner_val_loss, color=P.SERIES[0], lw=2, label="inner-validation loss")
    axs[1].plot(cur["v2"].epoch, cur["v2"].train_loss, color=P.SERIES[1], lw=2, label="training loss")
    axs[1].axhline(np.log(6), color=P.MUTED, ls="--", lw=1)
    axs[1].text(60, np.log(6), "uniform guess (ln 6)", color=P.MUTED, fontsize=7, va="bottom", ha="right")
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("cross-entropy")
    axs[1].set_title(f"v2 loss curves (mean over {4 * len(keys)} inner models)", loc="left", fontsize=9)
    axs[1].legend(fontsize=7, frameon=False)
    axs[1].yaxis.grid(True)
    fig.savefig(os.path.join(OUT, "fig_epoch_selection.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------------- summary text
    L = ["HELD-OUT RESULTS (outer test folds; 25 identical splits, 5 seeds; rs-fMRI only)"]
    for r in comp.to_dict("records"):
        L.append(f"  {r['model']:9s} {r['description']}: accuracy {100 * r['accuracy']:.1f}% (seed sd "
                 f"{100 * r['accuracy_seed_sd']:.1f}), balanced {100 * r['balanced_accuracy']:.1f}% (seed sd "
                 f"{100 * r['bal_seed_sd']:.1f}, fold sd {100 * r['bal_fold_sd']:.1f}, range {100 * r['bal_fold_min']:.0f}-"
                 f"{100 * r['bal_fold_max']:.0f}, CI {r['bal_ci']}), macro-F1 {100 * r['macro_f1']:.1f}% (fold sd "
                 f"{100 * r['macro_f1_fold_sd']:.1f}); recall " + ", ".join(f"{g} {100 * r['recall_' + g]:.0f}" for g in STAGES)
                 + "; predicted AD share " + f"{100 * r['predicted_share_AD']:.0f}%")
    L.append("\nPAIRED (A - B; balanced accuracy, seed-pooled with bootstrap CI; split-level counts and sign test)")
    for r in pr.to_dict("records"):
        L.append(f"  {r['A']} vs {r['B']}: {100 * r['bal_diff']:+.1f} CI {r['bal_diff_ci']}, seeds A better "
                 f"{r['seeds_A_better']}/{r['n_seeds']}, splits up/down {r['balanced_accuracy_splits_up']}/"
                 f"{r['balanced_accuracy_splits_down']} (sign p {r['bal_sign_test_p']:.2f}); macro-F1 "
                 f"{100 * r['macro_f1_split_mean_diff']:+.1f}, accuracy {100 * r['accuracy_split_mean_diff']:+.1f} -> {r['verdict']}")
    L.append("\nEPOCH SELECTION (inner CV only)")
    L.append(f"  fixed epoch (loss): median {sel.fixed_epoch.median():.0f}, IQR {sel.fixed_epoch.quantile(.25):.0f}-"
             f"{sel.fixed_epoch.quantile(.75):.0f}, range {sel.fixed_epoch.min()}-{sel.fixed_epoch.max()}; within 25-35: "
             f"{int(sel.fixed_epoch.between(25, 35).sum())}/25; values {sorted(sel.fixed_epoch.tolist())}")
    L.append(f"  fixed epoch (balanced acc.): median {sel.fixed_bal_epoch.median():.0f}, IQR "
             f"{sel.fixed_bal_epoch.quantile(.25):.0f}-{sel.fixed_bal_epoch.quantile(.75):.0f}, range "
             f"{sel.fixed_bal_epoch.min()}-{sel.fixed_bal_epoch.max()}")
    es_all = np.concatenate([np.array(json.loads(x)) for x in sel.es_epochs])
    c0_all = np.concatenate([np.array(json.loads(x)) for x in sel.c0_es_epochs])
    L.append(f"  nested early stopping, best epochs (100 inner models): median {np.median(es_all):.0f}, IQR "
             f"{np.percentile(es_all, 25):.0f}-{np.percentile(es_all, 75):.0f}, <= 5: {100 * np.mean(es_all <= 5):.0f}% "
             f"(C0: median {np.median(c0_all):.0f}, IQR {np.percentile(c0_all, 25):.0f}-{np.percentile(c0_all, 75):.0f})")
    L.append("\nINNER (validation) vs HELD-OUT balanced accuracy, mean over splits")
    for x in ("ES", "FIXED", "FIXED_BAL", "LAST"):
        a, b = sel[f"inner_oof_bal_{x}"], sel[f"heldout_bal_{x}"]
        L.append(f"  {x}: inner {100 * a.mean():.1f}% vs held-out {100 * b.mean():.1f}% (optimism {100 * (a - b).mean():+.1f}, "
                 f"CI {np.round(100 * np.array(split_ci(a - b)), 1).tolist()})")
    L.append("\nLEAKAGE CHECK: inner-val minus held-out balanced accuracy of the same single models, epochs 10-60, "
             "no epoch selected (should be ~0 when the validation subjects are truly unseen)")
    for tag in ("c0", "v2"):
        v = gaps[f"no_selection_gap_{tag}"]
        L.append(f"  {tag}: {100 * v.mean():+.1f} pts (CI over splits {np.round(100 * np.array(split_ci(v)), 1).tolist()}, "
                 f"positive in {int((v > 0).sum())}/25 splits)")
    dv = gaps.no_selection_gap_v2 - gaps.no_selection_gap_c0
    L.append(f"  v2 - c0: {100 * dv.mean():+.1f} pts (CI {np.round(100 * np.array(split_ci(dv)), 1).tolist()}, "
             f"lower in {int((dv < 0).sum())}/25 splits)")
    for tag in ("c0", "v2"):
        v = gaps[f"best_epoch_gap_{tag}"]
        L.append(f"  at each model's early-stopping epoch, {tag}: {100 * v.mean():+.1f} pts")
    L.append("\nFEATURE SELECTION STABILITY (SVM-RFE k=50)")
    L.append(f"  between the 4 inner fits of a split: Jaccard {feats.inner_pairwise_jaccard.mean():.2f}; inner fit vs C0's "
             f"outer-training fit: {feats.inner_vs_c0_outer_jaccard.mean():.2f}; regions per model {feats.K_mean.mean():.1f}")
    rt = [v["seconds"] for v in v2.values()]
    L.append(f"\nRUNTIME: {len(rt)} splits, median {np.median(rt):.0f}s, total {sum(rt) / 60:.1f} min")
    open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "splits": len(keys), "seeds": seeds,
               "primary_rule": "FIXED", "selection_uses_heldout": False,
               "sources": {"v2": "derivatives/experiments/transformer_nested_v2/grid",
                           "C0": "derivatives/experiments/transformer_diagnostics_v1/grid/C0",
                           "TABPFN": "derivatives/model_training/tabpfn/cv116_tabpfn_svmrfe50.json"}},
              open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
    print("\n".join(L))


if __name__ == "__main__":
    main()
