"""
Analysis for experiment v4 (PLAN.md): TabPFN on all 492 mALFF + mReHo features vs the TabPFN baseline (SVM-RFE
k = 50 from 984 features), on the same 25 outer splits. Reads saved predictions only; refuses to overwrite.

Statistics are v3's (analyze_v3.compare): paired class-stratified subject bootstrap of the seed-averaged
difference, seeds agreeing, splits up/down with a sign test, and the corrected resampled t-test (Nadeau-Bengio).
Added here: bootstrap CIs for macro-F1, per-class recall differences, and the pre-declared intermediate-stage
recall (SMC, EMCI, MCI, LMCI; and EMCI, MCI, LMCI).

Run:  python scripts/experiments/tabpfn_local492_v4/analyze_local492.py
"""
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "biomarker_ablation_v3"))
import analyze_v3 as V3A  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

A1, K, P, T = V3A.A1, V3A.K, V3A.P, V3A.T
EXP = os.path.join(K.ROOT, "derivatives", "experiments", "tabpfn_local492_v4")
OUT = os.path.join(EXP, "analysis")
V3 = os.path.join(K.ROOT, "derivatives", "experiments", "biomarker_ablation_v3", "grid", "A_all4")
STAGES, NS, SEEDS = K.STAGES, K.NS, list(K.SEEDS)
INTER = {"intermediate (SMC, EMCI, MCI, LMCI)": [1, 2, 3, 4], "MCI spectrum (EMCI, MCI, LMCI)": [2, 3, 4]}
NAMES = {"BASE": "TabPFN baseline: SVM-RFE k=50 of 984 features",
         "L492": "TabPFN all 492 mALFF + mReHo features, no selection",
         "BASE_BAL": "baseline + class balancing", "L492_BAL": "492 features + class balancing"}
V3A.LABEL.update(NAMES)
B = 2000


def load(path_glob):
    return {(r["seed"], r["fold"]): r for r in (json.load(open(f)) for f in glob.glob(path_glob))}


def oof_matrix(src, n=116):
    return np.vstack([A1.oof(src, s, n) for s in SEEDS])


def boot_idx(y, rng):
    return [np.concatenate([rng.choice(np.flatnonzero(y == g), (y == g).sum(), replace=True) for g in range(NS)])
            for _ in range(B)]


def recall_mean(y, M, classes):
    return float(np.mean([np.mean([(m[y == g] == g).mean() for g in classes]) for m in M]))


def f1_mean(y, M):
    return float(np.mean([K.H.evaluate(y, m)["macro_f1"] for m in M]))


def main():
    if os.path.exists(os.path.join(OUT, "summary.txt")):
        raise SystemExit("analysis already written; not overwritten")
    y = np.load(os.path.join(K.ROOT, "derivatives", "model_ready", "y.npy")).astype(int)
    new, base = load(os.path.join(EXP, "grid", "*.json")), load(os.path.join(V3, "*.json"))
    if len(new) < 25:
        raise SystemExit(f"incomplete: {len(new)}/25 splits")
    os.makedirs(OUT, exist_ok=True)
    keys = sorted(base)
    orig = A1.from_records(T.TABPFN_BASELINE)
    assert sorted(new) == keys
    assert all(new[k]["test_idx"] == base[k]["test_idx"] == list(orig[k][0]) for k in keys), "outer splits differ"
    assert all(np.array_equal(np.array(base[k]["pred"]), orig[k][1]) for k in keys), "baseline records differ"
    srcs = {"BASE": V3A.as_src(base), "L492": V3A.as_src(new), "BASE_BAL": V3A.as_src(base, "pred_balanced"),
            "L492_BAL": V3A.as_src(new, "pred_balanced")}

    # ------------------------------------------------------------- per-model metrics (+ macro-F1 CI)
    rng_idx = boot_idx(y, np.random.default_rng(0))
    rows, cms, folds = [], {}, []
    for m in srcs:
        row, cm, fm = V3A.describe(m, srcs[m], y)
        M = oof_matrix(srcs[m])
        f1b = [f1_mean(y[i], M[:, i]) for i in rng_idx]
        row["macro_f1_ci_low"], row["macro_f1_ci_high"] = np.percentile(f1b, [2.5, 97.5])
        st, _, _ = A1.summarize(srcs[m], y, SEEDS)
        row["acc_ci_low"], row["acc_ci_high"] = st["acc_ci"]
        for name, cls in INTER.items():
            row[name] = recall_mean(y, M, cls)
        rows.append(row)
        cms[m] = cm
        fm.insert(0, "model", m)
        folds.append(fm)
    desc = pd.DataFrame(rows).set_index("model")
    desc.to_csv(os.path.join(OUT, "metrics.csv"))
    pd.concat(folds).to_csv(os.path.join(OUT, "fold_level.csv"), index=False)
    pd.DataFrame([{"model": m, "true": STAGES[a], **{f"pred_{STAGES[b]}": int(cms[m][a, b]) for b in range(NS)}}
                  for m in cms for a in range(NS)]).to_csv(os.path.join(OUT, "confusion_matrices.csv"), index=False)

    # ------------------------------------------------------------- paired comparisons
    n_te = np.mean([len(base[k]["test_idx"]) for k in keys])
    ratio = n_te / (116 - n_te)
    pairs = [("L492", "BASE", "primary"), ("L492_BAL", "BASE", "secondary"), ("L492_BAL", "BASE_BAL", "secondary")]
    cmp_rows, rec_rows = [], []
    for a, b, role in pairs:
        c = V3A.compare(a, b, srcs, y, ratio)
        Ma, Mb = oof_matrix(srcs[a]), oof_matrix(srcs[b])
        for name, cls in INTER.items():
            d = [recall_mean(y[i], Ma[:, i], cls) - recall_mean(y[i], Mb[:, i], cls) for i in rng_idx]
            c[f"{name} diff"] = recall_mean(y, Ma, cls) - recall_mean(y, Mb, cls)
            c[f"{name} ci"] = np.round(np.percentile(d, [2.5, 97.5]), 4).tolist()
        fd = [f1_mean(y[i], Ma[:, i]) - f1_mean(y[i], Mb[:, i]) for i in rng_idx]
        c["macro_f1_diff"] = f1_mean(y, Ma) - f1_mean(y, Mb)
        c["macro_f1_diff_ci"] = np.round(np.percentile(fd, [2.5, 97.5]), 4).tolist()
        sa = [K.H.evaluate(y, Ma[s])["balanced_accuracy"] - K.H.evaluate(y, Mb[s])["balanced_accuracy"] for s in range(5)]
        c["per_seed_bal_diff"] = np.round(sa, 4).tolist()
        c["splits_tied"] = 25 - c["splits_up"] - c["splits_down"]
        c["consistent_improvement"] = bool(c["reliably_different"] and c["bal_diff"] > 0 and c["splits_up"] > c["splits_down"])
        c["role"] = role
        cmp_rows.append(c)
        for g in range(NS):
            d = [recall_mean(y[i], Ma[:, i], [g]) - recall_mean(y[i], Mb[:, i], [g]) for i in rng_idx]
            rec_rows.append({"A": a, "B": b, "stage": STAGES[g], "recall_A": recall_mean(y, Ma, [g]),
                             "recall_B": recall_mean(y, Mb, [g]), "diff": recall_mean(y, Ma, [g]) - recall_mean(y, Mb, [g]),
                             "ci_low": np.percentile(d, 2.5), "ci_high": np.percentile(d, 97.5)})
    cmp = pd.DataFrame(cmp_rows)
    cmp.to_csv(os.path.join(OUT, "paired_comparisons.csv"), index=False)
    recd = pd.DataFrame(rec_rows)
    recd.to_csv(os.path.join(OUT, "per_class_recall_differences.csv"), index=False)
    fl = pd.concat(folds).set_index(["model", "seed", "fold"])
    split_diff = pd.DataFrame([{"seed": s, "fold": f, "bal_diff": fl.loc[("L492", s, f), "balanced_accuracy"]
                                - fl.loc[("BASE", s, f), "balanced_accuracy"],
                                "f1_diff": fl.loc[("L492", s, f), "macro_f1"] - fl.loc[("BASE", s, f), "macro_f1"]}
                               for s, f in keys])
    split_diff.to_csv(os.path.join(OUT, "per_split_differences.csv"), index=False)

    # ------------------------------------------------------------- figures
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": P.AXIS, "axes.labelcolor": P.INK2, "xtick.color": P.INK2,
                         "ytick.color": P.INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "grid.color": P.GRID, "figure.facecolor": P.SURFACE, "axes.facecolor": P.SURFACE})
    show = ["BASE", "L492", "L492_BAL"]
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3))
    for ax, m in zip(axs, show):
        Mn = cms[m] / cms[m].sum(1, keepdims=True)
        ax.imshow(Mn, cmap="Blues", vmin=0, vmax=0.6)
        for a in range(NS):
            for b in range(NS):
                ax.text(b, a, f"{100 * Mn[a, b]:.0f}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if Mn[a, b] > 0.35 else P.INK2)
        ax.set_xticks(range(NS), STAGES, fontsize=7, rotation=45)
        ax.set_yticks(range(NS), STAGES, fontsize=7)
        ax.set_title(NAMES[m], fontsize=7.5, loc="left", wrap=True)
        ax.set_xlabel("predicted", fontsize=7)
    axs[0].set_ylabel("true (row %, 5 seeds pooled)", fontsize=7)
    fig.savefig(os.path.join(OUT, "fig_confusion.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1, 1.2]})
    for i, s in enumerate(SEEDS):
        v = split_diff[split_diff.seed == s].bal_diff * 100
        axs[0].scatter(np.full(v.size, i) + np.random.default_rng(i).uniform(-0.12, 0.12, v.size), v, s=18,
                       color=P.SERIES[0], alpha=0.7, edgecolor="none")
        axs[0].scatter([i + 0.3], [100 * cmp_rows[0]["per_seed_bal_diff"][i]], marker="D", s=40, color=P.INK)
    axs[0].axhline(0, color=P.AXIS, lw=1)
    axs[0].set_xticks(range(5), [str(s) for s in SEEDS], fontsize=7)
    axs[0].set_ylabel("balanced accuracy difference (points)")
    axs[0].set_title("492 features minus baseline: each outer split (dots), each seed pooled (diamond)", loc="left", fontsize=8.5)
    axs[0].yaxis.grid(True)
    r = recd[(recd.A == "L492") & (recd.B == "BASE")]
    x = np.arange(NS)
    axs[1].bar(x - 0.2, 100 * r.recall_B, 0.38, color=P.MUTED, label=NAMES["BASE"])
    axs[1].bar(x + 0.2, 100 * r.recall_A, 0.38, color=P.SERIES[0], label=NAMES["L492"])
    for xi, (ra, rb) in enumerate(zip(r.recall_A, r.recall_B)):
        axs[1].text(xi + 0.2, 100 * ra + 1, f"{100 * ra:.0f}", ha="center", fontsize=7, color=P.INK)
        axs[1].text(xi - 0.2, 100 * rb + 1, f"{100 * rb:.0f}", ha="center", fontsize=7, color=P.INK2)
    axs[1].axhline(100 / 6, color=P.MUTED, ls="--", lw=1)
    axs[1].set_xticks(x, STAGES)
    axs[1].set_ylabel("recall (%, seed-pooled)")
    axs[1].set_title("Per-class recall", loc="left", fontsize=8.5)
    axs[1].legend(fontsize=7, frameon=False)
    axs[1].yaxis.grid(True)
    fig.savefig(os.path.join(OUT, "fig_paired_and_recall.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------- summary
    L = ["METRICS (seed-pooled out-of-fold, 25 identical outer splits; 95% CI = stratified subject bootstrap)"]
    for m, r_ in desc.iterrows():
        L.append(f"  {m:9s} {NAMES[m]}: balanced {100 * r_.balanced_accuracy:.1f} [{100 * r_.bal_ci_low:.1f}, "
                 f"{100 * r_.bal_ci_high:.1f}] (fold {100 * r_.bal_fold_mean:.1f} +/- {100 * r_.bal_fold_sd:.1f}); macro-F1 "
                 f"{100 * r_.macro_f1:.1f} [{100 * r_.macro_f1_ci_low:.1f}, {100 * r_.macro_f1_ci_high:.1f}]; accuracy "
                 f"{100 * r_.accuracy:.1f} [{100 * r_.acc_ci_low:.1f}, {100 * r_.acc_ci_high:.1f}]; recall "
                 + " ".join(f"{g} {100 * r_['recall_' + g]:.0f}" for g in STAGES)
                 + "; " + "; ".join(f"{k} {100 * r_[k]:.1f}" for k in INTER))
    L.append("\nPAIRED")
    for c in cmp_rows:
        L.append(f"  [{c['role']}] {c['A']} vs {c['B']}: balanced {100 * c['bal_diff']:+.1f} "
                 f"[{100 * c['bal_diff_ci_low']:+.1f}, {100 * c['bal_diff_ci_high']:+.1f}], per seed "
                 f"{[round(100 * v, 1) for v in c['per_seed_bal_diff']]}, seeds A better {c['seeds_A_better']}/5, splits "
                 f"up/down/tied {c['splits_up']}/{c['splits_down']}/{c['splits_tied']} (sign p {c['sign_p']:.2f}), corrected t "
                 f"p {c['corrected_t_p']:.3f}; macro-F1 {100 * c['macro_f1_diff']:+.1f} {np.round(100 * np.array(c['macro_f1_diff_ci']), 1).tolist()}; "
                 + "; ".join(f"{k} {100 * c[k + ' diff']:+.1f} {np.round(100 * np.array(c[k + ' ci']), 1).tolist()}" for k in INTER)
                 + f" -> reliably different: {c['reliably_different']}; consistent improvement: {c['consistent_improvement']}")
    L.append("\nPER-CLASS RECALL DIFFERENCES (L492 - BASE) [95% CI]")
    for r_ in recd[(recd.A == "L492") & (recd.B == "BASE")].itertuples():
        L.append(f"  {r_.stage}: {100 * r_.recall_B:.0f} -> {100 * r_.recall_A:.0f} ({100 * r_.diff:+.1f} "
                 f"[{100 * r_.ci_low:+.1f}, {100 * r_.ci_high:+.1f}])")
    L.append(f"\nRUNTIME: {sum(r_['seconds'] for r_ in new.values()) / 60:.1f} min")
    open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "baseline_records_match_original": True,
               "splits_identical": True, "primary": {k: (v if not isinstance(v, np.generic) else v.item())
                                                     for k, v in cmp_rows[0].items()}},
              open(os.path.join(OUT, "manifest.json"), "w"), indent=2, default=str)
    print("\n".join(L))


if __name__ == "__main__":
    main()
