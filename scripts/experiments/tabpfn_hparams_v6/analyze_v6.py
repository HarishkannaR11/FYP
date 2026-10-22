"""
Analysis for the TabPFN v2 hyperparameter study (v6; PLAN.md). Reads saved predictions only; refuses to overwrite.

Verification: identical outer splits and labels; each configuration's resolved settings (from the fitted models);
no unexpected warnings; SVM-RFE support identical to the baseline's; C1 reproduces the stored baseline C0.
Comparisons: C2 and C3 vs C0 (Holm across the two), C1 vs C0 as the reproduction check; statistics as in v5
(seed-pooled metrics, class-stratified subject bootstrap, seeds, splits, corrected resampled t-test).

Run:  python scripts/experiments/tabpfn_hparams_v6/analyze_v6.py
"""
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "ordinal_transformer_v1"))
import analyze_ordinal as AO  # noqa: E402  (paired statistics; imports analyze_v3)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

V3A, A1, K, P, T = AO.V3A, AO.A1, AO.K, AO.P, AO.T
ROOT = K.ROOT
EXP = os.path.join(ROOT, "derivatives", "experiments", "tabpfn_hparams_v6")
OUT = os.path.join(EXP, "analysis")
STAGES, NS, SEEDS = K.STAGES, K.NS, list(K.SEEDS)
NAMES = {"C0": "C0 stored baseline (defaults: 8 estimators, T 0.9)", "C1": "C1 explicit defaults (reproduction)",
         "C2": "C2 n_estimators = 32", "C3": "C3 average_before_softmax = True"}
EXPECT = {"C1": {"n_estimators_": 8, "softmax_temperature_": 0.9, "n_ensemble_configs": 8, "average_before_softmax": False},
          "C2": {"n_estimators_": 32, "softmax_temperature_": 0.9, "n_ensemble_configs": 32, "average_before_softmax": False},
          "C3": {"n_estimators_": 8, "softmax_temperature_": 0.9, "n_ensemble_configs": 8, "average_before_softmax": True}}
AO.NAMES.update(NAMES)
V3A.LABEL.update(NAMES)


def main():
    if os.path.exists(os.path.join(OUT, "summary.txt")):
        raise SystemExit("analysis already written; not overwritten")
    y = np.load(os.path.join(ROOT, "derivatives", "model_ready", "y.npy")).astype(int)
    base = {(r["seed"], r["fold"]): r for r in json.load(open(T.TABPFN_BASELINE))["records"]}
    runs = {c: {(r["seed"], r["fold"]): r for r in (json.load(open(f)) for f in glob.glob(os.path.join(EXP, "grid", c, "*.json")))}
            for c in ("C1", "C2", "C3")}
    missing = {c: 25 - len(v) for c, v in runs.items() if len(v) < 25}
    if missing:
        raise SystemExit(f"incomplete: {missing}")
    os.makedirs(OUT, exist_ok=True)
    keys = sorted(base)

    # ------------------------------------------------------------- verification
    V = []

    def check(name, ok, detail=""):
        V.append({"check": name, "pass": bool(ok), "detail": detail})

    check("labels: class counts 20/20/20/16/20/20", np.bincount(y).tolist() == [20, 20, 20, 16, 20, 20])
    for c, recs in runs.items():
        check(f"{c}: 25/25 outer test sets and stored labels identical to C0",
              all(recs[k]["test_idx"] == base[k]["test_index"] and recs[k]["true"] == base[k]["true"] for k in keys))
        check(f"{c}: SVM-RFE support identical to C0 in 25/25 splits", all(recs[k]["support"] == base[k]["support"] for k in keys))
        bad = [k for k in keys if any(recs[k]["resolved"][a] != v for a, v in EXPECT[c].items()) or recs[k]["resolved"]["balance_probabilities"]]
        check(f"{c}: resolved settings as pre-registered in 25/25 splits {EXPECT[c]}", not bad, f"mismatches {bad}")
        w = sorted({m for k in keys for m in recs[k]["warnings"]})
        check(f"{c}: no warnings raised by TabPFN", not w, "; ".join(w)[:300])
    same = sum(runs["C1"][k]["pred"] == base[k]["pred"] for k in keys)
    pdiff = max(np.abs(np.array(runs["C1"][k]["probs"]) - np.array(base[k]["probs"])).max() for k in keys)
    check("C1 reproduces the stored baseline (predictions identical)", same == 25, f"{same}/25 splits identical; max probability difference {pdiff:.1e}")
    json.dump(V, open(os.path.join(OUT, "verification.json"), "w"), indent=2)

    # ------------------------------------------------------------- metrics
    srcs = {"C0": {k: (np.array(base[k]["test_index"]), np.array(base[k]["pred"])) for k in keys}}
    for c, recs in runs.items():
        srcs[c] = {k: (np.array(recs[k]["test_idx"]), np.array(recs[k]["pred"])) for k in keys}
    rng = np.random.default_rng(0)
    boots = [np.concatenate([rng.choice(np.flatnonzero(y == g), (y == g).sum(), replace=True) for g in range(NS)])
             for _ in range(2000)]
    n_te = np.mean([len(base[k]["test_index"]) for k in keys])
    ratio = n_te / (116 - n_te)
    rows, cms, folds = [], {}, []
    for m in srcs:
        row, cm, _ = V3A.describe(m, srcs[m], y)
        M = AO.oof(srcs[m])
        for key in ("f1", "acc", "inter"):
            row[f"{key}_mean"] = AO.seed_avg(y, M, key)
            row[f"{key}_ci_low"], row[f"{key}_ci_high"] = np.percentile([AO.seed_avg(y, M, key, i) for i in boots], [2.5, 97.5])
        ft = AO.fold_table(srcs[m], y)
        row["f1_fold_mean"], row["f1_fold_sd"] = ft.f1.mean(), ft.f1.std(ddof=1)
        rows.append(row)
        cms[m] = cm
        ft.insert(0, "model", m)
        folds.append(ft)
    desc = pd.DataFrame(rows).set_index("model")
    desc.to_csv(os.path.join(OUT, "metrics.csv"))
    pd.concat(folds).to_csv(os.path.join(OUT, "fold_level.csv"), index=False)
    pd.DataFrame([{"model": m, "true": STAGES[a], **{f"pred_{STAGES[b]}": int(cms[m][a, b]) for b in range(NS)}}
                  for m in cms for a in range(NS)]).to_csv(os.path.join(OUT, "confusion_matrices.csv"), index=False)

    # ------------------------------------------------------------- paired comparisons and decisions
    plan = [("C1", "C0", "reproduction"), ("C2", "C0", "candidate"), ("C3", "C0", "candidate")]
    cmp = pd.DataFrame([{**AO.paired(a, b, srcs, y, boots, ratio, key, True), "role": role}
                        for a, b, role in plan for key in ("bal", "f1", "acc", "inter")])
    sel = (cmp.role == "candidate") & (cmp.metric == "bal")
    cmp["holm_p"] = np.nan
    cmp.loc[sel, "holm_p"] = V3A.holm(cmp.loc[sel, "corrected_t_p"].to_numpy())
    cmp.to_csv(os.path.join(OUT, "paired_comparisons.csv"), index=False)
    decisions = {}
    for a in ("C2", "C3"):
        q = cmp[(cmp.A == a) & (cmp.B == "C0")].set_index("metric")
        promising = bool(q.loc["bal", "diff"] >= 0.03 and q.loc["f1", "diff"] >= 0 and q.loc["inter", "diff"] >= -0.03)
        reliable = bool(promising and q.loc["bal", "ci_low"] > 0 and q.loc["bal", "seeds_A_better"] >= 4 and q.loc["bal", "holm_p"] < 0.05)
        decisions[a] = {"balanced_accuracy_change": round(100 * q.loc["bal", "diff"], 2), "macro_f1_change": round(100 * q.loc["f1", "diff"], 2),
                        "intermediate_recall_change": round(100 * q.loc["inter", "diff"], 2), "promising": promising,
                        "reliable": reliable, "decision": "ACCEPT" if reliable else "REJECT"}

    # ------------------------------------------------------------- figure
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": P.AXIS, "axes.labelcolor": P.INK2, "xtick.color": P.INK2,
                         "ytick.color": P.INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "grid.color": P.GRID, "figure.facecolor": P.SURFACE, "axes.facecolor": P.SURFACE})
    col = {"C0": P.MUTED, "C1": P.INK2, "C2": P.SERIES[0], "C3": P.SERIES[2]}
    short = {"C0": "C0 stored\nbaseline", "C1": "C1 explicit\ndefaults", "C2": "C2\n32 estimators", "C3": "C3 average\nbefore softmax"}
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw={"width_ratios": [1, 1.4]})
    for i, m in enumerate(srcs):
        r = desc.loc[m]
        axs[0].errorbar([i], [r.balanced_accuracy], yerr=[[r.balanced_accuracy - r.bal_ci_low], [r.bal_ci_high - r.balanced_accuracy]],
                        fmt="D", ms=6, color=col[m], elinewidth=2, capsize=0)
        axs[0].text(i + 0.1, r.balanced_accuracy, f"{100 * r.balanced_accuracy:.1f}", fontsize=7.5, color=P.INK, va="center")
    axs[0].axhline(1 / 6, color=P.MUTED, ls="--", lw=1)
    axs[0].set_xticks(range(4), [short[m] for m in srcs])
    axs[0].set_ylabel("balanced accuracy (seed-pooled, 95% CI)")
    axs[0].set_title("Same 25 outer splits", loc="left", fontsize=9)
    axs[0].yaxis.grid(True)
    x = np.arange(NS)
    for i, m in enumerate(("C0", "C2", "C3")):
        v = [100 * desc.loc[m, f"recall_{g}"] for g in STAGES]
        axs[1].bar(x + (i - 1) * 0.27, v, 0.26, color=col[m], label=short[m].replace("\n", " "))
    axs[1].axhline(100 / 6, color=P.MUTED, ls="--", lw=1)
    axs[1].set_xticks(x, STAGES)
    axs[1].set_ylabel("recall (%, seed-pooled)")
    axs[1].set_title("Per-stage recall", loc="left", fontsize=9)
    axs[1].legend(fontsize=7, frameon=False)
    axs[1].yaxis.grid(True)
    fig.savefig(os.path.join(OUT, "fig_hparams.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------- summary
    L = ["VERIFICATION"] + [f"  {'PASS' if v['pass'] else 'FAIL'}  {v['check']}" + (f"  -- {v['detail']}" if v['detail'] else "") for v in V]
    L.append("\nMETRICS (seed-pooled; 95% CI subject bootstrap)")
    for m, r in desc.iterrows():
        L.append(f"  {m} {NAMES[m]}: balanced {100 * r.balanced_accuracy:.1f} [{100 * r.bal_ci_low:.1f}, {100 * r.bal_ci_high:.1f}] "
                 f"(fold {100 * r.bal_fold_mean:.1f} +/- {100 * r.bal_fold_sd:.1f}); macro-F1 {100 * r.f1_mean:.1f} [{100 * r.f1_ci_low:.1f}, "
                 f"{100 * r.f1_ci_high:.1f}]; accuracy {100 * r.acc_mean:.1f} [{100 * r.acc_ci_low:.1f}, {100 * r.acc_ci_high:.1f}]; intermediate "
                 f"recall {100 * r.inter_mean:.1f}; recall " + " ".join(f"{g} {100 * r['recall_' + g]:.0f}" for g in STAGES))
    L.append("\nPAIRED (A - B)")
    for r in cmp.itertuples():
        L.append(f"  [{r.role}] {r.A} vs {r.B} {r.metric}: {100 * r.diff:+.1f} [{100 * r.ci_low:+.1f}, {100 * r.ci_high:+.1f}], seeds A better "
                 f"{r.seeds_A_better}/5, splits {r.splits_better}/{r.splits_worse}/{r.splits_tied}, corrected p {r.corrected_t_p:.3f}"
                 + (f", Holm {r.holm_p:.3f}" if not np.isnan(r.holm_p) else ""))
    L.append(f"\nDECISIONS: {json.dumps(decisions)}")
    rt = sum(r["seconds"] for recs in runs.values() for r in recs.values())
    L.append(f"RUNTIME: {rt / 60:.1f} min (C1 {sum(r['seconds'] for r in runs['C1'].values()) / 60:.1f}, C2 "
             f"{sum(r['seconds'] for r in runs['C2'].values()) / 60:.1f}, C3 {sum(r['seconds'] for r in runs['C3'].values()) / 60:.1f})")
    open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "verification_all_pass": all(v["pass"] for v in V),
               "decisions": decisions}, open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
    print("\n".join(L))


if __name__ == "__main__":
    main()
