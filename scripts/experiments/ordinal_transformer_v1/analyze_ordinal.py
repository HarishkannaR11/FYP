"""
Analysis for ordinal transformer v1 (PLAN.md). Reads saved predictions only; trains nothing; refuses to overwrite.

Compares the ordinal transformer (FIXED epoch rule, argmax; median rule secondary) with transformer v2 (FIXED,
argmax) on the same 25 outer splits; TabPFN is context only. Statistics follow the project's procedure: seed-pooled
out-of-fold metrics averaged over 5 seeds; class-stratified subject bootstrap (2000) for CIs and paired differences;
seeds agreeing; outer splits up/down/tied; corrected resampled t-test over the 25 splits (Nadeau-Bengio). Stage order
CN < SMC < EMCI < MCI < LMCI < AD (indices 0..5).

Run:  python scripts/experiments/ordinal_transformer_v1/analyze_ordinal.py
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
from scipy.stats import binomtest  # noqa: E402
from scipy.stats import t as tdist  # noqa: E402
from sklearn.metrics import precision_recall_fscore_support  # noqa: E402

A1, K, P, T = V3A.A1, V3A.K, V3A.P, V3A.T
ROOT = K.ROOT
EXP = os.path.join(ROOT, "derivatives", "experiments", "ordinal_transformer_v1")
OUT = os.path.join(EXP, "analysis")
V2 = os.path.join(ROOT, "derivatives", "experiments", "transformer_nested_v2", "grid")
STAGES, NS, SEEDS = K.STAGES, K.NS, list(K.SEEDS)
ORDER = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
INTER = [1, 2, 3, 4]
NAMES = {"T_v2": "Transformer v2 (6-class softmax)", "ORD": "Ordinal transformer (argmax, primary)",
         "ORD_MEDIAN": "Ordinal transformer (median rule, secondary)", "TABPFN": "TabPFN baseline (context only)",
         "T_v2_LAST": "Transformer v2, epoch 60", "ORD_LAST": "Ordinal transformer, epoch 60"}
V3A.LABEL.update(NAMES)
B = 2000


def load(pattern):
    return {(r["seed"], r["fold"]): r for r in (json.load(open(f)) for f in glob.glob(pattern))}


def oof(src):
    return np.vstack([A1.oof(src, s) for s in SEEDS])


def mets(y, pred):
    """All metrics of one set of predictions (higher is better except mae)."""
    e = K.H.evaluate(y, pred)
    rec = [(pred[y == g] == g).mean() for g in range(NS)]
    return {"bal": e["balanced_accuracy"], "f1": e["macro_f1"], "acc": e["accuracy"],
            "mae": float(np.abs(pred - y).mean()), "within1": float((np.abs(pred - y) <= 1).mean()),
            "inter": float(np.mean([rec[g] for g in INTER]))}


def seed_avg(y, M, key, idx=None):
    if idx is None:
        return float(np.mean([mets(y, m)[key] for m in M]))
    return float(np.mean([mets(y[idx], m[idx])[key] for m in M]))


def fold_table(src, y):
    return pd.DataFrame([{"seed": s, "fold": f, **mets(y[te], pr)} for (s, f), (te, pr) in sorted(src.items())])


def paired(a, b, srcs, y, boots, ratio, key, higher=True):
    Ma, Mb = oof(srcs[a]), oof(srcs[b])
    diff = seed_avg(y, Ma, key) - seed_avg(y, Mb, key)
    bd = [seed_avg(y, Ma, key, i) - seed_avg(y, Mb, key, i) for i in boots]
    lo, hi = np.percentile(bd, [2.5, 97.5])
    per_seed = [mets(y, Ma[s])[key] - mets(y, Mb[s])[key] for s in range(len(SEEDS))]
    fa, fb = fold_table(srcs[a], y), fold_table(srcs[b], y)
    d = (fa[key] - fb[key]).to_numpy()
    sgn = 1 if higher else -1
    up, dn = int((sgn * d > 0).sum()), int((sgn * d < 0).sum())
    J = d.size
    var = d.var(ddof=1)
    tval = d.mean() / np.sqrt((1 / J + ratio) * var) if var > 0 else 0.0
    p = float(2 * tdist.sf(abs(tval), J - 1))
    good = np.sort(sgn * d)[::-1]                              # most favourable first
    agree = int(sum(sgn * v > 0 for v in per_seed))           # seeds favouring A
    against = int(sum(sgn * v < 0 for v in per_seed))         # seeds favouring B
    better = (lo > 0) if higher else (hi < 0)
    worse = (hi < 0) if higher else (lo > 0)
    return {"A": a, "B": b, "metric": key, "higher_is_better": higher, "diff": diff, "ci_low": lo, "ci_high": hi,
            "per_seed_diff": np.round(per_seed, 4).tolist(), "seeds_A_better": agree, "splits_better": up,
            "splits_worse": dn, "splits_tied": J - up - dn, "sign_p": float(binomtest(up, up + dn).pvalue) if up + dn else 1.0,
            "corrected_t": float(tval), "corrected_t_p": p,
            "mean_favourable_split_diff_without_top3": float(good[3:].mean()),
            "reliably_better": bool(better and agree >= 4 and p < 0.05),
            "reliably_worse": bool(worse and against >= 4 and p < 0.05)}


def main():
    if os.path.exists(os.path.join(OUT, "summary.txt")):
        raise SystemExit("analysis already written; not overwritten")
    y = np.load(os.path.join(ROOT, "derivatives", "model_ready", "y.npy")).astype(int)
    ordr, v2 = load(os.path.join(EXP, "grid", "*.json")), load(os.path.join(V2, "*.json"))
    if len(ordr) < 25:
        raise SystemExit(f"incomplete: {len(ordr)}/25 splits")
    os.makedirs(OUT, exist_ok=True)
    keys = sorted(v2)
    tests = json.load(open(os.path.join(EXP, "tests", "test_results.json")))["results"]
    gate_ok = "gate: reference head reproduces transformer v2 exactly: True" in open(os.path.join(EXP, "run.log")).read()
    checks = {"tests_pass": all(r["pass"] for r in tests), "gate_reproduces_v2": gate_ok,
              "label_order": all(r["label_order"] == ORDER for r in ordr.values()) and list(STAGES) == ORDER,
              "splits_identical": sorted(ordr) == keys and all(ordr[k]["test_idx"] == v2[k]["test_idx"] for k in keys),
              "outer_labels_match": all(ordr[k]["true"] == v2[k]["true"] for k in keys)}
    srcs = {"T_v2": V3A.as_src(v2, "pred", "FIXED"), "ORD": V3A.as_src(ordr, "pred", "FIXED"),
            "ORD_MEDIAN": V3A.as_src(ordr, "pred_median", "FIXED"), "T_v2_LAST": V3A.as_src(v2, "pred", "LAST"),
            "ORD_LAST": V3A.as_src(ordr, "pred", "LAST"), "TABPFN": A1.from_records(T.TABPFN_BASELINE)}
    for n, s in srcs.items():
        assert sorted(s) == keys and all(np.array_equal(np.sort(s[k][0]), np.sort(v2[k]["test_idx"])) for k in keys), n

    rng = np.random.default_rng(0)
    boots = [np.concatenate([rng.choice(np.flatnonzero(y == g), (y == g).sum(), replace=True) for g in range(NS)])
             for _ in range(B)]
    n_te = np.mean([len(v2[k]["test_idx"]) for k in keys])
    ratio = n_te / (116 - n_te)

    # ------------------------------------------------------------- per-model metrics
    rows, cms, prf_rows, folds = [], {}, [], []
    for m in srcs:
        row, cm, _ = V3A.describe(m, srcs[m], y)
        M = oof(srcs[m])
        for key in ("f1", "acc", "mae", "within1", "inter"):
            vals = [seed_avg(y, M, key, i) for i in boots]
            row[f"{key}_mean"] = seed_avg(y, M, key)
            row[f"{key}_ci_low"], row[f"{key}_ci_high"] = np.percentile(vals, [2.5, 97.5])
        ft = fold_table(srcs[m], y)
        for key in ("mae", "within1"):
            row[f"{key}_fold_mean"], row[f"{key}_fold_sd"] = ft[key].mean(), ft[key].std(ddof=1)
        rows.append(row)
        cms[m] = cm
        ft.insert(0, "model", m)
        folds.append(ft)
        pr, rc, f1, sup = precision_recall_fscore_support(np.tile(y, len(SEEDS)), M.ravel(), labels=range(NS),
                                                          zero_division=0)
        prf_rows += [{"model": m, "stage": STAGES[g], "precision": pr[g], "recall": rc[g], "f1": f1[g],
                      "support_5_seeds": int(sup[g])} for g in range(NS)]
    desc = pd.DataFrame(rows).set_index("model")
    desc.to_csv(os.path.join(OUT, "metrics.csv"))
    prf = pd.DataFrame(prf_rows)
    prf.to_csv(os.path.join(OUT, "per_class_precision_recall_f1.csv"), index=False)
    pd.concat(folds).to_csv(os.path.join(OUT, "fold_level.csv"), index=False)
    pd.DataFrame([{"model": m, "true": STAGES[a], **{f"pred_{STAGES[b]}": int(cms[m][a, b]) for b in range(NS)}}
                  for m in cms for a in range(NS)]).to_csv(os.path.join(OUT, "confusion_matrices.csv"), index=False)
    ep_rows = []
    for k in keys:
        row = {"seed": k[0], "fold": k[1]}
        for tag, recs in (("v2", v2), ("ord", ordr)):
            row[f"{tag}_fixed_epoch"] = recs[k]["epochs"]["FIXED"][0] + 1
            row[f"{tag}_fixed_bal_epoch"] = recs[k]["epochs"]["FIXED_BAL"][0] + 1
            row[f"{tag}_es_epochs"] = json.dumps([e + 1 for e in recs[k]["epochs"]["ES"]])
        ep_rows.append(row)
    pd.DataFrame(ep_rows).to_csv(os.path.join(OUT, "selected_epochs.csv"), index=False)

    # ------------------------------------------------------------- paired comparisons
    metrics = [("bal", True), ("f1", True), ("acc", True), ("mae", False), ("within1", True), ("inter", True)]
    cmp = pd.DataFrame([{**paired(a, b, srcs, y, boots, ratio, k, h), "role": role}
                        for a, b, role in (("ORD", "T_v2", "primary"), ("ORD_MEDIAN", "T_v2", "secondary"),
                                           ("ORD", "TABPFN", "context only"))
                        for k, h in metrics])
    cmp.to_csv(os.path.join(OUT, "paired_comparisons.csv"), index=False)
    v3c = V3A.compare("ORD", "T_v2", srcs, y, ratio)                 # cross-check against the v3 implementation
    prim = cmp[(cmp.A == "ORD") & (cmp.B == "T_v2")].set_index("metric")
    assert abs(prim.loc["bal", "diff"] - v3c["bal_diff"]) < 1e-9 and abs(prim.loc["bal", "corrected_t_p"] - v3c["corrected_t_p"]) < 1e-9
    rec_rows = []
    Mo, Mr = oof(srcs["ORD"]), oof(srcs["T_v2"])
    for g in range(NS):
        r_o = lambda M, idx=None: float(np.mean([((m if idx is None else m[idx])[(y if idx is None else y[idx]) == g] == g).mean() for m in M]))  # noqa: E731
        bd = [r_o(Mo, i) - r_o(Mr, i) for i in boots]
        rec_rows.append({"stage": STAGES[g], "recall_v2": r_o(Mr), "recall_ordinal": r_o(Mo), "diff": r_o(Mo) - r_o(Mr),
                         "ci_low": np.percentile(bd, 2.5), "ci_high": np.percentile(bd, 97.5)})
    recd = pd.DataFrame(rec_rows)
    recd.to_csv(os.path.join(OUT, "per_class_recall_differences.csv"), index=False)

    # ------------------------------------------------------------- curves and overfitting
    cur, fit = {}, []
    for name, recs in (("T_v2", v2), ("ORD", ordr)):
        ms = [m for r in recs.values() for m in r["inner_models"]]
        c = {f"{s}_{k}": np.mean([m["curves"][s][k] for m in ms], 0) for s in ("train", "val") for k in ("loss", "bal")}
        c["heldout_loss"] = np.mean([m["outer_curves"]["loss"] for m in ms], 0)
        cur[name] = pd.DataFrame({"epoch": np.arange(1, 61), **c})
        cur[name].to_csv(os.path.join(OUT, f"curves_{name}.csv"), index=False)
        vl, tl = cur[name].val_loss.to_numpy(), cur[name].train_loss.to_numpy()
        e = [r["epochs"]["FIXED"][0] + 1 for r in recs.values()]
        fit.append({"model": name, "n_parameters": ms[0]["n_parameters"], "val_loss_min": vl.min(),
                    "val_loss_min_epoch": int(vl.argmin()) + 1, "val_loss_epoch60": vl[-1], "train_loss_epoch60": tl[-1],
                    "train_bal_epoch60": cur[name].train_bal.iloc[-1], "selected_epoch_median": float(np.median(e)),
                    "selected_epoch_iqr": [float(np.percentile(e, 25)), float(np.percentile(e, 75))],
                    "selected_epoch_range": [int(min(e)), int(max(e))],
                    "overfits_after_min": bool(vl[-1] > 1.05 * vl.min() and tl[-1] < tl[int(vl.argmin())])})
    fitd = pd.DataFrame(fit)
    fitd.to_csv(os.path.join(OUT, "fit_diagnostics.csv"), index=False)

    # ------------------------------------------------------------- decision (PLAN.md)
    g_ = lambda k: prim.loc[k]  # noqa: E731
    gain_metric = [k for k in ("bal", "f1") if g_(k).reliably_better]
    not_few_folds = any(g_(k).splits_better > g_(k).splits_worse and g_(k).mean_favourable_split_diff_without_top3 > 0
                        for k in gain_metric)
    inter_ok = g_("inter")["diff"] >= -0.03 and g_("inter").ci_high >= 0
    all_checks = all(checks.values())
    if not all_checks or g_("bal").reliably_worse or g_("f1").reliably_worse or g_("inter").ci_high < 0:
        decision = "REJECT"
    elif gain_metric and not_few_folds and inter_ok:
        decision = "ADOPT"
    else:
        decision = "INCONCLUSIVE"
    ordinal_error = ("reliably lower" if g_("mae").reliably_better else
                     "reliably higher" if g_("mae").reliably_worse else "not reliably different")

    # ------------------------------------------------------------- figures
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": P.AXIS, "axes.labelcolor": P.INK2, "xtick.color": P.INK2,
                         "ytick.color": P.INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "grid.color": P.GRID, "figure.facecolor": P.SURFACE, "axes.facecolor": P.SURFACE})
    col = {"T_v2": P.MUTED, "ORD": P.SERIES[0]}
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.6), gridspec_kw={"width_ratios": [1.3, 1]})
    for name in ("T_v2", "ORD"):
        axs[0].plot(cur[name].epoch, cur[name].val_loss, color=col[name], lw=2, label=f"{NAMES[name]}: inner validation")
        axs[0].plot(cur[name].epoch, cur[name].train_loss, color=col[name], lw=1.2, ls="--", label=f"{NAMES[name]}: training")
    axs[0].axhline(np.log(6), color=P.MUTED, ls=":", lw=1)
    axs[0].text(60, np.log(6), "uniform guess (ln 6)", color=P.MUTED, fontsize=7, va="bottom", ha="right")
    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("negative log-likelihood")
    axs[0].set_title("Loss curves, mean over 100 inner models", loc="left", fontsize=9)
    axs[0].legend(fontsize=6.5, frameon=False)
    axs[0].yaxis.grid(True)
    bins = np.arange(0.5, 61.5, 3)
    for i, (name, recs) in enumerate((("T_v2", v2), ("ORD", ordr))):
        e = [r["epochs"]["FIXED"][0] + 1 for r in recs.values()]
        h, _ = np.histogram(e, bins)
        axs[1].bar(bins[:-1] + 1.5 + (i - 0.5) * 1.3, h, 1.3, color=col[name], label=NAMES[name].split(" (")[0])
    axs[1].set_xlabel("epoch selected by inner-validation loss (one per outer split)")
    axs[1].set_ylabel("outer splits")
    axs[1].yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    axs[1].set_title("Selected epochs", loc="left", fontsize=9)
    axs[1].legend(fontsize=7, frameon=False)
    axs[1].yaxis.grid(True)
    fig.savefig(os.path.join(OUT, "fig_curves_epochs.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    show = ["T_v2", "ORD", "ORD_MEDIAN"]
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.4))
    for ax, m in zip(axs, show):
        Mn = cms[m] / cms[m].sum(1, keepdims=True)
        ax.imshow(Mn, cmap="Blues", vmin=0, vmax=0.6)
        for a in range(NS):
            for b in range(NS):
                ax.text(b, a, f"{100 * Mn[a, b]:.0f}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if Mn[a, b] > 0.35 else P.INK2)
        ax.set_xticks(range(NS), STAGES, fontsize=7, rotation=45)
        ax.set_yticks(range(NS), STAGES, fontsize=7)
        ax.set_title(NAMES[m], fontsize=7.5, loc="left")
        ax.set_xlabel("predicted", fontsize=7)
    axs[0].set_ylabel("true (row %, 5 seeds pooled)", fontsize=7)
    fig.savefig(os.path.join(OUT, "fig_confusion.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 3.5))
    x = np.arange(NS)
    axs[0].bar(x - 0.2, 100 * recd.recall_v2, 0.38, color=col["T_v2"], label=NAMES["T_v2"])
    axs[0].bar(x + 0.2, 100 * recd.recall_ordinal, 0.38, color=col["ORD"], label=NAMES["ORD"])
    for xi, r in enumerate(recd.itertuples()):
        axs[0].text(xi - 0.2, 100 * r.recall_v2 + 1, f"{100 * r.recall_v2:.0f}", ha="center", fontsize=7, color=P.INK2)
        axs[0].text(xi + 0.2, 100 * r.recall_ordinal + 1, f"{100 * r.recall_ordinal:.0f}", ha="center", fontsize=7, color=P.INK)
    axs[0].axhline(100 / 6, color=P.MUTED, ls="--", lw=1)
    axs[0].set_xticks(x, STAGES)
    axs[0].set_ylabel("recall (%, seed-pooled)")
    axs[0].set_title("Per-class recall", loc="left", fontsize=9)
    axs[0].legend(fontsize=7, frameon=False)
    axs[0].yaxis.grid(True)
    for i, m in enumerate(("T_v2", "ORD")):
        M = oof(srcs[m])
        dist = np.bincount(np.abs(M - y[None]).ravel(), minlength=NS) / M.size
        axs[1].bar(x + (i - 0.5) * 0.38, 100 * dist, 0.38, color=col[m], label=NAMES[m])
    axs[1].set_xticks(x, [str(v) for v in x])
    axs[1].set_xlabel("absolute error (stage steps)")
    axs[1].set_ylabel("share of predictions (%)")
    axs[1].set_title("Distance between predicted and true stage", loc="left", fontsize=9)
    axs[1].legend(fontsize=7, frameon=False)
    axs[1].yaxis.grid(True)
    fig.savefig(os.path.join(OUT, "fig_recall_error.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------------- summary
    L = [f"CHECKS: {json.dumps(checks)}", "", "METRICS (seed-pooled, 25 identical outer splits; 95% CI subject bootstrap)"]
    for m, r in desc.iterrows():
        L.append(f"  {m:10s} {NAMES[m]}: balanced {100 * r.balanced_accuracy:.1f} [{100 * r.bal_ci_low:.1f}, {100 * r.bal_ci_high:.1f}] "
                 f"(fold {100 * r.bal_fold_mean:.1f} +/- {100 * r.bal_fold_sd:.1f}); macro-F1 {100 * r.f1_mean:.1f} "
                 f"[{100 * r.f1_ci_low:.1f}, {100 * r.f1_ci_high:.1f}]; accuracy {100 * r.acc_mean:.1f} [{100 * r.acc_ci_low:.1f}, "
                 f"{100 * r.acc_ci_high:.1f}]; MAE {r.mae_mean:.2f} [{r.mae_ci_low:.2f}, {r.mae_ci_high:.2f}] (fold {r.mae_fold_mean:.2f} "
                 f"+/- {r.mae_fold_sd:.2f}); within one stage {100 * r.within1_mean:.1f} [{100 * r.within1_ci_low:.1f}, "
                 f"{100 * r.within1_ci_high:.1f}]; intermediate recall {100 * r.inter_mean:.1f}; recall "
                 + " ".join(f"{g} {100 * r['recall_' + g]:.0f}" for g in STAGES) + f"; predicted AD share {100 * r.pred_share_AD:.0f}%")
    L.append("\nPER-CLASS PRECISION / RECALL / F1 (5 seeds pooled)")
    for m in ("T_v2", "ORD", "ORD_MEDIAN"):
        q = prf[prf.model == m]
        L.append(f"  {m}: " + "; ".join(f"{r.stage} {100 * r.precision:.0f}/{100 * r.recall:.0f}/{100 * r.f1:.0f}" for r in q.itertuples()))
    L.append("\nPAIRED (A - B)")
    for r in cmp.itertuples():
        scale = 1 if r.metric == "mae" else 100
        L.append(f"  [{r.role}] {r.A} vs {r.B} {r.metric}: {scale * r.diff:+.2f} [{scale * r.ci_low:+.2f}, {scale * r.ci_high:+.2f}], "
                 f"per seed {[round(scale * v, 2) for v in r.per_seed_diff]}, seeds A better {r.seeds_A_better}/5, splits "
                 f"better/worse/tied {r.splits_better}/{r.splits_worse}/{r.splits_tied} (sign p {r.sign_p:.2f}), corrected t p "
                 f"{r.corrected_t_p:.3f}; favourable mean without top-3 splits {scale * r.mean_favourable_split_diff_without_top3:+.2f}"
                 f" -> reliably better {r.reliably_better}, reliably worse {r.reliably_worse}")
    L.append("\nPER-CLASS RECALL DIFFERENCES (ordinal - v2) [95% CI]")
    for r in recd.itertuples():
        L.append(f"  {r.stage}: {100 * r.recall_v2:.0f} -> {100 * r.recall_ordinal:.0f} ({100 * r.diff:+.1f} [{100 * r.ci_low:+.1f}, {100 * r.ci_high:+.1f}])")
    L.append("\nFIT DIAGNOSTICS")
    for r in fitd.itertuples():
        L.append(f"  {r.model}: {r.n_parameters} params; inner-val loss min {r.val_loss_min:.3f} at epoch {r.val_loss_min_epoch}, "
                 f"epoch 60 {r.val_loss_epoch60:.3f}; train loss 60 {r.train_loss_epoch60:.3f}; train bal 60 {100 * r.train_bal_epoch60:.1f}; "
                 f"selected epoch median {r.selected_epoch_median:.0f} IQR {r.selected_epoch_iqr} range {r.selected_epoch_range}; "
                 f"overfits after minimum: {r.overfits_after_min}")
    L.append(f"\nDECISION (PLAN.md rules): {decision}; ordinal error (MAE) {ordinal_error}; reliably better on "
             f"{gain_metric or 'neither balanced accuracy nor macro-F1'}; intermediate criterion met: {bool(inter_ok)}")
    L.append(f"RUNTIME: {sum(r['seconds'] for r in ordr.values()) / 60:.1f} min")
    open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "checks": checks, "decision": decision,
               "ordinal_error": ordinal_error, "label_order": ORDER, "primary": "FIXED epoch rule, argmax"},
              open(os.path.join(OUT, "manifest.json"), "w"), indent=2)
    print("\n".join(L))


if __name__ == "__main__":
    main()
