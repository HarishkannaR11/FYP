"""
Analysis for experiment v3 (pre-registered in derivatives/experiments/biomarker_ablation_v3/PLAN.md).
Reads saved predictions only; trains nothing; refuses to overwrite.

Point estimates: per seed, the 5 outer test folds are pooled into one prediction per subject (116), the metric is
computed on that, then averaged over the 5 seeds. Fold-level mean +/- sd over the 25 outer folds is reported too.
95% CI: percentile interval of 2000 class-stratified bootstraps of subjects; each resample reuses every seed's
predictions for the drawn subjects and averages the seeds. It reflects test-subject sampling given the trained
models. Paired comparisons add the corrected resampled t-test (Nadeau & Bengio 2003) over the 25 splits, whose
variance term (1/J + n_test/n_train) accounts for the overlap between training sets of repeated CV.
"Reliably different" = bootstrap CI excludes 0, >= 4 of 5 seeds agree, and corrected-t p < 0.05.

Run:  python scripts/experiments/biomarker_ablation_v3/analyze_v3.py
"""
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "transformer_diagnostics_v1"))
import analyze as A1  # noqa: E402  (pooling, bootstrap; imports leakage_safe_v1/report)
import run_k_sweep as KS  # noqa: E402
import run_tabpfn_configs as T  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import binomtest  # noqa: E402
from scipy.stats import t as tdist  # noqa: E402

R, K, P = A1.R, A1.K, A1.P
ROOT = K.ROOT
EXP = T.EXP
OUT = os.path.join(EXP, "analysis")
STAGES, NS, SEEDS = K.STAGES, K.NS, list(K.SEEDS)
V2 = os.path.join(ROOT, "derivatives", "experiments", "transformer_nested_v2", "grid")
MT = os.path.join(ROOT, "derivatives", "model_training", "tabpfn")
A_CONFIGS = T.NESTED_ORDER
LABEL = {"A_all4": "TabPFN baseline (all four, k=50)", "A_all4_k25": "TabPFN all four, k=25",
         "A_all4_k100": "TabPFN all four, k=100", "A_mALFF": "TabPFN mALFF only", "A_mReHo": "TabPFN mReHo only",
         "A_DCz": "TabPFN DC_z only", "A_FC": "TabPFN FC strength only", "A_local": "TabPFN mALFF + mReHo",
         "A_conn": "TabPFN DC_z + FC strength", "NESTED": "TabPFN nested choice of setting (inner CV)",
         "BAL_A_all4": "TabPFN baseline + class balancing", "T_v2": "Transformer v2 (baseline)",
         "T_small": "Transformer small / regularised", "T_unweighted": "Transformer unweighted loss",
         "T_chosen": "Transformer on the nested-chosen setting", "T_comorb8": "Transformer + 8 comorbidities",
         "TP_comorb8": "TabPFN + 8 comorbidities", "TP_comorb8_only": "TabPFN 8 comorbidities only",
         "SITE": "Site-only lookup rule (confound diagnostic)", "L_frac50": "TabPFN baseline, 50% of training",
         "L_frac75": "TabPFN baseline, 75% of training", "NESTED_K": "TabPFN nested choice of k (inner CV)",
         **{f"K_all4_k{k}": f"TabPFN all four, k={k}" for k in (10, 20, 30, 75)}}
K_CONFIGS = list(KS.K_CONFIGS)


# ------------------------------------------------------------------ sources
def grid(cname):
    return {(r["seed"], r["fold"]): r for r in (json.load(open(f)) for f in glob.glob(os.path.join(EXP, "grid", cname, "*.json")))}


def tgrid(var):
    return {(r["seed"], r["fold"]): r for r in
            (json.load(open(f)) for f in glob.glob(os.path.join(EXP, "transformer", var, "*.json")))}


def as_src(recs, key="pred", sub=None):
    return {k: (np.array(r["test_idx"]), np.array(r[key] if sub is None else r[key][sub])) for k, r in recs.items()}


def site_rule(y, subjects, plan):
    site = np.array([s.replace("sub-", "")[:3] for s in subjects])
    src = {}
    for s in SEEDS:
        for f, (tr, te) in enumerate(plan[s][1]):
            cnt_all = np.bincount(y[tr], minlength=NS)
            pred = []
            for i in te:
                ys = y[tr][site[tr] == site[i]]
                if ys.size:
                    c = np.bincount(ys, minlength=NS)
                    tie = np.flatnonzero(c == c.max())
                    pred.append(tie[np.argmax(cnt_all[tie])])
                else:
                    pred.append(cnt_all.argmax())
            src[(s, f)] = (te, np.array(pred))
    return src


# ------------------------------------------------------------------ statistics
def fold_metrics(src, y):
    rows = []
    for (s, f), (te, pr) in sorted(src.items()):
        e = K.H.evaluate(y[te], pr)
        rows.append({"seed": s, "fold": f, "accuracy": e["accuracy"], "balanced_accuracy": e["balanced_accuracy"],
                     "macro_f1": e["macro_f1"], **{f"recall_{g}": v for g, v in e["per_class_recall"].items()}})
    return pd.DataFrame(rows)


def describe(name, src, y):
    st, cm, _ = A1.summarize(src, y, SEEDS)
    fm = fold_metrics(src, y)
    return {"model": name, "description": LABEL.get(name, name), "accuracy": st["accuracy"],
            "balanced_accuracy": st["bal"], "bal_ci_low": st["bal_ci"][0], "bal_ci_high": st["bal_ci"][1],
            "macro_f1": st["f1"], "bal_seed_sd": st["bal_sd"],
            "bal_fold_mean": fm.balanced_accuracy.mean(), "bal_fold_sd": fm.balanced_accuracy.std(ddof=1),
            "f1_fold_mean": fm.macro_f1.mean(), "f1_fold_sd": fm.macro_f1.std(ddof=1),
            "acc_fold_mean": fm.accuracy.mean(), "acc_fold_sd": fm.accuracy.std(ddof=1),
            "bal_fold_min": fm.balanced_accuracy.min(), "bal_fold_max": fm.balanced_accuracy.max(), "qwk": st["qwk"],
            **{f"recall_{g}": v for g, v in st["recall"].items()},
            **{f"pred_share_{g}": v for g, v in st["pred_share"].items()}}, cm, fm


def compare(a, b, srcs, y, ratio):
    p = A1.paired(srcs[a], srcs[b], y, SEEDS)
    fa, fb = fold_metrics(srcs[a], y), fold_metrics(srcs[b], y)
    d = (fa.balanced_accuracy - fb.balanced_accuracy).to_numpy()
    dm = (fa.macro_f1 - fb.macro_f1).to_numpy()
    J = d.size
    se = np.sqrt((1 / J + ratio) * d.var(ddof=1)) if d.var() > 0 else np.inf
    tval = d.mean() / se if np.isfinite(se) else 0.0
    pt = float(2 * tdist.sf(abs(tval), J - 1))
    up, dn = int((d > 0).sum()), int((d < 0).sum())
    agree = p["seeds_better"] if p["diff"] > 0 else p["n_seeds"] - p["seeds_better"]
    reliable = (p["ci"][0] > 0 or p["ci"][1] < 0) and agree >= 4 and pt < 0.05
    return {"A": a, "B": b, "bal_diff": p["diff"], "bal_diff_ci_low": p["ci"][0], "bal_diff_ci_high": p["ci"][1],
            "seeds_A_better": p["seeds_better"], "fold_bal_diff_mean": d.mean(), "fold_bal_diff_sd": d.std(ddof=1),
            "fold_f1_diff_mean": dm.mean(), "splits_up": up, "splits_down": dn,
            "sign_p": float(binomtest(up, up + dn).pvalue) if up + dn else 1.0, "corrected_t": tval,
            "corrected_t_p": pt, "reliably_different": bool(reliable),
            "direction": ("A better" if p["diff"] > 0 else "B better") if reliable else "no reliable difference"}


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj


def main():
    if os.path.exists(os.path.join(OUT, "comparison_table.csv")):
        raise SystemExit("analysis already written; not overwritten")
    d = T.S.load_cohort()
    y, subjects = d[3], d[4]
    plan = T.S.cv_plan(y, subjects, 5)
    keys = [(s, f) for s in SEEDS for f in range(5)]
    tab = {c: grid(c) for c in list(T.CONFIGS) + K_CONFIGS}
    missing = {c: 25 - len(v) for c, v in tab.items() if len(v) < 25}
    tv = {v: tgrid(v) for v in ("small", "unweighted", "comorb8", "chosen")}
    missing.update({f"T_{v}": 25 - len(r) for v, r in tv.items() if len(r) < 25})
    if missing:
        raise SystemExit(f"incomplete runs: {missing}")
    os.makedirs(OUT, exist_ok=True)

    srcs = {c: as_src(tab[c]) for c in tab}
    srcs["BAL_A_all4"] = as_src(tab["A_all4"], "pred_balanced")
    for c in A_CONFIGS[1:]:
        srcs[f"BAL_{c}"] = as_src(tab[c], "pred_balanced")
    choice = {k: T.nested_choice({c: tab[c][k]["inner"]["score"] for c in A_CONFIGS}) for k in keys}
    srcs["NESTED"] = {k: srcs[choice[k]][k] for k in keys}
    choice_k = {}
    for k in keys:                                                 # NESTED_K: Addendum 1 rule
        sc = {c: tab[c][k]["inner"]["score"] for c in KS.NESTED_K_ORDER}
        choice_k[k] = next(c for c in KS.NESTED_K_ORDER if sc[c] == max(sc.values()))
    srcs["NESTED_K"] = {k: srcs[choice_k[k]][k] for k in keys}
    v2 = {(r["seed"], r["fold"]): r for r in (json.load(open(f)) for f in glob.glob(os.path.join(V2, "*.json")))}
    srcs["T_v2"] = as_src(v2, "pred", "FIXED")
    srcs["T_v2_ES"] = as_src(v2, "pred", "ES")
    for v, recs in tv.items():
        srcs[f"T_{v}"] = as_src(recs, "pred", "FIXED")
        srcs[f"T_{v}_ES"] = as_src(recs, "pred", "ES")
    srcs["TP_comorb8"] = A1.from_records(os.path.join(MT, "cv116_tabpfn_svmrfe50_comorb8.json"))
    srcs["TP_comorb8_only"] = A1.from_records(os.path.join(MT, "cv116_tabpfn_comorb8.json"))
    srcs["SITE"] = site_rule(y, subjects, plan)
    for name, s in srcs.items():                                  # identical outer folds everywhere
        assert sorted(s) == keys, name
        assert all(np.array_equal(np.sort(s[k][0]), np.sort(plan[k[0]][1][k[1]][1])) for k in keys), name

    # ---------------------------------------------------------- reproduction checks
    repro = {"A_all4_vs_existing_baseline": sum(np.array_equal(srcs["A_all4"][k][1], A1.from_records(
        os.path.join(MT, "cv116_tabpfn_svmrfe50.json"))[k][1]) for k in keys)}
    same = [k for k in keys if tv["chosen"][k]["config"] == "A_all4"]
    repro["T_chosen_equals_v2_where_baseline_chosen"] = f"{sum(np.array_equal(srcs['T_chosen'][k][1], srcs['T_v2'][k][1]) for k in same)}/{len(same)}"

    # ---------------------------------------------------------- descriptive table
    order = ["A_all4", "NESTED", "NESTED_K", "K_all4_k10", "K_all4_k20", "A_all4_k25", "K_all4_k30", "K_all4_k75",
             "A_all4_k100", "A_mALFF", "A_mReHo", "A_DCz", "A_FC", "A_local", "A_conn", "BAL_A_all4", "L_frac75", "L_frac50", "T_v2", "T_small", "T_unweighted", "T_chosen",
             "TP_comorb8", "T_comorb8", "TP_comorb8_only", "SITE"]
    rows, cms, folds = [], {}, []
    for m in order + [f"BAL_{c}" for c in A_CONFIGS[1:]] + [f"P_site{r}" for r in range(1, 6)] + \
            [f"P_glob{r}" for r in range(1, 4)] + ["T_v2_ES", "T_small_ES", "T_unweighted_ES", "T_chosen_ES", "T_comorb8_ES"]:
        row, cm, fm = describe(m, srcs[m], y)
        rows.append(row)
        cms[m] = cm
        fm.insert(0, "model", m)
        folds.append(fm)
    desc = pd.DataFrame(rows).set_index("model")
    pd.concat(folds).to_csv(os.path.join(OUT, "fold_level.csv"), index=False)
    pd.DataFrame([{"model": m, "true": STAGES[a], **{f"pred_{STAGES[b]}": int(cms[m][a, b]) for b in range(NS)}}
                  for m in cms for a in range(NS)]).to_csv(os.path.join(OUT, "confusion_matrices.csv"), index=False)

    # ---------------------------------------------------------- paired comparisons
    n_te = np.mean([len(plan[s][1][f][1]) for s, f in keys])
    ratio = n_te / (116 - n_te)
    pairs = [(c, "A_all4", "R1 exploratory") for c in A_CONFIGS[1:] + K_CONFIGS] + \
            [("NESTED", "A_all4", "R1 primary"), ("NESTED_K", "A_all4", "R1 primary"), ("BAL_A_all4", "A_all4", "R2"),
             ("T_unweighted", "T_v2", "R2"), ("T_small", "T_v2", "R3"), ("T_chosen", "T_v2", "transformer setting"),
             ("T_v2", "A_all4", "transformer vs TabPFN"), ("T_small", "A_all4", "transformer vs TabPFN"),
             ("T_chosen", "NESTED", "transformer vs TabPFN, same settings"),
             ("TP_comorb8", "A_all4", "R5 separate"), ("T_comorb8", "T_v2", "R5 separate"),
             ("T_comorb8", "TP_comorb8", "R5 separate"), ("TP_comorb8_only", "A_all4", "R5 separate"),
             ("A_all4", "SITE", "confound"), ("L_frac75", "A_all4", "learning curve"), ("L_frac50", "A_all4", "learning curve")]
    cmp = pd.DataFrame([{**compare(a, b, srcs, y, ratio), "rule": r} for a, b, r in pairs])
    ex = cmp.rule == "R1 exploratory"
    cmp["holm_p"] = np.nan
    cmp.loc[ex, "holm_p"] = holm(cmp.loc[ex, "corrected_t_p"].to_numpy())
    pr = cmp.rule == "R1 primary"                                  # NESTED and NESTED_K adjusted together
    cmp.loc[pr, "holm_p"] = holm(cmp.loc[pr, "corrected_t_p"].to_numpy())
    cmp.loc[pr, "reliably_different"] = cmp.loc[pr, "reliably_different"] & (cmp.loc[pr, "holm_p"] < 0.05)
    cmp.to_csv(os.path.join(OUT, "paired_comparisons.csv"), index=False)

    table = desc.loc[order].copy()
    base = {m: ("T_v2" if m.startswith("T_") else "A_all4") for m in order}
    for m in order:
        b = base[m]
        table.loc[m, "diff_vs_baseline"] = desc.loc[m, "balanced_accuracy"] - desc.loc[b, "balanced_accuracy"]
        table.loc[m, "f1_diff_vs_baseline"] = desc.loc[m, "macro_f1"] - desc.loc[b, "macro_f1"]
        table.loc[m, "baseline"] = b
    table.to_csv(os.path.join(OUT, "comparison_table.csv"))
    desc.to_csv(os.path.join(OUT, "all_models.csv"))

    # ---------------------------------------------------------- nested choice, inner vs held-out, stability
    ch = pd.Series(choice).value_counts()
    ch_k = pd.Series(choice_k).value_counts()
    inner_rows = []
    for c in A_CONFIGS + K_CONFIGS:
        sc = [tab[c][k]["inner"]["score"] for k in keys]
        fo = [K.H.evaluate(y[np.array(tab[c][k]["test_idx"])], np.array(tab[c][k]["pred"]))["balanced_accuracy"] for k in keys]
        sups = [tab[c][k]["support"] for k in keys]
        jac = [len(set(a) & set(b)) / len(set(a) | set(b)) for i, a in enumerate(sups) for b in sups[i + 1:]]
        freq = np.bincount(np.concatenate(sups), minlength=984) / len(sups)
        inner_rows.append({"config": c, "inner_bal_mean": np.mean(sc), "heldout_fold_bal_mean": np.mean(fo),
                           "inner_minus_heldout": np.mean(sc) - np.mean(fo), "times_chosen": int(ch.get(c, 0)),
                           "times_chosen_k": int(ch_k.get(c, 0)),
                           "selection_jaccard": np.mean(jac), "features_in_half_of_fits": int((freq >= 0.5).sum()),
                           "inner_bal_balanced_mean": np.mean([tab[c][k]["inner"]["score_balanced"] for k in keys])})
    inner = pd.DataFrame(inner_rows)
    inner.to_csv(os.path.join(OUT, "settings_inner_vs_heldout_and_stability.csv"), index=False)
    pd.DataFrame([{"seed": k[0], "fold": k[1], "chosen": choice[k], "chosen_k": choice_k[k],
                   **{c: tab[c][k]["inner"]["score"] for c in A_CONFIGS + K_CONFIGS}} for k in keys]).to_csv(
        os.path.join(OUT, "nested_choice_per_split.csv"), index=False)

    fit_rows = []
    for name, recs in (("T_v2", v2), *((f"T_{v}", r) for v, r in tv.items())):
        ms = [m for r in recs.values() for m in r["inner_models"]]
        vl = np.mean([m["curves"]["val"]["loss"] for m in ms], 0)
        tl = np.mean([m["curves"]["train"]["loss"] for m in ms], 0)
        tb = np.mean([m["curves"]["train"]["bal"] for m in ms], 0)
        gap = np.mean([np.mean(np.array(m["curves"]["val"]["bal"][9:]) - np.array(m["outer_curves"]["bal"][9:])) for m in ms])
        e = [r["epochs"]["FIXED"][0] + 1 for r in recs.values()]
        fit_rows.append({"model": name, "n_parameters": ms[0]["n_parameters"], "val_loss_min": vl.min(),
                         "val_loss_min_epoch": int(vl.argmin()) + 1, "val_loss_epoch60": vl[-1],
                         "train_loss_epoch60": tl[-1], "train_bal_epoch60": tb[-1], "fixed_epoch_median": np.median(e),
                         "inner_minus_heldout_no_selection": gap,
                         "overfit_condition": bool(vl[-1] > 1.05 * vl.min() and tl[-1] < tl[int(vl.argmin())])})
        pd.DataFrame({"epoch": np.arange(1, vl.size + 1), "train_loss": tl, "val_loss": vl, "train_bal": tb,
                      "val_bal": np.mean([m["curves"]["val"]["bal"] for m in ms], 0),
                      "heldout_bal": np.mean([m["outer_curves"]["bal"] for m in ms], 0)}).to_csv(
            os.path.join(OUT, f"epoch_curves_{name}.csv"), index=False)
    fit = pd.DataFrame(fit_rows)
    fit.to_csv(os.path.join(OUT, "transformer_fit_diagnostics.csv"), index=False)

    # ---------------------------------------------------------- confound: within-site null, ComBat (existing)
    null_site = [desc.loc[f"P_site{r}", "balanced_accuracy"] for r in range(1, 6)]
    null_glob = [desc.loc[f"P_glob{r}", "balanced_accuracy"] for r in range(1, 4)]
    b0 = desc.loc["A_all4", "balanced_accuracy"]
    r4 = bool(b0 > max(null_site) and b0 - np.mean(null_site) > 0.02)
    cb = {}
    for tag in ("none", "combat"):
        s15 = A1.from_v1(f"tabpfn|img|k50|{tag}|-")
        sd15 = A1.full_seeds(s15)
        st, _, _ = A1.summarize(s15, y, sd15)
        cb[tag] = (st["bal"], st["f1"], sd15, s15)
    cbp = A1.paired(cb["combat"][3], cb["none"][3], y, cb["none"][2])

    # ---------------------------------------------------------- decisions
    get = lambda a, b: cmp[(cmp.A == a) & (cmp.B == b)].iloc[0]  # noqa: E731
    dec = {"R1_nested_replaces_baseline": bool(get("NESTED", "A_all4").reliably_different and get("NESTED", "A_all4").bal_diff > 0),
           "R1_nested_k_replaces_baseline": bool(get("NESTED_K", "A_all4").reliably_different
                                                 and get("NESTED_K", "A_all4").bal_diff > 0),
           "R2_tabpfn_balancing_adopted": bool(get("BAL_A_all4", "A_all4").reliably_different and get("BAL_A_all4", "A_all4").bal_diff > 0),
           "R2_transformer_unweighted_adopted": bool(get("T_unweighted", "T_v2").reliably_different and get("T_unweighted", "T_v2").bal_diff > 0),
           "R3_transformer_retained": ("T_small" if not get("T_small", "T_v2").reliably_different
                                       else ("T_small" if get("T_small", "T_v2").bal_diff > 0 else "T_v2")),
           "R4_baseline_exceeds_within_site_null": r4}

    # ---------------------------------------------------------- figures
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": P.AXIS, "axes.labelcolor": P.INK2, "xtick.color": P.INK2,
                         "ytick.color": P.INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "grid.color": P.GRID, "figure.facecolor": P.SURFACE, "axes.facecolor": P.SURFACE})
    site_bal = desc.loc["SITE", "balanced_accuracy"]

    def refs(ax, x_right):
        ax.axhline(1 / 6, color=P.MUTED, ls="--", lw=1)
        ax.text(x_right, 1 / 6, "chance 16.7%", color=P.MUTED, fontsize=7, va="bottom", ha="right")
        ax.axhline(site_bal, color=P.INK2, ls=":", lw=1)
        ax.text(x_right, site_bal, f"site-only rule {100 * site_bal:.1f}%", color=P.INK2, fontsize=7, va="bottom", ha="right")
        ax.axhspan(min(null_site), max(null_site), color=P.GRID, zorder=0)
        ax.text(x_right, max(null_site), "within-site permutation null (range)", color=P.MUTED, fontsize=7,
                va="bottom", ha="right")

    def dots(ax, names, colours):
        for i, m in enumerate(names):
            r = desc.loc[m]
            ax.errorbar([i], [r.balanced_accuracy], yerr=[[r.balanced_accuracy - r.bal_ci_low], [r.bal_ci_high - r.balanced_accuracy]],
                        fmt="D", ms=6, color=colours[i], elinewidth=2, capsize=0)
            ax.text(i + 0.12, r.balanced_accuracy, f"{100 * r.balanced_accuracy:.1f}", fontsize=7, color=P.INK, va="center")
        ax.set_xticks(range(len(names)), [LABEL[m].replace("TabPFN ", "").replace("Transformer ", "") for m in names],
                      rotation=35, ha="right")
        ax.set_ylabel("balanced accuracy (seed-pooled, 95% CI)")
        ax.yaxis.grid(True)

    tnames = ["A_all4", "NESTED", "A_all4_k25", "A_all4_k100", "A_mALFF", "A_mReHo", "A_DCz", "A_FC", "A_local", "A_conn",
              "BAL_A_all4"]
    fig, axs = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    dots(axs[0], tnames, [P.SERIES[0] if m in ("A_all4", "NESTED") else P.MUTED for m in tnames])
    refs(axs[0], len(tnames) - 0.5)
    axs[0].set_title("TabPFN v2: feature-selection settings and class balancing (25 identical outer splits)", loc="left", fontsize=9)
    trn = ["T_v2", "T_small", "T_unweighted", "T_chosen"]
    dots(axs[1], trn, [P.SERIES[1] if m == "T_v2" else P.MUTED for m in trn])
    refs(axs[1], len(trn) - 0.5)
    axs[1].set_title("Corrected transformer variants (fixed-epoch rule)", loc="left", fontsize=9)
    fig.savefig(os.path.join(OUT, "fig_settings.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    kc = [("K_all4_k10", 10), ("K_all4_k20", 20), ("A_all4_k25", 25), ("K_all4_k30", 30), ("A_all4", 50),
          ("K_all4_k75", 75), ("A_all4_k100", 100)]
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    vv = [desc.loc[c] for c, _ in kc]
    ax.errorbar([k for _, k in kc], [v.balanced_accuracy for v in vv],
                yerr=[[v.balanced_accuracy - v.bal_ci_low for v in vv], [v.bal_ci_high - v.balanced_accuracy for v in vv]],
                fmt="-o", color=P.SERIES[0], ms=6, lw=2, elinewidth=1.5, capsize=0)
    for (c, k), v in zip(kc, vv):
        ax.text(k + 1.5, v.balanced_accuracy, f"{100 * v.balanced_accuracy:.1f}", fontsize=7, color=P.INK, va="center")
    refs(ax, 103)
    ax.set_xlabel("SVM-RFE features kept (k), all four biomarkers")
    ax.set_ylabel("balanced accuracy (seed-pooled, 95% CI)")
    ax.set_title("TabPFN: number of selected features", loc="left", fontsize=9)
    ax.yaxis.grid(True)
    fig.savefig(os.path.join(OUT, "fig_k_curve.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(12, 3.6))
    sizes =[np.mean([tab[c][k]["n_train"] for k in keys]) for c in ("L_frac50", "L_frac75", "A_all4")]
    vals = [desc.loc[c] for c in ("L_frac50", "L_frac75", "A_all4")]
    axs[0].errorbar(sizes, [v.balanced_accuracy for v in vals],
                    yerr=[[v.balanced_accuracy - v.bal_ci_low for v in vals], [v.bal_ci_high - v.balanced_accuracy for v in vals]],
                    fmt="-o", color=P.SERIES[0], ms=6, lw=2, elinewidth=1.5, capsize=0)
    for x, v in zip(sizes, vals):
        axs[0].text(x + 1, v.balanced_accuracy, f"{100 * v.balanced_accuracy:.1f}%", fontsize=7, color=P.INK, va="center")
    refs(axs[0], max(sizes) + 3)
    axs[0].set_xlabel("training subjects per outer split")
    axs[0].set_ylabel("balanced accuracy (seed-pooled, 95% CI)")
    axs[0].set_title("TabPFN baseline learning curve (same outer test folds)", loc="left", fontsize=9)
    axs[0].yaxis.grid(True)
    cols = {"T_v2": P.SERIES[1], "T_small": P.SERIES[0], "T_unweighted": P.SERIES[2], "T_comorb8": P.MUTED}
    for name, c in cols.items():
        cur = pd.read_csv(os.path.join(OUT, f"epoch_curves_{name}.csv"))
        axs[1].plot(cur.epoch, cur.val_loss, color=c, lw=2, label=LABEL[name].replace("Transformer ", ""))
        axs[1].plot(cur.epoch, cur.train_loss, color=c, lw=1, ls="--")
    axs[1].axhline(np.log(6), color=P.MUTED, ls=":", lw=1)
    axs[1].text(60, np.log(6), "uniform guess (ln 6)", color=P.MUTED, fontsize=7, va="bottom", ha="right")
    axs[1].set_xlabel("epoch")
    axs[1].set_ylabel("cross-entropy (solid: inner validation, dashed: training)")
    axs[1].set_title("Transformer learning curves (mean over 100 inner models)", loc="left", fontsize=9)
    axs[1].legend(fontsize=7, frameon=False)
    axs[1].yaxis.grid(True)
    fig.savefig(os.path.join(OUT, "fig_learning_curves.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    show = ["A_all4", "NESTED", "T_v2", "T_small", "TP_comorb8", "SITE"]
    fig, axs = plt.subplots(1, len(show), figsize=(3.1 * len(show), 3.2))
    for ax, m in zip(axs, show):
        Mn = cms[m] / cms[m].sum(1, keepdims=True)
        ax.imshow(Mn, cmap="Blues", vmin=0, vmax=0.6)
        for a in range(NS):
            for b in range(NS):
                ax.text(b, a, f"{100 * Mn[a, b]:.0f}", ha="center", va="center", fontsize=6.5,
                        color="#ffffff" if Mn[a, b] > 0.35 else P.INK2)
        ax.set_xticks(range(NS), STAGES, fontsize=6.5, rotation=45)
        ax.set_yticks(range(NS), STAGES, fontsize=6.5)
        ax.set_title(LABEL[m].replace(" (confound diagnostic)", "\n(confound diagnostic)"), fontsize=7.5, loc="left")
        ax.set_xlabel("predicted", fontsize=7)
    axs[0].set_ylabel("true (row %, 5 seeds pooled)", fontsize=7)
    fig.savefig(os.path.join(OUT, "fig_confusion.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---------------------------------------------------------- summary
    L = [f"REPRODUCTION: {json.dumps(repro)}", "", "COMPARISON TABLE (balanced acc. seed-pooled [95% CI]; fold mean +/- sd; "
         "macro-F1; recall CN SMC EMCI MCI LMCI AD; difference vs baseline)"]
    for m, r in table.iterrows():
        L.append(f"  {m:16s} {100 * r.balanced_accuracy:5.1f} [{100 * r.bal_ci_low:.1f}, {100 * r.bal_ci_high:.1f}]  "
                 f"fold {100 * r.bal_fold_mean:.1f} +/- {100 * r.bal_fold_sd:.1f}  F1 {100 * r.macro_f1:.1f} "
                 f"(fold {100 * r.f1_fold_mean:.1f} +/- {100 * r.f1_fold_sd:.1f})  acc {100 * r.accuracy:.1f}  recall "
                 + " ".join(f"{100 * r['recall_' + g]:.0f}" for g in STAGES)
                 + f"  diff {100 * r.diff_vs_baseline:+.1f} vs {r.baseline}")
    L.append("\nPAIRED COMPARISONS")
    for r in cmp.itertuples():
        L.append(f"  [{r.rule}] {r.A} vs {r.B}: {100 * r.bal_diff:+.1f} [{100 * r.bal_diff_ci_low:+.1f}, {100 * r.bal_diff_ci_high:+.1f}], "
                 f"seeds A better {r.seeds_A_better}/5, splits up/down {r.splits_up}/{r.splits_down}, corrected t p "
                 f"{r.corrected_t_p:.3f}" + (f" (Holm {r.holm_p:.3f})" if not np.isnan(r.holm_p) else "")
                 + f", macro-F1 fold diff {100 * r.fold_f1_diff_mean:+.1f} -> {r.direction}")
    L.append(f"\nNESTED CHOICE per split: {ch.to_dict()}; NESTED_K choice: {ch_k.to_dict()}")
    L.append("\nINNER vs HELD-OUT and SELECTION STABILITY per setting")
    for r in inner.itertuples():
        L.append(f"  {r.config}: inner {100 * r.inner_bal_mean:.1f} vs held-out fold mean {100 * r.heldout_fold_bal_mean:.1f} "
                 f"({100 * r.inner_minus_heldout:+.1f}); chosen {r.times_chosen}x; Jaccard {r.selection_jaccard:.2f}; "
                 f"features in >= half of fits {r.features_in_half_of_fits}")
    L.append("\nTRANSFORMER FIT DIAGNOSTICS")
    for r in fit.itertuples():
        L.append(f"  {r.model}: {r.n_parameters} params; val loss min {r.val_loss_min:.3f} at epoch {r.val_loss_min_epoch}, "
                 f"epoch 60 {r.val_loss_epoch60:.3f}; train loss 60 {r.train_loss_epoch60:.3f}; train bal 60 "
                 f"{100 * r.train_bal_epoch60:.1f}; fixed epoch median {r.fixed_epoch_median:.0f}; inner-heldout gap "
                 f"{100 * r.inner_minus_heldout_no_selection:+.1f}; overfit condition {r.overfit_condition}")
    L.append(f"\nSITE CONFOUND: site-only rule {100 * site_bal:.1f}%; within-site permutation null "
             f"{[round(100 * v, 1) for v in null_site]} (mean {100 * np.mean(null_site):.1f}); global null "
             f"{[round(100 * v, 1) for v in null_glob]}; baseline {100 * b0:.1f}% -> R4 exceeds within-site null: {r4}")
    L.append(f"COMBAT (existing leakage_safe_v1, seeds {cb['none'][2]}): none {100 * cb['none'][0]:.1f}% / F1 "
             f"{100 * cb['none'][1]:.1f}; combat {100 * cb['combat'][0]:.1f}% / F1 {100 * cb['combat'][1]:.1f}; "
             f"combat - none {100 * cbp['diff']:+.1f} [{100 * cbp['ci'][0]:+.1f}, {100 * cbp['ci'][1]:+.1f}]")
    L.append(f"\nDECISIONS (pre-registered rules): {json.dumps(dec)}")
    rt = sum(r["seconds"] for c in tab for r in tab[c].values()) + sum(r["seconds"] for v in tv for r in tv[v].values())
    L.append(f"RUNTIME: {rt / 60:.0f} min of compute")
    open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "decisions": dec, "reproduction": repro,
               "ci_definition": __doc__.split("Run:")[0].strip(), "nested_choice_counts": ch.to_dict(),
               "nested_k_choice_counts": ch_k.to_dict(),
               "within_site_null": null_site, "global_null": null_glob, "site_only_rule": site_bal},
              open(os.path.join(OUT, "manifest.json"), "w"), indent=2, default=str)
    print("\n".join(L))


if __name__ == "__main__":
    main()
