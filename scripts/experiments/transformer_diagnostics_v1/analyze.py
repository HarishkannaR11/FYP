"""
Diagnostic analysis: Transformer vs TabPFN, from saved predictions and curves only.

Sources (all read-only):
  derivatives/experiments/transformer_diagnostics_v1/grid/C*/        epoch curves + predictions (this study)
  derivatives/model_training/transformer_svmrfe/cv116_*.json          earlier transformer runs (k = 50, all 246)
  derivatives/model_training/tabpfn/cv116*_tabpfn_*.json             earlier TabPFN runs (k = 50)
  derivatives/experiments/leakage_safe_v1/grid/main/                  TabPFN at k = 25..246 (completed splits)
  derivatives/experiments/leakage_safe_v1/audit/audit.json            data audit

Writes derivatives/experiments/transformer_diagnostics_v1/analysis/ (refuses to overwrite).

Metric convention: for each seed, the 5 test folds are pooled into one out-of-fold
prediction per subject (116), metrics are computed on that, then mean +/- sd over
seeds. Fold-level sd is reported separately. CIs: stratified subject bootstrap
(2000) of the seed-averaged balanced accuracy; paired differences use the same
bootstrap samples for both models.

Run:  python scripts/experiments/transformer_diagnostics_v1/analyze.py
"""
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "leakage_safe_v1"))
import report as R  # noqa: E402  (bootstrap, palette, common)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

K, P = R.K, R.P
ROOT = K.ROOT
EXP = os.path.join(ROOT, "derivatives", "experiments", "transformer_diagnostics_v1")
OUT = os.path.join(EXP, "analysis")
STAGES, NS = K.STAGES, K.NS
CONFIG_TEXT = {"C0": "baseline (lr 1e-3, dropout 0.3, wd 1e-2, d16 L1)", "C1": "lr 3e-4", "C2": "lr 3e-3",
               "C3": "dropout 0.1", "C4": "weight decay 1e-1", "C5": "larger (d32, 4 heads, 2 layers)"}


# ------------------------------------------------------------------ sources
def from_records(path, key="pred"):
    out = {}
    if os.path.exists(path):
        for r in json.load(open(path))["records"]:
            out[(r["seed"], r["fold"])] = (np.array(r["test_index"]), np.array(r[key]))
    return out


def from_grid(cname, key="pred_es"):
    out, recs = {}, {}
    for f in glob.glob(os.path.join(EXP, "grid", cname, "cv5_*.json")):
        r = json.load(open(f))
        recs[(r["seed"], r["fold"])] = r
        out[(r["seed"], r["fold"])] = (np.array(r["test_idx"]), np.array(r[key]))
    return out, recs


def from_v1(cid):
    out = {}
    for f in glob.glob(os.path.join(ROOT, "derivatives", "experiments", "leakage_safe_v1", "grid", "main", "cv5_*.json")):
        r = json.load(open(f))
        if cid in r["configs"]:
            out[(r["seed"], r["fold"])] = (np.array(r["test_idx"]), np.array(r["configs"][cid]["pred"]))
    return out


def full_seeds(src, seeds=None):
    return [s for s in (seeds or K.SEEDS) if all((s, f) in src for f in range(5))]


def oof(src, seed, n=116):
    p = np.full(n, -1)
    for f in range(5):
        te, pr = src[(seed, f)]
        p[te] = pr
    assert (p >= 0).all()
    return p


def summarize(src, y, seeds):
    rows = [K.H.evaluate(y, oof(src, s)) for s in seeds]
    folds = [K.bal_acc(y[src[(s, f)][0]], src[(s, f)][1]) for s in seeds for f in range(5)]
    A = np.vstack([oof(src, s) for s in seeds])
    (blo, bhi), (alo, ahi) = R.boot(y, A)
    cm = np.zeros((NS, NS), int)
    for a, b in zip(np.tile(y, len(seeds)), A.ravel()):
        cm[a, b] += 1
    m = lambda k: (float(np.mean([r[k] for r in rows])), float(np.std([r[k] for r in rows])))  # noqa: E731
    out = {"seeds": len(seeds), "predictions": int(A.size), "accuracy": m("accuracy")[0], "acc_sd": m("accuracy")[1],
           "acc_ci": [alo, ahi], "bal": m("balanced_accuracy")[0], "bal_sd": m("balanced_accuracy")[1],
           "bal_fold_sd": float(np.std(folds)), "bal_ci": [blo, bhi], "f1": m("macro_f1")[0], "f1_sd": m("macro_f1")[1],
           "qwk": m("qwk")[0], "mae": m("mae_stages")[0],
           "per_seed_bal": [r["balanced_accuracy"] for r in rows],
           "recall": {STAGES[g]: float(cm[g, g] / cm[g].sum()) for g in range(NS)},
           "pred_share": {STAGES[g]: float(cm[:, g].sum() / cm.sum()) for g in range(NS)}}
    return out, cm, A


def paired(srcA, srcB, y, seeds):
    a = np.vstack([oof(srcA, s) for s in seeds])
    b = np.vstack([oof(srcB, s) for s in seeds])
    da = np.array([K.bal_acc(y, x) for x in a]) - np.array([K.bal_acc(y, x) for x in b])
    lo, hi = R.boot(y, a, b)
    return {"diff": float(da.mean()), "ci": [lo, hi], "seeds_better": int((da > 0).sum()), "n_seeds": len(seeds),
            "verdict": ("first better" if lo > 0 and (da > 0).sum() >= int(np.ceil(0.8 * len(seeds))) else
                        "second better" if hi < 0 and (da < 0).sum() >= int(np.ceil(0.8 * len(seeds))) else
                        "no significant difference")}


# ------------------------------------------------------------------ curves
def curve_stats(recs):
    """Per epoch mean/sd over all inner models of all splits."""
    out = {}
    for st in ("train", "val", "outer"):
        for mt in ("loss", "acc", "bal", "f1"):
            M = np.array([m["curves"][st][mt] for r in recs.values() for m in r["models"]])
            out[f"{st}_{mt}_mean"], out[f"{st}_{mt}_sd"] = M.mean(0), M.std(0)
    B = np.array([r["bag_outer_curve"]["bal"] for r in recs.values()])
    out["bag_outer_bal_mean"], out["bag_outer_bal_sd"] = B.mean(0), B.std(0)
    return pd.DataFrame({"epoch": np.arange(1, B.shape[1] + 1), **out})


def fit_diagnosis(recs):
    rows = []
    for (s, f), r in recs.items():
        for j, m in enumerate(r["models"]):
            c = m["curves"]
            be, last = m["best_epoch"], len(c["val"]["bal"]) - 1
            vl = np.array(c["val"]["loss"])
            tl = np.array(c["train"]["loss"])
            rows.append({"seed": s, "fold": f, "inner_model": j, "best_epoch": be + 1, "stop_epoch": m["stop_epoch"] + 1,
                         "train_loss_at_best": tl[be], "val_loss_at_best": float(vl[be]),
                         "loss_gap_at_best": float(vl[be] - tl[be]), "loss_gap_final": float(vl[last] - tl[last]),
                         "train_f1_at_best": c["train"]["f1"][be], "val_f1_at_best": c["val"]["f1"][be],
                         "train_bal_at_best": c["train"]["bal"][be], "val_bal_at_best": c["val"]["bal"][be],
                         "outer_bal_at_best": c["outer"]["bal"][be], "train_bal_final": c["train"]["bal"][last],
                         "val_bal_final": c["val"]["bal"][last], "train_loss_first": tl[0], "train_loss_final": tl[last],
                         "val_loss_min_epoch": int(vl.argmin()) + 1, "val_loss_min": float(vl.min()),
                         "val_loss_final": float(vl[last]),
                         "val_bal_jitter": float(np.mean(np.abs(np.diff(c["val"]["bal"])))),
                         "early": be + 1 <= 5,
                         "underfit_at_best": c["train"]["bal"][be] < 0.5,
                         "overfit_later": (c["train"]["bal"][last] - c["val"]["bal"][last] > 0.3) and vl[last] > 1.05 * vl.min()})
    return pd.DataFrame(rows)


def main():
    if os.path.exists(os.path.join(OUT, "model_comparison.csv")):
        raise SystemExit("analysis already written; not overwritten")
    os.makedirs(OUT, exist_ok=True)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    y = np.load(os.path.join(ROOT, "derivatives", "model_ready", "y.npy")).astype(int)
    L = []

    # ---------------- sources
    grid = {c: from_grid(c) for c in ("C0", "C1", "C2", "C3", "C4", "C5")}
    srcs = {
        "Transformer C0 (rerun, logged)": grid["C0"][0],
        "Transformer k=50 (earlier saved run)": from_records(os.path.join(ROOT, "derivatives", "model_training",
                                                                          "transformer_svmrfe", "cv116_transformer_svmrfe50.json")),
        "Transformer, all 246 ROI tokens (earlier)": from_records(os.path.join(ROOT, "derivatives", "model_training",
                                                                               "transformer_svmrfe", "cv116_transformer_all246.json")),
        "TabPFN k=50 (earlier saved run)": from_records(os.path.join(ROOT, "derivatives", "model_training", "tabpfn",
                                                                     "cv116_tabpfn_svmrfe50.json")),
    }
    for k in (25, 50, 100, 150, 246):
        srcs[f"TabPFN k={k} (leakage-safe grid)"] = from_v1(f"tabpfn|img|k{k}|none|-")
    for c in ("C1", "C2", "C3", "C4", "C5"):
        srcs[f"Transformer {c}: {CONFIG_TEXT[c]}"] = grid[c][0]
    srcs["Transformer C0, no early stopping (epoch 60)"] = from_grid("C0", "pred_last")[0]
    # nested choice of configuration per split (inner-validation score only)
    nested, choice = {}, []
    common = [k for k in grid["C0"][1] if all(k in grid[c][1] for c in grid)]
    for key in common:
        sc = {c: grid[c][1][key]["inner_score"] for c in grid}
        best = max(sorted(sc), key=lambda c: (sc[c], c == "C0"))
        nested[key] = grid[best][0][key]
        choice.append({"seed": key[0], "fold": key[1], "chosen": best, **{f"inner_{c}": v for c, v in sc.items()}})
    srcs["Transformer, configuration chosen per split by inner CV"] = nested

    # ---------------- reproduction
    rep = {}
    a, b = srcs["Transformer C0 (rerun, logged)"], srcs["Transformer k=50 (earlier saved run)"]
    rep["transformer_C0_vs_earlier"] = {"splits": len(set(a) & set(b)),
                                        "identical": int(sum(np.array_equal(a[k][1], b[k][1]) for k in set(a) & set(b)))}
    t1, t2 = srcs["TabPFN k=50 (leakage-safe grid)"], srcs["TabPFN k=50 (earlier saved run)"]
    rep["tabpfn_k50_grid_vs_earlier"] = {"splits": len(set(t1) & set(t2)),
                                         "identical": int(sum(np.array_equal(t1[k][1], t2[k][1]) for k in set(t1) & set(t2)))}

    # ---------------- model comparison on all available seeds
    rows, cms = [], {}
    for name, src in srcs.items():
        seeds = full_seeds(src)
        if not seeds:
            continue
        s, cm, _ = summarize(src, y, seeds)
        cms[name] = cm
        rows.append({"model": name, **{k: v for k, v in s.items() if k not in ("recall", "pred_share", "per_seed_bal")},
                     **{f"recall_{g}": v for g, v in s["recall"].items()},
                     **{f"predicted_share_{g}": v for g, v in s["pred_share"].items()},
                     "per_seed_bal": json.dumps(np.round(s["per_seed_bal"], 4).tolist()),
                     "seeds_used": json.dumps(seeds)})
    comp = pd.DataFrame(rows)
    comp.to_csv(os.path.join(OUT, "model_comparison.csv"), index=False)
    shared_seeds = sorted(set.intersection(*[set(full_seeds(grid[c][0])) for c in grid if grid[c][0]]))
    mrows = []
    for name in ["TabPFN k=50 (earlier saved run)", "Transformer C0 (rerun, logged)"] + \
            [f"Transformer {c}: {CONFIG_TEXT[c]}" for c in ("C1", "C2", "C3", "C4", "C5")] + \
            ["Transformer, configuration chosen per split by inner CV", "Transformer C0, no early stopping (epoch 60)"]:
        if name in srcs and set(shared_seeds) <= set(full_seeds(srcs[name])) and shared_seeds:
            s_, _, _ = summarize(srcs[name], y, shared_seeds)
            fb = [K.bal_acc(y[srcs[name][(sd, f)][0]], srcs[name][(sd, f)][1]) for sd in shared_seeds for f in range(5)]
            ff = [K.H.evaluate(y[srcs[name][(sd, f)][0]], srcs[name][(sd, f)][1])["macro_f1"] for sd in shared_seeds for f in range(5)]
            mrows.append({"model": name, "seeds": json.dumps(shared_seeds), "heldout_accuracy": s_["accuracy"],
                          "heldout_bal": s_["bal"], "heldout_bal_ci": json.dumps(np.round(s_["bal_ci"], 4).tolist()),
                          "heldout_macro_f1": s_["f1"], "fold_bal_mean": float(np.mean(fb)), "fold_bal_sd": float(np.std(fb)),
                          "fold_bal_min": float(np.min(fb)), "fold_bal_max": float(np.max(fb)),
                          "fold_f1_mean": float(np.mean(ff)), "fold_f1_sd": float(np.std(ff)), "qwk": s_["qwk"],
                          **{f"recall_{g}": v for g, v in s_["recall"].items()}})
    matched = pd.DataFrame(mrows)
    matched.to_csv(os.path.join(OUT, "model_comparison_shared_seeds.csv"), index=False)

    # ---------------- paired comparisons on identical folds
    prs = []
    for a_, b_ in [("Transformer C0 (rerun, logged)", "TabPFN k=50 (earlier saved run)"),
                   ("Transformer, all 246 ROI tokens (earlier)", "Transformer C0 (rerun, logged)"),
                   ("Transformer, configuration chosen per split by inner CV", "Transformer C0 (rerun, logged)"),
                   ("Transformer, configuration chosen per split by inner CV", "TabPFN k=50 (earlier saved run)"),
                   ("Transformer C0, no early stopping (epoch 60)", "Transformer C0 (rerun, logged)")] + \
                  [(f"Transformer {c}: {CONFIG_TEXT[c]}", "Transformer C0 (rerun, logged)") for c in ("C1", "C2", "C3", "C4", "C5")] + \
                  [(f"Transformer {c}: {CONFIG_TEXT[c]}", "TabPFN k=50 (earlier saved run)") for c in ("C1", "C2", "C3", "C4", "C5")] + \
                  [(f"TabPFN k={k} (leakage-safe grid)", "Transformer C0 (rerun, logged)") for k in (25, 50, 100, 150, 246)]:
        seeds = sorted(set(full_seeds(srcs[a_])) & set(full_seeds(srcs[b_])))
        if seeds:
            prs.append({"A": a_, "B": b_, "seeds": json.dumps(seeds), **paired(srcs[a_], srcs[b_], y, seeds)})
    pr = pd.DataFrame(prs)
    pr.to_csv(os.path.join(OUT, "paired_comparisons.csv"), index=False)
    pd.DataFrame(choice).to_csv(os.path.join(OUT, "config_choice_per_split.csv"), index=False)

    # ---------------- curves + fit diagnosis
    diag_rows, cstats = [], {}
    for c, (_, recs) in grid.items():
        if not recs:
            continue
        cstats[c] = curve_stats(recs)
        cstats[c].to_csv(os.path.join(OUT, f"curves_{c}.csv"), index=False)
        fd = fit_diagnosis(recs)
        fd.insert(0, "config", c)
        diag_rows.append(fd)
    fdall = pd.concat(diag_rows)
    fdall.to_csv(os.path.join(OUT, "best_epochs_and_fit.csv"), index=False)

    # ---------------- per-split validation table and consistency of each configuration vs C0
    split_rows = []
    for c, (src, recs) in grid.items():
        for key, r in recs.items():
            te_true = np.array(r["true"])
            ms = r["models"]
            be = [m["best_epoch"] for m in ms]
            split_rows.append({
                "config": c, "seed": key[0], "fold": key[1],
                "outer_bal": K.bal_acc(te_true, np.array(r["pred_es"])),
                "outer_f1": K.H.evaluate(te_true, np.array(r["pred_es"]))["macro_f1"],
                "inner_val_bal_at_best": float(np.mean([m["curves"]["val"]["bal"][b] for m, b in zip(ms, be)])),
                "inner_val_f1_at_best": float(np.mean([m["curves"]["val"]["f1"][b] for m, b in zip(ms, be)])),
                "train_bal_at_best": float(np.mean([m["curves"]["train"]["bal"][b] for m, b in zip(ms, be)])),
                "loss_gap_at_best": float(np.mean([m["curves"]["val"]["loss"][b] - m["curves"]["train"]["loss"][b]
                                                   for m, b in zip(ms, be)])),
                "loss_gap_final": float(np.mean([m["curves"]["val"]["loss"][-1] - m["curves"]["train"]["loss"][-1] for m in ms])),
                "best_epoch_median": float(np.median(be)) + 1})
    sp = pd.DataFrame(split_rows)
    sp.to_csv(os.path.join(OUT, "per_split_validation.csv"), index=False)
    from scipy.stats import binomtest
    cons = []
    base = sp[sp.config == "C0"].set_index(["seed", "fold"])
    for c in ("C1", "C2", "C3", "C4", "C5"):
        q = sp[sp.config == c].set_index(["seed", "fold"])
        common_k = q.index.intersection(base.index)
        if not len(common_k):
            continue
        row = {"config": c, "description": CONFIG_TEXT[c], "n_splits": len(common_k)}
        for mt in ("outer_bal", "outer_f1", "inner_val_bal_at_best", "inner_val_f1_at_best", "train_bal_at_best",
                   "loss_gap_at_best", "loss_gap_final"):
            dd = (q.loc[common_k, mt] - base.loc[common_k, mt]).to_numpy()
            row[f"{mt}_diff_mean"] = float(dd.mean())
            row[f"{mt}_splits_up"] = int((dd > 0).sum())
            row[f"{mt}_splits_down"] = int((dd < 0).sum())
        n_up, n_dn = row["outer_bal_splits_up"], row["outer_bal_splits_down"]
        row["outer_bal_sign_test_p"] = float(binomtest(n_up, n_up + n_dn).pvalue) if n_up + n_dn else 1.0
        name = f"Transformer {c}: {CONFIG_TEXT[c]}"
        prr = pr[(pr.A == name) & (pr.B == "Transformer C0 (rerun, logged)")]
        if len(prr):
            row["seed_pooled_bal_diff"] = prr.iloc[0]["diff"]
            row["seed_pooled_bal_ci"] = json.dumps(np.round(prr.iloc[0]["ci"], 4).tolist())
        row["consistent_validation_gain"] = bool(n_up >= int(np.ceil(0.8 * (n_up + n_dn))) and row["outer_bal_diff_mean"] > 0
                                                 and row["outer_bal_sign_test_p"] < 0.05)
        cons.append(row)
    consd = pd.DataFrame(cons)
    consd.to_csv(os.path.join(OUT, "config_consistency_vs_C0.csv"), index=False)

    # ---------------- confusion matrices
    cmrows = []
    for name, cm in cms.items():
        for g in range(NS):
            for h in range(NS):
                cmrows.append({"model": name, "true": STAGES[g], "pred": STAGES[h], "count": int(cm[g, h])})
    pd.DataFrame(cmrows).to_csv(os.path.join(OUT, "confusion_matrices.csv"), index=False)

    # ---------------- figures
    c0 = cstats["C0"]
    fig, axs = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, mt, ttl in ((axs[0], "loss", "Cross-entropy loss"), (axs[1], "bal", "Balanced accuracy"), (axs[2], "f1", "Macro-F1")):
        for j, (st, lab) in enumerate((("train", "training subjects"), ("val", "inner validation (early stopping)"),
                                       ("outer", "held-out fold (diagnostic only)"))):
            m_, s_ = c0[f"{st}_{mt}_mean"], c0[f"{st}_{mt}_sd"]
            ax.plot(c0.epoch, m_, color=P.SERIES[j], label=lab)
            ax.fill_between(c0.epoch, m_ - s_, m_ + s_, color=P.SERIES[j], alpha=0.12, lw=0)
        if mt != "loss":
            ax.axhline(1 / 6, color=P.MUTED, ls="--", lw=1)
        else:
            ax.axhline(np.log(6), color=P.MUTED, ls="--", lw=1)
        ax.set_title(ttl, loc="left", fontsize=9)
        ax.set_xlabel("epoch")
        ax.yaxis.grid(True)
    be = fdall[fdall.config == "C0"].best_epoch
    axs[1].axvline(be.median(), color=P.INK2, lw=1, ls=":")
    axs[1].text(be.median(), axs[1].get_ylim()[1], f" median best epoch {be.median():.0f}", fontsize=7.5,
                va="top", color=P.INK2)
    axs[0].legend(loc="upper left", bbox_to_anchor=(0, -0.18), ncol=3, fontsize=8)
    fig.suptitle("Transformer (baseline C0) epoch-wise curves: mean +/- sd over 25 outer splits x 4 inner models "
                 "(dashed: chance / uniform-prediction loss)", x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.savefig(os.path.join(OUT, "fig_curves_C0.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
    axs[0].hist(be, bins=np.arange(0.5, 61.5, 2), color=P.SERIES[0])
    axs[0].set_xlabel("best inner-validation epoch")
    axs[0].set_ylabel("inner models")
    axs[0].set_title(f"Best epoch, C0 (100 inner models; median {be.median():.0f})", loc="left", fontsize=9)
    for c, st in cstats.items():
        axs[1].plot(st.epoch, st.bag_outer_bal_mean, label=f"{c}: {CONFIG_TEXT[c]}")
    axs[1].axhline(1 / 6, color=P.MUTED, ls="--", lw=1)
    axs[1].set_xlabel("epoch at which the 4-model bag is read out")
    axs[1].set_ylabel("held-out balanced accuracy")
    axs[1].set_title("Held-out accuracy if training stopped at a fixed epoch (diagnostic only)", loc="left", fontsize=9)
    axs[1].legend(fontsize=6.5, loc="lower right")
    axs[1].yaxis.grid(True)
    fig.savefig(os.path.join(OUT, "fig_best_epoch_and_fixed_epoch.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axs = plt.subplots(1, 3, figsize=(16, 3.8))
    colours = [P.SERIES[0], P.SERIES[1], P.SERIES[2], P.INK2, P.MUTED, "#9085e9"]
    for j, (c, st) in enumerate(cstats.items()):
        gap = st["val_loss_mean"] - st["train_loss_mean"]
        axs[0].plot(st.epoch, gap, color=colours[j], label=f"{c}: {CONFIG_TEXT[c]}")
    axs[0].axhline(0, color=P.AXIS, lw=1)
    axs[0].set_xlabel("epoch")
    axs[0].set_ylabel("validation loss - training loss")
    axs[0].set_title("Training-validation loss gap (mean over inner models)", loc="left", fontsize=9)
    axs[0].legend(fontsize=6.5)
    axs[0].yaxis.grid(True)
    cfgs = [c for c in cstats]
    axs[1].boxplot([fdall[fdall.config == c].best_epoch for c in cfgs], labels=cfgs, showfliers=True,
                   medianprops={"color": P.INK})
    axs[1].set_ylabel("best inner-validation epoch")
    axs[1].set_title("Best-epoch distribution per configuration", loc="left", fontsize=9)
    axs[1].yaxis.grid(True)
    shared = set(sp[sp.config == "C1"][["seed", "fold"]].apply(tuple, axis=1).tolist())
    for j, c in enumerate(cfgs):
        q = sp[sp.config == c]
        v = q[q[["seed", "fold"]].apply(tuple, axis=1).isin(shared)].outer_bal
        axs[2].scatter(np.full(len(v), j) + np.random.default_rng(j).uniform(-0.12, 0.12, len(v)), v, s=12,
                       color=P.SERIES[0], alpha=0.6)
        axs[2].scatter([j], [v.mean()], s=60, marker="D", color=P.SERIES[1], zorder=3)
    axs[2].axhline(1 / 6, color=P.MUTED, ls="--", lw=1)
    axs[2].set_xticks(range(len(cfgs)), cfgs)
    axs[2].set_ylabel("held-out balanced accuracy per split")
    axs[2].set_title("Held-out fold accuracy on the 10 shared splits (diamond = mean)", loc="left", fontsize=9)
    axs[2].yaxis.grid(True)
    fig.savefig(os.path.join(OUT, "fig_gap_bestepoch_configs.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    keys = ["Transformer C0 (rerun, logged)", "TabPFN k=50 (earlier saved run)"]
    fig, axs = plt.subplots(1, 3, figsize=(14, 4))
    for ax, nm in zip(axs[:2], keys):
        cm = cms[nm]
        Mn = cm / cm.sum(1, keepdims=True)
        ax.imshow(Mn, cmap=P.SEQ, vmin=0, vmax=max(0.5, Mn.max()))
        for a_ in range(NS):
            for b_ in range(NS):
                ax.text(b_, a_, f"{Mn[a_, b_]:.2f}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if Mn[a_, b_] > 0.3 else P.INK2)
        ax.set_xticks(range(NS), STAGES, fontsize=7)
        ax.set_yticks(range(NS), STAGES, fontsize=7)
        ax.set_title(nm, loc="left", fontsize=8)
        ax.set_xlabel("predicted")
        ax.tick_params(length=0)
        for s_ in ax.spines.values():
            s_.set_visible(False)
    axs[0].set_ylabel("true")
    x = np.arange(NS)
    for j, nm in enumerate(keys):
        r = comp[comp.model == nm].iloc[0]
        axs[2].bar(x + (j - 0.5) * 0.38, [r[f"recall_{g}"] for g in STAGES], width=0.38, color=P.SERIES[j], label=nm)
    axs[2].axhline(1 / 6, color=P.MUTED, ls="--", lw=1)
    axs[2].set_xticks(x, STAGES)
    axs[2].set_title("Per-class recall (pooled over 5 seeds)", loc="left", fontsize=9)
    axs[2].legend(fontsize=7)
    axs[2].yaxis.grid(True)
    fig.savefig(os.path.join(OUT, "fig_confusion_recall.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---------------- text summary
    def line(r):
        return (f"{r.model}: accuracy {100 * r.accuracy:.1f}% (sd {100 * r.acc_sd:.1f}), balanced {100 * r.bal:.1f}% "
                f"(seed sd {100 * r.bal_sd:.1f}, fold sd {100 * r.bal_fold_sd:.1f}, 95% CI {100 * json.loads(json.dumps(r.bal_ci))[0]:.1f}-"
                f"{100 * json.loads(json.dumps(r.bal_ci))[1]:.1f}), macro-F1 {100 * r.f1:.1f}%, QWK {r.qwk:+.3f}, "
                f"seeds {r.seeds}, predictions {r.predictions}")
    L.append("MODEL COMPARISON (seed-pooled out-of-fold metrics)")
    L += ["  " + line(r) for r in comp.itertuples()]
    L.append("\nPAIRED (balanced accuracy, A - B, identical folds)")
    for r in pr.itertuples():
        L.append(f"  {r.A}  vs  {r.B}: {100 * r.diff:+.1f} [{100 * r.ci[0]:+.1f}, {100 * r.ci[1]:+.1f}], "
                 f"seeds A better {r.seeds_better}/{r.n_seeds} -> {r.verdict}")
    L.append("\nFIT DIAGNOSIS per configuration (inner models)")
    for c, g in fdall.groupby("config"):
        L.append(f"  {c}: n {len(g)}, best epoch median {g.best_epoch.median():.0f} (IQR {g.best_epoch.quantile(.25):.0f}-"
                 f"{g.best_epoch.quantile(.75):.0f}), best <= 5: {100 * g.early.mean():.0f}%, train bal at best "
                 f"{100 * g.train_bal_at_best.mean():.1f}%, val bal at best {100 * g.val_bal_at_best.mean():.1f}%, "
                 f"outer bal at best {100 * g.outer_bal_at_best.mean():.1f}%, train bal final {100 * g.train_bal_final.mean():.1f}%, "
                 f"val bal final {100 * g.val_bal_final.mean():.1f}%, val-loss min epoch median {g.val_loss_min_epoch.median():.0f}, "
                 f"val loss min {g.val_loss_min.mean():.3f} -> final {g.val_loss_final.mean():.3f}, "
                 f"epoch-to-epoch val jitter {100 * g.val_bal_jitter.mean():.1f} pts, overfit later {100 * g.overfit_later.mean():.0f}%")
    L.append("\nLOSS GAP (validation - training) and validation metrics, per configuration (all splits run)")
    for c, g in fdall.groupby("config"):
        q = sp[sp.config == c]
        L.append(f"  {c}: gap at best epoch {g.loss_gap_at_best.mean():+.3f} (sd {g.loss_gap_at_best.std():.3f}), "
                 f"gap at epoch 60 {g.loss_gap_final.mean():+.3f} (sd {g.loss_gap_final.std():.3f}); "
                 f"inner-val bal at best {100 * q.inner_val_bal_at_best.mean():.1f}% (sd {100 * q.inner_val_bal_at_best.std():.1f}), "
                 f"inner-val F1 at best {100 * q.inner_val_f1_at_best.mean():.1f}%; held-out bal {100 * q.outer_bal.mean():.1f}% "
                 f"(sd {100 * q.outer_bal.std():.1f}), held-out F1 {100 * q.outer_f1.mean():.1f}% (sd {100 * q.outer_f1.std():.1f}); "
                 f"splits {len(q)}")
    if len(consd):
        L.append("\nCONSISTENCY vs C0 (same splits; held-out = outer fold, never used for any choice)")
        for r in consd.to_dict("records"):
            L.append(f"  {r['config']} ({r['description']}): held-out bal {100 * r['outer_bal_diff_mean']:+.1f} pts, up in "
                     f"{r['outer_bal_splits_up']}/{r['n_splits']} splits (sign p {r['outer_bal_sign_test_p']:.2f}); held-out F1 "
                     f"{100 * r['outer_f1_diff_mean']:+.1f}; inner-val bal {100 * r['inner_val_bal_at_best_diff_mean']:+.1f}; "
                     f"TRAIN bal {100 * r['train_bal_at_best_diff_mean']:+.1f}; gap at best {r['loss_gap_at_best_diff_mean']:+.3f}; "
                     f"seed-pooled diff {100 * r.get('seed_pooled_bal_diff', float('nan')):+.1f} "
                     f"CI {r.get('seed_pooled_bal_ci', '-')}; consistent gain: {r['consistent_validation_gain']}")
    if choice:
        L.append("\nCONFIG CHOSEN PER SPLIT (inner CV): " + json.dumps(pd.DataFrame(choice).chosen.value_counts().to_dict()))
    if len(matched):
        L.append(f"\nMATCHED HELD-OUT COMPARISON on the seeds every configuration shares {json.loads(matched.seeds.iloc[0])} "
                 "(10 identical outer splits)")
        for r in matched.to_dict("records"):
            L.append(f"  {r['model']}: held-out bal {100 * r['heldout_bal']:.1f}% CI {r['heldout_bal_ci']}, macro-F1 "
                     f"{100 * r['heldout_macro_f1']:.1f}%, fold bal sd {100 * r['fold_bal_sd']:.1f} (range "
                     f"{100 * r['fold_bal_min']:.0f}-{100 * r['fold_bal_max']:.0f}), recall "
                     + ", ".join(f"{g} {100 * r['recall_' + g]:.0f}" for g in STAGES))
    L.append("\nREPRODUCTION: " + json.dumps(rep))
    open(os.path.join(OUT, "summary.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump({"started": started, "reproduction": rep,
               "sources": {"grid": os.path.relpath(os.path.join(EXP, "grid"), ROOT)}},
              open(os.path.join(OUT, "manifest.json"), "w"), indent=1)
    print("\n".join(L))


if __name__ == "__main__":
    main()
