"""
Aggregate the leakage-safe experiment set into the Phase 4 deliverables.
Reads only saved per-split records (grid/*), the site diagnostic and the
locked test results; trains nothing.

  results/oof_predictions.csv      every subject's out-of-fold prediction, per config and seed
  results/model_comparison.csv     all metrics, 95% bootstrap CIs, paired tests vs the baseline (A)
  results/confusion_matrices.csv   per-class confusion counts, pooled over seeds
  results/feature_count.csv        balanced accuracy vs number of features
  results/harmonization_comparison.csv   ComBat vs none, paired, + site predictability
  results/nested_selection.csv     configuration chosen by the inner loop in every outer split
  results/reproduction_check.json  new TabPFN k=50 runs vs the earlier saved predictions
  results/summary_tables.md        tables used in the written report
  figures/*.png                    feature count, harmonisation, confusion matrices

"Tuned" rows: for LR / SVM / ordinal LR the hyperparameter is chosen per outer
split by inner-CV balanced accuracy. "Nested" rows: the whole configuration
(model, feature count, harmonisation, hyperparameters) is chosen per outer
split by inner CV. An improvement is claimed only if the 95% CI of the paired
difference excludes 0 AND at least 4 of 5 seeds agree.

Run:  python scripts/experiments/leakage_safe_v1/report.py
"""
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as K  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                                "scripts", "model"))
import stage_patterns as P  # noqa: E402  (plot style + palette)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

RES, FIG = os.path.join(K.EXP, "results"), os.path.join(K.EXP, "figures")
B = 2000
FAM_LABEL = {"tabpfn": "TabPFN v2", "lr": "Logistic regression (L2)", "svm": "RBF-SVM",
             "ordinal": "Ordinal logistic regression", "majority": "Majority class", "nested": "Nested selection"}
IN_LABEL = {"img": "rs-fMRI", "imgcom": "rs-fMRI + 8 comorbidities", "-": "-"}


def load_job(job):
    out = {}
    for f in sorted(glob.glob(os.path.join(K.EXP, "grid", job, "cv5_*.json"))):
        r = json.load(open(f))
        out[(r["seed"], r["fold"])] = r
    return out


# --------------------------------------------------------------- assembling predictions
class Table:
    """Per config: {(seed, fold): {"pred", "probs", "test_idx", "inner", "train_bal"}}"""

    def __init__(self):
        self.c = {}

    def add(self, cid, key, test_idx, pred, probs=None, inner=None, train_bal=None):
        self.c.setdefault(cid, {})[key] = {"test_idx": test_idx, "pred": np.asarray(pred), "probs": probs,
                                           "inner": inner, "train_bal": train_bal}

    def seeds(self, cid):
        keys = self.c.get(cid, {})
        return [s for s in K.SEEDS if all((s, f) in keys for f in range(5))]

    def oof(self, cid, seed, n=116):
        pred = np.full(n, -1)
        probs = np.full((n, K.NS), np.nan)
        fold = np.full(n, -1)
        for f in range(5):
            r = self.c[cid][(seed, f)]
            pred[r["test_idx"]] = r["pred"]
            fold[r["test_idx"]] = f
            if r["probs"] is not None:
                probs[r["test_idx"]] = np.asarray(r["probs"])
        return pred, probs, fold


def build(main, nested, bio, shuf):
    T, chosen = Table(), []
    for key, r in main.items():
        te = r["test_idx"]
        for c, v in r["configs"].items():
            T.add(c, key, te, v["pred"], v.get("probs"), v.get("inner"), v.get("train_bal"))
    for key, r in nested.items():
        for c, v in r["configs"].items():
            if c in T.c and key in T.c[c]:
                T.c[c][key]["inner"] = v["inner"]
    for key, r in bio.items():
        for c, v in r["configs"].items():
            T.add(c, key, r["test_idx"], v["pred"], v.get("probs"), v.get("inner"), v.get("train_bal"))
    # tuned light models: hyperparameter chosen per split by inner CV
    groups = {}
    for c in list(T.c):
        fam, inputs, view, harm, hp = K.parse_cid(c)
        if fam in K.LIGHT:
            groups.setdefault((fam, inputs, view, harm), []).append(c)
    for (fam, inputs, view, harm), cs in groups.items():
        for key in T.c[cs[0]]:
            sc = {c: float(np.mean(T.c[c][key]["inner"])) for c in cs if key in T.c[c] and T.c[c][key]["inner"]}
            if not sc:
                continue
            best = K.select_best(sc)
            b = T.c[best][key]
            T.add(K.cid(fam, inputs, view, harm, "tuned"), key, b["test_idx"], b["pred"], b["probs"],
                  [sc[best]], b["train_bal"])
    # nested selection over the whole grid, per input set
    for inputs in K.INPUTS:
        for key in main:
            sc = {}
            for c, v in T.c.items():
                fam, inp, view, harm, hp = K.parse_cid(c)
                if inp != inputs or hp == "tuned" or not view.startswith("k") or key not in v or not v[key]["inner"]:
                    continue
                if fam == "tabpfn" and not ((harm == "none" and int(view[1:]) in K.TABPFN_NESTED_K)
                                            or (harm == "combat" and view == "k50")):
                    continue
                sc[c] = float(np.mean(v[key]["inner"]))
            if not sc:
                continue
            best = K.select_best(sc)
            b = T.c[best][key]
            T.add(K.cid("nested", inputs, "-", "-", "-"), key, b["test_idx"], b["pred"], b["probs"], [sc[best]])
            chosen.append({"inputs": inputs, "seed": key[0], "fold": key[1], "chosen": best,
                           "inner_score": sc[best], "n_candidates": len(sc)})
    # shuffled-label controls
    S_ = Table()
    for key, r in shuf.items():
        for c, v in r["configs"].items():
            if c.startswith("tabpfn|") and c.split("|")[2] == "k50":
                S_.add(c, key, r["test_idx"], v["pred"], v.get("probs"))
        if "nested" in r:                       # first seed: full nested procedure, imgcom
            sc = {}
            for c, v in r["configs"].items():
                fam, inp, view, harm, hp = K.parse_cid(c)
                if inp == "imgcom" and fam in K.LIGHT and v.get("inner"):
                    sc[c] = float(np.mean(v["inner"]))
            for c, v in r["nested"].items():
                sc[c] = float(np.mean(v["inner"]))
            best = K.select_best(sc)
            S_.add(K.cid("nested", "imgcom", "-", "-", "-"), key, r["test_idx"], r["configs"][best]["pred"])
    return T, S_, pd.DataFrame(chosen)


# --------------------------------------------------------------- metrics
def per_seed_preds(T, cid, seeds):
    return np.vstack([T.oof(cid, s)[0] for s in seeds])


def boot(y, A, Bm=None, seed=0):
    """Vectorised stratified bootstrap over subjects. A, Bm: [n_seeds, 116] predictions.
    Returns (bal CI, acc CI) for A, or the CI of bal(A) - bal(Bm)."""
    rng = np.random.default_rng(seed)
    CA = (A == y[None, :]).astype(float)
    CB = None if Bm is None else (Bm == y[None, :]).astype(float)
    rec_a, rec_b, acc_a = [], [], []
    for g in range(K.NS):
        ids = np.flatnonzero(y == g)
        smp = rng.choice(ids, size=(B, len(ids)), replace=True)          # [B, n_g]
        rec_a.append(CA[:, smp].mean(2))                                  # [seeds, B]
        acc_a.append(CA[:, smp].sum(2))
        if CB is not None:
            rec_b.append(CB[:, smp].mean(2))
    bal_a = np.mean(rec_a, 0).mean(0)                                     # [B]
    if CB is not None:
        d = bal_a - np.mean(rec_b, 0).mean(0)
        return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
    acc = (np.sum(acc_a, 0) / len(y)).mean(0)
    return ((float(np.percentile(bal_a, 2.5)), float(np.percentile(bal_a, 97.5))),
            (float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))))


def metrics(T, cid, y):
    seeds = T.seeds(cid)
    if not seeds:
        return None
    rows = []
    for s in seeds:
        pred = T.oof(cid, s)[0]
        rows.append(K.H.evaluate(y, pred))
    folds = [K.bal_acc(y[v["test_idx"]], v["pred"]) for k, v in T.c[cid].items() if k[0] in seeds]
    A = per_seed_preds(T, cid, seeds)
    (blo, bhi), (alo, ahi) = boot(y, A)
    pooled_t, pooled_p = np.tile(y, len(seeds)), A.ravel()
    cm = np.zeros((K.NS, K.NS), int)
    for a, b in zip(pooled_t, pooled_p):
        cm[a, b] += 1
    tb = [v["train_bal"] for k, v in T.c[cid].items() if k[0] in seeds and v["train_bal"] is not None]

    def ms(key):
        v = [r[key] for r in rows]
        return float(np.mean(v)), float(np.std(v))
    out = {"n_seeds": len(seeds), "accuracy": ms("accuracy")[0], "accuracy_sd_seed": ms("accuracy")[1],
           "accuracy_ci_lo": alo, "accuracy_ci_hi": ahi,
           "balanced_accuracy": ms("balanced_accuracy")[0], "bal_sd_seed": ms("balanced_accuracy")[1],
           "bal_sd_fold": float(np.std(folds)), "bal_ci_lo": blo, "bal_ci_hi": bhi,
           "macro_f1": ms("macro_f1")[0], "mae_stages": ms("mae_stages")[0], "qwk": ms("qwk")[0],
           "train_balanced_accuracy": float(np.mean(tb)) if tb else np.nan}
    for g in range(K.NS):
        out[f"recall_{K.STAGES[g]}"] = float(cm[g, g] / cm[g].sum())
    return out, cm, [r["balanced_accuracy"] for r in rows]


def label_of(cid):
    fam, inputs, view, harm, hp = K.parse_cid(cid)
    feat = {"-": "-"}.get(view, view.replace("bio_", "biomarker: ").replace("k", "k = ") if view.startswith(("k", "bio")) else view)
    sel = "nested selection" if fam == "nested" else ("hyperparameters tuned in inner CV" if hp == "tuned" else "fixed")
    return FAM_LABEL.get(fam, fam), IN_LABEL.get(inputs, inputs), feat, harm, sel


def main():
    for p in (os.path.join(RES, "model_comparison.csv"), FIG):
        if os.path.exists(p):
            raise SystemExit(f"COLLISION: {p} exists; not overwritten")
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    os.makedirs(RES, exist_ok=True)
    os.makedirs(FIG)
    d = K.load()
    y, subjects, site = d["y"], d["subjects"], d["site"]
    main_, nested, bio, shuf = (load_job(j) for j in ("main", "nested", "biomarker", "shuffled"))
    print(f"split files: main {len(main_)}, nested {len(nested)}, biomarker {len(bio)}, shuffled {len(shuf)}")
    T, S_, chosen = build(main_, nested, bio, shuf)

    # ---------------- which configs to report
    report = ["majority|-|-|-|-"]
    for inputs in K.INPUTS:
        report += [K.cid("tabpfn", inputs, f"k{k}", "none", "-") for k in K.TABPFN_K]
        report += [K.cid("tabpfn", inputs, "k50", "combat", "-")]
        report += [K.cid(f, inputs, f"k{k}", h, "tuned") for f in K.LIGHT for k in K.K_GRID for h in K.HARMS]
        report += [K.cid("nested", inputs, "-", "-", "-")]
    for b in K.BIOMARKERS:
        report += [K.cid("tabpfn", "img", f"bio_{b}", "none", "-")]
        report += [K.cid(f, "img", f"bio_{b}", "none", "tuned") for f in K.LIGHT]
    report = [c for c in report if T.seeds(c)]
    ref = {inp: K.cid("tabpfn", inp, "k50", "none", "-") for inp in K.INPUTS}

    rows, cms, oof_rows, seedbal = [], [], [], {}
    for c in report:
        m, cm, sb = metrics(T, c, y)
        seedbal[c] = sb
        fam, inputs_l, feat, harm, sel = label_of(c)
        inp = K.parse_cid(c)[1]
        row = {"config_id": c, "model": fam, "inputs": inputs_l, "features": feat, "harmonisation": harm,
               "selection": sel, **m}
        r_ = ref.get(inp)
        if r_ and r_ != c and T.seeds(r_) and T.seeds(c) == T.seeds(r_):
            dd = np.array(sb) - np.array(seedbal.get(r_) or metrics(T, r_, y)[2])
            lo, hi = boot(y, per_seed_preds(T, c, T.seeds(c)), per_seed_preds(T, r_, T.seeds(r_)))
            better, worse = int((dd > 0).sum()), int((dd < 0).sum())
            row.update({"reference": r_, "diff_vs_reference": float(dd.mean()), "diff_ci_lo": lo, "diff_ci_hi": hi,
                        "seeds_better": better,
                        "verdict": ("better" if lo > 0 and better >= 4 else
                                    "worse" if hi < 0 and worse >= 4 else "no significant difference")})
        rows.append(row)
        for g in range(K.NS):
            for h in range(K.NS):
                cms.append({"config_id": c, "true_stage": K.STAGES[g], "pred_stage": K.STAGES[h], "count": int(cm[g, h])})
        for s in T.seeds(c):
            pred, probs, fold = T.oof(c, s)
            for i in range(len(y)):
                oof_rows.append({"config_id": c, "seed": s, "fold": int(fold[i]), "subject_id": subjects[i],
                                 "site": site[i], "true_stage": K.STAGES[y[i]], "pred_stage": K.STAGES[pred[i]],
                                 "correct": int(pred[i] == y[i]),
                                 **{f"p_{K.STAGES[g]}": (None if np.isnan(probs[i, g]) else round(float(probs[i, g]), 6))
                                    for g in range(K.NS)}})
    # shuffled-label controls
    for c in sorted(S_.c):
        if not S_.seeds(c):
            continue
        m, cm, _ = metrics(S_, c, y)
        fam, inputs_l, feat, harm, sel = label_of(c)
        rows.append({"config_id": c + "|SHUFFLED", "model": fam, "inputs": inputs_l, "features": feat,
                     "harmonisation": harm, "selection": sel + ", training labels permuted", **m})
    comp = pd.DataFrame(rows)
    comp.to_csv(os.path.join(RES, "model_comparison.csv"), index=False)
    pd.DataFrame(cms).to_csv(os.path.join(RES, "confusion_matrices.csv"), index=False)
    pd.DataFrame(oof_rows).to_csv(os.path.join(RES, "oof_predictions.csv"), index=False)
    chosen.to_csv(os.path.join(RES, "nested_selection.csv"), index=False)

    # ---------------- feature count
    fc_rows = []
    for inputs in K.INPUTS:
        for fam in ["tabpfn"] + K.LIGHT:
            for k in K.K_GRID:
                c = K.cid(fam, inputs, f"k{k}", "none", "-" if fam == "tabpfn" else "tuned")
                r = comp[comp.config_id == c]
                if len(r):
                    r = r.iloc[0]
                    fc_rows.append({"inputs": inputs, "model": FAM_LABEL[fam], "k": k,
                                    "balanced_accuracy": r.balanced_accuracy, "sd_seed": r.bal_sd_seed,
                                    "ci_lo": r.bal_ci_lo, "ci_hi": r.bal_ci_hi, "accuracy": r.accuracy})
    fcd = pd.DataFrame(fc_rows)
    fcd.to_csv(os.path.join(RES, "feature_count.csv"), index=False)
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    for ax, inputs in zip(axs, K.INPUTS):
        for j, fam in enumerate(["tabpfn"] + K.LIGHT):
            q = fcd[(fcd.inputs == inputs) & (fcd.model == FAM_LABEL[fam])]
            if len(q):
                ax.errorbar(q.k, q.balanced_accuracy, yerr=q.sd_seed, marker="os^D"[j], ms=6, capsize=3,
                            color=[P.SERIES[0], P.SERIES[1], P.SERIES[2], P.MUTED][j], label=FAM_LABEL[fam])
        ax.axhline(1 / 6, color=P.MUTED, ls="--", lw=1)
        ax.set_xscale("log")
        ax.set_xticks(K.K_GRID, [str(k) for k in K.K_GRID])
        ax.set_xlabel("number of SVM-RFE features (log scale; 984 = all)")
        ax.set_title(IN_LABEL[inputs], loc="left")
        ax.yaxis.grid(True)
    axs[0].set_ylabel("balanced accuracy (mean +/- sd of 5 seeds)")
    axs[0].text(25, 1 / 6, " chance", color=P.MUTED, fontsize=8, va="bottom")
    axs[0].legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=4, fontsize=8)
    fig.suptitle("Feature count vs performance, 5-fold x 5 CV, no harmonisation (TabPFN v2 stops at 500 features)",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.savefig(os.path.join(FIG, "feature_count.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---------------- harmonisation
    h_rows = []
    for inputs in K.INPUTS:
        pairs = [(K.cid("tabpfn", inputs, "k50", "none", "-"), K.cid("tabpfn", inputs, "k50", "combat", "-"), "TabPFN v2", 50)]
        pairs += [(K.cid(f, inputs, f"k{k}", "none", "tuned"), K.cid(f, inputs, f"k{k}", "combat", "tuned"), FAM_LABEL[f], k)
                  for f in K.LIGHT for k in K.K_GRID]
        for a, b, nm, k in pairs:
            if not (T.seeds(a) and T.seeds(b)):
                continue
            ra, rb = comp[comp.config_id == a].iloc[0], comp[comp.config_id == b].iloc[0]
            dd = np.array(seedbal[b]) - np.array(seedbal[a])
            lo, hi = boot(y, per_seed_preds(T, b, T.seeds(b)), per_seed_preds(T, a, T.seeds(a)))
            h_rows.append({"inputs": inputs, "model": nm, "k": k, "bal_none": ra.balanced_accuracy,
                           "bal_combat": rb.balanced_accuracy, "diff_combat_minus_none": float(dd.mean()),
                           "diff_ci_lo": lo, "diff_ci_hi": hi, "seeds_combat_better": int((dd > 0).sum()),
                           "verdict": ("ComBat better" if lo > 0 and (dd > 0).sum() >= 4 else
                                       "ComBat worse" if hi < 0 and (dd < 0).sum() >= 4 else "no significant difference")})
    hd = pd.DataFrame(h_rows)
    hd.to_csv(os.path.join(RES, "harmonization_comparison.csv"), index=False)
    site_csv = os.path.join(RES, "site_prediction.csv")
    sp = pd.read_csv(site_csv) if os.path.exists(site_csv) else None
    unadj = [r["combat_unadjusted_test"].get("combat") for r in main_.values() if r["combat_unadjusted_test"].get("combat") is not None]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 2]})
    if sp is not None:
        sm = sp.groupby(["harmonisation", "seed"]).site_balanced_accuracy.mean().reset_index()
        for j, h in enumerate(K.HARMS):
            v = sm[sm.harmonisation == h].site_balanced_accuracy
            axs[0].bar(j, v.mean(), color=P.SERIES[j], width=0.55)
            axs[0].scatter(np.full(len(v), j), v, color=P.INK, s=12, zorder=3)
        axs[0].axhline(sp.chance.iloc[0], color=P.MUTED, ls="--", lw=1)
        axs[0].set_xticks([0, 1], ["no harmonisation", "ComBat"])
        axs[0].set_ylabel("site balanced accuracy")
        axs[0].set_title("Predicting scanner SITE from features", loc="left", fontsize=9)
        axs[0].yaxis.grid(True)
    q = hd[hd.k == 50]
    labels = [f"{r.model}\n{IN_LABEL[r.inputs]}" for r in q.itertuples()]
    for i, r in enumerate(q.itertuples()):
        axs[1].plot([0, 1], [r.bal_none, r.bal_combat], color=P.SERIES[0] if r.inputs == "img" else P.SERIES[1],
                    marker="o", ms=5, alpha=0.8)
        axs[1].text(1.03, r.bal_combat, labels[i].replace("\n", ", "), fontsize=7, va="center", color=P.INK2)
    axs[1].set_xticks([0, 1], ["no harmonisation", "ComBat"])
    axs[1].set_xlim(-0.2, 2.2)
    axs[1].axhline(1 / 6, color=P.MUTED, ls="--", lw=1)
    axs[1].set_ylabel("stage balanced accuracy (k = 50)")
    axs[1].set_title("Stage prediction with and without ComBat (paired, same folds)", loc="left", fontsize=9)
    axs[1].yaxis.grid(True)
    fig.savefig(os.path.join(FIG, "harmonization.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---------------- confusion matrices of the key configs
    keys = [ref["img"], ref["imgcom"], "nested|img|-|-|-", "nested|imgcom|-|-|-"]
    keys = [k for k in keys if T.seeds(k)]
    fig, axs = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 4), squeeze=False)
    cmdf = pd.DataFrame(cms)
    for ax, c in zip(axs[0], keys):
        M_ = cmdf[cmdf.config_id == c].pivot(index="true_stage", columns="pred_stage", values="count").loc[K.STAGES, K.STAGES].to_numpy()
        Mn = M_ / M_.sum(1, keepdims=True)
        ax.imshow(Mn, cmap=P.SEQ, vmin=0, vmax=max(0.5, Mn.max()))
        for a in range(K.NS):
            for b in range(K.NS):
                ax.text(b, a, f"{Mn[a, b]:.2f}", ha="center", va="center", fontsize=7,
                        color="#ffffff" if Mn[a, b] > 0.3 else P.INK2)
        ax.set_xticks(range(K.NS), K.STAGES, fontsize=7)
        ax.set_yticks(range(K.NS), K.STAGES, fontsize=7)
        fam, inp, *_ = label_of(c)
        ax.set_title(f"{fam}\n{inp}", loc="left", fontsize=8)
        ax.set_xlabel("predicted")
        ax.tick_params(length=0)
        for s_ in ax.spines.values():
            s_.set_visible(False)
    axs[0][0].set_ylabel("true")
    fig.suptitle("Confusion matrices, row share, pooled over 5 seeds (5-fold CV, 116 subjects)", x=0.01,
                 ha="left", fontweight="bold", fontsize=10)
    fig.savefig(os.path.join(FIG, "confusion_key_models.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ---------------- reproduction of the earlier TabPFN run
    rep = {}
    old_dir = os.path.join(K.ROOT, "derivatives", "model_training", "tabpfn")
    for inputs, old in (("img", "cv116_tabpfn_svmrfe50.json"), ("imgcom", "cv116_tabpfn_svmrfe50_comorb8.json")):
        p_ = os.path.join(old_dir, old)
        c = ref[inputs]
        if not (os.path.exists(p_) and c in T.c):
            continue
        olds = {(r["seed"], r["fold"]): r for r in json.load(open(p_))["records"]}
        diffs, same = [], []
        for key, v in T.c[c].items():
            if key in olds:
                diffs.append(float(np.max(np.abs(np.asarray(v["probs"]) - np.asarray(olds[key]["probs"])))))
                same.append(bool(np.array_equal(v["pred"], olds[key]["pred"])))
        rep[inputs] = {"earlier_file": os.path.relpath(p_, K.ROOT), "splits_compared": len(diffs),
                       "identical_predictions": int(sum(same)), "max_abs_prob_diff": max(diffs) if diffs else None}
    json.dump(rep, open(os.path.join(RES, "reproduction_check.json"), "w"), indent=1)

    # ---------------- summary tables
    def fmt(r):
        return (f"{100 * r.accuracy:.1f}% | {100 * r.balanced_accuracy:.1f}% ± {100 * r.bal_sd_seed:.1f} "
                f"[{100 * r.bal_ci_lo:.1f}, {100 * r.bal_ci_hi:.1f}] | {100 * r.macro_f1:.1f}%")
    L = ["# Summary tables (generated by report.py)", "",
         f"Generated {time.strftime('%Y-%m-%d %H:%M')} from {len(main_)} main / {len(nested)} nested / "
         f"{len(bio)} biomarker / {len(shuf)} shuffled split files.", "",
         "Accuracy | balanced accuracy ± sd over seeds [95% bootstrap CI] | macro-F1; vs = paired difference in "
         "balanced accuracy against TabPFN k = 50 on the same inputs [95% CI], seeds better of 5.", ""]
    for title, ids in (("Key models", [c for c in report if c.startswith(("majority", "nested")) or c in ref.values()]),
                       ("All fixed / tuned models", report)):
        L += [f"## {title}", "", "| config | accuracy | balanced accuracy | macro-F1 | vs TabPFN k=50 | verdict |",
              "|---|---|---|---|---|---|"]
        for c in ids:
            r = comp[comp.config_id == c].iloc[0]
            vs = (f"{100 * r.diff_vs_reference:+.1f} [{100 * r.diff_ci_lo:+.1f}, {100 * r.diff_ci_hi:+.1f}], "
                  f"{int(r.seeds_better)}/5" if pd.notna(r.get("diff_vs_reference", np.nan)) else "-")
            L.append(f"| `{c}` | {fmt(r).replace(' | ', ' | ')} | {vs} | {r.get('verdict', '-') if pd.notna(r.get('verdict', np.nan)) else '-'} |")
        L.append("")
    L += ["## Shuffled-label controls", "", "| config | balanced accuracy | seeds |", "|---|---|---|"]
    for r in comp[comp.config_id.str.endswith("SHUFFLED")].itertuples():
        L.append(f"| `{r.config_id}` | {100 * r.balanced_accuracy:.1f}% [{100 * r.bal_ci_lo:.1f}, {100 * r.bal_ci_hi:.1f}] | {r.n_seeds} |")
    L += ["", "## Nested selection: configurations chosen by the inner loop", ""]
    if len(chosen):
        for inputs in K.INPUTS:
            q = chosen[chosen.inputs == inputs].chosen.value_counts()
            L.append(f"- {IN_LABEL[inputs]}: " + ", ".join(f"`{k}` x{v}" for k, v in q.items()))
    L += ["", "## Harmonisation (ComBat minus none)", "", "| inputs | model | k | none | ComBat | diff [95% CI] | seeds | verdict |",
          "|---|---|---|---|---|---|---|---|"]
    for r in hd.itertuples():
        L.append(f"| {IN_LABEL[r.inputs]} | {r.model} | {r.k} | {100 * r.bal_none:.1f}% | {100 * r.bal_combat:.1f}% | "
                 f"{100 * r.diff_combat_minus_none:+.1f} [{100 * r.diff_ci_lo:+.1f}, {100 * r.diff_ci_hi:+.1f}] | "
                 f"{r.seeds_combat_better}/5 | {r.verdict} |")
    if sp is not None:
        sm = sp.groupby(["harmonisation", "seed"]).site_balanced_accuracy.mean().groupby("harmonisation").agg(["mean", "std"])
        L += ["", f"Site predictability (balanced accuracy, chance {sp.chance.iloc[0]:.3f}): "
              + ", ".join(f"{h} {100 * sm.loc[h, 'mean']:.1f}% ± {100 * sm.loc[h, 'std']:.1f}" for h in sm.index)]
    if unadj:
        L += [f"ComBat: test subjects from a site not estimable in training (left unadjusted): "
              f"{sum(unadj)} subject-folds over {len(unadj)} outer splits (max {max(unadj)} in one split)."]
    L += ["", "## Reproduction of the earlier TabPFN k = 50 run", "", "```", json.dumps(rep, indent=1), "```"]
    for inputs in K.INPUTS:
        p_ = os.path.join(RES, f"test_locked_{inputs}.json")
        if os.path.exists(p_):
            t = json.load(open(p_))
            L += [f"", f"## Locked test ({inputs}, {t['role']})", "",
                  f"selected `{t['selected']}`; accuracy {100 * t['metrics']['accuracy']:.1f}% "
                  f"({t['accuracy_correct']}/35, Wilson 95% CI {100 * t['accuracy_wilson95'][0]:.1f}-"
                  f"{100 * t['accuracy_wilson95'][1]:.1f}%), balanced accuracy {100 * t['metrics']['balanced_accuracy']:.1f}% "
                  f"[{100 * t['balanced_accuracy_boot95'][0]:.1f}, {100 * t['balanced_accuracy_boot95'][1]:.1f}]"]
    open(os.path.join(RES, "summary_tables.md"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    outs = [os.path.relpath(p_, K.ROOT) for p_ in glob.glob(os.path.join(RES, "*")) + glob.glob(os.path.join(FIG, "*"))]
    K.manifest_entry("python scripts/experiments/leakage_safe_v1/report.py", started, outs)
    print("\n".join(L[:60]))


if __name__ == "__main__":
    main()
