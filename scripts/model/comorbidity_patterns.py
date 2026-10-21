"""
Analysis of the comorbidity-aware transformer (train_transformer_comorbidity.py).
Reads only saved outputs; nothing is retrained.

  0. QC           metrics recompute from saved predictions; identical folds and
                  identical SVM-RFE features as the imaging-only run
  1. outcomes     70/30 split (35 test) and 5x5 CV on 116, every candidate
  2. paired       does comorbidity help? (fusion vs imaging-only, same folds)
  3. ordering     expected stage vs true stage, fusion vs imaging-only
  4. modalities   brain-vs-comorbidity gate by stage; P(true stage) with one
                  modality removed
  5. variables    permutation importance of each comorbidity variable
  6. confounds    age, sex, APOE4, GDS by stage in the cohort itself
  7. ceiling      MMSE + CDR alone (circular; the label-defining ceiling)

Run:  python scripts/model/comorbidity_patterns.py --features comorbidity23
      python scripts/model/comorbidity_patterns.py --features block5
      python scripts/model/comorbidity_patterns.py --features comorbidity8   (model A)
      python scripts/model/comorbidity_patterns.py --features clinical10     (model B)
"""
import argparse
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage_patterns as P  # noqa: E402  (plot style, palette, statistics helpers)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

from train_transformer import evaluate  # noqa: E402
from train_transformer_comorbidity import FEATURE_SETS, OUT_DIRS, load_clinical  # noqa: E402

H = P.H
ROOT = P.ROOT
TRAIN_DIR = os.path.join(ROOT, "derivatives", "model_training")
SHARED = os.path.join(TRAIN_DIR, OUT_DIRS["comorbidity23"])    # imaging-only 70/30 + ceiling live here
IMG = os.path.join(TRAIN_DIR, "transformer_svmrfe")            # imaging-only CV records
CF = os.path.join(ROOT, "derivatives", "comorbidity_features")
STAGES, NS, CHANCE = P.STAGES, P.NS, P.CHANCE
PRIMARY = "transformer_svmrfe50_clin"
SHUF = PRIMARY + "_SHUFFLED"
ORDER = [PRIMARY, "transformer_svmrfe50", "logreg_svmrfe50_clin", "logreg_svmrfe50",
         "logreg_clin", SHUF, "ceiling_mmse_cdrsb", "ceiling_cdr_mmse"]
LABEL = {PRIMARY: "Transformer + comorbidity (primary)",
         "transformer_svmrfe50": "Transformer, imaging only",
         "logreg_svmrfe50_clin": "Logistic reg., imaging + comorbidity",
         "logreg_svmrfe50": "Logistic reg., imaging only",
         "logreg_clin": "Logistic reg., comorbidity only",
         SHUF: "Transformer + comorbidity, shuffled labels",
         "ceiling_mmse_cdrsb": "MMSE + CDR-SB only (circular ceiling)",
         "ceiling_cdr_mmse": "MMSE + CDR only (circular ceiling)"}
NICE = {"age": "age", "sex_male": "sex (male)", "APOE4_count": "APOE4 count", "GDS_total": "GDS (depression)",
        "MMSE_total": "MMSE", "CDR_SOB": "CDR sum of boxes", "CDR_global": "CDR global"}
CLIN_COLS = []
KEYS = P.KEYS

_log = []


def log(m=""):
    print(m, flush=True)
    _log.append(str(m))


def savefig(fig, path):
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log(f"    figure {os.path.relpath(path, ROOT)}")


def nice(c):
    return NICE.get(c, c.replace("MH_", "").replace("_", " "))


def main():
    global CLIN_COLS
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", choices=list(FEATURE_SETS), default="comorbidity23")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    SRC = os.path.join(TRAIN_DIR, OUT_DIRS[args.features])
    args.out = args.out or SRC
    CLIN_COLS = FEATURE_SETS[args.features]
    block = {"comorbidity23": "comorbidity", "block5": "clinical block", "comorbidity8": "comorbidity",
             "clinical10": "comorbidity + MMSE/CDR-SB"}[args.features]
    for k_ in (PRIMARY, "logreg_svmrfe50_clin", "logreg_clin", SHUF):
        LABEL[k_] = LABEL[k_].replace("comorbidity", block)
    figdir = os.path.join(args.out, "figures")
    for p in (os.path.join(args.out, "comorbidity_patterns.json"), figdir):
        if os.path.exists(p):
            raise SystemExit(f"COLLISION: {p} exists; not overwritten")
    os.makedirs(figdir)

    sp = {c: P.load("split70", c, SRC) or P.load("split70", c, SHARED) for c in ORDER if c != SHUF}
    cv = {c: P.load("cv116", c, SRC) for c in ORDER}
    cv["ceiling_cdr_mmse"] = cv["ceiling_cdr_mmse"] or P.load("cv116", "ceiling_cdr_mmse", SHARED)
    for c in ("transformer_svmrfe50", "logreg_svmrfe50"):
        cv[c] = P.load("cv116", c, IMG)                   # reused imaging-only records
    saved_cols = json.load(open(os.path.join(SRC, f"cv116_{PRIMARY}.json")))["extra"]["clinical_columns"]
    if saved_cols != CLIN_COLS:
        raise SystemExit(f"saved records use {saved_cols}, expected {CLIN_COLS}")
    split_fp = {r["fold_fingerprint"] for v in sp.values() if v for r in v}
    if cv[PRIMARY] is None or sp.get(PRIMARY) is None:
        raise SystemExit("primary records missing")
    missing = [f"cv116_{c}" for c, v in cv.items() if v is None] + \
              [f"split70_{c}" for c, v in sp.items() if v is None]
    y = np.load(os.path.join(H.SRC, "y.npy")).astype(int)
    out = {"missing_inputs": missing}

    log("=" * 100)
    log(f"COMORBIDITY-AWARE TRANSFORMER -- SVM-RFE 50 rs-fMRI tokens + {len(CLIN_COLS)} {block} features "
        f"('{args.features}'), 116 subjects")
    log("=" * 100)
    log(f"{block}: {', '.join(nice(c) for c in CLIN_COLS)}")
    if any(c in CLIN_COLS for c in ("MMSE_total", "CDR_SOB", "CDR_global")):
        log("NOTE: MMSE / CDR are inputs here by the user's decision. ADNI assigns the stage labels with them,")
        log("      so compare against 'MMSE + CDR only' and 'imaging only' to see what the brain data adds.")
    else:
        log("MMSE and CDR are NOT inputs (they define the ADNI stage labels); they appear only as the ceiling row")
    if missing:
        log(f"  inputs not found (left out): {missing}")

    # ---------------------------------------------------------------- 0. QC
    log("\n[0] QC")
    checks = bad = 0
    for group in (sp, cv):
        for recs in group.values():
            for r in recs or []:
                m = evaluate(np.asarray(r["probs"]), np.asarray(r["true"]), pred=np.asarray(r["pred"]))
                for k in KEYS:
                    checks += 1
                    bad += abs(m[k] - r[k]) > 1e-9
                checks += 1
                bad += r["true"] != [int(y[i]) for i in r["test_index"]]
    log(f"  {checks} checks, {bad} mismatches" + ("" if not bad else "   <-- FAIL"))
    if bad:
        raise SystemExit("saved predictions do not reproduce the recorded metrics")
    fps = {c: sorted({(r["seed"], r["fold_fingerprint"]) for r in v}) for c, v in cv.items() if v}
    same = len({json.dumps(v) for v in fps.values()}) == 1
    log(f"  every CV candidate (incl. reused imaging-only records) used identical folds: {same}")
    log(f"  every 70/30 candidate used the same split: {len(split_fp) == 1} {sorted(split_fp)}")
    if not same or len(split_fp) != 1:
        raise SystemExit("candidates used different folds or splits")
    if cv["transformer_svmrfe50"]:
        ss = all(a["support"] == b["support"] for a, b in zip(cv[PRIMARY], cv["transformer_svmrfe50"]))
        log(f"  fusion and imaging-only transformer used the same 50 rs-fMRI features in every fold: {ss}")
    out["qc"] = {"checks": checks, "mismatches": bad, "identical_folds": same}

    # ---------------------------------------------------------- 1. outcomes
    log("\n[1] STAGE PREDICTION (balanced accuracy; chance 0.167)")
    log(f"  {'model':44s} {'70/30 test (35)':>15s} {'correct':>8s} {'p vs chance':>11s} | "
        f"{'5x5 CV on 116':>14s} {'within1':>7s} {'train':>6s}")
    out["outcomes"] = {}
    for c in ORDER:
        row = {}
        left = f"{'':15s} {'':8s} {'':11s}"
        if sp.get(c):
            r = sp[c][0]
            k = int(round(r["accuracy"] * r["n_test"]))
            ci = stats.binomtest(k, r["n_test"], CHANCE).proportion_ci(0.95)
            pv = stats.binomtest(k, r["n_test"], CHANCE, alternative="greater").pvalue
            row["split70"] = {**{q: r[q] for q in KEYS}, "correct": k, "n_test": r["n_test"],
                              "accuracy_95ci": [ci.low, ci.high], "p_vs_chance": pv,
                              "within_one": P.within_one(r["true"], r["pred"]),
                              "per_class_recall": r["per_class_recall"],
                              "confusion_matrix": r["confusion_matrix"]}
            left = f"{r['balanced_accuracy']:15.3f} {k:4d}/{r['n_test']:<3d} {pv:11.3f}"
        right = ""
        if cv.get(c):
            s = H.summarise(cv[c])
            w1 = np.mean([P.within_one(r["true"], r["pred"]) for r in cv[c]])
            tr = np.mean([r["train_balanced_accuracy"] for r in cv[c]])
            row["cv116"] = {**{q: s[q] for q in KEYS}, "within_one": float(w1), "train_balanced_accuracy": float(tr)}
            right = f"{s['balanced_accuracy'][0]:.3f}+/-{s['balanced_accuracy'][1]:.3f} {w1:7.3f} {tr:6.3f}"
        out["outcomes"][c] = row
        log(f"  {LABEL[c]:44s} {left} | {right}")
    log("  70/30: one test subject moves balanced accuracy by about 0.028-0.033")

    # ------------------------------------------------------------ 2. paired
    log("\n[2] PAIRED CV COMPARISONS (identical folds; rule: >= 4/5 seeds and mean > 0)")
    out["paired"] = {}
    pairs = [(PRIMARY, "transformer_svmrfe50", "does comorbidity help the transformer?"),
             ("logreg_svmrfe50_clin", "logreg_svmrfe50", "does comorbidity help logistic regression?"),
             (PRIMARY, "logreg_svmrfe50_clin", "transformer fusion vs logistic fusion"),
             (PRIMARY, "logreg_clin", "fusion vs comorbidity alone"),
             (PRIMARY, SHUF, "fusion vs shuffled labels")]
    for a, b, q in pairs:
        if not (cv.get(a) and cv.get(b)):
            continue
        d = P.seed_means(cv[a], "balanced_accuracy") - P.seed_means(cv[b], "balanced_accuracy")
        wins = int((d > 0).sum())
        verdict = (f"{a} BETTER" if wins >= 4 and d.mean() > 0 else
                   f"{b} BETTER" if (d < 0).sum() >= 4 and d.mean() < 0 else "TIED")
        out["paired"][f"{a} - {b}"] = {"question": q, "per_seed_diff": d.tolist(),
                                       "mean_diff": float(d.mean()), "seeds_favouring_first": wins,
                                       "verdict": verdict}
        log(f"  {q:44s} {d.mean():+.3f}  seeds {wins}/5  {np.round(d, 3).tolist()}  -> {verdict}")
    if cv.get(SHUF):
        sb = H.summarise(cv[SHUF])["balanced_accuracy"][0]
        log(f"  shuffled control {sb:.3f}: " + ("OK (near chance)" if abs(sb - CHANCE) < 0.06 else "CHECK"))

    # ---------------------------------------------------------- 3. ordering
    log("\n[3] STAGE ORDERING (Spearman rho of expected stage vs true stage, permutation p)")
    out["ordering"] = {}
    E = {}
    for c in (PRIMARY, "transformer_svmrfe50", "logreg_svmrfe50_clin", "logreg_clin", SHUF):
        if not cv.get(c):
            continue
        for r in cv[c]:
            r["_E"] = P.expected_stage(r).tolist()
        E[c] = P.per_subject(cv[c], "_E")
        rho, p = P.spearman_perm(E[c], y)
        means = [float(E[c][y == g].mean()) for g in range(NS)]
        out["ordering"][c] = {"rho": rho, "p": p, "stage_means": means}
        log(f"  {LABEL[c]:44s} rho {rho:+.3f}  p {p:.3f}   "
            + " ".join(f"{STAGES[g]} {means[g]:.2f}" for g in range(NS)))
    r = sp[PRIMARY][0]
    rho, p = P.spearman_perm(P.expected_stage(r), np.asarray(r["true"]))
    out["ordering"]["split70_primary"] = {"rho": rho, "p": p}
    log(f"  70/30 primary (35 test subjects): rho {rho:+.3f}  p {p:.3f}")

    # -------------------------------------------------------- 4. modalities
    log(f"\n[4] BRAIN vs {block.upper()} inside the fusion model (out-of-fold, mean over 5 seeds)")
    recs = cv[PRIMARY]
    gate = P.per_subject(recs, "gate")
    pt, pnc, pnb = (P.per_subject(recs, k) for k in ("p_true", "p_true_no_clin", "p_true_no_brain"))
    gm = [float(gate[y == g].mean()) for g in range(NS)]
    ge = [float(gate[y == g].std(ddof=1) / np.sqrt((y == g).sum())) for g in range(NS)]
    gr = P.spear(gate, y)
    out["modalities"] = {"gate_stage_means": gm, "gate_trend": {"rho": gr[0], "p": gr[1]},
                         "p_true_full": float(pt.mean()), "p_true_no_comorbidity": float(pnc.mean()),
                         "p_true_no_brain": float(pnb.mean())}
    log("  gate = share of the fused representation from the brain (1) vs comorbidity (0)")
    log("    " + "  ".join(f"{STAGES[g]} {gm[g]:.3f}" for g in range(NS))
        + f"   trend rho {gr[0]:+.3f} p {gr[1]:.3f}")
    log(f"  mean P(true stage): full {pt.mean():.3f}   comorbidity removed {pnc.mean():.3f}   "
        f"brain removed {pnb.mean():.3f}   (uniform guess {1 / NS:.3f})")
    for g in range(NS):
        m_ = y == g
        log(f"    {STAGES[g]:5s} full {pt[m_].mean():.3f}  no comorbidity {pnc[m_].mean():.3f}  "
            f"no brain {pnb[m_].mean():.3f}")

    # --------------------------------------------------------- 5. variables
    log(f"\n[5] {block.upper()} VARIABLE IMPORTANCE (permutation within test folds; drop in mean P(true stage))")
    imp = np.array([[r["perm_importance"][c] for c in CLIN_COLS] for r in recs])     # [25, 23]
    im, ie = imp.mean(0), imp.std(0, ddof=1) / np.sqrt(len(imp))
    order = np.argsort(-im)
    coef = None
    if cv.get("logreg_clin"):
        coef = np.mean([[r["coef_abs_mean"][c] for c in CLIN_COLS] for r in cv["logreg_clin"]], axis=0)
    out["variables"] = {CLIN_COLS[j]: {"perm_importance": float(im[j]), "se": float(ie[j]),
                                       "logreg_abs_coef": None if coef is None else float(coef[j])}
                        for j in order}
    for j in order:
        log(f"    {nice(CLIN_COLS[j]):34s} {im[j] * 1000:+7.2f} x0.001  (SE {ie[j] * 1000:.2f})"
            + ("" if coef is None else f"   |logreg coef| {coef[j]:.3f}"))
    fig, ax = plt.subplots(figsize=(6.6, max(2.4, 0.28 * len(order) + 0.8)))
    oo = order[::-1]
    ax.barh(range(len(oo)), im[oo] * 1000, xerr=ie[oo] * 1000, color=P.SERIES[0], height=0.7,
            error_kw={"ecolor": P.INK2, "lw": 1})
    ax.axvline(0, color=P.AXIS, lw=1)
    ax.set_yticks(range(len(oo)), [nice(CLIN_COLS[j]) for j in oo], fontsize=8)
    ax.set_xlabel("drop in P(true stage) when shuffled (x 0.001, mean +/- SE over 25 folds)")
    ax.xaxis.grid(True)
    ax.set_title(f"Which {block} variables the fusion model uses", loc="left")
    savefig(fig, os.path.join(figdir, "fig2_comorbidity_importance.png"))

    # --------------------------------------------------------- 6. confounds
    log("\n[6] CONFOUNDS IN THE COHORT (how strongly each input differs by stage)")
    Craw, _ = load_clinical(y, np.load(os.path.join(H.SRC, "subjects.npy")).astype(str), CLIN_COLS)
    conf = {}
    log(f"  {'':22s} " + " ".join(f"{s_:>6s}" for s_ in STAGES) + "   test")
    for j, c in enumerate(CLIN_COLS):
        v = Craw[:, j]
        ok_ = ~np.isnan(v)
        if set(np.unique(v[ok_])) <= {0.0, 1.0}:
            tab = np.array([[((y == g) & ok_ & (v == b_)).sum() for b_ in (0, 1)] for g in range(NS)])
            p_, test = stats.chi2_contingency(tab)[1], "chi2"
        else:
            p_, test = stats.kruskal(*[v[(y == g) & ok_] for g in range(NS)]).pvalue, "Kruskal"
        means = [float(np.nanmean(v[y == g])) for g in range(NS)]
        conf[c] = {"stage_means": means, "test": test, "p": float(p_)}
        log(f"  {nice(c)[:22]:22s} " + " ".join(f"{m_:6.2f}" for m_ in means) + f"   {test} p {p_:.3g}")
    out["confounds"] = conf
    show = [c for c in CLIN_COLS if c in ("age", "APOE4_count", "GDS_total", "MMSE_total", "CDR_SOB")]
    fig, axs = plt.subplots(1, len(show), figsize=(3.4 * len(show), 3.1), squeeze=False)
    for ax, c in zip(axs[0], show):
        v = Craw[:, CLIN_COLS.index(c)]
        ax.boxplot([v[(y == g) & ~np.isnan(v)] for g in range(NS)], positions=range(NS), widths=0.5,
                   patch_artist=True, showfliers=False, medianprops={"color": P.INK},
                   boxprops={"facecolor": "#cde2fb", "edgecolor": P.SERIES[0]},
                   whiskerprops={"color": P.SERIES[0]}, capprops={"color": P.SERIES[0]})
        ax.set_title(f"{nice(c)}\n{conf[c]['test']} p {conf[c]['p']:.2g}", loc="left", fontsize=9)
        ax.set_xticks(range(NS), STAGES, fontsize=7)
        ax.yaxis.grid(True)
    fig.suptitle("Clinical inputs by stage in the 116-subject cohort", x=0.01, ha="left",
                 fontweight="bold", fontsize=10)
    fig.tight_layout()
    savefig(fig, os.path.join(figdir, "fig5_inputs_by_stage.png"))

    # ----------------------------------------------------------- figures
    # fig0: comparison
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    rows = [c for c in ORDER if cv.get(c) or sp.get(c)]
    for k, c in enumerate(rows):
        if cv.get(c):
            sm = P.seed_means(cv[c], "balanced_accuracy")
            ax.scatter(sm, [k] * len(sm), s=26, color=P.SERIES[0], alpha=0.45, zorder=3,
                       label="CV seed mean" if k == 0 else None)
            ax.scatter([sm.mean()], [k], s=90, color=P.SERIES[0], marker="D", zorder=4,
                       label="CV mean (5x5 on 116)" if k == 0 else None)
        if sp.get(c):
            ax.scatter([sp[c][0]["balanced_accuracy"]], [k], s=80, facecolor=P.SURFACE,
                       edgecolor=P.SERIES[1], lw=2, zorder=5, label="70/30 test (35)" if k == 0 else None)
    ax.axvline(CHANCE, color=P.MUTED, ls="--", lw=1)
    ax.set_yticks(range(len(rows)), [LABEL[c] for c in rows])
    ax.set_ylim(len(rows) - 0.5, -0.8)
    ax.text(CHANCE, -0.65, " chance 0.167", color=P.MUTED, fontsize=8, va="center")
    ax.set_xlabel("balanced accuracy (6 stages)")
    ax.xaxis.grid(True)
    ax.set_title(f"Adding the {block} to the rs-fMRI transformer", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.16), ncol=3, fontsize=8)
    savefig(fig, os.path.join(figdir, "fig0_model_comparison.png"))

    # fig1: confusion matrices, 70/30 primary and CV primary
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.2))
    for ax, C_, ttl in ((axs[0], np.array(sp[PRIMARY][0]["confusion_matrix"]), "70/30 test (35 subjects)"),
                        (axs[1], np.sum([np.array(r["confusion_matrix"]) for r in recs], axis=0),
                         "5x5 CV on 116, pooled")):
        Cn = C_ / C_.sum(1, keepdims=True)
        ax.imshow(Cn, cmap=P.SEQ, vmin=0, vmax=max(0.5, Cn.max()))
        for a, b in itertools.product(range(NS), range(NS)):
            ax.text(b, a, f"{Cn[a, b]:.2f}\n({C_[a, b]})", ha="center", va="center", fontsize=7,
                    color="#ffffff" if Cn[a, b] > 0.3 else P.INK2)
        ax.set_xticks(range(NS), STAGES)
        ax.set_yticks(range(NS), STAGES)
        ax.set_xlabel("predicted stage")
        ax.tick_params(length=0)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        ax.set_title(ttl, loc="left")
    axs[0].set_ylabel("true stage")
    fig.suptitle(f"{LABEL[PRIMARY]}: confusion (row share, counts)", x=0.01, ha="left",
                 fontweight="bold", fontsize=10)
    savefig(fig, os.path.join(figdir, "fig1_confusion_primary.png"))

    # fig3: gate and modality ablation by stage
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
    axs[0].errorbar(range(NS), gm, yerr=ge, color=P.SERIES[0], marker="o", ms=6, capsize=3)
    axs[0].set_title(f"Brain share of the fused representation (gate)\ntrend rho {gr[0]:+.2f}, p {gr[1]:.3f}",
                     loc="left", fontsize=9)
    axs[0].set_ylabel(f"gate (1 = brain, 0 = {block})")
    for k_, (v, lab) in enumerate(((pt, "full model"), (pnc, f"{block} removed"), (pnb, "brain removed"))):
        mm = [v[y == g].mean() for g in range(NS)]
        axs[1].plot(range(NS), mm, color=P.SERIES[k_], marker="o", ms=6, label=lab)
    axs[1].axhline(1 / NS, color=P.MUTED, ls="--", lw=1)
    axs[1].text(0, 1 / NS, " uniform guess", color=P.MUTED, fontsize=8, ha="left", va="top")
    axs[1].set_title("P(true stage) with one input removed", loc="left", fontsize=9)
    axs[1].set_ylabel("mean probability of the true stage")
    axs[1].legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=3, fontsize=8)
    for ax in axs:
        ax.set_xticks(range(NS), STAGES)
        ax.yaxis.grid(True)
    savefig(fig, os.path.join(figdir, "fig3_brain_vs_comorbidity.png"))

    # fig4: ordering, fusion vs imaging only
    pan = [c for c in (PRIMARY, "transformer_svmrfe50") if c in E]
    fig, axs = plt.subplots(1, len(pan), figsize=(5 * len(pan), 3.5), squeeze=False)
    for ax, c in zip(axs[0], pan):
        data = [E[c][y == g] for g in range(NS)]
        ax.boxplot(data, positions=range(NS), widths=0.5, patch_artist=True, showfliers=False,
                   medianprops={"color": P.INK}, boxprops={"facecolor": "#cde2fb", "edgecolor": P.SERIES[0]},
                   whiskerprops={"color": P.SERIES[0]}, capprops={"color": P.SERIES[0]})
        ax.plot(range(NS), [d.mean() for d in data], color=P.SERIES[1], marker="D", ms=6, label="stage mean")
        o = out["ordering"][c]
        ax.set_title(f"{LABEL[c]}\nrho {o['rho']:+.2f}, permutation p {o['p']:.3f}", loc="left", fontsize=9)
        ax.set_xticks(range(NS), STAGES)
        ax.set_xlabel("true stage")
        ax.yaxis.grid(True)
    axs[0, 0].set_ylabel("expected stage  E = sum p(k) k")
    axs[0, 0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Does adding comorbidity sharpen the CN -> AD ordering? (out-of-fold, 5 seeds)",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    savefig(fig, os.path.join(figdir, "fig4_stage_ordering.png"))

    json.dump(out, open(os.path.join(args.out, "comorbidity_patterns.json"), "w"), indent=1, default=float)
    open(os.path.join(args.out, "comorbidity_patterns.log"), "w", encoding="utf-8").write("\n".join(_log) + "\n")
    log(f"\nwrote {os.path.relpath(os.path.join(args.out, 'comorbidity_patterns.json'), ROOT)}")


if __name__ == "__main__":
    main()
