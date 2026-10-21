"""
Summary: 5-fold x 5 vs 10-fold x 5 cross-validation, and the learning curve.

Reads only saved records:
  5x5   model A  derivatives/model_training/transformer_comorbidity8/cv116_*.json
        model B  derivatives/model_training/transformer_clinical10/cv116_*.json
        imaging  derivatives/model_training/transformer_svmrfe/cv116_*.json
  10x5  the same folders, cv116k10_*.json (imaging-only rerun in transformer_comorbidity8)
  curve derivatives/model_training/learning_curve/learning_curve_records.json

Answers
  * does training on ~104 instead of ~93 subjects per fold change accuracy?
  * is the 5x5 estimate stable when the folds change?
  * does accuracy still rise with more training subjects (would more data help)?

Run:  python scripts/model/cv_scheme_summary.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import stage_patterns as P  # noqa: E402  (plot style, palette, helpers)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from train_transformer import evaluate  # noqa: E402

H = P.H
ROOT = P.ROOT
TD = os.path.join(ROOT, "derivatives", "model_training")
A_DIR, B_DIR = os.path.join(TD, "transformer_comorbidity8"), os.path.join(TD, "transformer_clinical10")
IMG_DIR = os.path.join(TD, "transformer_svmrfe")
LC = os.path.join(TD, "learning_curve", "learning_curve_records.json")
OUT = os.path.join(TD, "cv_scheme_and_learning_curve")
STAGES, NS, CHANCE = P.STAGES, P.NS, P.CHANCE

# (label, folder for 5x5, folder for 10x5, candidate)
ROWS = [("Model A: transformer + 8 comorbidities", A_DIR, A_DIR, "transformer_svmrfe50_clin"),
        ("Model B: transformer + 8 comorbidities + MMSE + CDR-SB", B_DIR, B_DIR, "transformer_svmrfe50_clin"),
        ("Transformer, rs-fMRI only", IMG_DIR, A_DIR, "transformer_svmrfe50"),
        ("Logistic reg., rs-fMRI + 8 comorbidities", A_DIR, A_DIR, "logreg_svmrfe50_clin"),
        ("Logistic reg., rs-fMRI only", IMG_DIR, A_DIR, "logreg_svmrfe50"),
        ("Logistic reg., 8 comorbidities only", A_DIR, A_DIR, "logreg_clin"),
        ("Logistic reg., MMSE + CDR-SB only (ceiling)", B_DIR, B_DIR, "ceiling_mmse_cdrsb")]
LC_LABEL = {"transformer_svmrfe50_clin": "Transformer + 8 comorbidities (A)",
            "transformer_svmrfe50": "Transformer, rs-fMRI only",
            "logreg_svmrfe50_clin": "Logistic reg., rs-fMRI + 8 comorbidities",
            "logreg_svmrfe50": "Logistic reg., rs-fMRI only"}

# colour = inputs (blue: with comorbidities, orange: rs-fMRI only); line = model (solid transformer, dashed logistic)
LC_STYLE = {"transformer_svmrfe50_clin": (0, "-", "o"), "transformer_svmrfe50": (1, "-", "o"),
            "logreg_svmrfe50_clin": (0, "--", "s"), "logreg_svmrfe50": (1, "--", "s")}

_log = []


def log(m=""):
    print(m, flush=True)
    _log.append(str(m))


def qc(recs, y, n_folds):
    bad = 0
    for r in recs:
        m = evaluate(np.asarray(r["probs"]), np.asarray(r["true"]), pred=np.asarray(r["pred"]))
        bad += sum(abs(m[k] - r[k]) > 1e-9 for k in P.KEYS)
        bad += r["true"] != [int(y[i]) for i in r["test_index"]]
    for s in H.SEEDS:
        sub = [r for r in recs if r["seed"] == s]
        bad += len(sub) != n_folds
        bad += sorted(i for r in sub for i in r["test_index"]) != list(range(len(y)))
    return bad


def main():
    for p in (os.path.join(OUT, "summary.json"), os.path.join(OUT, "figures")):
        if os.path.exists(p):
            raise SystemExit(f"COLLISION: {p} exists; not overwritten")
    os.makedirs(os.path.join(OUT, "figures"))
    y = np.load(os.path.join(H.SRC, "y.npy")).astype(int)
    out = {"schemes": {}, "paired": {}, "learning_curve": {}}

    log("=" * 100)
    log("CROSS-VALIDATION SCHEME (5-fold x 5 vs 10-fold x 5) AND LEARNING CURVE -- 116 subjects, 6 stages")
    log("=" * 100)

    # ------------------------------------------------------------ schemes
    log("\n[1] SAME MODELS, TWO CV SCHEMES (balanced accuracy; chance 0.167; mean +/- sd of 5 seed means)")
    log(f"  {'model':54s} {'5x5: train':>10s} {'validation':>14s} | {'10x5: train':>11s} {'validation':>14s} {'change':>7s}")
    data = {}
    checks = 0
    for label, d5, d10, cand in ROWS:
        row = {}
        for scheme, d, k in (("5x5", d5, 5), ("10x5", d10, 10)):
            recs = P.load("cv116" if k == 5 else "cv116k10", cand, d)
            if not recs:
                continue
            bad = qc(recs, y, k)
            checks += 1
            if bad:
                raise SystemExit(f"QC failed for {cand} {scheme}: {bad} problems")
            s = H.summarise(recs)
            row[scheme] = {"train": float(np.mean([r["train_balanced_accuracy"] for r in recs])),
                           "val": s["balanced_accuracy"], "accuracy": s["accuracy"],
                           "within_one": float(np.mean([P.within_one(r["true"], r["pred"]) for r in recs])),
                           "n_train_mean": float(np.mean([r["n_train"] for r in recs])),
                           "seed_means": P.seed_means(recs, "balanced_accuracy").tolist()}
        data[cand + "|" + label] = row
        out["schemes"][label] = row
        f5 = (f"{row['5x5']['train']:10.3f} {row['5x5']['val'][0]:.3f}+/-{row['5x5']['val'][1]:.3f}"
              if "5x5" in row else f"{'-':>10s} {'-':>14s}")
        f10 = (f"{row['10x5']['train']:11.3f} {row['10x5']['val'][0]:.3f}+/-{row['10x5']['val'][1]:.3f}"
               if "10x5" in row else f"{'-':>11s} {'-':>14s}")
        ch = (f"{row['10x5']['val'][0] - row['5x5']['val'][0]:+7.3f}" if len(row) == 2 else "")
        log(f"  {label:54s} {f5} | {f10} {ch}")
    log(f"  QC: {checks} record sets recomputed from saved predictions, every subject tested once per seed")
    log("  training subjects per fold: 5-fold ~93, 10-fold ~104")

    log("\n[2] PAIRED COMPARISONS INSIDE EACH SCHEME (rule: >= 4/5 seeds and mean > 0)")

    def get(cand, label_start):
        for key, row in data.items():
            if key.startswith(cand + "|") and key.split("|", 1)[1].startswith(label_start):
                return row
        return {}
    pairs = [("Model A vs rs-fMRI-only transformer", get("transformer_svmrfe50_clin", "Model A"),
              get("transformer_svmrfe50", "Transformer, rs-fMRI")),
             ("Model B vs Model A", get("transformer_svmrfe50_clin", "Model B"),
              get("transformer_svmrfe50_clin", "Model A")),
             ("Model B vs MMSE + CDR-SB only", get("transformer_svmrfe50_clin", "Model B"),
              get("ceiling_mmse_cdrsb", "Logistic reg., MMSE"))]
    for name, a, b in pairs:
        for scheme in ("5x5", "10x5"):
            if scheme in a and scheme in b:
                d = np.array(a[scheme]["seed_means"]) - np.array(b[scheme]["seed_means"])
                wins = int((d > 0).sum())
                verdict = ("FIRST BETTER" if wins >= 4 and d.mean() > 0 else
                           "SECOND BETTER" if (d < 0).sum() >= 4 and d.mean() < 0 else "TIED")
                out["paired"][f"{name} ({scheme})"] = {"mean_diff": float(d.mean()), "seeds": wins,
                                                      "verdict": verdict}
                log(f"  {name:38s} {scheme:5s} {d.mean():+.3f}  seeds {wins}/5  -> {verdict}")

    # ------------------------------------------------------------ figure 1
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    labels = [k.split("|", 1)[1] for k in data]
    for i, key in enumerate(data):
        for j, (scheme, col, mk) in enumerate((("5x5", P.SERIES[0], "D"), ("10x5", P.SERIES[1], "o"))):
            if scheme in data[key]:
                v = data[key][scheme]["val"]
                ax.errorbar(v[0], i + (j - 0.5) * 0.28, xerr=v[1], fmt=mk, color=col, ms=7, capsize=3,
                            label=("5-fold x 5 (~93 train)" if scheme == "5x5" else "10-fold x 5 (~104 train)")
                            if i == 0 else None)
    ax.axvline(CHANCE, color=P.MUTED, ls="--", lw=1)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_ylim(len(labels) - 0.5, -0.8)
    ax.text(CHANCE, -0.65, " chance 0.167", color=P.MUTED, fontsize=8, va="center")
    ax.set_xlabel("validation balanced accuracy (mean +/- sd of 5 seed means)")
    ax.xaxis.grid(True)
    ax.set_title("Same models, 5-fold vs 10-fold cross-validation", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=2, fontsize=8)
    fig.savefig(os.path.join(OUT, "figures", "fig1_cv_schemes.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------- learning curve
    log("\n[3] LEARNING CURVE (test fold fixed; training subsets nested and stratified)")
    if not os.path.exists(LC):
        log("  learning-curve records not found; skipped")
    else:
        lc = json.load(open(LC))["records"]
        fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
        for k, model in enumerate(LC_LABEL):
            rr = [r for r in lc if r["model"] == model]
            if not rr:
                continue
            sizes = list(dict.fromkeys(r["size"] for r in rr))
            xs, vm, ve, tm = [], [], [], []
            for sz in sizes:
                q = [r for r in rr if r["size"] == sz]
                v = np.array([r["balanced_accuracy"] for r in q])
                xs.append(np.mean([r["n_train"] for r in q]))
                vm.append(v.mean())
                ve.append(v.std(ddof=1) / np.sqrt(len(v)))
                tm.append(np.mean([r["train_balanced_accuracy"] for r in q]))
            # fold-paired trend: does each fold's score rise with training size?
            rho = P.spear([r["n_train"] for r in rr], [r["balanced_accuracy"] for r in rr])
            folds = sorted({(r["seed"], r["fold"]) for r in rr})
            gain = np.mean([[r["balanced_accuracy"] for r in rr if (r["seed"], r["fold"]) == f and r["size"] == "all"][0]
                            - [r["balanced_accuracy"] for r in rr if (r["seed"], r["fold"]) == f and r["size"] == 30][0]
                            for f in folds])
            last = np.mean([[r["balanced_accuracy"] for r in rr if (r["seed"], r["fold"]) == f and r["size"] == "all"][0]
                            - [r["balanced_accuracy"] for r in rr if (r["seed"], r["fold"]) == f and r["size"] == 75][0]
                            for f in folds])
            out["learning_curve"][model] = {"n_train": xs, "val_mean": vm, "val_se": ve, "train_mean": tm,
                                            "rho": rho[0], "p": rho[1], "gain_30_to_all": float(gain),
                                            "gain_75_to_all": float(last), "n_folds": len(folds)}
            log(f"  {LC_LABEL[model]:42s} " + "  ".join(f"n={x:5.1f}: {m_:.3f}" for x, m_ in zip(xs, vm)))
            log(f"  {'':42s} gain 30 -> all {gain:+.3f}, 75 -> all {last:+.3f}; "
                f"trend rho {rho[0]:+.2f} p {rho[1]:.3f} ({len(folds)} folds)")
            ci, ls, mk = LC_STYLE[model]
            axs[0].errorbar(xs, vm, yerr=ve, color=P.SERIES[ci], marker=mk, ms=6, capsize=3, ls=ls,
                            label=LC_LABEL[model])
            axs[1].plot(xs, tm, color=P.SERIES[ci], marker=mk, ms=6, ls=ls, label=LC_LABEL[model])
        for ax, ttl in ((axs[0], "Validation (held-out fold)"), (axs[1], "Training subjects")):
            ax.axhline(CHANCE, color=P.MUTED, ls="--", lw=1)
            ax.set_xlabel("training subjects")
            ax.set_ylabel("balanced accuracy")
            ax.yaxis.grid(True)
            ax.set_title(ttl, loc="left")
        axs[0].legend(loc="upper left", bbox_to_anchor=(0, -0.18), ncol=2, fontsize=8)
        fig.suptitle("Learning curve: accuracy vs number of training subjects (mean +/- SE over folds)",
                     x=0.01, ha="left", fontweight="bold", fontsize=10)
        fig.savefig(os.path.join(OUT, "figures", "fig2_learning_curve.png"), dpi=160, bbox_inches="tight")
        plt.close(fig)

    json.dump(out, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
    open(os.path.join(OUT, "summary.log"), "w", encoding="utf-8").write("\n".join(_log) + "\n")
    log(f"\nwrote {os.path.relpath(os.path.join(OUT, 'summary.json'), ROOT)}")


if __name__ == "__main__":
    main()
