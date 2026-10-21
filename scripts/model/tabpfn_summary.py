"""
TabPFN vs the Multi-Attention Transformer vs logistic regression, on identical
inputs, splits and folds. Reads only saved records.

  TabPFN       derivatives/model_training/tabpfn/{split70,cv116,cv116k10}_tabpfn_*.json
  transformer  rs-fMRI only : transformer_svmrfe (cv116), transformer_comorbidity8 (cv116k10),
                              transformer_comorbidity (split70)
               + 8 comorb.  : transformer_comorbidity8
               + MMSE/CDR-SB: transformer_clinical10
  logistic     the same folders (logreg_svmrfe50 / logreg_svmrfe50_clin / logreg_clin)

Run:  python scripts/model/tabpfn_summary.py
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
TAB = os.path.join(TD, "tabpfn")
IMG, C23 = os.path.join(TD, "transformer_svmrfe"), os.path.join(TD, "transformer_comorbidity")
A, B = os.path.join(TD, "transformer_comorbidity8"), os.path.join(TD, "transformer_clinical10")
CU = os.path.join(TD, "transformer_custom5")             # logistic references for the custom5 inputs
DESIGNS = [("cv116", "5-fold x 5"), ("cv116k10", "10-fold x 5"), ("split70", "70/30 test (35)")]
GROUPS = [
    ("rs-fMRI only", "tabpfn_svmrfe50",
     {"cv116": (IMG, "transformer_svmrfe50"), "cv116k10": (A, "transformer_svmrfe50"),
      "split70": (C23, "transformer_svmrfe50")},
     {"cv116": (IMG, "logreg_svmrfe50"), "cv116k10": (A, "logreg_svmrfe50"), "split70": (C23, "logreg_svmrfe50")}),
    ("rs-fMRI + 8 comorbidities", "tabpfn_svmrfe50_comorb8",
     {d: (A, "transformer_svmrfe50_clin") for d, _ in DESIGNS},
     {d: (A, "logreg_svmrfe50_clin") for d, _ in DESIGNS}),
    ("rs-fMRI + 8 comorbidities + MMSE + CDR-SB", "tabpfn_svmrfe50_clin10",
     {d: (B, "transformer_svmrfe50_clin") for d, _ in DESIGNS},
     {d: (B, "logreg_svmrfe50_clin") for d, _ in DESIGNS}),
    ("8 comorbidities only", "tabpfn_comorb8", {}, {d: (A, "logreg_clin") for d, _ in DESIGNS}),
    ("MMSE + CDR-SB + GDS + APOE4 + cardiovascular", "tabpfn_custom5", {},
     {d: (CU, "logreg_clin") for d, _ in DESIGNS}),
    ("rs-fMRI + MMSE + CDR-SB + GDS + APOE4 + cardiovascular", "tabpfn_svmrfe50_custom5", {},
     {d: (CU, "logreg_svmrfe50_clin") for d, _ in DESIGNS}),
]
OUT = TAB

_log = []


def log(m=""):
    print(m, flush=True)
    _log.append(str(m))


def qc(recs, y):
    bad = 0
    for r in recs:
        m = evaluate(np.asarray(r["probs"]), np.asarray(r["true"]), pred=np.asarray(r["pred"]))
        bad += sum(abs(m[k] - r[k]) > 1e-9 for k in P.KEYS)
        bad += r["true"] != [int(y[i]) for i in r["test_index"]]
    return bad


def stats_of(recs, design):
    if design == "split70":
        r = recs[0]
        return {"acc": r["accuracy"], "bal": r["balanced_accuracy"], "bal_sd": None,
                "correct": int(round(r["accuracy"] * r["n_test"])), "n_test": r["n_test"],
                "train": r.get("train_balanced_accuracy")}
    s = H.summarise(recs)
    tr = [r["train_balanced_accuracy"] for r in recs if r.get("train_balanced_accuracy") is not None]
    return {"acc": s["accuracy"][0], "bal": s["balanced_accuracy"][0], "bal_sd": s["balanced_accuracy"][1],
            "train": float(np.mean(tr)) if tr else None,
            "seed_means": P.seed_means(recs, "balanced_accuracy").tolist()}


def main():
    for p in (os.path.join(OUT, "tabpfn_summary.json"), os.path.join(OUT, "figures")):
        if os.path.exists(p):
            raise SystemExit(f"COLLISION: {p} exists; not overwritten")
    os.makedirs(os.path.join(OUT, "figures"))
    y = np.load(os.path.join(H.SRC, "y.npy")).astype(int)
    out = {"designs": {}, "paired": {}, "qc": {}}

    log("=" * 100)
    log("TabPFN v2 vs MULTI-ATTENTION TRANSFORMER vs LOGISTIC REGRESSION -- 116 subjects, 6 stages, chance 0.167")
    log("=" * 100)
    log("same splits, same folds, same SVM-RFE features per fold; accuracy = plain, bal = balanced accuracy")

    checks = bad = 0
    for design, dlabel in DESIGNS:
        log(f"\n[{dlabel}]")
        log(f"  {'inputs':56s} {'TabPFN':>16s} {'transformer':>16s} {'logistic reg.':>16s}   (accuracy / balanced)")
        out["designs"][design] = {}
        for label, tcand, tr_map, lr_map in GROUPS:
            row = {}
            tab = P.load(design, tcand, TAB)
            for name, recs in (("tabpfn", tab),
                               ("transformer", P.load(design, tr_map[design][1], tr_map[design][0]) if design in tr_map else None),
                               ("logreg", P.load(design, lr_map[design][1], lr_map[design][0]) if design in lr_map else None)):
                if not recs:
                    continue
                checks += 1
                bad += qc(recs, y)
                row[name] = stats_of(recs, design)
                row[name]["_recs"] = recs
            if tab and row.get("transformer") and "support" in tab[0]:
                same_feat = all(a.get("support") == b.get("support")
                                for a, b in zip(tab, row["transformer"]["_recs"]))
                same_folds = [(a["seed"], a["fold"], a["fold_fingerprint"]) for a in tab] == \
                             [(b["seed"], b["fold"], b["fold_fingerprint"]) for b in row["transformer"]["_recs"]]
                out["qc"][f"{design}|{label}"] = {"same_features": same_feat, "same_folds": same_folds}
                if not same_folds:
                    raise SystemExit(f"{design} {label}: TabPFN and transformer folds differ")

            def cell(k):
                v = row.get(k)
                return f"{v['acc']:.3f} / {v['bal']:.3f}" if v else "-"
            log(f"  {label:56s} {cell('tabpfn'):>16s} {cell('transformer'):>16s} {cell('logreg'):>16s}")
            out["designs"][design][label] = {k: {kk: vv for kk, vv in v.items() if kk != "_recs"}
                                             for k, v in row.items()}
            if design != "split70" and "tabpfn" in row:
                for other in ("transformer", "logreg"):
                    if other in row:
                        d = np.array(row["tabpfn"]["seed_means"]) - np.array(row[other]["seed_means"])
                        wins = int((d > 0).sum())
                        verdict = ("TabPFN BETTER" if wins >= 4 and d.mean() > 0 else
                                   f"{other.upper()} BETTER" if (d < 0).sum() >= 4 and d.mean() < 0 else "TIED")
                        out["paired"][f"{design}|{label}|tabpfn-{other}"] = {
                            "mean_diff": float(d.mean()), "seeds": wins, "verdict": verdict}
        if design == "cv116":
            sh = P.load("cv116", "tabpfn_svmrfe50_comorb8_SHUFFLED", TAB)
            if sh:
                checks += 1
                bad += qc(sh, y)
                b = H.summarise(sh)["balanced_accuracy"][0]
                out["shuffled_control"] = b
                log(f"  {'shuffled-label control (rs-fMRI + 8 comorb.)':44s} {'':>4s}{b:.3f} balanced  "
                    + ("OK (near chance)" if abs(b - P.CHANCE) < 0.06 else "CHECK"))
            trn = {lab: v["tabpfn"]["train"] for lab, v in out["designs"]["cv116"].items()
                   if "tabpfn" in v and v["tabpfn"]["train"] is not None}
            log("  TabPFN training-subject balanced accuracy (its own context, so near 100% by design): "
                + ", ".join(f"{k} {v:.2f}" for k, v in trn.items()))

    log("\n[PAIRED, balanced accuracy, identical folds; rule: >= 4/5 seeds and mean > 0]")
    for k, v in out["paired"].items():
        log(f"  {k:70s} {v['mean_diff']:+.3f}  seeds {v['seeds']}/5  -> {v['verdict']}")
    log(f"\nQC: {checks} record sets, {bad} metric mismatches; "
        f"TabPFN used the transformer's 50 features in every fold: "
        f"{all(v['same_features'] for v in out['qc'].values()) if out['qc'] else 'n/a'}")
    if bad:
        raise SystemExit("QC failed")

    # figure: 5x5 CV balanced accuracy per input set, three models side by side
    rows = [g[0] for g in GROUPS]
    fig, ax = plt.subplots(figsize=(9.2, 4.8))
    for j, (name, lab) in enumerate((("tabpfn", "TabPFN v2"), ("transformer", "Multi-Attention Transformer"),
                                     ("logreg", "Logistic regression"))):
        for i, r in enumerate(rows):
            v = out["designs"]["cv116"].get(r, {}).get(name)
            if v:
                ax.errorbar(v["bal"], i + (j - 1) * 0.22, xerr=v["bal_sd"], fmt="o", ms=7, capsize=3,
                            color=P.SERIES[j], label=lab if i == 0 or (name == "logreg" and i == 3) else None)
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.axvline(P.CHANCE, color=P.MUTED, ls="--", lw=1)
    ax.set_yticks(range(len(rows)), rows, fontsize=8.5)
    ax.set_ylim(len(rows) - 0.5, -0.8)
    ax.text(P.CHANCE, -0.65, " chance 0.167", color=P.MUTED, fontsize=8, va="center")
    ax.set_xlabel("balanced accuracy, 5-fold x 5 CV (mean +/- sd of 5 seed means)")
    ax.xaxis.grid(True)
    ax.set_title("TabPFN vs transformer vs logistic regression, same inputs and folds", loc="left")
    ax.legend(uniq.values(), uniq.keys(), loc="upper left", bbox_to_anchor=(0, -0.18), ncol=3, fontsize=8)
    fig.savefig(os.path.join(OUT, "figures", "fig_tabpfn_comparison.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    log("    figure derivatives/model_training/tabpfn/figures/fig_tabpfn_comparison.png")

    json.dump(out, open(os.path.join(OUT, "tabpfn_summary.json"), "w"), indent=1, default=float)
    open(os.path.join(OUT, "tabpfn_summary.log"), "w", encoding="utf-8").write("\n".join(_log) + "\n")
    log(f"\nwrote {os.path.relpath(os.path.join(OUT, 'tabpfn_summary.json'), ROOT)}")


if __name__ == "__main__":
    main()
