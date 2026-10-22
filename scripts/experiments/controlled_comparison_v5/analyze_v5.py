"""
v5 verification and controlled comparison (PLAN.md). Reads stored predictions only; trains nothing; refuses to
overwrite. Step 1 verifies labels, splits and metric calculations; steps 2-4 compare stored candidates and the new
TabPFN MMSE + CDR-SB reference on the identical 25 outer splits with the project's statistics.

Run:  python scripts/experiments/controlled_comparison_v5/analyze_v5.py
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
from sklearn.metrics import balanced_accuracy_score, f1_score  # noqa: E402

V3A, A1, K, P, T = AO.V3A, AO.A1, AO.K, AO.P, AO.T
ROOT = K.ROOT
EXP = os.path.join(ROOT, "derivatives", "experiments", "controlled_comparison_v5")
D = os.path.join(ROOT, "derivatives")
STAGES, NS, SEEDS = K.STAGES, K.NS, list(K.SEEDS)
FPS = ["3edbea19c1ee3c08", "d53ad26a06b7d5c5", "2a8e70dd0eb80cae", "d54bf96735be41dd", "8b7f4cd9e512b26a"]
NAMES = {"BASE": "TabPFN imaging-only baseline (SVM-RFE k=50 of 984)",
         "L492": "TabPFN imaging-only, 492 mALFF+mReHo, no selection",
         "BAL": "TabPFN imaging-only baseline + class balancing",
         "T_V2": "Transformer imaging-only (corrected v2)",
         "COMORB8": "TabPFN imaging + 8 non-circular comorbidities",
         "T_COMORB8": "Transformer imaging + 8 non-circular comorbidities",
         "MMSE_CDR": "CLINICAL REFERENCE: TabPFN MMSE + CDR-SB only",
         "CLIN_FULL": "CLINICAL REFERENCE: TabPFN imaging + comorbidities + MMSE + CDR-SB"}
GROUP = {"BASE": "imaging-only", "L492": "imaging-only", "BAL": "imaging-only", "T_V2": "imaging-only",
         "COMORB8": "non-circular multimodal", "T_COMORB8": "non-circular multimodal",
         "MMSE_CDR": "clinical reference", "CLIN_FULL": "clinical reference"}
SOURCE = {"BASE": "experiments/biomarker_ablation_v3/grid/A_all4 (= model_training/tabpfn/cv116_tabpfn_svmrfe50.json)",
          "L492": "experiments/tabpfn_local492_v4/grid", "BAL": "experiments/biomarker_ablation_v3/grid/A_all4 (pred_balanced)",
          "T_V2": "experiments/transformer_nested_v2/grid (FIXED)", "COMORB8": "model_training/tabpfn/cv116_tabpfn_svmrfe50_comorb8.json",
          "T_COMORB8": "experiments/biomarker_ablation_v3/transformer/comorb8 (FIXED)",
          "MMSE_CDR": "experiments/controlled_comparison_v5/clinical_reference/grid (NEW RUN)",
          "CLIN_FULL": "model_training/tabpfn/cv116_tabpfn_svmrfe50_clin10.json"}
SHORT = {"T_V2": "Transformer\nimaging", "BASE": "TabPFN\nimaging\n(baseline)", "L492": "TabPFN\n492 mALFF+mReHo",
         "BAL": "TabPFN\n+ balancing", "T_COMORB8": "Transformer\n+ comorbidities", "COMORB8": "TabPFN\n+ comorbidities",
         "MMSE_CDR": "TabPFN\nMMSE + CDR-SB", "CLIN_FULL": "TabPFN imaging\n+ comorb. + MMSE/CDR"}
AO.NAMES.update(NAMES)
V3A.LABEL.update(NAMES)


def draw_figure(desc, path):
    """Balanced accuracy with 95% CI per model, grouped; desc = metrics table indexed by model."""
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": P.AXIS, "axes.labelcolor": P.INK2, "xtick.color": P.INK2,
                         "ytick.color": P.INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "grid.color": P.GRID, "figure.facecolor": P.SURFACE, "axes.facecolor": P.SURFACE})
    order = ["T_V2", "BASE", "L492", "BAL", "T_COMORB8", "COMORB8", "MMSE_CDR", "CLIN_FULL"]
    gcol = {"imaging-only": P.SERIES[0], "non-circular multimodal": P.SERIES[2], "clinical reference": P.MUTED}
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axvspan(5.5, 7.5, color=P.GRID, zorder=0)
    for i, m in enumerate(order):
        r = desc.loc[m]
        ax.errorbar([i], [r.balanced_accuracy], yerr=[[r.balanced_accuracy - r.bal_ci_low], [r.bal_ci_high - r.balanced_accuracy]],
                    fmt="D", ms=6, color=gcol[GROUP[m]], elinewidth=2, capsize=0)
        ax.text(i + 0.12, r.balanced_accuracy, f"{100 * r.balanced_accuracy:.1f}", fontsize=7.5, color=P.INK, va="center")
    ax.axhline(1 / 6, color=P.MUTED, ls="--", lw=1)
    ax.text(5.4, 1 / 6, "chance 16.7%", color=P.MUTED, fontsize=7, va="bottom", ha="right")
    ax.text(6.5, 0.565, "clinical reference: MMSE/CDR are\npart of the diagnosis (not an fMRI result)", ha="center", va="top",
            fontsize=7, color=P.INK2)
    ax.set_ylim(0.14, 0.58)
    ax.set_xticks(range(len(order)), [SHORT[m] for m in order], fontsize=7)
    for lab, c in gcol.items():
        ax.plot([], [], "D", color=c, label=lab)
    ax.legend(fontsize=7, frameon=False, loc="upper left")
    ax.set_ylabel("balanced accuracy (seed-pooled, 95% CI)")
    ax.set_title("Same 25 outer splits: imaging-only, non-circular multimodal, and clinical reference", loc="left", fontsize=9)
    ax.yaxis.grid(True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def grid(path):
    return {(r["seed"], r["fold"]): r for r in (json.load(open(f)) for f in glob.glob(os.path.join(path, "*.json")))}


def records(path):
    return {(r["seed"], r["fold"]): r for r in json.load(open(path))["records"]}


def main():
    if os.path.exists(os.path.join(EXP, "analysis", "summary.txt")):
        raise SystemExit("analysis already written; not overwritten")
    for sub in ("verification", "analysis"):
        os.makedirs(os.path.join(EXP, sub), exist_ok=True)
    y = np.load(os.path.join(D, "model_ready", "y.npy")).astype(int)
    d = T.S.load_cohort()
    plan = T.S.cv_plan(y, d[4], 5)
    keys = [(s, f) for s in SEEDS for f in range(5)]
    raw = {"BASE": grid(os.path.join(D, "experiments", "biomarker_ablation_v3", "grid", "A_all4")),
           "L492": grid(os.path.join(D, "experiments", "tabpfn_local492_v4", "grid")),
           "T_V2": grid(os.path.join(D, "experiments", "transformer_nested_v2", "grid")),
           "COMORB8": records(os.path.join(D, "model_training", "tabpfn", "cv116_tabpfn_svmrfe50_comorb8.json")),
           "T_COMORB8": grid(os.path.join(D, "experiments", "biomarker_ablation_v3", "transformer", "comorb8")),
           "MMSE_CDR": grid(os.path.join(EXP, "clinical_reference", "grid")),
           "CLIN_FULL": records(os.path.join(D, "model_training", "tabpfn", "cv116_tabpfn_svmrfe50_clin10.json"))}
    raw["BAL"] = raw["BASE"]
    orig = records(T.TABPFN_BASELINE)

    def te_of(r):
        return r.get("test_idx", r.get("test_index"))

    def pred_of(m, r):
        if m == "BAL":
            return r["pred_balanced"]
        return r["pred"]["FIXED"] if isinstance(r["pred"], dict) else r["pred"]

    srcs = {m: {k: (np.array(te_of(r[k])), np.array(pred_of(m, r[k]))) for k in keys} for m, r in raw.items()}

    # ------------------------------------------------------------- step 1: verification
    meta = json.load(open(os.path.join(D, "model_ready", "metadata.json")))
    stages = np.load(os.path.join(D, "model_ready", "stages.npy")).astype(str)
    acq = np.load(os.path.join(D, "model_ready", "acq_ids.npy")).astype(str)
    V, lines = [], []

    def check(name, ok, detail=""):
        V.append({"check": name, "pass": bool(ok), "detail": detail})
        lines.append(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))

    check("label mapping CN=0, SMC=1, EMCI=2, MCI=3, LMCI=4, AD=5",
          meta["stage_to_index"] == {s: i for i, s in enumerate(["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"])}
          and sorted(set(zip(y.tolist(), stages.tolist()))) == list(enumerate(["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"])))
    counts = np.bincount(y, minlength=NS).tolist()
    check("class counts CN 20, SMC 20, EMCI 20, MCI 16, LMCI 20, AD 20", counts == [20, 20, 20, 16, 20, 20], str(counts))
    check("one scan per subject (116 subjects, 116 acquisitions)", len({a.split("|")[0] for a in acq}) == 116 == len(acq))
    check("fold fingerprints are the project's 5", [plan[s][0] for s in SEEDS] == FPS, ", ".join(plan[s][0] for s in SEEDS))
    for m in raw:
        same = all(sorted(srcs[m][k][0].tolist()) == sorted(plan[k[0]][1][k[1]][1].tolist()) for k in keys)
        fp_ok = all(raw[m][k].get("fold_fingerprint", plan[k[0]][0]) == plan[k[0]][0] for k in keys)
        lab = all(np.array_equal(np.array(raw[m][k]["true"]), y[srcs[m][k][0]]) for k in keys if "true" in raw[m][k])
        check(f"{m}: 25/25 outer test sets identical to the plan, fingerprints and stored labels match", same and fp_ok and lab)
    check("baseline records equal the original stored TabPFN baseline (predictions, 25/25)",
          all(np.array_equal(srcs["BASE"][k][1], np.array(orig[k]["pred"])) for k in keys))
    worst, stored_diff = 0.0, 0.0
    for m in raw:
        for k in keys:
            te, pr = srcs[m][k]
            e = K.H.evaluate(y[te], pr)
            worst = max(worst, abs(e["balanced_accuracy"] - balanced_accuracy_score(y[te], pr)),
                        abs(e["macro_f1"] - f1_score(y[te], pr, labels=range(NS), average="macro", zero_division=0)))
            r = raw[m][k]
            st = (r.get("heldout", {}).get("balanced" if m == "BAL" else "raw")
                  or r.get("heldout", {}).get("FIXED") if "heldout" in r else r)
            if isinstance(st, dict) and "balanced_accuracy" in st:
                stored_diff = max(stored_diff, abs(st["balanced_accuracy"] - e["balanced_accuracy"]),
                                  abs(st.get("macro_f1", e["macro_f1"]) - e["macro_f1"]),
                                  abs(st.get("accuracy", e["accuracy"]) - e["accuracy"]))
    check("balanced accuracy and macro-F1 from project code equal scikit-learn (all sources, all splits)", worst < 1e-12,
          f"max difference {worst:.1e}")
    check("recomputed per-split metrics equal each record's stored metrics", stored_diff < 1e-9, f"max difference {stored_diff:.1e}")
    json.dump(V, open(os.path.join(EXP, "verification", "verification.json"), "w"), indent=2)
    open(os.path.join(EXP, "verification", "verification.log"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
    if not all(v["pass"] for v in V):
        print("\n".join(lines))
        raise SystemExit("STOP: verification failed; nothing interpreted")

    # ------------------------------------------------------------- per-model metrics
    rng = np.random.default_rng(0)
    boots = [np.concatenate([rng.choice(np.flatnonzero(y == g), (y == g).sum(), replace=True) for g in range(NS)])
             for _ in range(2000)]
    n_te = np.mean([len(srcs["BASE"][k][0]) for k in keys])
    ratio = n_te / (116 - n_te)
    rows, cms, folds = [], {}, []
    for m in NAMES:
        row, cm, _ = V3A.describe(m, srcs[m], y)
        M = AO.oof(srcs[m])
        for key in ("f1", "acc", "inter"):
            row[f"{key}_mean"] = AO.seed_avg(y, M, key)
            row[f"{key}_ci_low"], row[f"{key}_ci_high"] = np.percentile([AO.seed_avg(y, M, key, i) for i in boots], [2.5, 97.5])
        ft = AO.fold_table(srcs[m], y)
        row["f1_fold_mean"], row["f1_fold_sd"] = ft.f1.mean(), ft.f1.std(ddof=1)
        row["group"], row["source"] = GROUP[m], SOURCE[m]
        rows.append(row)
        cms[m] = cm
        ft.insert(0, "model", m)
        folds.append(ft)
    desc = pd.DataFrame(rows).set_index("model")
    desc.to_csv(os.path.join(EXP, "analysis", "metrics.csv"))
    pd.concat(folds).to_csv(os.path.join(EXP, "analysis", "fold_level.csv"), index=False)
    pd.DataFrame([{"model": m, "true": STAGES[a], **{f"pred_{STAGES[b]}": int(cms[m][a, b]) for b in range(NS)}}
                  for m in cms for a in range(NS)]).to_csv(os.path.join(EXP, "analysis", "confusion_matrices.csv"), index=False)

    # ------------------------------------------------------------- comparisons
    plan_cmp = [("L492", "BASE", "candidate"), ("BAL", "BASE", "candidate"), ("COMORB8", "BASE", "candidate"),
                ("T_COMORB8", "T_V2", "context: transformer"), ("MMSE_CDR", "BASE", "clinical reference"),
                ("CLIN_FULL", "BASE", "clinical reference"), ("CLIN_FULL", "MMSE_CDR", "clinical reference")]
    cmp = pd.DataFrame([{**AO.paired(a, b, srcs, y, boots, ratio, key, key != "mae"), "role": role}
                        for a, b, role in plan_cmp for key in ("bal", "f1", "acc", "inter")])
    cand_bal = (cmp.role == "candidate") & (cmp.metric == "bal")
    cmp["holm_p"] = np.nan
    cmp.loc[cand_bal, "holm_p"] = V3A.holm(cmp.loc[cand_bal, "corrected_t_p"].to_numpy())
    cmp.to_csv(os.path.join(EXP, "analysis", "paired_comparisons.csv"), index=False)
    decisions = {}
    for a, b, role in plan_cmp[:3]:
        q = cmp[(cmp.A == a) & (cmp.B == b)].set_index("metric")
        promising = bool(q.loc["bal", "diff"] >= 0.03 and q.loc["f1", "diff"] >= 0 and q.loc["inter", "diff"] >= -0.03)
        reliable = bool(q.loc["bal", "ci_low"] > 0 and q.loc["bal", "seeds_A_better"] >= 4 and q.loc["bal", "holm_p"] < 0.05)
        decisions[a] = {"balanced_accuracy_gain": round(100 * q.loc["bal", "diff"], 2), "macro_f1_change": round(100 * q.loc["f1", "diff"], 2),
                        "intermediate_recall_change": round(100 * q.loc["inter", "diff"], 2), "promising": promising, "reliable": reliable}
    stop = not any(v["promising"] for v in decisions.values())

    # ------------------------------------------------------------- figure
    draw_figure(desc, os.path.join(EXP, "analysis", "fig_models.png"))

    # ------------------------------------------------------------- summary
    L = ["VERIFICATION"] + [f"  {x}" for x in lines] + ["", "METRICS (seed-pooled; 95% CI subject bootstrap)"]
    for m, r in desc.iterrows():
        L.append(f"  [{r.group}] {m}: balanced {100 * r.balanced_accuracy:.1f} [{100 * r.bal_ci_low:.1f}, {100 * r.bal_ci_high:.1f}] "
                 f"(fold {100 * r.bal_fold_mean:.1f} +/- {100 * r.bal_fold_sd:.1f}); macro-F1 {100 * r.f1_mean:.1f} [{100 * r.f1_ci_low:.1f}, "
                 f"{100 * r.f1_ci_high:.1f}]; accuracy {100 * r.acc_mean:.1f} [{100 * r.acc_ci_low:.1f}, {100 * r.acc_ci_high:.1f}]; "
                 f"intermediate recall {100 * r.inter_mean:.1f}; recall " + " ".join(f"{g} {100 * r['recall_' + g]:.0f}" for g in STAGES))
    L.append("\nPAIRED (A - B)")
    for r in cmp.itertuples():
        L.append(f"  [{r.role}] {r.A} vs {r.B} {r.metric}: {100 * r.diff:+.1f} [{100 * r.ci_low:+.1f}, {100 * r.ci_high:+.1f}], seeds A "
                 f"better {r.seeds_A_better}/5, splits {r.splits_better}/{r.splits_worse}/{r.splits_tied}, corrected p {r.corrected_t_p:.3f}"
                 + (f", Holm {r.holm_p:.3f}" if not np.isnan(r.holm_p) else ""))
    L.append(f"\nCANDIDATE DECISIONS (promising = +3.0 balanced, macro-F1 not lower, intermediate recall not >3 lower): {json.dumps(decisions)}")
    L.append(f"STOPPING RULE: {'no candidate is promising -> stop; no further experiment, no confirmation' if stop else 'a candidate is promising'}")
    open(os.path.join(EXP, "analysis", "summary.txt"), "w", encoding="utf-8").write("\n".join(L) + "\n")
    json.dump({"created": time.strftime("%Y-%m-%d %H:%M:%S"), "decisions": decisions, "stop": stop, "sources": SOURCE},
              open(os.path.join(EXP, "analysis", "manifest.json"), "w"), indent=2)
    print("\n".join(L))


if __name__ == "__main__":
    main()
