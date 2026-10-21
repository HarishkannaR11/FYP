"""
Reptile meta-learning vs the same transformer trained from scratch (and
TabPFN for context), on identical inputs, splits and folds. Reads only saved
records.

Run:  python scripts/model/meta_summary.py
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
META_DIR, TAB = os.path.join(TD, "meta_reptile"), os.path.join(TD, "tabpfn")
IMG, C23, A = (os.path.join(TD, d) for d in ("transformer_svmrfe", "transformer_comorbidity", "transformer_comorbidity8"))
DESIGNS = [("cv116", "5-fold x 5 CV (116)"), ("split70", "70/30 test (35)")]
GROUPS = [
    ("rs-fMRI only",
     {"meta": (META_DIR, "meta_transformer_svmrfe50"),
      "scratch": {"cv116": (IMG, "transformer_svmrfe50"), "split70": (C23, "transformer_svmrfe50")},
      "tabpfn": (TAB, "tabpfn_svmrfe50")}),
    ("rs-fMRI + 8 comorbidities",
     {"meta": (META_DIR, "meta_transformer_svmrfe50_comorb8"),
      "scratch": {"cv116": (A, "transformer_svmrfe50_clin"), "split70": (A, "transformer_svmrfe50_clin")},
      "tabpfn": (TAB, "tabpfn_svmrfe50_comorb8")}),
]
NAMES = {"scratch": "Transformer, from scratch", "meta": "Transformer, Reptile meta-learned start", "tabpfn": "TabPFN v2"}

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


def main():
    for p in (os.path.join(META_DIR, "meta_summary.json"), os.path.join(META_DIR, "figures")):
        if os.path.exists(p):
            raise SystemExit(f"COLLISION: {p} exists; not overwritten")
    os.makedirs(os.path.join(META_DIR, "figures"))
    y = np.load(os.path.join(H.SRC, "y.npy")).astype(int)
    out = {"designs": {}, "paired": {}, "qc": {}}
    log("=" * 100)
    log("REPTILE META-LEARNING (biomarker tasks) vs FROM-SCRATCH TRANSFORMER vs TabPFN -- 116 subjects, 6 stages")
    log("=" * 100)
    log("identical folds, SVM-RFE features and fine-tuning seeds; only the transformer's starting weights differ")
    checks = bad = 0
    for design, dlabel in DESIGNS:
        log(f"\n[{dlabel}]  accuracy / balanced accuracy (+/- sd of 5 seed means in CV), training balanced accuracy")
        out["designs"][design] = {}
        for label, src in GROUPS:
            row = {}
            for name in ("scratch", "meta", "tabpfn"):
                d, c = src[name][design] if name == "scratch" else src[name]
                recs = P.load(design, c, d)
                if not recs:
                    continue
                checks += 1
                bad += qc(recs, y)
                if design == "cv116":
                    s = H.summarise(recs)
                    tr = [r["train_balanced_accuracy"] for r in recs if r.get("train_balanced_accuracy") is not None]
                    row[name] = {"acc": s["accuracy"][0], "bal": s["balanced_accuracy"][0],
                                 "bal_sd": s["balanced_accuracy"][1], "train": float(np.mean(tr)) if tr else None,
                                 "seed_means": P.seed_means(recs, "balanced_accuracy").tolist(), "_recs": recs}
                else:
                    r = recs[0]
                    row[name] = {"acc": r["accuracy"], "bal": r["balanced_accuracy"], "bal_sd": None,
                                 "correct": int(round(r["accuracy"] * r["n_test"])),
                                 "train": r.get("train_balanced_accuracy"), "_recs": recs}
            if "meta" in row and "scratch" in row:
                a, b = row["meta"]["_recs"], row["scratch"]["_recs"]
                same = [(x["seed"], x["fold"], x["fold_fingerprint"]) for x in a] == \
                       [(x["seed"], x["fold"], x["fold_fingerprint"]) for x in b]
                feat = all(x.get("support") == z.get("support") for x, z in zip(a, b))
                out["qc"][f"{design}|{label}"] = {"same_folds": same, "same_features": feat}
                if not same:
                    raise SystemExit(f"{design} {label}: meta and scratch folds differ")
            for name, v in row.items():
                sd = f" +/- {v['bal_sd']:.3f}" if v["bal_sd"] is not None else ""
                ex = f"  ({v['correct']}/35)" if "correct" in v else ""
                tr = f"  train {v['train']:.3f}" if v["train"] is not None else ""
                log(f"  {label:28s} {NAMES[name]:42s} {v['acc']:.3f} / {v['bal']:.3f}{sd}{ex}{tr}")
            out["designs"][design][label] = {k: {kk: vv for kk, vv in v.items() if kk != "_recs"} for k, v in row.items()}
            if design == "cv116":
                for other in ("scratch", "tabpfn"):
                    if "meta" in row and other in row:
                        d_ = np.array(row["meta"]["seed_means"]) - np.array(row[other]["seed_means"])
                        wins = int((d_ > 0).sum())
                        verdict = ("META BETTER" if wins >= 4 and d_.mean() > 0 else
                                   f"{other.upper()} BETTER" if (d_ < 0).sum() >= 4 and d_.mean() < 0 else "TIED")
                        out["paired"][f"{label}|meta-{other}"] = {"mean_diff": float(d_.mean()), "seeds": wins,
                                                                 "per_seed": d_.tolist(), "verdict": verdict}
                if "meta" in row:
                    tl = {}
                    for t in ("mALFF", "mReHo", "DC_z", "FC"):
                        v = [r["meta_task_loss_last10"][t] for r in row["meta"]["_recs"]
                             if r["meta_task_loss_last10"].get(t) is not None]
                        tl[t] = float(np.mean(v)) if v else None
                    out["designs"][design][label]["meta_task_loss"] = tl
                    log(f"  {'':28s} Reptile inner loss per biomarker task (last 10 visits; uniform guess 1.792): "
                        + ", ".join(f"{k} {v:.3f}" for k, v in tl.items() if v is not None))
        if design == "cv116":
            sh = P.load("cv116", "meta_transformer_svmrfe50_comorb8_SHUFFLED", META_DIR)
            if sh:
                checks += 1
                bad += qc(sh, y)
                b = H.summarise(sh)["balanced_accuracy"][0]
                out["shuffled_control"] = b
                log(f"  shuffled-label control (meta, rs-fMRI + 8 comorb.): {b:.3f} balanced  "
                    + ("OK (near chance)" if abs(b - P.CHANCE) < 0.06 else "CHECK"))
    log("\n[PAIRED, 5-fold x 5, balanced accuracy, identical folds; rule: >= 4/5 seeds and mean > 0]")
    for k, v in out["paired"].items():
        log(f"  {k:46s} {v['mean_diff']:+.3f}  seeds {v['seeds']}/5  per seed {np.round(v['per_seed'], 3).tolist()}  -> {v['verdict']}")
    log(f"\nQC: {checks} record sets, {bad} metric mismatches; meta and scratch used the same folds and features: "
        f"{all(v['same_folds'] and v['same_features'] for v in out['qc'].values()) if out['qc'] else 'n/a'}")
    if bad:
        raise SystemExit("QC failed")

    fig, ax = plt.subplots(figsize=(8.4, 3.0))
    # same colours as the TabPFN figure: TabPFN blue, from-scratch transformer orange; meta-learned green
    colour = {"scratch": P.SERIES[1], "meta": P.SERIES[2], "tabpfn": P.SERIES[0]}
    for j, name in enumerate(("scratch", "meta", "tabpfn")):
        for i, (label, _) in enumerate(GROUPS):
            v = out["designs"]["cv116"].get(label, {}).get(name)
            if v:
                ax.errorbar(v["bal"], i + (j - 1) * 0.22, xerr=v["bal_sd"], fmt="o", ms=7, capsize=3,
                            color=colour[name], label=NAMES[name] if i == 0 else None)
    ax.axvline(P.CHANCE, color=P.MUTED, ls="--", lw=1)
    ax.set_yticks(range(len(GROUPS)), [g[0] for g in GROUPS])
    ax.set_ylim(len(GROUPS) - 0.5, -0.8)
    ax.text(P.CHANCE, -0.65, " chance 0.167", color=P.MUTED, fontsize=8, va="center")
    ax.set_xlabel("balanced accuracy, 5-fold x 5 CV (mean +/- sd of 5 seed means)")
    ax.xaxis.grid(True)
    ax.set_title("Reptile meta-learning vs training from scratch, same inputs and folds", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.24), ncol=3, fontsize=8)
    fig.savefig(os.path.join(META_DIR, "figures", "fig_meta_comparison.png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    log("    figure derivatives/model_training/meta_reptile/figures/fig_meta_comparison.png")

    json.dump(out, open(os.path.join(META_DIR, "meta_summary.json"), "w"), indent=1, default=float)
    open(os.path.join(META_DIR, "meta_summary.log"), "w", encoding="utf-8").write("\n".join(_log) + "\n")
    log(f"\nwrote {os.path.relpath(os.path.join(META_DIR, 'meta_summary.json'), ROOT)}")


if __name__ == "__main__":
    main()
