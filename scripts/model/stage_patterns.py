"""
Stage-pattern analysis for the transformer trained on SVM-RFE features.

Reads ONLY the outputs saved by train_transformer_svmrfe.py (no retraining)
and answers: does the transformer see the Alzheimer's progression pattern
across CN -> SMC -> EMCI -> MCI -> LMCI -> AD?

  0. QC          every metric recomputes from the saved predictions
  1. outcomes    balanced accuracy, accuracy, macro-F1, MAE, QWK, within-one-
                 stage accuracy, train/test gap; paired comparison vs references
  2. ordering    expected stage E = sum_k p_k k, and the first principal axis
                 of the learned brain embedding, against the true stage
                 (Spearman rho, permutation p)
  3. regions     stage x region occlusion importance and attention pooling
  4. branches    global / local / network attention weight by stage
  5. attention   stage-averaged network attention between the top regions
  6. data check  raw biomarker trajectories of the stable SVM-RFE features,
                 and how many of the 984 features trend with stage at all
  7. uncertainty MC-dropout predictive entropy by stage
  8. stability   how often each feature was selected across the 25 folds

Gate: the region/attention patterns are called LEARNED only if the primary
model beats the shuffled-label control (higher in >= 4 of 5 seeds, mean
paired difference > 0) AND the expected-stage ordering is significant
(permutation p < 0.05). Otherwise every pattern figure is labelled
NOT RELIABLE.

Run:  python scripts/model/stage_patterns.py
"""
import argparse
import itertools
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import feature_selection_cv as H  # noqa: E402
import feature_selection_methods as M  # noqa: E402

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm  # noqa: E402
from scipy import stats  # noqa: E402

from train_transformer import evaluate  # noqa: E402

ROOT = H.ROOT
SRC_DIR = os.path.join(ROOT, "derivatives", "model_training", "transformer_svmrfe")
STAGES = H.STAGES
NS = H.N_STAGES
CHANCE = H.CHANCE
PRIMARY = "transformer_svmrfe50"
REFS = ["logreg_svmrfe50", "transformer_all246"]
SHUF = "transformer_svmrfe50_SHUFFLED"
CANDS = [PRIMARY] + REFS + [SHUF]
LABEL = {PRIMARY: "Transformer, SVM-RFE 50 (primary)",
         "logreg_svmrfe50": "Logistic regression, SVM-RFE 50",
         "transformer_all246": "Transformer, all 246 regions",
         SHUF: "Transformer, shuffled labels (control)"}
KEYS = ("balanced_accuracy", "accuracy", "macro_f1", "mae_stages", "qwk")
N_PERM = 1000
RNG = np.random.default_rng(20261020)

# --- palette (reference instance of the dataviz skill), light surface
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]           # first three slots: all-pairs safe
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef",
                                                     "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("div_blue_red", ["#104281", "#3987e5", "#f0efec",
                                                         "#e66767", "#a32a2a"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 9,
    "text.color": INK, "axes.labelcolor": INK2, "axes.edgecolor": AXIS,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "axes.grid": False,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "legend.frameon": False,
})

_log = []


def log(m=""):
    print(m, flush=True)
    _log.append(str(m))


def load(design, cand, src):
    p = os.path.join(src, f"{design}_{cand}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))["records"]


def within_one(y, p):
    return float((np.abs(np.asarray(y) - np.asarray(p)) <= 1).mean())


def spear(x, y):
    r = stats.spearmanr(x, y)
    return float(r.statistic), float(r.pvalue)


def spearman_perm(x, y, n=N_PERM):
    rho = spear(x, y)[0]
    null = np.array([spear(x, RNG.permutation(y))[0] for _ in range(n)])
    return float(rho), float((1 + (null >= rho).sum()) / (n + 1))


def seed_means(recs, key):
    return np.array([np.mean([r[key] for r in recs if r["seed"] == s]) for s in H.SEEDS])


def per_subject(recs, field, n_sub=116):
    """Average a per-test-subject scalar over the 5 seeds -> [116]."""
    acc, cnt = np.zeros(n_sub), np.zeros(n_sub)
    for r in recs:
        for j, i in enumerate(r["test_index"]):
            acc[i] += r[field][j]
            cnt[i] += 1
    return acc / np.maximum(cnt, 1)


def expected_stage(r):
    return (np.asarray(r["probs"]) * np.arange(NS)).sum(1)


def bh_fdr(p):
    p = np.asarray(p)
    o = np.argsort(p)
    q = p[o] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[o] = np.minimum(q, 1)
    return out


def savefig(fig, path, unreliable=False):
    if unreliable:
        fig.text(0.995, 0.005, "NOT RELIABLE: gate failed (see stage_patterns.log)",
                 ha="right", va="bottom", color="#d03b3b", fontsize=8, fontweight="bold")
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    log(f"    figure {os.path.relpath(path, ROOT)}")


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=SRC_DIR)
    ap.add_argument("--out", default=SRC_DIR)
    args = ap.parse_args()
    figdir = os.path.join(args.out, "figures")
    for p in (os.path.join(args.out, "stage_patterns.json"), figdir):
        if os.path.exists(p):
            raise SystemExit(f"COLLISION: {p} exists; not overwritten")
    os.makedirs(figdir)

    cv = {c: load("cv116", c, args.src) for c in CANDS}
    sp = {c: load("split_80_20", c, args.src) for c in CANDS if c != SHUF}
    if cv[PRIMARY] is None:
        raise SystemExit("primary CV records missing")
    missing = [f"cv116_{c}" for c, v in cv.items() if v is None] + \
              [f"split_80_20_{c}" for c, v in sp.items() if v is None]
    roi = M.roi_names()
    fnames = M.feature_names("all4")
    y_all = np.load(os.path.join(H.SRC, "y.npy")).astype(int)
    out = {"missing_inputs": missing}

    log("=" * 100)
    log("STAGE PATTERNS -- transformer on SVM-RFE (all4 984 -> 50) features, 116-subject cohort")
    log("=" * 100)
    if missing:
        log(f"  inputs not found (left out): {missing}")

    # ---------------------------------------------------------------- 0. QC
    log("\n[0] QC: metrics recomputed from saved predictions")
    checks = bad = 0
    for design, group in (("cv116", cv), ("split_80_20", sp)):
        for c, recs in group.items():
            for r in recs or []:
                m = evaluate(np.asarray(r["probs"]), np.asarray(r["true"]),
                             pred=np.asarray(r["pred"]))
                for k in KEYS:
                    checks += 1
                    bad += abs(m[k] - r[k]) > 1e-9
                if r["true"] != [int(y_all[i]) for i in r["test_index"]]:
                    bad += 1
            if design == "cv116" and recs:
                for s in H.SEEDS:
                    idx = sorted(i for r in recs if r["seed"] == s for i in r["test_index"])
                    checks += 1
                    bad += idx != list(range(116))
    log(f"  {checks} checks, {bad} mismatches" + ("" if bad == 0 else "   <-- FAIL"))
    if bad:
        raise SystemExit("saved predictions do not reproduce the recorded metrics")
    fps = {c: sorted({(r["seed"], r["fold_fingerprint"]) for r in v}) for c, v in cv.items() if v}
    same_folds = len({json.dumps(v) for v in fps.values()}) == 1
    log(f"  all candidates used identical CV folds: {same_folds}")
    if not same_folds:
        raise SystemExit("candidates used different folds")
    if cv.get("logreg_svmrfe50"):
        same_sel = all(a["support"] == b["support"] for a, b in zip(cv[PRIMARY], cv["logreg_svmrfe50"]))
        log(f"  transformer and logistic regression used the same 50 features in every fold: {same_sel}")
    out["qc"] = {"checks": checks, "mismatches": bad, "identical_folds": same_folds}

    # -------------------------------------------------------- 1. outcomes
    log("\n[1] STAGE PREDICTION -- 5x5 CV on 116 (seed-level mean +/- sd) and the 80/20 split (23 test)")
    log(f"  chance {CHANCE:.3f}")
    log(f"  {'candidate':34s} {'bal.acc':>13s} {'acc':>6s} {'macroF1':>7s} {'MAE':>5s} "
        f"{'QWK':>6s} {'within1':>7s} {'train':>6s} | {'80/20 bal.acc':>13s} {'within1':>7s}")
    out["outcomes"] = {}
    for c in CANDS:
        if not cv[c]:
            continue
        recs = cv[c]
        s = H.summarise(recs)
        w1 = np.mean([within_one(r["true"], r["pred"]) for r in recs])
        tr = np.mean([r["train_balanced_accuracy"] for r in recs])
        row = {k: s[k] for k in KEYS}
        row.update({"within_one": float(w1), "train_balanced_accuracy": float(tr),
                    "classes_min": s["classes_min"], "classes_max": s["classes_max"]})
        splitcol = ""
        if sp.get(c):
            r = sp[c][0]
            n_ok = int(round(r["accuracy"] * r["n_test"]))
            ci = stats.binomtest(n_ok, r["n_test"], CHANCE)                       # two-sided CI
            pv = stats.binomtest(n_ok, r["n_test"], CHANCE, alternative="greater").pvalue
            row["split"] = {**{k: r[k] for k in KEYS}, "within_one": within_one(r["true"], r["pred"]),
                            "train_balanced_accuracy": r["train_balanced_accuracy"],
                            "n_test": r["n_test"], "correct": int(round(r["accuracy"] * r["n_test"])),
                            "accuracy_95ci": list(ci.proportion_ci(0.95)), "p_vs_chance": pv,
                            "confusion_matrix": r["confusion_matrix"]}
            splitcol = (f"| {r['balanced_accuracy']:13.3f} "
                        f"{row['split']['within_one']:7.3f}")
        out["outcomes"][c] = row
        b = s["balanced_accuracy"]
        log(f"  {LABEL[c]:34s} {b[0]:.3f}+/-{b[1]:.3f} {s['accuracy'][0]:6.3f} {s['macro_f1'][0]:7.3f} "
            f"{s['mae_stages'][0]:5.2f} {s['qwk'][0]:+6.3f} {w1:7.3f} {tr:6.3f} {splitcol}")
    if sp.get(PRIMARY):
        o = out["outcomes"][PRIMARY]["split"]
        log(f"  80/20 primary: {o['correct']}/{o['n_test']} correct, accuracy 95% CI "
            f"{o['accuracy_95ci'][0]:.2f}-{o['accuracy_95ci'][1]:.2f}, p vs chance {o['p_vs_chance']:.3f}; "
            f"one subject moves balanced accuracy by about {1 / 4 / NS:.3f}")
    # within-one by chance for these class counts (uniform random predictor)
    cnt = np.bincount(y_all, minlength=NS) / len(y_all)
    w1_chance = float(sum(cnt[t] * min(3 if 0 < t < NS - 1 else 2, 3) / NS for t in range(NS)))
    log(f"  within-one-stage accuracy of random guessing: {w1_chance:.3f}")
    out["within_one_chance"] = w1_chance

    log("\n  paired comparison on identical folds (primary minus reference), rule: >= 4/5 seeds and mean > 0")
    out["paired"] = {}
    pm = seed_means(cv[PRIMARY], "balanced_accuracy")
    for c in REFS + [SHUF]:
        if not cv[c]:
            continue
        d = pm - seed_means(cv[c], "balanced_accuracy")
        wins = int((d > 0).sum())
        verdict = "PRIMARY BETTER" if wins >= 4 and d.mean() > 0 else (
            "REFERENCE BETTER" if (d < 0).sum() >= 4 and d.mean() < 0 else "TIED")
        out["paired"][c] = {"per_seed_diff": d.tolist(), "mean_diff": float(d.mean()),
                            "seeds_favouring_primary": wins, "verdict": verdict}
        log(f"    vs {LABEL[c]:40s} mean {d.mean():+.3f}  seeds {wins}/5  "
            f"per seed {np.round(d, 3).tolist()}  -> {verdict}")
    if cv[SHUF]:
        sb = H.summarise(cv[SHUF])["balanced_accuracy"][0]
        log(f"  shuffled control {sb:.3f}: "
            + ("OK (near chance)" if abs(sb - CHANCE) < 0.06 else "CHECK: away from chance"))

    # -------------------------------------------------------- 2. ordering
    log("\n[2] DOES THE MODEL ORDER THE STAGES? (Spearman rho vs true stage, permutation p, 1000 shuffles)")
    out["ordering"] = {}
    for c in [PRIMARY, "logreg_svmrfe50", "transformer_all246", SHUF]:
        if not cv[c]:
            continue
        for r in cv[c]:
            r["_E"] = expected_stage(r).tolist()
        E = per_subject(cv[c], "_E")
        rho, p = spearman_perm(E, y_all)
        res = {"expected_stage_rho": rho, "expected_stage_p": p,
               "stage_means": [float(E[y_all == g].mean()) for g in range(NS)]}
        line = f"  {LABEL[c]:40s} E[stage] rho {rho:+.3f} p {p:.3f}"
        if c == PRIMARY:
            A = per_subject(cv[c], "emb_axis")
            rho_a, p_a = spearman_perm(A, y_all)
            res.update({"embedding_axis_rho": rho_a, "embedding_axis_p": p_a,
                        "embedding_axis_stage_means": [float(A[y_all == g].mean()) for g in range(NS)]})
            line += f" | embedding axis rho {rho_a:+.3f} p {p_a:.3f}"
            out["_E_primary"], out["_A_primary"] = E, A
        out["ordering"][c] = res
        log(line)
        log("    mean E[stage] by true stage: "
            + "  ".join(f"{STAGES[g]} {res['stage_means'][g]:.2f}" for g in range(NS)))
    if sp.get(PRIMARY):
        r = sp[PRIMARY][0]
        rho, p = spearman_perm(expected_stage(r), np.asarray(r["true"]))
        out["ordering"]["split_primary"] = {"expected_stage_rho": rho, "expected_stage_p": p}
        log(f"  80/20 primary (23 test subjects): E[stage] rho {rho:+.3f} p {p:.3f}")

    # ------------------------------------------------------------ gate
    beats_shuffled = bool(cv[SHUF]) and out["paired"].get(SHUF, {}).get("verdict") == "PRIMARY BETTER"
    ordered = out["ordering"][PRIMARY]["expected_stage_p"] < 0.05
    gate = beats_shuffled and ordered
    out["gate"] = {"beats_shuffled": beats_shuffled, "ordering_significant": ordered, "passed": gate}
    log(f"\n  GATE: beats shuffled control {beats_shuffled}; ordering significant {ordered} "
        f"-> patterns {'LEARNED' if gate else 'NOT RELIABLE'}")

    # ------------------------------------------- 3. regions by stage
    log("\n[3] WHICH REGIONS DRIVE EACH STAGE (primary, out-of-fold, pooled over 5 seeds)")
    recs = cv[PRIMARY]
    n_folds = len(recs)
    roi_freq = Counter(i for r in recs for i in r["rois"])
    top = [i for i, _ in roi_freq.most_common(20)]
    occ = np.full((NS, 246), np.nan)
    pw = np.full((NS, 246), np.nan)
    occ_s, pw_s, cnt_s = np.zeros((NS, 246)), np.zeros((NS, 246)), np.zeros((NS, 246))
    for r in recs:
        K = len(r["rois"])
        for j, g in enumerate(r["true"]):
            occ_s[g, r["rois"]] += r["occlusion"][j]
            pw_s[g, r["rois"]] += np.asarray(r["pool_w"][j]) * K      # 1.0 = uniform share
            cnt_s[g, r["rois"]] += 1
    ok = cnt_s > 0
    occ[ok], pw[ok] = occ_s[ok] / cnt_s[ok], pw_s[ok] / cnt_s[ok]
    out["regions"] = {"selection_frequency": {roi[i]: c / n_folds for i, c in roi_freq.most_common(40)},
                      "top20": [roi[i] for i in top],
                      "occlusion_stage_x_top20": occ[:, top].tolist(),
                      "pooling_stage_x_top20": pw[:, top].tolist()}
    log(f"  20 regions most often kept as tokens (folds out of {n_folds}): "
        + ", ".join(f"{roi[i]} {roi_freq[i]}" for i in top))
    eligible = [i for i in roi_freq if roi_freq[i] >= 0.4 * n_folds]
    log(f"  top 3 regions per stage by occlusion importance (regions kept in >= 40% of folds, n={len(eligible)}):")
    out["regions"]["top_by_stage"] = {}
    for g in range(NS):
        best = sorted(eligible, key=lambda i: -occ[g, i])[:3]
        out["regions"]["top_by_stage"][STAGES[g]] = [(roi[i], float(occ[g, i])) for i in best]
        log(f"    {STAGES[g]:5s} " + ", ".join(f"{roi[i]} {occ[g, i]:+.3f}" for i in best))

    def heat(mat, title, cmap, norm, fname, cbar_label, fmt):
        fig, ax = plt.subplots(figsize=(11, 3.4))
        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
        ax.set_yticks(range(NS), STAGES)
        ax.set_xticks(range(len(top)), [roi[i] for i in top], rotation=60, ha="right", fontsize=8)
        ax.tick_params(length=0)
        for s_ in ax.spines.values():
            s_.set_visible(False)
        for a, b in itertools.product(range(NS), range(len(top))):
            v = mat[a, b]
            if np.isfinite(v):
                dark = abs(norm(v) - 0.5) > 0.3 if isinstance(norm, TwoSlopeNorm) else norm(v) > 0.6
                ax.text(b, a, fmt.format(v), ha="center", va="center", fontsize=6.5,
                        color="#ffffff" if dark else INK2)
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
        cb.set_label(cbar_label, color=INK2)
        cb.outline.set_visible(False)
        ax.set_title(title, loc="left")
        ax.set_xlabel("Brainnetome region (20 most often selected, most frequent first)")
        savefig(fig, os.path.join(figdir, fname), unreliable=not gate)

    occ_k = occ[:, top] * 1000                                     # thousandths of probability
    lim = np.nanmax(np.abs(occ_k)) or 1.0
    heat(occ_k, "Occlusion importance: drop in P(true stage) when the region is removed",
         DIV, TwoSlopeNorm(0, -lim, lim), "fig3_region_stage_occlusion.png",
         "probability drop (x 0.001)", "{:+.1f}")
    pv = pw[:, top]
    heat(pv, "Attention pooling weight per region by stage (1.0 = equal share of all tokens)",
         SEQ, matplotlib.colors.Normalize(np.nanmin(pv), np.nanmax(pv)),
         "fig4_region_stage_pooling.png", "relative weight", "{:.2f}")

    # ---------------------------------------------- 4. branch weights
    log("\n[4] WHICH ATTENTION BRANCH THE MODEL USES, BY STAGE (mean over test subjects)")
    bw = np.zeros((116, 3))
    cnt = np.zeros(116)
    for r in recs:
        for j, i in enumerate(r["test_index"]):
            bw[i] += r["branch_w"][j]
            cnt[i] += 1
    bw /= cnt[:, None]
    bm = np.array([bw[y_all == g].mean(0) for g in range(NS)])
    be = np.array([bw[y_all == g].std(0, ddof=1) / np.sqrt((y_all == g).sum()) for g in range(NS)])
    trends = {n: spear(bw[:, k], y_all) for k, n in enumerate(("global", "local", "network"))}
    out["branches"] = {"stage_means": bm.tolist(),
                       "trend": {n: {"rho": float(t[0]), "p": float(t[1])} for n, t in trends.items()}}
    for g in range(NS):
        log(f"    {STAGES[g]:5s} global {bm[g, 0]:.3f}  local {bm[g, 1]:.3f}  network {bm[g, 2]:.3f}")
    log("    trend with stage: " + "  ".join(f"{n} rho {t[0]:+.3f} p {t[1]:.3f}" for n, t in trends.items()))
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(NS)
    for k, n in enumerate(("Global", "Local (atlas neighbours)", "Network (FC)")):
        ax.errorbar(x, bm[:, k], yerr=be[:, k], color=SERIES[k], marker="o", ms=6, capsize=3,
                    label=n, lw=2)
        ax.annotate(n.split(" ")[0], (x[-1], bm[-1, k]), xytext=(8, 0), textcoords="offset points",
                    va="center", color=INK2, fontsize=8)
    ax.set_xticks(x, STAGES)
    ax.set_ylabel("mean branch weight (sums to 1)")
    ax.yaxis.grid(True)
    ax.set_title("Share of each attention branch by stage (mean +/- SE)", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.14), ncol=3, fontsize=8)
    savefig(fig, os.path.join(figdir, "fig5_branch_weights_by_stage.png"), unreliable=not gate)

    # ---------------------------------------------- 5. attention maps
    log("\n[5] NETWORK ATTENTION BETWEEN THE 10 MOST-SELECTED REGIONS, BY STAGE")
    pa = os.path.join(args.src, f"cv116_{PRIMARY}_attention.npz")
    if os.path.exists(pa):
        A = np.load(pa)
        t10 = top[:10]
        ix = np.ix_(t10, t10)
        fig, axs = plt.subplots(1, NS, figsize=(15, 3.0))
        mats = []
        for g in range(NS):
            c_ = A["count"][g][ix]
            mats.append(np.where(c_ > 0, A["network"][g][ix] / np.maximum(c_, 1), np.nan))
        # attention is close to uniform (1/K), so centre the scale on the overall mean
        mid = float(np.nanmean(mats))
        half = float(np.nanmax(np.abs(np.array(mats) - mid))) or 1e-6
        norm = TwoSlopeNorm(mid, mid - half, mid + half)
        for g, ax in enumerate(axs):
            im = ax.imshow(mats[g], cmap=DIV, norm=norm)
            ax.set_title(STAGES[g], loc="center")
            ax.set_xticks(range(10), [roi[i] for i in t10], rotation=90, fontsize=6)
            ax.set_yticks(range(10), [roi[i] for i in t10] if g == 0 else [], fontsize=6)
            ax.tick_params(length=0)
            for s_ in ax.spines.values():
                s_.set_visible(False)
        cb = fig.colorbar(im, ax=axs, fraction=0.012, pad=0.01)
        cb.set_label(f"attention row -> column (centre = mean {mid:.4f})", color=INK2)
        cb.outline.set_visible(False)
        fig.suptitle("Network-branch attention between the 10 most-selected regions, averaged by stage",
                     x=0.01, ha="left", fontweight="bold", fontsize=10)
        savefig(fig, os.path.join(figdir, "fig8_network_attention_by_stage.png"), unreliable=not gate)
        diff = np.nanmean(np.abs(np.array(mats) - np.nanmean(mats, axis=0)), axis=(1, 2))
        log("    mean absolute difference from the all-stage average: "
            + "  ".join(f"{STAGES[g]} {diff[g]:.4f}" for g in range(NS)))
        out["attention"] = {"regions": [roi[i] for i in t10], "network_by_stage": [m.tolist() for m in mats]}

    # ---------------------------------------- 6. raw data trajectories
    log("\n[6] DATA CHECK: do the selected features change with stage in the raw data? (model-independent)")
    reg = np.load(os.path.join(H.SRC, "X_regional.npy")).astype(np.float64)
    fcu = np.load(os.path.join(H.SRC, "X_fc_roi_unsigned.npy")).astype(np.float64)
    flat = np.hstack([reg.reshape(116, -1), fcu])
    z = (flat - flat.mean(0)) / flat.std(0)                       # display only
    rho_all = np.array([spear(flat[:, j], y_all) for j in range(flat.shape[1])])
    q_all = bh_fdr(rho_all[:, 1])
    log(f"  all 984 features vs stage: {(rho_all[:, 1] < 0.05).sum()} with p < 0.05 "
        f"(about {0.05 * 984:.0f} expected by chance), {(q_all < 0.05).sum()} with FDR q < 0.05")
    feat_freq = Counter(i for r in recs for i in r["support"])
    stable = [i for i, c in feat_freq.most_common() if c >= 0.5 * n_folds]
    rule = "selected in >= 50% of folds"
    if len(stable) < 4:
        stable = [i for i, _ in feat_freq.most_common(8)]
        rule = "fewer than 4 features reached 50% of folds, so the 8 most frequent are shown"
    elif len(stable) > 12:
        stable = stable[:12]
        rule += " (12 most frequent shown)"
    p_sel = [rho_all[i, 1] for i in stable]
    q_sel = bh_fdr(p_sel)
    log(f"  stable features ({rule}): {len(stable)}")
    out["trajectories"] = {"n_p05_of_984": int((rho_all[:, 1] < 0.05).sum()),
                           "n_fdr05_of_984": int((q_all < 0.05).sum()), "rule": rule, "features": []}
    for k, i in enumerate(stable):
        means = [float(z[y_all == g, i].mean()) for g in range(NS)]
        out["trajectories"]["features"].append(
            {"feature": fnames[i], "selected_folds": feat_freq[i], "rho": float(rho_all[i, 0]),
             "p": float(rho_all[i, 1]), "q": float(q_sel[k]), "stage_means_z": means})
        log(f"    {fnames[i]:24s} folds {feat_freq[i]:2d}/{n_folds}  rho {rho_all[i, 0]:+.3f}  "
            f"p {rho_all[i, 1]:.3f}  q {q_sel[k]:.3f}  z by stage "
            + " ".join(f"{v:+.2f}" for v in means))
    ncol = 4
    nrow = int(np.ceil(len(stable) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(12, 2.5 * nrow), sharey=True, squeeze=False)
    for k, ax in enumerate(axs.flat):
        if k >= len(stable):
            ax.set_visible(False)
            continue
        i = stable[k]
        m = [z[y_all == g, i].mean() for g in range(NS)]
        e = [z[y_all == g, i].std(ddof=1) / np.sqrt((y_all == g).sum()) for g in range(NS)]
        ax.axhline(0, color=AXIS, lw=0.8)
        ax.errorbar(range(NS), m, yerr=e, color=SERIES[0], marker="o", ms=5, capsize=3, lw=2)
        ax.set_xticks(range(NS), STAGES, fontsize=7)
        ax.set_title(f"{fnames[i]}\nrho {rho_all[i, 0]:+.2f}, q {q_sel[k]:.2f}, "
                     f"{feat_freq[i]}/{n_folds} folds", fontsize=8, loc="left")
        ax.yaxis.grid(True)
    axs[0, 0].set_ylabel("z-score (mean +/- SE)")
    fig.suptitle("Raw biomarker value by stage for the features SVM-RFE kept most often",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    savefig(fig, os.path.join(figdir, "fig6_biomarker_trajectories.png"))

    # -------------------------------------------------- 7. uncertainty
    log("\n[7] UNCERTAINTY: MC-dropout predictive entropy by stage (max possible ln 6 = 1.792)")
    ent = per_subject(recs, "entropy")
    em = [float(ent[y_all == g].mean()) for g in range(NS)]
    ee = [float(ent[y_all == g].std(ddof=1) / np.sqrt((y_all == g).sum())) for g in range(NS)]
    out["uncertainty"] = {"entropy_stage_means": em}
    log("    " + "  ".join(f"{STAGES[g]} {em[g]:.3f}" for g in range(NS)))
    correct = np.zeros(116)
    for r in recs:
        for j, i in enumerate(r["test_index"]):
            correct[i] += r["pred"][j] == r["true"][j]
    correct /= len(H.SEEDS)
    rho_ec = spear(ent, correct)
    log(f"    entropy vs fraction of seeds predicted correctly: rho {rho_ec[0]:+.3f} p {rho_ec[1]:.3f}")
    out["uncertainty"]["entropy_vs_correct"] = {"rho": float(rho_ec[0]), "p": float(rho_ec[1])}

    # -------------------------------------------------- 8. stability
    log("\n[8] FEATURE STABILITY across the 25 CV folds")
    sets = [set(r["support"]) for r in recs]
    jac = np.mean([len(a & b) / len(a | b) for a, b in itertools.combinations(sets, 2)])
    share = Counter(fnames[i].split("|")[1] for r in recs for i in r["support"])
    tot = sum(share.values())
    out["stability"] = {"mean_pairwise_jaccard": float(jac),
                        "biomarker_share": {k: v / tot for k, v in share.items()},
                        "top_features": [(fnames[i], c) for i, c in feat_freq.most_common(25)]}
    log(f"    mean pairwise Jaccard overlap {jac:.3f} (1 = same 50 features every fold)")
    log("    share of picks: " + "  ".join(f"{k} {v / tot:.2f}" for k, v in share.most_common()))
    log("    most frequent: " + ", ".join(f"{fnames[i]} {c}" for i, c in feat_freq.most_common(12)))
    tf = feat_freq.most_common(25)[::-1]
    fig, ax = plt.subplots(figsize=(6.4, 6.2))
    ax.barh(range(len(tf)), [c for _, c in tf], color=SERIES[0], height=0.7)
    ax.set_yticks(range(len(tf)), [fnames[i] for i, _ in tf], fontsize=7.5)
    ax.axvline(0.5 * n_folds, color=MUTED, lw=1, ls="--")
    ax.text(0.5 * n_folds, len(tf) - 0.4, " 50% of folds", color=MUTED, fontsize=7.5, va="bottom")
    ax.set_xlabel(f"folds in which SVM-RFE kept the feature (of {n_folds})")
    ax.xaxis.grid(True)
    ax.set_title(f"25 most frequently kept features (mean overlap between folds {jac:.2f})", loc="left")
    savefig(fig, os.path.join(figdir, "fig9_feature_stability.png"))

    # ------------------------------------------------- remaining figures
    # fig0: model comparison
    fig, ax = plt.subplots(figsize=(7.4, 3.2))
    order = [c for c in CANDS if cv[c]]
    for k, c in enumerate(order):
        sm = seed_means(cv[c], "balanced_accuracy")
        ax.scatter(sm, [k] * len(sm), s=26, color=SERIES[0], alpha=0.45, zorder=3,
                   label="CV seed mean" if k == 0 else None)
        ax.scatter([sm.mean()], [k], s=90, color=SERIES[0], zorder=4, marker="D",
                   label="CV mean of 5 seeds" if k == 0 else None)
        if sp.get(c):
            ax.scatter([sp[c][0]["balanced_accuracy"]], [k], s=80, facecolor=SURFACE,
                       edgecolor=SERIES[1], lw=2, zorder=5, marker="o",
                       label="80/20 split (23 test)" if k == 0 else None)
    ax.axvline(CHANCE, color=MUTED, ls="--", lw=1)
    ax.set_yticks(range(len(order)), [LABEL[c] for c in order])
    ax.set_ylim(len(order) - 0.5, -0.7)                             # top row first, room for label
    ax.text(CHANCE, -0.6, " chance 0.167", color=MUTED, fontsize=8, va="center")
    ax.set_xlabel("balanced accuracy (6 stages)")
    ax.xaxis.grid(True)
    ax.set_title("Stage prediction on the 116-subject cohort", loc="left")
    ax.legend(loc="upper left", bbox_to_anchor=(0, -0.2), ncol=3, fontsize=8)
    savefig(fig, os.path.join(figdir, "fig0_model_comparison.png"))

    # fig1: confusion matrix, primary, pooled CV
    C = np.sum([np.array(r["confusion_matrix"]) for r in recs], axis=0)
    Cn = C / C.sum(1, keepdims=True)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(Cn, cmap=SEQ, vmin=0, vmax=max(0.5, Cn.max()))
    for a, b in itertools.product(range(NS), range(NS)):
        ax.text(b, a, f"{Cn[a, b]:.2f}\n({C[a, b]})", ha="center", va="center", fontsize=7,
                color="#ffffff" if Cn[a, b] > 0.3 else INK2)
    ax.set_xticks(range(NS), STAGES)
    ax.set_yticks(range(NS), STAGES)
    ax.set_xlabel("predicted stage")
    ax.set_ylabel("true stage")
    ax.tick_params(length=0)
    for s_ in ax.spines.values():
        s_.set_visible(False)
    o = out["outcomes"][PRIMARY]
    ax.set_title(f"Primary model, 5x5 CV pooled (row share)\nbal.acc {o['balanced_accuracy'][0]:.3f}, "
                 f"within one stage {o['within_one']:.2f} (random {w1_chance:.2f})", loc="left", fontsize=9)
    savefig(fig, os.path.join(figdir, "fig1_confusion_primary_cv.png"))

    # fig2: ordering
    E, A_ = out.pop("_E_primary"), out.pop("_A_primary")
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, vals, ttl, res_k in ((axs[0], E, "Expected stage  E = sum p(k) k", "expected_stage"),
                                 (axs[1], A_, "First axis of the learned brain embedding",
                                  "embedding_axis")):
        data = [vals[y_all == g] for g in range(NS)]
        ax.boxplot(data, positions=range(NS), widths=0.5, patch_artist=True, showfliers=False,
                        medianprops={"color": INK, "lw": 1.5},
                        boxprops={"facecolor": "#cde2fb", "edgecolor": SERIES[0]},
                        whiskerprops={"color": SERIES[0]}, capprops={"color": SERIES[0]})
        for g in range(NS):
            jit = RNG.uniform(-0.12, 0.12, len(data[g]))
            ax.scatter(g + jit, data[g], s=10, color=SERIES[0], alpha=0.55, zorder=3, lw=0)
        ax.plot(range(NS), [d.mean() for d in data], color=SERIES[1], marker="D", ms=6, lw=2,
                zorder=4, label="stage mean")
        r_ = out["ordering"][PRIMARY]
        ax.set_title(f"{ttl}\nSpearman rho {r_[res_k + '_rho']:+.2f}, permutation p {r_[res_k + '_p']:.3f}",
                     loc="left", fontsize=9)
        ax.set_xticks(range(NS), STAGES)
        ax.yaxis.grid(True)
        ax.set_xlabel("true stage")
    axs[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Does the transformer order the stages CN -> AD? (out-of-fold, mean over 5 seeds)",
                 x=0.01, ha="left", fontweight="bold", fontsize=10)
    fig.tight_layout()
    savefig(fig, os.path.join(figdir, "fig2_stage_ordering.png"))

    # fig7: uncertainty
    fig, ax = plt.subplots(figsize=(5.6, 3.2))
    ax.bar(range(NS), em, yerr=ee, color=SERIES[0], width=0.6, capsize=3,
           error_kw={"ecolor": INK2, "lw": 1})
    ax.axhline(np.log(NS), color=MUTED, ls="--", lw=1)
    ax.text(NS - 0.5, np.log(NS), "maximum (ln 6)", color=MUTED, fontsize=8, ha="right", va="bottom")
    ax.set_xticks(range(NS), STAGES)
    ax.set_ylabel("predictive entropy (mean +/- SE)")
    ax.yaxis.grid(True)
    ax.set_ylim(0, np.log(NS) * 1.1)
    ax.set_title("MC-dropout uncertainty by true stage", loc="left")
    savefig(fig, os.path.join(figdir, "fig7_uncertainty_by_stage.png"))

    json.dump(out, open(os.path.join(args.out, "stage_patterns.json"), "w"), indent=1, default=float)
    open(os.path.join(args.out, "stage_patterns.log"), "w", encoding="utf-8").write("\n".join(_log) + "\n")
    log(f"\nwrote {os.path.relpath(os.path.join(args.out, 'stage_patterns.json'), ROOT)}")


if __name__ == "__main__":
    main()
