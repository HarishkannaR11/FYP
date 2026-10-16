"""
4 biomarkers x 6 stages one-vs-rest summary, by multivariate permutation.

Each of the 24 cells is a genuine test computed from the subject-level data.
Nothing is derived from the 15 pairwise comparisons.

Statistic
---------
For a stage s and biomarker b, subjects are split into the stage group and
the pooled rest. Features are z-scored across subjects (one scaler, fitted
on the whole cohort -- this is a descriptive/inferential analysis, not a
predictive model, and the same scaling is applied under every permutation
so it cannot bias the null).

The statistic is a distance-based pseudo-F (the PERMANOVA statistic for a
two-group design on squared Euclidean distances):

    F = [ SS_between / (k-1) ] / [ SS_within / (n-k) ],   k = 2

SS_between is the dispersion of the two group centroids about the grand
centroid; SS_within is the dispersion of subjects about their own group
centroid. This is multivariate: ROIs are features of a subject, never
independent observations.

Null
----
Stage membership is permuted ACROSS SUBJECTS. Each subject contributes
exactly one acquisition, so permuting subject labels is exactly permuting
observations; no within-subject dependence exists to preserve.

    p = (#{F_perm >= F_obs} + 1) / (N_PERM + 1)

The +1 makes p exact-conservative and bounds it away from zero, so the
smallest attainable p with N_PERM=9999 is 1e-4.

FC
--
FC uses the 30,135 unique upper-triangle Fisher-z edges as the subject's
feature vector; the diagonal is excluded. The same pseudo-F is used. No
dimensionality reduction is applied -- a distance-based statistic is well
defined for p >> n, unlike Hotelling's T^2 which would require inverting a
singular covariance matrix. That difference in method is deliberate and is
the reason a distance-based statistic was chosen for all four biomarkers
rather than a parametric multivariate test.

Limitation
----------
One-vs-rest pools the other five stages and therefore ignores the ordinal
structure CN -> AD. It answers "is this stage separable from everything
else", not "do the stages progress". It is exploratory.
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "derivatives", "model_ready")
OUTROOT = os.path.join(ROOT, "derivatives", "INTRA_INTER_CLASS_HEATMAPS")

STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
FEATS = ["mALFF", "mReHo", "DC_z", "FC"]
N_PERM = 9999
SEED = 20261013
Q_FLOOR = 1e-4          # = smallest attainable p; -log10 -> 4.0
ALPHA = 0.05
UBLUE, DGRAY, RED = "#103E6E", "#374151", "#C0392B"

_log = []


def log(m=""):
    print(m, flush=True)
    _log.append(str(m))


def bh(p):
    p = np.asarray(p, float)
    n = p.size
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def pseudo_f(X, mask):
    """Two-group distance-based pseudo-F on z-scored features."""
    n = X.shape[0]
    g = X[mask]
    r = X[~mask]
    if len(g) < 2 or len(r) < 2:
        return np.nan
    grand = X.mean(0)
    ssb = len(g) * ((g.mean(0) - grand) ** 2).sum() + \
          len(r) * ((r.mean(0) - grand) ** 2).sum()
    ssw = ((g - g.mean(0)) ** 2).sum() + ((r - r.mean(0)) ** 2).sum()
    if ssw <= 0:
        return np.nan
    return (ssb / 1.0) / (ssw / (n - 2))


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    reg = np.load(os.path.join(SRC, "X_regional.npy"))
    fc = np.load(os.path.join(SRC, "X_fc.npy"))
    y = np.load(os.path.join(SRC, "y.npy"))
    subs = np.load(os.path.join(SRC, "subjects.npy"))
    iu = np.triu_indices(246, 1)

    X = {n: reg[:, :, i].astype(np.float64) for i, n in enumerate(FEATS[:3])}
    X["FC"] = fc[:, iu[0], iu[1]].astype(np.float64)
    del reg, fc

    n = len(y)
    log("=" * 74)
    log("ONE-VS-REST MULTIVARIATE PERMUTATION SUMMARY  (4 biomarkers x 6 stages)")
    log("=" * 74)
    log(f"cohort: {n} acquisitions / {len(set(subs))} distinct subjects "
        f"({'one per subject' if len(set(subs)) == n else 'REPEATS PRESENT'})")
    for i, s in enumerate(STAGES):
        log(f"   {s:5s} n = {int((y == i).sum()):3d}")
    log(f"\nstatistic   : distance-based pseudo-F (two-group PERMANOVA)")
    log(f"permutations: {N_PERM} (subject-level label permutation)")
    log(f"seed        : {SEED}")
    log(f"correction  : Benjamini-Hochberg across all "
        f"{len(FEATS)*len(STAGES)} tests")
    log(f"feature dims: " + ", ".join(f"{k}={v.shape[1]}" for k, v in X.items()))
    log("")

    rng = np.random.default_rng(SEED)
    rows, pvals = [], []
    for bm in FEATS:
        Z = X[bm]
        Z = (Z - Z.mean(0)) / np.where(Z.std(0) > 0, Z.std(0), 1.0)
        Z = np.nan_to_num(Z)
        for si, s in enumerate(STAGES):
            mask = (y == si)
            obs = pseudo_f(Z, mask)
            null = np.empty(N_PERM)
            for k in range(N_PERM):
                null[k] = pseudo_f(Z, rng.permutation(mask))
            p = (np.sum(null >= obs) + 1) / (N_PERM + 1)
            rows.append(dict(biomarker=bm, stage=s, n_stage=int(mask.sum()),
                             n_rest=int((~mask).sum()),
                             pseudo_F=float(obs),
                             null_mean=float(np.nanmean(null)),
                             null_p95=float(np.nanpercentile(null, 95)),
                             p_value=float(p)))
            pvals.append(p)
            log(f"  {bm:6s} {s:5s}  n={int(mask.sum()):3d} vs {int((~mask).sum()):3d}  "
                f"F={obs:7.4f}  null mean={np.nanmean(null):6.4f}  p={p:.4f}")
        log("")

    q = bh(pvals)
    for r, qq in zip(rows, q):
        r["q_value_BH"] = float(qq)
        r["significant_q05"] = bool(qq < ALPHA)

    log("after Benjamini-Hochberg across 24 tests:")
    log(f"  {'biomarker':8s} {'stage':6s} {'p':>8s} {'q':>8s}  verdict")
    for r in rows:
        log(f"  {r['biomarker']:8s} {r['stage']:6s} {r['p_value']:8.4f} "
            f"{r['q_value_BH']:8.4f}  "
            f"{'SIGNIFICANT' if r['significant_q05'] else 'not significant'}")
    n_sig = sum(r["significant_q05"] for r in rows)
    log(f"\nsignificant cells: {n_sig} / {len(rows)}")

    # ------------------------------------------------------------ figure
    Q = np.array([r["q_value_BH"] for r in rows]).reshape(len(FEATS), len(STAGES))
    P = np.array([r["p_value"] for r in rows]).reshape(len(FEATS), len(STAGES))
    S = -np.log10(np.maximum(Q, Q_FLOOR))
    thr = -np.log10(ALPHA)

    fig, ax = plt.subplots(figsize=(9.5, 5.0), facecolor="white")
    im = ax.imshow(S, cmap="magma_r", vmin=0, vmax=max(thr * 1.2, S.max() * 1.05),
                   aspect="auto", interpolation="nearest")
    ax.set_xticks(range(len(STAGES))); ax.set_xticklabels(STAGES, fontsize=11)
    ax.set_yticks(range(len(FEATS))); ax.set_yticklabels(FEATS, fontsize=11)
    for i in range(len(FEATS)):
        for j in range(len(STAGES)):
            star = "*" if Q[i, j] < ALPHA else ""
            ax.text(j, i, f"q={Q[i, j]:.3f}{star}", ha="center", va="center",
                    fontsize=9,
                    color="white" if S[i, j] > 0.6 * S.max() else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(f"-log10(BH-adjusted q)   floor {Q_FLOOR:g}", fontsize=9.5)
    cb.ax.axhline(thr, color=RED, lw=1.5)
    cb.ax.text(1.6, thr, " q=0.05", color=RED, fontsize=8, va="center")
    ax.set_title("Biomarker significance across Alzheimer's disease stages\n"
                 "one-vs-rest multivariate permutation test",
                 fontsize=13.5, fontweight="bold", color=UBLUE, pad=12)
    fig.text(0.5, -0.17,
             f"Each cell: that stage's subjects vs the other five stages pooled. "
             f"Statistic = distance-based pseudo-F (two-group PERMANOVA) on "
             f"z-scored features;\n{N_PERM} subject-level label permutations, "
             f"seed {SEED}; p = (#F_perm >= F_obs + 1)/(N+1), so the smallest "
             f"attainable p is {1/(N_PERM+1):.0e}.\n"
             f"Benjamini-Hochberg FDR across all 24 cells. "
             f"* marks q < {ALPHA}. Significant cells: {n_sig}/24.\n"
             f"EXPLORATORY: pooling the other five stages ignores the ordinal "
             f"structure CN->AD. Not derived from the 15 pairwise comparisons.",
             ha="center", fontsize=8.5, color=DGRAY)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(fig_dir, f"one_vs_rest_summary_4x6.{ext}"),
                    dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"\nsaved one_vs_rest_summary_4x6.png/.pdf")

    with open(os.path.join(out_dir, "one_vs_rest_results.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    np.save(os.path.join(out_dir, "one_vs_rest_qvalues_4x6.npy"), Q)
    np.save(os.path.join(out_dir, "one_vs_rest_pvalues_4x6.npy"), P)
    with open(os.path.join(out_dir, "one_vs_rest_method.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"n_acquisitions": int(n), "n_subjects": int(len(set(subs))),
                   "per_stage": {s: int((y == i).sum()) for i, s in enumerate(STAGES)},
                   "statistic": "distance-based pseudo-F (two-group PERMANOVA)",
                   "permutations": N_PERM, "seed": SEED,
                   "p_definition": "(#F_perm >= F_obs + 1)/(N_PERM+1)",
                   "min_attainable_p": 1.0 / (N_PERM + 1),
                   "correction": "Benjamini-Hochberg across 24 tests",
                   "alpha": ALPHA, "q_floor_for_display": Q_FLOOR,
                   "feature_dims": {k: int(v.shape[1]) for k, v in X.items()},
                   "n_significant": int(n_sig),
                   "limitation": "one-vs-rest pools the other five stages and "
                                 "ignores the ordinal structure CN->AD"},
                  fh, indent=2)
    with open(os.path.join(out_dir, "one_vs_rest_log.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(_log))
    log(f"outputs in {os.path.relpath(out_dir, ROOT)}")
    return rows


if __name__ == "__main__":
    import sys
    out = sys.argv[1] if len(sys.argv) > 1 else os.path.join(OUTROOT, "v2_cohort116")
    main(out)
