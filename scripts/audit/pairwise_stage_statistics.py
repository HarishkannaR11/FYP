"""
Pairwise stage statistics: effect sizes, FDR-corrected significance, and
cross-biomarker consensus.

Descriptive and inferential outputs are kept separate. Nothing here is a
model and nothing is trained.

Design
------
15 pairwise comparisons in a fixed order. For every ROI and every
comparison:
    effect size   Hedges' g  (bias-corrected standardised mean difference,
                  first stage minus second)
    test          Mann-Whitney U, two-sided
    correction    Benjamini-Hochberg within a prespecified family of
                  246 ROIs x 15 comparisons = 3690 tests, PER BIOMARKER

Why a rank test with a parametric effect size: group sizes are 14-20, so
normality cannot be assumed and Mann-Whitney is the safer inferential
choice. Hedges' g is reported alongside it because it is interpretable as
a magnitude; it is NOT the statistic the p-value comes from, and the report
says so. Rank-biserial correlation is also stored as the effect size that
matches the test.

FC edges are a SEPARATE family (30,135 edges per comparison), never pooled
with the ROI-level tests.

Input : derivatives/model_ready/   (one acquisition per subject, so the
        samples are independent and no repeated-measures model is needed)
Output: derivatives/INTRA_INTER_CLASS_HEATMAPS/pairwise/
"""
import csv
import itertools
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "derivatives", "model_ready")
import sys as _sys
OUT = (_sys.argv[1] if len(_sys.argv) > 1 else
       os.path.join(ROOT, "derivatives", "INTRA_INTER_CLASS_HEATMAPS", "pairwise"))
FIG = os.path.join(OUT, "figures")
LUT = os.path.join(ROOT, "derivatives", "fsfast", "FINAL_6GROUP_AUDIT",
                   "dataset_brainnetome_summary.csv")

STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
REGIONAL = ["mALFF", "mReHo", "DC_z"]
N_ROI = 246
ALPHA = 0.05
# fixed comparison order, used identically in every output
COMPARISONS = [(a, b) for i, a in enumerate(STAGES) for b in STAGES[i + 1:]]
COMP_LABELS = [f"{a}-{b}" for a, b in COMPARISONS]
UBLUE, DGRAY, RED = "#103E6E", "#374151", "#C0392B"

_log = []


def log(m=""):
    print(m, flush=True)
    _log.append(str(m))


def bh(p):
    """Benjamini-Hochberg adjusted p-values over a flat family."""
    p = np.asarray(p, dtype=float)
    flat = p.ravel()
    n = flat.size
    o = np.argsort(flat)
    q = np.empty(n)
    q[o] = np.minimum.accumulate(
        (flat[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(q, 0, 1).reshape(p.shape)


def hedges_g(a, b):
    """Bias-corrected standardised mean difference, a minus b."""
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return np.nan
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if sp2 <= 0:
        return np.nan
    d = (a.mean() - b.mean()) / np.sqrt(sp2)
    J = 1.0 - 3.0 / (4 * (na + nb) - 9)      # small-sample correction
    return d * J


def roi_names():
    names = {}
    if os.path.isfile(LUT):
        with open(LUT, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                names[int(r["ROI_ID"])] = r["ROI_NAME"]
    return names


def load():
    reg = np.load(os.path.join(SRC, "X_regional.npy"))
    fc = np.load(os.path.join(SRC, "X_fc.npy"))
    y = np.load(os.path.join(SRC, "y.npy"))
    subs = np.load(os.path.join(SRC, "subjects.npy"))
    iu = np.triu_indices(N_ROI, 1)
    feats = {n: reg[:, :, i].astype(np.float64) for i, n in enumerate(REGIONAL)}
    edges = fc[:, iu[0], iu[1]].astype(np.float64)
    del reg, fc
    return feats, edges, y, subs, iu


# ------------------------------------------------------------ regional
def regional_tests(feats, y):
    res = {}
    for bm in REGIONAL:
        X = feats[bm]
        g = np.full((N_ROI, len(COMPARISONS)), np.nan)
        rb = np.full_like(g, np.nan)
        p = np.ones_like(g)
        for c, (sa, sb) in enumerate(COMPARISONS):
            A = X[y == STAGES.index(sa)]
            B = X[y == STAGES.index(sb)]
            na, nb = len(A), len(B)
            for r in range(N_ROI):
                a, b = A[:, r], B[:, r]
                g[r, c] = hedges_g(a, b)
                try:
                    U, pv = stats.mannwhitneyu(a, b, alternative="two-sided")
                    p[r, c] = pv
                    rb[r, c] = 2.0 * U / (na * nb) - 1.0   # rank-biserial
                except ValueError:
                    p[r, c] = 1.0
        q = bh(p)                      # family = 246 x 15, per biomarker
        res[bm] = dict(g=g, rb=rb, p=p, q=q, sig=(q < ALPHA))
        log(f"  {bm:6s} tests={g.size}  p<0.05 raw={int((p<ALPHA).sum()):5d}  "
            f"q<0.05={int((q<ALPHA).sum()):4d}  min q={q.min():.4f}  "
            f"max |g|={np.nanmax(np.abs(g)):.3f}")
    return res


# ------------------------------------------------------------------ FC
def fc_tests(edges, y):
    n_edge = edges.shape[1]
    log(f"\n  FC: {n_edge:,} edges x {len(COMPARISONS)} comparisons = "
        f"{n_edge*len(COMPARISONS):,} tests")
    log("  Family for FDR: the edges of ONE comparison (never pooled with ROI tests).")
    out = {}
    for c, (sa, sb) in enumerate(COMPARISONS):
        A = edges[y == STAGES.index(sa)]
        B = edges[y == STAGES.index(sb)]
        na, nb = len(A), len(B)
        sp2 = ((na - 1) * A.var(0, ddof=1) + (nb - 1) * B.var(0, ddof=1)) / (na + nb - 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            d = (A.mean(0) - B.mean(0)) / np.sqrt(sp2)
        d *= 1.0 - 3.0 / (4 * (na + nb) - 9)
        U, p = stats.mannwhitneyu(A, B, alternative="two-sided", axis=0)
        q = bh(p)
        out[COMP_LABELS[c]] = dict(g=d, p=p, q=q, n_sig=int((q < ALPHA).sum()))
        log(f"    {COMP_LABELS[c]:12s} n={na:2d} vs {nb:2d}  "
            f"edges q<0.05: {int((q<ALPHA).sum()):5d}  min q={q.min():.4f}")
    return out


# -------------------------------------------------------------- figures
def heat(ax, M, cmap, vmin, vmax, title, cbar_label, fig):
    im = ax.imshow(M, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(range(len(COMP_LABELS)))
    ax.set_xticklabels(COMP_LABELS, rotation=90, fontsize=7)
    ax.set_ylabel("Brainnetome ROI (1-246)", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", color=UBLUE)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label=cbar_label)


def plot_regional(res):
    for bm in REGIONAL:
        d = res[bm]
        lim = np.nanpercentile(np.abs(d["g"]), 99)
        fig, axes = plt.subplots(1, 3, figsize=(17, 7.5), facecolor="white")
        heat(axes[0], d["g"], "RdBu_r", -lim, lim,
             f"{bm}: Hedges' g", "Hedges' g  (first stage - second)", fig)
        # effect size with significance overlaid
        heat(axes[1], d["g"], "RdBu_r", -lim, lim,
             f"{bm}: g with q<0.05 marked", "Hedges' g", fig)
        ys, xs = np.where(d["sig"])
        axes[1].scatter(xs, ys, s=4, c="black", marker="s")
        heat(axes[2], d["sig"].astype(float), "Greys", 0, 1,
             f"{bm}: significant (q<{ALPHA})", "1 = significant", fig)
        fig.suptitle(f"Pairwise stage differences: {bm}",
                     fontsize=15.5, fontweight="bold", color=UBLUE, y=1.0)
        fig.text(0.5, -0.04,
                 f"Mann-Whitney U, two-sided. Benjamini-Hochberg FDR over "
                 f"{N_ROI} ROIs x {len(COMPARISONS)} comparisons = "
                 f"{N_ROI*len(COMPARISONS)} tests for this biomarker. "
                 f"Significant cells: {int(d['sig'].sum())}.",
                 ha="center", fontsize=9.5, color=DGRAY)
        fig.tight_layout()
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(FIG, f"pairwise_{bm}.{ext}"), dpi=200,
                        bbox_inches="tight", facecolor="white")
        plt.close(fig)
        log(f"   saved pairwise_{bm}.png/.pdf")


def plot_consensus(cons):
    fig, ax = plt.subplots(figsize=(8.5, 8.5), facecolor="white")
    from matplotlib.colors import BoundaryNorm, ListedColormap
    cmap = ListedColormap(["#f2f4f7", "#9FB6CE", "#1E88E5", "#103E6E"])
    im = ax.imshow(cons, aspect="auto", cmap=cmap,
                   norm=BoundaryNorm([0, 1, 2, 3, 4], 4),
                   interpolation="nearest")
    ax.set_xticks(range(len(COMP_LABELS)))
    ax.set_xticklabels(COMP_LABELS, rotation=90, fontsize=7.5)
    ax.set_ylabel("Brainnetome ROI (1-246)", fontsize=10.5)
    ax.set_title("Cross-biomarker consensus", fontsize=14,
                 fontweight="bold", color=UBLUE)
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, ticks=[0.5, 1.5, 2.5, 3.5])
    cb.ax.set_yticklabels(["0", "1", "2", "3"])
    cb.set_label("number of biomarkers with q < 0.05", fontsize=9.5)
    fig.text(0.5, -0.03,
             "Descriptive count over mALFF, mReHo and DC_z. These biomarkers "
             "are correlated, so this is NOT independent replication and NOT a "
             "combined p-value.",
             ha="center", fontsize=9, color=DGRAY)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(FIG, f"cross_biomarker_consensus.{ext}"),
                    dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log("   saved cross_biomarker_consensus.png/.pdf")


# ------------------------------------------------------------------ main
def main():
    for d in (OUT, FIG):
        os.makedirs(d, exist_ok=True)
    feats, edges, y, subs, iu = load()
    names = roi_names()
    n = len(y)
    log("=" * 74)
    log("PAIRWISE STAGE STATISTICS")
    log("=" * 74)
    log(f"acquisitions {n}   distinct subjects {len(set(subs))}"
        f"   ({'independent samples' if len(set(subs)) == n else 'REPEATS -- not independent'})")
    for i, s in enumerate(STAGES):
        log(f"   {s:5s} n = {int((y == i).sum()):3d}")
    log(f"\ncomparisons ({len(COMPARISONS)}): {', '.join(COMP_LABELS)}")
    log(f"\ntest: Mann-Whitney U (two-sided); effect size: Hedges' g")
    log(f"FDR family: {N_ROI} ROIs x {len(COMPARISONS)} comparisons "
        f"= {N_ROI*len(COMPARISONS)} tests PER biomarker\n")

    res = regional_tests(feats, y)
    fc = fc_tests(edges, y)

    cons = np.zeros((N_ROI, len(COMPARISONS)), dtype=int)
    for bm in REGIONAL:
        cons += res[bm]["sig"].astype(int)
    log(f"\nconsensus: cells with >=1 biomarker significant: "
        f"{int((cons >= 1).sum())}, >=2: {int((cons >= 2).sum())}, "
        f"=3: {int((cons == 3).sum())}")

    # ---------------------------------------------------------- saves
    for bm in REGIONAL:
        d = res[bm]
        for key, arr in [("effect_sizes", d["g"]), ("p_values", d["p"]),
                         ("q_values", d["q"]),
                         ("significance_masks", d["sig"].astype(int)),
                         ("rank_biserial", d["rb"])]:
            with open(os.path.join(OUT, f"{key}_{bm}.csv"), "w", newline="",
                      encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["ROI_ID", "ROI_NAME"] + COMP_LABELS)
                for r in range(N_ROI):
                    w.writerow([r + 1, names.get(r + 1, "")] +
                               [f"{v:.6g}" for v in arr[r]])
    with open(os.path.join(OUT, "cross_biomarker_consensus.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ROI_ID", "ROI_NAME"] + COMP_LABELS)
        for r in range(N_ROI):
            w.writerow([r + 1, names.get(r + 1, "")] + list(cons[r]))

    # which biomarkers contributed to each non-zero consensus cell
    with open(os.path.join(OUT, "consensus_contributors.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["ROI_ID", "ROI_NAME", "comparison", "consensus_count",
                    "biomarkers", "q_values", "hedges_g"])
        for r in range(N_ROI):
            for c in range(len(COMPARISONS)):
                if cons[r, c] == 0:
                    continue
                bms = [b for b in REGIONAL if res[b]["sig"][r, c]]
                w.writerow([r + 1, names.get(r + 1, ""), COMP_LABELS[c],
                            cons[r, c], ";".join(bms),
                            ";".join(f"{res[b]['q'][r, c]:.4g}" for b in bms),
                            ";".join(f"{res[b]['g'][r, c]:.3f}" for b in bms)])

    # FC: effect-size and significance matrices per comparison
    fcdir = os.path.join(OUT, "fc_matrices")
    os.makedirs(fcdir, exist_ok=True)
    tot_sig = 0
    for lab, d in fc.items():
        G = np.zeros((N_ROI, N_ROI)); Q = np.ones((N_ROI, N_ROI))
        G[iu] = d["g"]; G += G.T
        Q[iu] = d["q"]; Q = np.minimum(Q, Q.T)
        np.save(os.path.join(fcdir, f"fc_effectsize_{lab}.npy"), G)
        np.save(os.path.join(fcdir, f"fc_qvalues_{lab}.npy"), Q)
        tot_sig += d["n_sig"]

    plot_regional(res)
    plot_consensus(cons)

    manifest = []
    for bm in REGIONAL:
        d = res[bm]
        manifest.append(dict(biomarker=bm, level="ROI", n_tests=int(d["g"].size),
                             family=f"{N_ROI}x{len(COMPARISONS)}",
                             n_raw_p_lt_05=int((d["p"] < ALPHA).sum()),
                             n_q_lt_05=int(d["sig"].sum()),
                             min_q=float(d["q"].min()),
                             max_abs_g=float(np.nanmax(np.abs(d["g"])))))
    manifest.append(dict(biomarker="FC", level="edge",
                         n_tests=int(edges.shape[1] * len(COMPARISONS)),
                         family="30135 edges per comparison",
                         n_raw_p_lt_05="", n_q_lt_05=tot_sig, min_q="",
                         max_abs_g=""))
    with open(os.path.join(OUT, "analysis_manifest.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(manifest[0].keys()))
        w.writeheader(); w.writerows(manifest)

    with open(os.path.join(OUT, "analysis_log.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(_log))
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"n_acquisitions": int(n), "n_subjects": int(len(set(subs))),
                   "per_stage": {s: int((y == i).sum()) for i, s in enumerate(STAGES)},
                   "comparisons": COMP_LABELS, "alpha": ALPHA,
                   "test": "Mann-Whitney U two-sided",
                   "effect_size": "Hedges' g (first - second)",
                   "fdr_family_regional": f"{N_ROI} ROIs x {len(COMPARISONS)} comparisons per biomarker",
                   "fdr_family_fc": "edges within one comparison",
                   "regional_significant": {b: int(res[b]["sig"].sum()) for b in REGIONAL},
                   "fc_significant_edges_total": tot_sig,
                   "consensus_ge1": int((cons >= 1).sum()),
                   "consensus_eq3": int((cons == 3).sum())}, fh, indent=2)
    log(f"\noutputs in {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
