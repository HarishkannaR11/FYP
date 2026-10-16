"""
Figure C -- overall 4 x 6 biomarker-by-stage significance heatmap
Figure D -- optional multi-panel summary

Figure C plots the EXISTING stage-level multivariate one-vs-rest results
from one_vs_rest_statistics.csv. Nothing is recomputed. The script verifies
before plotting that:

  * all 24 tests are status COMPLETED
  * all 24 share one cohort fingerprint and one seed
  * Benjamini-Hochberg over the full 24-test family reproduces the stored
    q-values exactly
  * the matrix is exactly 4 biomarkers x 6 stages

The correction family for this figure is the complete set of 24 tests
(4 biomarkers x 6 stages) corrected together, which is what the stored
q_value_BH column already contains.

Statistic: distance-based pseudo-F, two-group (stage versus pooled rest),
assessed by permutation with 9,999 permutations. With the +1 convention the
smallest attainable p is 1e-4, so -log10(p) is bounded above by 4. The
colour axis is -log10(q) and is extended to at least -log10(0.05) = 1.301
so the significance threshold is always visible.

Figure D places the three regional Z-score clustering heatmaps beside the
4 x 6 summary. The full-resolution 246-ROI figures remain separate files;
the panels here are for orientation only and reuse the exact clustered
orders written by zscore_hierarchical_heatmaps.py so the two agree.
"""
import csv
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "derivatives", "model_ready")
V2 = os.path.join(ROOT, "derivatives", "INTRA_INTER_CLASS_HEATMAPS", "v2_cohort116")
OUT = os.path.join(V2, "zscore_and_pscore_heatmaps")

STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
BIOMARKERS = ["mALFF", "mReHo", "DC_z", "FC"]
STAGE_COLORS = {
    "CN": "#1A5276", "SMC": "#2E86C1", "EMCI": "#7FB3D5",
    "MCI": "#F5B041", "LMCI": "#D35400", "AD": "#922B21",
}
ALPHA = 0.05
THRESH = -np.log10(ALPHA)
FLOOR = 1e-12
N_ROI = 246
DDOF = 1

_log = []


def log(m=""):
    print(m, flush=True)
    _log.append(str(m))


def bh(p):
    flat = np.asarray(p, float).ravel()
    n = flat.size
    o = np.argsort(flat)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((flat[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(q, 0, 1).reshape(np.shape(p))


def neglog10(x):
    return -np.log10(np.maximum(np.asarray(x, float), FLOOR))


def load_overall():
    rows = list(csv.DictReader(open(os.path.join(V2, "one_vs_rest_statistics.csv"))))
    if len(rows) != 24:
        raise SystemExit(f"expected 24 rows, found {len(rows)}")
    bad = [r for r in rows if r["status"] != "COMPLETED"]
    if bad:
        raise SystemExit(f"{len(bad)} tests are not COMPLETED -- refusing to plot")
    fps = {r["cohort_fingerprint"] for r in rows}
    seeds = {r["seed"] for r in rows}
    perms = {r["n_permutations"] for r in rows}
    if len(fps) != 1 or len(seeds) != 1:
        raise SystemExit(f"inconsistent provenance: fingerprints={fps} seeds={seeds}")
    log(f"24/24 COMPLETED | fingerprint {fps.pop()} | seed {seeds.pop()} | "
        f"permutations {perms.pop()}")

    P = np.full((4, 6), np.nan)
    Q = np.full((4, 6), np.nan)
    F = np.full((4, 6), np.nan)
    for r in rows:
        i = BIOMARKERS.index(r["biomarker"])
        j = STAGES.index(r["stage"])
        P[i, j] = float(r["p_value"])
        Q[i, j] = float(r["q_value_BH"])
        F[i, j] = float(r["pseudo_F"])
    if np.isnan(P).any() or np.isnan(Q).any():
        raise SystemExit("missing cells in the 4 x 6 matrix -- not filling with zero")
    if not np.allclose(bh(P), Q, atol=1e-9):
        raise SystemExit("BH over the 24-test family does not reproduce stored q")
    log(f"BH over the full 24-test family reproduces stored q exactly "
        f"(max diff {np.abs(bh(P) - Q).max():.2e})")
    log(f"shape {Q.shape} = {Q.size} cells; q < {ALPHA}: {int((Q < ALPHA).sum())}")
    return P, Q, F


def figure_c(P, Q, F, outdir, csvdir):
    V = neglog10(Q)
    vmax = float(max(V.max(), THRESH)) * 1.05

    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    fig.subplots_adjust(left=0.115, right=0.875, top=0.80, bottom=0.175)
    im = ax.imshow(V, cmap="YlOrRd", vmin=0, vmax=vmax, aspect="auto",
                   interpolation="nearest")
    ax.set_xticks(range(6))
    ax.set_xticklabels(STAGES, fontsize=11)
    for t, s in zip(ax.get_xticklabels(), STAGES):
        t.set_color(STAGE_COLORS[s])
        t.set_fontweight("bold")
    ax.set_yticks(range(4))
    ax.set_yticklabels(["Functional\nConnectivity" if b == "FC" else b
                        for b in BIOMARKERS], fontsize=11)
    ax.set_xlabel("disease stage (one-vs-rest: this stage versus the pooled other five)",
                  fontsize=10, labelpad=8)

    for i in range(4):
        for j in range(6):
            shade = V[i, j] / vmax
            sig = Q[i, j] < ALPHA
            ax.text(j, i - 0.13, f"q = {Q[i, j]:.3f}", ha="center", va="center",
                    fontsize=10, fontweight="bold" if sig else "normal",
                    color="#FFFFFF" if shade > 0.55 else "#111827")
            ax.text(j, i + 0.19, f"p = {P[i, j]:.4f}   F = {F[i, j]:.2f}",
                    ha="center", va="center", fontsize=7.3,
                    color="#F3F4F6" if shade > 0.55 else "#4B5563")
            if sig:
                ax.text(j, i + 0.38, "*", ha="center", va="center", fontsize=14,
                        color="#FFFFFF" if shade > 0.55 else "#111827")
    for j in range(1, 6):
        ax.axvline(j - 0.5, color="white", lw=1.4)
    for i in range(1, 4):
        ax.axhline(i - 0.5, color="white", lw=1.4)

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.025)
    cb.set_label("-log10(q), Benjamini-Hochberg adjusted\n"
                 "family = 4 biomarkers x 6 stages = 24 tests, corrected together",
                 fontsize=8.5)
    cb.ax.tick_params(labelsize=8)
    cb.ax.axhline(THRESH, color="#111827", lw=1.5)
    cb.ax.text(0.5, THRESH, f"q = {ALPHA}", transform=cb.ax.get_yaxis_transform(),
               ha="center", va="bottom", fontsize=7.5, fontweight="bold",
               color="#111827",
               bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none", alpha=0.85))

    fig.suptitle("Stage-Level Multivariate Significance by Biomarker and Stage",
                 fontsize=14, fontweight="bold", y=0.975)
    fig.text(0.5, 0.885,
             "distance-based pseudo-F, two-group permutation test (9,999 permutations) "
             "on all 246 ROIs jointly\n"
             "cells show the FDR-adjusted q-value, with the raw p and the pseudo-F "
             "statistic beneath",
             ha="center", va="top", fontsize=9, color="#374151", linespacing=1.5)

    n_sig = int((Q < ALPHA).sum())
    msg = (f"no biomarker-stage combination reaches q < {ALPHA}  "
           f"(0 of 24 tests; smallest q = {Q.min():.3f})" if n_sig == 0 else
           f"{n_sig} of 24 combinations reach q < {ALPHA}")
    fig.text(0.5, 0.055, msg, ha="center", fontsize=10.5,
             color="#B91C1C" if n_sig == 0 else "#111827", fontweight="bold")

    png = os.path.join(outdir, "overall_zscore_significance_heatmap.png")
    pdf = os.path.join(outdir, "overall_zscore_significance_heatmap.pdf")
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    log(f"wrote {os.path.relpath(png, ROOT)}")
    log(f"wrote {os.path.relpath(pdf, ROOT)}")

    for path in (os.path.join(outdir, "overall_zscore_significance_heatmap.csv"),
                 os.path.join(csvdir, "overall_summary_4x6.csv")):
        with open(path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["biomarker", "stage", "pseudo_F", "p_value_permutation",
                        "q_value_BH_24family", "neg_log10_q", "significant_q05"])
            for i, b in enumerate(BIOMARKERS):
                for j, s in enumerate(STAGES):
                    # shortest round-trip precision: the copy is bit-identical
                    # to one_vs_rest_statistics.csv, not a rounded version
                    w.writerow([b, s, repr(float(F[i, j])), repr(float(P[i, j])),
                                repr(float(Q[i, j])), repr(float(V[i, j])),
                                bool(Q[i, j] < ALPHA)])
        log(f"wrote {os.path.relpath(path, ROOT)}")
    return V, vmax


def load_order(bm, csvdir):
    rows = list(csv.DictReader(open(os.path.join(
        csvdir, f"clustering_order_{bm}_rois.csv"))))
    roi_order = np.array([int(r["original_index"]) for r in rows])
    rows = list(csv.DictReader(open(os.path.join(
        csvdir, f"clustering_order_{bm}_subjects.csv"))))
    subj_order = np.array([int(r["original_index"]) for r in rows])
    stages = [r["stage"] for r in rows]
    return roi_order, subj_order, stages


def figure_d(Vq, vmax_q, Q, outdir, csvdir):
    Xr = np.load(os.path.join(SRC, "X_regional.npy")).astype(np.float64)
    feats = {"mALFF": Xr[:, :, 0].T, "mReHo": Xr[:, :, 1].T, "DC_z": Xr[:, :, 2].T}

    fig = plt.figure(figsize=(17.5, 11.2))
    gs = GridSpec(2, 2, figure=fig, wspace=0.17, hspace=0.26,
                  left=0.055, right=0.945, top=0.838, bottom=0.075)
    panels = ["mALFF", "mReHo", "DC_z"]

    for k, bm in enumerate(panels):
        ax = fig.add_subplot(gs[k // 2, k % 2])
        M = feats[bm]
        z = (M - M.mean(1, keepdims=True)) / M.std(1, ddof=DDOF, keepdims=True)
        roi_order, subj_order, stages = load_order(bm, csvdir)
        disp = z[np.ix_(roi_order, subj_order)]
        vmax = float(np.nanpercentile(np.abs(disp), 99))
        im = ax.imshow(disp, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([0, N_ROI - 1])
        ax.set_yticklabels(["1", "246"], fontsize=8)
        ax.set_title(f"{chr(65 + k)}.  {bm}  --  Z-score across subjects, clustered",
                     fontsize=11, fontweight="bold", loc="left", pad=16)
        ax.set_ylabel("ROI (clustered)", fontsize=8.5)
        ax.set_xlabel("subjects (clustered)", fontsize=8.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.015)
        cb.set_label("Z-score", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        # stage strip drawn just above the panel
        pos = ax.get_position()
        sax = fig.add_axes([pos.x0, pos.y1 + 0.004, pos.width, 0.012])
        sax.imshow(np.array([[matplotlib.colors.to_rgb(STAGE_COLORS[s])
                              for s in stages]]),
                   aspect="auto", interpolation="nearest")
        sax.set_xticks([])
        sax.set_yticks([])

    # panel D -- the 4 x 6 summary
    ax = fig.add_subplot(gs[1, 1])
    im = ax.imshow(Vq, cmap="YlOrRd", vmin=0, vmax=vmax_q, aspect="auto",
                   interpolation="nearest")
    ax.set_xticks(range(6))
    ax.set_xticklabels(STAGES, fontsize=9.5)
    for t, s in zip(ax.get_xticklabels(), STAGES):
        t.set_color(STAGE_COLORS[s])
        t.set_fontweight("bold")
    ax.set_yticks(range(4))
    ax.set_yticklabels(["FC" if b == "FC" else b for b in BIOMARKERS], fontsize=10)
    ax.set_title("D.  Stage-level multivariate significance (4 x 6)",
                 fontsize=11, fontweight="bold", loc="left", pad=16)
    for i in range(4):
        for j in range(6):
            shade = Vq[i, j] / vmax_q
            ax.text(j, i, f"q = {Q[i, j]:.3f}", ha="center", va="center",
                    fontsize=9, color="#FFFFFF" if shade > 0.55 else "#111827")
    for j in range(1, 6):
        ax.axvline(j - 0.5, color="white", lw=1.2)
    for i in range(1, 4):
        ax.axhline(i - 0.5, color="white", lw=1.2)
    cb = fig.colorbar(im, ax=ax, fraction=0.028, pad=0.015)
    cb.set_label("-log10(q)", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    cb.ax.axhline(THRESH, color="#111827", lw=1.3)
    cb.ax.text(1.5, THRESH, f" q={ALPHA}", transform=cb.ax.get_yaxis_transform(),
               va="center", fontsize=7, fontweight="bold")

    fig.legend(handles=[Patch(facecolor=STAGE_COLORS[s], label=s) for s in STAGES],
               loc="lower center", ncol=6, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.012))

    fig.suptitle("Regional Z-Score Structure and Stage-Level Significance "
                 "Across Alzheimer's Disease Stages",
                 fontsize=15, fontweight="bold", y=0.985)
    fig.text(0.5, 0.947,
             "Panels A-C are DESCRIPTIVE: between-subject Z-scores with hierarchical "
             "clustering, no statistical test. Panel D is INFERENTIAL: FDR-adjusted "
             "permutation q-values.\n"
             "The two must not be read as one result. Full-resolution 246-ROI versions "
             "of A-C are separate files in zscore_hierarchical/.",
             ha="center", va="top", fontsize=9, color="#374151", linespacing=1.6)

    png = os.path.join(outdir, "combined_summary_panels.png")
    pdf = os.path.join(outdir, "combined_summary_panels.pdf")
    fig.savefig(png, dpi=190)
    fig.savefig(pdf)
    plt.close(fig)
    log(f"wrote {os.path.relpath(png, ROOT)}")
    log(f"wrote {os.path.relpath(pdf, ROOT)}")


def main():
    outdir = os.path.join(OUT, "overall_summary")
    csvdir = os.path.join(OUT, "csv")
    for d in (outdir, csvdir, os.path.join(OUT, "logs")):
        os.makedirs(d, exist_ok=True)

    log("=" * 78)
    log("FIGURE C -- overall 4 x 6 biomarker x stage significance")
    log("=" * 78)
    P, Q, F = load_overall()
    Vq, vmax_q = figure_c(P, Q, F, outdir, csvdir)
    if (Q < ALPHA).sum() == 0:
        log(f"No biomarker-stage combination reaches q < {ALPHA}. Smallest q = "
            f"{Q.min():.4f} ({BIOMARKERS[np.unravel_index(Q.argmin(), Q.shape)[0]]} "
            f"at {STAGES[np.unravel_index(Q.argmin(), Q.shape)[1]]}).")

    log("")
    log("=" * 78)
    log("FIGURE D -- combined multi-panel summary")
    log("=" * 78)
    figure_d(Vq, vmax_q, Q, outdir, csvdir)

    lp = os.path.join(OUT, "logs", "overall_and_combined.log")
    with open(lp, "w", encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")
    print(f"\nlog -> {os.path.relpath(lp, ROOT)}")


if __name__ == "__main__":
    main()
