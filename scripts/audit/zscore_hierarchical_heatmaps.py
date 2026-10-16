"""
Figure A -- between-subject Z-score hierarchical clustering heatmaps.

For each of the four biomarkers a 246 ROI x 116 subject matrix is
standardised ACROSS SUBJECTS within each ROI and displayed with
hierarchical clustering dendrograms on both axes and a disease-stage
annotation strip above the subject columns.

Z-score definition
------------------
    z[i, j] = (x[i, j] - mean_j x[i, :]) / sd_j x[i, :]

    i = ROI (1..246), j = subject (1..116)
    sd uses ddof = 1 (sample standard deviation, n - 1 denominator)

The standardisation scope is therefore ACROSS SUBJECTS WITHIN AN ROI. A
value of +2 means "this subject is two subject-level SDs above the cohort
mean for this ROI". It says nothing about statistical significance and
nothing about progression.

Relationship to the existing normalisation
------------------------------------------
The input features are already normalised WITHIN each acquisition:

    mALFF, mReHo  ratio to that acquisition's own 246-ROI mean
    DC_z          z-score across the 246 ROIs of that acquisition
    FC            Fisher-z, signed, unthresholded, diagonal zeroed

Those are within-acquisition operations. The Z-score applied here is a
second, orthogonal standardisation along the OTHER axis (across subjects).
For DC_z this means a z-score of a z-score: the first is across ROIs within
a subject, the second across subjects within an ROI. They are different
operations on different axes and are not redundant, but the double
standardisation is why DC_z is labelled explicitly in the figure.

Zero-variance ROIs are not silently zeroed -- they are detected, written to
csv/zero_variance_rois.csv, excluded from clustering and left blank.

FC representation
-----------------
FC is a 246x246 matrix, not a regional scalar. The ROI-level summary is the
UNSIGNED mean connectivity strength

    FCstr_i = mean_{j != i} |z_ij|

computed by build_transformer_inputs.py (X_fc_roi_unsigned.npy). The signed
row mean was rejected because it equals DC/245 and correlates with DC_z at
r = +0.994; the unsigned strength correlates with DC_z at r = -0.090 and so
carries different information. A separate 246x246 per-stage mean
connectivity figure is also produced, labelled distinctly.

Clustering
----------
    distance   correlation distance, 1 - Pearson r
    linkage    average (UPGMA)
    ordering   scipy optimal_leaf_ordering, deterministic

Correlation distance is used because the question is whether ROIs vary
TOGETHER across subjects, which is a question about profile shape, not
magnitude. Average linkage is used because it does not assume spherical,
equally sized clusters the way Ward does.

Unsupervised clusters are NOT claimed to correspond to clinical stages.
"""
import csv
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap, TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy.cluster import hierarchy
from scipy.spatial.distance import squareform

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "derivatives", "model_ready")
V2 = os.path.join(ROOT, "derivatives", "INTRA_INTER_CLASS_HEATMAPS", "v2_cohort116")
OUT = os.path.join(V2, "zscore_and_pscore_heatmaps")
LUT = os.path.join(ROOT, "derivatives", "fsfast", "FINAL_6GROUP_AUDIT",
                   "dataset_brainnetome_summary.csv")

STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
STAGE_COLORS = {
    "CN": "#1A5276", "SMC": "#2E86C1", "EMCI": "#7FB3D5",
    "MCI": "#F5B041", "LMCI": "#D35400", "AD": "#922B21",
}
N_ROI = 246
DDOF = 1
LINKAGE_METHOD = "average"
DISTANCE_METRIC = "correlation"

TITLES = {
    "mALFF": "Hierarchical Clustering of Regional mALFF Across Alzheimer's Disease Stages",
    "mReHo": "Hierarchical Clustering of Regional mReHo Across Alzheimer's Disease Stages",
    "DC_z": "Hierarchical Clustering of Regional DC_z Across Alzheimer's Disease Stages",
    "FC": "Hierarchical Clustering of Regional Functional Connectivity\nAcross Alzheimer's Disease Stages",
}
SUBTITLES = {
    "mALFF": "Z-score across subjects within each ROI  |  input: mALFF (ratio to within-acquisition mean)",
    "mReHo": "Z-score across subjects within each ROI  |  input: mReHo (ratio to within-acquisition mean)",
    "DC_z": "Z-score across subjects within each ROI  |  input: DC_z (already z-scored across ROIs within acquisition)",
    "FC": "Z-score across subjects within each ROI  |  input: unsigned mean Fisher-z strength, mean_{j!=i} |z_ij|",
}

_log = []


def log(m=""):
    print(m, flush=True)
    _log.append(str(m))


def load_cohort():
    y = np.load(os.path.join(SRC, "y.npy"))
    subjects = np.load(os.path.join(SRC, "subjects.npy")).astype(str)
    meta = json.load(open(os.path.join(SRC, "metadata.json")))
    order = meta["stage_order"]
    if order != STAGES:
        raise SystemExit(f"stage order mismatch: metadata={order} script={STAGES}")
    if len(subjects) != len(set(subjects.tolist())):
        raise SystemExit("subjects are not unique -- refusing to proceed")
    counts = {s: int((y == i).sum()) for i, s in enumerate(STAGES)}
    if counts != meta["class_counts"]:
        raise SystemExit(f"class counts disagree with metadata: {counts}")
    log(f"cohort: {len(subjects)} acquisitions, {len(set(subjects.tolist()))} unique subjects")
    log(f"stage order (from metadata, not reassigned): {order}")
    log(f"class counts: {counts}")
    return y, subjects, counts


def load_features():
    Xr = np.load(os.path.join(SRC, "X_regional.npy")).astype(np.float64)
    fc_u = np.load(os.path.join(SRC, "X_fc_roi_unsigned.npy")).astype(np.float64)
    if Xr.shape[1] != N_ROI or fc_u.shape[1] != N_ROI:
        raise SystemExit(f"ROI count wrong: {Xr.shape} {fc_u.shape}")
    # [subjects, ROI] -> transpose to [ROI, subjects] for display
    return {
        "mALFF": Xr[:, :, 0].T,
        "mReHo": Xr[:, :, 1].T,
        "DC_z": Xr[:, :, 2].T,
        "FC": fc_u.T,
    }


def load_roi_labels():
    rows = list(csv.DictReader(open(LUT)))
    if len(rows) != N_ROI:
        raise SystemExit(f"LUT has {len(rows)} rows, expected {N_ROI}")
    ids = [int(r["ROI_ID"]) for r in rows]
    names = [r["ROI_NAME"] for r in rows]
    if ids != list(range(1, N_ROI + 1)):
        raise SystemExit("LUT ROI_IDs are not 1..246 in order")
    return ids, names


def zscore_across_subjects(M):
    """M is [ROI, subjects]. Returns z, and a boolean mask of zero-variance ROIs."""
    mu = M.mean(axis=1, keepdims=True)
    sd = M.std(axis=1, ddof=DDOF, keepdims=True)
    flat = (sd[:, 0] == 0) | ~np.isfinite(sd[:, 0])
    safe = np.where(flat[:, None], 1.0, sd)
    z = (M - mu) / safe
    z[flat, :] = np.nan          # not silently zeroed
    return z, flat


def cluster(Z, axis_name):
    """Z rows are the objects to cluster. Returns (leaf order, linkage matrix)."""
    n = Z.shape[0]
    if n < 3:
        return np.arange(n), None
    sd = Z.std(axis=1, ddof=DDOF)
    if np.any(sd == 0):
        raise SystemExit(f"{axis_name}: zero-variance rows reached clustering")
    # correlation distance via the correlation matrix (exact, and cheap at this size)
    C = np.corrcoef(Z)
    C = np.clip(C, -1.0, 1.0)
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0
    if not np.all(np.isfinite(D)):
        raise SystemExit(f"{axis_name}: non-finite correlation distances")
    L = hierarchy.linkage(squareform(D, checks=False), method=LINKAGE_METHOD)
    L = hierarchy.optimal_leaf_ordering(L, squareform(D, checks=False))
    order = hierarchy.leaves_list(L)
    coph, _ = hierarchy.cophenet(L, squareform(D, checks=False))
    log(f"  {axis_name}: n={n}, cophenetic r={coph:.3f}, "
        f"distance={DISTANCE_METRIC}, linkage={LINKAGE_METHOD}")
    return order, L


def gray_links(*a, **k):
    return "#8A8A8A"


def make_figure(bm, z, flat, roi_ids, roi_names, subjects, y, outdir):
    keep = ~flat
    zk = z[keep, :]
    kept_idx = np.flatnonzero(keep)

    log(f"{bm}: clustering")
    roi_order_k, roi_L = cluster(zk, "ROI")
    subj_order, subj_L = cluster(zk.T, "subject")
    roi_order = kept_idx[roi_order_k]

    disp = zk[np.ix_(roi_order_k, subj_order)]
    vmax = float(np.nanpercentile(np.abs(disp), 99))
    vmax = max(vmax, 1e-6)

    fig = plt.figure(figsize=(14.5, 16.5))
    gs = GridSpec(
        4, 3, figure=fig,
        width_ratios=[0.11, 1.0, 0.035],
        height_ratios=[0.085, 0.022, 1.0, 0.055],
        wspace=0.22, hspace=0.015,
        left=0.105, right=0.93, top=0.915, bottom=0.045,
    )

    # top dendrogram (subjects)
    ax_top = fig.add_subplot(gs[0, 1])
    hierarchy.dendrogram(subj_L, ax=ax_top, no_labels=True, color_threshold=0,
                         link_color_func=gray_links)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    for s in ax_top.spines.values():
        s.set_visible(False)
    ax_top.set_title("subject clustering (unsupervised -- not clinical stage)",
                     fontsize=8.5, color="#374151", pad=3)

    # stage annotation strip
    ax_strip = fig.add_subplot(gs[1, 1])
    strip = np.array([[matplotlib.colors.to_rgb(STAGE_COLORS[STAGES[y[j]]])
                       for j in subj_order]])
    ax_strip.imshow(strip, aspect="auto", interpolation="nearest")
    ax_strip.set_xticks([])
    ax_strip.set_yticks([])
    ax_strip.set_ylabel("stage", rotation=0, ha="right", va="center",
                        fontsize=8.5, labelpad=8)

    # left dendrogram (ROIs)
    ax_left = fig.add_subplot(gs[2, 0])
    hierarchy.dendrogram(roi_L, ax=ax_left, orientation="left", no_labels=True,
                         color_threshold=0, link_color_func=gray_links)
    ax_left.set_xticks([])
    ax_left.set_yticks([])
    ax_left.invert_yaxis()
    for s in ax_left.spines.values():
        s.set_visible(False)

    # main heatmap
    ax = fig.add_subplot(gs[2, 1])
    im = ax.imshow(disp, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks([])
    step = 12
    ticks = list(range(0, len(roi_order), step))
    ax.set_yticks(ticks)
    ax.set_yticklabels(
        [f"{roi_ids[roi_order[t]]:>3d} {roi_names[roi_order[t]]}" for t in ticks],
        fontsize=6.2, family="monospace")
    fig.text(0.016, 0.49,
             f"Brainnetome ROI  (all {len(roi_order)} rows plotted, "
             f"every {step}th labelled, clustered order)",
             rotation=90, va="center", ha="center", fontsize=9)
    ax.set_xlabel(f"subjects (n = {len(subj_order)}, clustered order; "
                  f"identifiers in csv/clustering_order_{bm}_subjects.csv)",
                  fontsize=9)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_color("#9CA3AF")

    # colorbar
    cax = fig.add_subplot(gs[2, 2])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(f"Z-score of {bm} across subjects within ROI\n"
                 f"(blue = below cohort mean, red = above; "
                 f"scale clipped at the 99th percentile of |Z| = {vmax:.2f})",
                 fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)

    # stage legend
    ax_leg = fig.add_subplot(gs[3, 1])
    ax_leg.axis("off")
    ax_leg.legend(
        handles=[Patch(facecolor=STAGE_COLORS[s], label=s) for s in STAGES],
        loc="center", ncol=6, frameon=False, fontsize=9.5,
        title="disease stage (progression order CN -> AD, as encoded in the data)",
        title_fontsize=8.5)

    fig.suptitle(TITLES[bm], fontsize=14, y=0.975, fontweight="bold")
    fig.text(0.5075, 0.9365, SUBTITLES[bm], ha="center", fontsize=9, color="#374151")
    fig.text(0.5, 0.012,
             "Descriptive clustering only. A Z-score is a relative position within the cohort, "
             "not a statistical test: no claim of significance or progression is made here. "
             "For inference see the FDR-corrected ROI x stage figures.",
             ha="center", fontsize=8, color="#6B7280", style="italic")

    png = os.path.join(outdir, f"zscore_hierarchical_{bm}.png")
    pdf = os.path.join(outdir, f"zscore_hierarchical_{bm}.pdf")
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    log(f"  wrote {os.path.relpath(png, ROOT)}")
    log(f"  wrote {os.path.relpath(pdf, ROOT)}")
    return roi_order, subj_order, vmax


def write_csvs(bm, z, flat, roi_ids, roi_names, subjects, y, roi_order, subj_order, csvdir):
    # full z matrix in ORIGINAL ROI x original subject order
    p = os.path.join(csvdir, f"zscore_{bm}_roi_by_subject.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ROI_ID", "ROI_NAME", "zero_variance"] + list(subjects))
        for i in range(N_ROI):
            row = ["" if not np.isfinite(v) else f"{v:.6f}" for v in z[i, :]]
            w.writerow([roi_ids[i], roi_names[i], int(flat[i])] + row)
    log(f"  wrote {os.path.relpath(p, ROOT)}")

    p = os.path.join(csvdir, f"clustering_order_{bm}_subjects.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["display_column", "subject_id", "stage", "stage_index",
                    "original_index"])
        for col, j in enumerate(subj_order):
            w.writerow([col, subjects[j], STAGES[y[j]], int(y[j]), int(j)])
    log(f"  wrote {os.path.relpath(p, ROOT)}")

    p = os.path.join(csvdir, f"clustering_order_{bm}_rois.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["display_row", "ROI_ID", "ROI_NAME", "original_index"])
        for row, i in enumerate(roi_order):
            w.writerow([row, roi_ids[i], roi_names[i], int(i)])
    log(f"  wrote {os.path.relpath(p, ROOT)}")


def fc_stage_means(y, roi_ids, outdir, csvdir):
    """Separate 246x246 per-stage mean connectivity figure -- NOT the ROI x stage
    significance heatmap, and not a Z-score figure."""
    X = np.load(os.path.join(SRC, "X_fc.npy"), mmap_mode="r")
    means = np.empty((len(STAGES), N_ROI, N_ROI), dtype=np.float64)
    for i, s in enumerate(STAGES):
        idx = np.flatnonzero(y == i)
        acc = np.zeros((N_ROI, N_ROI), dtype=np.float64)
        for j in idx:                      # one at a time, low memory
            acc += np.asarray(X[j], dtype=np.float64)
        means[i] = acc / len(idx)
        log(f"  {s}: mean of {len(idx)} subjects, "
            f"off-diagonal mean z = {means[i][~np.eye(N_ROI, dtype=bool)].mean():+.4f}")
    del X

    vmax = float(np.percentile(np.abs(means[:, ~np.eye(N_ROI, dtype=bool)]), 99))
    fig, axes = plt.subplots(2, 3, figsize=(15, 10.6))
    for i, (s, ax) in enumerate(zip(STAGES, axes.ravel())):
        im = ax.imshow(means[i], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       interpolation="nearest")
        ax.set_title(f"{s}  (n = {int((y == i).sum())})", fontsize=11,
                     color=STAGE_COLORS[s], fontweight="bold")
        ax.set_xticks([0, 122, 245])
        ax.set_yticks([0, 122, 245])
        ax.set_xticklabels(["1", "123", "246"], fontsize=7)
        ax.set_yticklabels(["1", "123", "246"], fontsize=7)
        ax.tick_params(length=2)
    fig.suptitle("Mean Within-Stage Functional Connectivity (246 x 246 Fisher-z)",
                 fontsize=14, fontweight="bold", y=0.975)
    fig.text(0.5, 0.935,
             "Signed, unthresholded, diagonal zeroed. This is a 246 x 246 connectivity "
             "pattern per stage -- it is NOT the 246 x 6 ROI-by-stage significance heatmap "
             "and carries no statistical test.",
             ha="center", fontsize=9, color="#374151")
    cb = fig.colorbar(im, ax=axes, fraction=0.022, pad=0.02)
    cb.set_label(f"mean Fisher-z connectivity (scale clipped at 99th percentile "
                 f"of |z| = {vmax:.3f})", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    png = os.path.join(outdir, "fc_stage_mean_connectivity_246x246.png")
    pdf = os.path.join(outdir, "fc_stage_mean_connectivity_246x246.pdf")
    fig.savefig(png, dpi=180, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    log(f"  wrote {os.path.relpath(png, ROOT)}")
    log(f"  wrote {os.path.relpath(pdf, ROOT)}")

    p = os.path.join(csvdir, "fc_stage_mean_offdiagonal_summary.csv")
    off = ~np.eye(N_ROI, dtype=bool)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["stage", "n_subjects", "mean_offdiag_z", "sd_offdiag_z",
                    "frac_positive_edges"])
        for i, s in enumerate(STAGES):
            v = means[i][off]
            w.writerow([s, int((y == i).sum()), f"{v.mean():.6f}",
                        f"{v.std(ddof=DDOF):.6f}", f"{(v > 0).mean():.6f}"])
    log(f"  wrote {os.path.relpath(p, ROOT)}")


def main():
    for sub in ("zscore_hierarchical", "roi_stage_significance", "overall_summary",
                "csv", "logs"):
        os.makedirs(os.path.join(OUT, sub), exist_ok=True)
    figdir = os.path.join(OUT, "zscore_hierarchical")
    csvdir = os.path.join(OUT, "csv")

    log("=" * 78)
    log("FIGURE A -- between-subject Z-score hierarchical clustering")
    log("=" * 78)
    y, subjects, counts = load_cohort()
    roi_ids, roi_names = load_roi_labels()
    feats = load_features()

    zv_rows = []
    for bm in ["mALFF", "mReHo", "DC_z", "FC"]:
        log("")
        log(f"--- {bm} ---")
        M = feats[bm]
        log(f"  input [ROI, subjects] = {M.shape}, finite = {bool(np.isfinite(M).all())}")
        z, flat = zscore_across_subjects(M)
        nflat = int(flat.sum())
        log(f"  zero-variance ROIs across subjects: {nflat}")
        for i in np.flatnonzero(flat):
            zv_rows.append([bm, roi_ids[i], roi_names[i], f"{M[i].mean():.6f}"])
        finite = z[~flat]
        log(f"  Z-score check: mean={finite.mean():+.3e} (expect 0), "
            f"sd={finite.std(ddof=DDOF):.6f} (expect ~1), "
            f"range=[{finite.min():+.2f}, {finite.max():+.2f}]")
        roi_order, subj_order, vmax = make_figure(
            bm, z, flat, roi_ids, roi_names, subjects, y, figdir)
        write_csvs(bm, z, flat, roi_ids, roi_names, subjects, y,
                   roi_order, subj_order, csvdir)

    p = os.path.join(csvdir, "zero_variance_rois.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["biomarker", "ROI_ID", "ROI_NAME", "constant_value"])
        w.writerows(zv_rows)
    log("")
    log(f"zero-variance ROIs total: {len(zv_rows)} -> {os.path.relpath(p, ROOT)}")

    log("")
    log("--- additional: per-stage mean 246x246 FC (labelled separately) ---")
    fc_stage_means(y, roi_ids, figdir, csvdir)

    lp = os.path.join(OUT, "logs", "zscore_hierarchical.log")
    with open(lp, "w", encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")
    print(f"\nlog -> {os.path.relpath(lp, ROOT)}")


if __name__ == "__main__":
    main()
