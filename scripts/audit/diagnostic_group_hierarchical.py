"""
Hierarchical clustering of the six DIAGNOSTIC GROUPS by their regional
biomarker profiles.

This is a different question from every clustering already in this project:

    existing Z-score figures   cluster 246 ROIs and 116 SUBJECTS
    existing P-score figures   cluster 246 ROIs by SIGNIFICANCE profile
    THIS script                clusters the 6 DIAGNOSTIC GROUPS by their mean
                               regional biomarker profile

Group matrix
------------
For each biomarker a 246 x 6 matrix is built whose cell (i, g) is the MEAN of
biomarker i over the subjects in diagnostic group g, computed from the actual
subject-level features. Group sizes are 20/20/20/16/20/20.

Standardisation (display only)
------------------------------
    zg[i, g] = (m[i, g] - mean_g m[i, :]) / sd_g(m[i, :]),  ddof = 1

i.e. each ROI is standardised ACROSS THE SIX GROUPS. Raw unstandardised group
means are preserved in csv/group_means_raw_<bm>.csv and are never overwritten.

Because every row is centred across the 6 groups, the six column profiles are
linearly constrained: their pairwise correlations are biased downward, with an
expected mean of -1/(k-1) = -0.2 for k = 6. This is a property of row-centring,
not a finding about the data, and it is why the colour scale is fixed and why
the dendrogram is checked against the unstandardised means and by bootstrap.

With k = 6 values and ddof = 1, |z| is bounded above by (k-1)/sqrt(k) = 2.041,
so a FIXED symmetric colour scale of +/-2.05 covers every attainable value and
is directly comparable across all four biomarkers.

Clustering
----------
    6 diagnostic groups  correlation distance + average linkage, on the
                         standardised 246-dimensional profile. 246 points per
                         group, so the correlation is well determined. The
                         question is whether two groups deviate in the SAME
                         PATTERN across regions.

    246 ROIs             Euclidean distance + average linkage, on the
                         standardised matrix. Since every row is standardised,
                         ||a-b||^2 = 2(k-1)(1-r), so for rows this is a
                         monotone transform of correlation distance; Euclidean
                         is used because it is stable over only 6 columns.

Nothing here is a test. Branch heights are distances, not p-values. The
dendrogram may order the groups differently from the clinical progression
CN -> SMC -> EMCI -> MCI -> LMCI -> AD; that reflects unsupervised similarity
under the chosen metric, not disease progression.
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
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist, squareform

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "derivatives", "model_ready")
V2 = os.path.join(ROOT, "derivatives", "INTRA_INTER_CLASS_HEATMAPS", "v2_cohort116")
OUT = os.path.join(V2, "diagnostic_group_hierarchical")
FIG = os.path.join(OUT, "figures")
CSVD = os.path.join(OUT, "csv")
LOGD = os.path.join(OUT, "logs")
LUT = os.path.join(ROOT, "derivatives", "fsfast", "FINAL_6GROUP_AUDIT",
                   "dataset_brainnetome_summary.csv")

STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
BIOMARKERS = ["mALFF", "mReHo", "DC_z", "FC"]
STAGE_COLORS = {
    "CN": "#1A5276", "SMC": "#2E86C1", "EMCI": "#7FB3D5",
    "MCI": "#F5B041", "LMCI": "#D35400", "AD": "#922B21",
}
N_ROI = 246
K = len(STAGES)
DDOF = 1
VMAX = 2.05                        # (k-1)/sqrt(k) = 2.041 for k = 6
GROUP_METRIC, GROUP_LINK = "correlation", "average"
ROI_METRIC, ROI_LINK = "euclidean", "average"
BOOT_N = 1000
BOOT_SEED = 20261014

LONG = {"mALFF": "Regional amplitude of low-frequency fluctuation (mALFF)",
        "mReHo": "Regional homogeneity (mReHo)",
        "DC_z": "Standardised signed degree centrality (DC_z)",
        "FC": "Unsigned mean Fisher-z connectivity strength (FC)"}

_log = []


def log(m=""):
    print(m, flush=True)
    _log.append(str(m))


def load():
    y = np.load(os.path.join(SRC, "y.npy"))
    subjects = np.load(os.path.join(SRC, "subjects.npy")).astype(str)
    meta = json.load(open(os.path.join(SRC, "metadata.json")))
    if meta["stage_order"] != STAGES:
        raise SystemExit(f"stage order mismatch: {meta['stage_order']}")
    if meta["stage_to_index"] != {s: i for i, s in enumerate(STAGES)}:
        raise SystemExit(f"stage index mapping mismatch: {meta['stage_to_index']}")
    if len(subjects) != 116 or len(set(subjects.tolist())) != 116:
        raise SystemExit(f"cohort is not 116 unique subjects ({len(subjects)})")
    counts = {s: int((y == i).sum()) for i, s in enumerate(STAGES)}
    if counts != meta["class_counts"]:
        raise SystemExit(f"class counts disagree with metadata: {counts}")
    log(f"cohort: {len(subjects)} acquisitions, {len(set(subjects.tolist()))} "
        f"unique subjects, one per subject")
    log(f"stage_to_index verified from source: {meta['stage_to_index']}")
    log(f"group sizes: {counts}")

    Xr = np.load(os.path.join(SRC, "X_regional.npy")).astype(np.float64)
    fc_u = np.load(os.path.join(SRC, "X_fc_roi_unsigned.npy")).astype(np.float64)
    if Xr.shape != (116, N_ROI, 3) or fc_u.shape != (116, N_ROI):
        raise SystemExit(f"feature shapes wrong: {Xr.shape}, {fc_u.shape}")
    if not (np.isfinite(Xr).all() and np.isfinite(fc_u).all()):
        raise SystemExit("non-finite values in the source features")
    log(f"X_regional {Xr.shape} channels {meta['regional_channels']}; "
        f"X_fc_roi_unsigned {fc_u.shape}; all finite")
    feats = {"mALFF": Xr[:, :, 0], "mReHo": Xr[:, :, 1],
             "DC_z": Xr[:, :, 2], "FC": fc_u}       # each [subjects, ROI]
    return y, subjects, feats, counts


def roi_labels():
    rows = list(csv.DictReader(open(LUT)))
    ids = [int(r["ROI_ID"]) for r in rows]
    names = [r["ROI_NAME"] for r in rows]
    if ids != list(range(1, N_ROI + 1)):
        raise SystemExit("LUT ROI_IDs are not 1..246 in order")
    return ids, names


def group_means(M, y):
    """M is [subjects, ROI]. Returns [ROI, 6] of per-group means."""
    out = np.empty((N_ROI, K))
    for g in range(K):
        out[:, g] = M[y == g].mean(axis=0)
    return out


def standardise_rows(G):
    """Standardise each ROI across the six groups. Returns (z, flat_mask)."""
    mu = G.mean(axis=1, keepdims=True)
    sd = G.std(axis=1, ddof=DDOF, keepdims=True)
    flat = (sd[:, 0] == 0) | ~np.isfinite(sd[:, 0])
    z = (G - mu) / np.where(flat[:, None], 1.0, sd)
    z[flat, :] = np.nan                     # never silently zeroed
    return z, flat


def link_groups(Z):
    D = pdist(Z.T, metric=GROUP_METRIC)
    L = hierarchy.linkage(D, method=GROUP_LINK)
    L = hierarchy.optimal_leaf_ordering(L, D)
    return L, D


def link_rois(Z):
    D = pdist(Z, metric=ROI_METRIC)
    L = hierarchy.linkage(D, method=ROI_LINK)
    L = hierarchy.optimal_leaf_ordering(L, D)
    return L, D


def clusters_of(L, labels):
    """Non-trivial leaf sets (bipartitions) encoded by a linkage matrix."""
    n = len(labels)
    members = {i: {labels[i]} for i in range(n)}
    out = []
    for r, row in enumerate(L):
        a, b = int(row[0]), int(row[1])
        s = members[a] | members[b]
        members[n + r] = s
        if 1 < len(s) < n:
            out.append(frozenset(s))
    return out


def bootstrap_stability(M, y, observed):
    """Resample subjects WITHIN each group, recompute group means, recluster.
    Reports how often each observed cluster reappears. This is a descriptive
    stability frequency, NOT a p-value and NOT a significance test."""
    rng = np.random.default_rng(BOOT_SEED)
    idx_by_g = [np.flatnonzero(y == g) for g in range(K)]
    tally = {c: 0 for c in observed}
    skipped = 0
    for _ in range(BOOT_N):
        G = np.empty((N_ROI, K))
        for g in range(K):
            pick = rng.choice(idx_by_g[g], size=idx_by_g[g].size, replace=True)
            G[:, g] = M[pick].mean(axis=0)
        z, flat = standardise_rows(G)
        if flat.all():
            skipped += 1
            continue
        z = z[~flat]
        L, _ = link_groups(z)
        for c in clusters_of(L, STAGES):
            if c in tally:
                tally[c] += 1
    return {c: tally[c] / max(BOOT_N - skipped, 1) for c in observed}, skipped


def gray(*a, **k):
    return "#8A8A8A"


def figure_a(bm, G, Z, flat, ids, names, Lg, Lr, stab):
    keep = ~flat
    zk = Z[keep]
    kept = np.flatnonzero(keep)
    roi_order = kept[hierarchy.leaves_list(Lr)]
    grp_order = hierarchy.leaves_list(Lg)
    disp = Z[np.ix_(roi_order, grp_order)]

    fig = plt.figure(figsize=(9.0, 15.2))
    gs = GridSpec(4, 3, figure=fig,
                  width_ratios=[0.17, 1.0, 0.055],
                  height_ratios=[0.11, 1.0, 0.07, 0.075],
                  wspace=0.52, hspace=0.045,
                  left=0.235, right=0.855, top=0.880, bottom=0.042)

    # group dendrogram
    ax_t = fig.add_subplot(gs[0, 1])
    hierarchy.dendrogram(Lg, ax=ax_t, no_labels=True, color_threshold=0,
                         link_color_func=gray)
    ax_t.set_xticks([])
    ax_t.set_yticks([])
    for s in ax_t.spines.values():
        s.set_visible(False)
    ax_t.set_title("diagnostic-group clustering "
                   f"({GROUP_METRIC} distance, {GROUP_LINK} linkage)",
                   fontsize=8.6, color="#374151", pad=3)

    # main heatmap
    ax = fig.add_subplot(gs[1, 1])
    cmap = matplotlib.colormaps["RdBu_r"].with_extremes(bad="#D1D5DB")
    im = ax.imshow(disp, aspect="auto", cmap=cmap, vmin=-VMAX, vmax=VMAX,
                   interpolation="nearest")
    ax.set_xticks(range(K))
    ax.set_xticklabels([STAGES[g] for g in grp_order], fontsize=11)
    for t, g in zip(ax.get_xticklabels(), grp_order):
        t.set_color(STAGE_COLORS[STAGES[g]])
        t.set_fontweight("bold")
    step = 10
    ticks = list(range(0, len(roi_order), step))
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{ids[roi_order[t]]:>3d} {names[roi_order[t]]}"
                        for t in ticks], fontsize=5.8, family="monospace")
    for j in range(1, K):
        ax.axvline(j - 0.5, color="white", lw=1.0)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_color("#9CA3AF")
    fig.text(0.017, 0.50,
             f"Brainnetome ROI  (all {len(roi_order)} rows plotted, every "
             f"{step}th labelled, clustered order)",
             rotation=90, va="center", ha="center", fontsize=9)

    # ROI dendrogram
    ax_l = fig.add_subplot(gs[1, 0])
    hierarchy.dendrogram(Lr, ax=ax_l, orientation="left", no_labels=True,
                         color_threshold=0, link_color_func=gray)
    ax_l.set_xticks([])
    ax_l.set_yticks([])
    ax_l.invert_yaxis()
    for s in ax_l.spines.values():
        s.set_visible(False)
    ax_l.set_title("ROI\nclustering", fontsize=7.8, color="#374151", pad=3)

    cax = fig.add_subplot(gs[1, 2])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("group-mean biomarker value, standardised across the six "
                 "diagnostic groups\n"
                 f"(blue = below this ROI's six-group mean, red = above; "
                 f"fixed scale +/-{VMAX}, identical for all biomarkers)",
                 fontsize=8.0)
    cb.ax.tick_params(labelsize=7.5)

    # caption + clinical order reminder
    ax_c = fig.add_subplot(gs[2, 1])
    ax_c.axis("off")
    ax_c.text(0.5, 0.72,
              "columns are ordered by the dendrogram above, NOT by disease stage",
              ha="center", va="center", fontsize=8.4, color="#111827",
              transform=ax_c.transAxes)
    ax_c.text(0.5, 0.22,
              f"clinical progression order:  {' -> '.join(STAGES)}",
              ha="center", va="center", fontsize=8.2, color="#374151",
              transform=ax_c.transAxes)

    ax_f = fig.add_subplot(gs[3, 1])
    ax_f.axis("off")
    best = sorted(stab.items(), key=lambda kv: -kv[1])[:3]
    txt = ";  ".join(f"{{{','.join(sorted(c, key=STAGES.index))}}} {v * 100:.0f}%"
                     for c, v in best)
    ax_f.text(0.5, 0.72,
              f"bootstrap stability of the branches ({BOOT_N} resamples of "
              f"subjects within groups):  {txt}",
              ha="center", va="center", fontsize=7.6, color="#374151",
              transform=ax_f.transAxes)
    ax_f.text(0.5, 0.26,
              "Descriptive only. Branch heights are distances, not p-values. "
              "No ROI x stage test in this project reaches q < 0.05.",
              ha="center", va="center", fontsize=7.8, color="#B91C1C",
              style="italic", transform=ax_f.transAxes)

    fig.suptitle("Hierarchical Clustering of Alzheimer's Diagnostic Groups\n"
                 f"{LONG[bm]}", fontsize=13.0, fontweight="bold", y=0.980)
    fig.text(0.5, 0.925,
             f"cells are the MEAN {bm} of the subjects in each diagnostic "
             f"group (n = 20/20/20/16/20/20), standardised across the six groups "
             f"per ROI (ddof = {DDOF})\n"
             f"groups clustered by {GROUP_METRIC} distance + {GROUP_LINK} "
             f"linkage over the 246-D profile; ROIs by {ROI_METRIC} distance + "
             f"{ROI_LINK} linkage",
             ha="center", va="top", fontsize=7.9, color="#374151", linespacing=1.6)
    if bm == "FC":
        fig.text(0.5, 0.898,
                 "FC per ROI = unsigned mean Fisher-z strength, mean_{j!=i} "
                 "|z_ij| (connectivity MAGNITUDE, not signed net connectivity)",
                 ha="center", va="top", fontsize=7.2, color="#6B21A8",
                 style="italic")

    png = os.path.join(FIG, f"group_hierarchical_{bm}.png")
    pdf = os.path.join(FIG, f"group_hierarchical_{bm}.pdf")
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    log(f"  wrote {os.path.relpath(png, ROOT)}")
    log(f"  wrote {os.path.relpath(pdf, ROOT)}")
    return roi_order, grp_order


def figure_b(store, ids, names):
    fig = plt.figure(figsize=(16.0, 10.4))
    gs = GridSpec(2, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.04],
                  height_ratios=[0.17, 1.0],
                  wspace=0.19, hspace=0.045,
                  left=0.045, right=0.935, top=0.815, bottom=0.165)
    cmap = matplotlib.colormaps["RdBu_r"].with_extremes(bad="#D1D5DB")
    for k, bm in enumerate(BIOMARKERS):
        Z, Lg, Lr, roi_order, grp_order = store[bm]
        ax_t = fig.add_subplot(gs[0, k])
        hierarchy.dendrogram(Lg, ax=ax_t, no_labels=True, color_threshold=0,
                             link_color_func=gray)
        ax_t.set_xticks([])
        ax_t.set_yticks([])
        for s in ax_t.spines.values():
            s.set_visible(False)
        ax_t.set_title(f"{chr(65 + k)}.  {bm}", fontsize=11.5,
                       fontweight="bold", pad=6)

        ax = fig.add_subplot(gs[1, k])
        im = ax.imshow(Z[np.ix_(roi_order, grp_order)], aspect="auto", cmap=cmap,
                       vmin=-VMAX, vmax=VMAX, interpolation="nearest")
        ax.set_xticks(range(K))
        ax.set_xticklabels([STAGES[g] for g in grp_order], fontsize=9,
                           rotation=45, ha="right")
        for t, g in zip(ax.get_xticklabels(), grp_order):
            t.set_color(STAGE_COLORS[STAGES[g]])
            t.set_fontweight("bold")
        ax.set_yticks([0, N_ROI - 1])
        ax.set_yticklabels(["1", "246"], fontsize=8)
        if k == 0:
            ax.set_ylabel("Brainnetome ROI (clustered per panel)", fontsize=9)
        for j in range(1, K):
            ax.axvline(j - 0.5, color="white", lw=0.8)

    cax = fig.add_subplot(gs[1, 4])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(f"group mean, standardised across the six groups  "
                 f"[fixed +/-{VMAX}, all panels]", fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)

    fig.legend(handles=[Patch(facecolor=STAGE_COLORS[s], label=s) for s in STAGES],
               loc="lower center", ncol=6, frameon=False, fontsize=10,
               bbox_to_anchor=(0.5, 0.012),
               title=f"clinical progression order: {' -> '.join(STAGES)}",
               title_fontsize=9)

    fig.suptitle("Diagnostic-Group Hierarchical Clustering Across All Four "
                 "Biomarkers", fontsize=15, fontweight="bold", y=0.975)
    fig.text(0.5, 0.925,
             "Each panel clusters the SIX DIAGNOSTIC GROUPS by their mean "
             "regional profile. Column order is the dendrogram's, not the "
             "clinical order; ROI rows are clustered independently per panel.\n"
             "The colour scale is fixed and comparable across panels. "
             "Descriptive only -- branch heights are distances, not p-values.",
             ha="center", va="top", fontsize=9, color="#374151", linespacing=1.6)

    png = os.path.join(FIG, "group_hierarchical_combined.png")
    pdf = os.path.join(FIG, "group_hierarchical_combined.pdf")
    fig.savefig(png, dpi=190)
    fig.savefig(pdf)
    plt.close(fig)
    log(f"wrote {os.path.relpath(png, ROOT)}")
    log(f"wrote {os.path.relpath(pdf, ROOT)}")


def fc_matrix_level(y, store):
    """Spec option 2: cluster groups on the FULL 246x246 mean connectivity
    pattern, and compare with the ROI-summary dendrogram."""
    X = np.load(os.path.join(SRC, "X_fc.npy"), mmap_mode="r")
    iu = np.triu_indices(N_ROI, k=1)
    E = np.empty((K, iu[0].size))
    for g in range(K):
        idx = np.flatnonzero(y == g)
        acc = np.zeros((N_ROI, N_ROI))
        for j in idx:
            acc += np.asarray(X[j], dtype=np.float64)
        E[g] = (acc / idx.size)[iu]
    del X
    log(f"  matrix-level: {E.shape[1]} unique edges per group "
        f"(upper triangle, diagonal excluded)")
    D = pdist(E, metric="correlation")
    L = hierarchy.linkage(D, method=GROUP_LINK)
    L = hierarchy.optimal_leaf_ordering(L, D)
    order_mat = [STAGES[g] for g in hierarchy.leaves_list(L)]
    Lg_roi = store["FC"][1]
    order_roi = [STAGES[g] for g in hierarchy.leaves_list(Lg_roi)]
    cl_mat = set(clusters_of(L, STAGES))
    cl_roi = set(clusters_of(Lg_roi, STAGES))
    agree = cl_mat & cl_roi
    log(f"  ROI-summary leaf order : {order_roi}")
    log(f"  matrix-level leaf order: {order_mat}")
    log(f"  shared branches: {len(agree)} of {len(cl_roi)} "
        f"({[sorted(c, key=STAGES.index) for c in agree]})")

    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.0))
    fig.subplots_adjust(left=0.07, right=0.97, top=0.76, bottom=0.14, wspace=0.22)
    for ax, LL, ttl, sub in (
            (axes[0], Lg_roi, "A.  ROI-level summary",
             "246-D profile of unsigned mean Fisher-z strength"),
            (axes[1], L, "B.  Matrix-level",
             f"{iu[0].size}-D vector of mean edge weights")):
        dn = hierarchy.dendrogram(LL, ax=ax, labels=STAGES, color_threshold=0,
                                  link_color_func=gray)
        ax.set_title(f"{ttl}\n{sub}", fontsize=10.5, fontweight="bold", pad=8)
        ax.set_ylabel("correlation distance", fontsize=9)
        ax.tick_params(labelsize=10)
        for t in ax.get_xticklabels():
            t.set_color(STAGE_COLORS[t.get_text()])
            t.set_fontweight("bold")
        for s in ax.spines.values():
            s.set_color("#9CA3AF")
    fig.suptitle("FC Representation Check: Does the Group Dendrogram Depend on "
                 "How Connectivity Is Summarised?",
                 fontsize=13, fontweight="bold", y=0.965)
    fig.text(0.5, 0.875,
             f"Both panels cluster the six diagnostic groups by {GROUP_METRIC} "
             f"distance + {GROUP_LINK} linkage; only the FC representation "
             f"differs.  Shared branches: {len(agree)} of {len(cl_roi)}.",
             ha="center", va="top", fontsize=8.8, color="#374151")
    fig.text(0.5, 0.025,
             "Signed connectivity lets positive and negative edges cancel; "
             "unsigned connectivity measures total connection magnitude. "
             "Neither is 'superior' -- they answer different questions.",
             ha="center", fontsize=8, color="#6B21A8", style="italic")
    png = os.path.join(FIG, "fc_representation_comparison.png")
    pdf = os.path.join(FIG, "fc_representation_comparison.pdf")
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    log(f"  wrote {os.path.relpath(png, ROOT)}")
    log(f"  wrote {os.path.relpath(pdf, ROOT)}")
    return order_roi, order_mat, len(agree), len(cl_roi)


def figure_c(y, subjects, feats):
    """Subject-level clustering, with the diagnostic label used ONLY as an
    annotation, never as clustering input. Quantified by a nearest-neighbour
    same-group rate against the chance baseline."""
    blocks = []
    for bm in ("mALFF", "mReHo", "DC_z"):
        M = feats[bm]
        blocks.append((M - M.mean(0)) / M.std(0, ddof=DDOF))
    Xc = np.hstack(blocks)                       # [116, 738]
    log(f"  concatenated regional feature space: {Xc.shape}")

    D = pdist(Xc, metric="correlation")
    L = hierarchy.linkage(D, method="average")
    L = hierarchy.optimal_leaf_ordering(L, D)
    order = hierarchy.leaves_list(L)
    coph, _ = hierarchy.cophenet(L, D)

    # nearest-neighbour same-group rate vs chance
    S = squareform(D)
    np.fill_diagonal(S, np.inf)
    nn = S.argmin(axis=1)
    rate = float((y[nn] == y).mean())
    counts = np.array([(y == g).sum() for g in range(K)], dtype=float)
    chance = float(((counts / counts.sum()) * ((counts - 1) / (counts.sum() - 1))).sum())
    per = {}
    for g in range(K):
        m = y == g
        per[STAGES[g]] = float((y[nn][m] == g).mean())
    log(f"  nearest-neighbour same-group rate: {rate:.3f} "
        f"(chance {chance:.3f}), cophenetic r = {coph:.3f}")
    log(f"  per group: " + ", ".join(f"{k} {v:.2f}" for k, v in per.items()))

    fig = plt.figure(figsize=(15.0, 8.2))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.0, 0.30],
                  height_ratios=[1.0, 0.135], wspace=0.13, hspace=0.045,
                  left=0.055, right=0.975, top=0.80, bottom=0.115)

    ax_d = fig.add_subplot(gs[0, 0])
    hierarchy.dendrogram(L, ax=ax_d, no_labels=True, color_threshold=0,
                         link_color_func=gray)
    ax_d.set_xticks([])
    ax_d.set_yticks([])
    for s in ax_d.spines.values():
        s.set_visible(False)
    ax_d.set_ylabel("correlation\ndistance", fontsize=8)

    ax = fig.add_subplot(gs[1, 0])
    strip = np.array([[matplotlib.colors.to_rgb(STAGE_COLORS[STAGES[y[j]]])
                       for j in order]])
    ax.imshow(np.repeat(strip, 8, axis=0), aspect="auto", interpolation="nearest")
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlabel(f"116 subjects, ordered by the dendrogram above "
                  f"(clustered on the {Xc.shape[1]}-dimensional standardised "
                  f"regional feature space; labels NOT used as input)",
                  fontsize=9.5)
    ax.set_ylabel("diagnostic\ngroup", rotation=0, ha="right", va="center",
                  fontsize=9, labelpad=12)

    ax_b = fig.add_subplot(gs[:, 1])
    xs = np.arange(K)
    ax_b.bar(xs, [per[s] for s in STAGES],
             color=[STAGE_COLORS[s] for s in STAGES], width=0.68)
    ax_b.axhline(chance, color="#B91C1C", lw=1.8, ls="--")
    ax_b.text(K - 0.4, chance + 0.012, f"chance {chance:.2f}", color="#B91C1C",
              fontsize=8.5, ha="right", fontweight="bold")
    ax_b.set_xticks(xs)
    ax_b.set_xticklabels(STAGES, fontsize=9, rotation=45, ha="right")
    for t, s in zip(ax_b.get_xticklabels(), STAGES):
        t.set_color(STAGE_COLORS[s])
        t.set_fontweight("bold")
    ax_b.set_ylim(0, max(0.45, max(per.values()) * 1.25))
    ax_b.set_ylabel("fraction whose nearest neighbour\nshares the diagnostic group",
                    fontsize=8.5)
    ax_b.set_title(f"overall {rate:.3f} vs chance {chance:.3f}",
                   fontsize=9.5, fontweight="bold", pad=6)
    ax_b.tick_params(labelsize=8)
    for s in ax_b.spines.values():
        s.set_color("#9CA3AF")

    fig.suptitle("Subject-Level Hierarchical Clustering, Annotated by "
                 "Diagnostic Group", fontsize=14.5, fontweight="bold", y=0.975)
    fig.text(0.5, 0.915,
             "Subjects are clustered on their own feature profiles; the "
             "diagnostic label is shown only as the colour strip and is never "
             f"an input. Cophenetic r = {coph:.3f}.\n"
             "If subjects of one group clustered together, the strip would show "
             "contiguous blocks of one colour.",
             ha="center", va="top", fontsize=9, color="#374151", linespacing=1.6)
    fig.text(0.5, 0.028,
             "The nearest-neighbour rate is a descriptive summary, NOT a "
             "classification accuracy and NOT evidence of diagnostic "
             "discrimination: no model has been trained and no held-out "
             "evaluation exists.",
             ha="center", fontsize=8.4, color="#B91C1C", style="italic")

    png = os.path.join(FIG, "subject_level_clustering.png")
    pdf = os.path.join(FIG, "subject_level_clustering.pdf")
    fig.savefig(png, dpi=195)
    fig.savefig(pdf)
    plt.close(fig)
    log(f"  wrote {os.path.relpath(png, ROOT)}")
    log(f"  wrote {os.path.relpath(pdf, ROOT)}")
    return rate, chance, per, float(coph), order


def main():
    for d in (FIG, CSVD, LOGD):
        os.makedirs(d, exist_ok=True)
    log("=" * 78)
    log("DIAGNOSTIC-GROUP HIERARCHICAL CLUSTERING")
    log("=" * 78)
    y, subjects, feats, counts = load()
    ids, names = roi_labels()
    log(f"fixed colour scale +/-{VMAX}; theoretical max |z| over k={K} groups "
        f"= (k-1)/sqrt(k) = {(K - 1) / np.sqrt(K):.4f}")
    log(f"groups: {GROUP_METRIC} + {GROUP_LINK}; ROIs: {ROI_METRIC} + {ROI_LINK}")

    store = {}
    rows_group_order, rows_roi_order, rows_stab, rows_cfg = [], [], [], []
    for bm in BIOMARKERS:
        log("")
        log(f"--- {bm} ---")
        G = group_means(feats[bm], y)
        Z, flat = standardise_rows(G)
        log(f"  group-mean matrix {G.shape}; zero-variance ROIs across groups: "
            f"{int(flat.sum())}")
        fin = Z[~flat]
        log(f"  standardised: mean {fin.mean():+.2e}, "
            f"range [{fin.min():+.3f}, {fin.max():+.3f}]")
        Lg, Dg = link_groups(Z[~flat])
        Lr, Dr = link_rois(Z[~flat])
        obs = clusters_of(Lg, STAGES)
        leaf = [STAGES[g] for g in hierarchy.leaves_list(Lg)]
        log(f"  group dendrogram leaf order: {leaf}")
        log(f"  group branches: "
            f"{[sorted(c, key=STAGES.index) for c in obs]}")
        cophg, _ = hierarchy.cophenet(Lg, Dg)
        log(f"  group cophenetic r = {cophg:.3f}; "
            f"pairwise correlation distance range "
            f"[{Dg.min():.3f}, {Dg.max():.3f}]")

        stab, skipped = bootstrap_stability(feats[bm], y, obs)
        log(f"  bootstrap stability ({BOOT_N} resamples, seed {BOOT_SEED}, "
            f"{skipped} skipped):")
        for c in sorted(obs, key=lambda c: -stab[c]):
            log(f"    {{{','.join(sorted(c, key=STAGES.index))}}}: "
                f"{stab[c] * 100:.1f}%")

        roi_order, grp_order = figure_a(bm, G, Z, flat, ids, names, Lg, Lr, stab)
        store[bm] = (Z, Lg, Lr, roi_order, grp_order)

        # CSVs
        p = os.path.join(CSVD, f"group_means_raw_{bm}.csv")
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ROI_ID", "ROI_NAME"] + STAGES)
            for i in range(N_ROI):
                w.writerow([ids[i], names[i]] + [repr(float(v)) for v in G[i]])
        log(f"  wrote {os.path.relpath(p, ROOT)}")
        p = os.path.join(CSVD, f"group_means_standardized_{bm}.csv")
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ROI_ID", "ROI_NAME", "zero_variance"] + STAGES)
            for i in range(N_ROI):
                w.writerow([ids[i], names[i], int(flat[i])]
                           + ["" if np.isnan(v) else repr(float(v)) for v in Z[i]])
        log(f"  wrote {os.path.relpath(p, ROOT)}")

        for pos, g in enumerate(grp_order):
            rows_group_order.append([bm, pos, STAGES[g], g,
                                     STAGES.index(STAGES[g]), counts[STAGES[g]]])
        for pos, i in enumerate(roi_order):
            rows_roi_order.append([bm, pos, int(i), ids[i], names[i]])
        for c, v in stab.items():
            rows_stab.append([bm, "{" + ",".join(sorted(c, key=STAGES.index)) + "}",
                              len(c), f"{v:.4f}", BOOT_N, skipped])
        rows_cfg.append([f"group_hierarchical_{bm}", "diagnostic groups (6)",
                         GROUP_METRIC, GROUP_LINK, "optimal_leaf_ordering",
                         "standardised group means (246-D)", VMAX])
        rows_cfg.append([f"group_hierarchical_{bm}", "ROIs (246)",
                         ROI_METRIC, ROI_LINK, "optimal_leaf_ordering",
                         "standardised group means (6-D)", VMAX])

    log("")
    log("--- combined overview ---")
    figure_b(store, ids, names)

    log("")
    log("--- FC representation comparison ---")
    o_roi, o_mat, n_agree, n_tot = fc_matrix_level(y, store)
    rows_cfg.append(["fc_representation_comparison", "diagnostic groups (6)",
                     "correlation", GROUP_LINK, "optimal_leaf_ordering",
                     "mean FC edge vector (30135-D, upper triangle)", ""])

    log("")
    log("--- subject-level clustering ---")
    rate, chance, per, cophs, sorder = figure_c(y, subjects, feats)
    rows_cfg.append(["subject_level_clustering", "subjects (116)",
                     "correlation", "average", "optimal_leaf_ordering",
                     "concatenated standardised mALFF+mReHo+DC_z (738-D)", ""])

    # shared CSVs
    p = os.path.join(CSVD, "group_dendrogram_leaf_order.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["biomarker", "display_column", "stage_label",
                    "source_stage_index", "clinical_order_index", "n_subjects"])
        w.writerows(rows_group_order)
    log(f"\nwrote {os.path.relpath(p, ROOT)}")

    p = os.path.join(CSVD, "roi_dendrogram_leaf_order.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["biomarker", "display_row", "original_index", "ROI_ID",
                    "ROI_NAME"])
        w.writerows(rows_roi_order)
    log(f"wrote {os.path.relpath(p, ROOT)}")

    p = os.path.join(CSVD, "group_cluster_stability_bootstrap.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["biomarker", "cluster", "cluster_size", "bootstrap_frequency",
                    "n_resamples", "n_skipped"])
        w.writerows(rows_stab)
    log(f"wrote {os.path.relpath(p, ROOT)}")

    p = os.path.join(CSVD, "clustering_configuration.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["figure", "objects_clustered", "distance_metric",
                    "linkage_method", "leaf_ordering", "input_space",
                    "colour_scale_abs_max"])
        w.writerows(rows_cfg)
    log(f"wrote {os.path.relpath(p, ROOT)}")

    p = os.path.join(CSVD, "fc_transformation_details.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["property", "value"])
        w.writerows([
            ["source_array", "derivatives/model_ready/X_fc.npy (116,246,246)"],
            ["source_normalisation", "Fisher-z, signed, unthresholded, diagonal zeroed"],
            ["roi_level_summary", "FCstr_i = mean_{j != i} |z_ij|"],
            ["roi_level_array", "derivatives/model_ready/X_fc_roi_unsigned.npy (116,246)"],
            ["signed_or_absolute", "absolute (unsigned magnitude)"],
            ["diagonal_handling", "zeroed at construction, excluded from the mean"],
            ["missing_edges", "none; matrix is unthresholded"],
            ["interpretation", "total connection magnitude, not signed net connectivity"],
            ["signed_alternative", "signed row mean = DC/245 exactly; r = +0.994 with DC_z"],
            ["correlation_with_DC_z", "-0.090 (different information, not better)"],
            ["matrix_level_space", f"upper triangle, {N_ROI * (N_ROI - 1) // 2} unique edges"],
            ["matrix_level_leaf_order", " -> ".join(o_mat)],
            ["roi_summary_leaf_order", " -> ".join(o_roi)],
            ["shared_branches", f"{n_agree} of {n_tot}"],
        ])
    log(f"wrote {os.path.relpath(p, ROOT)}")

    p = os.path.join(CSVD, "subject_neighbour_purity.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scope", "nearest_neighbour_same_group_rate", "chance_baseline",
                    "n"])
        w.writerow(["overall", f"{rate:.6f}", f"{chance:.6f}", 116])
        for s in STAGES:
            w.writerow([s, f"{per[s]:.6f}", f"{chance:.6f}", counts[s]])
    log(f"wrote {os.path.relpath(p, ROOT)}")

    lp = os.path.join(LOGD, "diagnostic_group_hierarchical.log")
    with open(lp, "w", encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")
    print(f"\nlog -> {os.path.relpath(lp, ROOT)}")


if __name__ == "__main__":
    main()
