"""
P-score hierarchical clustering heatmaps: 246 ROIs x 6 stages, per biomarker.

    P-score = -log10(q)

where q is the Benjamini-Hochberg FDR-adjusted p-value for that ROI x stage
hypothesis, read from the stored ROI x stage analysis. Nothing is recomputed:
the q-values come from csv/roi_stage_qvalues_<bm>.csv, which validation has
already confirmed to be bit-identical to the source analysis output.

Correction family: 246 ROIs x 6 stages = 1476 tests, declared PER BIOMARKER.
Test: two-sided Mann-Whitney U, stage versus the pooled other five.
Threshold: q < 0.05, i.e. P-score > -log10(0.05) = 1.301.

Fixed colour scale
------------------
All four biomarker figures share ONE fixed scale, PSCORE_VMIN to PSCORE_VMAX,
so colours are directly comparable between biomarkers. The upper bound is set
above the q = 0.05 threshold so that the threshold is always visible on the
colourbar and a non-significant dataset cannot be made to look significant by
autoscaling. Each figure also carries a histogram of its own P-score
distribution, so fixing the scale does not hide where the values actually lie.

Column order
------------
The six stage columns are NEVER clustered. They stay in the established
clinical order CN -> SMC -> EMCI -> MCI -> LMCI -> AD, as encoded in the data.
Only the 246 ROI rows are clustered.

Clustering
----------
    objects    246 ROIs, each a 6-dimensional vector of P-scores
    distance   Euclidean
    linkage    Ward
    ordering   scipy optimal_leaf_ordering, deterministic

Euclidean distance is used here, rather than the correlation distance used for
the Z-score figures, because a P-score profile is a vector of magnitudes: two
ROIs are alike if their evidence is of similar strength at the same stages.
Correlation distance would discard magnitude entirely, calling an ROI with
q ~ 1 everywhere "identical" to one with strong evidence at the same stage,
and it is undefined for an ROI whose six q-values are constant. Ward linkage
is the standard pairing with Euclidean distance.

WHAT THE CLUSTERING MEANS -- and does not
-----------------------------------------
This groups ROIs with similar STATISTICAL SIGNIFICANCE PROFILES. It does not
group ROIs that are biologically similar, anatomically related, or functionally
connected. A tight cluster is not evidence of significance, and a row's
membership of a distinct cluster says nothing about whether its q-values pass
any threshold.

In this cohort NO ROI x stage test reaches q < 0.05 in any biomarker. Where
nothing is significant, the P-score surface being clustered is dominated by
sampling noise, and the resulting dendrogram should be read as a display
ordering rather than as a finding. See the effect-size alternative produced by
this script, and the README, for the defensible complementary view.
"""
import csv
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
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "derivatives", "model_ready")
V2 = os.path.join(ROOT, "derivatives", "INTRA_INTER_CLASS_HEATMAPS", "v2_cohort116")
BASE = os.path.join(V2, "zscore_and_pscore_heatmaps")
CSVD = os.path.join(BASE, "csv")
OUT = os.path.join(BASE, "pscore_hierarchical")

STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
BIOMARKERS = ["mALFF", "mReHo", "DC_z", "FC"]
STAGE_COLORS = {
    "CN": "#1A5276", "SMC": "#2E86C1", "EMCI": "#7FB3D5",
    "MCI": "#F5B041", "LMCI": "#D35400", "AD": "#922B21",
}
N_ROI = 246
ALPHA = 0.05
THRESH = -np.log10(ALPHA)          # 1.3010299956639813
FLOOR = 1e-12
FAMILY = N_ROI * len(STAGES)       # 1476

# Fixed, shared colour scale -- identical for all four biomarker figures.
PSCORE_VMIN = 0.0
PSCORE_VMAX = 1.40                 # > THRESH, so q = 0.05 is always on the bar
CMAP = "YlOrRd"

DISTANCE_METRIC = "euclidean"
LINKAGE_METHOD = "ward"

FC_DERIVATION = (
    "FC has no scalar per ROI, so the ROI-level statistic was computed on the "
    "UNSIGNED mean Fisher-z strength  FCstr_i = mean_{j!=i} |z_ij|  "
    "(source signed and unthresholded; diagonal zeroed before summarising; no "
    "missing edges). The signed row mean was rejected: it equals DC/245 and "
    "correlates with DC_z at r = +0.994. The unsigned strength correlates with "
    "DC_z at r = -0.090, so it is not a restatement of DC_z."
)

_log = []


def log(m=""):
    print(m, flush=True)
    _log.append(str(m))


def read_matrix(path, cols=STAGES):
    rows = list(csv.DictReader(open(path)))
    if len(rows) != N_ROI:
        raise SystemExit(f"{path}: {len(rows)} rows, expected {N_ROI}")
    ids = [int(r["ROI_ID"]) for r in rows]
    names = [r["ROI_NAME"] for r in rows]
    if ids != list(range(1, N_ROI + 1)):
        raise SystemExit(f"{path}: ROI_IDs are not 1..246 in order")
    M = np.full((N_ROI, len(cols)), np.nan)
    missing = []
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            v = r[c]
            if v in ("", "nan", "NaN", "NA", "None"):
                missing.append((ids[i], c))   # left as NaN, never zero
            else:
                M[i, j] = float(v)
    return M, ids, names, missing


def verify_cohort():
    y = np.load(os.path.join(SRC, "y.npy"))
    subjects = np.load(os.path.join(SRC, "subjects.npy")).astype(str)
    import json
    meta = json.load(open(os.path.join(SRC, "metadata.json")))
    if meta["stage_order"] != STAGES:
        raise SystemExit(f"stage order mismatch: {meta['stage_order']}")
    if len(subjects) != 116 or len(set(subjects.tolist())) != 116:
        raise SystemExit(f"cohort is not 116 unique subjects: {len(subjects)}")
    counts = {s: int((y == i).sum()) for i, s in enumerate(STAGES)}
    log(f"cohort verified: {len(subjects)} acquisitions, "
        f"{len(set(subjects.tolist()))} unique subjects, one each")
    log(f"stage order (from data): {meta['stage_order']}")
    log(f"class counts: {counts}")
    log(f"ROIs: {N_ROI} (Brainnetome)")


def cluster_rows(M, label):
    """Cluster the 246 ROI rows. Columns are never clustered."""
    if not np.isfinite(M).all():
        raise SystemExit(f"{label}: non-finite values reached clustering")
    D = pdist(M, metric=DISTANCE_METRIC)
    L = hierarchy.linkage(D, method=LINKAGE_METHOD)
    L = hierarchy.optimal_leaf_ordering(L, D)
    order = hierarchy.leaves_list(L)
    coph, _ = hierarchy.cophenet(L, D)
    log(f"  {label}: n={M.shape[0]}, distance={DISTANCE_METRIC}, "
        f"linkage={LINKAGE_METHOD}, cophenetic r={coph:.3f}")
    return order, L, float(coph)


def gray_links(*a, **k):
    return "#8A8A8A"


def make_figure(bm, P, Q, ids, names, order, L, coph, missing):
    disp = P[order, :]
    n_sig = int(np.nansum(Q < ALPHA))
    obs_max = float(np.nanmax(P))
    obs_med = float(np.nanmedian(P))

    fig = plt.figure(figsize=(9.6, 15.4))
    gs = GridSpec(4, 3, figure=fig,
                  width_ratios=[0.15, 1.0, 0.05],
                  height_ratios=[1.0, 0.05, 0.16, 0.055],
                  wspace=0.46, hspace=0.11,
                  left=0.215, right=0.86, top=0.885, bottom=0.048)

    # --- ROI dendrogram
    ax_d = fig.add_subplot(gs[0, 0])
    hierarchy.dendrogram(L, ax=ax_d, orientation="left", no_labels=True,
                         color_threshold=0, link_color_func=gray_links)
    ax_d.set_xticks([])
    ax_d.set_yticks([])
    ax_d.invert_yaxis()
    for s in ax_d.spines.values():
        s.set_visible(False)
    ax_d.set_title("ROI clustering", fontsize=8, color="#374151", pad=4)

    # --- main heatmap
    ax = fig.add_subplot(gs[0, 1])
    cmap = matplotlib.colormaps[CMAP].with_extremes(bad="#D1D5DB")
    im = ax.imshow(disp, aspect="auto", cmap=cmap,
                   vmin=PSCORE_VMIN, vmax=PSCORE_VMAX, interpolation="nearest")
    ax.set_xticks(range(len(STAGES)))
    ax.set_xticklabels(STAGES, fontsize=10.5)
    for t, s in zip(ax.get_xticklabels(), STAGES):
        t.set_color(STAGE_COLORS[s])
        t.set_fontweight("bold")
    step = 10
    ticks = list(range(0, N_ROI, step))
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{ids[order[t]]:>3d} {names[order[t]]}" for t in ticks],
                       fontsize=5.8, family="monospace")
    fig.text(0.018, 0.55,
             f"Brainnetome ROI  (all {N_ROI} rows plotted, every {step}th "
             f"labelled, clustered order)",
             rotation=90, va="center", ha="center", fontsize=9)
    for j in range(1, len(STAGES)):
        ax.axvline(j - 0.5, color="white", lw=0.9)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_color("#9CA3AF")

    ys, xs = np.nonzero(np.nan_to_num(Q[order, :], nan=1.0) < ALPHA)
    if len(ys):
        ax.scatter(xs, ys, marker="*", s=26, c="#111827", linewidths=0)

    # --- colourbar, fixed scale, threshold marked
    cax = fig.add_subplot(gs[0, 2])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("P-score = -log10(FDR-adjusted q)\n"
                 f"fixed scale {PSCORE_VMIN:.1f}-{PSCORE_VMAX:.1f}, identical "
                 f"for all four biomarkers",
                 fontsize=8.2)
    cb.ax.tick_params(labelsize=7.5)
    cb.ax.axhline(THRESH, color="#111827", lw=1.6)
    cb.ax.text(0.5, THRESH, f"q = {ALPHA}", transform=cb.ax.get_yaxis_transform(),
               ha="center", va="bottom", fontsize=7.4, fontweight="bold",
               color="#111827",
               bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none",
                         alpha=0.9))

    # --- stage axis caption
    ax_c = fig.add_subplot(gs[1, 1])
    ax_c.axis("off")
    ax_c.text(0.5, 0.45,
              "disease stage, fixed clinical order -- columns are NOT clustered "
              "(one-vs-rest: this stage versus the pooled other five)",
              ha="center", va="center", fontsize=8.4, color="#374151",
              transform=ax_c.transAxes)

    # --- observed distribution, so the fixed scale hides nothing
    ax_h = fig.add_subplot(gs[2, 1])
    vals = P[np.isfinite(P)]
    ax_h.hist(vals, bins=60, range=(PSCORE_VMIN, PSCORE_VMAX),
              color="#E8873A", edgecolor="#9A5520", linewidth=0.3)
    ax_h.axvline(THRESH, color="#B91C1C", lw=1.6)
    ax_h.text(THRESH, ax_h.get_ylim()[1] * 0.93, f"  q = {ALPHA}", color="#B91C1C",
              fontsize=8, fontweight="bold", va="top")
    ax_h.set_xlim(PSCORE_VMIN, PSCORE_VMAX)
    ax_h.set_xlabel("P-score", fontsize=8.5)
    ax_h.set_ylabel("ROI x stage\ncells", fontsize=8)
    ax_h.tick_params(labelsize=7.5)
    for s in ax_h.spines.values():
        s.set_color("#9CA3AF")
    ax_h.set_title(f"observed P-score distribution across all {FAMILY} cells  "
                   f"(max {obs_max:.3f}, median {obs_med:.3f})",
                   fontsize=8.2, color="#374151", pad=4)

    # --- verdict
    ax_f = fig.add_subplot(gs[3, 1])
    ax_f.axis("off")
    if n_sig:
        head, hc = (f"{n_sig} of {FAMILY} ROI x stage cells reach q < {ALPHA} "
                    f"(marked with a star)", "#111827")
    else:
        head, hc = (f"no ROI passes FDR correction: 0 of {FAMILY} cells reach "
                    f"q < {ALPHA}  (highest P-score {obs_max:.3f} < {THRESH:.3f})",
                    "#B91C1C")
    ax_f.text(0.5, 0.5, head, ha="center", va="center", fontsize=9.3,
              color=hc, fontweight="bold", transform=ax_f.transAxes)

    nice = "Functional Connectivity" if bm == "FC" else bm
    fig.suptitle(f"P-Score Hierarchical Clustering: ROI-Level Stage-vs-Rest "
                 f"Significance\n{nice}",
                 fontsize=13.2, fontweight="bold", y=0.983)
    sub = (f"P-score = -log10(q), Benjamini-Hochberg over "
           f"{N_ROI} x {len(STAGES)} = {FAMILY} tests per biomarker; "
           f"two-sided Mann-Whitney U per ROI per stage\n"
           f"rows clustered by {DISTANCE_METRIC} distance + {LINKAGE_METHOD} "
           f"linkage on the 6-D significance profile (cophenetic r = {coph:.3f}) "
           f"-- this groups ROIs by SIMILARITY OF EVIDENCE, not by biology")
    fig.text(0.5, 0.932, sub, ha="center", va="top", fontsize=7.9,
             color="#374151", linespacing=1.6)
    if bm == "FC":
        fig.text(0.5, 0.902, "FC ROI-level statistic: unsigned mean Fisher-z "
                 "strength (see README)", ha="center", va="top", fontsize=7.2,
                 color="#6B21A8", style="italic")
    if missing:
        fig.text(0.5, 0.016, f"{len(missing)} missing q-values shown in grey, "
                 "not substituted with zero", ha="center", fontsize=7.5,
                 color="#B91C1C")

    png = os.path.join(OUT, f"pscore_hierarchical_{bm}.png")
    pdf = os.path.join(OUT, f"pscore_hierarchical_{bm}.pdf")
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    log(f"  wrote {os.path.relpath(png, ROOT)}")
    log(f"  wrote {os.path.relpath(pdf, ROOT)}")


def combined_figure(store, ids, names):
    fig = plt.figure(figsize=(15.6, 11.0))
    gs = GridSpec(1, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.035],
                  wspace=0.16, left=0.05, right=0.93, top=0.835, bottom=0.085)
    cmap = matplotlib.colormaps[CMAP].with_extremes(bad="#D1D5DB")
    for k, bm in enumerate(BIOMARKERS):
        P, Q, order, coph = store[bm]
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(P[order, :], aspect="auto", cmap=cmap,
                       vmin=PSCORE_VMIN, vmax=PSCORE_VMAX, interpolation="nearest")
        ax.set_xticks(range(len(STAGES)))
        ax.set_xticklabels(STAGES, fontsize=8.5, rotation=45, ha="right")
        for t, s in zip(ax.get_xticklabels(), STAGES):
            t.set_color(STAGE_COLORS[s])
            t.set_fontweight("bold")
        ax.set_yticks([0, N_ROI - 1])
        ax.set_yticklabels(["1", "246"], fontsize=8)
        if k == 0:
            ax.set_ylabel("Brainnetome ROI (clustered per panel)", fontsize=9)
        nice = "FC" if bm == "FC" else bm
        ax.set_title(f"{chr(65 + k)}.  {nice}\nmax P-score "
                     f"{float(np.nanmax(P)):.3f}", fontsize=10,
                     fontweight="bold", pad=8)
        for j in range(1, len(STAGES)):
            ax.axvline(j - 0.5, color="white", lw=0.7)
    cax = fig.add_subplot(gs[0, 4])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("P-score = -log10(FDR-adjusted q)   [fixed scale, all panels]",
                 fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)
    cb.ax.axhline(THRESH, color="#111827", lw=1.6)
    cb.ax.text(0.5, THRESH, f"q = {ALPHA}", transform=cb.ax.get_yaxis_transform(),
               ha="center", va="bottom", fontsize=7.4, fontweight="bold",
               bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none",
                         alpha=0.9))

    fig.suptitle("P-Score Hierarchical Clustering Across Biomarkers  "
                 "(ROI-level stage-vs-rest significance)",
                 fontsize=14.5, fontweight="bold", y=0.975)
    fig.text(0.5, 0.925,
             "Each panel is clustered independently, so row positions are NOT "
             "comparable between panels; the colour scale IS fixed and comparable.\n"
             "Stage columns are never clustered. Rows are ordered by Euclidean/Ward "
             "clustering of the 6-D P-score profile.",
             ha="center", va="top", fontsize=8.6, color="#374151", linespacing=1.6)
    fig.text(0.5, 0.032,
             f"No ROI x stage test reaches q < {ALPHA} in any biomarker "
             f"(0 of {FAMILY * 4} cells). Clustering here groups ROIs by "
             f"similarity of statistical evidence, not by biology.",
             ha="center", fontsize=9.3, color="#B91C1C", fontweight="bold")

    png = os.path.join(OUT, "pscore_hierarchical_combined.png")
    pdf = os.path.join(OUT, "pscore_hierarchical_combined.pdf")
    fig.savefig(png, dpi=190)
    fig.savefig(pdf)
    plt.close(fig)
    log(f"wrote {os.path.relpath(png, ROOT)}")
    log(f"wrote {os.path.relpath(pdf, ROOT)}")


def effectsize_alternative(ids, names):
    """The defensible complement: cluster on effect size, which is a measured
    magnitude with a direction, rather than on a significance score that is
    uniformly null in this cohort."""
    fig = plt.figure(figsize=(15.6, 11.0))
    gs = GridSpec(1, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.035],
                  wspace=0.16, left=0.05, right=0.93, top=0.835, bottom=0.085)
    gmax = 0.0
    mats = {}
    for bm in BIOMARKERS:
        G, _, _, _ = read_matrix(os.path.join(
            CSVD, f"roi_stage_effectsize_hedges_g_{bm}.csv"))
        mats[bm] = G
        gmax = max(gmax, float(np.nanmax(np.abs(G))))
    vmax = float(np.ceil(gmax * 10) / 10)
    log(f"  effect-size alternative: shared symmetric scale +/-{vmax:.1f} "
        f"(observed max |g| = {gmax:.3f})")

    for k, bm in enumerate(BIOMARKERS):
        G = mats[bm]
        order, L, coph = cluster_rows(G, f"{bm} (Hedges g)")
        ax = fig.add_subplot(gs[0, k])
        im = ax.imshow(G[order, :], aspect="auto", cmap="RdBu_r",
                       vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_xticks(range(len(STAGES)))
        ax.set_xticklabels(STAGES, fontsize=8.5, rotation=45, ha="right")
        for t, s in zip(ax.get_xticklabels(), STAGES):
            t.set_color(STAGE_COLORS[s])
            t.set_fontweight("bold")
        ax.set_yticks([0, N_ROI - 1])
        ax.set_yticklabels(["1", "246"], fontsize=8)
        if k == 0:
            ax.set_ylabel("Brainnetome ROI (clustered per panel)", fontsize=9)
        nice = "FC" if bm == "FC" else bm
        ax.set_title(f"{chr(65 + k)}.  {nice}\ncophenetic r = {coph:.3f}",
                     fontsize=10, fontweight="bold", pad=8)
        for j in range(1, len(STAGES)):
            ax.axvline(j - 0.5, color="white", lw=0.7)
    cax = fig.add_subplot(gs[0, 4])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Hedges' g, stage minus rest   [shared symmetric scale]",
                 fontsize=8.5)
    cb.ax.tick_params(labelsize=7.5)

    fig.suptitle("Alternative View: Hierarchical Clustering of EFFECT SIZE "
                 "(not significance)", fontsize=14.5, fontweight="bold", y=0.975)
    fig.text(0.5, 0.925,
             "Complement to the P-score figures. Effect size is a measured, "
             "signed magnitude that exists whether or not a test is significant, so "
             "clustering it groups ROIs by\nthe direction and size of their "
             "stage-vs-rest difference. Blue = stage below the pooled rest, "
             "red = above. Columns are never clustered.",
             ha="center", va="top", fontsize=8.6, color="#374151", linespacing=1.6)
    fig.text(0.5, 0.032,
             "Effect size is NOT significance: none of these differences survives "
             "FDR correction at this sample size. Shown for structure, not for "
             "inference.",
             ha="center", fontsize=9.3, color="#B91C1C", fontweight="bold")

    png = os.path.join(OUT, "effectsize_hierarchical_alternative.png")
    pdf = os.path.join(OUT, "effectsize_hierarchical_alternative.pdf")
    fig.savefig(png, dpi=190)
    fig.savefig(pdf)
    plt.close(fig)
    log(f"wrote {os.path.relpath(png, ROOT)}")
    log(f"wrote {os.path.relpath(pdf, ROOT)}")


def main():
    os.makedirs(OUT, exist_ok=True)
    log("=" * 78)
    log("P-SCORE HIERARCHICAL CLUSTERING HEATMAPS")
    log("=" * 78)
    verify_cohort()
    log(f"P-score = -log10(q); floor {FLOOR:g} applied only if q == 0")
    log(f"fixed colour scale: {PSCORE_VMIN} to {PSCORE_VMAX} (threshold "
        f"{THRESH:.4f}), identical for all four biomarkers")
    log(f"columns fixed in clinical order, never clustered: {STAGES}")

    store = {}
    order_rows = []
    ids = names = None
    total_missing = 0
    for bm in BIOMARKERS:
        log("")
        log(f"--- {bm} ---")
        Q, ids, names, missing = read_matrix(
            os.path.join(CSVD, f"roi_stage_qvalues_{bm}.csv"))
        total_missing += len(missing)
        if missing:
            log(f"  {len(missing)} missing q-values -- left as NaN, never zeroed")
        if (Q == 0).any():
            log(f"  {(Q == 0).sum()} q-values are exactly 0; floor {FLOOR:g} applied")
        P = -np.log10(np.where(np.isnan(Q), np.nan, np.maximum(Q, FLOOR)))
        # q == 1 gives -log10(1) = -0.0; normalise the sign of zero
        P[P == 0] = 0.0
        log(f"  q: min {np.nanmin(Q):.6f}, max {np.nanmax(Q):.6f}")
        log(f"  P-score: min {np.nanmin(P):.4f}, max {np.nanmax(P):.4f}, "
            f"median {np.nanmedian(P):.4f}")
        log(f"  cells with q < {ALPHA}: {int(np.nansum(Q < ALPHA))} of {FAMILY}")

        order, L, coph = cluster_rows(P, bm)
        store[bm] = (P, Q, order, coph)
        make_figure(bm, P, Q, ids, names, order, L, coph, missing)

        p = os.path.join(CSVD, f"pscore_matrix_{bm}.csv")
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["ROI_ID", "ROI_NAME"] + STAGES)
            for i in range(N_ROI):
                w.writerow([ids[i], names[i]]
                           + ["" if np.isnan(v) else repr(float(v)) for v in P[i]])
        log(f"  wrote {os.path.relpath(p, ROOT)}")

        for pos, i in enumerate(order):
            order_rows.append([bm, pos, int(i), ids[i], names[i]])

    p = os.path.join(CSVD, "pscore_clustering_order.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["biomarker", "display_row", "dendrogram_leaf_index",
                    "original_index", "ROI_ID", "ROI_NAME"])
        for bm, pos, i, rid, rname in order_rows:
            w.writerow([bm, pos, pos, i, rid, rname])
    log("")
    log(f"wrote {os.path.relpath(p, ROOT)}  ({len(order_rows)} rows = "
        f"4 biomarkers x {N_ROI} ROIs)")

    log("")
    log("--- combined figure ---")
    combined_figure(store, ids, names)
    log("")
    log("--- effect-size alternative ---")
    effectsize_alternative(ids, names)

    log("")
    log(f"missing q-values across all biomarkers: {total_missing}")
    nsig = sum(int(np.nansum(store[b][1] < ALPHA)) for b in BIOMARKERS)
    log(f"ROI x stage cells reaching q < {ALPHA}: {nsig} of {FAMILY * 4}")
    if nsig == 0:
        log("No ROI passes FDR correction in any biomarker. Reported as observed; "
            "no value was altered and the colour scale was not rescaled to "
            "exaggerate the null surface.")

    lp = os.path.join(BASE, "logs", "pscore_hierarchical.log")
    with open(lp, "w", encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")
    print(f"\nlog -> {os.path.relpath(lp, ROOT)}")


if __name__ == "__main__":
    main()
