"""
P-score heatmap with a diagnostic-group dendrogram on top and NO ROI dendrogram.

Figure contents
---------------
    rows      246 Brainnetome ROIs, all retained, sparse tick labels
    columns   6 diagnostic groups, ordered by the dendrogram above them
    cells     P-score = -log10(FDR-adjusted q)
    top       dendrogram over the six diagnostic groups
    right     colorbar with the q = 0.05 threshold marked
    bottom    clinical progression order, stated explicitly

What the heatmap and the tree each encode -- they are NOT the same quantity
--------------------------------------------------------------------------
Both are derived from the SAME P-score matrix, so this figure is internally
consistent: the dendrogram clusters the six groups by their 246-dimensional
P-SCORE PROFILES, i.e. by how similar their pattern of statistical evidence
is across regions. It is labelled "P-score-profile clustering" on the figure.

This is deliberately NOT the biomarker-profile dendrogram produced by
scripts/audit/diagnostic_group_hierarchical.py, which clusters the same six
groups by their mean regional biomarker ACTIVITY. Mixing a P-score heatmap
with a biomarker-derived tree would put two different quantities in one figure
without saying so.

Clustering
    distance  correlation, 1 - Pearson r, over the 246-D profile
    linkage   average (UPGMA)
    ordering  scipy optimal_leaf_ordering, deterministic

Source statistics are read, never recomputed:
    <bm>_roi_stage_qvalues.csv   Benjamini-Hochberg q over 246 x 6 = 1476
                                 tests per biomarker, two-sided Mann-Whitney U
                                 per ROI per stage, stage versus pooled rest

Verified at run time: 0 of 5904 cells reach q < 0.05 and the largest P-score
anywhere is 0.9278, below the 1.3010 threshold. The colour scale is fixed and
is never rescaled to make regions look significant.
"""
import csv
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V2 = os.path.join(ROOT, "derivatives", "INTRA_INTER_CLASS_HEATMAPS", "v2_cohort116")
OUT = os.path.join(V2, "diagnostic_group_hierarchical", "figures")
CSVD = os.path.join(V2, "diagnostic_group_hierarchical", "csv")
LUT = os.path.join(ROOT, "derivatives", "fsfast", "FINAL_6GROUP_AUDIT",
                   "dataset_brainnetome_summary.csv")

STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
BIOMARKERS = ["mALFF", "mReHo", "DC_z", "FC"]
STAGE_COLORS = {"CN": "#1A5276", "SMC": "#2E86C1", "EMCI": "#7FB3D5",
                "MCI": "#F5B041", "LMCI": "#D35400", "AD": "#922B21"}
N_ROI, ALPHA, FLOOR = 246, 0.05, 1e-12
THRESH = -np.log10(ALPHA)              # 1.3010
VMIN, VMAX, CMAP = 0.0, 1.40, "YlOrRd"
METRIC, LINKAGE = "correlation", "average"
LONG = {"mALFF": "mALFF", "mReHo": "mReHo", "DC_z": "DC_z",
        "FC": "Functional Connectivity (unsigned mean Fisher-z strength)"}

_log = []


def log(m=""):
    print(m, flush=True)
    _log.append(str(m))


def read_q(bm):
    rows = list(csv.DictReader(open(os.path.join(V2, f"{bm}_roi_stage_qvalues.csv"))))
    if len(rows) != N_ROI:
        raise SystemExit(f"{bm}: {len(rows)} rows, expected {N_ROI}")
    ids = [int(r["ROI_ID"]) for r in rows]
    if ids != list(range(1, N_ROI + 1)):
        raise SystemExit(f"{bm}: ROI_IDs are not 1..246 in order")
    Q = np.full((N_ROI, len(STAGES)), np.nan)
    missing = 0
    for i, r in enumerate(rows):
        for j, s in enumerate(STAGES):
            if r[s] in ("", "nan", "NaN", "NA"):
                missing += 1              # left as NaN, never zeroed
            else:
                Q[i, j] = float(r[s])
    return Q, missing


def roi_labels():
    rows = list(csv.DictReader(open(LUT)))
    return ([int(r["ROI_ID"]) for r in rows], [r["ROI_NAME"] for r in rows])


def make_figure(bm, P, Q, ids, names, order, L, coph, missing, outdir):
    disp = P[:, order]
    n_sig = int(np.nansum(Q < ALPHA))
    obs_max = float(np.nanmax(P))

    fig = plt.figure(figsize=(8.4, 14.6))
    gs = GridSpec(4, 2, figure=fig,
                  width_ratios=[1.0, 0.045],
                  height_ratios=[0.32, 1.0, 0.075, 0.105],
                  wspace=0.05, hspace=0.10,
                  left=0.195, right=0.865, top=0.872, bottom=0.045)

    # --- top: diagnostic-group dendrogram
    ax_t = fig.add_subplot(gs[0, 0])
    dn = hierarchy.dendrogram(L, ax=ax_t, no_labels=True, color_threshold=0,
                              link_color_func=lambda *a, **k: "#6B7280")
    ax_t.set_xticks([])
    ax_t.set_yticks([])
    for s in ax_t.spines.values():
        s.set_visible(False)

    # Branch labels are MERGE DISTANCES only. A merge height is a geometric
    # quantity; it is not a p-value and is never printed as one. No
    # significance value is assigned to a branch here.
    icoord, dcoord = np.array(dn["icoord"]), np.array(dn["dcoord"])
    MIN_SEP, BASE, STEP = 64.0, 9.0, 23.0
    placed = []
    for xm, h in sorted(((0.5 * (xs[1] + xs[2]), ys[1])
                         for xs, ys in zip(icoord, dcoord)), key=lambda t: t[0]):
        level = 0
        while any(abs(xm - px) < MIN_SEP and pl == level for px, pl in placed):
            level += 1
        placed.append((xm, level))
        dy = BASE + level * STEP
        ax_t.annotate(f"d = {h:.3f}", (xm, h), textcoords="offset points",
                      xytext=(0, dy), ha="center", fontsize=7.0,
                      color="#111827", fontweight="bold")
        if level:
            ax_t.annotate("", (xm, h), textcoords="offset points",
                          xytext=(0, dy - 2),
                          arrowprops=dict(arrowstyle="-", lw=0.5,
                                          color="#9CA3AF"))
    max_level = max((l for _, l in placed), default=0)
    ax_t.set_ylim(0, max(dcoord[:, 1].max(), 1e-9) * (1.10 + 0.10 * max_level))
    ax_t.set_title(
        f"diagnostic-group clustering by P-SCORE PROFILE  "
        f"({METRIC} distance, {LINKAGE} linkage, cophenetic r = {coph:.3f})"
        + chr(10)
        + "d = merge distance between significance profiles -- a distance, "
          "NOT a p-value",
        fontsize=7.8, color="#374151", pad=9)

    # --- main heatmap, no ROI dendrogram
    ax = fig.add_subplot(gs[1, 0])
    cmap = matplotlib.colormaps[CMAP].with_extremes(bad="#D1D5DB")
    im = ax.imshow(disp, aspect="auto", cmap=cmap, vmin=VMIN, vmax=VMAX,
                   interpolation="nearest")
    ax.set_xticks(range(len(STAGES)))
    ax.set_xticklabels([STAGES[g] for g in order], fontsize=11)
    for t, g in zip(ax.get_xticklabels(), order):
        t.set_color(STAGE_COLORS[STAGES[g]])
        t.set_fontweight("bold")
    ax.xaxis.set_ticks_position("bottom")
    ax.tick_params(axis="x", pad=6)
    step = 10
    ticks = list(range(0, N_ROI, step))
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{ids[t]:>3d} {names[t]}" for t in ticks],
                       fontsize=6.0, family="monospace")
    ax.set_ylabel(f"Brainnetome ROI 1-246  (all {N_ROI} rows shown, "
                  f"every {step}th labelled, atlas order)", fontsize=9)
    for j in range(1, len(STAGES)):
        ax.axvline(j - 0.5, color="white", lw=1.0)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_color("#9CA3AF")

    # --- colorbar with the threshold marked
    cax = fig.add_subplot(gs[1, 1])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(f"P-score = -log10(FDR-adjusted q)\n"
                 f"higher = stronger evidence  |  fixed scale {VMIN:.1f}-{VMAX:.1f}",
                 fontsize=8.4)
    cb.ax.tick_params(labelsize=7.5)
    cb.ax.axhline(THRESH, color="#111827", lw=1.6)
    cb.ax.text(0.5, THRESH, f"q = {ALPHA}\n({THRESH:.3f})",
               transform=cb.ax.get_yaxis_transform(), ha="center", va="bottom",
               fontsize=7.0, fontweight="bold", color="#111827",
               bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none",
                         alpha=0.9))

    # --- clinical progression order
    ax_c = fig.add_subplot(gs[2, 0])
    ax_c.axis("off")
    ax_c.text(0.5, 0.52, "columns ordered by the dendrogram above, NOT by "
              "disease stage", ha="center", va="center", fontsize=8.4,
              color="#111827", transform=ax_c.transAxes)
    ax_c.text(0.5, 0.06, f"clinical progression order:  "
              f"{'  ->  '.join(STAGES)}",
              ha="center", va="center", fontsize=9.0, color="#374151",
              fontweight="bold", transform=ax_c.transAxes)

    # --- verdict + caption
    ax_f = fig.add_subplot(gs[3, 0])
    ax_f.axis("off")
    head = (f"no ROI reaches q < {ALPHA}: 0 of {N_ROI * len(STAGES)} tests  "
            f"(highest P-score {obs_max:.3f} < {THRESH:.3f})" if n_sig == 0
            else f"{n_sig} of {N_ROI * len(STAGES)} cells reach q < {ALPHA}")
    ax_f.text(0.5, 0.74, head, ha="center", va="center", fontsize=9.0,
              color="#B91C1C" if n_sig == 0 else "#111827",
              fontweight="bold", transform=ax_f.transAxes)
    ax_f.text(0.5, 0.24,
              "Two-sided Mann-Whitney U per ROI per stage (stage vs pooled "
              "rest); Benjamini-Hochberg over 246 x 6 = 1476 tests.\n"
              "The tree measures similarity between STATISTICAL-SIGNIFICANCE "
              "PROFILES -- not between raw biomarker measurements, and not "
              "clinical progression.\n"
              "Because no ROI-stage test survives FDR correction, it is NOT "
              "evidence of significant diagnostic-group separation.",
              ha="center", va="center", fontsize=7.1, color="#374151",
              linespacing=1.5, transform=ax_f.transAxes)

    fig.suptitle(f"ROI-Level Stage-vs-Rest Significance\n{LONG[bm]}",
                 fontsize=13.0, fontweight="bold", y=0.972)
    if missing:
        fig.text(0.5, 0.012, f"{missing} missing q-values shown in grey, not "
                 "replaced with zero", ha="center", fontsize=7.2,
                 color="#B91C1C")

    png = os.path.join(outdir, f"pscore_grouptree_dist_{bm}.png")
    pdf = os.path.join(outdir, f"pscore_grouptree_dist_{bm}.pdf")
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    log(f"  wrote {os.path.relpath(png, ROOT)}")
    log(f"  wrote {os.path.relpath(pdf, ROOT)}")


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CSVD, exist_ok=True)
    ids, names = roi_labels()
    log("=" * 80)
    log("P-SCORE HEATMAP WITH DIAGNOSTIC-GROUP TREE (no ROI dendrogram)")
    log("=" * 80)
    log(f"fixed colour scale {VMIN}-{VMAX}; threshold {THRESH:.4f}")
    log(f"group tree: {METRIC} distance + {LINKAGE} linkage on the 246-D "
        f"P-score profile")

    total_sig, overall_max, order_rows = 0, 0.0, []

    def leaf_sets(L, labels):
        """Leaf membership of each internal node, in merge order."""
        n = len(labels)
        mem = {i: {labels[i]} for i in range(n)}
        out = []
        for r, row in enumerate(L):
            a, b = int(row[0]), int(row[1])
            A, B = mem[a], mem[b]
            mem[n + r] = A | B
            out.append((r, sorted(A, key=STAGES.index),
                        sorted(B, key=STAGES.index), float(row[2])))
        return out

    for bm in BIOMARKERS:
        log("")
        log(f"--- {bm} ---")
        Q, missing = read_q(bm)
        if missing:
            log(f"  {missing} missing q-values retained as NaN (never zeroed)")
        P = -np.log10(np.where(np.isnan(Q), np.nan, np.maximum(Q, FLOOR)))
        P[P == 0] = 0.0
        n_sig = int(np.nansum(Q < ALPHA))
        total_sig += n_sig
        overall_max = max(overall_max, float(np.nanmax(P)))
        log(f"  q: min {np.nanmin(Q):.6f}  P-score: max {np.nanmax(P):.4f}  "
            f"q < {ALPHA}: {n_sig} of {N_ROI * len(STAGES)}")

        cols = P.T                                    # [6, 246]
        if not np.isfinite(cols).all():
            raise SystemExit(f"{bm}: non-finite values reached clustering")
        D = pdist(cols, metric=METRIC)
        L = hierarchy.linkage(D, method=LINKAGE)
        L = hierarchy.optimal_leaf_ordering(L, D)
        order = hierarchy.leaves_list(L)
        coph = float(hierarchy.cophenet(L, D)[0])
        leaf = [STAGES[g] for g in order]
        log(f"  dendrogram leaf order: {leaf}   cophenetic r = {coph:.3f}")
        for pos, g in enumerate(order):
            order_rows.append([bm, "leaf", pos, STAGES[g], g,
                               STAGES.index(STAGES[g]), "", "", ""])
        for r, A, B, dist in leaf_sets(L, STAGES):
            order_rows.append([bm, "merge", r, "", "", "",
                               "{" + ",".join(A) + "}",
                               "{" + ",".join(B) + "}", f"{dist:.6f}"])

        make_figure(bm, P, Q, ids, names, order, L, coph, missing, OUT)

    log("")
    log(f"TOTAL q < {ALPHA} across all four biomarkers: {total_sig} of "
        f"{4 * N_ROI * len(STAGES)}")
    log(f"Largest P-score anywhere: {overall_max:.4f} (threshold {THRESH:.4f})")
    if total_sig == 0:
        log("No ROI survives FDR correction. Reported as observed; the colour "
            "scale was not rescaled and no value was altered.")

    p = os.path.join(CSVD, "pscore_group_tree_leaf_order_and_merges.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["biomarker", "record_type", "index", "stage_label",
                    "source_stage_index", "clinical_order_index",
                    "cluster_A", "cluster_B", "merge_distance"])
        w.writerows(order_rows)
    log(f"wrote {os.path.relpath(p, ROOT)}")
    with open(os.path.join(OUT, "pscore_group_tree.log"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")


if __name__ == "__main__":
    main()
