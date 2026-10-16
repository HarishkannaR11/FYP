"""
ONE combined four-panel figure: P-scores for mALFF, mReHo, DC_z and FC.

Mirrors the layout of group_hierarchical_combined.png, but every panel shows
P-scores instead of standardised biomarker means:

    per panel   top dendrogram over the six diagnostic groups, clustered on
                their 246-D P-SCORE profiles; below it the 246 x 6 P-score
                heatmap, columns in that dendrogram's leaf order
    no panel    has a left-side ROI dendrogram
    shared      one colorbar and one stage legend for the whole figure

    P-score = -log10(FDR-adjusted q)

Colour scale
------------
Sequential YlOrRd on a FIXED 0.0-1.40 range, not the diverging red-blue scale
used for Z-scores. P-scores are non-negative and have no meaningful midpoint,
so a diverging map centred on zero would imply a sign distinction that does
not exist and would exaggerate small values. The upper bound sits above the
q = 0.05 threshold (1.3010) so the threshold is always visible on the bar --
and visibly unreached.

Clustering, per biomarker independently
    distance  correlation, 1 - Pearson r, over the 246-D P-score profile
    linkage   average (UPGMA)
    ordering  scipy optimal_leaf_ordering, deterministic

Clustering uses P-score profiles only, never raw biomarker values. Branch
heights are merge distances, not p-values, so no numerical branch labels are
drawn and the caption says so.

Source q-values are read, never recomputed.
"""
import csv
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V2 = os.path.join(ROOT, "derivatives", "INTRA_INTER_CLASS_HEATMAPS", "v2_cohort116")
BASE = os.path.join(V2, "diagnostic_group_hierarchical")
OUT, CSVD = os.path.join(BASE, "figures"), os.path.join(BASE, "csv")

STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
BIOMARKERS = ["mALFF", "mReHo", "DC_z", "FC"]
STAGE_COLORS = {"CN": "#1A5276", "SMC": "#2E86C1", "EMCI": "#7FB3D5",
                "MCI": "#F5B041", "LMCI": "#D35400", "AD": "#922B21"}
N_ROI, ALPHA, FLOOR = 246, 0.05, 1e-12
THRESH = -np.log10(ALPHA)
VMIN, VMAX, CMAP = 0.0, 1.40, "YlOrRd"
METRIC, LINKAGE = "correlation", "average"

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
    hdr = list(csv.reader(open(os.path.join(V2, f"{bm}_roi_stage_qvalues.csv"))))[0]
    if hdr[2:] != STAGES:
        raise SystemExit(f"{bm}: column labels {hdr[2:]} != {STAGES}")
    Q = np.full((N_ROI, len(STAGES)), np.nan)
    missing = 0
    for i, r in enumerate(rows):
        for j, s in enumerate(STAGES):
            if r[s] in ("", "nan", "NaN", "NA"):
                missing += 1
            else:
                Q[i, j] = float(r[s])
    return Q, missing


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CSVD, exist_ok=True)
    log("=" * 80)
    log("COMBINED FOUR-PANEL P-SCORE FIGURE")
    log("=" * 80)
    log(f"colour: sequential {CMAP}, fixed {VMIN}-{VMAX}; threshold {THRESH:.4f}")
    log(f"clustering per panel: {METRIC} distance + {LINKAGE} linkage on the "
        f"246-D P-score profile")

    panels, rows_csv = {}, []
    total_sig, overall_max, total_missing = 0, 0.0, 0
    for bm in BIOMARKERS:
        Q, missing = read_q(bm)
        total_missing += missing
        P = -np.log10(np.where(np.isnan(Q), np.nan, np.maximum(Q, FLOOR)))
        P[P == 0] = 0.0
        n_sig = int(np.nansum(Q < ALPHA))
        total_sig += n_sig
        overall_max = max(overall_max, float(np.nanmax(P)))

        D = pdist(P.T, metric=METRIC)
        L = hierarchy.linkage(D, method=LINKAGE)
        L = hierarchy.optimal_leaf_ordering(L, D)
        order = hierarchy.leaves_list(L)
        coph = float(hierarchy.cophenet(L, D)[0])
        panels[bm] = (P, Q, L, order, coph)
        log(f"  {bm:6s} shape {P.shape}  max P {np.nanmax(P):.4f}  "
            f"q<{ALPHA}: {n_sig}  missing {missing}  "
            f"leaves {[STAGES[g] for g in order]}  coph {coph:.3f}")
        for pos, g in enumerate(order):
            rows_csv.append([bm, pos, STAGES[g], g, STAGES.index(STAGES[g]),
                             f"{float(P[:, g].max()):.6f}",
                             int((P[:, g] >= THRESH).sum())])

    # ------------------------------------------------------------- figure
    fig = plt.figure(figsize=(17.0, 11.6))
    gs = GridSpec(2, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.038],
                  height_ratios=[0.17, 1.0], wspace=0.17, hspace=0.045,
                  left=0.048, right=0.935, top=0.830, bottom=0.185)
    cmap = matplotlib.colormaps[CMAP].with_extremes(bad="#D1D5DB")

    for k, bm in enumerate(BIOMARKERS):
        P, Q, L, order, coph = panels[bm]

        ax_t = fig.add_subplot(gs[0, k])
        hierarchy.dendrogram(L, ax=ax_t, no_labels=True, color_threshold=0,
                             link_color_func=lambda *a, **kw: "#6B7280")
        ax_t.set_xticks([])
        ax_t.set_yticks([])
        for sp in ax_t.spines.values():
            sp.set_visible(False)
        nice = "Functional Connectivity" if bm == "FC" else bm
        ax_t.set_title(f"{chr(65 + k)}.  {nice}\n"
                       f"max P-score {float(np.nanmax(P)):.3f}   "
                       f"cophenetic r = {coph:.3f}",
                       fontsize=10.5, fontweight="bold", pad=7)

        ax = fig.add_subplot(gs[1, k])
        im = ax.imshow(P[:, order], aspect="auto", cmap=cmap,
                       vmin=VMIN, vmax=VMAX, interpolation="nearest")
        ax.set_xticks(range(len(STAGES)))
        ax.set_xticklabels([STAGES[g] for g in order], fontsize=9.5,
                           rotation=45, ha="right")
        for t, g in zip(ax.get_xticklabels(), order):
            t.set_color(STAGE_COLORS[STAGES[g]])
            t.set_fontweight("bold")
        ax.set_yticks([0, N_ROI - 1])
        ax.set_yticklabels(["1", "246"], fontsize=8)
        if k == 0:
            ax.set_ylabel("Brainnetome ROI 1-246  (all rows shown)", fontsize=9)
        for j in range(1, len(STAGES)):
            ax.axvline(j - 0.5, color="white", lw=0.8)
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
            sp.set_color("#9CA3AF")

    cax = fig.add_subplot(gs[1, 4])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("P-score = -log10(FDR-adjusted q-value)\n"
                 f"higher = stronger evidence  |  fixed scale "
                 f"{VMIN:.1f}-{VMAX:.1f}, identical in all panels", fontsize=9)
    cb.ax.tick_params(labelsize=8)
    cb.ax.axhline(THRESH, color="#111827", lw=1.7)
    cb.ax.text(0.5, THRESH, f"q = {ALPHA}\n({THRESH:.3f})",
               transform=cb.ax.get_yaxis_transform(), ha="center", va="bottom",
               fontsize=7.6, fontweight="bold",
               bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none",
                         alpha=0.9))

    fig.legend(handles=[Patch(facecolor=STAGE_COLORS[s], label=s) for s in STAGES],
               loc="lower center", ncol=6, frameon=False, fontsize=10.5,
               bbox_to_anchor=(0.5, 0.058),
               title=f"clinical progression order:  {'  ->  '.join(STAGES)}",
               title_fontsize=9.5)

    fig.suptitle("Diagnostic-Group Hierarchical Clustering Across All Four "
                 "Biomarkers  --  P-Scores", fontsize=15.5,
                 fontweight="bold", y=0.975)
    fig.text(0.5, 0.918,
             "Each panel clusters the SIX DIAGNOSTIC GROUPS by their 246-D "
             "P-SCORE profiles (correlation distance, average linkage); "
             "columns follow that panel's leaf order, NOT the clinical order.\n"
             "ROI rows are in atlas order and are NOT clustered. Branch heights "
             "are merge distances, not p-values, so no numerical branch labels "
             "are drawn.",
             ha="center", va="top", fontsize=9, color="#374151",
             linespacing=1.6)
    fig.text(0.5, 0.030,
             f"No ROI-stage cell reaches the significance threshold in any "
             f"panel: 0 of {4 * N_ROI * len(STAGES):,} tests at q < {ALPHA}; "
             f"the largest P-score anywhere is {overall_max:.3f} < {THRESH:.3f}.",
             ha="center", fontsize=10.5, color="#B91C1C", fontweight="bold")

    png = os.path.join(OUT, "pscore_hierarchical_combined_4panel.png")
    pdf = os.path.join(OUT, "pscore_hierarchical_combined_4panel.pdf")
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    log("")
    log(f"wrote {os.path.relpath(png, ROOT)}")
    log(f"wrote {os.path.relpath(pdf, ROOT)}")

    p = os.path.join(CSVD, "pscore_combined_4panel_leaf_order.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["biomarker", "display_column", "stage_label",
                    "source_stage_index", "clinical_order_index",
                    "max_P_score_in_column", "n_ROIs_at_or_above_1.301"])
        w.writerows(rows_csv)
    log(f"wrote {os.path.relpath(p, ROOT)}")
    log("")
    log(f"TOTAL q < {ALPHA}: {total_sig} of {4 * N_ROI * len(STAGES)}; "
        f"max P-score {overall_max:.4f} (threshold {THRESH:.4f}); "
        f"missing values {total_missing}")
    with open(os.path.join(OUT, "pscore_combined_4panel.log"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")


if __name__ == "__main__":
    main()
