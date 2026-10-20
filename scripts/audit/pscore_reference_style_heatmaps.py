"""
P-score heatmaps in the layout of group_hierarchical_mReHo.png.

A direct P-score analogue of the reference figure, one per biomarker:

    left    hierarchical dendrogram over the 246 ROIs      (RESTORED)
    top     hierarchical dendrogram over the 6 groups
    cells   P-score = -log10(FDR-adjusted q)
    right   colourbar
    bottom  clinical progression order and captions

Two deliberate differences from the reference, both requested
-------------------------------------------------------------
1. COLOUR SCALE IS PER-FIGURE, 0 to that biomarker's own maximum.

   The reference spans 99% of its colour bar because Z-scores reach +/-2.
   P-scores here reach only 0.137 (DC_z) to 0.928 (mReHo), so on a scale
   stretched to the q = 0.05 threshold of 1.301 they occupy the bottom tenth
   and the figure looks blank. Scaling to the observed range restores the
   contrast of the reference.

   THE COST, STATED ON EVERY FIGURE: the q = 0.05 threshold is now OFF SCALE,
   and the panels are NOT comparable to one another -- the same colour means a
   different P-score in each. Colour intensity here shows RELATIVE variation
   within one biomarker, never proximity to significance.

2. Sequential YlOrRd instead of the reference's diverging red-blue. P-scores
   are non-negative with no meaningful midpoint; a diverging map centred on
   zero would imply a sign distinction that does not exist.

Clustering, reusing the configuration documented in
DIAGNOSTIC_GROUP_HIERARCHICAL_README.md but applied to P-SCORE profiles:

    6 groups   correlation distance + average linkage over the 246-D profile
    246 ROIs   euclidean distance + average linkage over the 6-D profile

Source q-values are read, never recomputed. No existing figure, CSV or
statistical output is modified.
"""
import csv
import json
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
BASE = os.path.join(V2, "diagnostic_group_hierarchical")
OUT, CSVD = os.path.join(BASE, "figures"), os.path.join(BASE, "csv")
SRC = os.path.join(ROOT, "derivatives", "model_ready")
LUT = os.path.join(ROOT, "derivatives", "fsfast", "FINAL_6GROUP_AUDIT",
                   "dataset_brainnetome_summary.csv")

STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
BIOMARKERS = ["mALFF", "mReHo", "DC_z", "FC"]
LONG = {"mALFF": "Amplitude of low-frequency fluctuation (mALFF)",
        "mReHo": "Regional homogeneity (mReHo)",
        "DC_z": "Degree centrality (DC_z)",
        "FC": "Functional connectivity (unsigned mean Fisher-z strength)"}
STAGE_COLORS = {"CN": "#1A5276", "SMC": "#2E86C1", "EMCI": "#7FB3D5",
                "MCI": "#F5B041", "LMCI": "#D35400", "AD": "#922B21"}
N_ROI, ALPHA, FLOOR = 246, 0.05, 1e-12
THRESH = -np.log10(ALPHA)
CMAP = "YlOrRd"
GRP_METRIC, GRP_LINK = "correlation", "average"
ROI_METRIC, ROI_LINK = "euclidean", "average"

_log = []


def log(m=""):
    print(m, flush=True)
    _log.append(str(m))


def read_q(bm):
    path = os.path.join(V2, f"{bm}_roi_stage_qvalues.csv")
    rows = list(csv.DictReader(open(path)))
    if len(rows) != N_ROI:
        raise SystemExit(f"{bm}: {len(rows)} rows, expected {N_ROI}")
    if list(csv.reader(open(path)))[0][2:] != STAGES:
        raise SystemExit(f"{bm}: unexpected column headers")
    Q = np.full((N_ROI, len(STAGES)), np.nan)
    missing = 0
    for i, r in enumerate(rows):
        for j, s in enumerate(STAGES):
            if r[s] in ("", "nan", "NaN", "NA"):
                missing += 1
            else:
                Q[i, j] = float(r[s])
    return Q, missing, [int(r["ROI_ID"]) for r in rows]


def gray(*a, **k):
    return "#9A9A9A"


def make_figure(bm, P, Q, ids, names, Lg, grp_order, Lr, roi_order,
                cg, cr, missing):
    vmax = float(np.nanmax(P))
    disp = P[np.ix_(roi_order, grp_order)]
    n_sig = int(np.nansum(Q < ALPHA))

    fig = plt.figure(figsize=(9.4, 15.6))
    gs = GridSpec(5, 3, figure=fig,
                  width_ratios=[0.15, 1.0, 0.05],
                  height_ratios=[0.085, 1.0, 0.045, 0.055, 0.075],
                  wspace=0.45, hspace=0.05,
                  left=0.215, right=0.855, top=0.880, bottom=0.044)

    # --- top: diagnostic-group dendrogram
    ax_t = fig.add_subplot(gs[0, 1])
    hierarchy.dendrogram(Lg, ax=ax_t, no_labels=True, color_threshold=0,
                         link_color_func=gray)
    ax_t.set_xticks([])
    ax_t.set_yticks([])
    for sp in ax_t.spines.values():
        sp.set_visible(False)
    ax_t.set_title(f"diagnostic-group clustering by P-score profile "
                   f"({GRP_METRIC} distance, {GRP_LINK} linkage)",
                   fontsize=8.4, color="#374151", pad=4)

    # --- left: ROI dendrogram (restored)
    ax_l = fig.add_subplot(gs[1, 0])
    hierarchy.dendrogram(Lr, ax=ax_l, orientation="left", no_labels=True,
                         color_threshold=0, link_color_func=gray)
    ax_l.set_xticks([])
    ax_l.set_yticks([])
    ax_l.invert_yaxis()
    ax_l.patch.set_visible(False)          # do not paint over the tick labels
    for sp in ax_l.spines.values():
        sp.set_visible(False)
    ax_l.set_title("ROI\nclustering", fontsize=7.6, color="#374151", pad=4)

    # --- heatmap
    ax = fig.add_subplot(gs[1, 1])
    cmap = matplotlib.colormaps[CMAP].with_extremes(bad="#D1D5DB")
    im = ax.imshow(disp, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(range(len(STAGES)))
    ax.set_xticklabels([STAGES[g] for g in grp_order], fontsize=12)
    for t, g in zip(ax.get_xticklabels(), grp_order):
        t.set_color(STAGE_COLORS[STAGES[g]])
        t.set_fontweight("bold")
    ax.xaxis.set_ticks_position("bottom")
    ax.tick_params(axis="x", pad=5)
    step = 10
    ticks = list(range(0, N_ROI, step))
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{ids[roi_order[t]]:>3d} {names[roi_order[t]]}"
                        for t in ticks], fontsize=6.0, family="monospace")
    for j in range(1, len(STAGES)):
        ax.axvline(j - 0.5, color="white", lw=0.9)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
        sp.set_color("#9CA3AF")
    fig.text(0.016, 0.50, f"Brainnetome ROI  (all {N_ROI} rows plotted, "
             f"every {step}th labelled, clustered order)",
             rotation=90, va="center", ha="center", fontsize=9)

    # --- colourbar, scaled to this biomarker's own range
    cax = fig.add_subplot(gs[1, 2])
    cb = fig.colorbar(im, cax=cax)
    # Translate the P-score scale back into q so the reader can see at a
    # glance how far the darkest cell still is from significance.
    q_pale, q_dark = 1.0, float(10.0 ** (-vmax))
    cb.set_label(f"P-score = -log10(FDR-adjusted q)\n"
                 f"scale 0.000-{vmax:.3f}, THIS BIOMARKER'S OWN RANGE\n"
                 f"palest = q {q_pale:.3f}    darkest = q {q_dark:.3f}\n"
                 f"q = 0.05 threshold ({THRESH:.3f}) is OFF THIS SCALE",
                 fontsize=7.8)
    cb.ax.tick_params(labelsize=7.5)

    ax_c = fig.add_subplot(gs[2, 1])
    ax_c.axis("off")
    ax_c.text(0.5, 0.45, "columns are ordered by the dendrogram above, NOT by "
              "disease stage", ha="center", va="center", fontsize=8.4,
              color="#111827", transform=ax_c.transAxes)

    ax_p = fig.add_subplot(gs[3, 1])
    ax_p.axis("off")
    ax_p.text(0.5, 0.45, f"clinical progression order:  "
              f"{'  →  '.join(STAGES)}", ha="center", va="center",
              fontsize=9.5, color="#374151", fontweight="bold",
              transform=ax_p.transAxes)

    ax_f = fig.add_subplot(gs[4, 1])
    ax_f.axis("off")
    ax_f.text(0.5, 0.86,
              f"COLOUR SHOWS RELATIVE VARIATION WITHIN {bm.upper()} ONLY, NOT "
              f"PROXIMITY TO SIGNIFICANCE.",
              ha="center", va="center", fontsize=8.2, color="#B91C1C",
              fontweight="bold", transform=ax_f.transAxes)
    ax_f.text(0.5, 0.50,
              f"The darkest cell here is q = {q_dark:.3f} and the palest is "
              f"q = {q_pale:.3f} — the entire colour range sits far above "
              f"the {ALPHA} threshold.\n"
              f"Scale tops out at this biomarker's own maximum ({vmax:.3f}); "
              f"q = {ALPHA} would require {THRESH:.3f}. Panels are NOT "
              f"comparable across biomarkers.\n"
              f"No ROI reaches q < {ALPHA}: {n_sig} of {N_ROI * len(STAGES)} "
              f"tests. Branch heights are distances, not p-values.",
              ha="center", va="center", fontsize=7.4, color="#374151",
              linespacing=1.5, transform=ax_f.transAxes)

    fig.suptitle("Hierarchical Clustering of Alzheimer's Diagnostic Groups\n"
                 f"{LONG[bm]} — P-Scores",
                 fontsize=13.5, fontweight="bold", y=0.975)
    fig.text(0.5, 0.928,
             f"cells are -log10(FDR-adjusted q) for each ROI and diagnostic "
             f"group (stage vs pooled rest; BH over 246 x 6 = 1476 tests)\n"
             f"groups clustered by {GRP_METRIC} distance + {GRP_LINK} linkage; "
             f"ROIs by {ROI_METRIC} distance + {ROI_LINK} linkage, both on "
             f"P-SCORE profiles",
             ha="center", va="top", fontsize=8.6, color="#374151",
             linespacing=1.55)
    if missing:
        fig.text(0.5, 0.010, f"{missing} missing q-values in grey, not zeroed",
                 ha="center", fontsize=7.2, color="#B91C1C")

    png = os.path.join(OUT, f"pscore_reference_style_{bm}.png")
    pdf = os.path.join(OUT, f"pscore_reference_style_{bm}.pdf")
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    log(f"  wrote {os.path.relpath(png, ROOT)}")
    log(f"  wrote {os.path.relpath(pdf, ROOT)}")
    return vmax


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(os.path.join(SRC, "metadata.json")))
    if meta["stage_to_index"] != {s: i for i, s in enumerate(STAGES)}:
        raise SystemExit(f"stage mapping mismatch: {meta['stage_to_index']}")
    lut = list(csv.DictReader(open(LUT)))
    names = [r["ROI_NAME"] for r in lut]

    log("=" * 80)
    log("P-SCORE HEATMAPS IN THE REFERENCE LAYOUT")
    log("=" * 80)
    log(f"stage mapping verified: {meta['stage_to_index']}")
    log("colour scale: PER FIGURE, 0 to that biomarker's own maximum")
    log(f"q = 0.05 threshold ({THRESH:.4f}) is OFF SCALE on every figure")

    rows, total_sig = [], 0
    for bm in BIOMARKERS:
        log("")
        log(f"--- {bm} ---")
        Q, missing, ids = read_q(bm)
        P = -np.log10(np.where(np.isnan(Q), np.nan, np.maximum(Q, FLOOR)))
        P[P == 0] = 0.0
        n_sig = int(np.nansum(Q < ALPHA))
        total_sig += n_sig

        Dg = pdist(P.T, metric=GRP_METRIC)
        Lg = hierarchy.optimal_leaf_ordering(
            hierarchy.linkage(Dg, method=GRP_LINK), Dg)
        grp_order = hierarchy.leaves_list(Lg)
        cg = float(hierarchy.cophenet(Lg, Dg)[0])

        Dr = pdist(P, metric=ROI_METRIC)
        Lr = hierarchy.optimal_leaf_ordering(
            hierarchy.linkage(Dr, method=ROI_LINK), Dr)
        roi_order = hierarchy.leaves_list(Lr)
        cr = float(hierarchy.cophenet(Lr, Dr)[0])

        log(f"  max P {np.nanmax(P):.4f}  q<{ALPHA}: {n_sig}  missing {missing}")
        log(f"  groups {[STAGES[g] for g in grp_order]}  cophenetic {cg:.3f}")
        log(f"  ROI tree cophenetic {cr:.3f}")
        vmax = make_figure(bm, P, Q, ids, names, Lg, grp_order, Lr, roi_order,
                           cg, cr, missing)
        for pos, g in enumerate(grp_order):
            rows.append([bm, pos, STAGES[g], g, f"{vmax:.6f}",
                         f"{float(P[:, g].max()):.6f}",
                         int((P[:, g] >= THRESH).sum()), f"{cg:.4f}",
                         f"{cr:.4f}"])

    p = os.path.join(CSVD, "pscore_reference_style_leaf_order.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["biomarker", "display_column", "stage_label",
                    "source_stage_index", "figure_colour_vmax",
                    "max_P_score_in_column", "n_ROIs_at_or_above_1.301",
                    "group_tree_cophenetic_r", "roi_tree_cophenetic_r"])
        w.writerows(rows)
    log("")
    log(f"wrote {os.path.relpath(p, ROOT)}")
    log(f"TOTAL q < {ALPHA}: {total_sig} of {4 * N_ROI * len(STAGES)}")
    with open(os.path.join(OUT, "pscore_reference_style.log"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")


if __name__ == "__main__":
    main()
