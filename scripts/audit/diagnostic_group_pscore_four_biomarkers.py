"""
Diagnostic-Group Hierarchical Clustering Across All Four Biomarkers
-- P-Score Profiles.

ONE combined four-panel figure (A mALFF, B mReHo, C DC_z, D FC) styled after
group_hierarchical_mReHo.png, but showing STATISTICAL SIGNIFICANCE instead of
standardised biomarker means.

    cell value   P-score = -log10(FDR-adjusted q)
    rows         all 246 Brainnetome ROIs, atlas order, every 10th labelled
    columns      the 6 diagnostic groups, in that panel's dendrogram order
    top          per-panel dendrogram over the groups
    left         ROI identifiers and names -- NO ROI dendrogram

The reference figure clusters ROIs on the left and colours cells with a
diverging red-blue map. Both are deliberately changed here:

  * the left ROI dendrogram is removed, as requested, and the space is given
    to ROI identifiers;
  * the diverging map is replaced by a sequential one. P-scores are
    non-negative with no meaningful midpoint, so a red-blue map centred on
    zero would imply a sign distinction that does not exist.

Clustering, per biomarker independently -- the configuration documented in
DIAGNOSTIC_GROUP_HIERARCHICAL_README.md and reused here:

    distance  correlation, 1 - Pearson r, over the 246-D P-SCORE profile
    linkage   average (UPGMA)
    ordering  scipy optimal_leaf_ordering, deterministic

The input is the P-score profile, never the raw biomarker values. Branch
heights are merge distances, not p-values, so no numerical branch labels are
drawn.

Sources are read and never recomputed; the existing statistical outputs,
figures and CSVs are not modified.
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
PANEL_TITLE = {"mALFF": "mALFF", "mReHo": "mReHo", "DC_z": "DC_z",
               "FC": "FC"}
STAGE_COLORS = {"CN": "#1A5276", "SMC": "#2E86C1", "EMCI": "#7FB3D5",
                "MCI": "#F5B041", "LMCI": "#D35400", "AD": "#922B21"}
N_ROI, ALPHA, FLOOR = 246, 0.05, 1e-12
THRESH = -np.log10(ALPHA)
VMIN, VMAX, CMAP = 0.0, 1.40, "YlOrRd"
METRIC, LINKAGE = "correlation", "average"
STEM = "diagnostic_group_hierarchical_pscore_four_biomarkers"

_log = []


def log(m=""):
    print(m, flush=True)
    _log.append(str(m))


def verify_mapping():
    meta = json.load(open(os.path.join(SRC, "metadata.json")))
    want = {s: i for i, s in enumerate(STAGES)}
    if meta["stage_to_index"] != want:
        raise SystemExit(f"stage mapping mismatch: {meta['stage_to_index']}")
    log(f"stage mapping verified from source: {meta['stage_to_index']}")


def read_q(bm):
    path = os.path.join(V2, f"{bm}_roi_stage_qvalues.csv")
    rows = list(csv.DictReader(open(path)))
    if len(rows) != N_ROI:
        raise SystemExit(f"{bm}: {len(rows)} rows, expected {N_ROI}")
    hdr = list(csv.reader(open(path)))[0]
    if hdr[2:] != STAGES:
        raise SystemExit(f"{bm}: columns {hdr[2:]} != {STAGES}")
    ids = [int(r["ROI_ID"]) for r in rows]
    if ids != list(range(1, N_ROI + 1)):
        raise SystemExit(f"{bm}: ROI_IDs are not 1..246 in order")
    Q = np.full((N_ROI, len(STAGES)), np.nan)
    missing = 0
    for i, r in enumerate(rows):
        for j, s in enumerate(STAGES):
            if r[s] in ("", "nan", "NaN", "NA"):
                missing += 1              # kept NaN, never zeroed
            else:
                Q[i, j] = float(r[s])
    return Q, missing, os.path.relpath(path, ROOT)


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CSVD, exist_ok=True)
    log("=" * 82)
    log("DIAGNOSTIC-GROUP HIERARCHICAL CLUSTERING -- P-SCORE PROFILES")
    log("=" * 82)
    verify_mapping()
    names = [r["ROI_NAME"] for r in csv.DictReader(open(LUT))]
    ids = [int(r["ROI_ID"]) for r in csv.DictReader(open(LUT))]

    panels, rows_csv, sources = {}, [], {}
    total_sig, overall_max, total_missing = 0, 0.0, 0
    for bm in BIOMARKERS:
        Q, missing, src = read_q(bm)
        sources[bm] = src
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
        panels[bm] = (P, L, order, coph)
        log(f"  {bm:6s} {P.shape}  max P {np.nanmax(P):.4f}  q<{ALPHA}: {n_sig}  "
            f"missing {missing}  leaves {[STAGES[g] for g in order]}  "
            f"coph {coph:.3f}  <- {src}")
        for pos, g in enumerate(order):
            rows_csv.append([bm, pos, STAGES[g], g, STAGES.index(STAGES[g]),
                             f"{float(P[:, g].max()):.6f}",
                             int((P[:, g] >= THRESH).sum()), src])

    # ------------------------------------------------------------- figure
    fig = plt.figure(figsize=(18.0, 12.4))
    gs = GridSpec(2, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.034],
                  height_ratios=[0.16, 1.0], wspace=0.30, hspace=0.035,
                  left=0.105, right=0.935, top=0.838, bottom=0.138)
    cmap = matplotlib.colormaps[CMAP].with_extremes(bad="#D1D5DB")

    for k, bm in enumerate(BIOMARKERS):
        P, L, order, coph = panels[bm]

        ax_t = fig.add_subplot(gs[0, k])
        hierarchy.dendrogram(L, ax=ax_t, no_labels=True, color_threshold=0,
                             link_color_func=lambda *a, **kw: "#8A8A8A")
        ax_t.set_xticks([])
        ax_t.set_yticks([])
        for sp in ax_t.spines.values():
            sp.set_visible(False)
        # Title split in two: at one font size the stats line is wider than a
        # panel and collides with the neighbouring panel's title.
        ax_t.set_title(f"{chr(65 + k)}.  {PANEL_TITLE[bm]}",
                       fontsize=12.5, fontweight="bold", pad=24)
        ax_t.text(0.5, 1.05,
                  f"max P-score {float(np.nanmax(P)):.3f}   |   "
                  f"cophenetic r = {coph:.3f}",
                  transform=ax_t.transAxes, ha="center", va="bottom",
                  fontsize=8.8, color="#374151")

        ax = fig.add_subplot(gs[1, k])
        im = ax.imshow(P[:, order], aspect="auto", cmap=cmap,
                       vmin=VMIN, vmax=VMAX, interpolation="nearest")
        ax.set_xticks(range(len(STAGES)))
        ax.set_xticklabels([STAGES[g] for g in order], fontsize=11)
        for t, g in zip(ax.get_xticklabels(), order):
            t.set_color(STAGE_COLORS[STAGES[g]])
            t.set_fontweight("bold")
        ax.tick_params(axis="x", pad=5)
        step = 10
        ticks = list(range(0, N_ROI, step))
        if k == 0:
            # ROI identifiers replace the removed left dendrogram
            ax.set_yticks(ticks)
            ax.set_yticklabels([f"{ids[t]:>3d} {names[t]}" for t in ticks],
                               fontsize=5.6, family="monospace")
            ax.set_ylabel(f"Brainnetome ROI  (all {N_ROI} rows plotted, "
                          f"every {step}th labelled, atlas order)", fontsize=9.5)
        else:
            ax.set_yticks(ticks)
            ax.set_yticklabels([str(ids[t]) for t in ticks], fontsize=5.6,
                               family="monospace")
        for j in range(1, len(STAGES)):
            ax.axvline(j - 0.5, color="white", lw=0.9)
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
            sp.set_color("#9CA3AF")

    cax = fig.add_subplot(gs[1, 4])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("P-score = -log10(FDR-adjusted q-value)\n"
                 f"higher = stronger evidence  |  fixed scale "
                 f"{VMIN:.1f}-{VMAX:.1f}, identical in all four panels",
                 fontsize=9.5)
    cb.ax.tick_params(labelsize=8.5)
    cb.ax.axhline(THRESH, color="#111827", lw=1.8)
    cb.ax.text(0.5, THRESH, f"P-score = {THRESH:.3f}\n(q = {ALPHA})",
               transform=cb.ax.get_yaxis_transform(), ha="center", va="bottom",
               fontsize=8.0, fontweight="bold", color="#111827",
               bbox=dict(boxstyle="square,pad=0.2", fc="white", ec="none",
                         alpha=0.92))

    fig.suptitle("Diagnostic-Group Hierarchical Clustering Across All Four "
                 "Biomarkers — P-Score Profiles",
                 fontsize=16, fontweight="bold", y=0.975)
    fig.text(0.5, 0.920,
             "cells are the ROI-level statistical significance of each "
             "diagnostic group versus the pooled rest:  "
             "P-score = -log10(FDR-adjusted q)\n"
             f"groups clustered per panel by {METRIC} distance + {LINKAGE} "
             f"linkage over the 246-D P-SCORE profile; ROI rows are in atlas "
             f"order and are NOT clustered",
             ha="center", va="top", fontsize=9.5, color="#374151",
             linespacing=1.6)

    fig.text(0.5, 0.098,
             "columns are ordered by each panel's dendrogram, NOT by disease "
             "stage", ha="center", fontsize=9.5, color="#111827")
    fig.text(0.5, 0.068,
             f"clinical progression order:  {'  →  '.join(STAGES)}",
             ha="center", fontsize=10.5, color="#374151", fontweight="bold")
    fig.text(0.5, 0.036,
             f"No ROI-stage test reaches the significance threshold in any "
             f"panel: 0 of {4 * N_ROI * len(STAGES):,} tests at q < {ALPHA}; "
             f"the largest P-score anywhere is {overall_max:.3f} < "
             f"{THRESH:.3f}.",
             ha="center", fontsize=10.5, color="#B91C1C", fontweight="bold")
    fig.text(0.5, 0.012,
             "Descriptive only. Branch heights are merge distances, not "
             "p-values. The tree reflects similarity of statistical-evidence "
             "profiles, not biology or disease progression.",
             ha="center", fontsize=8.6, color="#6B7280", style="italic")

    png, pdf = os.path.join(OUT, STEM + ".png"), os.path.join(OUT, STEM + ".pdf")
    for p in (png, pdf):
        if os.path.exists(p):
            raise SystemExit(f"refusing to overwrite existing file: {p}")
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    log("")
    log(f"wrote {os.path.relpath(png, ROOT)}")
    log(f"wrote {os.path.relpath(pdf, ROOT)}")

    p = os.path.join(CSVD, STEM + "_leaf_order.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["biomarker", "display_column", "stage_label",
                    "source_stage_index", "clinical_order_index",
                    "max_P_score_in_column", "n_ROIs_at_or_above_1.301",
                    "source_file"])
        w.writerows(rows_csv)
    log(f"wrote {os.path.relpath(p, ROOT)}")
    log("")
    log(f"TOTAL q < {ALPHA}: {total_sig} of {4 * N_ROI * len(STAGES)}; "
        f"max P-score {overall_max:.4f} (threshold {THRESH:.4f}); "
        f"missing values {total_missing}")
    with open(os.path.join(OUT, STEM + ".log"), "w", encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")


if __name__ == "__main__":
    main()
