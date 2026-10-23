"""
Poster-sized version of the combined four-panel P-score figure.

Same data, clustering and colour scale as pscore_combined_four_panel.py
(P = -log10(FDR q), correlation distance + average linkage on the 246-D
P-score profile, fixed 0.0-1.4 YlOrRd scale so the q = 0.05 line at 1.301
is visible on the bar). Only the layout differs: the figure is drawn at its
final printed size (10.1 x 5.2 in) with 10-13 pt text, and the title,
subtitle, legend and red footer are left to the poster itself.

Pass "blue" as the first argument for the blue-red variant: RdBu_r with a
per-panel scale (blue = P of 0, pale = that panel's median, red = its 99th
percentile, as in group_hierarchical_with_pscore_top.py). Colour is then
RELATIVE within each panel and the panels are not comparable.

Reads the q-value CSVs through the original script's read_q(); nothing is
recomputed.
"""
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pscore_combined_four_panel import (ALPHA, BASE, BIOMARKERS, CMAP, FLOOR,
                                        LINKAGE, METRIC, N_ROI, STAGES,
                                        STAGE_COLORS, THRESH, VMAX, VMIN,
                                        read_q)

BLUE = len(sys.argv) > 1 and sys.argv[1] == "blue"
CLIP_PCT = 99
OUT = os.path.join(BASE, "figures",
                   "pscore_poster_panel_blue.png" if BLUE else "pscore_poster_panel.png")


def main():
    panels = {}
    for bm in BIOMARKERS:
        Q, _ = read_q(bm)
        P = -np.log10(np.maximum(Q, FLOOR))
        P[P == 0] = 0.0
        D = pdist(P.T, metric=METRIC)
        L = hierarchy.optimal_leaf_ordering(hierarchy.linkage(D, method=LINKAGE), D)
        panels[bm] = (P, L, hierarchy.leaves_list(L))

    fig = plt.figure(figsize=(10.1, 5.2))
    gs = GridSpec(2, 5, figure=fig, width_ratios=[1, 1, 1, 1, 0.07],
                  height_ratios=[0.16, 1.0], wspace=0.16, hspace=0.05,
                  left=0.035, right=0.915, top=0.885, bottom=0.135)
    cmap = matplotlib.colormaps["RdBu_r" if BLUE else CMAP].with_extremes(bad="#D1D5DB")

    for k, bm in enumerate(BIOMARKERS):
        P, L, order = panels[bm]
        ax_t = fig.add_subplot(gs[0, k])
        hierarchy.dendrogram(L, ax=ax_t, no_labels=True, color_threshold=0,
                             link_color_func=lambda *a, **kw: "#6B7280")
        ax_t.set_xticks([]); ax_t.set_yticks([])
        for sp in ax_t.spines.values():
            sp.set_visible(False)
        nice = "FC" if bm == "FC" else bm
        ax_t.set_title(f"{chr(65 + k)}  {nice}\nmax P {float(np.nanmax(P)):.2f}",
                       fontsize=12, fontweight="bold", pad=4, linespacing=1.15)

        ax = fig.add_subplot(gs[1, k])
        if BLUE:
            vmid, vtop = float(np.median(P)), float(np.percentile(P, CLIP_PCT))
            kw = dict(norm=TwoSlopeNorm(vmin=0.0, vcenter=vmid, vmax=vtop))
        else:
            kw = dict(vmin=VMIN, vmax=VMAX)
        im = ax.imshow(P[:, order], aspect="auto", cmap=cmap,
                       interpolation="nearest", **kw)
        ax.set_xticks(range(len(STAGES)))
        ax.set_xticklabels([STAGES[g] for g in order], fontsize=11,
                           rotation=45, ha="right", rotation_mode="anchor")
        for t, g in zip(ax.get_xticklabels(), order):
            t.set_color(STAGE_COLORS[STAGES[g]])
            t.set_fontweight("bold")
        ax.set_yticks([0, N_ROI - 1])
        ax.set_yticklabels(["1", "246"] if k == 0 else ["", ""], fontsize=9)
        if k == 0:
            ax.set_ylabel("Brainnetome ROI", fontsize=10)
        for j in range(1, len(STAGES)):
            ax.axvline(j - 0.5, color="white", lw=0.8)
        for sp in ax.spines.values():
            sp.set_linewidth(0.6); sp.set_color("#9CA3AF")

    cax = fig.add_subplot(gs[1, 4])
    if BLUE:
        # one shared bar in panel-relative units: 0 = P of 0, 0.5 = median, 1 = 99th pct
        sm = ScalarMappable(norm=TwoSlopeNorm(vmin=0.0, vcenter=0.5, vmax=1.0), cmap=cmap)
        cb = fig.colorbar(sm, cax=cax, extend="max")
        cb.set_ticks([0, 0.5, 1.0])
        cb.set_ticklabels(["P = 0", "median", "top 1%"])
        cb.set_label("P-score, relative within each panel", fontsize=10)
        cb.ax.tick_params(labelsize=9)
    else:
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("P-score = \u2212log10(q)", fontsize=10)
        cb.ax.tick_params(labelsize=9)
        cb.ax.axhline(THRESH, color="#111827", lw=1.7)
        cb.ax.text(1.15, THRESH, f"q={ALPHA}", transform=cb.ax.get_yaxis_transform(),
                   ha="left", va="center", fontsize=8.5, fontweight="bold")

    fig.savefig(OUT, dpi=300)
    plt.close(fig)
    print("wrote", os.path.relpath(OUT, os.path.dirname(BASE)))


if __name__ == "__main__":
    main()
