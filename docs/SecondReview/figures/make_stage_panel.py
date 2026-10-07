"""Combine the three row-z stage heatmaps (mALFF, mReHo, DC_z) into one legible
figure with a single shared colour bar.

Reads derivatives/STAGE_HEATMAPS/<biomarker>_rowz.csv (written by
scripts/biomarkers/stage_heatmaps.py) and writes figures/stage_rowz_panel.pdf.
Run from docs/SecondReview/:  python figures/make_stage_panel.py
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "..", "derivatives", "STAGE_HEATMAPS")
STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
PANELS = [("mALFF", "mALFF"), ("mReHo", "mReHo"), ("DC_z", r"DC$_z$")]

plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 3, figsize=(6.3, 6.6), sharey=True)
vmax = 2.2
for ax, (key, title) in zip(axes, PANELS):
    d = pd.read_csv(os.path.join(SRC, f"{key}_rowz.csv"))
    m = d[STAGES].to_numpy()
    assert m.shape == (246, 6)
    im = ax.imshow(m, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest", extent=(-0.5, 5.5, 246.5, 0.5))
    ax.axhline(210.5, color="k", ls="--", lw=0.8)
    ax.set_xticks(range(6), STAGES, rotation=45, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
axes[0].set_yticks([1, 50, 100, 150, 200, 246])
axes[0].set_ylabel("Brainnetome-246 ROI (atlas order)", fontsize=8)
fig.subplots_adjust(left=0.12, right=0.98, top=0.95, bottom=0.2, wspace=0.08)
cax = fig.add_axes([0.2, 0.065, 0.6, 0.022])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label("Row-wise z-score across the six stages", fontsize=8)
cb.ax.tick_params(labelsize=8)
fig.savefig(os.path.join(HERE, "stage_rowz_panel.pdf"))
print("written", os.path.join(HERE, "stage_rowz_panel.pdf"))
