"""Slide-sized version of the report's three-panel row-z heatmap (same data,
larger relative fonts so stage labels stay legible on a 16:9 slide).
Run from docs/SecondReview/ppt/:  python figures/make_slide_panel.py
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "..", "..", "..", "..", "derivatives", "STAGE_HEATMAPS")
STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
PANELS = [("mALFF", "mALFF"), ("mReHo", "mReHo"), ("DC_z", r"DC$_z$")]
plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
fig, axes = plt.subplots(1, 3, figsize=(5.2, 3.3), sharey=True)
for ax, (key, title) in zip(axes, PANELS):
    m = pd.read_csv(os.path.join(SRC, f"{key}_rowz.csv"))[STAGES].to_numpy()
    im = ax.imshow(m, aspect="auto", cmap="RdBu_r", vmin=-2.2, vmax=2.2,
                   interpolation="nearest", extent=(-0.5, 5.5, 246.5, 0.5))
    ax.axhline(210.5, color="k", ls="--", lw=0.7)
    ax.set_xticks(range(6), STAGES, rotation=60, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10, fontweight="bold")
axes[0].set_yticks([1, 100, 210, 246])
axes[0].set_ylabel("Brainnetome ROI", fontsize=8)
fig.subplots_adjust(left=0.13, right=0.99, top=0.9, bottom=0.27, wspace=0.07)
cax = fig.add_axes([0.25, 0.07, 0.5, 0.035])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label("Row-wise z-score across stages", fontsize=8)
cb.ax.tick_params(labelsize=8)
fig.savefig(os.path.join(HERE, "stage_rowz_panel_slide.pdf"))
