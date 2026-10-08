"""Slide-sized versions of the two ROI-space figures for the backup slides.

Same data, same panels and same representative acquisition as
scripts/audit/atlas_feature_space_figures.py (derivatives/REVIEW2_FINDINGS/
fig_roi_qc.png and fig_fc_matrix.png), redrawn at the size they are shown on a
16:9 slide (5.75 in = the full text width, so fonts are shown at 1:1) so every
label stays legible. Read-only; claims nothing about any individual region
being disease-relevant.

Writes into this folder:
    atlas_roi_qc_slide.pdf   voxel counts, per-ROI variance, 135 x 246 matrix
    atlas_fc_slide.pdf       246 x 246 signed FC matrix and edge distribution

Run from anywhere:  python3 docs/SecondReview/ppt/figures/make_atlas_backup_figs.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
DRV = os.path.join(ROOT, "derivatives", "fsfast")
AUD = os.path.join(DRV, "FINAL_6GROUP_AUDIT")
UBLUE, ABLUE, DGRAY, GREEN, RED = "#103E6E", "#1E88E5", "#374151", "#2E7D32", "#C0392B"

plt.rcParams.update({"font.size": 7.5, "pdf.fonttype": 42, "axes.linewidth": 0.6,
                     "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "xtick.major.pad": 2, "ytick.major.pad": 2})
TITLE = dict(fontsize=8, fontweight="bold", color=UBLUE, pad=4)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.22, lw=0.5)
    ax.set_axisbelow(True)


# representative acquisition, same rule as the original script:
# highest-tSNR CN scan, excluding the flagged artefact scan
q = pd.read_csv(os.path.join(AUD, "dataset_qc_summary.csv"))
q = q[(q.group == "CN_Final") & (q.subject != "sub-018S4313")]
rep = q.sort_values("mean_tSNR", ascending=False).iloc[0]
base = os.path.join(DRV, rep.group, rep.subject, rep.session, rep.run, "biomarkers")
ts = np.load(os.path.join(base, "roi_timeseries.npy"))      # (135, 246)
fc = np.load(os.path.join(base, "fc_matrix.npy"))           # (246, 246)
bn = pd.read_csv(os.path.join(AUD, "dataset_brainnetome_summary.csv"))
fc_files = sorted(glob.glob(os.path.join(DRV, "*", "sub-*", "ses-*", "run-*",
                                         "biomarkers", "fc_matrix.npy")))
n_acq = len(fc_files)
print("representative:", rep.subject, rep.session, rep.run, "| acquisitions:", n_acq)

# ------------------------------------------------------------------ ROI QC
var = ts.var(axis=0)
W, H = 5.75, 2.15
fig = plt.figure(figsize=(W, H))

ax = fig.add_axes([0.072, 0.19, 0.255, 0.66])
ax.hist(bn.VOXEL_COUNT, bins=30, color=UBLUE, alpha=0.9, edgecolor="white", lw=0.4)
ymax = ax.get_ylim()[1]
ax.set_ylim(0, ymax * 1.38)
for v, c in [(bn.VOXEL_COUNT.min(), RED), (bn.VOXEL_COUNT.median(), GREEN),
             (bn.VOXEL_COUNT.max(), RED)]:
    ax.axvline(v, color=c, ls="--", lw=1)
ax.text(0.905, 0.97, "min %d\nmedian %d\nmax %d" % (
    bn.VOXEL_COUNT.min(), bn.VOXEL_COUNT.median(), bn.VOXEL_COUNT.max()),
    transform=ax.transAxes, ha="right", va="top", fontsize=7.5, fontweight="bold",
    color=UBLUE, linespacing=1.25)
ax.set_xlabel("voxels per ROI (4 mm grid)")
ax.set_ylabel("number of ROIs")
ax.set_title("(a) every ROI is sampled", **TITLE)
style(ax)

ax = fig.add_axes([0.415, 0.19, 0.245, 0.66])
ax.scatter(np.arange(1, 247), var, s=4, color=ABLUE, alpha=0.75, edgecolors="none")
ax.set_yscale("log")
ax.set_ylim(var.min() / 14, var.max() * 2)
ax.axhline(var.min(), color=GREEN, ls="--", lw=1)
expo = int(np.floor(np.log10(var.min())))
ax.text(0.5, 0.115, "lowest $%.1f\\times10^{%d}$\nnone is flat" % (var.min() / 10 ** expo, expo),
        transform=ax.transAxes, ha="center", va="center", fontsize=7.5, color=GREEN,
        fontweight="bold", linespacing=1.25)
ax.set_xlabel("ROI index (1–246)")
ax.set_ylabel("variance (log scale)")
ax.set_xlim(0, 247)
ax.set_title("(b) no ROI is flat", **TITLE)
style(ax)

ax = fig.add_axes([0.745, 0.19, 0.195, 0.66])
z = (ts - ts.mean(0)) / ts.std(0)
im = ax.imshow(z.T, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3, interpolation="nearest")
ax.set_xlabel("volume (1–135)")
ax.set_ylabel("ROI (1–246)")
ax.set_xticks([0, 45, 90, 135 - 1])
ax.set_xticklabels(["1", "45", "90", "135"])
ax.set_title("(c) ROI time series", **TITLE)
cax = fig.add_axes([0.952, 0.19, 0.012, 0.66])
cb = fig.colorbar(im, cax=cax, ticks=[-3, 0, 3])
cb.ax.tick_params(labelsize=6.5, length=2)
cb.ax.set_title("z", fontsize=7, pad=2)
fig.savefig(os.path.join(HERE, "atlas_roi_qc_slide.pdf"))
plt.close(fig)

# --------------------------------------------------------------- FC matrix
iu = np.triu_indices(246, 1)
off = fc[iu]
neg_rep = 100 * (off < 0).mean()
neg_all = 100 * np.mean([(np.load(f)[iu] < 0).mean() for f in fc_files])
print("negative edges: this scan %.1f%%, mean over all scans %.1f%%" % (neg_rep, neg_all))

W, H = 5.75, 2.2
fig = plt.figure(figsize=(W, H))
side = 0.70 * H                                    # square matrix panel (inches)
ax = fig.add_axes([0.09, 0.18, side / W, 0.70])
lim = np.abs(off).max()
im = ax.imshow(fc, cmap="RdBu_r", vmin=-lim, vmax=lim, interpolation="nearest")
ax.set_xlabel("ROI (1–246)")
ax.set_ylabel("ROI (1–246)")
ax.set_title("246 × 246 FC (signed)", **TITLE)
cax = fig.add_axes([0.09 + side / W + 0.013, 0.18, 0.012, 0.70])
cb = fig.colorbar(im, cax=cax)
cb.set_label("Pearson r", fontsize=7, labelpad=2)
cb.ax.tick_params(labelsize=6.5, length=2)

ax2 = fig.add_axes([0.595, 0.18, 0.385, 0.70])
counts, _, _ = ax2.hist(off, bins=70, color=UBLUE, alpha=0.9, edgecolor="white", lw=0.3)
ax2.set_ylim(0, counts.max() * 1.45)
ax2.vlines(0, 0, counts.max() * 1.04, color=DGRAY, lw=1, ls=":", zorder=5)
ax2.axvspan(off.min(), 0, ymax=1.04 / 1.45, color=ABLUE, alpha=0.08)
ax2.text(0.03, 0.97, "%.1f%% of edges are negative" % neg_rep,
         transform=ax2.transAxes, fontsize=7.5, va="top", color=ABLUE, fontweight="bold")
ax2.text(0.03, 0.865, "(%.1f%% across all %d scans)" % (neg_all, n_acq),
         transform=ax2.transAxes, fontsize=7, va="top", color=ABLUE)
ax2.set_xlabel("edge weight (Pearson r)")
ax2.set_ylabel("number of edges")
ax2.set_title("%s unique edges, negatives kept" % format(len(off), ","), **TITLE)
style(ax2)
fig.savefig(os.path.join(HERE, "atlas_fc_slide.pdf"))
plt.close(fig)
print("done")
