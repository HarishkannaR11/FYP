"""Explainer figures for the Review-II deck (every panel is drawn from real
project outputs; nothing is simulated).

One typical-quality acquisition is used throughout: CN sub-002S1280 ses-01
run-01 (mean FD 0.298 mm, tSNR 132; dataset medians 0.317 mm and 127.95).

Needs the Git-LFS objects for that acquisition and the Brainnetome atlas
(git lfs pull --include="derivatives/fsfast/CN_Final/sub-002S1280/**,atlases/**").

Run from anywhere:  python3 docs/SecondReview/ppt/figures/make_explainer_figs.py
Writes into the same folder:
    ts_example.pdf         three regional BOLD time series
    scan_to_features.pdf   scan -> atlas -> time series -> features -> FC
    bm_alff.png bm_reho.png bm_dc.png bm_fc.png   biomarker thumbnails
    malff_annotated.pdf    mALFF stage heatmap with reading guide
    atlas_overview.png     crop of derivatives/ATLAS_FIGURES/brainnetome246_combined.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
SUB = os.path.join(ROOT, "derivatives", "fsfast", "CN_Final", "sub-002S1280",
                   "ses-01", "run-01")
BM = os.path.join(SUB, "biomarkers")
LUT = os.path.join(ROOT, "atlases", "Brainnetome246", "BN_Atlas_246_LUT.txt")
COLOURS = os.path.join(ROOT, "derivatives", "ATLAS_FIGURES",
                       "brainnetome246_roi_colors.csv")
HEAT = os.path.join(ROOT, "derivatives", "STAGE_HEATMAPS")
STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
Z_MM = 20  # axial slice height in MNI millimetres

plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.linewidth": 0.6})

names = {}
for line in open(LUT):
    p = line.split()
    if len(p) >= 2 and p[0].isdigit():
        names[int(p[0])] = p[1]


def roi_index(name):
    """0-based column of a Brainnetome region in the ROI matrices."""
    return next(k for k, v in names.items() if v == name) - 1


def axial(vol, z_mm=Z_MM):
    """Axial slice of a 4 mm MNI volume, anterior up, as a 2-D array."""
    k = int(round((z_mm + 72) / 4))
    return vol[:, :, k].T[::-1, :]


ts = np.load(os.path.join(BM, "roi_timeseries.npy"))            # (135, 246)
fc = np.load(os.path.join(BM, "fc_matrix.npy"))                 # (246, 246)
reg = np.load(os.path.join(BM, "regional_biomarkers.npy"))      # (246, 4)
meta = json.load(open(os.path.join(
    SUB, "sub-002S1280_ses-01_task-rest_run-01_desc-preproc_bold.json")))
TR = float(meta["TR_seconds"])
t = np.arange(ts.shape[0]) * TR

bold = nib.load(os.path.join(
    SUB, "sub-002S1280_ses-01_task-rest_run-01_desc-preproc_bold.nii.gz"))
mean_bold = bold.get_fdata().mean(axis=3)
atlas = np.asarray(nib.load(os.path.join(BM, "brainnetome_246_4mm.nii.gz")).dataobj)
alff_map = nib.load(os.path.join(BM, "alff_map.nii.gz")).get_fdata()
reho_map = nib.load(os.path.join(BM, "reho_map.nii.gz")).get_fdata()

hexes = pd.read_csv(COLOURS).sort_values("ROI_ID")["hex_colour"].tolist()
atlas_cmap = ListedColormap(["#ffffff"] + hexes)


def pct(x):
    return (x / x.mean() - 1.0) * 100.0


def blank(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


# ---------------------------------------------------------------- ts_example
sel = [("A31_L", "#1f5fa8"), ("A31_R", "#d9822b"), ("mOccG_L", "#6b6b6b")]
fig, ax = plt.subplots(figsize=(4.1, 2.3))
step = 3.2
for i, (nm, col) in enumerate(sel):
    y = pct(ts[:, roi_index(nm)])
    ax.plot(t, y - i * step, color=col, lw=0.9)
    ax.text(t[-1] + 6, -i * step, nm.replace("_", " "), color=col, va="center",
            fontsize=8, fontweight="bold")
r_pair = np.corrcoef(ts[:, roi_index("A31_L")], ts[:, roi_index("A31_R")])[0, 1]
r_occ = np.corrcoef(ts[:, roi_index("A31_L")], ts[:, roi_index("mOccG_L")])[0, 1]
ax.set_yticks([])
ax.set_xlabel("time (s)   one point every TR = %.0f s" % TR)
ax.set_xlim(0, t[-1])
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
fig.text(0.02, 0.97, "left-right pair A31: r = %.2f      A31 L vs mOccG L: r = %.2f" % (r_pair, r_occ),
         fontsize=7.5, va="top")
fig.subplots_adjust(left=0.02, right=0.78, top=0.86, bottom=0.2)
fig.savefig(os.path.join(HERE, "ts_example.pdf"))
plt.close(fig)
print("ts_example: r_pair=%.3f r_occ=%.3f TR=%s n=%d" % (r_pair, r_occ, TR, ts.shape[0]))

# ------------------------------------------------------------ scan_to_features
fig = plt.figure(figsize=(5.7, 2.05))
gs = fig.add_gridspec(1, 5, left=0.01, right=0.995, top=0.83, bottom=0.2,
                      wspace=0.16)
titles = ["1  Scan", "2  Atlas", "3  Signals", "4  Features", "5  Connectivity"]
shapes = ["135 volumes", "246 regions", "135 x 246", "246 x 3", "246 x 246"]
axs = [fig.add_subplot(gs[0, i]) for i in range(5)]

axs[0].imshow(axial(mean_bold), cmap="gray", vmin=np.percentile(mean_bold[mean_bold > 0], 2),
              vmax=np.percentile(mean_bold, 99.5), interpolation="nearest")
blank(axs[0])

lab = axial(atlas).astype(int)
img = np.where(lab > 0, lab, 0)
axs[1].imshow(img, cmap=atlas_cmap, vmin=0, vmax=246, interpolation="nearest")
blank(axs[1])

rng = np.random.RandomState(1)
pick = [roi_index(n) for n in ("A31_L", "A31_R", "mOccG_L", "rHipp_L", "A7m_L")]
for i, c in enumerate(pick):
    axs[2].plot(pct(ts[:, c]) * 0.55 - i * 2.2, color=["#1f5fa8", "#d9822b", "#6b6b6b",
                                                       "#2a9d5c", "#a33b8d"][i], lw=0.7)
axs[2].set_xlim(0, ts.shape[0] - 1)
blank(axs[2])

feat = reg[:, :3].copy()
feat = (feat - feat.mean(axis=0)) / feat.std(axis=0)
axs[3].imshow(feat, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5,
              interpolation="nearest")
axs[3].set_xticks([0, 1, 2], ["ALFF", "ReHo", "DC"], fontsize=6.5)
axs[3].set_yticks([])
axs[3].tick_params(length=0, pad=1)
for s in axs[3].spines.values():
    s.set_visible(False)

axs[4].imshow(fc, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
blank(axs[4])

for a, ttl, shp in zip(axs, titles, shapes):
    a.set_title(ttl, fontsize=8, fontweight="bold", pad=3)
    a.text(0.5, -0.09 if a is not axs[3] else -0.1, shp, transform=a.transAxes,
           ha="center", va="top", fontsize=7)
for i in range(4):
    x0 = axs[i].get_position().x1
    x1 = axs[i + 1].get_position().x0
    fig.add_artist(matplotlib.patches.FancyArrowPatch(
        (x0 + 0.002, 0.52), (x1 - 0.002, 0.52), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=7, lw=0.9, color="#134074"))
fig.savefig(os.path.join(HERE, "scan_to_features.pdf"), dpi=300)
plt.close(fig)

# ------------------------------------------------------ biomarker thumbnails
def thumb(path, draw, size=(1.55, 1.35)):
    fig, ax = plt.subplots(figsize=size)
    draw(ax)
    fig.subplots_adjust(0, 0, 1, 1)
    fig.savefig(os.path.join(HERE, path), dpi=300, transparent=True)
    plt.close(fig)


def brain_map(vol, cmap):
    def draw(ax):
        sl = axial(vol)
        sl = np.ma.masked_where(sl <= 0, sl)
        hi = np.percentile(vol[vol > 0], 98)
        ax.imshow(sl, cmap=cmap, vmin=0, vmax=hi, interpolation="nearest")
        blank(ax)
    return draw


thumb("bm_alff.png", brain_map(alff_map, "magma"))
thumb("bm_reho.png", brain_map(reho_map, "viridis"))

dc = reg[:, 2]
dcz = (dc - dc.mean()) / dc.std()
lut_img = np.zeros(atlas.shape)
for r in range(1, 247):
    lut_img[atlas == r] = dcz[r - 1]
mask_img = (atlas > 0)


def draw_dc(ax):
    sl = axial(np.where(mask_img, lut_img, np.nan))
    ax.imshow(np.ma.masked_invalid(sl), cmap="RdBu_r", vmin=-2.5, vmax=2.5,
              interpolation="nearest")
    blank(ax)


thumb("bm_dc.png", draw_dc)


def draw_fc(ax):
    ax.imshow(fc, cmap="RdBu_r", vmin=-1, vmax=1, interpolation="nearest")
    i = roi_index("A31_L")
    ax.add_patch(plt.Rectangle((-0.5, i - 0.5), 246, 1.0, fill=False, ec="k", lw=1.0))
    blank(ax)


thumb("bm_fc.png", draw_fc, size=(1.4, 1.4))

# ------------------------------------------------------------ malff annotated
m = pd.read_csv(os.path.join(HEAT, "mALFF_rowz.csv"))[STAGES].to_numpy()
fig, ax = plt.subplots(figsize=(3.4, 2.55))
im = ax.imshow(m, aspect="auto", cmap="RdBu_r", vmin=-2.2, vmax=2.2,
               interpolation="nearest", extent=(-0.5, 5.5, 246.5, 0.5))
ax.set_xticks(range(6), STAGES, fontsize=8)
ax.set_yticks([1, 100, 200, 246])
ax.tick_params(labelsize=7.5)
ax.set_ylabel("246 brain regions (rows)", fontsize=8)
ax.set_xlabel("stage (columns), in disease order", fontsize=8)
ax.xaxis.set_label_position("top")
ax.xaxis.tick_top()
fig.subplots_adjust(left=0.19, right=0.97, top=0.84, bottom=0.2)
cax = fig.add_axes([0.25, 0.08, 0.62, 0.04])
cb = fig.colorbar(im, cax=cax, orientation="horizontal")
cb.set_label("region's level vs its own average over the six stages", fontsize=7)
cb.set_ticks([-2, 0, 2], labels=["lower", "average", "higher"])
cb.ax.tick_params(labelsize=7)
fig.savefig(os.path.join(HERE, "malff_annotated.pdf"))
plt.close(fig)

# ------------------------------------------------------------- atlas overview
# Re-composition of derivatives/ATLAS_FIGURES/brainnetome246_combined.png (made
# by scripts/biomarkers/plot_brainnetome_atlas.py): the six panels are cropped
# from that image unchanged and given larger, slide-legible labels.
src = np.asarray(Image.open(os.path.join(ROOT, "derivatives", "ATLAS_FIGURES",
                                         "brainnetome246_combined.png")).convert("RGB"))
panels = [
    ("Left hemisphere", (193, 1085, 440, 1250)),
    ("Top view", (1531, 2161, 440, 1250)),
    ("Right hemisphere", (2616, 3497, 440, 1250)),
    ("Sagittal slice", (218, 1026, 1592, 2290)),
    ("Coronal slice", (1486, 2177, 1592, 2290)),
    ("Axial slice", (2752, 3321, 1592, 2290)),
]
CW, CH = 920, 830
fig, axes = plt.subplots(2, 3, figsize=(6.2, 3.55))
for ax, (ttl, (x0, x1, y0, y1)) in zip(axes.ravel(), panels):
    crop = src[y0:y1, x0:x1]
    canvas = np.full((CH, CW, 3), 255, np.uint8)
    h, w = crop.shape[:2]
    oy, ox = (CH - h) // 2, (CW - w) // 2
    canvas[oy:oy + h, ox:ox + w] = crop
    ax.imshow(canvas, interpolation="lanczos")
    ax.set_title(ttl, fontsize=8.5, fontweight="bold", pad=1)
    ax.axis("off")
fig.subplots_adjust(left=0.005, right=0.995, top=0.94, bottom=0.005, wspace=0.01, hspace=0.12)
fig.savefig(os.path.join(HERE, "atlas_overview.png"), dpi=260)
plt.close(fig)
print("done")
