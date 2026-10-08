"""Gallery of every processed scan, grouped by stage.

Each thumbnail is the axial panel of that acquisition's own QC picture
(derivatives/fsfast/<STAGE>/sub-*/ses-*/run-*/qc/mean_bold.png, the
time-averaged BOLD image). Nothing is selected or edited: all 165 processed
scans appear, sorted by subject, session and run. The one scan flagged as an
artefact in the QC audit (CN sub-018S4313 ses-01) is outlined in red.

Run from anywhere:  python3 docs/SecondReview/ppt/figures/make_stage_gallery.py
Writes stage_gallery.png next to this script and prints the per-stage counts.
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
FSFAST = os.path.join(ROOT, "derivatives", "fsfast")
STAGES = [("CN_Final", "CN"), ("SMC_Final", "SMC"), ("EMCI", "EMCI"),
          ("MCI", "MCI"), ("LMCI", "LMCI"), ("AD", "AD")]
# the axial panel of every QC picture sits at the same pixel box
X0, X1, Y0, Y1 = 645, 845, 47, 285
NROW, NCOL = 4, 8

ARTEFACT = os.path.join("sub-018S4313", "ses-01")

plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42})


def thumb(path):
    im = Image.open(path).convert("L").crop((X0, Y0, X1, Y1))
    return np.asarray(im.resize((50, 60), Image.LANCZOS))


counts = {}
fig = plt.figure(figsize=(5.75, 2.95))
outer = fig.add_gridspec(2, 3, left=0.005, right=0.995, top=0.93, bottom=0.005,
                         wspace=0.04, hspace=0.2)
for k, (directory, label) in enumerate(STAGES):
    files = sorted(glob.glob(os.path.join(
        FSFAST, directory, "sub-*", "ses-*", "run-*", "qc", "mean_bold.png")))
    counts[label] = len(files)
    assert len(files) <= NROW * NCOL, (label, len(files))
    inner = outer[k // 3, k % 3].subgridspec(NROW, NCOL, wspace=0.05, hspace=0.05)
    for i in range(NROW * NCOL):
        ax = fig.add_subplot(inner[i // NCOL, i % NCOL])
        ax.axis("off")
        if i < len(files):
            ax.imshow(thumb(files[i]), cmap="gray", vmin=0, vmax=255)
            if ARTEFACT in files[i]:
                ax.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax.transAxes,
                                           fill=False, ec="#d62728", lw=1.6, clip_on=False))
        if i == 0:
            ax.text(0, 1.25, "%s  (%d scans)" % (label, len(files)),
                    transform=ax.transAxes, fontsize=8.5, fontweight="bold",
                    color="#103e6e", va="bottom", ha="left")
fig.savefig(os.path.join(HERE, "stage_gallery.png"), dpi=300)
print(counts, sum(counts.values()))
