"""Text-free axial mosaic of the Brainnetome-246 atlas for the compact slide.

Every pixel is read straight from atlases/Brainnetome246/BN_Atlas_246_2mm.nii.gz
(2 mm MNI grid) and coloured with the ROI colour table written by
scripts/biomarkers/plot_brainnetome_atlas.py, so the colours match the 3D
renders: each colour is one of the 246 ROIs. Nothing is smoothed or redrawn;
slices are shown at the atlas's own resolution (nearest-neighbour upscaling).

Writes atlas_axial_panel.png next to this script.
Run from anywhere:  python3 docs/SecondReview/ppt/figures/make_atlas_axial.py
"""
import os

import nibabel as nib
import numpy as np
import pandas as pd
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", "..", ".."))
ATLAS = os.path.join(ROOT, "atlases", "Brainnetome246", "BN_Atlas_246_2mm.nii.gz")
COLOURS = os.path.join(ROOT, "derivatives", "ATLAS_FIGURES", "brainnetome246_roi_colors.csv")

Z_MM = [-16, -2, 12, 26, 40, 56]      # temporal pole ... vertex
ROWS, COLS = 2, 3
SCALE, GAP = 8, 28

img = nib.as_closest_canonical(nib.load(ATLAS))   # RAS+: axis 0 = x (left to right)
vol = np.asarray(img.dataobj).astype(int)
inv = np.linalg.inv(img.affine)
tab = pd.read_csv(COLOURS)
lut = np.full((int(vol.max()) + 1, 3), 255, dtype=np.uint8)
for r, h in zip(tab.ROI_ID, tab.hex_colour):
    lut[int(r)] = [int(h[i:i + 2], 16) for i in (1, 3, 5)]


def axial(z_mm):
    k = int(round((inv @ np.array([0, 0, z_mm, 1.0]))[2]))
    lab = vol[:, :, k]                              # (x, y)
    rgb = lut[lab]                                  # white where label 0
    rgb = np.transpose(rgb, (1, 0, 2))[::-1]        # rows = y with anterior at top
    return rgb, lab


slices = [axial(z) for z in Z_MM]
# one common brain-shaped frame so all six tiles have the same size
any_lab = np.max([np.transpose(l, (1, 0))[::-1] > 0 for _, l in slices], axis=0)
ys, xs = np.where(any_lab)
y0, y1, x0, x1 = ys.min() - 1, ys.max() + 2, xs.min() - 1, xs.max() + 2
tiles = []
for rgb, _ in slices:
    t = Image.fromarray(rgb[y0:y1, x0:x1])
    tiles.append(t.resize((t.width * SCALE, t.height * SCALE), Image.NEAREST))

tw, th = tiles[0].size
canvas = Image.new("RGB", (COLS * tw + (COLS - 1) * GAP, ROWS * th + (ROWS - 1) * GAP), "white")
for i, t in enumerate(tiles):
    r, c = divmod(i, COLS)
    canvas.paste(t, (c * (tw + GAP), r * (th + GAP)))
out = os.path.join(HERE, "atlas_axial_panel.png")
canvas.save(out, optimize=True)
print("wrote", out, canvas.size, "slices at z =", Z_MM, "mm")
