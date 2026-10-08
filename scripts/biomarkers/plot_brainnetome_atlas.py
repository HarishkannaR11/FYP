"""
Presentation figures of the Brainnetome-246 parcellation.

Every boundary is read directly from the supplied atlas volume
(atlases/Brainnetome246/BN_Atlas_246_2mm.nii.gz). Nothing is approximated,
drawn by hand or synthesised: the 3D surface is extracted from that volume by
marching cubes and each surface vertex is coloured by the atlas label of the
voxel it sits in, so the rendered parcel boundaries are the atlas's own.

Outputs (derivatives/ATLAS_FIGURES/):
    brainnetome246_left_lateral.png
    brainnetome246_right_lateral.png
    brainnetome246_superior.png
    brainnetome246_combined.png        3D views + orthographic slices
    brainnetome246_slices.png          axial mosaic through the volume
    brainnetome246_roi_colors.csv      ROI id -> name -> hex colour
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from matplotlib.colors import ListedColormap, to_hex
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage import measure

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ATLAS = os.path.join(ROOT, "atlases", "Brainnetome246", "BN_Atlas_246_2mm.nii.gz")
LUT = os.path.join(ROOT, "atlases", "Brainnetome246", "BN_Atlas_246_LUT.txt")
OUT = os.path.join(ROOT, "derivatives", "ATLAS_FIGURES")

N_ROI = 246
TITLE = "Brainnetome 246-ROI Brain Parcellation"
ANNOT = ("246 anatomical ROIs used for resting-state fMRI "
         "regional biomarker extraction.")
CAPTION = ("Brainnetome 246-ROI parcellation used for regional "
           "resting-state fMRI biomarker extraction.")


def log(m):
    print(m, flush=True)


def load_lut():
    lut = {}
    if os.path.isfile(LUT):
        for line in open(LUT):
            p = line.split()
            if len(p) >= 2 and p[0].isdigit():
                lut[int(p[0])] = p[1]
    return lut


def roi_colours(seed=0):
    """
    246 maximally separable colours.

    Hues are spread with the golden-angle increment so neighbouring ROI ids do
    not get neighbouring hues, and lightness/saturation are cycled so that
    parcels adjacent in space are unlikely to share a shade.
    """
    import colorsys
    cols = []
    golden = 0.6180339887498949
    h = seed
    for i in range(N_ROI):
        h = (h + golden) % 1.0
        s = [0.62, 0.78, 0.52, 0.88][i % 4]
        v = [0.92, 0.72, 0.84, 0.62][(i // 2) % 4]
        cols.append(colorsys.hsv_to_rgb(h, s, v))
    return np.array(cols)


def hemisphere_surface(lab, affine, side):
    """
    Marching-cubes surface of one hemisphere, with every vertex assigned the
    atlas label of the voxel containing it. Boundaries therefore come from the
    atlas itself.
    """
    mask = lab > 0
    # Brainnetome numbering: odd ids left, even ids right.
    if side == "left":
        mask &= (lab % 2 == 1)
    elif side == "right":
        mask &= (lab % 2 == 0)

    vol = mask.astype(np.float32)
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.5, step_size=1)

    vox = np.clip(np.rint(verts).astype(int), 0,
                  np.array(lab.shape) - 1)
    vlab = lab[vox[:, 0], vox[:, 1], vox[:, 2]]

    # a few vertices land on a zero voxel at the rim; take the nearest labelled
    # neighbour rather than dropping or inventing a label
    bad = np.where(vlab == 0)[0]
    for i in bad:
        x, y, z = vox[i]
        found = 0
        for r in (1, 2):
            sl = lab[max(0, x - r):x + r + 1, max(0, y - r):y + r + 1,
                     max(0, z - r):z + r + 1]
            nz = sl[sl > 0]
            if nz.size:
                found = np.bincount(nz).argmax()
                break
        vlab[i] = found

    xyz = nib.affines.apply_affine(affine, verts)
    return xyz, faces, vlab


def render(ax, xyz, faces, vlab, colours, elev, azim, title):
    tri = xyz[faces]
    face_lab = vlab[faces[:, 0]]
    fc = np.ones((len(faces), 4))
    valid = face_lab > 0
    fc[valid, :3] = colours[face_lab[valid] - 1]
    fc[~valid, :3] = 0.85

    # cheap lambert shading so parcel borders stay visible on a 3D form
    v0, v1, v2 = tri[:, 0], tri[:, 1], tri[:, 2]
    n = np.cross(v1 - v0, v2 - v0)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.where(ln == 0, 1, ln)
    light = np.array([0.3, 0.5, 0.8])
    light = light / np.linalg.norm(light)
    shade = 0.55 + 0.45 * np.clip(n @ light, 0, 1)
    fc[:, :3] *= shade[:, None]
    fc[:, :3] = np.clip(fc[:, :3], 0, 1)

    coll = Poly3DCollection(tri, facecolors=fc, linewidths=0, antialiased=False)
    ax.add_collection3d(coll)

    # Use the true data extents rather than a cube: a brain is much wider than
    # it is tall, and forcing equal axes leaves most of the frame empty.
    lo, hi = xyz.min(axis=0), xyz.max(axis=0)
    pad = (hi - lo) * 0.02
    lo, hi = lo - pad, hi + pad
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])
    ax.set_zlim(lo[2], hi[2])
    ax.view_init(elev=elev, azim=azim)
    ax.set_axis_off()
    try:
        ax.set_box_aspect(tuple(hi - lo))   # real anatomical proportions
    except Exception:
        pass
    # pull the camera in so the mesh fills the axes
    try:
        ax.set_position(ax.get_position())
        ax.dist = 6.2
    except Exception:
        pass
    if title:
        ax.set_title(title, fontsize=11, fontweight="bold", pad=2)


def single_view(xyz, faces, vlab, colours, elev, azim, title, path,
                figsize=(9.0, 6.2)):
    fig = plt.figure(figsize=figsize, facecolor="white")
    ax = fig.add_axes([0.01, 0.13, 0.98, 0.76], projection="3d",
                      facecolor="white")
    render(ax, xyz, faces, vlab, colours, elev, azim, "")
    fig.suptitle(TITLE, fontsize=15, fontweight="bold", y=0.975)
    fig.text(0.5, 0.075, title, ha="center", fontsize=12)
    fig.text(0.5, 0.028, ANNOT, ha="center", fontsize=9, alpha=0.8)
    fig.savefig(path, dpi=300, facecolor="white")
    plt.close(fig)
    log(f"saved {os.path.basename(path)}")


def main():
    os.makedirs(OUT, exist_ok=True)
    img = nib.load(ATLAS)
    lab = np.asarray(img.dataobj).astype(np.int16)
    present = sorted(int(v) for v in np.unique(lab) if v > 0)
    log(f"atlas: {ATLAS}")
    log(f"shape {lab.shape}  labels present {len(present)}  "
        f"min {min(present)}  max {max(present)}")
    if len(present) != N_ROI:
        log(f"WARNING: expected {N_ROI} labels, found {len(present)}")

    lut = load_lut()
    colours = roi_colours()

    import csv
    with open(os.path.join(OUT, "brainnetome246_roi_colors.csv"), "w",
              newline="") as f:
        w = csv.writer(f)
        w.writerow(["ROI_ID", "ROI_Name", "hex_colour", "voxel_count",
                    "hemisphere"])
        for i in range(1, N_ROI + 1):
            w.writerow([i, lut.get(i, f"ROI_{i}"), to_hex(colours[i - 1]),
                        int((lab == i).sum()), "left" if i % 2 else "right"])
    log("saved brainnetome246_roi_colors.csv")

    log("extracting surfaces (marching cubes on the atlas volume)...")
    whole = hemisphere_surface(lab, img.affine, "both")
    left = hemisphere_surface(lab, img.affine, "left")
    right = hemisphere_surface(lab, img.affine, "right")
    log(f"  whole: {len(whole[0])} vertices, {len(whole[1])} faces")
    log(f"  left : {len(left[0])} vertices")
    log(f"  right: {len(right[0])} vertices")

    # individual views
    single_view(*left, colours, 0, 180, "Left hemisphere - lateral view",
                os.path.join(OUT, "brainnetome246_left_lateral.png"))
    single_view(*right, colours, 0, 0, "Right hemisphere - lateral view",
                os.path.join(OUT, "brainnetome246_right_lateral.png"))
    single_view(*whole, colours, 90, 270, "Superior view (both hemispheres)",
                os.path.join(OUT, "brainnetome246_superior.png"))

    # ---------- combined figure ----------
    fig = plt.figure(figsize=(15.5, 9.2), facecolor="white")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1.0], hspace=0.04,
                          wspace=0.02, top=0.88, bottom=0.11)
    views = [
        (left, 0, 180, "Left hemisphere\nlateral"),
        (whole, 90, 270, "Both hemispheres\nsuperior"),
        (right, 0, 0, "Right hemisphere\nlateral"),
    ]
    for j, (surf, elev, azim, ttl) in enumerate(views):
        ax = fig.add_subplot(gs[0, j], projection="3d", facecolor="white")
        render(ax, *surf, colours, elev, azim, ttl)

    # orthographic slices straight from the volume
    mid = [s // 2 for s in lab.shape]
    cmap = ListedColormap(np.vstack([[1, 1, 1], colours]))
    planes = [
        (np.rot90(lab[mid[0], :, :]), f"Sagittal (x={mid[0]})"),
        (np.rot90(lab[:, mid[1], :]), f"Coronal (y={mid[1]})"),
        (np.rot90(lab[:, :, mid[2]]), f"Axial (z={mid[2]})"),
    ]
    for j, (sl, ttl) in enumerate(planes):
        ax = fig.add_subplot(gs[1, j], facecolor="white")
        ax.imshow(sl, cmap=cmap, vmin=0, vmax=N_ROI, interpolation="nearest")
        ax.set_title(ttl, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)

    fig.suptitle(TITLE, fontsize=19, fontweight="bold", y=0.965)
    fig.text(0.5, 0.925, ANNOT, ha="center", fontsize=11, alpha=0.85)
    fig.text(0.5, 0.055,
             "Surfaces are extracted directly from BN_Atlas_246_2mm.nii.gz; each "
             "colour is one of the 246 atlas ROIs (odd labels left, even right).",
             ha="center", fontsize=9, alpha=0.75)
    fig.text(0.5, 0.022, CAPTION, ha="center", fontsize=10, style="italic")
    p = os.path.join(OUT, "brainnetome246_combined.png")
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log("saved brainnetome246_combined.png")

    # ---------- axial mosaic ----------
    zs = np.linspace(int(lab.shape[2] * 0.22), int(lab.shape[2] * 0.82), 12).astype(int)
    fig, axes = plt.subplots(3, 4, figsize=(12.5, 9.6), facecolor="white")
    for ax, z in zip(axes.ravel(), zs):
        ax.imshow(np.rot90(lab[:, :, z]), cmap=cmap, vmin=0, vmax=N_ROI,
                  interpolation="nearest")
        ax.set_title(f"z = {z}", fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.suptitle(TITLE + " - axial series", fontsize=16, fontweight="bold")
    fig.text(0.5, 0.055, ANNOT, ha="center", fontsize=10, alpha=0.85)
    fig.text(0.5, 0.022, CAPTION, ha="center", fontsize=9.5, style="italic")
    fig.tight_layout(rect=[0, 0.075, 1, 0.955])
    p = os.path.join(OUT, "brainnetome246_slices.png")
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log("saved brainnetome246_slices.png")

    log(f"\noutputs in {OUT}")
    log("ATLAS_FIGURES_DONE")


if __name__ == "__main__":
    main()
