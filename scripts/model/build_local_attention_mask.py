"""
Derive the local attention mask from the Brainnetome atlas itself.

The MultiAttentionTransformer takes `local_mask [246, 246]` as a caller-supplied
argument and this repository has never provided one, so the local branch has
never been usable as designed. The mask is MEASURED from the atlas here, not
invented.

Two definitions are computed from BN_Atlas_246_2mm.nii.gz:

  1. CENTROID k-NN  (default)
     Each ROI's centroid is its mean voxel position mapped to MNI mm through
     the image affine. ROI j is a neighbour of ROI i if j is among i's k
     nearest centroids. The relation is symmetrised with OR, so the mask is
     symmetric and every ROI keeps at least k neighbours.

  2. VOXEL-BOUNDARY ADJACENCY
     ROIs i and j are adjacent if any voxel of i is 6-connected to any voxel
     of j. This is the stricter anatomical definition.

k-NN is the default because adjacency leaves some ROIs with very few
neighbours, and a near-empty row makes the local branch contribute nothing for
that ROI. Both are saved so the choice can be revisited and compared.

The mask encodes SPATIAL PROXIMITY ONLY. It is not a functional network, not a
connectivity measure, and carries no information about diagnosis. Diagonal is
always True, so every ROI attends to itself.
"""
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ATLAS = os.path.join(ROOT, "atlases", "Brainnetome246", "BN_Atlas_246_2mm.nii.gz")
OUT = os.path.join(ROOT, "derivatives", "model_ready")
N_ROI = 246
K_NEIGHBOURS = 15


def centroids():
    img = nib.load(ATLAS)
    data = np.asarray(img.dataobj)
    labels = np.unique(data)
    labels = labels[labels > 0]
    if labels.tolist() != list(range(1, N_ROI + 1)):
        raise SystemExit(f"atlas does not contain labels 1..246 (found {len(labels)})")
    cent = np.zeros((N_ROI, 3))
    counts = np.zeros(N_ROI, dtype=int)
    for lab in labels:
        vox = np.argwhere(data == lab)
        counts[lab - 1] = vox.shape[0]
        mm = nib.affines.apply_affine(img.affine, vox.astype(float))
        cent[lab - 1] = mm.mean(axis=0)
    if counts.min() == 0:
        raise SystemExit("an ROI has zero voxels")
    return cent, counts, data, img


def knn_mask(cent, k):
    d = np.linalg.norm(cent[:, None, :] - cent[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    idx = np.argsort(d, axis=1)[:, :k]
    m = np.zeros((N_ROI, N_ROI), dtype=bool)
    rows = np.repeat(np.arange(N_ROI), k)
    m[rows, idx.ravel()] = True
    m |= m.T                                  # symmetrise
    np.fill_diagonal(m, True)
    np.fill_diagonal(d, 0.0)
    return m, d


def adjacency_mask(data):
    m = np.zeros((N_ROI, N_ROI), dtype=bool)
    for axis in (0, 1, 2):
        a = np.take(data, np.arange(0, data.shape[axis] - 1), axis=axis)
        b = np.take(data, np.arange(1, data.shape[axis]), axis=axis)
        pairs = np.stack([a.ravel(), b.ravel()], axis=1)
        pairs = pairs[(pairs[:, 0] > 0) & (pairs[:, 1] > 0)]
        pairs = pairs[pairs[:, 0] != pairs[:, 1]]
        m[pairs[:, 0] - 1, pairs[:, 1] - 1] = True
    m |= m.T
    np.fill_diagonal(m, True)
    return m


def describe(m, name):
    deg = m.sum(axis=1) - 1                   # exclude self
    dens = float(m.sum()) / (N_ROI * N_ROI)
    print(f"{name:22s} density {dens * 100:5.2f}%  degree min {deg.min():3d} "
          f"median {int(np.median(deg)):3d} max {deg.max():3d}  "
          f"symmetric {bool((m == m.T).all())}  "
          f"ROIs with no neighbour {int((deg == 0).sum())}")
    return {"density": dens, "degree_min": int(deg.min()),
            "degree_median": int(np.median(deg)), "degree_max": int(deg.max()),
            "symmetric": bool((m == m.T).all()),
            "rois_without_neighbour": int((deg == 0).sum())}


def main():
    os.makedirs(OUT, exist_ok=True)
    print("deriving local attention mask from", os.path.relpath(ATLAS, ROOT))
    cent, counts, data, img = centroids()
    print(f"246 ROI centroids computed; voxel counts {counts.min()}-{counts.max()}")

    knn, dist = knn_mask(cent, K_NEIGHBOURS)
    adj = adjacency_mask(data)
    s_knn = describe(knn, f"centroid k-NN (k={K_NEIGHBOURS})")
    s_adj = describe(adj, "voxel adjacency")
    shared = float((knn & adj).sum() - N_ROI) / max((adj.sum() - N_ROI), 1)
    print(f"fraction of adjacency edges also in the k-NN mask: {shared:.3f}")

    np.save(os.path.join(OUT, "local_attention_mask.npy"), knn)
    np.save(os.path.join(OUT, "local_attention_mask_adjacency.npy"), adj)
    np.save(os.path.join(OUT, "roi_centroids_mm.npy"), cent)
    np.save(os.path.join(OUT, "roi_centroid_distance_mm.npy"), dist)

    meta = {
        "default_mask": "local_attention_mask.npy",
        "definition": f"symmetric centroid {K_NEIGHBOURS}-nearest-neighbour graph, "
                      "diagonal True",
        "k": K_NEIGHBOURS,
        "source_atlas": "atlases/Brainnetome246/BN_Atlas_246_2mm.nii.gz",
        "centroid_space": "MNI mm via the image affine",
        "encodes": "spatial proximity only; no functional, connectivity or "
                   "diagnostic information",
        "knn": s_knn,
        "adjacency": s_adj,
        "adjacency_edges_covered_by_knn": shared,
        "alternative_mask": "local_attention_mask_adjacency.npy",
    }
    with open(os.path.join(OUT, "local_attention_mask.json"), "w",
              encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nwrote {os.path.relpath(OUT, ROOT)}/local_attention_mask.npy "
          f"(+ adjacency, centroids, distances, json)")


if __name__ == "__main__":
    main()
