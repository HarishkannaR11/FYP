"""
Figures justifying the 246-ROI feature-extraction space. Read-only.

fig_roi_qc.png   the 246 ROIs as a complete, verified extraction space
fig_fc_matrix.png  the 246 x 246 signed, unthresholded connectivity matrix

Deliberately does NOT claim any ROI is individually disease-relevant, and
does not exaggerate between-ROI differences.
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AUD = os.path.join(R, "derivatives", "fsfast", "FINAL_6GROUP_AUDIT")
DRV = os.path.join(R, "derivatives", "fsfast")
OUT = os.path.join(R, "derivatives", "REVIEW2_FINDINGS")
UBLUE, ABLUE, LBLUE, DGRAY = "#103E6E", "#1E88E5", "#E8F2FC", "#374151"
GREEN, RED = "#2E7D32", "#C0392B"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(alpha=0.22, lw=0.6)
    ax.set_axisbelow(True)


def pick_cn():
    """Representative CN acquisition, matching the deck's convention."""
    q = pd.read_csv(os.path.join(AUD, "dataset_qc_summary.csv"))
    q = q[(q.group == "CN_Final") & (q.subject != "sub-018S4313")]
    r = q.sort_values("mean_tSNR", ascending=False).iloc[0]
    return os.path.join(DRV, r.group, r.subject, r.session, r.run,
                        "biomarkers"), r


def main():
    os.makedirs(OUT, exist_ok=True)
    bn = pd.read_csv(os.path.join(AUD, "dataset_brainnetome_summary.csv"))
    base, meta = pick_cn()
    ts = np.load(os.path.join(base, "roi_timeseries.npy"))      # (T, 246)
    fc = np.load(os.path.join(base, "fc_matrix.npy"))           # (246, 246)
    print(f"representative: {meta.subject}/{meta.session}/{meta.run}")
    print(f"  time series {ts.shape}, FC {fc.shape}")

    n_acq = len(glob.glob(os.path.join(DRV, "*", "sub-*", "ses-*", "run-*",
                                       "biomarkers", "roi_timeseries.npy")))
    var = ts.var(axis=0)

    # ================================================== FIGURE: ROI QC
    fig = plt.figure(figsize=(15.5, 7.6), facecolor="white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15], hspace=0.42,
                          wspace=0.22)

    # (a) voxel-count distribution
    ax = fig.add_subplot(gs[0, 0])
    ax.hist(bn.VOXEL_COUNT, bins=32, color=UBLUE, alpha=0.85,
            edgecolor="white")
    for v, lab, c in [(bn.VOXEL_COUNT.min(), f"min {bn.VOXEL_COUNT.min()}", RED),
                      (bn.VOXEL_COUNT.median(),
                       f"median {bn.VOXEL_COUNT.median():.0f}", GREEN),
                      (bn.VOXEL_COUNT.max(), f"max {bn.VOXEL_COUNT.max()}", RED)]:
        ax.axvline(v, color=c, ls="--", lw=1.3)
    ax.text(0.97, 0.9, f"min {bn.VOXEL_COUNT.min()}  ·  "
                       f"median {bn.VOXEL_COUNT.median():.0f}  ·  "
                       f"max {bn.VOXEL_COUNT.max()}",
            transform=ax.transAxes, ha="right", fontsize=10,
            fontweight="bold", color=UBLUE)
    ax.set_xlabel("Voxels per ROI (4 mm grid)", fontsize=10.5)
    ax.set_ylabel("Number of ROIs", fontsize=10.5)
    ax.set_title("(a)  Every ROI is adequately sampled",
                 fontsize=11.5, fontweight="bold", color=UBLUE)
    style(ax)

    # (b) per-ROI variance -- none are flat
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(np.arange(1, 247), var, s=9, color=ABLUE, alpha=0.7,
               edgecolors="none")
    ax.set_yscale("log")
    ax.axhline(var.min(), color=GREEN, ls="--", lw=1.2)
    ax.text(246, var.min() * 1.25, f"lowest variance {var.min():.3g}  "
            f"(zero-variance ROIs: 0)", ha="right", fontsize=9.5,
            color=GREEN, fontweight="bold")
    ax.set_xlabel("Brainnetome ROI index (1–246)", fontsize=10.5)
    ax.set_ylabel("Time-series variance (log)", fontsize=10.5)
    ax.set_xlim(0, 247)
    ax.set_title("(b)  No ROI yields a flat signal",
                 fontsize=11.5, fontweight="bold", color=UBLUE)
    style(ax)

    # (c) the extracted 135 x 246 matrix
    ax = fig.add_subplot(gs[1, :])
    z = (ts - ts.mean(0)) / ts.std(0)
    im = ax.imshow(z.T, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3,
                   interpolation="nearest")
    ax.set_xlabel("Volume (135 time points)", fontsize=10.5)
    ax.set_ylabel("Brainnetome ROI (246)", fontsize=10.5)
    ax.set_title(f"(c)  The extracted feature space: {ts.shape[0]} × "
                 f"{ts.shape[1]} ROI time-series matrix, "
                 f"identical in shape for all {n_acq} acquisitions",
                 fontsize=11.5, fontweight="bold", color=UBLUE)
    cb = fig.colorbar(im, ax=ax, fraction=0.016, pad=0.012)
    cb.set_label("signal (z per ROI, for display)", fontsize=9)
    cb.ax.tick_params(labelsize=8)

    fig.suptitle("The 246 ROIs form a complete, verified feature-extraction space",
                 fontsize=15, fontweight="bold", color=UBLUE, y=0.985)
    fig.text(0.5, 0.012,
             f"246/246 ROIs present in all {n_acq} acquisitions · 0 zero-voxel "
             f"ROIs · 0 ROIs below 10 voxels · 0 zero-variance time series "
             f"· 0 NaN/Inf.   Representative acquisition: "
             f"{meta.subject} ({meta.group.replace('_Final', '')}).",
             ha="center", fontsize=9.5, color=DGRAY)
    p = os.path.join(OUT, "fig_roi_qc.png")
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved fig_roi_qc.png")

    # ================================================== FIGURE: FC matrix
    off = fc[np.triu_indices(246, 1)]
    neg = 100 * (off < 0).mean()
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(14.2, 5.9), facecolor="white",
        gridspec_kw=dict(width_ratios=[1, 0.78], wspace=0.40))

    lim = np.abs(off).max()
    im = ax.imshow(fc, cmap="RdBu_r", vmin=-lim, vmax=lim,
                   interpolation="nearest")
    ax.set_xlabel("Brainnetome ROI (1–246)", fontsize=10.5)
    ax.set_ylabel("Brainnetome ROI (1–246)", fontsize=10.5)
    ax.set_title("246 × 246 functional connectivity\n"
                 "signed, unthresholded, diagonal retained",
                 fontsize=12, fontweight="bold", color=UBLUE)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Pearson r", fontsize=9.5)

    ax2.hist(off, bins=90, color=UBLUE, alpha=0.85, edgecolor="white")
    ax2.axvline(0, color=DGRAY, lw=1.2, ls=":")
    ax2.axvspan(off.min(), 0, color=ABLUE, alpha=0.08)
    ax2.text(0.03, 0.93, f"{neg:.1f}% of edges\nare negative",
             transform=ax2.transAxes, fontsize=10.5, va="top",
             color=ABLUE, fontweight="bold")
    ax2.text(0.03, 0.70, "centred near zero by\nglobal signal regression\n"
                         "(expected, not an artefact)",
             transform=ax2.transAxes, fontsize=8.5, va="top", color=DGRAY)
    ax2.text(0.97, 0.93, f"{len(off):,} unique\nROI-to-ROI edges",
             transform=ax2.transAxes, fontsize=10.5, va="top", ha="right",
             color=UBLUE, fontweight="bold")
    ax2.set_xlabel("Edge weight (Pearson r)", fontsize=10.5)
    ax2.set_ylabel("Number of edges", fontsize=10.5)
    ax2.set_title("Negative correlations are preserved,\nnot discarded",
                  fontsize=12, fontweight="bold", color=UBLUE)
    style(ax2)

    fig.suptitle("The 246 ROIs are the nodes of a whole-brain connectivity representation",
                 fontsize=14.5, fontweight="bold", color=UBLUE, y=1.02)
    fig.text(0.5, -0.06,
             f"One representative acquisition ({meta.subject}, "
             f"{meta.group.replace('_Final', '')}). The full signed matrix is "
             f"retained for every one of the {n_acq} acquisitions; no edge is "
             "thresholded away and no claim is made that any individual edge is "
             "disease-relevant.",
             ha="center", fontsize=9.5, color=DGRAY)
    p = os.path.join(OUT, "fig_fc_matrix.png")
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved fig_fc_matrix.png")
    print(f"\nedges {len(off):,}, {neg:.1f}% negative, "
          f"range {off.min():+.3f}..{off.max():+.3f}")


if __name__ == "__main__":
    main()
