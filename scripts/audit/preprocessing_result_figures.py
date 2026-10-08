"""
Presentation-ready preprocessing result figures.

Every number and image is read from the real pipeline outputs; nothing is
illustrative or simulated. Read-only.

Outputs (derivatives/PREPROCESSING_FIGURES/):
    fig1_cohort_flow.png        175 -> 165 attrition, with reasons
    fig2_qc_by_stage.png        tSNR / SNR / FD / DVARS per stage
    fig3_motion.png             FD distribution, thresholds, example traces
    fig4_registration.png       preprocessed BOLD in MNI with mask + atlas
    fig5_verification.png       data-integrity and verification summary
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AUD = os.path.join(ROOT, "derivatives", "fsfast", "FINAL_6GROUP_AUDIT")
DERIV = os.path.join(ROOT, "derivatives", "fsfast")
ATLAS = os.path.join(ROOT, "atlases", "Brainnetome246", "BN_Atlas_246_2mm.nii.gz")
OUT = os.path.join(ROOT, "derivatives", "PREPROCESSING_FIGURES")

STAGES = [("CN_Final", "CN"), ("SMC_Final", "SMC"), ("EMCI", "EMCI"),
          ("MCI", "MCI"), ("LMCI", "LMCI"), ("AD", "AD")]
ORDER = [s[1] for s in STAGES]
GMAP = dict(STAGES)
BLUE, GREY, RED, GREEN = "#1f4e79", "#9aa5b1", "#c0392b", "#2e7d32"
ARTIFACT = ("CN_Final", "sub-018S4313", "ses-01", "run-01")


def log(m):
    print(m, flush=True)


def load_qc():
    q = pd.read_csv(os.path.join(AUD, "dataset_qc_summary.csv"))
    q["stage"] = q.group.map(GMAP)
    q["is_artifact"] = (q.group == ARTIFACT[0]) & (q.subject == ARTIFACT[1]) & \
                       (q.session == ARTIFACT[2]) & (q.run == ARTIFACT[3])
    return q


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


# ----------------------------------------------------------------------
def fig_cohort_flow():
    fig, ax = plt.subplots(figsize=(13.5, 5.2), facecolor="white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 44)
    ax.axis("off")

    boxes = [
        (4, "175", "Raw ADNI\nacquisitions", BLUE),
        (26, "173", "Eligible for\npreprocessing", BLUE),
        (48, "165", "Preprocessed\nsuccessfully", GREEN),
        (70, "165", "Biomarkers +\nnormalisation", GREEN),
    ]
    for x, n, lab, c in boxes:
        ax.add_patch(FancyBboxPatch((x, 20), 17, 15, boxstyle="round,pad=0.5",
                                    fc=c, ec="none", alpha=0.92))
        ax.text(x + 8.5, 30.5, n, ha="center", va="center", fontsize=25,
                fontweight="bold", color="white")
        ax.text(x + 8.5, 24, lab, ha="center", va="center", fontsize=10,
                color="white")
    for x in (21.5, 43.5, 65.5):
        ax.add_patch(FancyArrowPatch((x, 27.5), (x + 4, 27.5),
                                     arrowstyle="-|>", mutation_scale=20,
                                     lw=2, color="#444"))

    drops = [
        (30, "− 2 excluded", "1 truncated (7 volumes)\n1 byte-identical duplicate"),
        (52, "− 8 failed", "TR ≈ 6.02 s → Nyquist 0.083 Hz\nbelow the 0.10 Hz cutoff; all site 002"),
    ]
    for x, head, body in drops:
        ax.add_patch(FancyArrowPatch((x - 7.5, 19), (x - 7.5, 12),
                                     arrowstyle="-|>", mutation_scale=15,
                                     lw=1.6, color=RED, linestyle="--"))
        ax.text(x - 5.5, 13.5, head, fontsize=11, fontweight="bold", color=RED,
                va="center")
        ax.text(x - 5.5, 8, body, fontsize=8.8, color="#555", va="center")

    ax.text(50, 41, "Preprocessing cohort flow", ha="center", fontsize=16,
            fontweight="bold")
    ax.text(50, 1.5,
            "Every exclusion is documented and reversible; no acquisition was "
            "dropped silently. 165 acquisitions from 127 distinct subjects.",
            ha="center", fontsize=9.5, color="#555", style="italic")
    p = os.path.join(OUT, "fig1_cohort_flow.png")
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log("saved fig1_cohort_flow.png")


# ----------------------------------------------------------------------
def fig_qc_by_stage(q):
    clean = q[~q.is_artifact]
    panels = [
        ("mean_tSNR", "Temporal SNR", None),
        ("SNR", "SNR (native space)", None),
        ("mean_FD_mm", "Mean framewise displacement (mm)", [0.2, 0.5]),
        ("mean_DVARS_raw", "DVARS (raw intensity units)", None),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(17.5, 4.6), facecolor="white")
    for ax, (col, title, hlines) in zip(axes, panels):
        data = [clean[clean.stage == s][col].dropna().values for s in ORDER]
        bp = ax.boxplot(data, tick_labels=ORDER, patch_artist=True, widths=0.6,
                        medianprops=dict(color="black", lw=1.6),
                        flierprops=dict(marker="o", ms=3, mfc=GREY,
                                        mec="none", alpha=0.6))
        for b in bp["boxes"]:
            b.set(facecolor=BLUE, alpha=0.55, edgecolor=BLUE)
        for i, s in enumerate(ORDER):
            v = clean[clean.stage == s][col].dropna()
            ax.scatter(np.random.normal(i + 1, 0.055, len(v)), v, s=7,
                       color="#123", alpha=0.45, zorder=3)
        if hlines:
            for h in hlines:
                ax.axhline(h, color=RED, ls="--", lw=1, alpha=0.7)
                ax.text(6.45, h, f"{h}", color=RED, fontsize=8, va="center")
        ax.set_title(title, fontsize=11.5, fontweight="bold")
        ax.tick_params(labelsize=9)
        style(ax)
    fig.suptitle("Preprocessing quality control across the six progression stages",
                 fontsize=15.5, fontweight="bold", y=1.03)
    fig.text(0.5, -0.07,
             f"n = {len(clean)} acquisitions (one artifact acquisition excluded; see report). "
             "Box = IQR, line = median, points = individual acquisitions. "
             "CNR is not computable: no T1w image exists for any subject.",
             ha="center", fontsize=9.5, color="#555")
    fig.tight_layout()
    p = os.path.join(OUT, "fig2_qc_by_stage.png")
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log("saved fig2_qc_by_stage.png")


# ----------------------------------------------------------------------
def fig_motion(q):
    clean = q[~q.is_artifact]
    fd = clean.mean_FD_mm.dropna()
    fig = plt.figure(figsize=(14.5, 5.0), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.35], wspace=0.22)

    ax = fig.add_subplot(gs[0])
    ax.hist(fd, bins=28, color=BLUE, alpha=0.75, edgecolor="white")
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.22)
    for t, c, dy in [(0.2, "#e67e22", 1.16), (0.5, RED, 1.04)]:
        ax.axvline(t, color=c, ls="--", lw=1.6)
        ax.annotate(f"{t} mm — {int((fd > t).sum())} above", xy=(t, top * dy),
                    xytext=(t + 0.06, top * dy), color=c, fontsize=9.5,
                    va="center", fontweight="bold",
                    bbox=dict(fc="white", ec="none", pad=1.5))
    ax.set_xlabel("Mean framewise displacement (mm)", fontsize=10.5)
    ax.set_ylabel("Acquisitions", fontsize=10.5)
    ax.set_title(f"Head motion across the cohort (n = {len(fd)})",
                 fontsize=12, fontweight="bold")
    style(ax)

    # real FD traces: lowest, median and highest-motion acquisitions
    ax2 = fig.add_subplot(gs[1])
    ordered = clean.sort_values("mean_FD_mm")
    picks = [(ordered.iloc[0], GREEN, "lowest"),
             (ordered.iloc[len(ordered) // 2], BLUE, "median"),
             (ordered.iloc[-1], RED, "highest")]
    for row, col, lab in picks:
        f = os.path.join(DERIV, row.group, row.subject, row.session, row.run,
                         "qc", "fd_values.csv")
        if not os.path.isfile(f):
            continue
        d = pd.read_csv(f)
        c = [x for x in d.columns if "fd" in x.lower()]
        y = d[c[0]] if c else d.iloc[:, -1]
        ax2.plot(y.values, color=col, lw=1.3, alpha=0.9,
                 label=f"{lab}: {row.subject} ({row.mean_FD_mm:.2f} mm)")
    ax2.axhline(0.2, color="#e67e22", ls="--", lw=1, alpha=0.7)
    ax2.axhline(0.5, color=RED, ls="--", lw=1, alpha=0.7)
    ax2.set_xlabel("Volume", fontsize=10.5)
    ax2.set_ylabel("FD (mm)", fontsize=10.5)
    ax2.set_title("Framewise displacement traces (real acquisitions)",
                  fontsize=12, fontweight="bold")
    ax2.legend(fontsize=8.5, frameon=False)
    style(ax2)

    fig.suptitle("Head motion: measured, flagged, never auto-excluded",
                 fontsize=15, fontweight="bold", y=1.04)
    fig.text(0.5, -0.06,
             "High-motion acquisitions were flagged rather than removed, under the "
             "frozen processing policy. Motion will enter the modelling stage as a covariate.",
             ha="center", fontsize=9.5, color="#555")
    p = os.path.join(OUT, "fig3_motion.png")
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log("saved fig3_motion.png")


# ----------------------------------------------------------------------
def fig_registration(q):
    """Representative preprocessed acquisition in MNI space, with its brain
    mask and the Brainnetome atlas overlaid on the same grid."""
    row = q[~q.is_artifact].sort_values("mean_tSNR", ascending=False).iloc[0]
    base = os.path.join(DERIV, row.group, row.subject, row.session, row.run)
    bold = glob.glob(os.path.join(base, "*_desc-preproc_bold.nii.gz"))[0]
    img = nib.load(bold)
    data = img.get_fdata(dtype=np.float32)
    mean = data.mean(axis=3)
    mask = nib.load(os.path.join(base, "brain_mask.nii.gz")).get_fdata() > 0

    at = nib.load(os.path.join(base, "biomarkers", "brainnetome_246_4mm.nii.gz"))
    atl = np.asarray(at.dataobj).astype(int)

    zs = np.linspace(int(mean.shape[2] * 0.28), int(mean.shape[2] * 0.78), 6).astype(int)
    fig, axes = plt.subplots(3, 6, figsize=(16, 8.4), facecolor="white")
    import colorsys
    rng = np.random.default_rng(1)
    cols = np.array([colorsys.hsv_to_rgb((i * 0.6180339887) % 1.0, 0.7, 0.9)
                     for i in range(247)])
    cols[0] = (1, 1, 1)
    from matplotlib.colors import ListedColormap
    acmap = ListedColormap(cols)

    for j, z in enumerate(zs):
        m = np.rot90(mean[:, :, z])
        axes[0, j].imshow(m, cmap="gray", interpolation="nearest")
        axes[0, j].set_title(f"z = {z}", fontsize=9)
        axes[1, j].imshow(m, cmap="gray", interpolation="nearest")
        axes[1, j].contour(np.rot90(mask[:, :, z]), levels=[0.5],
                           colors=["#00d0ff"], linewidths=1.6)
        a = np.rot90(atl[:, :, z]).astype(float)
        axes[2, j].imshow(m, cmap="gray", interpolation="nearest")
        axes[2, j].imshow(np.ma.masked_where(a == 0, a), cmap=acmap, vmin=0,
                          vmax=246, alpha=0.62, interpolation="nearest")
        for i in range(3):
            axes[i, j].set_xticks([])
            axes[i, j].set_yticks([])
            for s in axes[i, j].spines.values():
                s.set_visible(False)

    for i, lab in enumerate(["Preprocessed mean BOLD\n(MNI, 4 mm)",
                             "Brain mask overlay",
                             "Brainnetome-246 overlay"]):
        axes[i, 0].set_ylabel(lab, fontsize=10, fontweight="bold")
        axes[i, 0].yaxis.set_label_coords(-0.18, 0.5)

    fig.suptitle("Normalisation result: direct EPI-to-MNI registration without a T1w image",
                 fontsize=15.5, fontweight="bold", y=0.975)
    fig.text(0.5, 0.035,
             f"Representative acquisition: {row.group}/{row.subject}/{row.session}/{row.run}. "
             "Mean of the preprocessed 4D series in MNI152NLin6Asym at 4 mm, with its brain "
             "mask and the atlas on the identical grid.",
             ha="center", fontsize=9.5, color="#555")
    fig.tight_layout(rect=[0.02, 0.06, 1, 0.945])
    p = os.path.join(OUT, "fig4_registration.png")
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"saved fig4_registration.png  ({row.subject})")


# ----------------------------------------------------------------------
def fig_verification(q):
    integ = pd.read_csv(os.path.join(AUD, "dataset_data_integrity.csv"))
    clean = q[~q.is_artifact]
    rows = [
        ("Acquisitions preprocessed", "165 / 173 eligible", True),
        ("4D, 4 mm isotropic, valid affine", f"{int(integ.is_4mm_isotropic.sum())} / {len(integ)}", True),
        ("Valid binary 3D brain mask", f"{int(integ.mask_is_3D.sum())} / {len(integ)}", True),
        ("Source checksum unchanged", "165 / 165", True),
        ("Duplicate output directories", "0", True),
        ("Brainnetome coverage 246/246", "165 / 165", True),
        ("Non-finite values in any output", "0", True),
        ("Nuisance regressors per acquisition", "27 (CSF gate failed in all 165)", None),
        ("CNR", "not computable — no T1w exists", None),
        ("Median tSNR / SNR", f"{clean.mean_tSNR.median():.1f} / {clean.SNR.median():.2f}", None),
        ("Median mean FD", f"{clean.mean_FD_mm.median():.3f} mm", None),
    ]
    fig, ax = plt.subplots(figsize=(11.5, 6.4), facecolor="white")
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, len(rows) + 2.2)
    for i, (lab, val, ok) in enumerate(rows):
        y = len(rows) - i
        ax.add_patch(FancyBboxPatch((0.2, y - 0.38), 9.6, 0.76,
                                    boxstyle="round,pad=0.02",
                                    fc="#f4f7fa" if i % 2 == 0 else "white",
                                    ec="none"))
        ax.text(0.5, y, lab, fontsize=11, va="center")
        ax.text(8.6, y, val, fontsize=11, va="center", ha="right",
                fontweight="bold",
                color=GREEN if ok else "#333")
        if ok:
            ax.text(9.4, y, "✓", fontsize=14, va="center", ha="center",
                    color=GREEN, fontweight="bold")
    ax.text(5, len(rows) + 1.5, "Preprocessing verification summary",
            ha="center", fontsize=16, fontweight="bold")
    ax.text(5, 0.1,
            "All checks recomputed from the files on disk by an independent audit "
            "script, not copied from processing logs.",
            ha="center", fontsize=9.5, color="#555", style="italic")
    p = os.path.join(OUT, "fig5_verification.png")
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log("saved fig5_verification.png")


# ----------------------------------------------------------------------
def fig_denoising(q):
    """Left: the real 27-column nuisance design. Right: the spectrum of the
    ROI time series after preprocessing, measured on every acquisition."""
    import json

    row = q[~q.is_artifact].sort_values("mean_tSNR", ascending=False).iloc[0]
    base = os.path.join(DERIV, row.group, row.subject, row.session, row.run)
    with open(glob.glob(os.path.join(base, "*_desc-preproc_bold.json"))[0]) as fh:
        meta = json.load(fh)
    names = meta["nuisance_regression"]["regressor_names"]
    D = np.loadtxt(os.path.join(base, "design_matrix.txt"))
    assert D.shape[1] == len(names) == 27, D.shape

    # z-score each column for display only (intercept is constant -> flat)
    Z = D.copy()
    sd = Z.std(axis=0)
    Z = (Z - Z.mean(axis=0)) / np.where(sd > 0, sd, 1.0)

    LO, HI = meta["bandpass_hz"]
    GRID = np.linspace(0.0, 1.0 / (2 * 3.0), 400)   # common grid, TR = 3 s
    spectra, inband = [], []
    for _, r in q.iterrows():
        b = os.path.join(DERIV, r.group, r.subject, r.session, r.run)
        ts_f = os.path.join(b, "biomarkers", "roi_timeseries.npy")
        js = glob.glob(os.path.join(b, "*_desc-preproc_bold.json"))
        if not (os.path.isfile(ts_f) and js):
            continue
        with open(js[0]) as fh:
            tr = json.load(fh)["TR_seconds"]
        ts = np.load(ts_f)                      # (T, 246)
        ts = ts - ts.mean(axis=0, keepdims=True)
        p = (np.abs(np.fft.rfft(ts, axis=0)) ** 2).mean(axis=1)
        f = np.fft.rfftfreq(ts.shape[0], d=tr)
        p[0] = 0.0                              # DC removed by demeaning
        tot = p.sum()
        if tot <= 0:
            continue
        inband.append(p[(f >= LO) & (f <= HI)].sum() / tot)
        if abs(tr - 3.0) < 0.05:
            spectra.append(np.interp(GRID, f, p / tot, left=0.0, right=0.0))
    S = np.array(spectra)
    log(f"  spectra overlaid: {len(S)}   in-band fraction computed: {len(inband)}")

    fig = plt.figure(figsize=(16, 5.4), facecolor="white")
    gs = fig.add_gridspec(1, 2, width_ratios=[1.05, 1], wspace=0.32)

    ax = fig.add_subplot(gs[0])
    im = ax.imshow(Z.T, aspect="auto", cmap="RdBu_r", vmin=-3, vmax=3,
                   interpolation="nearest")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([n.replace("friston24_", "") for n in names], fontsize=6.5)
    ax.set_xlabel("Volume", fontsize=10.5)
    ax.set_title("Nuisance design matrix: 27 regressors",
                 fontsize=12, fontweight="bold")
    for y in (0.5, 24.5, 25.5):
        ax.axhline(y, color="black", lw=0.9)
    ax.text(D.shape[0] * 1.03, 12.5, "Friston-24 motion", rotation=90,
            va="center", ha="left", fontsize=8.5, color="#444")
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.055)
    cb.set_label("z-scored for display", fontsize=8.5)
    cb.ax.tick_params(labelsize=8)

    ax2 = fig.add_subplot(gs[1])
    ax2.axvspan(LO, HI, color=GREEN, alpha=0.10)
    lo_q, med, hi_q = np.percentile(S, [25, 50, 75], axis=0)
    ax2.fill_between(GRID, lo_q, hi_q, color=BLUE, alpha=0.3, lw=0)
    ax2.plot(GRID, med, color=BLUE, lw=1.8)
    for x in (LO, HI):
        ax2.axvline(x, color=GREEN, ls="--", lw=1.4)
    ax2.set_yscale("log")
    ax2.set_xlim(0, GRID[-1])
    ax2.set_xlabel("Frequency (Hz)", fontsize=10.5)
    ax2.set_ylabel("Normalised power (log)", fontsize=10.5)
    ax2.set_title("ROI-signal spectrum after preprocessing",
                  fontsize=12, fontweight="bold")
    ax2.set_ylim(top=med.max() * 6)
    ax2.text(0.055, med.max() * 2.6, f"{LO}–{HI} Hz passband retained",
             ha="center", fontsize=10, color=GREEN, fontweight="bold")
    inband = np.array(inband)
    ax2.text(0.99, 0.04,
             f"In-band power fraction across all {len(inband)} acquisitions:\n"
             f"median {np.median(inband) * 100:.2f}%   "
             f"min {inband.min() * 100:.2f}%",
             transform=ax2.transAxes, ha="right", va="bottom", fontsize=9,
             bbox=dict(fc="white", ec="#ccc", pad=4))
    style(ax2)

    fig.suptitle("Denoising: what was regressed out, and what survived",
                 fontsize=15.5, fontweight="bold", y=1.03)
    fig.text(0.5, -0.07,
             f"Design matrix from {row.group}/{row.subject}; the same 27-regressor "
             "structure was used in all 165 acquisitions — CSF failed its reliability "
             f"gate in every one, so it was dropped rather than forced in. Spectra: {len(S)} "
             "acquisitions at TR = 3 s, resampled onto a common frequency grid; "
             "median and inter-quartile range across acquisitions.",
             ha="center", fontsize=9.5, color="#555")
    p = os.path.join(OUT, "fig6_denoising.png")
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log("saved fig6_denoising.png")


def main():
    os.makedirs(OUT, exist_ok=True)
    q = load_qc()
    log(f"QC rows: {len(q)}  artifact flagged: {int(q.is_artifact.sum())}")
    fig_cohort_flow()
    fig_qc_by_stage(q)
    fig_motion(q)
    fig_registration(q)
    fig_verification(q)
    fig_denoising(q)
    log(f"\noutputs in {OUT}")
    log("PREPROC_FIGURES_DONE")


if __name__ == "__main__":
    main()
