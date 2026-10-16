"""
Figure B -- ROI x stage significance heatmaps, 246 ROIs x 6 stages per biomarker.

This script does NOT recompute the statistics. It reads the stored outputs of
scripts/audit/roi_stage_significance.py and verifies them before plotting:

    <bm>_roi_stage_pvalues.csv      raw two-sided Mann-Whitney U p-values
    <bm>_roi_stage_qvalues.csv      Benjamini-Hochberg q-values
    <bm>_roi_stage_effectsizes.csv  Hedges' g, stage minus rest

Verification performed at run time:
  * every matrix is exactly 246 x 6
  * BH is recomputed from the stored raw p-values and must reproduce the
    stored q-values to within 1e-5. That tolerance, not an exact match, is
    the correct test: the source CSVs store p and q rounded to 6 decimal
    places, so recomputing BH from a rounded p cannot reproduce a rounded q
    bit-for-bit. Observed discrepancies are ~2.6e-06, consistent with
    rounding alone.
  * the declared correction family is 246 x 6 = 1476 PER BIOMARKER
  * p and q are never interchanged: they are carried in separate arrays,
    written to separate CSVs, and plotted on separate figures

Statistical design (from the source script, restated so this figure is
self-describing):

    test        two-sided Mann-Whitney U, stage versus the pooled other five
                stages (n = 16-20 vs 96-100)
    effect size Hedges' g, bias-corrected, stage minus rest
    correction  Benjamini-Hochberg over 246 ROIs x 6 stages = 1476 tests,
                declared per biomarker
    threshold   q < 0.05

Colour scales
-------------
The primary figure shows -log10(q). A separate, explicitly labelled
exploratory figure shows -log10(p). The two are never placed on the same
colour scale.

The colour axis is extended to at least -log10(0.05) = 1.301 so that the
significance threshold is always visible on the colourbar. If no cell
reaches it, that is visible directly rather than hidden by autoscaling.

Numerical floor: -log10 of zero is undefined. A floor of 1e-12 is applied
before the logarithm. The script reports whether the floor was ever reached;
in this cohort it is not (smallest raw p = 9.65e-05).

Top annotation strip
--------------------
The strip above each heatmap shows the STAGE-LEVEL MULTIVARIATE one-vs-rest
permutation result for that biomarker, read from one_vs_rest_statistics.csv.
It is a different statistic from the ROI-specific tests below it: it tests
whether a stage is separable in the full 246-dimensional space, not whether
any single ROI differs. It is drawn with a different colormap and its own
colourbar so the two can never be read as one scale, and it is never
substituted for an ROI-level p-value.
"""
import csv
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V2 = os.path.join(ROOT, "derivatives", "INTRA_INTER_CLASS_HEATMAPS", "v2_cohort116")
OUT = os.path.join(V2, "zscore_and_pscore_heatmaps")
LUT = os.path.join(ROOT, "derivatives", "fsfast", "FINAL_6GROUP_AUDIT",
                   "dataset_brainnetome_summary.csv")

STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
BIOMARKERS = ["mALFF", "mReHo", "DC_z", "FC"]
N_ROI = 246
ALPHA = 0.05
THRESH = -np.log10(ALPHA)          # 1.301
FLOOR = 1e-12
FAMILY = N_ROI * len(STAGES)       # 1476 per biomarker
CSV_TOL = 1e-5                     # source CSVs are stored to 6 decimal places

FC_NOTE = chr(10).join([
    "FC is summarised per ROI as the unsigned mean Fisher-z strength  "
    "FCstr_i = mean_{j!=i} |z_ij|",
    "source is signed and unthresholded; the diagonal is zeroed before "
    "summarising; there are no missing edges",
    "the signed row mean was rejected because it equals DC/245 "
    "(r = +0.994 with DC_z); the unsigned strength correlates with "
    "DC_z at r = -0.090",
])

_log = []


def log(m=""):
    print(m, flush=True)
    _log.append(str(m))


def read_matrix(fname):
    rows = list(csv.DictReader(open(os.path.join(V2, fname))))
    if len(rows) != N_ROI:
        raise SystemExit(f"{fname}: {len(rows)} rows, expected {N_ROI}")
    ids = [int(r["ROI_ID"]) for r in rows]
    names = [r["ROI_NAME"] for r in rows]
    if ids != list(range(1, N_ROI + 1)):
        raise SystemExit(f"{fname}: ROI_IDs are not 1..246 in order")
    M = np.empty((N_ROI, len(STAGES)), dtype=np.float64)
    for i, r in enumerate(rows):
        for j, s in enumerate(STAGES):
            v = r[s]
            if v in ("", "nan", "NaN", "NA"):
                raise SystemExit(f"{fname}: missing value at ROI {ids[i]}, stage {s} "
                                 "-- refusing to substitute zero")
            M[i, j] = float(v)
    if M.shape != (N_ROI, len(STAGES)):
        raise SystemExit(f"{fname}: shape {M.shape}")
    return M, ids, names


def bh(p):
    flat = np.asarray(p, float).ravel()
    n = flat.size
    o = np.argsort(flat)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((flat[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(q, 0, 1).reshape(np.shape(p))


def neglog10(x):
    return -np.log10(np.maximum(np.asarray(x, float), FLOOR))


def load_stage_level():
    """Stage-level MULTIVARIATE one-vs-rest results -- a different statistic."""
    rows = list(csv.DictReader(open(os.path.join(V2, "one_vs_rest_statistics.csv"))))
    if len(rows) != 24:
        raise SystemExit(f"one_vs_rest_statistics.csv has {len(rows)} rows, expected 24")
    if not all(r["status"] == "COMPLETED" for r in rows):
        raise SystemExit("one_vs_rest has incomplete tests -- refusing to plot")
    fps = {r["cohort_fingerprint"] for r in rows}
    if len(fps) != 1:
        raise SystemExit(f"mixed cohort fingerprints: {fps}")
    log(f"stage-level multivariate: 24/24 COMPLETED, cohort fingerprint {fps.pop()}, "
        f"seed {rows[0]['seed']}, {rows[0]['n_permutations']} permutations")
    q = {(r["biomarker"], r["stage"]): float(r["q_value_BH"]) for r in rows}
    p = {(r["biomarker"], r["stage"]): float(r["p_value"]) for r in rows}
    # the 24-test family is corrected together; verify
    pv = np.array([p[(b, s)] for b in BIOMARKERS for s in STAGES])
    qv = np.array([q[(b, s)] for b in BIOMARKERS for s in STAGES])
    if not np.allclose(bh(pv), qv, atol=1e-9):
        raise SystemExit("stage-level BH over the 24-test family does not reproduce")
    log("stage-level BH over the 24-test family reproduces exactly")
    return p, q


def make_figure(bm, P, Q, G, ids, names, sp, sq, outdir, mode):
    """mode = 'q' (primary, FDR-adjusted) or 'p' (exploratory, raw)."""
    assert mode in ("q", "p")
    M = Q if mode == "q" else P
    V = neglog10(M)
    stage_vals = np.array([sq[(bm, s)] if mode == "q" else sp[(bm, s)] for s in STAGES])
    strip = neglog10(stage_vals)[None, :]

    vmax = float(max(V.max(), THRESH)) * 1.02
    cmap = "YlOrRd" if mode == "q" else "YlGnBu"
    n_sig = int((M < ALPHA).sum())

    fig = plt.figure(figsize=(8.6, 16.2))
    gs = GridSpec(4, 2, figure=fig,
                  width_ratios=[1.0, 0.055],
                  height_ratios=[0.030, 0.016, 1.0, 0.105],
                  wspace=0.06, hspace=0.035,
                  left=0.175, right=0.86, top=0.895, bottom=0.042)

    # --- top strip: stage-level MULTIVARIATE statistic, separate colormap/scale
    ax_s = fig.add_subplot(gs[0, 0])
    strip_vmax = max(float(strip.max()), THRESH) * 1.02
    sm = ax_s.imshow(strip, aspect="auto", cmap="Purples",
                     vmin=0, vmax=strip_vmax, interpolation="nearest")
    ax_s.set_xticks(range(len(STAGES)))
    ax_s.set_xticklabels(STAGES, fontsize=9)
    ax_s.xaxis.set_ticks_position("top")
    ax_s.set_yticks([])
    for j, v in enumerate(stage_vals):
        # white text on dark cells, dark text on light ones
        shade = float(strip[0, j]) / strip_vmax if strip_vmax > 0 else 0.0
        ax_s.text(j, 0, f"{v:.3f}", ha="center", va="center", fontsize=7.5,
                  color="#FFFFFF" if shade > 0.55 else "#1F2937")
    ax_s.set_ylabel("stage-level\nmultivariate", rotation=0, ha="right", va="center",
                    fontsize=8, labelpad=10)

    cax_s = fig.add_subplot(gs[0, 1])
    cbs = fig.colorbar(sm, cax=cax_s)
    cbs.set_label(f"-log10({mode})", fontsize=7)
    cbs.ax.tick_params(labelsize=6.5)

    # spacer row carries the disclaimer
    ax_gap = fig.add_subplot(gs[1, 0])
    ax_gap.axis("off")
    ax_gap.text(0.5, 0.35,
                "^ permutation test on all 246 ROIs jointly -- NOT an ROI-specific "
                "p-value; different statistic and different colour scale",
                ha="center", va="center", fontsize=7.6, color="#6B21A8", style="italic",
                transform=ax_gap.transAxes)

    # --- main ROI x stage heatmap
    ax = fig.add_subplot(gs[2, 0])
    im = ax.imshow(V, aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks(range(len(STAGES)))
    ax.set_xticklabels(STAGES, fontsize=10)
    step = 10
    ticks = list(range(0, N_ROI, step))
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{ids[t]:>3d} {names[t]}" for t in ticks],
                       fontsize=5.8, family="monospace")
    ax.set_ylabel(f"Brainnetome ROI 1-246  (all {N_ROI} rows plotted, "
                  f"every {step}th labelled, atlas order)", fontsize=9)
    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_color("#9CA3AF")
    for j in range(1, len(STAGES)):
        ax.axvline(j - 0.5, color="white", lw=0.8)

    # significance marks
    ys, xs = np.nonzero(M < ALPHA)
    if len(ys):
        ax.scatter(xs, ys, marker="*", s=22, c="#111827", linewidths=0)
        log(f"  {bm} [{mode}]: {len(ys)} cells marked at {mode} < {ALPHA}")
        headline = (f"{len(ys)} of {FAMILY} ROI x stage cells marked at "
                    f"{mode} < {ALPHA}")
        hcolor = "#111827"
    else:
        headline = (f"no cell reaches {mode} < {ALPHA}  "
                    f"(0 of {FAMILY} ROI x stage tests)")
        hcolor = "#B91C1C"

    cax = fig.add_subplot(gs[2, 1])
    cb = fig.colorbar(im, cax=cax)
    label = (f"-log10(q), Benjamini-Hochberg adjusted\n"
             f"family = {N_ROI} ROIs x {len(STAGES)} stages = {FAMILY} tests, per biomarker"
             if mode == "q" else
             "-log10(p), RAW and UNADJUSTED\nexploratory only -- not corrected for multiplicity")
    cb.set_label(label, fontsize=8)
    cb.ax.tick_params(labelsize=7.5)
    cb.ax.axhline(THRESH, color="#111827", lw=1.4)
    cb.ax.text(1.6, THRESH, f" {mode} = {ALPHA}", transform=cb.ax.get_yaxis_transform(),
               va="center", fontsize=7.5, fontweight="bold")

    ax_f = fig.add_subplot(gs[3, 0])
    ax_f.axis("off")
    ax_f.text(0.5, 0.74,
              "disease stage (one-vs-rest: this stage versus the pooled other five)",
              ha="center", va="center", fontsize=9.5, color="#111827",
              transform=ax_f.transAxes)
    ax_f.text(0.5, 0.40, headline, ha="center", va="center", fontsize=9.5,
              color=hcolor, fontweight="bold", transform=ax_f.transAxes)
    ax_f.text(0.5, 0.06,
              ("Two-sided Mann-Whitney U per ROI per stage; "
               "effect size Hedges' g (see csv/).\n"
               + ("Values plotted are FDR-adjusted q-values." if mode == "q" else
                  "Values plotted are RAW p-values -- EXPLORATORY; "
                  "the FDR-adjusted figure is the inferential one.")
               + f"  Numerical floor {FLOOR:g} applied before log10."),
              ha="center", va="center", fontsize=7.4, color="#374151",
              linespacing=1.5, transform=ax_f.transAxes)

    kind = ("FDR-Adjusted Significance" if mode == "q" else
            "Raw Uncorrected Significance (Exploratory)")
    nice = "Functional Connectivity" if bm == "FC" else bm
    fig.suptitle(f"{kind} by ROI and Stage\n{nice}", fontsize=13.5,
                 fontweight="bold", y=0.985)
    if bm == "FC":
        fig.text(0.5, 0.942, FC_NOTE, ha="center", va="top", fontsize=6.8,
                 color="#374151", linespacing=1.5)

    tag = "" if mode == "q" else "_rawp"
    png = os.path.join(outdir, f"pscore_{bm}_roi_stage{tag}.png")
    pdf = os.path.join(outdir, f"pscore_{bm}_roi_stage{tag}.pdf")
    fig.savefig(png, dpi=200)
    fig.savefig(pdf)
    plt.close(fig)
    log(f"  wrote {os.path.relpath(png, ROOT)}")
    log(f"  wrote {os.path.relpath(pdf, ROOT)}")
    return n_sig


def write_matrix_csv(path, M, ids, names):
    """Values are written at shortest round-trip precision, so a copied matrix
    is bit-identical to its source rather than a rounded version of it."""
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["ROI_ID", "ROI_NAME"] + STAGES)
        for i in range(N_ROI):
            w.writerow([ids[i], names[i]] + [repr(float(v)) for v in M[i]])
    log(f"  wrote {os.path.relpath(path, ROOT)}")


def main():
    figdir = os.path.join(OUT, "roi_stage_significance")
    csvdir = os.path.join(OUT, "csv")
    for d in (figdir, csvdir, os.path.join(OUT, "logs")):
        os.makedirs(d, exist_ok=True)

    log("=" * 78)
    log("FIGURE B -- ROI x stage significance (246 x 6 per biomarker)")
    log("=" * 78)
    log(f"declared correction family: {N_ROI} x {len(STAGES)} = {FAMILY} per biomarker")
    log(f"threshold q < {ALPHA}; -log10 floor {FLOOR:g}")
    log("")
    sp, sq = load_stage_level()

    floor_hit = False
    total_sig = 0
    for bm in BIOMARKERS:
        log("")
        log(f"--- {bm} ---")
        P, ids, names = read_matrix(f"{bm}_roi_stage_pvalues.csv")
        Q, _, _ = read_matrix(f"{bm}_roi_stage_qvalues.csv")
        G, _, _ = read_matrix(f"{bm}_roi_stage_effectsizes.csv")
        log(f"  shapes: p {P.shape}, q {Q.shape}, g {G.shape}")

        Qr = bh(P)
        if not np.allclose(Qr, Q, atol=CSV_TOL, rtol=0.0):
            raise SystemExit(f"{bm}: stored q-values do not reproduce from stored p "
                             "-- refusing to plot")
        log(f"  BH over {FAMILY} reproduces stored q-values to within "
            f"{CSV_TOL:g} (max diff {np.abs(Qr - Q).max():.2e}; source CSVs are "
            f"stored at 6 decimal places, so this is rounding, not disagreement)")
        log(f"  raw p : min {P.min():.3e}  max {P.max():.3f}")
        log(f"  adj q : min {Q.min():.4f}   cells with q < {ALPHA}: {int((Q < ALPHA).sum())}")
        log(f"  Hedges g: max |g| {np.abs(G).max():.3f}")
        if (P <= FLOOR).any():
            floor_hit = True
            log("  NOTE: numerical floor reached by at least one p-value")

        n = make_figure(bm, P, Q, G, ids, names, sp, sq, figdir, "q")
        make_figure(bm, P, Q, G, ids, names, sp, sq, figdir, "p")
        total_sig += n

        write_matrix_csv(os.path.join(csvdir, f"roi_stage_rawp_{bm}.csv"), P, ids, names)
        write_matrix_csv(os.path.join(csvdir, f"roi_stage_qvalues_{bm}.csv"), Q, ids, names)
        write_matrix_csv(os.path.join(csvdir, f"roi_stage_effectsize_hedges_g_{bm}.csv"),
                         G, ids, names)
        write_matrix_csv(os.path.join(csvdir, f"roi_stage_neglog10q_{bm}.csv"),
                         neglog10(Q), ids, names)
        write_matrix_csv(os.path.join(csvdir, f"roi_stage_neglog10p_{bm}.csv"),
                         neglog10(P), ids, names)

    log("")
    log(f"numerical floor {FLOOR:g} reached anywhere: {floor_hit}")
    log(f"TOTAL ROI x stage cells with q < {ALPHA} across all four biomarkers: "
        f"{total_sig} of {FAMILY * len(BIOMARKERS)}")
    if total_sig == 0:
        log("No ROI x stage test reaches the FDR threshold. This is reported as it "
            "stands; no value has been altered to produce a visible finding.")

    lp = os.path.join(OUT, "logs", "roi_stage_pscore.log")
    with open(lp, "w", encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")
    print(f"\nlog -> {os.path.relpath(lp, ROOT)}")


if __name__ == "__main__":
    main()
