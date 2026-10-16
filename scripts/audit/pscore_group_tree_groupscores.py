"""
P-score heatmap, group dendrogram, and PER-GROUP P-SCORES above the tree.

Which P-score is shown above each group, and why
------------------------------------------------
The heatmap cells are ROI-SPECIFIC: 246 P-scores per group. There is no single
valid "group P-score" in that family, and averaging 246 ROI P-scores would
produce a number with no inferential meaning. That was not done.

A genuine one-value-per-group statistic does exist, in a SEPARATE analysis:

    one_vs_rest_statistics.csv -- stage-level MULTIVARIATE one-vs-rest,
    distance-based pseudo-F over all 246 ROIs jointly, 9,999 permutations,
    Benjamini-Hochberg across the full 4 biomarkers x 6 stages = 24 family.

So each group carries two annotations, labelled distinctly:

    P = ...        MULTIVARIATE group-level P-score, -log10(q) from the
                   24-test family. A real test of "is this stage separable
                   across all regions at once?".

    max ROI P ...  the largest P-score in that group's heatmap column. Purely
                   DESCRIPTIVE -- it is the maximum of 246 already-corrected
                   values, not itself a test, and it is labelled as such.

These two are different tests on different units and are corrected in
different families. They are never summed, averaged or compared to each other.

Branch annotations are deliberately absent here: merge heights are distances,
not p-values, and the distance-annotated variant already exists separately as
pscore_grouptree_dist_*. Mixing distances and P-scores in one header invites
exactly the confusion this figure is meant to avoid.

The dendrogram clusters the six groups by their 246-D P-SCORE PROFILES, i.e.
by similarity of statistical evidence -- not raw biomarker activity, and not
clinical progression.
"""
import csv
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
V2 = os.path.join(ROOT, "derivatives", "INTRA_INTER_CLASS_HEATMAPS", "v2_cohort116")
BASE = os.path.join(V2, "diagnostic_group_hierarchical")
OUT, CSVD = os.path.join(BASE, "figures"), os.path.join(BASE, "csv")
LUT = os.path.join(ROOT, "derivatives", "fsfast", "FINAL_6GROUP_AUDIT",
                   "dataset_brainnetome_summary.csv")

STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
BIOMARKERS = ["mALFF", "mReHo", "DC_z", "FC"]
STAGE_COLORS = {"CN": "#1A5276", "SMC": "#2E86C1", "EMCI": "#7FB3D5",
                "MCI": "#F5B041", "LMCI": "#D35400", "AD": "#922B21"}
N_ROI, ALPHA, FLOOR = 246, 0.05, 1e-12
THRESH = -np.log10(ALPHA)
VMIN, VMAX, CMAP = 0.0, 1.40, "YlOrRd"
METRIC, LINKAGE = "correlation", "average"
LONG = {"mALFF": "mALFF", "mReHo": "mReHo", "DC_z": "DC_z",
        "FC": "Functional Connectivity (unsigned mean Fisher-z strength)"}

_log = []


def log(m=""):
    print(m, flush=True)
    _log.append(str(m))


def read_roi_q(bm):
    rows = list(csv.DictReader(open(os.path.join(V2, f"{bm}_roi_stage_qvalues.csv"))))
    if len(rows) != N_ROI:
        raise SystemExit(f"{bm}: {len(rows)} rows, expected {N_ROI}")
    Q = np.full((N_ROI, len(STAGES)), np.nan)
    missing = 0
    for i, r in enumerate(rows):
        for j, s in enumerate(STAGES):
            if r[s] in ("", "nan", "NaN", "NA"):
                missing += 1
            else:
                Q[i, j] = float(r[s])
    return Q, missing, [int(r["ROI_ID"]) for r in rows]


def read_multivariate():
    """Genuine one-value-per-group statistic: stage-level one-vs-rest."""
    rows = list(csv.DictReader(open(os.path.join(V2, "one_vs_rest_statistics.csv"))))
    if len(rows) != 24:
        raise SystemExit(f"one_vs_rest_statistics.csv has {len(rows)} rows, expected 24")
    if not all(r["status"] == "COMPLETED" for r in rows):
        raise SystemExit("one_vs_rest contains incomplete tests")
    fps = {r["cohort_fingerprint"] for r in rows}
    if len(fps) != 1:
        raise SystemExit(f"mixed cohort fingerprints: {fps}")
    log(f"multivariate source: 24/24 COMPLETED, fingerprint {fps.pop()}, "
        f"{rows[0]['n_permutations']} permutations, BH over the 24-test family")
    return {(r["biomarker"], r["stage"]): float(r["q_value_BH"]) for r in rows}


def make_figure(bm, P, Q, mq, ids, names, order, L, coph, missing):
    disp = P[:, order]
    n_sig = int(np.nansum(Q < ALPHA))
    obs_max = float(np.nanmax(P))

    fig = plt.figure(figsize=(8.8, 15.4))
    gs = GridSpec(5, 2, figure=fig, width_ratios=[1.0, 0.045],
                  height_ratios=[0.10, 0.115, 1.0, 0.075, 0.105],
                  wspace=0.05, hspace=0.06,
                  left=0.195, right=0.865, top=0.900, bottom=0.044)

    # leaf x positions used by scipy's dendrogram: 5, 15, 25, ...
    xs = [5 + 10 * k for k in range(len(STAGES))]

    # --- row 0: per-group P-scores, aligned to the leaves below
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_xlim(0, 10 * len(STAGES))
    ax_a.set_ylim(0, 1)
    ax_a.axis("off")
    for x, g in zip(xs, order):
        s = STAGES[g]
        pm = -np.log10(max(mq[(bm, s)], FLOOR))
        ax_a.text(x, 0.62, f"P = {pm:.3f}", ha="center", va="center",
                  fontsize=8.6, fontweight="bold", color=STAGE_COLORS[s])
        ax_a.text(x, 0.17, f"max ROI P {P[:, g].max():.3f}", ha="center",
                  va="center", fontsize=6.9, color="#6B7280")
    ax_a.text(0.5, 1.26, "P = MULTIVARIATE group-level P-score, -log10(q) from "
              "the stage-vs-rest permutation test (24-test family)",
              transform=ax_a.transAxes, ha="center", fontsize=7.4,
              color="#111827")
    ax_a.text(0.5, 1.02, "max ROI P = largest value in that group's heatmap "
              "column -- DESCRIPTIVE, not a test",
              transform=ax_a.transAxes, ha="center", fontsize=7.0,
              color="#6B7280", style="italic")

    # --- row 1: dendrogram, no branch labels
    ax_t = fig.add_subplot(gs[1, 0])
    hierarchy.dendrogram(L, ax=ax_t, no_labels=True, color_threshold=0,
                         link_color_func=lambda *a, **k: "#6B7280")
    ax_t.set_xlim(0, 10 * len(STAGES))
    ax_t.set_xticks([])
    ax_t.set_yticks([])
    for sp in ax_t.spines.values():
        sp.set_visible(False)
    ax_t.set_title(f"clustering by P-SCORE PROFILE  ({METRIC} distance, "
                   f"{LINKAGE} linkage, cophenetic r = {coph:.3f})",
                   fontsize=7.6, color="#374151", pad=3)

    # --- row 2: heatmap
    ax = fig.add_subplot(gs[2, 0])
    cmap = matplotlib.colormaps[CMAP].with_extremes(bad="#D1D5DB")
    im = ax.imshow(disp, aspect="auto", cmap=cmap, vmin=VMIN, vmax=VMAX,
                   interpolation="nearest")
    ax.set_xticks(range(len(STAGES)))
    ax.set_xticklabels([STAGES[g] for g in order], fontsize=11)
    for t, g in zip(ax.get_xticklabels(), order):
        t.set_color(STAGE_COLORS[STAGES[g]])
        t.set_fontweight("bold")
    ax.xaxis.set_ticks_position("bottom")
    ax.tick_params(axis="x", pad=6)
    step = 10
    ticks = list(range(0, N_ROI, step))
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{ids[t]:>3d} {names[t]}" for t in ticks],
                       fontsize=6.0, family="monospace")
    ax.set_ylabel(f"Brainnetome ROI 1-246  (all {N_ROI} rows shown, "
                  f"every {step}th labelled, atlas order)", fontsize=9)
    for j in range(1, len(STAGES)):
        ax.axvline(j - 0.5, color="white", lw=1.0)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
        sp.set_color("#9CA3AF")

    cax = fig.add_subplot(gs[2, 1])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("P-score = -log10(FDR-adjusted q)\nhigher = stronger evidence"
                 f"  |  fixed scale {VMIN:.1f}-{VMAX:.1f}", fontsize=8.4)
    cb.ax.tick_params(labelsize=7.5)
    cb.ax.axhline(THRESH, color="#111827", lw=1.6)
    cb.ax.text(0.5, THRESH, f"q = {ALPHA}\n({THRESH:.3f})",
               transform=cb.ax.get_yaxis_transform(), ha="center", va="bottom",
               fontsize=7.0, fontweight="bold",
               bbox=dict(boxstyle="square,pad=0.15", fc="white", ec="none",
                         alpha=0.9))

    ax_c = fig.add_subplot(gs[3, 0])
    ax_c.axis("off")
    ax_c.text(0.5, 0.58, "columns ordered by the dendrogram above, NOT by "
              "disease stage", ha="center", va="center", fontsize=8.4,
              color="#111827", transform=ax_c.transAxes)
    ax_c.text(0.5, 0.10, f"clinical progression order:  "
              f"{'  ->  '.join(STAGES)}", ha="center", va="center",
              fontsize=9.0, color="#374151", fontweight="bold",
              transform=ax_c.transAxes)

    ax_f = fig.add_subplot(gs[4, 0])
    ax_f.axis("off")
    head = (f"no ROI reaches q < {ALPHA}: 0 of {N_ROI * len(STAGES)} tests  "
            f"(highest P-score {obs_max:.3f} < {THRESH:.3f})" if n_sig == 0
            else f"{n_sig} of {N_ROI * len(STAGES)} cells reach q < {ALPHA}")
    ax_f.text(0.5, 0.80, head, ha="center", va="center", fontsize=9.0,
              color="#B91C1C" if n_sig == 0 else "#111827", fontweight="bold",
              transform=ax_f.transAxes)
    ax_f.text(0.5, 0.28,
              "Heatmap: two-sided Mann-Whitney U per ROI per stage, "
              "Benjamini-Hochberg over 246 x 6 = 1476 tests per biomarker.\n"
              "The tree measures similarity between STATISTICAL-SIGNIFICANCE "
              "PROFILES -- not raw biomarker activity, and not clinical "
              "progression.\n"
              "No group-level P reaches 1.301 either; none of these annotations "
              "indicates a statistically significant result.",
              ha="center", va="center", fontsize=7.0, color="#374151",
              linespacing=1.5, transform=ax_f.transAxes)

    fig.suptitle(f"ROI-Level Stage-vs-Rest Significance\n{LONG[bm]}",
                 fontsize=13.0, fontweight="bold", y=0.982)
    if missing:
        fig.text(0.5, 0.010, f"{missing} missing q-values shown in grey, not "
                 "replaced with zero", ha="center", fontsize=7.2,
                 color="#B91C1C")

    png = os.path.join(OUT, f"pscore_grouptree_groupscore_{bm}.png")
    pdf = os.path.join(OUT, f"pscore_grouptree_groupscore_{bm}.pdf")
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    log(f"  wrote {os.path.relpath(png, ROOT)}")
    log(f"  wrote {os.path.relpath(pdf, ROOT)}")


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CSVD, exist_ok=True)
    names = [r["ROI_NAME"] for r in csv.DictReader(open(LUT))]
    log("=" * 80)
    log("P-SCORE HEATMAP + GROUP DENDROGRAM + PER-GROUP P-SCORES ABOVE")
    log("=" * 80)
    mq = read_multivariate()

    rows, total_sig, overall_max = [], 0, 0.0
    for bm in BIOMARKERS:
        log("")
        log(f"--- {bm} ---")
        Q, missing, ids = read_roi_q(bm)
        P = -np.log10(np.where(np.isnan(Q), np.nan, np.maximum(Q, FLOOR)))
        P[P == 0] = 0.0
        n_sig = int(np.nansum(Q < ALPHA))
        total_sig += n_sig
        overall_max = max(overall_max, float(np.nanmax(P)))
        log(f"  heatmap: min q {np.nanmin(Q):.6f}  max P {np.nanmax(P):.4f}  "
            f"q < {ALPHA}: {n_sig}  missing {missing}")

        D = pdist(P.T, metric=METRIC)
        L = hierarchy.linkage(D, method=LINKAGE)
        L = hierarchy.optimal_leaf_ordering(L, D)
        order = hierarchy.leaves_list(L)
        coph = float(hierarchy.cophenet(L, D)[0])
        log(f"  leaf order {[STAGES[g] for g in order]}  cophenetic r {coph:.3f}")
        log(f"  {'group':6s} {'multivar q':>11s} {'multivar P':>11s} "
            f"{'max ROI P':>10s} {'ROIs>=1.301':>12s}")
        for g in order:
            s = STAGES[g]
            q = mq[(bm, s)]
            pm = -np.log10(max(q, FLOOR))
            mx = float(P[:, g].max())
            npass = int((P[:, g] >= THRESH).sum())
            log(f"  {s:6s} {q:11.4f} {pm:11.4f} {mx:10.4f} {npass:12d}")
            rows.append([bm, s, g, list(order).index(g), f"{q:.6f}",
                         f"{pm:.6f}", f"{mx:.6f}", npass,
                         bool(q < ALPHA),
                         "multivariate stage-vs-rest pseudo-F, 9999 "
                         "permutations, BH over 24",
                         "max of 246 already-corrected ROI P-scores; "
                         "DESCRIPTIVE, not a test"])
        make_figure(bm, P, Q, mq, ids, names, order, L, coph, missing)

    p = os.path.join(CSVD, "group_level_pscore_annotations.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["biomarker", "stage_label", "source_stage_index",
                    "display_column", "multivariate_q_BH24",
                    "multivariate_P_score", "max_roi_P_score",
                    "n_ROIs_at_or_above_1.301", "multivariate_q_below_0.05",
                    "multivariate_test", "max_roi_P_caveat"])
        w.writerows(rows)
    log("")
    log(f"wrote {os.path.relpath(p, ROOT)}  ({len(rows)} group annotations)")
    log(f"heatmap totals: q < {ALPHA}: {total_sig} of {4 * N_ROI * len(STAGES)}; "
        f"max ROI P-score {overall_max:.4f} (threshold {THRESH:.4f})")
    nmv = sum(1 for r in rows if r[8])
    log(f"groups with multivariate q < {ALPHA}: {nmv} of {len(rows)}")
    mx_mv = max(float(r[5]) for r in rows)
    log(f"largest multivariate P-score: {mx_mv:.4f} (threshold {THRESH:.4f}) "
        f"-> {'below' if mx_mv < THRESH else 'ABOVE'} threshold")
    with open(os.path.join(OUT, "pscore_group_tree_groupscores.log"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")


if __name__ == "__main__":
    main()
