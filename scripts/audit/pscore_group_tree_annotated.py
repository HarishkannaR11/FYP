"""
P-score heatmap with an ANNOTATED diagnostic-group dendrogram.

Each internal merge of the top dendrogram carries two labels:

    d = ...     the MERGE DISTANCE. A geometric quantity, not a p-value.
    q = ...*    a permutation q-value for the contrast that merge represents.

Why the merge distance alone is not enough, and why the q carries an asterisk
----------------------------------------------------------------------------
A dendrogram merge height is a distance. Printing it as a "P-score" would be
a category error, so it is labelled "d =" throughout.

A genuine test for a merge IS definable: the merge joins cluster A and cluster
B, so the question "do the subjects in A differ from those in B?" is a
well-posed two-group comparison. It is answered here with exactly the
machinery already validated elsewhere in this project -- distance-based
pseudo-F (two-group PERMANOVA) on the subject-level 246-D biomarker vectors,
assessed by 9,999 permutations of the group labels, Benjamini-Hochberg
corrected across the 5 merges of that biomarker.

BUT the clusters were chosen by looking at the data. Testing a data-derived
partition on the same data is selective inference ("double dipping"): the
split was selected to separate the groups, so the permutation null -- which
assumes the partition was fixed in advance -- is too generous. The resulting
p-values are ANTI-CONSERVATIVE and are not valid unconditional p-values.

They are therefore marked with an asterisk everywhere they appear and
described as descriptive, not confirmatory. A valid unconditional test would
need either sample splitting (not feasible at n = 116 with six classes) or a
selective-inference correction for hierarchical clustering. Neither is done
here, and no claim of significance is made from them.

Nothing is recomputed from the stored statistics: the heatmap q-values are
read from <bm>_roi_stage_qvalues.csv as before. The merge test is a NEW,
separate analysis on subject-level data, reported separately.
"""
import csv
import json
import os
import sys

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
SRC = os.path.join(ROOT, "derivatives", "model_ready")
BASE = os.path.join(V2, "diagnostic_group_hierarchical")
OUT = os.path.join(BASE, "figures")
CSVD = os.path.join(BASE, "csv")
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
N_PERM, SEED = 9999, 20261016
LONG = {"mALFF": "mALFF", "mReHo": "mReHo", "DC_z": "DC_z",
        "FC": "Functional Connectivity (unsigned mean Fisher-z strength)"}

_log = []


def log(m=""):
    print(m, flush=True)
    _log.append(str(m))


# ------------------------------------------------- statistics (reused design)
def zscore(A):
    sd = A.std(0)
    return (A - A.mean(0)) / np.where(sd > 0, sd, 1.0)


def pseudo_f(Z, mask):
    """Two-group PERMANOVA pseudo-F over the rows of Z selected by mask vs not."""
    n = Z.shape[0]
    n_g = int(mask.sum())
    n_r = n - n_g
    if n_g < 2 or n_r < 2:
        return np.nan
    S = Z.sum(0)
    g = Z.mean(0)
    SST = float(((Z - g) ** 2).sum())
    s_g = Z[mask].sum(0)
    diff = s_g / n_g - (S - s_g) / n_r
    ssb = (n_g * n_r / n) * float(diff @ diff)
    ssw = SST - ssb
    if ssw <= 0:
        return np.nan
    return ssb / (ssw / (n - 2))


def bh(p):
    p = np.asarray(p, float)
    n = p.size
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def merge_sets(L, labels):
    """Leaf sets for each internal node, in merge order."""
    n = len(labels)
    members = {i: {labels[i]} for i in range(n)}
    out = []
    for r, row in enumerate(L):
        a, b = int(row[0]), int(row[1])
        A, B = members[a], members[b]
        members[n + r] = A | B
        out.append({"index": r, "A": sorted(A, key=STAGES.index),
                    "B": sorted(B, key=STAGES.index), "distance": float(row[2])})
    return out


def test_merges(merges, Xbm, y, rng):
    """For each merge, contrast the subjects of cluster A against cluster B."""
    pvals = []
    for m in merges:
        ia = np.array([i for i in range(len(y))
                       if STAGES[y[i]] in m["A"]])
        ib = np.array([i for i in range(len(y))
                       if STAGES[y[i]] in m["B"]])
        idx = np.concatenate([ia, ib])
        Z = zscore(Xbm[idx])
        mask = np.zeros(len(idx), dtype=bool)
        mask[:len(ia)] = True
        obs = pseudo_f(Z, mask)
        null = np.empty(N_PERM)
        for k in range(N_PERM):
            null[k] = pseudo_f(Z, rng.permutation(mask))
        p = (1 + int((null >= obs).sum())) / (N_PERM + 1)
        m.update({"n_A": int(len(ia)), "n_B": int(len(ib)),
                  "pseudo_F": float(obs), "p_perm": float(p)})
        pvals.append(p)
    q = bh(pvals)
    for m, qq in zip(merges, q):
        m["q_BH"] = float(qq)
    return merges


# ------------------------------------------------------------------ plotting
def read_q(bm):
    rows = list(csv.DictReader(open(os.path.join(V2, f"{bm}_roi_stage_qvalues.csv"))))
    Q = np.full((N_ROI, len(STAGES)), np.nan)
    missing = 0
    for i, r in enumerate(rows):
        for j, s in enumerate(STAGES):
            if r[s] in ("", "nan", "NaN", "NA"):
                missing += 1
            else:
                Q[i, j] = float(r[s])
    return Q, missing, [int(r["ROI_ID"]) for r in rows]


def make_figure(bm, P, Q, ids, names, order, L, coph, merges, missing):
    disp = P[:, order]
    n_sig = int(np.nansum(Q < ALPHA))
    obs_max = float(np.nanmax(P))

    fig = plt.figure(figsize=(9.0, 15.2))
    gs = GridSpec(4, 2, figure=fig, width_ratios=[1.0, 0.042],
                  height_ratios=[0.275, 1.0, 0.075, 0.095],
                  wspace=0.05, hspace=0.10,
                  left=0.195, right=0.865, top=0.882, bottom=0.042)

    # --- top dendrogram with annotated merges
    ax_t = fig.add_subplot(gs[0, 0])
    dn = hierarchy.dendrogram(L, ax=ax_t, no_labels=True, color_threshold=0,
                              link_color_func=lambda *a, **k: "#6B7280")
    ax_t.set_xticks([])
    ax_t.set_yticks([])
    for s in ax_t.spines.values():
        s.set_visible(False)

    # dendrogram() returns links in plot order; match them to merge rows by height
    icoord, dcoord = np.array(dn["icoord"]), np.array(dn["dcoord"])
    # Merges can sit at nearly identical heights (e.g. 0.898, 0.899, 0.907),
    # so their labels would overlap. Stack colliding labels onto separate
    # levels: a label takes the lowest level where no already-placed label is
    # within MIN_SEP on the x axis.
    MIN_SEP, BASE, STEP = 46.0, 9.0, 19.0
    placed = []
    items = sorted(((0.5 * (xs[1] + xs[2]), ys[1]) for xs, ys in
                    zip(icoord, dcoord)), key=lambda t: t[0])
    for xm, h in items:
        m = min(merges, key=lambda mm: abs(mm["distance"] - h))
        level = 0
        while any(abs(xm - px) < MIN_SEP and pl == level for px, pl in placed):
            level += 1
        placed.append((xm, level))
        dy = BASE + level * STEP
        ax_t.annotate(f"d = {m['distance']:.3f}", (xm, h),
                      textcoords="offset points", xytext=(0, dy + 8),
                      ha="center", fontsize=6.9, color="#111827",
                      fontweight="bold")
        ax_t.annotate(f"q = {m['q_BH']:.3f}*", (xm, h),
                      textcoords="offset points", xytext=(0, dy - 1),
                      ha="center", fontsize=6.6, color="#6B21A8")
        if level:                       # leader line back to its own bar
            ax_t.annotate("", (xm, h), textcoords="offset points",
                          xytext=(0, dy - 2), arrowprops=dict(
                              arrowstyle="-", lw=0.5, color="#9CA3AF"))
    max_level = max((l for _, l in placed), default=0)
    ax_t.set_ylim(0, max(dcoord[:, 1].max(), 1e-9)
                  * (1.34 + 0.30 * max_level))
    ax_t.set_title(
        f"diagnostic-group clustering by P-SCORE PROFILE  "
        f"({METRIC} distance, {LINKAGE} linkage, cophenetic r = {coph:.3f})"
        + chr(10)
        + "d = merge DISTANCE (geometric, not a p-value)    "
          "q* = permutation q for that contrast, SELECTIVE",
        fontsize=7.8, color="#374151", pad=10)

    # --- heatmap
    ax = fig.add_subplot(gs[1, 0])
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
    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_color("#9CA3AF")

    cax = fig.add_subplot(gs[1, 1])
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

    ax_c = fig.add_subplot(gs[2, 0])
    ax_c.axis("off")
    ax_c.text(0.5, 0.56, "columns ordered by the dendrogram above, NOT by "
              "disease stage", ha="center", va="center", fontsize=8.4,
              color="#111827", transform=ax_c.transAxes)
    ax_c.text(0.5, 0.10, f"clinical progression order:  "
              f"{'  ->  '.join(STAGES)}", ha="center", va="center",
              fontsize=9.0, color="#374151", fontweight="bold",
              transform=ax_c.transAxes)

    ax_f = fig.add_subplot(gs[3, 0])
    ax_f.axis("off")
    head = (f"no ROI reaches q < {ALPHA}: 0 of {N_ROI * len(STAGES)} tests  "
            f"(highest P-score {obs_max:.3f} < {THRESH:.3f})" if n_sig == 0
            else f"{n_sig} of {N_ROI * len(STAGES)} cells reach q < {ALPHA}")
    ax_f.text(0.5, 0.80, head, ha="center", va="center", fontsize=9.0,
              color="#B91C1C" if n_sig == 0 else "#111827", fontweight="bold",
              transform=ax_f.transAxes)
    ax_f.text(0.5, 0.30,
              "Heatmap: two-sided Mann-Whitney U per ROI per stage, "
              "Benjamini-Hochberg over 246 x 6 = 1476 tests.\n"
              "*Branch q: two-group PERMANOVA pseudo-F on subject-level data, "
              f"{N_PERM:,} permutations, BH over the 5 merges. The clusters "
              "were DERIVED FROM THE DATA, so this is\nselective inference -- "
              "the q is anti-conservative and descriptive only. No branch is "
              "claimed to be significant.",
              ha="center", va="center", fontsize=6.9, color="#374151",
              linespacing=1.5, transform=ax_f.transAxes)

    fig.suptitle(f"ROI-Level Stage-vs-Rest Significance\n{LONG[bm]}",
                 fontsize=13.0, fontweight="bold", y=0.983)

    png = os.path.join(OUT, f"pscore_grouptree_annotated_{bm}.png")
    pdf = os.path.join(OUT, f"pscore_grouptree_annotated_{bm}.pdf")
    fig.savefig(png, dpi=220)
    fig.savefig(pdf)
    plt.close(fig)
    log(f"  wrote {os.path.relpath(png, ROOT)}")
    log(f"  wrote {os.path.relpath(pdf, ROOT)}")


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CSVD, exist_ok=True)
    lut = list(csv.DictReader(open(LUT)))
    names = [r["ROI_NAME"] for r in lut]

    reg = np.load(os.path.join(SRC, "X_regional.npy")).astype(np.float64)
    fcu = np.load(os.path.join(SRC, "X_fc_roi_unsigned.npy")).astype(np.float64)
    y = np.load(os.path.join(SRC, "y.npy"))
    X = {"mALFF": reg[:, :, 0], "mReHo": reg[:, :, 1], "DC_z": reg[:, :, 2],
         "FC": fcu}

    log("=" * 82)
    log("ANNOTATED DIAGNOSTIC-GROUP TREE ON THE P-SCORE HEATMAP")
    log("=" * 82)
    log(f"branch labels: d = merge distance (geometric), "
        f"q* = permutation q for the cluster contrast")
    log(f"merge test: two-group PERMANOVA pseudo-F, {N_PERM:,} permutations, "
        f"seed {SEED}, BH over the 5 merges per biomarker")
    log("CAVEAT: clusters are data-derived, so the merge q is SELECTIVE "
        "INFERENCE and anti-conservative")

    rows, total_sig, overall_max = [], 0, 0.0
    for bm in BIOMARKERS:
        log("")
        log(f"--- {bm} ---")
        Q, missing, ids = read_q(bm)
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

        merges = merge_sets(L, STAGES)
        merges = test_merges(merges, X[bm], y,
                             np.random.default_rng(SEED + BIOMARKERS.index(bm)))
        log(f"  leaf order {[STAGES[g] for g in order]}  cophenetic r {coph:.3f}")
        log(f"  {'merge':28s} {'n_A':>4s} {'n_B':>4s} {'dist':>7s} "
            f"{'pseudo-F':>9s} {'p*':>8s} {'q*':>8s}")
        for m in merges:
            lab = "{" + ",".join(m["A"]) + "} + {" + ",".join(m["B"]) + "}"
            log(f"  {lab:28.28s} {m['n_A']:4d} {m['n_B']:4d} "
                f"{m['distance']:7.3f} {m['pseudo_F']:9.3f} "
                f"{m['p_perm']:8.4f} {m['q_BH']:8.4f}")
            rows.append([bm, m["index"], "{" + ",".join(m["A"]) + "}",
                         "{" + ",".join(m["B"]) + "}", m["n_A"], m["n_B"],
                         f"{m['distance']:.6f}", f"{m['pseudo_F']:.6f}",
                         f"{m['p_perm']:.6f}", f"{m['q_BH']:.6f}",
                         bool(m["q_BH"] < ALPHA),
                         "two-group PERMANOVA pseudo-F, 9999 permutations, "
                         "BH over 5 merges",
                         "SELECTIVE: clusters derived from the same data; "
                         "anti-conservative, descriptive only"])
        make_figure(bm, P, Q, ids, names, order, L, coph, merges, missing)

    p = os.path.join(CSVD, "dendrogram_merge_statistics.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["biomarker", "merge_index", "cluster_A", "cluster_B",
                    "n_subjects_A", "n_subjects_B", "merge_distance",
                    "pseudo_F", "p_permutation", "q_BH_over_5_merges",
                    "q_below_0.05", "test_and_correction", "validity_caveat"])
        w.writerows(rows)
    log("")
    log(f"wrote {os.path.relpath(p, ROOT)}  ({len(rows)} merges)")
    log(f"heatmap totals: q < {ALPHA}: {total_sig} of {4 * N_ROI * len(STAGES)}; "
        f"max P-score {overall_max:.4f} (threshold {THRESH:.4f})")
    nmq = sum(1 for r in rows if r[10])
    log(f"merges with selective q < {ALPHA}: {nmq} of {len(rows)} "
        f"(reported as descriptive, NOT as significance)")
    with open(os.path.join(OUT, "pscore_group_tree_annotated.log"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")


if __name__ == "__main__":
    main()
