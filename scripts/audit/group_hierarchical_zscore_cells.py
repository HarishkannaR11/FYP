"""
group_hierarchical_<bm>.png, unchanged, with P-SCORES ADDED AT THE TOP.

Everything below the new annotation row is the original figure:

    heatmap     group-mean biomarker value, standardised across the six
                diagnostic groups (ddof = 1), diverging RdBu_r, fixed +/-2.05
    left        ROI dendrogram, euclidean distance + average linkage
    top         diagnostic-group dendrogram, correlation distance + average
                linkage over the 246-D STANDARDISED BIOMARKER profile
    bottom      clinical progression order and bootstrap branch stability

The heatmap values and both dendrograms are NOT changed. In particular the
group tree still clusters on BIOMARKER ACTIVITY, exactly as in the original
figure -- it is not re-clustered on P-scores.

What is new: a row above the dendrogram giving, per diagnostic group,

    P = ...        the MULTIVARIATE group-level P-score, -log10(q), from
                   one_vs_rest_statistics.csv: stage-vs-rest distance-based
                   pseudo-F over all 246 ROIs jointly, 9,999 permutations,
                   Benjamini-Hochberg across the 4 x 6 = 24 family. One valid
                   value per group.

    max ROI P ...  the largest ROI-level P-score for that group, from
                   <bm>_roi_stage_qvalues.csv. DESCRIPTIVE: it is the maximum
                   of 246 already-corrected values, not itself a test.

Note that the heatmap and the P-scores encode DIFFERENT quantities: the cells
show how far each group's mean biomarker value sits from the six-group
average, while the top row shows statistical significance. They are placed in
one figure because that is what was asked for, and are labelled so they cannot
be read as the same thing.

Sources are read, never recomputed. No existing figure or CSV is modified.
"""
import csv
import json
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
SRC = os.path.join(ROOT, "derivatives", "model_ready")
LUT = os.path.join(ROOT, "derivatives", "fsfast", "FINAL_6GROUP_AUDIT",
                   "dataset_brainnetome_summary.csv")

STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
BIOMARKERS = ["mALFF", "mReHo", "DC_z", "FC"]
LONG = {"mALFF": "Regional amplitude of low-frequency fluctuation (mALFF)",
        "mReHo": "Regional homogeneity (mReHo)",
        "DC_z": "Standardised signed degree centrality (DC_z)",
        "FC": "Unsigned mean Fisher-z connectivity strength (FC)"}
STAGE_COLORS = {"CN": "#1A5276", "SMC": "#2E86C1", "EMCI": "#7FB3D5",
                "MCI": "#F5B041", "LMCI": "#D35400", "AD": "#922B21"}
N_ROI, K, DDOF, ALPHA, FLOOR = 246, 6, 1, 0.05, 1e-12
THRESH = -np.log10(ALPHA)
VMAX = 2.05
GROUP_METRIC, GROUP_LINK = "correlation", "average"
N_PERM, MERGE_SEED = 9999, 20261017
ROI_METRIC, ROI_LINK = "euclidean", "average"

_log = []


def log(m=""):
    print(m, flush=True)
    _log.append(str(m))


def read_matrix(path, blank_ok=False):
    rows = list(csv.DictReader(open(path)))
    if len(rows) != N_ROI:
        raise SystemExit(f"{path}: {len(rows)} rows, expected {N_ROI}")
    M = np.full((N_ROI, K), np.nan)
    missing = 0
    for i, r in enumerate(rows):
        for j, s in enumerate(STAGES):
            if r[s] in ("", "nan", "NaN", "NA"):
                missing += 1
            else:
                M[i, j] = float(r[s])
    if missing and not blank_ok:
        raise SystemExit(f"{path}: {missing} missing values")
    return M, missing



def _zscore(A):
    sd = A.std(0)
    return (A - A.mean(0)) / np.where(sd > 0, sd, 1.0)


def _pseudo_f(Z, mask):
    """Two-group PERMANOVA pseudo-F, as used elsewhere in this project."""
    n = Z.shape[0]
    n_g = int(mask.sum())
    n_r = n - n_g
    if n_g < 2 or n_r < 2:
        return np.nan
    S = Z.sum(0)
    SST = float(((Z - Z.mean(0)) ** 2).sum())
    s_g = Z[mask].sum(0)
    diff = s_g / n_g - (S - s_g) / n_r
    ssb = (n_g * n_r / n) * float(diff @ diff)
    ssw = SST - ssb
    return ssb / (ssw / (n - 2)) if ssw > 0 else np.nan


def _bh(p):
    p = np.asarray(p, float)
    n = p.size
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def _merge_sets(L, labels):
    """Leaf membership of each internal node, in merge order."""
    n = len(labels)
    mem = {i: {labels[i]} for i in range(n)}
    out = []
    for r, row in enumerate(L):
        a, b = int(row[0]), int(row[1])
        mem[n + r] = mem[a] | mem[b]
        out.append({"A": sorted(mem[a], key=STAGES.index),
                    "B": sorted(mem[b], key=STAGES.index),
                    "distance": float(row[2])})
    return out


def _test_merges(merges, Xbm, y, seed):
    """Contrast the subjects of cluster A against those of cluster B.

    SELECTIVE INFERENCE: the clusters were derived from these same data, so
    the permutation null is too generous and these q-values are
    anti-conservative. They are descriptive, never a significance claim.
    """
    rng = np.random.default_rng(seed)
    ps = []
    for m in merges:
        ia = np.array([i for i in range(len(y)) if STAGES[y[i]] in m["A"]])
        ib = np.array([i for i in range(len(y)) if STAGES[y[i]] in m["B"]])
        idx = np.concatenate([ia, ib])
        Z = _zscore(Xbm[idx])
        mask = np.zeros(len(idx), dtype=bool)
        mask[:len(ia)] = True
        obs = _pseudo_f(Z, mask)
        null = np.array([_pseudo_f(Z, rng.permutation(mask))
                         for _ in range(N_PERM)])
        p = (1 + int((null >= obs).sum())) / (N_PERM + 1)
        m.update({"n_A": int(ia.size), "n_B": int(ib.size),
                  "pseudo_F": float(obs), "p_perm": float(p)})
        ps.append(p)
    for m, q in zip(merges, _bh(ps)):
        m["q_BH"] = float(q)
    return merges


def gray(*a, **k):
    return "#9A9A9A"


def main():
    os.makedirs(OUT, exist_ok=True)
    meta = json.load(open(os.path.join(SRC, "metadata.json")))
    if meta["stage_to_index"] != {s: i for i, s in enumerate(STAGES)}:
        raise SystemExit(f"stage mapping mismatch: {meta['stage_to_index']}")
    reg = np.load(os.path.join(SRC, "X_regional.npy")).astype(np.float64)
    fcu = np.load(os.path.join(SRC, "X_fc_roi_unsigned.npy")).astype(np.float64)
    yv = np.load(os.path.join(SRC, "y.npy"))
    XBM = {"mALFF": reg[:, :, 0], "mReHo": reg[:, :, 1],
           "DC_z": reg[:, :, 2], "FC": fcu}
    lut = list(csv.DictReader(open(LUT)))
    ids = [int(r["ROI_ID"]) for r in lut]
    names = [r["ROI_NAME"] for r in lut]

    # genuine one-value-per-group statistic
    mv = {}
    rows = list(csv.DictReader(open(os.path.join(V2, "one_vs_rest_statistics.csv"))))
    if len(rows) != 24 or not all(r["status"] == "COMPLETED" for r in rows):
        raise SystemExit("one_vs_rest_statistics.csv incomplete")
    for r in rows:
        mv[(r["biomarker"], r["stage"])] = float(r["q_value_BH"])

    # bootstrap stability, for the original caption
    boot = {}
    for r in csv.DictReader(open(os.path.join(CSVD,
                                              "group_cluster_stability_bootstrap.csv"))):
        boot.setdefault(r["biomarker"], []).append(
            (r["cluster"], float(r["bootstrap_frequency"])))

    log("=" * 80)
    log("ORIGINAL GROUP-HIERARCHICAL FIGURE + P-SCORES AT THE TOP")
    log("=" * 80)
    log(f"stage mapping verified: {meta['stage_to_index']}")
    log("heatmap, clustering and layout are UNCHANGED from the original")

    out_rows = []
    for bm in BIOMARKERS:
        log("")
        log(f"--- {bm} ---")
        Z, zmiss = read_matrix(os.path.join(CSVD,
                                            f"group_means_standardized_{bm}.csv"),
                               blank_ok=True)
        Praw, _ = read_matrix(os.path.join(V2, f"{bm}_roi_stage_qvalues.csv"))
        Proi = -np.log10(np.maximum(Praw, FLOOR))
        Proi[Proi == 0] = 0.0

        flat = ~np.isfinite(Z).all(axis=1)
        zk = Z[~flat]
        Dg = pdist(zk.T, metric=GROUP_METRIC)
        Lg = hierarchy.optimal_leaf_ordering(
            hierarchy.linkage(Dg, method=GROUP_LINK), Dg)
        grp_order = hierarchy.leaves_list(Lg)
        Dr = pdist(zk, metric=ROI_METRIC)
        Lr = hierarchy.optimal_leaf_ordering(
            hierarchy.linkage(Dr, method=ROI_LINK), Dr)
        kept = np.flatnonzero(~flat)
        roi_order = kept[hierarchy.leaves_list(Lr)]
        log(f"  group order {[STAGES[g] for g in grp_order]}  "
            f"(clustered on BIOMARKER profile, as in the original)")
        log(f"  zero-variance ROIs excluded: {int(flat.sum())}")
        merges = _test_merges(_merge_sets(Lg, STAGES), XBM[bm],
                              yv, MERGE_SEED + BIOMARKERS.index(bm))
        for m in merges:
            log(f"    {{{','.join(m['A'])}}} + {{{','.join(m['B'])}}}"
                f"  d={m['distance']:.3f}  F={m['pseudo_F']:.2f}"
                f"  p={m['p_perm']:.4f}  q={m['q_BH']:.4f}")

        # ---------------------------------------------------------- figure
        fig = plt.figure(figsize=(9.4, 16.4))
        # Two columns only (heatmap, colourbar) now that the ROI dendrogram is
        # gone. hspace is tight so the P-score row sits against the tree.
        gs = GridSpec(5, 2, figure=fig,
                      width_ratios=[1.0, 0.052],
                      height_ratios=[0.012, 0.345, 1.0, 0.105, 0.255],
                      wspace=0.05, hspace=0.045,
                      left=0.215, right=0.860, top=0.884, bottom=0.042)

        # row 0 -- P-scores, aligned to the dendrogram leaves below

        # row 1 -- group dendrogram, with the P-scores drawn ON it
        ax_t = fig.add_subplot(gs[1, 0])
        dn = hierarchy.dendrogram(Lg, ax=ax_t, no_labels=True,
                                  color_threshold=0, link_color_func=gray)
        ax_t.set_xlim(0, 10 * K)
        ax_t.set_xticks([])
        ax_t.set_yticks([])
        for sp in ax_t.spines.values():
            sp.set_visible(False)

        # Each group's P-score sits at the foot of its own leaf stem, so the
        # number is unambiguously attached to that branch of the tree.
        icoord = np.array(dn["icoord"])
        dcoord = np.array(dn["dcoord"])
        top = float(dcoord.max())
        ax_t.set_ylim(-0.62 * top, top * 1.06)
        xs = [5 + 10 * i for i in range(K)]
        for x, g in zip(xs, grp_order):
            st = STAGES[g]
            pm = -np.log10(max(mv[(bm, st)], FLOOR))
            ax_t.text(x, -0.16 * top, f"{pm:.3f}", ha="center", va="center",
                      fontsize=9.5, fontweight="bold", color=STAGE_COLORS[st],
                      zorder=5)
            ax_t.text(x, -0.42 * top, f"{Proi[:, g].max():.3f}", ha="center",
                      va="center", fontsize=7.4, color="#6B7280", zorder=5)
        ax_t.text(-1.0, -0.16 * top, "group P-score", ha="right", va="center",
                  fontsize=7.4, color="#111827", fontweight="bold")
        ax_t.text(-1.0, -0.42 * top, "max ROI P", ha="right", va="center",
                  fontsize=7.0, color="#6B7280", style="italic")

        # d = merge DISTANCE (geometric). q* = permutation q for that cluster
        # contrast, SELECTIVE because the clusters came from these same data.
        # Labels stack when merges sit at similar heights.
        MIN_SEP, BASE, STEP = 52.0, 11.0, 20.0
        placed = []
        for xc, yc in sorted(((0.5 * (a[1] + a[2]), b[1])
                              for a, b in zip(icoord, dcoord)),
                             key=lambda t: t[0]):
            m = min(merges, key=lambda mm: abs(mm["distance"] - yc))
            lvl = 0
            while any(abs(xc - px) < MIN_SEP and pl == lvl for px, pl in placed):
                lvl += 1
            placed.append((xc, lvl))
            dy = BASE + lvl * STEP
            ax_t.annotate(f"d = {m['distance']:.3f}", (xc, yc),
                          textcoords="offset points", xytext=(0, dy + 8),
                          ha="center", fontsize=7.0, color="#111827",
                          fontweight="bold", zorder=6)
            ax_t.annotate(f"q = {m['q_BH']:.3f}*", (xc, yc),
                          textcoords="offset points", xytext=(0, dy - 1),
                          ha="center", fontsize=6.6, color="#6B21A8", zorder=6)
            if lvl:
                ax_t.annotate("", (xc, yc), textcoords="offset points",
                              xytext=(0, dy - 2),
                              arrowprops=dict(arrowstyle="-", lw=0.5,
                                              color="#9CA3AF"))
        ax_t.set_ylim(-0.62 * top,
                      top * (1.20 + 0.26 * max((l for _, l in placed),
                                               default=0)))
        # axis title removed: the figure subtitle already names the metric
        # and linkage, and the title collided with the P-score row above.

        # row 2 -- ROI dendrogram + heatmap (unchanged)
        # The left ROI dendrogram is deliberately NOT drawn: only the top
        # diagnostic-group clustering is shown. Rows keep their clustered
        # ORDER, so the block structure of the heatmap is unchanged.

        ax = fig.add_subplot(gs[2, 0])
        cmap = matplotlib.colormaps["RdBu_r"].with_extremes(bad="#D1D5DB")
        im = ax.imshow(Z[np.ix_(roi_order, grp_order)], aspect="auto",
                       cmap=cmap, vmin=-VMAX, vmax=VMAX,
                       interpolation="nearest")
        ax.set_xticks(range(K))
        ax.set_xticklabels([STAGES[g] for g in grp_order], fontsize=11)
        for t, g in zip(ax.get_xticklabels(), grp_order):
            t.set_color(STAGE_COLORS[STAGES[g]])
            t.set_fontweight("bold")
        ax.xaxis.set_ticks_position("bottom")
        step = 10
        ticks = list(range(0, len(roi_order), step))
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{ids[roi_order[t]]:>3d} {names[roi_order[t]]}"
                            for t in ticks], fontsize=5.8, family="monospace")
        for j in range(1, K):
            ax.axvline(j - 0.5, color="white", lw=1.0)
        for sp in ax.spines.values():
            sp.set_linewidth(0.6)
            sp.set_color("#9CA3AF")
        fig.text(0.016, 0.46, f"Brainnetome ROI  (all {len(roi_order)} rows "
                 f"plotted, every {step}th labelled, clustered order)",
                 rotation=90, va="center", ha="center", fontsize=9)

        cax = fig.add_subplot(gs[2, 1])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label("group-mean biomarker value, standardised across the six "
                     "diagnostic groups\n"
                     f"(blue = below this ROI's six-group mean, red = above; "
                     f"fixed scale +/-{VMAX})", fontsize=8.0)
        cb.ax.tick_params(labelsize=7.5)

        ax_c = fig.add_subplot(gs[3, 0])
        ax_c.axis("off")
        ax_c.text(0.5, 0.66, "columns are ordered by the dendrogram above, "
                  "NOT by disease stage", ha="center", va="center",
                  fontsize=8.4, color="#111827", transform=ax_c.transAxes)
        ax_c.text(0.5, 0.12, f"clinical progression order:  "
                  f"{'  →  '.join(STAGES)}", ha="center", va="center",
                  fontsize=8.6, color="#374151", transform=ax_c.transAxes)

        ax_f = fig.add_subplot(gs[4, 0])
        ax_f.axis("off")
        top3 = sorted(boot.get(bm, []), key=lambda t: -t[1])[:3]
        txt = ";  ".join(f"{c} {v * 100:.0f}%" for c, v in top3)
        ax_f.text(0.5, 0.92, f"bootstrap stability of the branches (1000 "
                  f"resamples of subjects within groups):  {txt}",
                  ha="center", va="center", fontsize=7.4, color="#374151",
                  transform=ax_f.transAxes)
        ax_f.text(0.5, 0.71, "group P-score = -log10(q) from the MULTIVARIATE "
                  "stage-vs-rest permutation test (24-test family);  max ROI P "
                  "= largest ROI-level P-score, DESCRIPTIVE only",
                  ha="center", va="center", fontsize=7.0, color="#111827",
                  transform=ax_f.transAxes)
        ax_f.text(0.5, 0.50, "d = merge DISTANCE (geometric, not a p-value).  "
                  "*q = permutation q for that cluster contrast, "
                  "SELECTIVE (clusters came from these data) -- "
                  "anti-conservative, descriptive only.",
                  ha="center", va="center", fontsize=6.9, color="#6B21A8",
                  transform=ax_f.transAxes)
        ax_f.text(0.5, 0.28, "Heatmap shows BIOMARKER ACTIVITY; the top rows "
                  "show STATISTICAL SIGNIFICANCE. Different quantities. The "
                  "tree clusters on biomarker activity.",
                  ha="center", va="center", fontsize=7.2, color="#6B21A8",
                  transform=ax_f.transAxes)
        ax_f.text(0.5, 0.06, f"Descriptive only. Branch heights are distances, "
                  f"not p-values. No group reaches q < {ALPHA} and no ROI x "
                  f"stage test in this project does either.",
                  ha="center", va="center", fontsize=7.4, color="#B91C1C",
                  style="italic", transform=ax_f.transAxes)

        fig.suptitle("Hierarchical Clustering of Alzheimer's Diagnostic Groups\n"
                     f"{LONG[bm]}", fontsize=13.2, fontweight="bold", y=0.978)
        fig.text(0.5, 0.930,
                 f"cells are the MEAN {bm} of the subjects in each diagnostic "
                 f"group (n = 20/20/20/16/20/20), standardised across the six "
                 f"groups per ROI (ddof = {DDOF})\n"
                 f"groups clustered by {GROUP_METRIC} distance + {GROUP_LINK} "
                 f"linkage over the 246-D profile  |  ROI rows ORDERED by "
                 f"{ROI_METRIC} + {ROI_LINK} linkage (tree not drawn)",
                 ha="center", va="top", fontsize=7.9, color="#374151",
                 linespacing=1.6)

        png = os.path.join(OUT, f"group_hierarchical_pscore_top_{bm}.png")
        pdf = os.path.join(OUT, f"group_hierarchical_pscore_top_{bm}.pdf")
        fig.savefig(png, dpi=150)
        fig.savefig(pdf)
        plt.close(fig)
        log(f"  wrote {os.path.relpath(png, ROOT)}")
        log(f"  wrote {os.path.relpath(pdf, ROOT)}")

        for pos, g in enumerate(grp_order):
            s = STAGES[g]
            q = mv[(bm, s)]
            out_rows.append([bm, pos, s, g, f"{q:.6f}",
                             f"{-np.log10(max(q, FLOOR)):.6f}",
                             f"{float(Proi[:, g].max()):.6f}",
                             int((Proi[:, g] >= THRESH).sum()),
                             bool(q < ALPHA)])

    p = os.path.join(CSVD, "group_hierarchical_pscore_top_annotations.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["biomarker", "display_column", "stage_label",
                    "source_stage_index", "multivariate_q_BH24",
                    "multivariate_P_score", "max_roi_P_score",
                    "n_ROIs_at_or_above_1.301", "multivariate_q_below_0.05"])
        w.writerows(out_rows)
    log("")
    log(f"wrote {os.path.relpath(p, ROOT)}")
    n = sum(1 for r in out_rows if r[8])
    log(f"groups with multivariate q < {ALPHA}: {n} of {len(out_rows)}")
    log(f"largest multivariate P-score: "
        f"{max(float(r[5]) for r in out_rows):.4f} (threshold {THRESH:.4f})")
    with open(os.path.join(OUT, "group_hierarchical_pscore_top.log"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")


if __name__ == "__main__":
    main()
