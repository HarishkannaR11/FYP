"""
ROI x stage one-vs-rest significance: 246 ROIs x 6 stages, per biomarker.

For every ROI r, biomarker b and stage s the subjects in stage s are
compared against the pooled subjects of the other five stages:

    test        Mann-Whitney U, two-sided  (n = 16-20 vs 96-100; normality
                cannot be assumed at these sizes)
    effect size Hedges' g, stage minus rest (bias-corrected)
    correction  Benjamini-Hochberg across 246 ROIs x 6 stages = 1476 tests,
                declared PER BIOMARKER

This is a genuinely per-stage design. It is NOT derived from the 15
pairwise comparisons, and the multivariate one-vs-rest q-values are never
reused as if they were ROI-level p-values -- they appear only as a labelled
annotation strip above each heatmap.

FC representation
-----------------
FC is a 246x246 matrix, not a regional scalar. The ROI-level summary used
here is the UNSIGNED mean connectivity strength of each ROI:

    FCstr_i = mean_{j != i} |z_ij|

The obvious alternative, the signed row mean, equals DC/245 and correlates
with DC_z at r = +0.994 across this cohort -- it would duplicate DC_z and
is therefore excluded. The unsigned strength correlates with DC_z at
r = -0.090, so it carries different information: net signed connectivity
versus total connectivity magnitude.

Limitation
----------
One-vs-rest pools the other five stages and ignores the ordinal structure
CN -> AD. It answers "is this stage separable", not "do the stages
progress". Exploratory.
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "derivatives", "model_ready")
LUT = os.path.join(ROOT, "derivatives", "fsfast", "FINAL_6GROUP_AUDIT",
                   "dataset_brainnetome_summary.csv")

# PROJECT order, fixed at Review I and encoded in model_ready (MCI=3, LMCI=4)
STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
BIOMARKERS = ["mALFF", "mReHo", "DC_z", "FC"]
N_ROI = 246
ALPHA = 0.05
Q_FLOOR = 1e-4
UBLUE, DGRAY, RED = "#103E6E", "#374151", "#C0392B"

_log = []


def log(m=""):
    print(m, flush=True)
    _log.append(str(m))


def bh(p):
    p = np.asarray(p, float)
    flat = p.ravel()
    n = flat.size
    o = np.argsort(flat)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((flat[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(q, 0, 1).reshape(p.shape)


def hedges_g(a, b):
    na, nb = len(a), len(b)
    sp2 = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    if sp2 <= 0:
        return np.nan
    return ((a.mean() - b.mean()) / np.sqrt(sp2)) * (1 - 3 / (4 * (na + nb) - 9))


def roi_names():
    d = {}
    if os.path.isfile(LUT):
        with open(LUT, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                d[int(r["ROI_ID"])] = r["ROI_NAME"]
    return d


def load():
    reg = np.load(os.path.join(SRC, "X_regional.npy"))
    y = np.load(os.path.join(SRC, "y.npy"))
    subs = np.load(os.path.join(SRC, "subjects.npy"))
    X = {n: reg[:, :, i].astype(np.float64) for i, n in enumerate(BIOMARKERS[:3])}
    f = os.path.join(SRC, "X_fc_roi_unsigned.npy")
    if not os.path.isfile(f):
        raise SystemExit(f"missing {f} -- run the FC ROI-summary step first")
    X["FC"] = np.load(f).astype(np.float64)
    return X, y, subs


def stage_annotation(out_dir):
    """Multivariate one-vs-rest q per (biomarker, stage), if available.
    Returned only for the annotation strip -- never used as an ROI p-value."""
    f = os.path.join(out_dir, "one_vs_rest_statistics.csv")
    if not os.path.isfile(f):
        return None
    d = {}
    with open(f, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("q_value_BH"):
                d[(r["biomarker"], r["stage"])] = float(r["q_value_BH"])
    return d or None


def main(out_dir):
    fig_dir = os.path.join(out_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    X, y, subs = load()
    names = roi_names()
    n = len(y)
    ann = stage_annotation(out_dir)

    log("=" * 74)
    log("ROI x STAGE ONE-VS-REST  (246 ROIs x 6 stages, per biomarker)")
    log("=" * 74)
    log(f"cohort {n} acquisitions / {len(set(subs))} subjects")
    for i, s in enumerate(STAGES):
        log(f"   {s:5s} n = {int((y == i).sum()):3d}  vs rest {int((y != i).sum()):3d}")
    log(f"\nstage order (project): {' -> '.join(STAGES)}")
    log("test: Mann-Whitney U two-sided; effect size: Hedges' g (stage - rest)")
    log(f"FDR family: {N_ROI} ROIs x {len(STAGES)} stages = "
        f"{N_ROI*len(STAGES)} tests PER biomarker")
    log(f"FC represented as unsigned mean |Fisher-z| per ROI "
        f"(signed mean = DC/245, r=+0.994 with DC_z, excluded)")
    log(f"stage annotation strip: "
        f"{'multivariate one-vs-rest q' if ann else 'NOT AVAILABLE (pending)'}\n")

    results = {}
    for bm in BIOMARKERS:
        A = X[bm]
        g = np.full((N_ROI, len(STAGES)), np.nan)
        p = np.ones_like(g)
        for si in range(len(STAGES)):
            m = (y == si)
            for r in range(N_ROI):
                a, b = A[m, r], A[~m, r]
                g[r, si] = hedges_g(a, b)
                try:
                    p[r, si] = stats.mannwhitneyu(a, b, alternative="two-sided")[1]
                except ValueError:
                    p[r, si] = 1.0
        q = bh(p)
        sig = q < ALPHA
        results[bm] = dict(g=g, p=p, q=q, sig=sig)
        log(f"  {bm:6s} tests={g.size}  raw p<0.05={int((p<ALPHA).sum()):4d} "
            f"(chance {ALPHA*g.size:.0f})  q<0.05={int(sig.sum()):4d}  "
            f"min q={q.min():.4f}  max |g|={np.nanmax(np.abs(g)):.3f}")

    # ------------------------------------------------------------- figures
    for bm in BIOMARKERS:
        d = results[bm]
        S = -np.log10(np.maximum(d["q"], Q_FLOOR))
        fig = plt.figure(figsize=(8.8, 11.5), facecolor="white")
        gs = GridSpec(2, 2, height_ratios=[1, 26], width_ratios=[40, 1],
                      hspace=0.02, wspace=0.04)

        # top annotation strip: stage-level multivariate q
        axt = fig.add_subplot(gs[0, 0])
        if ann:
            vals = np.array([[ann.get((bm, s), np.nan) for s in STAGES]])
            strip = -np.log10(np.maximum(vals, Q_FLOOR))
            axt.imshow(strip, cmap="magma_r", aspect="auto", vmin=0,
                       vmax=max(1.5, np.nanmax(strip) * 1.1))
            for j, s in enumerate(STAGES):
                v = ann.get((bm, s))
                axt.text(j, 0, "n/a" if v is None else f"q={v:.3f}",
                         ha="center", va="center", fontsize=7.5,
                         color="black")
            axt.set_title(f"stage-level multivariate one-vs-rest "
                          f"(permutation, BH over 24) — NOT an ROI p-value",
                          fontsize=8.5, color=DGRAY, pad=4)
        else:
            axt.text(0.5, 0.5, "stage-level multivariate test pending",
                     ha="center", va="center", fontsize=9, color=RED,
                     transform=axt.transAxes)
        axt.set_xticks([]); axt.set_yticks([])

        ax = fig.add_subplot(gs[1, 0])
        im = ax.imshow(S, aspect="auto", cmap="magma_r", vmin=0,
                       vmax=max(-np.log10(ALPHA) * 1.2, S.max() * 1.05),
                       interpolation="nearest")
        ys, xs = np.where(d["sig"])
        if len(ys):
            ax.scatter(xs, ys, s=10, marker="*", c="cyan",
                       label=f"q < {ALPHA}")
            ax.legend(fontsize=8, loc="upper right")
        ax.set_xticks(range(len(STAGES)))
        ax.set_xticklabels(STAGES, fontsize=10)
        ax.set_yticks(np.arange(0, N_ROI, 10))
        ax.set_yticklabels(np.arange(1, N_ROI + 1, 10), fontsize=6)
        ax.set_ylabel("Brainnetome ROI (1–246)", fontsize=10)
        ax.set_xlabel("disease stage (project order)", fontsize=10)

        cax = fig.add_subplot(gs[1, 1])
        cb = fig.colorbar(im, cax=cax)
        cb.set_label(f"-log10(BH q)   floor {Q_FLOOR:g}", fontsize=9)
        cb.ax.axhline(-np.log10(ALPHA), color="cyan", lw=1.5)

        nsig = int(d["sig"].sum())
        fig.suptitle(f"{bm}: ROI-level one-vs-rest significance\n"
                     f"each stage vs the other five pooled",
                     fontsize=13.5, fontweight="bold", color=UBLUE, y=0.965)
        fig.text(0.5, 0.055,
                 f"Mann-Whitney U, two-sided; BH FDR over {N_ROI}x{len(STAGES)}"
                 f"={N_ROI*len(STAGES)} tests for this biomarker.\n"
                 f"Significant ROIs (q<{ALPHA}): {nsig}. "
                 f"{'None pass correction.' if nsig == 0 else ''} "
                 f"Colour is -log10(q); the top strip is a separate "
                 f"stage-level multivariate test, not an ROI p-value.\n"
                 f"Exploratory: pooling the other five stages ignores the "
                 f"ordinal structure CN→AD.",
                 ha="center", fontsize=8.5, color=DGRAY)
        for ext in ("png", "pdf"):
            fig.savefig(os.path.join(
                fig_dir, f"{bm}_roi_stage_significance_heatmap.{ext}"),
                dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        log(f"   saved {bm}_roi_stage_significance_heatmap.png/.pdf")

        for key, arr in [("qvalues", d["q"]), ("pvalues", d["p"]),
                         ("effectsizes", d["g"]),
                         ("significance", d["sig"].astype(int))]:
            with open(os.path.join(out_dir,
                      f"{bm}_roi_stage_{key}.csv"), "w", newline="",
                      encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow(["ROI_ID", "ROI_NAME"] + STAGES)
                for r in range(N_ROI):
                    w.writerow([r + 1, names.get(r + 1, "")] +
                               [f"{v:.6g}" for v in arr[r]])

    with open(os.path.join(out_dir, "roi_stage_method.json"), "w",
              encoding="utf-8") as fh:
        json.dump({"n_acquisitions": int(n), "n_subjects": int(len(set(subs))),
                   "stage_order": STAGES,
                   "per_stage": {s: int((y == i).sum()) for i, s in enumerate(STAGES)},
                   "test": "Mann-Whitney U two-sided, stage vs pooled rest",
                   "effect_size": "Hedges' g (stage - rest)",
                   "fdr_family": f"{N_ROI} ROIs x {len(STAGES)} stages per biomarker",
                   "alpha": ALPHA, "q_floor_display": Q_FLOOR,
                   "fc_representation": "unsigned mean |Fisher-z| per ROI",
                   "fc_excluded_alternative": "signed row mean (= DC/245), r=+0.994 with DC_z",
                   "n_significant": {b: int(results[b]["sig"].sum()) for b in BIOMARKERS},
                   "limitation": "one-vs-rest ignores ordinal structure CN->AD"},
                  fh, indent=2)
    with open(os.path.join(out_dir, "roi_stage_log.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(_log))
    log(f"\noutputs in {os.path.relpath(out_dir, ROOT)}")
    return results


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.join(ROOT, "derivatives", "INTRA_INTER_CLASS_HEATMAPS",
                      "v2_cohort116"))
