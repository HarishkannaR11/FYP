"""
Intra-class and inter-class analysis across the six progression stages.

Descriptive heatmaps and inferential statistics are kept strictly separate:
sections 1-3 describe, section 4 tests. Nothing here is a model and nothing
is trained.

Input : derivatives/model_ready/ (114 acquisitions, one per subject)
Output: derivatives/INTRA_INTER_CLASS_HEATMAPS/

Metrics, stated once and used consistently:
  regional biomarkers  Pearson r across the 246 ROI values
  FC                   Pearson r across the 30,135 upper-triangle edges,
                       diagonal excluded
  inter-class          Pearson r between stage prototypes, plus Euclidean
                       distance as a magnitude-sensitive complement
"""
import csv
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "derivatives", "model_ready")
OUT = os.path.join(ROOT, "derivatives", "INTRA_INTER_CLASS_HEATMAPS")
FIG = os.path.join(OUT, "figures")
MAT = os.path.join(OUT, "matrices")

STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]   # FIXED order
REGIONAL = ["mALFF", "mReHo", "DC_z"]
ALL_FEATS = REGIONAL + ["FC"]
UBLUE, ABLUE, DGRAY, RED = "#103E6E", "#1E88E5", "#374151", "#C0392B"
N_PERM = 5000
RNG = np.random.default_rng(0)

_log = []


def log(m=""):
    print(m, flush=True)
    _log.append(str(m))


def bh(p):
    p = np.asarray(p, float)
    n = p.size
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ---------------------------------------------------------------- loading
def load():
    reg = np.load(os.path.join(SRC, "X_regional.npy"))
    fc = np.load(os.path.join(SRC, "X_fc.npy"))
    y = np.load(os.path.join(SRC, "y.npy"))
    subs = np.load(os.path.join(SRC, "subjects.npy"))
    iu = np.triu_indices(246, 1)            # diagonal excluded, by design
    feats = {n: reg[:, :, i].astype(np.float64)
             for i, n in enumerate(REGIONAL)}
    feats["FC"] = fc[:, iu[0], iu[1]].astype(np.float64)
    del fc, reg
    return feats, y, subs


def corr_matrix(X):
    """Subject x subject Pearson correlation across features."""
    Z = X - X.mean(axis=1, keepdims=True)
    Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    return np.clip(Z @ Z.T, -1.0, 1.0)


# ------------------------------------------------------- 1. intra-class
def intra_class(feats, y, subs):
    log("\n" + "=" * 74)
    log("1. INTRA-CLASS  --  subject-to-subject similarity within each stage")
    log("=" * 74)
    log("Pearson r between subjects. Diagonal (self-similarity = 1) is")
    log("excluded from every summary statistic.\n")
    log(f"{'biomarker':10s} {'stage':6s} {'n':>4s} {'pairs':>6s} "
        f"{'mean r':>8s} {'median':>8s} {'sd':>7s} {'min':>7s} {'max':>7s}")
    rows, mats = [], {}
    for f in ALL_FEATS:
        for si, s in enumerate(STAGES):
            idx = np.where(y == si)[0]
            C = corr_matrix(feats[f][idx])
            mats[(f, s)] = (C, subs[idx])
            off = C[np.triu_indices(len(idx), 1)]
            rows.append(dict(biomarker=f, stage=s, n_subjects=len(idx),
                             n_pairs=len(off), mean=off.mean(),
                             median=np.median(off), sd=off.std(ddof=1),
                             min=off.min(), max=off.max()))
            log(f"{f:10s} {s:6s} {len(idx):4d} {len(off):6d} "
                f"{off.mean():8.4f} {np.median(off):8.4f} "
                f"{off.std(ddof=1):7.4f} {off.min():7.4f} {off.max():7.4f}")
        log("")
    return rows, mats


def plot_intra(mats):
    for f in ALL_FEATS:
        fig, axes = plt.subplots(2, 3, figsize=(15, 9.6), facecolor="white")
        vals = np.concatenate([mats[(f, s)][0][np.triu_indices(
            len(mats[(f, s)][1]), 1)] for s in STAGES])
        lo, hi = np.percentile(vals, [2, 98])
        lim = max(abs(lo), abs(hi))
        for ax, s in zip(axes.ravel(), STAGES):
            C, sid = mats[(f, s)]
            im = ax.imshow(C, cmap="RdBu_r", vmin=-lim, vmax=lim,
                           interpolation="nearest")
            ax.set_title(f"{s}  (n = {len(sid)})", fontsize=11.5,
                         fontweight="bold", color=UBLUE)
            ax.set_xticks(range(len(sid)))
            ax.set_yticks(range(len(sid)))
            ax.set_xticklabels([x.replace("sub-", "") for x in sid],
                               rotation=90, fontsize=5.5)
            ax.set_yticklabels([x.replace("sub-", "") for x in sid],
                               fontsize=5.5)
        fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02,
                     label="Pearson r between subjects")
        fig.suptitle(f"Intra-class subject similarity: {f}",
                     fontsize=16, fontweight="bold", color=UBLUE, y=0.97)
        fig.text(0.5, 0.025,
                 "One subject per row/column; the unit-diagonal is excluded "
                 "from all reported statistics. High similarity here does not "
                 "imply shared disease biology.",
                 ha="center", fontsize=9.5, color=DGRAY)
        p = os.path.join(FIG, f"intra_class_{f}.png")
        fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        log(f"   saved {os.path.basename(p)}")


# ---------------------------------------------------- 2. stage prototypes
def prototypes(feats, y):
    log("\n" + "=" * 74)
    log("2. STAGE PROTOTYPES")
    log("=" * 74)
    log("Estimator: arithmetic mean over subjects within each stage.")
    log("Chosen because every subject contributes exactly one acquisition,")
    log("so no subject can be over-weighted and no trimming is warranted.\n")
    protos = {}
    for f in ALL_FEATS:
        P = np.vstack([feats[f][y == si].mean(0) for si in range(6)])
        protos[f] = P
        log(f"   {f:8s} prototype matrix {P.shape}")
    return protos


def plot_prototypes(protos):
    fig, axes = plt.subplots(3, 1, figsize=(15, 9), facecolor="white")
    for ax, f in zip(axes, REGIONAL):
        P = protos[f]
        im = ax.imshow(P, aspect="auto", cmap="viridis",
                       interpolation="nearest")
        ax.set_yticks(range(6)); ax.set_yticklabels(STAGES, fontsize=10)
        ax.set_ylabel("stage", fontsize=10)
        ax.set_title(f"{f} stage prototypes (absolute values)",
                     fontsize=12, fontweight="bold", color=UBLUE)
        fig.colorbar(im, ax=ax, fraction=0.015, pad=0.01, label=f)
    axes[-1].set_xlabel("Brainnetome ROI index (1-246)", fontsize=10.5)
    fig.suptitle("Stage-wise class prototypes across 246 ROIs",
                 fontsize=15.5, fontweight="bold", color=UBLUE, y=0.98)
    fig.text(0.5, 0.015,
             "Mean over subjects within each stage; stage order fixed "
             "CN->AD. Absolute scale, not row-normalised.",
             ha="center", fontsize=9.5, color=DGRAY)
    fig.tight_layout(rect=[0, 0.03, 1, 0.955])
    p = os.path.join(FIG, "stage_prototypes_regional.png")
    fig.savefig(p, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"   saved {os.path.basename(p)}")

    # FC prototypes, common colour scale across stages
    P = protos["FC"]
    full = np.zeros((6, 246, 246))
    iu = np.triu_indices(246, 1)
    for i in range(6):
        M = np.zeros((246, 246))
        M[iu] = P[i]
        full[i] = M + M.T
    lim = np.percentile(np.abs(full), 99)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10), facecolor="white")
    for ax, s, M in zip(axes.ravel(), STAGES, full):
        im = ax.imshow(M, cmap="RdBu_r", vmin=-lim, vmax=lim,
                       interpolation="nearest")
        ax.set_title(s, fontsize=12.5, fontweight="bold", color=UBLUE)
        ax.set_xticks([0, 122, 245]); ax.set_yticks([0, 122, 245])
        ax.tick_params(labelsize=8)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02,
                 label="mean Fisher-z connectivity (common scale)")
    fig.suptitle("Stage-level functional connectivity prototypes",
                 fontsize=15.5, fontweight="bold", color=UBLUE, y=0.96)
    fig.text(0.5, 0.03,
             f"Mean Fisher-z FC per stage, identical colour scale "
             f"(+/-{lim:.3f}). Diagonal is zero by the project's convention.",
             ha="center", fontsize=9.5, color=DGRAY)
    p = os.path.join(FIG, "stage_prototypes_FC.png")
    fig.savefig(p, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"   saved {os.path.basename(p)}")
    return full


# ------------------------------------------------------ 3. inter-class
def inter_class(protos):
    log("\n" + "=" * 74)
    log("3. INTER-CLASS  --  prototype-to-prototype comparison")
    log("=" * 74)
    log("Correlation measures pattern similarity and is blind to overall")
    log("magnitude, so Euclidean distance is reported alongside it.\n")
    sim, dist = {}, {}
    for f in ALL_FEATS:
        P = protos[f]
        sim[f] = corr_matrix(P)
        D = np.sqrt(((P[:, None, :] - P[None, :, :]) ** 2).sum(-1))
        dist[f] = D
        off = sim[f][np.triu_indices(6, 1)]
        log(f"{f:8s} off-diagonal r: min {off.min():.4f}  "
            f"max {off.max():.4f}  mean {off.mean():.4f}")
    return sim, dist


def plot_inter(sim, dist):
    for name, mats, cmap, lab in [
            ("similarity", sim, "RdBu_r", "Pearson r between prototypes"),
            ("distance", dist, "magma_r", "Euclidean distance")]:
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.6), facecolor="white")
        for ax, f in zip(axes, ALL_FEATS):
            M = mats[f]
            if name == "similarity":
                kw = dict(vmin=-1, vmax=1)
            else:
                kw = dict(vmin=0, vmax=M.max())
            im = ax.imshow(M, cmap=cmap, interpolation="nearest", **kw)
            ax.set_xticks(range(6)); ax.set_yticks(range(6))
            ax.set_xticklabels(STAGES, rotation=45, fontsize=9)
            ax.set_yticklabels(STAGES, fontsize=9)
            ax.set_title(f, fontsize=12, fontweight="bold", color=UBLUE)
            for i in range(6):
                for j in range(6):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if abs(M[i, j]) > 0.6 * np.abs(M).max()
                            else "black")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=lab)
        fig.suptitle(f"Inter-class {name} between stage prototypes",
                     fontsize=15.5, fontweight="bold", color=UBLUE, y=1.05)
        fig.text(0.5, -0.08,
                 "Stage order fixed CN->AD. Colour alone is not evidence of "
                 "a statistically significant difference; see the permutation "
                 "tests in the report.",
                 ha="center", fontsize=9.5, color=DGRAY)
        fig.tight_layout()
        p = os.path.join(FIG, f"inter_class_{name}.png")
        fig.savefig(p, dpi=250, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        log(f"   saved {os.path.basename(p)}")


# ------------------------------------------------------- 4. statistics
def permutation_test(feats, y):
    log("\n" + "=" * 74)
    log("4. STATISTICAL VALIDATION")
    log("=" * 74)
    log("H0: stage labels carry no information about feature similarity.")
    log("Statistic: mean within-stage r minus mean between-stage r, over")
    log("subject pairs. Labels are permuted across SUBJECTS, which preserves")
    log("the dependency between pairs sharing a subject -- individual pairs")
    log("are never treated as independent observations.")
    log(f"{N_PERM} permutations, Benjamini-Hochberg FDR over the "
        f"{len(ALL_FEATS)} biomarkers.\n")
    res, pvals = [], []
    for f in ALL_FEATS:
        C = corr_matrix(feats[f])
        iu = np.triu_indices(C.shape[0], 1)
        off = C[iu]

        def gap(lab):
            same = lab[iu[0]] == lab[iu[1]]
            return off[same].mean() - off[~same].mean()

        obs = gap(y)
        null = np.array([gap(RNG.permutation(y)) for _ in range(N_PERM)])
        p = (np.sum(null >= obs) + 1) / (N_PERM + 1)
        same = y[iu[0]] == y[iu[1]]
        w, b = off[same], off[~same]
        pooled = np.sqrt(((len(w) - 1) * w.var(ddof=1)
                          + (len(b) - 1) * b.var(ddof=1))
                         / (len(w) + len(b) - 2))
        d = (w.mean() - b.mean()) / pooled if pooled > 0 else np.nan
        ci = np.percentile(null, [2.5, 97.5])
        res.append(dict(biomarker=f, within_mean=w.mean(), between_mean=b.mean(),
                        gap=obs, cohens_d=d, p_value=p,
                        null_mean=null.mean(), null_ci_lo=ci[0], null_ci_hi=ci[1],
                        n_within_pairs=len(w), n_between_pairs=len(b)))
        pvals.append(p)
    q = bh(pvals)
    for r, qq in zip(res, q):
        r["q_value_BH"] = qq
    log(f"{'biomarker':10s} {'within':>8s} {'between':>8s} {'gap':>8s} "
        f"{'d':>7s} {'p':>8s} {'q(BH)':>8s}  verdict")
    for r in res:
        v = "significant" if r["q_value_BH"] < 0.05 else "not significant"
        log(f"{r['biomarker']:10s} {r['within_mean']:8.4f} "
            f"{r['between_mean']:8.4f} {r['gap']:8.4f} {r['cohens_d']:7.3f} "
            f"{r['p_value']:8.4f} {r['q_value_BH']:8.4f}  {v}")
    return res


def plot_summary(res, intra_rows):
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 4.8), facecolor="white")
    x = np.arange(len(ALL_FEATS))
    w = [r["within_mean"] for r in res]
    b = [r["between_mean"] for r in res]
    ax.bar(x - 0.2, w, 0.4, label="within stage", color=UBLUE)
    ax.bar(x + 0.2, b, 0.4, label="between stages", color="#9FB6CE")
    for i, r in enumerate(res):
        ax.text(i, max(w[i], b[i]) + 0.012,
                f"q={r['q_value_BH']:.3f}", ha="center", fontsize=8.5,
                color=RED if r["q_value_BH"] < .05 else DGRAY)
    ax.set_xticks(x); ax.set_xticklabels(ALL_FEATS)
    ax.set_ylabel("mean Pearson r between subjects", fontsize=10.5)
    ax.set_title("Within- vs between-stage similarity",
                 fontsize=12.5, fontweight="bold", color=UBLUE)
    ax.legend(fontsize=9, frameon=False)
    style(ax); ax.grid(axis="y", alpha=.25, lw=.6); ax.set_axisbelow(True)

    for f, c in zip(ALL_FEATS, [UBLUE, ABLUE, "#6FA8DC", "#9FB6CE"]):
        vals = [r["mean"] for r in intra_rows if r["biomarker"] == f]
        ax2.plot(STAGES, vals, marker="o", lw=1.8, label=f, color=c)
    ax2.set_ylabel("mean within-stage r", fontsize=10.5)
    ax2.set_title("Intra-class similarity by stage",
                  fontsize=12.5, fontweight="bold", color=UBLUE)
    ax2.legend(fontsize=9, frameon=False)
    style(ax2); ax2.grid(alpha=.25, lw=.6); ax2.set_axisbelow(True)

    fig.suptitle("Intra-class versus inter-class summary",
                 fontsize=15, fontweight="bold", color=UBLUE, y=1.04)
    fig.text(0.5, -0.08,
             "q = Benjamini-Hochberg corrected permutation p-value "
             f"({N_PERM} subject-label permutations).",
             ha="center", fontsize=9.5, color=DGRAY)
    fig.tight_layout()
    p = os.path.join(FIG, "summary_intra_vs_inter.png")
    fig.savefig(p, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    log(f"   saved {os.path.basename(p)}")


# ------------------------------------------------------------------ main
def main():
    for d in (OUT, FIG, MAT):
        os.makedirs(d, exist_ok=True)
    feats, y, subs = load()
    n = len(y)
    log("=" * 74)
    log("INTRA-CLASS / INTER-CLASS ANALYSIS")
    log("=" * 74)
    log(f"acquisitions {n}   distinct subjects {len(set(subs))}   "
        f"({'one per subject' if len(set(subs)) == n else 'REPEATS'})")
    for si, s in enumerate(STAGES):
        log(f"   {s:5s} n = {int((y == si).sum()):3d}")
    log(f"\nfeature dimensions: " +
        ", ".join(f"{k} {v.shape[1]}" for k, v in feats.items()))

    intra_rows, mats = intra_class(feats, y, subs)
    plot_intra(mats)
    protos = prototypes(feats, y)
    fc_full = plot_prototypes(protos)
    sim, dist = inter_class(protos)
    plot_inter(sim, dist)
    res = permutation_test(feats, y)
    plot_summary(res, intra_rows)

    # ------------------------------------------------------------ saves
    for f in REGIONAL:
        np.save(os.path.join(MAT, f"prototype_{f}.npy"), protos[f])
    np.save(os.path.join(MAT, "prototype_FC_full.npy"), fc_full)
    for f in ALL_FEATS:
        np.save(os.path.join(MAT, f"interclass_similarity_{f}.npy"), sim[f])
        np.save(os.path.join(MAT, f"interclass_distance_{f}.npy"), dist[f])
    for (f, s), (C, sid) in mats.items():
        np.save(os.path.join(MAT, f"intraclass_{f}_{s}.npy"), C)

    with open(os.path.join(OUT, "intra_class_summary.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(intra_rows[0].keys()))
        w.writeheader(); w.writerows(intra_rows)
    with open(os.path.join(OUT, "statistical_tests.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0].keys()))
        w.writeheader(); w.writerows(res)
    with open(os.path.join(OUT, "subject_stage_mapping.csv"), "w", newline="",
              encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(["row", "subject", "stage", "y"])
        for i, (sb, yy) in enumerate(zip(subs, y)):
            w.writerow([i, sb, STAGES[yy], yy])
    with open(os.path.join(OUT, "analysis_log.txt"), "w",
              encoding="utf-8") as fh:
        fh.write("\n".join(_log))
    log(f"\noutputs in {os.path.relpath(OUT, ROOT)}")


if __name__ == "__main__":
    main()
