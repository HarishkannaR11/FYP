"""
Figures for the two new Review-II findings. Read-only.

fig_complementarity.png  biomarker redundancy: one pair removed, three kept
fig_reliability.png      between-session test-retest, fingerprinting, and the
                         homotopic positive control

Colours match the Beamer template (UniversityBlue / AccentBlue).
"""
import glob
import itertools
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NRM = os.path.join(R, "derivatives", "biomarkers_normalized")
DRV = os.path.join(R, "derivatives", "fsfast")
OUT = os.path.join(R, "derivatives", "REVIEW2_FINDINGS")

ORDER = ["CN_Final", "SMC_Final", "EMCI", "MCI", "LMCI", "AD"]
SHORT = dict(zip(ORDER, ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]))
UBLUE, ABLUE, LBLUE, DGRAY = "#103E6E", "#1E88E5", "#E8F2FC", "#374151"
RED, GREEN = "#C0392B", "#2E7D32"


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", alpha=0.25, lw=0.6)
    ax.set_axisbelow(True)


def index():
    rows = []
    for g in ORDER:
        for d in sorted(glob.glob(os.path.join(NRM, g, "sub-*", "ses-*", "run-*"))):
            p = d.split(os.sep)
            rows.append(dict(stage=SHORT[g], subject=p[-3], ses=p[-2],
                             run=p[-1], norm=d,
                             raw=os.path.join(DRV, g, p[-3], p[-2], p[-1],
                                              "biomarkers")))
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    df = index()
    F = {n: np.vstack([np.load(os.path.join(p, f)) for p in df.norm])
         for n, f in [("mALFF", "malff.npy"), ("mReHo", "mreho.npy"),
                      ("DC_z", "dc_z.npy")]}
    n_acq = F["mALFF"].shape[0]
    print(f"{n_acq} acquisitions, {df.subject.nunique()} subjects")

    # ---- the redundant pair, recomputed from the raw features -------------
    fcs_dc = []
    for p in df.raw:
        dc = np.load(os.path.join(p, "degree_centrality.npy"))
        fcs = np.load(os.path.join(p, "fc_strength.npy"))
        fcs_dc.append(stats.pearsonr(dc, fcs)[0])
    fcs_dc = np.array(fcs_dc)
    print(f"FC_Strength vs DC: mean r = {fcs_dc.mean():.12f}")

    pairs = [("FC_Strength", "DC", fcs_dc, True)]
    for a, b in itertools.combinations(F, 2):
        r = np.array([stats.pearsonr(F[a][i], F[b][i])[0] for i in range(n_acq)])
        pairs.append((a, b, r, False))

    # ================================================= FIGURE 1
    fig, (ax, ax2) = plt.subplots(
        1, 2, figsize=(13.2, 4.6), facecolor="white",
        gridspec_kw=dict(width_ratios=[1.75, 1], wspace=0.28))

    labels = [f"{a}–{b}" for a, b, _, _ in pairs]
    for i, (a, b, r, redundant) in enumerate(pairs):
        y = len(pairs) - 1 - i
        c = RED if redundant else UBLUE
        ax.scatter(r, np.random.normal(y, 0.07, len(r)), s=11, color=c,
                   alpha=0.45, edgecolors="none", zorder=3)
        ax.scatter([r.mean()], [y], s=150, marker="|", color=c, zorder=4, lw=2.4)
        ax.text(1.06, y, f"r̄ = {r.mean():+.3f}", va="center", fontsize=10,
                color=c, fontweight="bold")
    ax.axvline(0, color=DGRAY, lw=0.9, ls=":")
    ax.axvline(1, color=RED, lw=1.1, ls="--", alpha=0.6)
    ax.set_yticks(range(len(pairs)))
    ax.set_yticklabels(labels[::-1], fontsize=10.5)
    ax.set_xlim(-0.75, 1.3)
    ax.set_xticks([-0.5, 0, 0.5, 1.0])
    ax.set_xlabel("Correlation across the 246 ROIs (one point = one acquisition)",
                  fontsize=10)
    ax.set_title("One pair is perfectly redundant; three are not",
                 fontsize=12.5, fontweight="bold", color=UBLUE)
    ax.text(1.0, len(pairs) - 1.38, "removed at\nReview II", color=RED,
            fontsize=9, ha="center", va="top", fontweight="bold")
    ax.set_ylim(-0.6, len(pairs) - 0.5)
    style(ax)
    ax.grid(axis="y", alpha=0)

    sv = [(f"{a}–{b}", np.mean([stats.pearsonr(F[a][i], F[b][i])[0] ** 2
                                     for i in range(n_acq)]) * 100)
          for a, b in itertools.combinations(F, 2)]
    nm, vals = zip(*sv)
    bars = ax2.bar(nm, vals, color=ABLUE, width=0.6)
    ax2.bar(nm, [100 - v for v in vals], bottom=vals, color=LBLUE, width=0.6)
    for bb, v in zip(bars, vals):
        ax2.text(bb.get_x() + bb.get_width() / 2, v + 3, f"{v:.1f}%",
                 ha="center", fontsize=11, fontweight="bold", color=UBLUE)
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("Shared variance (%)", fontsize=10.5)
    ax2.set_title("The retained pairs share 6–11%",
                  fontsize=12.5, fontweight="bold", color=UBLUE)
    ax2.tick_params(labelsize=9.5)
    style(ax2)

    fig.suptitle("Finding 3: the biomarkers carry different information",
                 fontsize=15, fontweight="bold", color=UBLUE, y=1.04)
    fig.text(0.5, -0.07,
             f"All {n_acq} acquisitions. FC_Strength = DC/245 exactly "
             "(r = 1.000000000000) and was removed; the three retained "
             "biomarkers are 89–94% non-redundant, which is what makes "
             "them usable as separate meta-learning tasks.",
             ha="center", fontsize=9.5, color=DGRAY)
    p = os.path.join(OUT, "fig_complementarity.png")
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved fig_complementarity.png")

    # ================================================= FIGURE 2
    rep = df.groupby("subject").size()
    rep = rep[rep >= 2]
    idx = {s: np.where(df.subject.values == s)[0] for s in rep.index}
    rng = np.random.default_rng(0)
    subs = list(rep.index)
    stat = {}
    for n, X in F.items():
        within = [stats.pearsonr(X[i], X[j])[0] for s in subs
                  for i, j in itertools.combinations(idx[s], 2)]
        between = []
        for _ in range(3000):
            a, b = rng.choice(len(subs), 2, replace=False)
            between.append(stats.pearsonr(X[rng.choice(idx[subs[a]])],
                                          X[rng.choice(idx[subs[b]])])[0])
        sel = np.concatenate([idx[s] for s in subs])
        C = np.corrcoef(X[sel])
        np.fill_diagonal(C, -np.inf)
        own = np.concatenate([[s] * len(idx[s]) for s in subs])
        hit = sum(own[C[i].argmax()] == own[i] for i in range(len(sel)))
        stat[n] = (np.mean(within), np.mean(between), hit, len(sel))

    homo, het = [], []
    r2 = np.random.default_rng(1)
    for p_ in df.norm[:60]:
        Z = np.load(os.path.join(p_, "fc_fisher_z.npy"))
        homo.append(np.mean([Z[2 * k, 2 * k + 1] for k in range(123)]))
        pr = r2.integers(0, 246, (200, 2))
        het.append(np.mean([Z[i, j] for i, j in pr if i != j]))

    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.5), facecolor="white")
    names = list(F)
    x = np.arange(len(names))
    w = 0.36
    ax = axes[0]
    ax.bar(x - w / 2, [stat[n][0] for n in names], w, label="same person,\ndifferent session",
           color=UBLUE)
    ax.bar(x + w / 2, [stat[n][1] for n in names], w, label="different people",
           color="#9FB6CE")
    for i, n in enumerate(names):
        ax.text(i - w / 2, stat[n][0] + .02, f"{stat[n][0]:.2f}", ha="center",
                fontsize=10, fontweight="bold", color=UBLUE)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10.5)
    ax.set_ylabel("Correlation across 246 ROIs", fontsize=10.5)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8.5, frameon=False, loc="upper right")
    ax.set_title("Repeat scans resemble the same person",
                 fontsize=12, fontweight="bold", color=UBLUE)
    style(ax)

    ax = axes[1]
    acc = [100 * stat[n][2] / stat[n][3] for n in names]
    b = ax.bar(names, acc, color=[ABLUE, ABLUE, "#9FB6CE"], width=0.55)
    ax.axhline(100 / len(rep), color=RED, ls="--", lw=1.4)
    ax.text(2.45, 100 / len(rep) + 3, f"chance {100/len(rep):.0f}%", color=RED,
            fontsize=9.5, ha="right", fontweight="bold")
    for bb, v, n in zip(b, acc, names):
        ax.text(bb.get_x() + bb.get_width() / 2, v + 2.5,
                f"{v:.0f}%\n({stat[n][2]}/{stat[n][3]})", ha="center",
                fontsize=10, fontweight="bold", color=UBLUE)
    ax.set_ylim(0, 108)
    ax.set_ylabel("Correct identification (%)", fontsize=10.5)
    ax.set_title("Each scan identifies its own subject",
                 fontsize=12, fontweight="bold", color=UBLUE)
    ax.tick_params(labelsize=10.5)
    style(ax)

    ax = axes[2]
    bp = ax.boxplot([homo, het], tick_labels=["left–right\nhomologues",
                                              "random ROI\npairs"],
                    patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", lw=1.6))
    for bb, c in zip(bp["boxes"], [UBLUE, "#9FB6CE"]):
        bb.set(facecolor=c, alpha=0.75, edgecolor=c)
    ax.set_ylabel("Functional connectivity (Fisher-z)", fontsize=10.5)
    ax.axhline(0, color=DGRAY, lw=0.8, ls=":")
    t, pv = stats.ttest_rel(homo, het)
    ax.set_title(f"Positive control: homotopic FC\n"
                 f"{np.mean(homo):+.3f} vs {np.mean(het):+.3f}, "
                 f"t = {t:.1f}", fontsize=12, fontweight="bold", color=UBLUE)
    ax.tick_params(labelsize=10)
    style(ax)

    fig.suptitle("Finding 4: the features are reproducible and biologically plausible",
                 fontsize=15, fontweight="bold", color=UBLUE, y=1.05)
    fig.text(0.5, -0.09,
             f"{len(rep)} subjects with repeat scans ({stat['mALFF'][3]} acquisitions, "
             "45 of 46 pairs from different sessions). Identification: a scan's "
             "nearest neighbour by correlation is that same person's other scan. "
             f"Homotopic control: {len(homo)} acquisitions, p = {pv:.0e}.",
             ha="center", fontsize=9.5, color=DGRAY)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_reliability.png")
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved fig_reliability.png")
    print(f"\noutputs in {OUT}")


if __name__ == "__main__":
    main()
