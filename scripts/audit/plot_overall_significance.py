"""
Overall 4 biomarkers x 6 stages significance heatmap.

Reads the completed one-vs-rest statistics (multivariate permutation test,
BH-corrected across all 24 cells) and renders a single 4x6 figure. Refuses
to plot if any of the 24 tests is missing -- a missing test is never drawn
as zero or blank-but-plausible.

    python scripts/audit/plot_overall_significance.py <out_dir>
"""
import csv
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]   # project order
BIOMARKERS = ["mALFF", "mReHo", "DC_z", "FC"]
ALPHA = 0.05
UBLUE, DGRAY, RED = "#103E6E", "#374151", "#C0392B"


def main(out_dir):
    stats_csv = os.path.join(out_dir, "one_vs_rest_statistics.csv")
    if not os.path.isfile(stats_csv):
        raise SystemExit(f"missing {stats_csv} -- the 24 tests have not all "
                         f"completed; refusing to plot a partial figure")
    rows = list(csv.DictReader(open(stats_csv, newline="", encoding="utf-8")))
    by = {(r["biomarker"], r["stage"]): r for r in rows}

    missing = [(b, s) for b in BIOMARKERS for s in STAGES if (b, s) not in by]
    if missing:
        raise SystemExit(f"{len(missing)} tests missing: {missing} -- refusing "
                         f"to plot; a missing test is not a zero")
    bad = [(b, s) for b in BIOMARKERS for s in STAGES
           if by[(b, s)].get("status") != "COMPLETED"]
    if bad:
        raise SystemExit(f"{len(bad)} tests not COMPLETED: {bad}")

    P = np.array([[float(by[(b, s)]["p_value"]) for s in STAGES]
                  for b in BIOMARKERS])
    Q = np.array([[float(by[(b, s)]["q_value_BH"]) for s in STAGES]
                  for b in BIOMARKERS])
    F = np.array([[float(by[(b, s)]["pseudo_F"]) for s in STAGES]
                  for b in BIOMARKERS])
    n_perm = int(rows[0]["n_permutations"])
    seed = int(rows[0]["seed"])
    fp = rows[0]["cohort_fingerprint"]
    floor = 1.0 / (n_perm + 1)

    assert P.shape == (4, 6) and Q.shape == (4, 6), "expected exactly 4x6"
    S = -np.log10(np.maximum(Q, floor))
    thr = -np.log10(ALPHA)
    n_sig = int((Q < ALPHA).sum())

    fig, ax = plt.subplots(figsize=(10.5, 5.4), facecolor="white")
    im = ax.imshow(S, cmap="magma_r", aspect="auto", vmin=0,
                   vmax=max(thr * 1.25, S.max() * 1.1),
                   interpolation="nearest")
    ax.set_xticks(range(len(STAGES)))
    ax.set_xticklabels(STAGES, fontsize=12)
    ax.set_yticks(range(len(BIOMARKERS)))
    ax.set_yticklabels(BIOMARKERS, fontsize=12)
    ax.set_xlabel("disease stage  (project order CN → AD)", fontsize=10.5)

    for i in range(len(BIOMARKERS)):
        for j in range(len(STAGES)):
            sig = Q[i, j] < ALPHA
            ax.text(j, i - 0.13, f"q={Q[i, j]:.3f}{'*' if sig else ''}",
                    ha="center", va="center", fontsize=9.5,
                    fontweight="bold" if sig else "normal",
                    color="white" if S[i, j] > 0.55 * S.max() else "black")
            ax.text(j, i + 0.17, f"p={P[i, j]:.4f}  F={F[i, j]:.2f}",
                    ha="center", va="center", fontsize=7.5,
                    color="white" if S[i, j] > 0.55 * S.max() else DGRAY)

    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label(f"−log10(BH-adjusted q)      floor {floor:.1e}",
                 fontsize=9.5)
    cb.ax.axhline(thr, color=RED, lw=1.8)
    cb.ax.text(1.7, thr, f" q={ALPHA}", color=RED, fontsize=8.5, va="center")

    ax.set_title("Overall Statistical Significance Across Four Biomarkers\n"
                 "and Six Alzheimer's Disease Stages",
                 fontsize=14, fontweight="bold", color=UBLUE, pad=14)

    verdict = (f"NO CELL REACHES q < {ALPHA}" if n_sig == 0
               else f"{n_sig}/24 cells reach q < {ALPHA} (marked *)")
    fig.text(0.5, -0.22,
             f"Each cell is one biomarker-stage one-vs-rest test: that stage's "
             f"subjects against the other five stages POOLED.\n"
             f"Test: distance-based pseudo-F (two-group PERMANOVA) on z-scored "
             f"features, {n_perm} subject-level label permutations, seed {seed}.\n"
             f"FC is represented by its {30135} unique upper-triangle Fisher-z "
             f"edges (diagonal excluded); regional biomarkers by 246 ROI values.\n"
             f"Benjamini-Hochberg FDR applied across all 24 tests. "
             f"Colour = −log10(q), floored at {floor:.1e} (the smallest "
             f"attainable p).\n"
             f"RESULT: {verdict}.   Cohort {fp}: 116 subjects "
             f"(CN 20, SMC 20, EMCI 20, MCI 16, LMCI 20, AD 20), one "
             f"acquisition each.\n"
             f"EXPLORATORY: pooling the other five stages ignores the ordinal "
             f"structure CN→AD. This is not a test of monotonic progression.",
             ha="center", fontsize=8.5, color=DGRAY)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        p = os.path.join(out_dir, "figures", f"overall_significance_heatmap.{ext}")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        fig.savefig(p, dpi=250, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    with open(os.path.join(out_dir, "overall_significance_heatmap.csv"), "w",
              newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["biomarker", "stage", "n_stage", "n_rest", "pseudo_F",
                    "p_value", "q_value_BH", "neg_log10_q", "significant_q05"])
        for i, b in enumerate(BIOMARKERS):
            for j, s in enumerate(STAGES):
                r = by[(b, s)]
                w.writerow([b, s, r["n_stage"], r["n_rest"],
                            f"{F[i, j]:.6g}", f"{P[i, j]:.6g}",
                            f"{Q[i, j]:.6g}", f"{S[i, j]:.6g}",
                            int(Q[i, j] < ALPHA)])

    print(f"4x6 verified: P{P.shape} Q{Q.shape}")
    print(f"significant cells: {n_sig}/24")
    print(f"min q = {Q.min():.4f}  (at "
          f"{BIOMARKERS[np.unravel_index(Q.argmin(), Q.shape)[0]]}/"
          f"{STAGES[np.unravel_index(Q.argmin(), Q.shape)[1]]})")
    print(f"saved overall_significance_heatmap.png/.pdf/.csv in "
          f"{os.path.relpath(out_dir, ROOT)}")
    return Q


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         os.path.join(ROOT, "derivatives", "INTRA_INTER_CLASS_HEATMAPS",
                      "v2_cohort116"))
