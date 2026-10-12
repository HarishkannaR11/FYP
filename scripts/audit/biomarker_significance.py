"""
Formal significance testing of the four biomarkers against disease stage.

Subject-level (127 subjects, not 165 acquisitions) to avoid pseudo-replication.
Benjamini-Hochberg FDR across all tests in each family. Read-only.
"""
import glob
import os

import numpy as np
import pandas as pd
from scipy import stats

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NRM = os.path.join(R, "derivatives", "biomarkers_normalized")
AUD = os.path.join(R, "derivatives", "fsfast", "FINAL_6GROUP_AUDIT")
ORDER = ["CN_Final", "SMC_Final", "EMCI", "MCI", "LMCI", "AD"]
SHORT = dict(zip(ORDER, ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]))
STAGES = list(SHORT.values())
RANK = {s: i for i, s in enumerate(STAGES)}


def bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    n = p.size
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def hdr(t):
    print("\n" + "=" * 76 + f"\n{t}\n" + "=" * 76)


def main():
    rows = []
    for g in ORDER:
        for d in sorted(glob.glob(os.path.join(NRM, g, "sub-*", "ses-*", "run-*"))):
            rows.append(dict(stage=SHORT[g], subject=d.split(os.sep)[-3], p=d))
    df = pd.DataFrame(rows)

    iu = np.triu_indices(246, 1)
    feats = {"mALFF": "malff.npy", "mReHo": "mreho.npy", "DC_z": "dc_z.npy"}
    subj, lab = [], []
    S = {k: [] for k in list(feats) + ["FC"]}
    for sub, g in df.groupby("subject"):
        subj.append(sub)
        lab.append(g.stage.iloc[0])
        for n, f in feats.items():
            S[n].append(np.mean([np.load(os.path.join(p, f)) for p in g.p], 0))
        S["FC"].append(np.mean([np.load(os.path.join(p, "fc_fisher_z.npy"))[iu]
                                for p in g.p], 0))
    S = {k: np.vstack(v) for k, v in S.items()}
    lab = np.array(lab)
    print(f"subject-level matrices: " +
          ", ".join(f"{k} {v.shape}" for k, v in S.items()))
    print("per stage: " + "  ".join(f"{s}={int((lab==s).sum())}" for s in STAGES))

    # motion covariate, subject level
    q = pd.read_csv(os.path.join(AUD, "dataset_qc_summary.csv"))
    fdm = q.groupby("subject").mean_FD_mm.mean()
    fd = np.array([fdm.get(s, np.nan) for s in subj])

    hdr("1.  OMNIBUS: does ANY region differ across the six stages?")
    print(f"{'biomarker':10s} {'tests':>7s} {'p<0.05':>8s} {'expected':>9s} "
          f"{'FDR q<0.05':>11s} {'min p':>10s} {'min q':>8s}")
    keep = {}
    for n in ["mALFF", "mReHo", "DC_z", "FC"]:
        X = S[n]
        p = np.array([stats.kruskal(*[X[lab == s, j] for s in STAGES])[1]
                      for j in range(X.shape[1])])
        q_ = bh(p)
        keep[n] = (p, q_)
        print(f"{n:10s} {X.shape[1]:7d} {int((p<.05).sum()):8d} "
              f"{0.05*X.shape[1]:9.0f} {int((q_<.05).sum()):11d} "
              f"{p.min():10.2e} {q_.min():8.3f}")

    hdr("2.  CN vs AD, the most-separated pair")
    print(f"{'biomarker':10s} {'p<0.05':>8s} {'expected':>9s} "
          f"{'FDR q<0.05':>11s} {'min p':>10s} {'min q':>8s}")
    cn, ad = lab == "CN", lab == "AD"
    print(f"(n = {cn.sum()} CN vs {ad.sum()} AD subjects)")
    cnad = {}
    for n in ["mALFF", "mReHo", "DC_z", "FC"]:
        X = S[n]
        p = np.array([stats.mannwhitneyu(X[cn, j], X[ad, j])[1]
                      for j in range(X.shape[1])])
        q_ = bh(p)
        cnad[n] = (p, q_)
        print(f"{n:10s} {int((p<.05).sum()):8d} {0.05*X.shape[1]:9.0f} "
              f"{int((q_<.05).sum()):11d} {p.min():10.2e} {q_.min():8.3f}")

    hdr("3.  MONOTONIC TREND across the ordered stages (Spearman)")
    print("tests whether a region changes steadily CN -> AD")
    y = np.array([RANK[s] for s in lab], float)
    for n in ["mALFF", "mReHo", "DC_z", "FC"]:
        X = S[n]
        r = np.array([stats.spearmanr(y, X[:, j])[0] for j in range(X.shape[1])])
        p = np.array([stats.spearmanr(y, X[:, j])[1] for j in range(X.shape[1])])
        q_ = bh(p)
        print(f"{n:10s} p<0.05: {int((p<.05).sum()):5d} / {X.shape[1]:5d} "
              f"(expected {0.05*X.shape[1]:.0f})   FDR q<0.05: "
              f"{int((q_<.05).sum()):4d}   max |rho| = {np.abs(r).max():.3f}")

    hdr("4.  MULTIVARIATE OMNIBUS (PERMANOVA-style permutation test)")
    print("is there ANY stage structure in the whole 246-dim space?")
    rng = np.random.default_rng(0)
    for n in ["mALFF", "mReHo", "DC_z", "FC"]:
        X = stats.zscore(S[n], axis=0, ddof=0)
        X = np.nan_to_num(X)
        gt = X.mean(0)

        def pseudo_f(l):
            ssb = sum((l == s).sum() * ((X[l == s].mean(0) - gt) ** 2).sum()
                      for s in STAGES)
            ssw = sum(((X[l == s] - X[l == s].mean(0)) ** 2).sum()
                      for s in STAGES)
            return (ssb / 5) / (ssw / (len(l) - 6))

        obs = pseudo_f(lab)
        null = np.array([pseudo_f(rng.permutation(lab)) for _ in range(2000)])
        pv = (np.sum(null >= obs) + 1) / (len(null) + 1)
        print(f"{n:10s} pseudo-F = {obs:.3f}   permutation p = {pv:.4f}   "
              f"{'SIGNIFICANT' if pv < .05 else 'not significant'}")

    hdr("5.  IF ANYTHING SURVIVED, IS IT MOTION?")
    any_hit = False
    for n in ["mALFF", "mReHo", "DC_z", "FC"]:
        p, q_ = keep[n]
        hits = np.where(q_ < 0.05)[0]
        if len(hits) == 0:
            continue
        any_hit = True
        ok = np.isfinite(fd)
        print(f"\n{n}: {len(hits)} regions/edges survive FDR. "
              f"Correlation with mean FD:")
        for j in hits[:10]:
            r, pm = stats.spearmanr(fd[ok], S[n][ok, j])
            print(f"   index {j:6d}   rho(FD) = {r:+.3f}  p = {pm:.4f}"
                  f"{'   <-- motion-linked' if pm < .05 else ''}")
    if not any_hit:
        print("Nothing survived FDR in any biomarker, so there is nothing to")
        print("attribute to motion. The negative result stands on its own.")


if __name__ == "__main__":
    main()
