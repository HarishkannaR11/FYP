"""
Numbers behind Findings 3 and 4 and the three backup slides "Why 246 Regions?",
"Residual Motion and Site Effects" (docs/SecondReview/ppt/r2_presentation.tex).
Read-only; no stage label is used for Findings 3 and 4 or for the granularity
check, and nothing here says that any biomarker or region predicts disease.

    python3 scripts/audit/findings_checks.py

Prints, from the preprocessed derivatives:
  * Finding 3: per-scan correlation and shared variance between biomarker pairs
  * Finding 4: same-person vs different-people similarity, closest-scan
    identification, and the left-right homotopic control over all 165 scans
  * granularity: identification against random subsets and merged labels,
    pairwise overlap between regions, components for 80 % / 95 % of the variance
  * residual motion: regions whose value tracks mean FD (n = 164, artefact
    scan excluded, as in the original analysis, and n = 165)
  * site concentration per stage (site = prefix of the ADNI subject ID)
"""
import glob
import itertools
import os

import numpy as np
import pandas as pd
from scipy import stats

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NRM = os.path.join(R, "derivatives", "biomarkers_normalized")
DRV = os.path.join(R, "derivatives", "fsfast")
AUD = os.path.join(DRV, "FINAL_6GROUP_AUDIT")
ORDER = ["CN_Final", "SMC_Final", "EMCI", "MCI", "LMCI", "AD"]
SHORT = dict(zip(ORDER, ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]))
ARTEFACT = "sub-018S4313"          # CN scan flagged as an artefact in the QC audit


def main():
    rows = []
    for g in ORDER:
        for d in sorted(glob.glob(os.path.join(NRM, g, "sub-*", "ses-*", "run-*"))):
            p = d.split(os.sep)
            rows.append(dict(group=g, stage=SHORT[g], subject=p[-3], ses=p[-2], run=p[-1],
                             norm=d, raw=os.path.join(DRV, g, p[-3], p[-2], p[-1], "biomarkers")))
    df = pd.DataFrame(rows)
    n = len(df)
    F = {k: np.vstack([np.load(os.path.join(p, f)) for p in df.norm])
         for k, f in [("mALFF", "malff.npy"), ("mReHo", "mreho.npy"), ("DC_z", "dc_z.npy")]}
    print(f"{n} acquisitions, {df.subject.nunique()} subjects\n")

    print("== Finding 3: are the biomarkers redundant?")
    r = np.array([stats.pearsonr(np.load(os.path.join(p, "degree_centrality.npy")),
                                 np.load(os.path.join(p, "fc_strength.npy")))[0] for p in df.raw])
    print(f"FC_Strength vs DC: r = {r.min():.12f} (min) to {r.max():.12f} (max)")
    for a, b in itertools.combinations(F, 2):
        r = np.array([stats.pearsonr(F[a][i], F[b][i])[0] for i in range(n)])
        print(f"{a}-{b}: mean r {r.mean():+.3f}; shared variance (mean r^2) {100 * np.mean(r ** 2):.1f}%; "
              f"r < 0 in {(r < 0).sum()}/{n} scans")

    print("\n== Finding 4: reproducibility and plausibility")
    rep = df.groupby("subject").size()
    rep = rep[rep >= 2]
    idx = {s: np.where(df.subject.values == s)[0] for s in rep.index}
    sel = np.concatenate([idx[s] for s in rep.index])
    own = np.concatenate([[s] * len(idx[s]) for s in rep.index])
    pairs = [(i, j) for s in rep.index for i, j in itertools.combinations(idx[s], 2)]
    diff = sum(df.ses[i] != df.ses[j] for i, j in pairs)
    chance_exact = 100 * sum(len(idx[s]) * (len(idx[s]) - 1) for s in rep.index) / (len(sel) * (len(sel) - 1))
    print(f"{len(rep)} people with repeat scans, {len(sel)} scans, {len(pairs)} same-person pairs "
          f"({diff} from different sessions); chance = 1/{len(rep)} = {100 / len(rep):.1f}% "
          f"(exactly {chance_exact:.1f}% for nearest-neighbour matching)")

    def ident(M):
        C = np.corrcoef(M)
        np.fill_diagonal(C, -np.inf)
        return 100 * sum(own[C[i].argmax()] == own[i] for i in range(len(sel))) / len(sel)

    rng = np.random.default_rng(0)
    subs = list(rep.index)
    for k, X in F.items():
        within = np.mean([stats.pearsonr(X[i], X[j])[0] for i, j in pairs])
        between = []
        for _ in range(3000):
            a, b = rng.choice(len(subs), 2, replace=False)
            between.append(stats.pearsonr(X[rng.choice(idx[subs[a]])], X[rng.choice(idx[subs[b]])])[0])
        v = ident(X[sel])
        print(f"{k}: same person {within:.2f}, different people {np.mean(between):.2f}, "
              f"closest-scan identification {v:.1f}% ({round(v * len(sel) / 100)}/{len(sel)})")
    iu = np.triu_indices(246, 1)
    is_hom = (iu[0] % 2 == 0) & (iu[1] == iu[0] + 1)
    hom, oth = [], []
    for p in df.norm:
        v = np.load(os.path.join(p, "fc_fisher_z.npy"))[iu]
        hom.append(v[is_hom].mean())
        oth.append(v[~is_hom].mean())
    hom, oth = np.array(hom), np.array(oth)
    bad = [(df.stage[i], df.subject[i], df.ses[i], df.run[i]) for i in np.where(hom <= oth)[0]]
    print(f"left-right homologous pairs: {is_hom.sum()}; mean Fisher-z {hom.mean():.3f} vs {oth.mean():.3f} "
          f"for the other {(~is_hom).sum():,} pairs; homologous > other in {(hom > oth).sum()}/{n} scans; "
          f"exceptions: {bad}")

    print("\n== Granularity (mALFF closest-scan identification)")
    X = F["mALFF"]
    rng = np.random.default_rng(0)
    ns = [5, 10, 25, 50, 75, 100, 150, 200, 246]
    mu = [np.mean([ident(X[sel][:, rng.choice(246, m, replace=False)]) for _ in range(1 if m == 246 else 40)])
          for m in ns]
    print("random subsets of regions:", ", ".join(f"{m}: {v:.1f}%" for m, v in zip(ns, mu)))
    mg = []
    for k in [1, 2, 3, 6, 41]:
        m = 246 // k
        mg.append((m, ident(X[sel][:, :m * k].reshape(len(sel), m, k).mean(axis=2))))
    print("merging consecutive atlas labels:", ", ".join(f"{m}: {v:.1f}%" for m, v in mg))
    c = np.corrcoef(X.T)
    ev = np.linalg.eigvalsh(c)[::-1]
    ev = ev[ev > 1e-10]
    cum = 100 * np.cumsum(ev) / ev.sum()
    print(f"two regions share {100 * np.mean(c[iu] ** 2):.1f}% of their variance on average ({n} scans); "
          f"components for 80% / 95%: {int(np.searchsorted(cum, 80)) + 1} / {int(np.searchsorted(cum, 95)) + 1} "
          f"(rank {len(ev)})")

    print("\n== Residual motion: regions whose value tracks mean FD (Pearson, p < 0.05, uncorrected)")
    q = pd.read_csv(os.path.join(AUD, "dataset_qc_summary.csv"))
    m = df.merge(q[["group", "subject", "session", "run", "mean_FD_mm"]],
                 left_on=["group", "subject", "ses", "run"], right_on=["group", "subject", "session", "run"])
    fd = m.mean_FD_mm.values
    for label, keep in [("164 scans (artefact scan excluded)", (m.subject != ARTEFACT).values),
                        ("all 165 scans", np.ones(n, bool))]:
        out = {k: sum(stats.pearsonr(X_[keep, j], fd[keep])[1] < 0.05 for j in range(246))
               for k, X_ in F.items()}
        print(f"{label}: " + ", ".join(f"{k} {v}/246" for k, v in out.items())
              + f"; expected by chance about {0.05 * 246:.0f}")

    print("\n== Site concentration (site = prefix of the ADNI subject ID)")
    df["site"] = df.subject.str.extract(r"sub-(\d{3})S")[0]
    print(f"{df.site.nunique()} sites in all")
    for st in ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]:
        vc = df[df.stage == st].site.value_counts()
        print(f"{st:5s} {vc.sum():3d} scans, {len(vc):2d} sites, largest site {100 * vc.iloc[0] / vc.sum():.0f}%, "
              f"two largest {100 * vc.iloc[:2].sum() / vc.sum():.0f}%")


if __name__ == "__main__":
    main()
