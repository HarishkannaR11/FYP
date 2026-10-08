"""
Numbers behind the "Brainnetome 246-ROI Parcellation for Feature Extraction"
slide and its two backup slides (docs/SecondReview/ppt/r2_presentation.tex).
Read-only: prints, claims nothing about any region being disease-relevant.

    python3 scripts/audit/roi_space_checks.py

Checks, over every acquisition that has biomarker outputs:
  * 246 x 246 FC and 135 x 246 ROI time series exist and are finite
  * no zero-variance ROI time series (and the smallest variance seen)
  * FC edges: 30,135 unique, share that are negative (global signal regression)
  * atlas coverage: voxels per ROI on the BOLD grid (min / median / max)
  * how much two ROIs overlap in mALFF across scans (mean r^2) and the number of
    principal components needed for 95 % of the variance
"""
import glob
import os

import numpy as np
import pandas as pd

R = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DRV = os.path.join(R, "derivatives", "fsfast")
AUD = os.path.join(DRV, "FINAL_6GROUP_AUDIT")
NRM = os.path.join(R, "derivatives", "biomarkers_normalized")
ORDER = ["CN_Final", "SMC_Final", "EMCI", "MCI", "LMCI", "AD"]
PAT = os.path.join("*", "sub-*", "ses-*", "run-*", "biomarkers")


def main():
    ts_files = sorted(glob.glob(os.path.join(DRV, PAT, "roi_timeseries.npy")))
    fc_files = sorted(glob.glob(os.path.join(DRV, PAT, "fc_matrix.npy")))
    print(f"acquisitions with ROI time series: {len(ts_files)}; with FC: {len(fc_files)}")

    shapes, bad, zero_var, min_var = set(), 0, 0, np.inf
    for f in ts_files:
        a = np.load(f)
        shapes.add(a.shape)
        bad += int((~np.isfinite(a)).sum())
        v = a.var(axis=0)
        zero_var += int((v == 0).sum())
        min_var = min(min_var, float(v.min()))
    print(f"ROI time-series shapes: {sorted(shapes)}; non-finite values: {bad}; "
          f"zero-variance ROI series: {zero_var}; smallest variance in any scan x ROI: {min_var:.2f}")

    iu = np.triu_indices(246, 1)
    fshapes, fbad, neg = set(), 0, []
    for f in fc_files:
        m = np.load(f)
        fshapes.add(m.shape)
        fbad += int((~np.isfinite(m)).sum())
        neg.append(100 * (m[iu] < 0).mean())
    neg = np.array(neg)
    print(f"FC shapes: {sorted(fshapes)}; non-finite values: {fbad}; unique edges per scan: {len(iu[0]):,}")
    print(f"negative edges: {neg.mean():.2f}% on average across scans "
          f"(per-scan range {neg.min():.1f}-{neg.max():.1f}%)")

    bn = pd.read_csv(os.path.join(AUD, "dataset_brainnetome_summary.csv"))
    print(f"atlas ROIs listed: {len(bn)}; status counts: {bn.STATUS.value_counts().to_dict()}; "
          f"voxels per ROI min/median/max: {bn.VOXEL_COUNT.min()}/{bn.VOXEL_COUNT.median():.0f}/"
          f"{bn.VOXEL_COUNT.max()}")
    print(f"cortical ROIs (IDs 1-210): {(bn.ROI_ID <= 210).sum()}; "
          f"subcortical (IDs 211-246): {(bn.ROI_ID > 210).sum()}")

    rows = []
    for g in ORDER:
        for d in sorted(glob.glob(os.path.join(NRM, g, "sub-*", "ses-*", "run-*"))):
            rows.append(dict(subject=d.split(os.sep)[-3], norm=d))
    df = pd.DataFrame(rows)
    X = np.vstack([np.load(os.path.join(p, "malff.npy")) for p in df.norm])

    def shared(M):
        c = np.corrcoef(M.T)
        return 100 * np.mean(c[iu] ** 2)

    rep = df.groupby("subject").size()
    rep = rep[rep >= 2]
    sel = np.concatenate([np.where(df.subject.values == s)[0] for s in rep.index])
    ev = np.linalg.eigvalsh(np.corrcoef(X.T))[::-1]
    ev = ev[ev > 0]
    cum = 100 * np.cumsum(ev) / ev.sum()
    print(f"mALFF between-ROI shared variance (mean r^2): {shared(X):.1f}% over all {len(df)} scans "
          f"from {df.subject.nunique()} subjects; {shared(X[sel]):.1f}% over the {len(sel)} scans of the "
          f"{len(rep)} subjects with repeat scans")
    print(f"principal components for 80% / 95% of the variance: "
          f"{int(np.searchsorted(cum, 80)) + 1} / {int(np.searchsorted(cum, 95)) + 1}")


if __name__ == "__main__":
    main()
