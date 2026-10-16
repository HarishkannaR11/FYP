"""
STEP 1b -- acquisition-level duplicate detection for the new CN/LMCI archives.

Matches each ZIP series against the existing dataset on
(ADNI subject, acquisition date), not just subject id. Read-only.
"""
import os

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(ROOT, "derivatives", "EXTRA_DATA_INVENTORY")
INV = os.path.join(OUT, "extra_series_inventory.csv")
MAP = os.path.join(ROOT, "audit", "bids_source_mapping.csv")

TIER1 = {"Resting_State_fMRI", "Extended_Resting_State_fMRI",
         "Axial_rsfMRI__Eyes_Open__10_min__-PJ"}
REVIEW = {"Extended_Resting_State_fMRI_CLEAR"}


def main():
    inv = pd.read_csv(INV)
    mp = pd.read_csv(MAP)

    # existing acquisitions keyed on (adni subject, YYYYMMDD)
    mp["date"] = mp.acquisition_datetime.astype(str).str[:8]
    existing = set(zip(mp.subject_id, mp.date))
    print(f"existing acquisitions with a known date: {len(existing)} "
          f"(from {mp.subject_id.nunique()} subjects)")

    inv["date"] = inv.acq_datetime.str.replace("-", "", regex=False).str[:8]
    inv["is_duplicate"] = [(s, d) in existing
                           for s, d in zip(inv.adni_subject, inv.date)]
    inv["protocol_tier"] = inv.description.map(
        lambda d: "TIER1" if d in TIER1 else
                  ("REVIEW" if d in REVIEW else "UNKNOWN"))

    print("\n" + "=" * 74)
    print("DUPLICATE CHECK  (subject + acquisition date)")
    print("=" * 74)
    for stage, g in inv.groupby("stage"):
        dup = int(g.is_duplicate.sum())
        print(f"\n{stage}: {len(g)} series in archive")
        print(f"   already in project (duplicate) : {dup}")
        print(f"   genuinely NEW                  : {len(g) - dup}")
        new = g[~g.is_duplicate]
        print(f"   new, by description:")
        for d, c in new.description.value_counts().items():
            tier = "TIER1" if d in TIER1 else ("REVIEW" if d in REVIEW else "UNKNOWN")
            print(f"      {c:3d}  [{tier:7s}] {d}")
        print(f"   new subjects among them        : "
              f"{new[~new.subject_already_in_project].adni_subject.nunique()}")
        print(f"   DICOM slices (new series)      : {int(new.n_dicom.sum()):,}")

    print("\n" + "=" * 74)
    print("SERIES SIZE SANITY (DICOM count per series)")
    print("=" * 74)
    for stage, g in inv[~inv.is_duplicate].groupby("stage"):
        for d, gg in g.groupby("description"):
            print(f"  {stage:5s} {d[:42]:44s} n={len(gg):3d}  "
                  f"slices min={gg.n_dicom.min():5d} "
                  f"median={int(gg.n_dicom.median()):5d} "
                  f"max={gg.n_dicom.max():5d}")

    f = os.path.join(OUT, "extra_series_dedup.csv")
    inv.to_csv(f, index=False)
    print(f"\nwrote {f}")

    print("\n" + "=" * 74)
    print("CONVERSION CANDIDATES")
    print("=" * 74)
    cand = inv[(~inv.is_duplicate) & (inv.protocol_tier == "TIER1")]
    for stage, g in cand.groupby("stage"):
        print(f"  {stage:5s} {len(g):3d} series  "
              f"({g.adni_subject.nunique()} subjects, "
              f"{int(g.n_dicom.sum()):,} DICOM files)")
    skipped = inv[(~inv.is_duplicate) & (inv.protocol_tier != "TIER1")]
    if len(skipped):
        print(f"\n  held for review ({len(skipped)}):")
        for _, r in skipped.iterrows():
            print(f"      {r.stage:5s} {r.adni_subject}  {r.description}")


if __name__ == "__main__":
    main()
