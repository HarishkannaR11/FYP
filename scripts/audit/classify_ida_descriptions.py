"""
Classify ADNI IDA image-collection CSV exports by Image Description.

IDA's `Modality = fMRI` filter returns three different kinds of series:
raw resting-state BOLD, scanner-derived products (MoCoSeries), and ASL
derivatives (Perfusion_Weighted, relCBF). Only the first is usable input
for this pipeline. This script reports what is in an export so the
selection can be made before any image is downloaded.

Read-only. Never downloads, never touches BIDS/ or derivatives/.

Usage:
    python scripts/audit/classify_ida_descriptions.py <csv-or-dir> [...]
    python scripts/audit/classify_ida_descriptions.py Comorbidity/
"""
import os
import re
import sys

import pandas as pd

# Tier 1: basic protocol, TR ~3 s, ~3.3 mm -- directly mergeable with the
#         existing 165 after truncation to a common volume count.
# Tier 2: a genuinely different acquisition. Keep as a separate protocol
#         domain; do not merge into the training pool without a decision.
# EXCLUDE: not raw resting-state BOLD at all.
RULES = [
    (r"^resting\s*state\s*fmri$",                 "TIER1", "ADNI2/3 basic rs-fMRI"),
    (r"^extended\s*resting\s*state\s*fmri$",      "TIER1", "basic sequence, more volumes"),
    (r"^axial[\s_]*rsfmri",                       "TIER1", "basic rs-fMRI, vendor naming"),
    (r"^axial[\s_]*fcmri",                        "TIER1", "fcMRI = basic rs-fMRI (Philips naming)"),
    (r"axial[\s_]*mb[\s_]*rsfmri",                "TIER2", "multiband/SMS, TR~600ms, 2.5mm"),
    (r"axial[\s_]*hb[\s_]*rsfmri",                "TIER2", "HB unconfirmed -- check DICOM headers"),
    (r"^mocoseries$",                             "EXCLUDE", "scanner-derived motion-corrected copy"),
    (r"^perfusion[\s_]*weighted$",                "EXCLUDE", "ASL perfusion map, not BOLD"),
    (r"^relcbf$",                                 "EXCLUDE", "ASL rel. CBF, not BOLD"),
]
ORDER = ["TIER1", "TIER2", "EXCLUDE", "UNKNOWN"]
NOTE = {
    "TIER1": "mergeable with the current cohort (verify TR from headers)",
    "TIER2": "different protocol -- hold out as a separate domain",
    "EXCLUDE": "NOT raw resting-state BOLD -- must not enter the cohort",
    "UNKNOWN": "unrecognised -- inspect before deciding",
}


def classify(desc):
    d = re.sub(r"\s+", " ", str(desc)).strip()
    bare = re.sub(r"\((msv\d+|eyes\s*open)\)", "", d, flags=re.I).strip()
    for pat, tier, why in RULES:
        if re.search(pat, bare, flags=re.I) or re.search(pat, d, flags=re.I):
            return tier, why
    return "UNKNOWN", "no rule matched"


def load(paths):
    frames = []
    for p in paths:
        files = ([os.path.join(p, f) for f in sorted(os.listdir(p))
                  if f.lower().endswith(".csv")] if os.path.isdir(p) else [p])
        for f in files:
            try:
                d = pd.read_csv(f)
            except Exception as e:                      # noqa: BLE001
                print(f"  skipped {f}: {e}")
                continue
            if "Description" not in d.columns:
                continue
            d["_file"] = os.path.basename(f)
            frames.append(d)
    if not frames:
        sys.exit("No CSV with a 'Description' column found.")
    return pd.concat(frames, ignore_index=True)


def main(paths):
    a = load(paths)
    a[["tier", "why"]] = a.Description.apply(
        lambda x: pd.Series(classify(x)))

    sub = "Subject" if "Subject" in a.columns else None
    grp = "Group" if "Group" in a.columns else None
    print(f"\nrows {len(a)}" + (f"   subjects {a[sub].nunique()}" if sub else ""))
    print(f"files: {', '.join(sorted(a._file.unique()))}")

    for t in ORDER:
        d = a[a.tier == t]
        if d.empty:
            continue
        print("\n" + "=" * 74)
        print(f"{t}  --  {NOTE[t]}")
        print("=" * 74)
        for desc, n in d.Description.value_counts().items():
            why = d[d.Description == desc].why.iloc[0]
            s = f"   subjects={d[d.Description == desc][sub].nunique():4d}" if sub else ""
            print(f"  {n:6d}{s}   {desc}")
            print(f"{'':>14}({why})")

    if grp:
        print("\n" + "=" * 74)
        print("TIER x RESEARCH GROUP")
        print("=" * 74)
        ct = pd.crosstab(a[grp], a.tier)
        print(ct.reindex(columns=[c for c in ORDER if c in ct.columns]).to_string())

    t1 = a[a.tier == "TIER1"]
    print("\n" + "=" * 74)
    print("WHAT YOU WOULD ACTUALLY DOWNLOAD (Tier 1 only)")
    print("=" * 74)
    print(f"  images   {len(t1)}")
    if sub:
        print(f"  subjects {t1[sub].nunique()}")
    if grp:
        for g, d in t1.groupby(grp):
            s = f"  subjects={d[sub].nunique():4d}" if sub else ""
            print(f"    {g:10s} images={len(d):5d}{s}")

    bad = a[a.tier == "EXCLUDE"]
    if len(bad):
        print(f"\n  {len(bad)} rows would be silently wrong if downloaded "
              f"({100 * len(bad) / len(a):.1f}% of the export).")
    unk = a[a.tier == "UNKNOWN"]
    if len(unk):
        print(f"\n  {len(unk)} UNRECOGNISED rows -- decide before downloading:")
        for desc, n in unk.Description.value_counts().items():
            print(f"      {n:6d}  {desc}")

    # A study visit whose ONLY fMRI series are EXCLUDE-tier is a subject you
    # lose by excluding them. Decide that consciously rather than by default.
    if sub and "Visit" in a.columns:
        key = [sub, "Visit"]
        usable = a[a.tier.isin(["TIER1", "TIER2"])].set_index(key).index
        orphan = a[a.tier == "EXCLUDE"].set_index(key)
        orphan = orphan[~orphan.index.isin(usable)]
        print("\n" + "=" * 74)
        print("VISITS WITH NO USABLE SERIES (lost by excluding derived products)")
        print("=" * 74)
        if orphan.empty:
            print("  none -- every excluded series has a raw companion at the "
                  "same visit.\n  Excluding them costs you nothing.")
        else:
            print(f"  {orphan.index.nunique()} visit(s), "
                  f"{orphan.reset_index()[sub].nunique()} subject(s):")
            for (s, v), d in orphan.groupby(level=[0, 1]):
                print(f"    {s}  {v}  -> only "
                      f"{', '.join(sorted(d.Description.unique()))}")
            print("\n  These are unusable anyway: a MoCoSeries carries no motion\n"
                  "  parameters, so Friston-24 cannot be built and the design\n"
                  "  matrix would drop from 27 regressors to 3.")

    out = "ida_description_classification.csv"
    cols = [c for c in [sub, grp, "Visit", "Modality", "Description",
                        "Acq Date", "Image Data ID", "tier", "why", "_file"]
            if c and c in a.columns]
    a[cols].to_csv(out, index=False)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["Comorbidity"])
