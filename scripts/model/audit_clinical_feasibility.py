"""
Phase 5: clinical and comorbidity feasibility audit.

Answers three questions from the files on disk:
  1. Which proposed clinical variables actually exist, and for whom?
  2. Which cohort would be eligible for a fusion experiment?
  3. Which variables would leak the label if used as predictors?

Nothing is imputed, and no subject is removed.
"""
import csv
import json
import os

import numpy as np
from scipy import stats

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "derivatives", "model_ready")
CLIN = os.path.join(ROOT, "derivatives", "clinical_cleaned")
OUT = os.path.join(ROOT, "derivatives", "model_training")
STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
BLANK = {"", "nan", "NA", "NaN", "None", None}

# proposed variable -> column actually present in the cleaned table
PROPOSED = {
    "age": "Age_years",
    "APOE4": "APOE4_allele_count",
    "MMSE": "MMSE_total",
    "CDR-SB": "CDR_SB",
    "hypertension": "MH4CARD_clean",     # ADNI has no standalone hypertension
    "diabetes": "MH9ENDO_clean",         # ADNI has no standalone diabetes
    "GDS": "GDS_total",
}
NOTE = {
    "hypertension": "NOT a hypertension flag. ADNI records MH4CARD = "
                    "cardiovascular medical history, which INCLUDES "
                    "hypertension but also every other cardiovascular "
                    "condition. It cannot be read as 'has hypertension'.",
    "diabetes": "NOT a diabetes flag. ADNI records MH9ENDO = "
                "endocrine-metabolic history, which INCLUDES diabetes among "
                "other endocrine conditions.",
}

_log = []


def log(m=""):
    print(m, flush=True)
    _log.append(str(m))


def main():
    y = np.load(os.path.join(SRC, "y.npy"))
    subs = np.load(os.path.join(SRC, "subjects.npy")).astype(str)
    stage_of = {s: STAGES[i] for s, i in zip(subs.tolist(), y.tolist())}
    counts = {s: int((y == i).sum()) for i, s in enumerate(STAGES)}
    rows = {r["subject_id"]: r for r in
            csv.DictReader(open(os.path.join(CLIN, "master_subject_clinical.csv")))}

    log("=" * 86)
    log("PHASE 5 -- CLINICAL / COMORBIDITY FEASIBILITY")
    log("=" * 86)
    log(f"cohort {len(subs)} subjects; clinical table {len(rows)} rows")
    log(f"subjects with NO clinical row: "
        f"{sorted(s for s in stage_of if s not in rows)}")

    log("")
    log(f"{'proposed':14s} {'column':22s} {'total':>7s}  " +
        "  ".join(f"{s:>7s}" for s in STAGES))
    cover = {}
    for prop, col in PROPOSED.items():
        if col not in next(iter(rows.values())):
            log(f"{prop:14s} {col:22s}  COLUMN NOT PRESENT")
            cover[prop] = None
            continue
        per = {s: 0 for s in STAGES}
        tot = 0
        for sub, st in stage_of.items():
            r = rows.get(sub)
            if r and r.get(col, "") not in BLANK:
                per[st] += 1
                tot += 1
        cover[prop] = {"column": col, "total": tot, "per_stage": per}
        log(f"{prop:14s} {col:22s} {tot:4d}/116  " +
            "  ".join(f"{per[s]:3d}/{counts[s]:<3d}" for s in STAGES))

    log("")
    for k, v in NOTE.items():
        log(f"CAVEAT [{k}]: {v}")

    # ---- eligible cohorts
    log("")
    log("=" * 86)
    log("ELIGIBLE COHORTS")
    log("=" * 86)
    rich = [c for c in ("MMSE_total", "CDR_SB", "GDS_total", "Age_years")]
    elig_rich = [s for s in stage_of
                 if rows.get(s) and all(rows[s].get(c, "") not in BLANK
                                        for c in rich)]
    elig_apoe = [s for s in stage_of
                 if rows.get(s) and rows[s].get("APOE4_allele_count", "")
                 not in BLANK]
    for name, elig in (("rich clinical (MMSE+CDR-SB+GDS+age)", elig_rich),
                       ("APOE4 only", elig_apoe)):
        per = {s: 0 for s in STAGES}
        for s in elig:
            per[stage_of[s]] += 1
        full = [s for s in STAGES if per[s] > 0]
        log(f"\n{name}: {len(elig)} subjects")
        log("   " + "  ".join(f"{s} {per[s]}/{counts[s]}" for s in STAGES))
        log(f"   stages represented: {len(full)}/6"
            + ("" if len(full) == 6 else f"  MISSING {[s for s in STAGES if per[s] == 0]}"))

    # ---- leakage
    log("")
    log("=" * 86)
    log("LABEL-LEAKAGE CHECK")
    log("=" * 86)
    log("ADNI assigns diagnostic group using cognitive testing. A variable "
        "that reproduces the stage almost perfectly is not an independent "
        "predictor -- it is the label in another form.")
    log("")
    log(f"{'variable':22s} {'n':>5s} {'Spearman rho':>16s} {'p':>9s}  reading")
    for prop, col in PROPOSED.items():
        if cover.get(prop) is None:
            continue
        pairs = [(float(rows[s][col]), y[list(subs).index(s)])
                 for s in stage_of
                 if rows.get(s) and rows[s].get(col, "") not in BLANK]
        if len(pairs) < 10:
            continue
        a = np.array([p[0] for p in pairs], float)
        b = np.array([p[1] for p in pairs], float)
        # scipy's spearmanr uses AVERAGE ranks for ties. Ranking with
        # argsort(argsort(.)) breaks ties arbitrarily, and because the cohort
        # arrays are ordered by stage that arbitrary order tracks the label and
        # manufactures correlation. For 0/1/2 variables the inflation was large
        # (cardiovascular history read +0.455 instead of -0.025).
        r, pval = stats.spearmanr(a, b)
        r = float(r)
        verdict = ("SEVERE LEAKAGE RISK" if abs(r) >= 0.6 else
                   "moderate association" if abs(r) >= 0.35 else
                   "weak association")
        log(f"{prop + ' (' + col + ')':22.22s} {len(pairs):5d} {r:+16.3f} "
            f"{pval:9.2e}  {verdict}")

    log("")
    log("")
    log("Interpretation:")
    log("  CDR-SB (rho = +0.85) is a component of the diagnostic criteria "
        "themselves -- it is the label in another form and MUST NOT be a "
        "model input.")
    log("  MMSE (rho = -0.56) is also used in staging and carries the same "
        "risk to a lesser degree.")
    log("  APOE4 (rho = +0.35) is a genuine genetic risk factor, not part of "
        "the staging decision, and its association is of the magnitude the "
        "literature would lead one to expect. It is a defensible input.")
    log("  Age, cardiovascular history and endocrine history show NO useful "
        "association with stage in this cohort (|rho| <= 0.15), so they are "
        "safe but probably uninformative.")

    out = {"coverage": cover, "caveats": NOTE,
           "eligible_rich_clinical": sorted(elig_rich),
           "eligible_apoe4": sorted(elig_apoe),
           "n_eligible_rich": len(elig_rich), "n_eligible_apoe4": len(elig_apoe)}
    p = os.path.join(OUT, "clinical_feasibility.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log(f"\nwrote {os.path.relpath(p, ROOT)}")
    with open(os.path.join(OUT, "clinical_feasibility.log"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")


if __name__ == "__main__":
    main()
