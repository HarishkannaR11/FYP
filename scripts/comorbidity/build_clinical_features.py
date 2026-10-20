"""Map the ADNI IDA clinical export (MMSE, GDSCALE, Global CDR) onto the
116-subject cohort, clean it, and report what is and is not determinable.

Inputs (read-only):
  Comorbidity/new_comorbidity/gdscale-cdr-mmse/idaSearch_10_20_2026_<GROUP>.csv
  derivatives/model_ready/{subjects,stages,acq_ids}.npy

Outputs (derivatives/comorbidity_features/):
  clinical_visits_116.csv     every visit for every cohort subject, cleaned
  clinical_subject_116.csv    one row per subject: baseline / median / last
  clinical_qc_report.txt      missingness, range violations, what failed

KNOWN LIMITATION, carried into every output file
------------------------------------------------
The IDA export gives Age but no study date, so a visit CANNOT be tied to the
rs-fMRI scan the cohort used. 103 of 116 subjects have several visits sharing
the same series description, and within a subject MMSE moves by up to 12
points and Global CDR by a full point. Any single value is therefore a choice
of visit, not a measurement of the scanned timepoint. Nothing here is
imputed; missing stays missing. See the README for the IDA re-export that
would fix this.
"""
import csv
import glob
import os
import re
import collections
import statistics

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "derivatives", "model_ready")
IDA = os.path.join(ROOT, "Comorbidity", "new_comorbidity", "gdscale-cdr-mmse")
OUT = os.path.join(ROOT, "derivatives", "comorbidity_features")

STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
SCORES = ["MMSE Total Score", "GDSCALE Total Score", "Global CDR"]
SHORT = {"MMSE Total Score": "MMSE",
         "GDSCALE Total Score": "GDSCALE",
         "Global CDR": "GlobalCDR"}
# documented instrument ranges -- values outside are reported, never silently fixed
VALID = {"MMSE": (0.0, 30.0),
         "GDSCALE": (0.0, 15.0),
         "GlobalCDR": (0.0, 3.0)}
CDR_ALLOWED = {0.0, 0.5, 1.0, 2.0, 3.0}

_log = []


def log(m=""):
    print(m)
    _log.append(m)


def to_adni(s):
    m = re.match(r"sub-(\d{3})S(\d+)$", s)
    if not m:
        raise SystemExit(f"unexpected subject id: {s}")
    return f"{m.group(1)}_S_{m.group(2)}"


def num(v):
    v = (v or "").strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def main():
    os.makedirs(OUT, exist_ok=True)

    subs = np.load(os.path.join(SRC, "subjects.npy")).astype(str)
    stages = np.load(os.path.join(SRC, "stages.npy")).astype(str)
    acqs = np.load(os.path.join(SRC, "acq_ids.npy")).astype(str)
    cohort = {}
    for s, st, a in zip(subs, stages, acqs):
        cohort[to_adni(s)] = {"bids": s, "stage": st, "acq": a}
    if len(cohort) != 116:
        raise SystemExit(f"expected 116 unique subjects, got {len(cohort)}")

    log("=" * 78)
    log("ADNI CLINICAL FEATURES -> 116-SUBJECT COHORT")
    log("=" * 78)
    log(f"cohort: {len(cohort)} subjects, "
        f"{dict(collections.Counter(stages))}")

    # ------------------------------------------------------------ read IDA
    raw, files = [], sorted(glob.glob(os.path.join(IDA, "*.csv")))
    if not files:
        raise SystemExit(f"no IDA csv found in {IDA}")
    for f in files:
        n = 0
        for r in csv.DictReader(open(f, encoding="utf-8-sig")):
            r["_src"] = os.path.basename(f)
            raw.append(r)
            n += 1
        log(f"  read {n:5d} rows  {os.path.basename(f)}")
    log(f"  total {len(raw)} rows across {len(files)} files")

    # ------------------------------------------------- filter to the cohort
    mine = [r for r in raw if r["Subject ID"] in cohort]
    log("")
    log(f"rows belonging to the 116 cohort subjects: {len(mine)} "
        f"(dropped {len(raw) - len(mine)} from other subjects)")
    present = set(r["Subject ID"] for r in mine)
    missing = sorted(set(cohort) - present)
    log(f"cohort subjects with at least one row: {len(present)} / {len(cohort)}")
    if missing:
        log(f"  NOT FOUND IN THE EXPORT: {missing}")

    # --------------------------------------- stage agreement cross-check
    log("")
    log("Research Group in the export vs the cohort stage label:")
    disagree = collections.Counter()
    for r in mine:
        g = r["Research Group"].strip()
        s = cohort[r["Subject ID"]]["stage"]
        if g != s:
            disagree[(s, g)] += 1
    if disagree:
        for (s, g), n in sorted(disagree.items(), key=lambda kv: -kv[1]):
            log(f"  cohort={s:5s} export={g:8s} {n:5d} rows")
        log("  NOTE: ADNI 'Research Group' is the group at that visit and may")
        log("  legitimately differ from the stage assigned to the scan used.")
    else:
        log("  every row agrees")

    # --------------------------------------------- clean + dedupe visits
    seen, visits, bad = set(), [], []
    for r in mine:
        sid = r["Subject ID"]
        age = num(r["Age"])
        rec = {"adni_id": sid,
               "bids_id": cohort[sid]["bids"],
               "cohort_stage": cohort[sid]["stage"],
               "export_research_group": r["Research Group"].strip(),
               "sex": r["Sex"].strip(),
               "age_at_visit": age,
               "modality": r["Modality"].strip(),
               "description": r["Description"].strip(),
               "source_file": r["_src"]}
        for col in SCORES:
            v = num(r[col])
            k = SHORT[col]
            if v is not None:
                lo, hi = VALID[k]
                if not (lo <= v <= hi):
                    bad.append((sid, k, v, "outside instrument range"))
                    v = None
                elif k == "GlobalCDR" and v not in CDR_ALLOWED:
                    bad.append((sid, k, v, "not an allowed CDR level"))
                    v = None
            rec[k] = v
        key = (sid, age, rec["description"],
               rec["MMSE"], rec["GDSCALE"], rec["GlobalCDR"])
        if key in seen:
            continue
        seen.add(key)
        visits.append(rec)

    visits.sort(key=lambda r: (r["adni_id"],
                               r["age_at_visit"] if r["age_at_visit"] is not None else -1))
    log("")
    log(f"visit rows after de-duplication: {len(visits)} "
        f"(removed {len(mine) - len(visits)} exact duplicates)")
    if bad:
        log(f"values rejected as out of range: {len(bad)}")
        for sid, k, v, why in bad[:10]:
            log(f"    {sid} {k}={v} -- {why}")
    else:
        log("values rejected as out of range: 0")

    p = os.path.join(OUT, "clinical_visits_116.csv")
    cols = ["adni_id", "bids_id", "cohort_stage", "export_research_group",
            "sex", "age_at_visit", "modality", "description",
            "MMSE", "GDSCALE", "GlobalCDR", "source_file"]
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows({c: ("" if r[c] is None else r[c]) for c in cols}
                    for r in visits)
    log(f"wrote {os.path.relpath(p, ROOT)}")

    # ------------------------------------------------- per-subject summary
    by = collections.defaultdict(list)
    for r in visits:
        by[r["adni_id"]].append(r)

    srows = []
    for sid in sorted(cohort):
        rr = sorted(by.get(sid, []),
                    key=lambda r: r["age_at_visit"] if r["age_at_visit"] is not None else -1)
        row = {"adni_id": sid,
               "bids_id": cohort[sid]["bids"],
               "cohort_stage": cohort[sid]["stage"],
               "stage_index": STAGES.index(cohort[sid]["stage"]),
               "n_visits": len(rr)}
        for k in ("MMSE", "GDSCALE", "GlobalCDR"):
            vals = [(r["age_at_visit"], r[k]) for r in rr if r[k] is not None]
            row[f"{k}_n_observed"] = len(vals)
            if vals:
                v = [x[1] for x in vals]
                row[f"{k}_baseline"] = v[0]
                row[f"{k}_last"] = v[-1]
                # median_low, not median: these are ordinal clinical scales.
                # Averaging the two middle visits invents levels that do not
                # exist -- MMSE 12.5 on an integer scale, Global CDR 0.25 or
                # 1.5 when the only defined levels are 0/0.5/1/2/3.
                # median_low always returns a value actually observed.
                row[f"{k}_median"] = statistics.median_low(v)
                row[f"{k}_min"] = min(v)
                row[f"{k}_max"] = max(v)
                row[f"{k}_range"] = max(v) - min(v)
            else:
                for suf in ("baseline", "last", "median", "min", "max", "range"):
                    row[f"{k}_{suf}"] = None
        srows.append(row)

    scols = ["adni_id", "bids_id", "cohort_stage", "stage_index", "n_visits"]
    for k in ("MMSE", "GDSCALE", "GlobalCDR"):
        scols += [f"{k}_n_observed", f"{k}_baseline", f"{k}_last",
                  f"{k}_median", f"{k}_min", f"{k}_max", f"{k}_range"]
    p = os.path.join(OUT, "clinical_subject_116.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=scols)
        w.writeheader()
        w.writerows({c: ("" if r[c] is None else r[c]) for c in scols}
                    for r in srows)
    log(f"wrote {os.path.relpath(p, ROOT)}")

    # ------------------------------------------- per-stage roster + check
    # Each stage's subjects must come from that stage's own export file.
    # A subject showing up under another stage would mean the cohort label
    # and the export disagree, so it is checked rather than assumed.
    src_of = collections.defaultdict(set)
    for r in visits:
        g = r["source_file"].replace("idaSearch_10_20_2026_", "").replace(".csv", "")
        src_of[r["adni_id"]].add(g)

    log("")
    log("=" * 78)
    log("PER-STAGE SUBJECT MAPPING")
    log("=" * 78)
    rrows, problems = [], 0
    for g in STAGES:
        mine = sorted(s for s in cohort if cohort[s]["stage"] == g)
        log("")
        log(f"--- {g}  ({len(mine)} subjects) ---")
        log(f"  {'#':>3s} {'ADNI ID':13s} {'BIDS ID':14s} {'file':6s} "
            f"{'vis':>4s} {'MMSE':>6s} {'GDS':>5s} {'CDR':>5s}")
        for i, sid in enumerate(mine, 1):
            sm = next(r for r in srows if r["adni_id"] == sid)
            files = sorted(src_of.get(sid, []))
            flag = "" if files == [g] else "  <-- WRONG FILE"
            if flag:
                problems += 1
            def fmt(k):
                v = sm[f"{k}_median"]
                return "--" if v is None else f"{v:g}"
            log(f"  {i:3d} {sid:13s} {sm['bids_id']:14s} "
                f"{(files[0] if files else '??'):6s} {sm['n_visits']:4d} "
                f"{fmt('MMSE'):>6s} {fmt('GDSCALE'):>5s} {fmt('GlobalCDR'):>5s}{flag}")
            # full provenance chain: the folder the features were built from,
            # through to the export row the clinical scores came from. The
            # folder is verified on disk, not assumed from the naming rule.
            _, ses, run = cohort[sid]["acq"].split("|")
            rel = os.path.join("derivatives", "phase1_selected_20", g,
                               sm["bids_id"], ses, run)
            rrows.append({"cohort_stage": g, "n_in_stage": i, "adni_id": sid,
                          "bids_id": sm["bids_id"],
                          "phase1_source_dir": rel.replace("\\", "/"),
                          "phase1_dir_exists": os.path.isdir(
                              os.path.join(ROOT, rel)),
                          "ida_source_file": (f"idaSearch_10_20_2026_{files[0]}.csv"
                                              if files else ""),
                          "n_visits": sm["n_visits"],
                          "MMSE_median": sm["MMSE_median"],
                          "GDSCALE_median": sm["GDSCALE_median"],
                          "GlobalCDR_median": sm["GlobalCDR_median"],
                          "all_three_observed": all(
                              sm[f"{k}_n_observed"] > 0
                              for k in ("MMSE", "GDSCALE", "GlobalCDR"))})
    log("")
    log(f"subjects sourced from the wrong stage file: {problems}")
    log(f"subjects appearing in more than one stage file: "
        f"{sum(1 for s in cohort if len(src_of.get(s, [])) > 1)}")

    p = os.path.join(OUT, "stage_subject_mapping_116.csv")
    rcols = ["cohort_stage", "n_in_stage", "adni_id", "bids_id",
             "phase1_source_dir", "phase1_dir_exists", "ida_source_file",
             "n_visits", "MMSE_median", "GDSCALE_median",
             "GlobalCDR_median", "all_three_observed"]
    n_bad = sum(1 for r in rrows if not r["phase1_dir_exists"])
    log(f"phase1_selected_20 folders verified on disk: "
        f"{len(rrows) - n_bad} / {len(rrows)}")
    if n_bad:
        for r in rrows:
            if not r["phase1_dir_exists"]:
                log(f"    MISSING FOLDER: {r['phase1_source_dir']}")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=rcols)
        w.writeheader()
        w.writerows({c: ("" if r[c] is None else r[c]) for c in rcols}
                    for r in rrows)
    log(f"wrote {os.path.relpath(p, ROOT)}")

    # -------------------------------------------------------- missingness
    log("")
    log("COMPLETENESS (subjects with at least one observed value)")
    log(f"  {'score':10s} {'subjects':>9s} {'missing':>8s} "
        f"{'visits obs':>11s} {'within-subj range: median':>26s}")
    for k in ("MMSE", "GDSCALE", "GlobalCDR"):
        have = [r for r in srows if r[f"{k}_n_observed"] > 0]
        rng = [r[f"{k}_range"] for r in have if r[f"{k}_range"] is not None]
        nvis = sum(1 for r in visits if r[k] is not None)
        med = statistics.median(rng) if rng else 0.0
        log(f"  {k:10s} {len(have):9d} {len(srows) - len(have):8d} "
            f"{nvis:11d} {med:26.2f}")

    log("")
    log("Subjects with NO observed value for a score:")
    for k in ("MMSE", "GDSCALE", "GlobalCDR"):
        none = [r["adni_id"] for r in srows if r[f"{k}_n_observed"] == 0]
        log(f"  {k:10s} {len(none)}  {none if none else ''}")

    # ------------------------------- the limitation, stated with evidence
    log("")
    log("=" * 78)
    log("WHY NO SINGLE SCAN-MATCHED VALUE IS PROVIDED")
    log("=" * 78)
    multi = sum(1 for r in srows if r["n_visits"] > 1)
    log(f"  subjects with more than one visit in the export : {multi} / 116")
    for k in ("MMSE", "GDSCALE", "GlobalCDR"):
        moved = [r for r in srows
                 if r[f"{k}_range"] is not None and r[f"{k}_range"] > 0]
        if moved:
            mx = max(r[f"{k}_range"] for r in moved)
            log(f"  {k:10s} changes across visits in {len(moved):3d} subjects; "
                f"largest within-subject swing {mx:.1f}")
    log("")
    log("  The export carries Age but no study date, so a visit cannot be")
    log("  matched to the scan the cohort used. Choosing a visit therefore")
    log("  changes the value by the swings above. baseline / last / median")
    log("  are all provided; pick one deliberately and say which was used.")

    p = os.path.join(OUT, "clinical_qc_report.txt")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")
    print(f"\nwrote {os.path.relpath(p, ROOT)}")


if __name__ == "__main__":
    main()
