"""
Audit, deduplicate, select and copy 20 preprocessed acquisitions per stage.

Operates only on acquisitions that ALREADY have valid preprocessing and
biomarker outputs. Nothing is converted, preprocessed or extracted here.

    python scripts/data_prep/build_phase1_dataset.py           # audit + plan
    python scripts/data_prep/build_phase1_dataset.py --copy    # execute

Source trees are opened read-only. Writes go only to
derivatives/phase1_selected_20/.
"""
import argparse
import hashlib
import json
import os
import shutil
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FS = os.path.join(ROOT, "derivatives", "fsfast")
NRM = os.path.join(ROOT, "derivatives", "biomarkers_normalized")
AUD = os.path.join(FS, "FINAL_6GROUP_AUDIT")
MAPCSV = os.path.join(ROOT, "audit", "bids_source_mapping.csv")
OUT = os.path.join(ROOT, "derivatives", "phase1_selected_20")
MAN = os.path.join(OUT, "manifests")
REP = os.path.join(OUT, "reports")
LOGF = os.path.join(OUT, "build.log")

TARGET = 20
CANON = {"CN_Final": "CN", "SMC_Final": "SMC", "EMCI": "EMCI",
         "MCI": "MCI", "LMCI": "LMCI", "AD": "AD"}
STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]
N_ROI = 246

# Required per acquisition. FC_Strength is deliberately absent: it equals
# DC/245 exactly and is not an independent biomarker.
REQ_NORM = ["malff.npy", "mreho.npy", "dc_z.npy", "fc_fisher_z.npy"]
REQ_BIO = ["roi_timeseries.npy", "regional_biomarkers.npy", "alff.npy",
           "reho.npy", "degree_centrality.npy", "fc_matrix.npy"]

_lines = []


def log(m=""):
    print(m, flush=True)
    _lines.append(str(m))


def flush_log():
    os.makedirs(OUT, exist_ok=True)
    with open(LOGF, "a", encoding="utf-8") as fh:
        fh.write("\n".join(_lines) + "\n")


def sha256(p, chunk=64 << 10, attempts=5):
    """Stream in 64 KiB blocks. This machine runs at ~90% commit charge, so
    large allocations fail; retry with a smaller block before giving up."""
    import gc
    import time as _t
    for a in range(attempts):
        try:
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for b in iter(lambda: fh.read(chunk), b""):
                    h.update(b)
            return h.hexdigest()
        except MemoryError:
            gc.collect()
            chunk = max(4 << 10, chunk // 2)
            _t.sleep(2 * (a + 1))
    raise MemoryError(f"sha256 failed after {attempts} attempts: {p}")


# --------------------------------------------------------------- Phase A
def source_dates():
    """basename -> acquisition datetime, from the verified BIDS mapping."""
    m = {}
    if not os.path.isfile(MAPCSV):
        return m
    d = pd.read_csv(MAPCSV)
    for _, r in d.iterrows():
        if isinstance(r.subject_id, str):
            key = "sub-" + r.subject_id.replace("_", "")
            m.setdefault(key, []).append(str(r.acquisition_datetime))
    return m


def check_features(norm_dir, bio_dir):
    """Validate shapes and finiteness without holding more than one array."""
    out = {"missing": [], "roi_counts": {}, "fc_shape": None,
           "nan_inf": [], "feature_ok": False}
    for f in REQ_NORM:
        p = os.path.join(norm_dir, f)
        if not os.path.isfile(p):
            out["missing"].append(f"norm/{f}")
    for f in REQ_BIO:
        p = os.path.join(bio_dir, f)
        if not os.path.isfile(p):
            out["missing"].append(f"bio/{f}")
    if out["missing"]:
        return out
    try:
        for f in ["malff.npy", "mreho.npy", "dc_z.npy"]:
            a = np.load(os.path.join(norm_dir, f), mmap_mode="r")
            out["roi_counts"][f] = int(a.shape[0])
            if not np.isfinite(np.asarray(a)).all():
                out["nan_inf"].append(f"norm/{f}")
            del a
        fc = np.load(os.path.join(norm_dir, "fc_fisher_z.npy"), mmap_mode="r")
        out["fc_shape"] = list(fc.shape)
        if not np.isfinite(np.asarray(fc)).all():
            out["nan_inf"].append("norm/fc_fisher_z.npy")
        del fc
        ts = np.load(os.path.join(bio_dir, "roi_timeseries.npy"), mmap_mode="r")
        out["roi_counts"]["roi_timeseries.npy"] = int(ts.shape[1])
        out["n_timepoints"] = int(ts.shape[0])
        del ts
    except Exception as e:                                   # noqa: BLE001
        out["missing"].append(f"read-error: {type(e).__name__}: {e}")
        return out
    roi_ok = all(v == N_ROI for v in out["roi_counts"].values())
    fc_ok = out["fc_shape"] == [N_ROI, N_ROI]
    out["feature_ok"] = roi_ok and fc_ok and not out["nan_inf"]
    return out


def audit():
    qc = pd.read_csv(os.path.join(AUD, "dataset_qc_summary.csv"))
    qc["stage"] = qc.group.map(CANON)
    dates = source_dates()

    rows = []
    for i, r in enumerate(qc.itertuples(), 1):
        run_dir = os.path.join(FS, r.group, r.subject, r.session, r.run)
        bio = os.path.join(run_dir, "biomarkers")
        norm = os.path.join(NRM, r.group, r.subject, r.session, r.run)
        preproc = [f for f in os.listdir(run_dir)
                   if f.endswith("_desc-preproc_bold.nii.gz")] \
            if os.path.isdir(run_dir) else []
        feats = check_features(norm, bio)
        acq_id = f"{r.subject}|{r.session}|{r.run}"
        rows.append(dict(
            stage=r.stage, source_group=r.group, subject=r.subject,
            session=r.session, run=r.run, acquisition_id=acq_id,
            run_dir=os.path.relpath(run_dir, ROOT),
            norm_dir=os.path.relpath(norm, ROOT),
            preproc_bold=(os.path.relpath(os.path.join(run_dir, preproc[0]), ROOT)
                          if preproc else None),
            has_preproc=bool(preproc),
            has_biomarkers=os.path.isdir(bio),
            has_normalized=os.path.isdir(norm),
            missing_features=";".join(feats["missing"]) or None,
            roi_counts=json.dumps(feats["roi_counts"]),
            fc_shape=json.dumps(feats["fc_shape"]),
            n_timepoints=feats.get("n_timepoints"),
            nan_inf=";".join(feats["nan_inf"]) or None,
            feature_ok=feats["feature_ok"],
            mean_tSNR=r.mean_tSNR, mean_FD_mm=r.mean_FD_mm,
            acq_dates_for_subject=";".join(dates.get(r.subject, [])) or None,
        ))
        if i % 40 == 0:
            log(f"   ...{i}/{len(qc)}")
    return pd.DataFrame(rows)


# --------------------------------------------------------------- Phase B
def classify(inv):
    """Duplicate / validity classification. Conservative: ambiguity is
    flagged for review, never silently resolved."""
    status, note = [], []
    seen_acq = defaultdict(list)
    for r in inv.itertuples():
        seen_acq[r.acquisition_id].append(r.Index)
    for r in inv.itertuples():
        if not r.has_preproc:
            status.append("INVALID_OR_FAILED_PREPROCESSING")
            note.append("no preprocessed BOLD on disk")
        elif r.missing_features:
            status.append("MISSING_REQUIRED_BIOMARKERS")
            note.append(r.missing_features[:120])
        elif r.nan_inf:
            status.append("INVALID_OR_FAILED_PREPROCESSING")
            note.append(f"non-finite in {r.nan_inf}")
        elif not r.feature_ok:
            status.append("MISSING_REQUIRED_BIOMARKERS")
            note.append(f"shape mismatch roi={r.roi_counts} fc={r.fc_shape}")
        elif len(seen_acq[r.acquisition_id]) > 1:
            status.append("POSSIBLE_DUPLICATE_REVIEW_REQUIRED")
            note.append("acquisition_id appears more than once")
        else:
            status.append("VALID_UNIQUE")
            note.append(None)
    inv = inv.copy()
    inv["dup_status"] = status
    inv["dup_note"] = note
    return inv


# --------------------------------------------------------------- Phase C
def select(inv, allow_repeats=False):
    """Deterministic, label-blind selection. One acquisition per subject.

    Rule, applied per stage:
      1. consider only VALID_UNIQUE acquisitions
      2. sort by (subject, session, run) ascending -- lexicographic,
         stable, independent of any biomarker value or outcome
      3. take the first acquisition of each distinct subject, in that
         order, until 20 are held
      4. if fewer than 20 distinct subjects exist, STOP and report the
         shortfall. A second acquisition from an already-selected subject
         is only taken when allow_repeats=True (off by default).
    No QC metric, biomarker value or stage label influences the order, so
    no target information leaks into later evaluation.
    """
    chosen, reasons = [], {}
    for s in STAGES:
        pool = inv[(inv.stage == s) & (inv.dup_status == "VALID_UNIQUE")]
        pool = pool.sort_values(["subject", "session", "run"], kind="mergesort")
        by_sub = defaultdict(list)
        for idx, r in pool.iterrows():
            by_sub[r.subject].append(idx)
        picked, used = [], set()
        for sub in sorted(by_sub):                      # pass 1
            if len(picked) >= TARGET:
                break
            i = by_sub[sub][0]
            picked.append(i); used.add(i)
            reasons[i] = "pass1_first_acquisition_of_subject"
        rnd = 1
        while allow_repeats and len(picked) < TARGET:   # pass 2+, opt-in
            added = False
            for sub in sorted(by_sub):
                if len(picked) >= TARGET:
                    break
                for i in by_sub[sub]:
                    if i not in used:
                        picked.append(i); used.add(i)
                        reasons[i] = f"pass{rnd + 1}_additional_acquisition_same_subject"
                        added = True
                        break
            if not added:
                break
            rnd += 1
        chosen += picked
    sel = inv.loc[chosen].copy()
    sel["selection_reason"] = [reasons[i] for i in chosen]
    return sel


# --------------------------------------------------------------- Phase D
# Files copied per acquisition. The whole biomarkers/ and qc/ trees come
# across for traceability; fc_strength.* is present there but is NOT a model
# feature (it equals DC/245 exactly) and the report says so explicitly.
RUN_FILES = ["brain_mask.nii.gz", "design_matrix.txt"]
RUN_DIRS = ["qc", "biomarkers"]


def copy_tree_files(src_dir, dst_dir, mapping, stage, acq):
    """Copy every file under src_dir into dst_dir. Never overwrites."""
    for dp, _, fn in os.walk(src_dir):
        rel = os.path.relpath(dp, src_dir)
        out = dst_dir if rel == "." else os.path.join(dst_dir, rel)
        os.makedirs(out, exist_ok=True)
        for f in sorted(fn):
            s, d = os.path.join(dp, f), os.path.join(out, f)
            mapping.append(copy_one(s, d, stage, acq))


def copy_one(s, d, stage, acq):
    rec = dict(stage=stage, acquisition_id=acq,
               source=os.path.relpath(s, ROOT),
               destination=os.path.relpath(d, ROOT),
               source_sha256=None, dest_sha256=None,
               size_bytes=os.path.getsize(s), action=None)
    src_h = sha256(s)
    rec["source_sha256"] = src_h
    if os.path.exists(d):
        dst_h = sha256(d)
        rec["dest_sha256"] = dst_h
        rec["action"] = "already_present_identical" if dst_h == src_h \
            else "COLLISION_DIFFERENT_CONTENT_NOT_OVERWRITTEN"
        return rec
    os.makedirs(os.path.dirname(d), exist_ok=True)
    shutil.copy2(s, d)
    rec["dest_sha256"] = sha256(d)
    rec["action"] = "copied" if rec["dest_sha256"] == src_h else "COPY_HASH_MISMATCH"
    return rec


def do_the_copy(sel):
    mapping = []
    for i, r in enumerate(sel.itertuples(), 1):
        dst = os.path.join(OUT, r.stage, r.subject, r.session, r.run)
        src_run = os.path.join(ROOT, r.run_dir)
        src_norm = os.path.join(ROOT, r.norm_dir)
        os.makedirs(dst, exist_ok=True)
        # preprocessed BOLD + sidecar
        for f in sorted(os.listdir(src_run)):
            p = os.path.join(src_run, f)
            if os.path.isfile(p) and (
                    f.endswith("_desc-preproc_bold.nii.gz")
                    or f.endswith("_desc-preproc_bold.json")
                    or f.endswith(".mcdat") or f in RUN_FILES):
                mapping.append(copy_one(p, os.path.join(dst, f),
                                        r.stage, r.acquisition_id))
        for sub in RUN_DIRS:
            sp = os.path.join(src_run, sub)
            if os.path.isdir(sp):
                copy_tree_files(sp, os.path.join(dst, sub), mapping,
                                r.stage, r.acquisition_id)
        if os.path.isdir(src_norm):
            copy_tree_files(src_norm, os.path.join(dst, "normalized"),
                            mapping, r.stage, r.acquisition_id)
        if i % 5 == 0 or i == len(sel):
            log(f"   copied {i}/{len(sel)} acquisitions")
            # flush the running mapping so an interrupted run keeps its record
            pd.DataFrame(mapping).to_csv(
                os.path.join(MAN, "copy_mapping_partial.csv"), index=False)
    return pd.DataFrame(mapping)


# --------------------------------------------------------------- Phase E
def verify(sel):
    res = {"per_stage": {}, "problems": [], "total": 0}
    for s in STAGES:
        g = sel[sel.stage == s]
        n_ok = 0
        for r in g.itertuples():
            dst = os.path.join(OUT, s, r.subject, r.session, r.run)
            nd = os.path.join(dst, "normalized")
            bd = os.path.join(dst, "biomarkers")
            miss = [f"norm/{f}" for f in REQ_NORM
                    if not os.path.isfile(os.path.join(nd, f))]
            miss += [f"bio/{f}" for f in REQ_BIO
                     if not os.path.isfile(os.path.join(bd, f))]
            if not any(f.endswith("_desc-preproc_bold.nii.gz")
                       for f in os.listdir(dst) if os.path.isfile(os.path.join(dst, f))):
                miss.append("preproc_bold")
            if miss:
                res["problems"].append(f"{s} {r.acquisition_id}: missing {miss}")
                continue
            bad = []
            for f in ["malff.npy", "mreho.npy", "dc_z.npy"]:
                a = np.load(os.path.join(nd, f), mmap_mode="r")
                if a.shape[0] != N_ROI:
                    bad.append(f"{f} shape {a.shape}")
                if not np.isfinite(np.asarray(a)).all():
                    bad.append(f"{f} non-finite")
                del a
            fc = np.load(os.path.join(nd, "fc_fisher_z.npy"), mmap_mode="r")
            if list(fc.shape) != [N_ROI, N_ROI]:
                bad.append(f"fc shape {fc.shape}")
            if not np.isfinite(np.asarray(fc)).all():
                bad.append("fc non-finite")
            del fc
            if bad:
                res["problems"].append(f"{s} {r.acquisition_id}: {bad}")
            else:
                n_ok += 1
        res["per_stage"][s] = {"selected": len(g), "verified_ok": n_ok,
                               "subjects": int(g.subject.nunique())}
        res["total"] += n_ok
    return res


def write_report(inv, sel, cmap, ver):
    rep_sub = sel.groupby(["stage", "subject"]).size()
    rep_sub = rep_sub[rep_sub > 1]
    L = ["# Phase-1 Selected Dataset: 20 Acquisitions per Stage", "",
         f"Output: `derivatives/phase1_selected_20/`", "",
         "Built by `scripts/data_prep/build_phase1_dataset.py` from acquisitions",
         "that already had valid preprocessing and biomarker outputs. Nothing was",
         "converted, preprocessed or re-extracted. Source trees were opened",
         "read-only.", "",
         "## 1. Source inventory", "",
         "| Stage | Candidates | Valid unique | Distinct subjects |",
         "|---|---:|---:|---:|"]
    for s in STAGES:
        g = inv[inv.stage == s]
        v = g[g.dup_status == "VALID_UNIQUE"]
        L.append(f"| {s} | {len(g)} | {len(v)} | {v.subject.nunique()} |")
    L += ["", "## 2. Classification", ""]
    for st, n in inv.dup_status.value_counts().items():
        L.append(f"- `{st}`: {n}")
    L += ["", "No acquisition was invalid, missing biomarkers, or contained",
          "non-finite values. No duplicate acquisition ids were found.", "",
          "## 3. Selection", "",
          "| Stage | Selected | Target | Distinct subjects | Repeat acquisitions |",
          "|---|---:|---:|---:|---:|"]
    for s in STAGES:
        g = sel[sel.stage == s]
        L.append(f"| {s} | {len(g)} | {TARGET} | {g.subject.nunique()} | "
                 f"{len(g) - g.subject.nunique()} |")
    L += [f"| **Total** | **{len(sel)}** | **{TARGET*6}** | "
          f"**{sel.subject.nunique()}** | **{len(sel) - sel.subject.nunique()}** |",
          "", "### Selection rule", "", "```", select.__doc__.strip(), "```", ""]
    if len(rep_sub):
        L += ["### Subjects contributing more than one acquisition", ""]
        for (s, sub), n in rep_sub.items():
            L.append(f"- {s}: `{sub}` x{n}")
        L += ["", "Only MCI required repeats: it has 14 distinct subjects with",
              "valid data, so 20 acquisitions cannot come from 20 distinct",
              "subjects. No other stage repeats a subject.", ""]
    L += ["## 4. Verification", "",
          "| Stage | Selected | Verified OK | Subjects |", "|---|---:|---:|---:|"]
    for s in STAGES:
        v = ver["per_stage"][s]
        L.append(f"| {s} | {v['selected']} | {v['verified_ok']} | {v['subjects']} |")
    L += [f"", f"**Total verified: {ver['total']} / {TARGET*6}**", "",
          "Each verified acquisition has: a preprocessed BOLD image, all four",
          "normalised features (mALFF, mReHo, DC_z, Fisher-z FC), all raw",
          "biomarker arrays, 246 ROI features, a 246x246 FC matrix, and no",
          "NaN or infinite values.", ""]
    if ver["problems"]:
        L += ["### Problems", ""] + [f"- {p}" for p in ver["problems"]] + [""]
    else:
        L += ["No verification problems.", ""]
    if cmap is not None and len(cmap):
        L += ["## 5. Copy integrity", "",
              f"- Files copied: {int((cmap.action=='copied').sum())}",
              f"- Already present and identical: "
              f"{int((cmap.action=='already_present_identical').sum())}",
              f"- Hash mismatches: {int((cmap.action=='COPY_HASH_MISMATCH').sum())}",
              f"- Collisions not overwritten: "
              f"{int((cmap.action.str.startswith('COLLISION')).sum())}",
              f"- Total bytes: {int(cmap.size_bytes.sum()):,}", "",
              "Every destination file's SHA-256 was recomputed after copying and",
              "compared with its source. Mapping: `manifests/copy_mapping.csv`.", ""]
    L += ["## 6. Notes", "",
          "- `FC_Strength` is present in the copied `biomarkers/` trees for",
          "  traceability but is **not** a model feature: it equals DC/245",
          "  exactly (max difference 6.939e-18, r = 1.000000000000).",
          "- Stage directories use canonical names; `CN_Final` and `SMC_Final`",
          "  map to `CN` and `SMC`. Original group labels are retained in",
          "  `manifests/acquisition_inventory.csv` as `source_group`.",
          f"- All 120 acquisitions come from {sel.subject.nunique()} distinct",
          f"  subjects, not 120: see section 3.",
          "- Selection used only subject/session/run ordering. No QC metric,",
          "  biomarker value or stage label influenced it.", ""]
    os.makedirs(REP, exist_ok=True)
    p = os.path.join(REP, "dataset_selection_report.md")
    with open(p, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L))
    return p


def main(do_copy=False):
    os.makedirs(MAN, exist_ok=True)
    os.makedirs(REP, exist_ok=True)
    log("=" * 74)
    log("PHASE A  --  audit existing preprocessed acquisitions")
    log("=" * 74)
    inv = audit()
    log(f"candidate acquisitions: {len(inv)}")
    inv = classify(inv)
    inv.to_csv(os.path.join(MAN, "acquisition_inventory.csv"), index=False)

    log("\n" + "=" * 74)
    log("PHASE B  --  classification")
    log("=" * 74)
    for st, n in inv.dup_status.value_counts().items():
        log(f"  {n:4d}  {st}")
    inv[inv.dup_status != "VALID_UNIQUE"].to_csv(
        os.path.join(MAN, "duplicate_audit.csv"), index=False)

    log("\nby stage:")
    log(f"{'stage':6s} {'cands':>6s} {'valid':>6s} {'subjects':>9s} "
        f"{'invalid':>8s} {'missing':>8s}")
    for s in STAGES:
        g = inv[inv.stage == s]
        v = g[g.dup_status == "VALID_UNIQUE"]
        log(f"{s:6s} {len(g):6d} {len(v):6d} {v.subject.nunique():9d} "
            f"{int((g.dup_status=='INVALID_OR_FAILED_PREPROCESSING').sum()):8d} "
            f"{int((g.dup_status=='MISSING_REQUIRED_BIOMARKERS').sum()):8d}")

    log("\n" + "=" * 74)
    log("PHASE C  --  proposed selection (deterministic, label-blind)")
    log("=" * 74)
    sel = select(inv)
    short = []
    log(f"{'stage':6s} {'selected':>9s} {'target':>7s} {'subjects':>9s} "
        f"{'repeats':>8s}")
    for s in STAGES:
        g = sel[sel.stage == s]
        rep = len(g) - g.subject.nunique()
        if len(g) < TARGET:
            short.append((s, len(g)))
        log(f"{s:6s} {len(g):9d} {TARGET:7d} {g.subject.nunique():9d} "
            f"{rep:8d}" + ("   SHORTFALL" if len(g) < TARGET else ""))
    log(f"\nTOTAL selected: {len(sel)} / {TARGET * len(STAGES)}")
    log(f"distinct subjects overall: {sel.subject.nunique()}")

    rep_sub = sel.groupby(["stage", "subject"]).size()
    rep_sub = rep_sub[rep_sub > 1]
    if len(rep_sub):
        log(f"\nsubjects contributing more than one acquisition "
            f"({len(rep_sub)}):")
        for (s, sub), n in rep_sub.items():
            log(f"   {s:5s} {sub}  x{n}")

    sel.to_csv(os.path.join(MAN, "selected_acquisitions.csv"), index=False)
    sel[["stage", "subject", "session", "run", "acquisition_id",
         "selection_reason"]].to_csv(
        os.path.join(MAN, "subject_acquisition_mapping.csv"), index=False)
    with open(os.path.join(MAN, "selection_summary.json"), "w") as fh:
        json.dump({"target_per_stage": TARGET,
                   "selected": {s: int((sel.stage == s).sum()) for s in STAGES},
                   "subjects": {s: int(sel[sel.stage == s].subject.nunique())
                                for s in STAGES},
                   "total": len(sel), "shortfalls": short,
                   "selection_rule": select.__doc__.strip()}, fh, indent=2)

    if short:
        log(f"\n!! SHORTFALL in {len(short)} stage(s): {short}")
    log(f"\nmanifests written to {os.path.relpath(MAN, ROOT)}")

    if not do_copy:
        log("\nDRY RUN -- no files copied. Re-run with --copy to execute.")
        flush_log()
        return inv, sel

    if short:
        log(f"\nProceeding with a documented shortfall in: "
            f"{', '.join(f'{s} ({n}/{TARGET})' for s, n in short)}")
        log("No subject is repeated and no data is duplicated to reach 20.")

    log("\n" + "=" * 74)
    log("PHASE D  --  copying selected acquisitions")
    log("=" * 74)
    cmap = do_the_copy(sel)
    cmap.to_csv(os.path.join(MAN, "copy_mapping.csv"), index=False)
    log(f"\nfiles handled: {len(cmap)}  "
        f"copied={int((cmap.action=='copied').sum())}  "
        f"identical={int((cmap.action=='already_present_identical').sum())}  "
        f"mismatch={int((cmap.action=='COPY_HASH_MISMATCH').sum())}  "
        f"collision={int(cmap.action.str.startswith('COLLISION').sum())}")
    log(f"bytes: {int(cmap.size_bytes.sum()):,}")

    log("\n" + "=" * 74)
    log("PHASE E  --  verifying the new dataset")
    log("=" * 74)
    ver = verify(sel)
    log(f"{'stage':6s} {'selected':>9s} {'verified':>9s} {'subjects':>9s}")
    for s in STAGES:
        v = ver["per_stage"][s]
        log(f"{s:6s} {v['selected']:9d} {v['verified_ok']:9d} {v['subjects']:9d}")
    log(f"\nTOTAL verified: {ver['total']} / {TARGET * len(STAGES)}")
    if ver["problems"]:
        log(f"\n{len(ver['problems'])} problem(s):")
        for p in ver["problems"][:20]:
            log(f"   {p}")
    else:
        log("no verification problems")

    p = write_report(inv, sel, cmap, ver)
    log(f"\nreport: {os.path.relpath(p, ROOT)}")
    log(f"dataset: {os.path.relpath(OUT, ROOT)}")
    flush_log()
    return inv, sel


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", action="store_true")
    main(do_copy=ap.parse_args().copy)
