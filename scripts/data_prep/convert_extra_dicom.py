"""
STEP 2/3 -- sequential DICOM->NIfTI conversion + validation for the new
CN and LMCI ADNI archives.

Design constraints (from the task brief):
  * ZIP archives are never modified; DICOMs are extracted one series at a
    time to a staging dir and deleted after conversion (disk is tight).
  * One series at a time, single worker.
  * Resumable: completed series are recorded in a JSONL state file and
    re-validated on disk before being skipped.
  * Never overwrites an existing output file.
  * Full length: no truncation of the 200-volume Extended scans.
  * Nothing outside NIfTI_extra/ and derivatives/EXTRA_DATA_INVENTORY/
    is written.

Usage:
    python scripts/data_prep/convert_extra_dicom.py            # convert
    python scripts/data_prep/convert_extra_dicom.py --report   # report only
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from collections import defaultdict

import nibabel as nib
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW = os.path.join(ROOT, "raw_data_extra")
INV = os.path.join(ROOT, "derivatives", "EXTRA_DATA_INVENTORY")
DEDUP = os.path.join(INV, "extra_series_dedup.csv")
OUTROOT = os.path.join(ROOT, "NIfTI_extra")
STAGE_DIR = os.path.join(ROOT, "_extract_staging")
STATE = os.path.join(INV, "conversion_state.jsonl")
DCM2NIIX = os.path.join(ROOT, "tools", "dcm2niix_win", "dcm2niix.exe")

GROUP_DIR = {"CN": "CN_Final", "SMC": "SMC_Final", "EMCI": "EMCI",
             "MCI": "MCI", "LMCI": "LMCI", "AD": "AD"}
ARCHIVE = {"CN": "CN_extra.zip", "LMCI": "LMCI_Extra_Final.zip",
           "MCI": "MCI_ersfMRI.zip"}
MIN_DICOM = 1000              # below this the series is a broken download
SLICES_GUESS = 48             # ADNI rs-fMRI slice count, for the volume check


def log(m):
    print(m, flush=True)


# ------------------------------------------------------------------ state
def load_state():
    done = {}
    if os.path.isfile(STATE):
        with open(STATE, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                done[r["image_uid"]] = r
    return done


def append_state(rec):
    with open(STATE, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec) + "\n")


# -------------------------------------------------------------- validation
def validate(nii_path, json_path, expected_vols):
    """Return (ok, dict_of_findings). Never raises."""
    v = {"nifti_readable": False, "is_4d": False, "n_volumes": None,
         "dims": None, "voxel_mm": None, "affine_valid": False,
         "data_finite": False, "data_nonconstant": False,
         "json_valid": False, "TR_seconds": None,
         "volume_count_matches_dicom": None, "problems": []}
    try:
        img = nib.load(nii_path)
        v["nifti_readable"] = True
        v["dims"] = list(img.shape)
        v["is_4d"] = img.ndim == 4
        if not v["is_4d"]:
            v["problems"].append(f"not 4D (ndim={img.ndim})")
        else:
            v["n_volumes"] = int(img.shape[3])
            if img.shape[3] < 2:
                v["problems"].append(f"time dimension is {img.shape[3]}")
        zooms = [float(z) for z in img.header.get_zooms()[:3]]
        v["voxel_mm"] = [round(z, 4) for z in zooms]
        if not all(np.isfinite(zooms)) or min(zooms) <= 0:
            v["problems"].append(f"implausible voxel size {zooms}")
        elif min(zooms) < 0.5 or max(zooms) > 8.0:
            v["problems"].append(f"voxel size outside 0.5-8mm: {zooms}")
        aff = img.affine
        if np.all(np.isfinite(aff)) and abs(np.linalg.det(aff[:3, :3])) > 1e-9:
            v["affine_valid"] = True
        else:
            v["problems"].append("affine singular or non-finite")
        # Stream volume-by-volume. Loading a 200-volume 4D array at once
        # needs ~300 MiB, which this machine does not reliably have free.
        if int(np.prod(img.shape)) == 0:
            v["problems"].append("empty data array")
        else:
            finite = True
            lo, hi = np.inf, -np.inf
            n_t = img.shape[3] if img.ndim == 4 else 1
            for t in range(n_t):
                vol = np.asanyarray(
                    img.dataobj[..., t] if img.ndim == 4 else img.dataobj,
                    dtype=np.float32)
                if not np.isfinite(vol).all():
                    finite = False
                    vmin = np.nanmin(vol[np.isfinite(vol)]) if np.isfinite(vol).any() else np.nan
                    vmax = np.nanmax(vol[np.isfinite(vol)]) if np.isfinite(vol).any() else np.nan
                else:
                    vmin, vmax = float(vol.min()), float(vol.max())
                if np.isfinite(vmin):
                    lo = min(lo, vmin)
                if np.isfinite(vmax):
                    hi = max(hi, vmax)
                del vol
            v["data_finite"] = bool(finite)
            if not finite:
                v["problems"].append("non-finite values present")
            v["data_min"] = None if not np.isfinite(lo) else round(float(lo), 4)
            v["data_max"] = None if not np.isfinite(hi) else round(float(hi), 4)
            v["data_nonconstant"] = bool(np.isfinite(lo) and np.isfinite(hi)
                                         and hi > lo)
            if not v["data_nonconstant"]:
                v["problems"].append("data is constant")
    except Exception as e:                                   # noqa: BLE001
        v["problems"].append(f"nibabel: {type(e).__name__}: {e}")

    try:
        with open(json_path, encoding="utf-8") as fh:
            j = json.load(fh)
        v["json_valid"] = True
        tr = j.get("RepetitionTime")
        v["TR_seconds"] = tr
        if tr is None:
            v["problems"].append("RepetitionTime absent from sidecar")
        v["SeriesDescription"] = j.get("SeriesDescription")
        v["Manufacturer"] = j.get("Manufacturer")
        v["ConversionSoftwareVersion"] = j.get("ConversionSoftwareVersion")
        v["slice_timing_present"] = "SliceTiming" in j
    except Exception as e:                                   # noqa: BLE001
        v["problems"].append(f"json: {type(e).__name__}: {e}")

    if expected_vols and v["n_volumes"]:
        v["volume_count_matches_dicom"] = (v["n_volumes"] == expected_vols)
        if not v["volume_count_matches_dicom"]:
            v["problems"].append(
                f"volume count {v['n_volumes']} != {expected_vols} "
                f"expected from DICOM count")
    ok = (v["nifti_readable"] and v["is_4d"] and v["affine_valid"]
          and v["data_finite"] and v["data_nonconstant"] and v["json_valid"]
          and not v["problems"])
    return ok, v


# ------------------------------------------------------------- conversion
def convert_one(zf, members, rec, out_dir):
    """Extract -> dcm2niix -> move -> validate. Returns the result record."""
    uid = rec["image_uid"]
    if os.path.isdir(STAGE_DIR):
        shutil.rmtree(STAGE_DIR, ignore_errors=True)
    os.makedirs(STAGE_DIR, exist_ok=True)
    tmp_out = os.path.join(STAGE_DIR, "_out")
    os.makedirs(tmp_out, exist_ok=True)

    t0 = time.time()
    for m in members:
        zf.extract(m, STAGE_DIR)
    src = os.path.join(STAGE_DIR, os.path.dirname(members[0]))
    n_extracted = sum(len(f) for _, _, f in os.walk(src))

    p = subprocess.run(
        [DCM2NIIX, "-z", "y", "-b", "y", "-f", "%n_%d_%t_%s",
         "-o", tmp_out, src],
        capture_output=True, text=True, timeout=1800)
    stdout = (p.stdout or "") + (p.stderr or "")
    warn = [l.strip() for l in stdout.splitlines()
            if "warning" in l.lower() or "error" in l.lower()]

    produced = sorted(f for f in os.listdir(tmp_out) if f.endswith(".nii.gz"))
    rec = dict(rec)
    rec.update(n_dicom_extracted=n_extracted, dcm2niix_returncode=p.returncode,
               dcm2niix_warnings=warn[:12], n_outputs=len(produced),
               seconds=round(time.time() - t0, 1))

    if not produced:
        rec.update(status="FAILED", reason="dcm2niix produced no NIfTI",
                   outputs=[])
        shutil.rmtree(STAGE_DIR, ignore_errors=True)
        return rec

    os.makedirs(out_dir, exist_ok=True)
    outputs, collisions = [], []
    for f in produced:
        dst = os.path.join(out_dir, f)
        if os.path.exists(dst):
            collisions.append(dst)       # never overwrite
            continue
        shutil.move(os.path.join(tmp_out, f), dst)
        js = os.path.join(tmp_out, f[:-7] + ".json")
        if os.path.isfile(js):
            shutil.move(js, os.path.join(out_dir, f[:-7] + ".json"))
        outputs.append(dst)

    if collisions and not outputs:
        rec.update(status="SKIPPED_EXISTS", reason="output already present",
                   outputs=[], collisions=collisions)
        shutil.rmtree(STAGE_DIR, ignore_errors=True)
        return rec

    exp = rec.get("expected_volumes")
    vals, all_ok = [], True
    for o in outputs:
        ok, v = validate(o, o[:-7] + ".json", exp if len(outputs) == 1 else None)
        v["file"] = os.path.relpath(o, ROOT)
        vals.append(v)
        all_ok = all_ok and ok

    rec.update(outputs=[os.path.relpath(o, ROOT) for o in outputs],
               collisions=collisions, validation=vals)
    if rec.get("suspicious"):
        rec["status"] = "SUSPICIOUS_VALIDATED" if all_ok else "SUSPICIOUS_FAILED"
    else:
        rec["status"] = "OK" if all_ok else "VALIDATION_FAILED"
    shutil.rmtree(STAGE_DIR, ignore_errors=True)
    return rec


# -------------------------------------------------------------------- main
def revalidate():
    """Re-check every converted file on disk and append corrected records.

    Conversion is the expensive step; validation is cheap. A file that was
    marked VALIDATION_FAILED only because the validator ran out of memory
    is re-judged here without being reconverted.
    """
    done = load_state()
    fixed = checked = still_bad = 0
    for uid, r in sorted(done.items()):
        outs = r.get("outputs") or []
        if not outs:
            continue
        present = [o for o in outs if os.path.isfile(os.path.join(ROOT, o))]
        if not present:
            continue
        checked += 1
        exp = r.get("expected_volumes")
        vals, all_ok = [], True
        for o in present:
            full = os.path.join(ROOT, o)
            ok, v = validate(full, full[:-7] + ".json",
                             exp if len(present) == 1 else None)
            v["file"] = o
            vals.append(v)
            all_ok = all_ok and ok
        if r.get("suspicious"):
            new = "SUSPICIOUS_VALIDATED" if all_ok else "SUSPICIOUS_FAILED"
        else:
            new = "OK" if all_ok else "VALIDATION_FAILED"
        if new != r.get("status"):
            rec = dict(r, status=new, validation=vals,
                       revalidated=True,
                       previous_status=r.get("status"))
            append_state(rec)
            log(f"  {uid} {r['adni_subject']}: {r.get('status')} -> {new}")
            fixed += 1
        if new not in ("OK", "SUSPICIOUS_VALIDATED"):
            still_bad += 1
            for v in vals:
                log(f"      {v['file']}: {v.get('problems')}")
    log(f"\nrevalidated {checked} series; {fixed} status changes; "
        f"{still_bad} still not passing")


def main(report_only=False, stages=("CN", "LMCI"), dedup=None, state=None):
    global STATE
    os.makedirs(INV, exist_ok=True)
    if not os.path.isfile(DCM2NIIX):
        sys.exit(f"dcm2niix not found: {DCM2NIIX}")
    if state:
        STATE = state
    d = pd.read_csv(dedup or DEDUP)
    cand = d[(~d.is_duplicate) & (d.n_dicom >= MIN_DICOM)].copy()
    cand["expected_volumes"] = (cand.n_dicom / SLICES_GUESS).apply(
        lambda x: int(x) if float(x).is_integer() else None)
    cand["suspicious"] = cand.expected_volumes.isna()
    log(f"candidates: {len(cand)}  "
        f"(CN {int((cand.stage=='CN').sum())}, "
        f"LMCI {int((cand.stage=='LMCI').sum())})  "
        f"suspicious: {int(cand.suspicious.sum())}")
    log(f"excluded as broken (<{MIN_DICOM} DICOMs): "
        f"{int(((~d.is_duplicate) & (d.n_dicom < MIN_DICOM)).sum())}")
    log(f"duplicates skipped: {int(d.is_duplicate.sum())}")

    done = load_state()
    log(f"already recorded in state: {len(done)}")
    if report_only:
        return done

    for stage in stages:
        sub = cand[cand.stage == stage]
        if sub.empty:
            continue
        _a = ARCHIVE[stage]
        zpath = _a if os.path.isabs(_a) else os.path.join(RAW, _a)
        log(f"\n{'='*72}\n{stage}: {len(sub)} series from {ARCHIVE[stage]}\n{'='*72}")
        with zipfile.ZipFile(zpath) as zf:
            by_uid = defaultdict(list)
            for n in zf.namelist():
                if n.endswith(".dcm"):
                    by_uid[n.rstrip("/").split("/")[-2]].append(n)
            for i, (_, r) in enumerate(sub.iterrows(), 1):
                uid = r.image_uid
                prev = done.get(uid)
                if prev and prev.get("status") in ("OK", "SUSPICIOUS_VALIDATED"):
                    if all(os.path.isfile(os.path.join(ROOT, o))
                           for o in prev.get("outputs", [])):
                        log(f"[{i}/{len(sub)}] {uid} {r.adni_subject}  SKIP (done)")
                        continue
                members = by_uid.get(uid, [])
                if not members:
                    log(f"[{i}/{len(sub)}] {uid}  NO MEMBERS -- skipped")
                    continue
                out_dir = os.path.join(OUTROOT, GROUP_DIR[stage], r.adni_subject)
                base = dict(stage=stage, archive=ARCHIVE[stage],
                            adni_subject=r.adni_subject, image_uid=uid,
                            description=r.description,
                            acq_datetime=r.acq_datetime,
                            n_dicom_listed=int(r.n_dicom),
                            expected_volumes=(None if pd.isna(r.expected_volumes)
                                              else int(r.expected_volumes)),
                            suspicious=bool(r.suspicious))
                log(f"[{i}/{len(sub)}] {uid} {r.adni_subject} "
                    f"{r.description[:34]} n={int(r.n_dicom)}"
                    f"{'  [SUSPICIOUS]' if r.suspicious else ''}")
                try:
                    rec = convert_one(zf, members, base, out_dir)
                except Exception as e:                       # noqa: BLE001
                    rec = dict(base, status="ERROR",
                               reason=f"{type(e).__name__}: {e}", outputs=[])
                    shutil.rmtree(STAGE_DIR, ignore_errors=True)
                nv = (rec.get("validation") or [{}])[0].get("n_volumes")
                log(f"      -> {rec['status']}  vols={nv}  "
                    f"{rec.get('seconds','?')}s"
                    + (f"  {rec.get('reason','')}" if rec.get("reason") else ""))
                append_state(rec)
                done[uid] = rec

    shutil.rmtree(STAGE_DIR, ignore_errors=True)
    log("\nCONVERSION PASS COMPLETE")
    return done


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--revalidate", action="store_true",
                    help="re-check converted files on disk; no conversion")
    ap.add_argument("--stages", default="CN,LMCI",
                    help="comma-separated stages to convert, e.g. MCI")
    ap.add_argument("--dedup", default=None,
                    help="dedup CSV to read (default: the CN/LMCI one)")
    ap.add_argument("--state", default=None,
                    help="JSONL state file (default: conversion_state.jsonl)")
    ap.add_argument("--archive", default=None,
                    help="override the archive path for the named stage")
    a = ap.parse_args()
    if a.archive:
        for _s in (x.strip() for x in a.stages.split(",") if x.strip()):
            ARCHIVE[_s] = a.archive
    if a.state:
        STATE = a.state
    if a.revalidate:
        revalidate()
    else:
        main(report_only=a.report,
             stages=tuple(x.strip() for x in a.stages.split(",") if x.strip()),
             dedup=a.dedup, state=a.state)
