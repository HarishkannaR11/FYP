"""
Audit, deduplicate and select 20 distinct acquisitions per diagnostic stage.

Dry run by default. `--copy` performs the copy after you have reviewed the
plan. Never modifies BIDS/, NIfTI_/, NIfTI_extra/ or any existing derivative.

Memory-safe: headers and hashes only, no image data is loaded (this machine
runs close to its RAM limit while conversion is active).

Usage:
    python scripts/data_prep/select_phase1_dataset.py            # dry run
    python scripts/data_prep/select_phase1_dataset.py --copy     # execute
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import defaultdict

import nibabel as nib
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NIFTI = os.path.join(ROOT, "NIfTI_", "NIfTI")
EXTRA = os.path.join(ROOT, "NIfTI_extra")
AUD = os.path.join(ROOT, "derivatives", "fsfast", "FINAL_6GROUP_AUDIT")
MAPCSV = os.path.join(ROOT, "audit", "bids_source_mapping.csv")
OUT = os.path.join(ROOT, "derivatives", "phase1_selected_20")
REPORTS = os.path.join(OUT, "_audit")

TARGET = 20
# group directory -> canonical stage. The _Final suffix is a directory
# naming artefact, not a separate diagnostic class.
CANON = {"CN_Final": "CN", "SMC_Final": "SMC", "EMCI": "EMCI",
         "MCI": "MCI", "LMCI": "LMCI", "AD": "AD"}
STAGES = ["CN", "SMC", "EMCI", "MCI", "LMCI", "AD"]


def log(m):
    print(m, flush=True)


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def header_facts(nii):
    """Header-only. Never touches the voxel data."""
    f = {"n_volumes": None, "dims": None, "voxel_mm": None,
         "header_ok": False, "problem": None}
    try:
        img = nib.load(nii)
        f["dims"] = list(int(x) for x in img.shape)
        f["n_volumes"] = int(img.shape[3]) if img.ndim == 4 else None
        f["voxel_mm"] = [round(float(z), 4) for z in img.header.get_zooms()[:3]]
        f["header_ok"] = True
        if img.ndim != 4:
            f["problem"] = f"not 4D (ndim={img.ndim})"
        elif img.shape[3] < 2:
            f["problem"] = f"time dimension {img.shape[3]}"
    except Exception as e:                                   # noqa: BLE001
        f["problem"] = f"{type(e).__name__}: {e}"
    return f


def sidecar_facts(js):
    f = {"TR_seconds": None, "SeriesDescription": None,
         "acq_datetime_json": None, "json_ok": False}
    if not (js and os.path.isfile(js)):
        return f
    try:
        with open(js, encoding="utf-8") as fh:
            d = json.load(fh)
        f["json_ok"] = True
        f["TR_seconds"] = d.get("RepetitionTime")
        f["SeriesDescription"] = d.get("SeriesDescription")
        f["acq_datetime_json"] = d.get("AcquisitionDateTime") or d.get("AcquisitionTime")
    except Exception:                                        # noqa: BLE001
        pass
    return f


def pair_for(nii):
    base = nii[:-7] if nii.endswith(".nii.gz") else nii[:-4]
    js = base + ".json"
    return js if os.path.isfile(js) else None


# ------------------------------------------------------------- eligibility
def eligibility():
    """What the validated pipeline already knows about each acquisition."""
    ok, fail, excl = set(), {}, {}
    q = os.path.join(AUD, "dataset_qc_summary.csv")
    if os.path.isfile(q):
        d = pd.read_csv(q)
        for _, r in d.iterrows():
            ok.add((CANON.get(r.group, r.group), r.subject))
    f = os.path.join(AUD, "dataset_failures.csv")
    if os.path.isfile(f):
        d = pd.read_csv(f)
        for _, r in d.iterrows():
            sub = str(r.acquisition).split("/")[0]
            fail[(CANON.get(r.group, r.group), sub)] = r.reason
    e = os.path.join(AUD, "dataset_exclusions.csv")
    if os.path.isfile(e):
        d = pd.read_csv(e)
        for _, r in d.iterrows():
            excl[(CANON.get(r.group, r.group), r.subject)] = r.reason
    return ok, fail, excl


# -------------------------------------------------------------- inventory
def scan_tree(root, origin, group_from_path):
    rows = []
    if not os.path.isdir(root):
        return rows
    for dp, _, fn in os.walk(root):
        for name in sorted(fn):
            if not (name.endswith(".nii") or name.endswith(".nii.gz")):
                continue
            nii = os.path.join(dp, name)
            grp = group_from_path(nii)
            if grp is None:
                continue
            rows.append(dict(origin=origin, group_dir=grp,
                             stage=CANON.get(grp, grp),
                             nifti_path=os.path.relpath(nii, ROOT),
                             json_path=(os.path.relpath(pair_for(nii), ROOT)
                                        if pair_for(nii) else None),
                             filename=name,
                             size_bytes=os.path.getsize(nii)))
    return rows


def enrich(rows, mapping):
    out = []
    for i, r in enumerate(rows, 1):
        nii = os.path.join(ROOT, r["nifti_path"])
        js = os.path.join(ROOT, r["json_path"]) if r["json_path"] else None
        r = dict(r)
        r.update(header_facts(nii))
        r.update(sidecar_facts(js))
        r["file_sha256"] = sha256(nii)
        # subject id from the containing directory (both trees use it)
        parts = r["nifti_path"].replace("\\", "/").split("/")
        r["subject_dir"] = parts[-2]
        key = r["filename"]
        m = mapping.get(key)
        r["acq_datetime"] = m["acquisition_datetime"] if m else r.get("acq_datetime_json")
        r["series_number"] = m["series_number"] if m else None
        r["adni_subject"] = m["subject_id"] if m else r["subject_dir"]
        if i % 25 == 0:
            log(f"   ...{i}/{len(rows)}")
        out.append(r)
    return out


def main(do_copy=False):
    os.makedirs(REPORTS, exist_ok=True)

    mapping = {}
    if os.path.isfile(MAPCSV):
        d = pd.read_csv(MAPCSV)
        for _, r in d.iterrows():
            rec = r.to_dict()          # dicts, not Series: avoids ambiguous truth
            for p in (r.nii_path, r.json_path):
                if isinstance(p, str):
                    mapping[os.path.basename(p)] = rec
            if isinstance(r.nii_path, str):
                b = os.path.basename(r.nii_path)
                mapping[b] = rec
                mapping[b.replace(".nii.gz", ".nii")] = rec
                mapping[b.replace(".nii", ".nii.gz")] = rec
    log(f"source mapping entries: {len(mapping)}")

    def grp_existing(p):
        parts = os.path.relpath(p, NIFTI).replace("\\", "/").split("/")
        return parts[0] if parts and parts[0] in CANON else None

    def grp_extra(p):
        parts = os.path.relpath(p, EXTRA).replace("\\", "/").split("/")
        return parts[0] if parts and parts[0] in CANON else None

    log("\nscanning existing NIfTI_ tree...")
    ex = scan_tree(NIFTI, "existing", grp_existing)
    log(f"  {len(ex)} files")
    log("scanning NIfTI_extra tree...")
    nw = scan_tree(EXTRA, "new", grp_extra)
    log(f"  {len(nw)} files")

    log("\nreading headers + hashing (no image data loaded)...")
    rows = enrich(ex + nw, mapping)
    inv = pd.DataFrame(rows)

    ok_set, fail_map, excl_map = eligibility()
    def elig(r):
        k = (r["stage"], r["adni_subject"])
        kb = (r["stage"], "sub-" + str(r["adni_subject"]).replace("_", ""))
        if kb in excl_map or k in excl_map:
            return "EXCLUDED", excl_map.get(kb, excl_map.get(k))
        if kb in fail_map or k in fail_map:
            return "PREPROC_FAILED", fail_map.get(kb, fail_map.get(k))
        if kb in ok_set or k in ok_set:
            return "PREPROC_OK", None
        return "NOT_YET_PROCESSED", None
    inv[["pipeline_status", "pipeline_note"]] = inv.apply(
        lambda r: pd.Series(elig(r)), axis=1)

    f = os.path.join(REPORTS, "acquisition_inventory.csv")
    inv.to_csv(f, index=False)
    log(f"\nwrote {f}  ({len(inv)} rows)")

    log("\n" + "=" * 74)
    log("INVENTORY BY STAGE")
    log("=" * 74)
    log(f"{'stage':6s} {'existing':>9s} {'new':>5s} {'subjects':>9s} "
        f"{'preproc_ok':>11s} {'failed':>7s} {'excluded':>9s}")
    for s in STAGES:
        g = inv[inv.stage == s]
        log(f"{s:6s} {int((g.origin=='existing').sum()):9d} "
            f"{int((g.origin=='new').sum()):5d} "
            f"{g.adni_subject.nunique():9d} "
            f"{int((g.pipeline_status=='PREPROC_OK').sum()):11d} "
            f"{int((g.pipeline_status=='PREPROC_FAILED').sum()):7d} "
            f"{int((g.pipeline_status=='EXCLUDED').sum()):9d}")
    return inv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--copy", action="store_true")
    main(do_copy=ap.parse_args().copy)
