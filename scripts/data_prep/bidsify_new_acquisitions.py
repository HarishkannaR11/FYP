"""
BIDS-ify newly converted acquisitions so the production pipeline can see them.

The production preprocessing reads ONLY from BIDS/, so a converted file in
NIfTI_extra/ is invisible to it until it is placed here under the project's
naming convention:

    BIDS/<sub>/<ses>/func/<sub>_<ses>_task-rest_<run>_bold.nii.gz  (+ .json)

Copies, never moves. Never overwrites. Appends to participants.tsv only for
subjects not already listed. Writes an explicit source->destination mapping.

    python scripts/data_prep/bidsify_new_acquisitions.py --dry-run
    python scripts/data_prep/bidsify_new_acquisitions.py --apply
"""
import argparse
import csv
import json
import os
import shutil

import nibabel as nib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXTRA = os.path.join(ROOT, "NIfTI_extra")
BIDS = os.path.join(ROOT, "BIDS")
PARTS = os.path.join(BIDS, "participants.tsv")
MAPOUT = os.path.join(ROOT, "audit", "bids_additions_mapping.csv")

# Acquisitions to BIDS-ify: (stage_dir, adni_subject, session, run)
# Both are new subjects with no prior session, hence ses-01/run-01.
TARGETS = [
    ("MCI", "100_S_0296", "ses-01", "run-01"),
    ("MCI", "007_S_2394", "ses-01", "run-01"),
]


def log(m=""):
    print(m, flush=True)


def bids_id(adni):
    return "sub-" + adni.replace("_", "")


def find_source(stage, adni):
    d = os.path.join(EXTRA, stage, adni)
    if not os.path.isdir(d):
        return None, None
    nii = sorted(f for f in os.listdir(d) if f.endswith(".nii.gz"))
    if not nii:
        return None, None
    n = os.path.join(d, nii[0])
    j = n[:-7] + ".json"
    return n, (j if os.path.isfile(j) else None)


def validate(nii, js):
    v = {"ok": False, "problems": []}
    try:
        img = nib.load(nii)
        v["dims"] = list(int(x) for x in img.shape)
        v["ndim"] = img.ndim
        v["n_volumes"] = int(img.shape[3]) if img.ndim == 4 else None
        v["voxel_mm"] = [round(float(z), 4) for z in img.header.get_zooms()[:3]]
        v["tr_header"] = round(float(img.header.get_zooms()[3]), 6) \
            if img.ndim == 4 else None
        if img.ndim != 4:
            v["problems"].append(f"not 4D (ndim={img.ndim})")
        elif img.shape[3] < 6:
            v["problems"].append(
                f"only {img.shape[3]} volumes; pipeline discards the first 5")
        aff = img.affine
        if not (np.all(np.isfinite(aff)) and abs(np.linalg.det(aff[:3, :3])) > 1e-9):
            v["problems"].append("affine singular or non-finite")
        # one volume only: enough to confirm readability without the full 4D
        _ = np.asanyarray(img.dataobj[..., 0], dtype=np.float32)
    except Exception as e:                                   # noqa: BLE001
        v["problems"].append(f"nibabel: {type(e).__name__}: {e}")
        return v
    if not js:
        v["problems"].append("JSON sidecar missing")
        return v
    try:
        with open(js, encoding="utf-8") as fh:
            d = json.load(fh)
        v["tr_json"] = d.get("RepetitionTime")
        v["series"] = d.get("SeriesDescription")
        v["manufacturer"] = d.get("Manufacturer")
        v["slice_timing_present"] = "SliceTiming" in d
        if v["tr_json"] is None:
            v["problems"].append("RepetitionTime absent from sidecar")
        else:
            nyq = 1.0 / (2.0 * v["tr_json"])
            v["nyquist_hz"] = round(nyq, 5)
            v["bandpass_viable"] = bool(nyq > 0.10)
            if not v["bandpass_viable"]:
                v["problems"].append(
                    f"TR {v['tr_json']}s -> Nyquist {nyq:.4f} Hz below the "
                    f"0.10 Hz cutoff; pipeline will reject")
    except Exception as e:                                   # noqa: BLE001
        v["problems"].append(f"json: {type(e).__name__}: {e}")
        return v
    v["ok"] = not v["problems"]
    return v


def main(apply_changes):
    rows = []
    log("=" * 74)
    log("BIDS-IFICATION OF NEW ACQUISITIONS" + ("" if apply_changes else "  (DRY RUN)"))
    log("=" * 74)

    for stage, adni, ses, run in TARGETS:
        sub = bids_id(adni)
        nii, js = find_source(stage, adni)
        log(f"\n{adni}  ->  {sub}/{ses}/{run}")
        if not nii:
            log("   SOURCE NOT FOUND -- skipped")
            continue
        log(f"   source : {os.path.relpath(nii, ROOT)}")
        v = validate(nii, js)
        log(f"   dims {v.get('dims')}  vols {v.get('n_volumes')}  "
            f"voxel {v.get('voxel_mm')}")
        log(f"   TR(json) {v.get('tr_json')}  Nyquist {v.get('nyquist_hz')} Hz  "
            f"band-pass viable: {v.get('bandpass_viable')}")
        log(f"   manufacturer {v.get('manufacturer')}  "
            f"slice timing present: {v.get('slice_timing_present')}")
        if v["problems"]:
            log(f"   PROBLEMS: {v['problems']}")
            log("   -> NOT BIDS-ified")
            rows.append(dict(adni_subject=adni, bids_subject=sub, session=ses,
                             run=run, source_nii=os.path.relpath(nii, ROOT),
                             dest_nii="", status="REJECTED",
                             reason="; ".join(v["problems"])))
            continue

        dest_dir = os.path.join(BIDS, sub, ses, "func")
        base = f"{sub}_{ses}_task-rest_{run}_bold"
        dn = os.path.join(dest_dir, base + ".nii.gz")
        dj = os.path.join(dest_dir, base + ".json")
        clash = [p for p in (dn, dj, dn[:-3]) if os.path.exists(p)]
        if clash:
            log(f"   DESTINATION ALREADY EXISTS -- refusing to overwrite:")
            for c in clash:
                log(f"      {os.path.relpath(c, ROOT)}")
            rows.append(dict(adni_subject=adni, bids_subject=sub, session=ses,
                             run=run, source_nii=os.path.relpath(nii, ROOT),
                             dest_nii=os.path.relpath(dn, ROOT),
                             status="SKIPPED_EXISTS",
                             reason="destination present"))
            continue

        log(f"   dest   : {os.path.relpath(dn, ROOT)}")
        if apply_changes:
            os.makedirs(dest_dir, exist_ok=True)
            shutil.copy2(nii, dn)
            shutil.copy2(js, dj)
            same = os.path.getsize(nii) == os.path.getsize(dn)
            log(f"   copied ({os.path.getsize(dn)/1e6:.1f} MB), "
                f"size match: {same}")
            status = "COPIED" if same else "SIZE_MISMATCH"
        else:
            status = "WOULD_COPY"
        rows.append(dict(adni_subject=adni, bids_subject=sub, session=ses,
                         run=run, source_nii=os.path.relpath(nii, ROOT),
                         dest_nii=os.path.relpath(dn, ROOT), status=status,
                         reason=""))

    # ---- participants.tsv: append only, never rewrite existing rows -----
    done = [r for r in rows if r["status"] in ("COPIED", "WOULD_COPY")]
    if done:
        with open(PARTS, newline="", encoding="utf-8") as fh:
            existing = list(csv.reader(fh, delimiter="\t"))
        known = {r[0] for r in existing[1:] if r}
        add = [r for r in done if r["bids_subject"] not in known]
        log(f"\nparticipants.tsv: {len(existing)-1} rows, "
            f"{len(add)} to append")
        for r in add:
            log(f"   + {r['bids_subject']}\tMCI")
        if apply_changes and add:
            with open(PARTS, "a", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh, delimiter="\t", lineterminator="\n")
                for r in add:
                    w.writerow([r["bids_subject"], "MCI"])

    os.makedirs(os.path.dirname(MAPOUT), exist_ok=True)
    if rows:
        mode = "a" if (os.path.isfile(MAPOUT) and apply_changes) else "w"
        with open(MAPOUT, mode, newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            if mode == "w":
                w.writeheader()
            w.writerows(rows)
        log(f"\nmapping -> {os.path.relpath(MAPOUT, ROOT)}")
    if not apply_changes:
        log("\nDRY RUN -- nothing written. Re-run with --apply.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    main(apply_changes=a.apply)
