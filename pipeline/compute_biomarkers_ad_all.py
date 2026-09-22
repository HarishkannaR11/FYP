"""
PRODUCTION: compute ALFF (fALFF), ReHo, and Degree Centrality (DC) for ALL 25
AD acquisitions. Reuses the exact same computation functions validated in the
single-subject pilot (scripts/biomarkers/compute_biomarkers_single_subject.py)
-- not reimplemented, same input choices per biomarker:
  - fALFF: pre-GLM smoothed intermediate (NOT the GLM residual)
  - ReHo:  unsmoothed motion-corrected intermediate
  - DC:    final masked GLM residual (derivatives/fsfast/AD/)

FC is NOT computed here -- it requires atlas parcellation (Brainnetome-246),
which is still blocked pending the atlas file + registration-tool decision.

Reads: Nipype work-dir cache (moco/smooth intermediates) + derivatives/fsfast/AD/
  (final bold + mask). All read-only.
Writes ONLY to: derivatives/biomarkers/AD/sub-<ID>/ses-<NN>/
Continues on a per-subject failure; never stops the whole batch; records
every failure explicitly.
"""
import os
import csv
import json
import time
import numpy as np
import nibabel as nib

MAPPING_CSV = "/mnt/c/Users/krish/FYP/audit/ad_session_mapping_final.csv"
WORK_ROOT = "/home/harish/fyp_work/fsfast_ad_masked/work/fsfast_ad_production_masked"
DERIV_AD = "/mnt/c/Users/krish/FYP/derivatives/fsfast/AD"
OUT_ROOT = "/mnt/c/Users/krish/FYP/derivatives/biomarkers/AD"
TR = 3.0


def load_runs():
    rows = list(csv.DictReader(open(MAPPING_CSV, newline="", encoding="utf-8")))
    return sorted([(r["participant_id"], r["session"]) for r in rows])


def compute_falff(data, mask, tr):
    T = data.shape[3]
    freqs = np.fft.rfftfreq(T, d=tr)
    low_band = (freqs >= 0.01) & (freqs <= 0.08)
    vox = data[mask]
    vox = vox - vox.mean(axis=1, keepdims=True)
    fft_amp = np.abs(np.fft.rfft(vox, axis=1))
    total_power = fft_amp.sum(axis=1)
    low_power = fft_amp[:, low_band].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        falff = np.where(total_power > 0, low_power / total_power, 0.0)
    falff_map = np.zeros(mask.shape, dtype=np.float32)
    falff_map[mask] = falff
    return falff_map


def compute_reho(data, mask):
    x, y, z, T = data.shape
    reho_map = np.zeros((x, y, z), dtype=np.float32)
    ranks = np.argsort(np.argsort(data, axis=3), axis=3).astype(np.float32) + 1
    mask_idx = np.array(np.where(mask)).T
    for (i, j, k) in mask_idx:
        i0, i1 = max(i - 1, 0), min(i + 2, x)
        j0, j1 = max(j - 1, 0), min(j + 2, y)
        k0, k1 = max(k - 1, 0), min(k + 2, z)
        neighborhood_mask = mask[i0:i1, j0:j1, k0:k1]
        if neighborhood_mask.sum() < 7:
            continue
        R = ranks[i0:i1, j0:j1, k0:k1, :][neighborhood_mask]
        Nn, Tt = R.shape
        Rj = R.sum(axis=0)
        mean_R = Nn * (Tt + 1) / 2.0
        SS = np.sum((Rj - mean_R) ** 2)
        W = 12 * SS / (Nn ** 2 * (Tt ** 3 - Tt)) if Tt > 1 else 0.0
        reho_map[i, j, k] = W
    return reho_map


def compute_dc(data, mask, r_thresh=0.25, chunk=500):
    vox = data[mask]
    vox = vox - vox.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(vox, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vox_n = vox / norms
    Nvox = vox_n.shape[0]
    degree = np.zeros(Nvox, dtype=np.float32)
    for start in range(0, Nvox, chunk):
        end = min(start + chunk, Nvox)
        corr_block = vox_n[start:end] @ vox_n.T
        corr_block[np.arange(end - start), np.arange(start, end)] = 0
        thresholded = np.where(corr_block > r_thresh, corr_block, 0.0)
        degree[start:end] = thresholded.sum(axis=1)
    dc_map = np.zeros(mask.shape, dtype=np.float32)
    dc_map[mask] = degree
    return dc_map


def process_one(sub, ses):
    t0 = time.time()
    smooth_file = os.path.join(WORK_ROOT, f"_run_key_{sub}_{ses}", "smooth", "smooth_bold.nii.gz")
    moco_file = os.path.join(WORK_ROOT, f"_run_key_{sub}_{ses}", "moco", "moco_bold.nii.gz")
    final_bold = os.path.join(DERIV_AD, sub, ses, "func", f"{sub}_{ses}_task-rest_run-01_desc-preproc_bold.nii.gz")
    mask_file = os.path.join(DERIV_AD, sub, ses, "func", f"{sub}_{ses}_task-rest_run-01_desc-brain_mask.nii.gz")

    for p in [smooth_file, moco_file, final_bold, mask_file]:
        if not os.path.exists(p):
            return {"sub": sub, "ses": ses, "status": "FAIL", "error": f"missing input: {p}", "time_s": time.time() - t0}

    try:
        mask = nib.load(mask_file).get_fdata().astype(bool)
        affine = nib.load(final_bold).affine

        out_dir = os.path.join(OUT_ROOT, sub, ses)
        os.makedirs(out_dir, exist_ok=True)

        smooth_data = nib.load(smooth_file).get_fdata(dtype=np.float32)
        falff_map = compute_falff(smooth_data, mask, TR)
        falff_out = os.path.join(out_dir, f"{sub}_{ses}_falff.nii.gz")
        nib.save(nib.Nifti1Image(falff_map, affine), falff_out)

        moco_data = nib.load(moco_file).get_fdata(dtype=np.float32)
        reho_map = compute_reho(moco_data, mask)
        reho_out = os.path.join(out_dir, f"{sub}_{ses}_reho.nii.gz")
        nib.save(nib.Nifti1Image(reho_map, affine), reho_out)

        final_data = nib.load(final_bold).get_fdata(dtype=np.float32)
        dc_map = compute_dc(final_data, mask, r_thresh=0.25)
        dc_out = os.path.join(out_dir, f"{sub}_{ses}_dc.nii.gz")
        nib.save(nib.Nifti1Image(dc_map, affine), dc_out)

        # free memory between subjects (important given known host RAM constraints)
        del smooth_data, moco_data, final_data

        metrics = {
            "falff_mean": float(falff_map[mask].mean()), "falff_std": float(falff_map[mask].std()),
            "reho_mean": float(reho_map[mask].mean()), "reho_std": float(reho_map[mask].std()),
            "dc_mean": float(dc_map[mask].mean()), "dc_std": float(dc_map[mask].std()),
            "mask_voxels": int(mask.sum()),
        }
        json_out = os.path.join(out_dir, f"{sub}_{ses}_biomarkers.json")
        with open(json_out, "w") as f:
            json.dump({"sub": sub, "ses": ses, "metrics": metrics,
                       "inputs": {"falff": smooth_file, "reho": moco_file, "dc": final_bold, "mask": mask_file},
                       "outputs": {"falff": falff_out, "reho": reho_out, "dc": dc_out}}, f, indent=2)

        return {"sub": sub, "ses": ses, "status": "PASS", "error": "", "time_s": round(time.time() - t0, 1), **metrics}
    except Exception as e:
        return {"sub": sub, "ses": ses, "status": "FAIL", "error": str(e), "time_s": round(time.time() - t0, 1)}


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)
    runs = load_runs()
    print(f"=== Computing ALFF/ReHo/DC for {len(runs)} AD acquisitions ===")
    results = []
    for i, (sub, ses) in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {sub} {ses} ...", end=" ", flush=True)
        r = process_one(sub, ses)
        print(r["status"], f"({r['time_s']}s)" if "time_s" in r else "")
        if r["status"] == "FAIL":
            print(f"    ERROR: {r['error']}")
        results.append(r)

    n_pass = sum(1 for r in results if r["status"] == "PASS")
    n_fail = sum(1 for r in results if r["status"] == "FAIL")
    print(f"\n=== DONE: {n_pass} PASS, {n_fail} FAIL / {len(results)} total ===")

    csv_out = os.path.join(OUT_ROOT, "biomarker_generation_summary.csv")
    fieldnames = sorted(set().union(*[r.keys() for r in results]))
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in results:
            for fn in fieldnames:
                r.setdefault(fn, "")
            w.writerow(r)
    print(f"Summary: {csv_out}")
    print("BIOMARKER_GENERATION_DONE")


if __name__ == "__main__":
    main()
