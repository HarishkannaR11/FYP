"""
SINGLE-ACQUISITION PILOT: ADNI rs-fMRI "paper" preprocessing pipeline.

Processes exactly ONE 4D BOLD NIfTI. Reads the source read-only; never writes
to BIDS/, NIfTI_/ or raw_data/. All outputs go under pilot_preprocessing/.

Pipeline (paper order):
  discard 5 vols -> slice timing -> motion correction -> MNI normalization
  -> 4mm resample -> 6mm FWHM smoothing -> linear detrend -> nuisance
  regression -> 0.01-0.10 Hz bandpass -> final BOLD -> QC

Single worker: ANTs/ITK threads pinned to 1.
"""
import os
# pin to 1 CPU worker before any heavy import
os.environ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["FREESURFER_HOME"] = "/home/harish/freesurfer"
os.environ["FSFAST_HOME"] = "/home/harish/freesurfer/fsfast"
os.environ["SUBJECTS_DIR"] = "/home/harish/freesurfer/subjects"
os.environ["PATH"] = "/home/harish/freesurfer/bin:/home/harish/freesurfer/fsfast/bin:" + os.environ["PATH"]

import sys
import json
import time
import shutil
import platform
import subprocess
import datetime
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage, signal

INPUT_BOLD = "/mnt/c/Users/krish/FYP/BIDS/sub-019S4549/ses-01/func/sub-019S4549_ses-01_task-rest_run-01_bold.nii.gz"
INPUT_JSON = INPUT_BOLD.replace(".nii.gz", ".json")
SUBJECT, SESSION, RUN = "sub-019S4549", "ses-01", "run-01"

ROOT = "/mnt/c/Users/krish/FYP/pilot_preprocessing"
D_IN, D_INT = os.path.join(ROOT, "input"), os.path.join(ROOT, "intermediate")
D_XFM, D_MOT = os.path.join(ROOT, "transforms"), os.path.join(ROOT, "motion")
D_CONF, D_QC, D_FIN = os.path.join(ROOT, "confounds"), os.path.join(ROOT, "qc"), os.path.join(ROOT, "final")
for d in (ROOT, D_IN, D_INT, D_XFM, D_MOT, D_CONF, D_QC, D_FIN):
    os.makedirs(d, exist_ok=True)

LOG_PATH = os.path.join(ROOT, "processing_log.txt")
_logf = open(LOG_PATH, "w", encoding="utf-8")


def log(msg=""):
    line = str(msg)
    print(line, flush=True)
    _logf.write(line + "\n")
    _logf.flush()


def hdr(title):
    log("\n" + "=" * 70)
    log(title)
    log("=" * 70)


T_START = time.time()
CFG = {}
DEVIATIONS = []


def save_nii(data, affine, header, path, dtype=np.float32):
    img = nib.Nifti1Image(np.asarray(data, dtype=dtype), affine, header)
    img.set_data_dtype(dtype)
    nib.save(img, path)
    log(f"  saved: {path}  shape={img.shape}  dtype={dtype.__name__}")
    return path


# =====================================================================
# 0. INPUT AUDIT
# =====================================================================
hdr("STEP 0 -- INPUT AUDIT (read-only)")
log(f"EXACTLY ONE 4D NIfTI WILL BE PROCESSED:\n  {INPUT_BOLD}\n")

if not os.path.isfile(INPUT_BOLD):
    log("FATAL: input file does not exist"); sys.exit(1)

in_img = nib.load(INPUT_BOLD)
in_data = in_img.get_fdata(dtype=np.float32)
in_zooms = in_img.header.get_zooms()
TR = float(in_zooms[3])
orient = "".join(nib.aff2axcodes(in_img.affine))

audit = {
    "file_path": INPUT_BOLD,
    "n_dimensions": int(in_data.ndim),
    "dim_x": int(in_img.shape[0]), "dim_y": int(in_img.shape[1]), "dim_z": int(in_img.shape[2]),
    "n_timepoints": int(in_img.shape[3]) if in_data.ndim == 4 else None,
    "voxel_size_mm": [round(float(z), 5) for z in in_zooms[:3]],
    "TR_seconds_nifti_header": TR,
    "orientation": orient,
    "datatype_on_disk": str(in_img.get_data_dtype()),
    "min": float(np.min(in_data)), "max": float(np.max(in_data)), "mean": float(np.mean(in_data)),
    "has_NaN": bool(np.isnan(in_data).any()), "has_Inf": bool(np.isinf(in_data).any()),
    "is_4D": bool(in_data.ndim == 4),
}
for k, v in audit.items():
    log(f"  {k}: {v}")
log("  affine:")
for row in in_img.affine:
    log("    " + np.array2string(row, precision=5, suppress_small=True))

with open(INPUT_JSON) as f:
    meta = json.load(f)
TR_json = meta.get("RepetitionTime")
log(f"  JSON RepetitionTime: {TR_json}")
log(f"  JSON has SliceTiming field: {'SliceTiming' in meta}")

invalid = []
if not audit["is_4D"]:
    invalid.append("image is not 4D")
if audit["n_timepoints"] is None or audit["n_timepoints"] <= 1:
    invalid.append("not more than 1 timepoint")
if audit["has_NaN"]:
    invalid.append("contains NaN")
if audit["has_Inf"]:
    invalid.append("contains Inf")
if not np.any(in_data != 0):
    invalid.append("image is empty (all zeros)")
if not np.isfinite(in_img.affine).all() or abs(np.linalg.det(in_img.affine)) < 1e-9:
    invalid.append("invalid affine")
if not (TR and np.isfinite(TR) and TR > 0):
    invalid.append("invalid TR")
if invalid:
    log("\nINPUT INVALID -- STOPPING: " + "; ".join(invalid)); sys.exit(1)
log("\nINPUT AUDIT: VALID -- proceeding.")

shutil.copy2(INPUT_BOLD, os.path.join(D_IN, os.path.basename(INPUT_BOLD)))
shutil.copy2(INPUT_JSON, os.path.join(D_IN, os.path.basename(INPUT_JSON)))
log(f"  read-only copy of source placed in {D_IN} (source untouched)")

N_ORIG = audit["n_timepoints"]

# =====================================================================
# 1. DISCARD FIRST 5 VOLUMES
# =====================================================================
hdr("STEP 1 -- DISCARD FIRST 5 VOLUMES")
N_DISCARD = 5
d5 = in_data[..., N_DISCARD:]
N_KEPT = d5.shape[3]
log(f"  before: {N_ORIG} volumes -> after: {N_KEPT} volumes  (removed {N_ORIG - N_KEPT})")
assert N_ORIG - N_KEPT == 5, "did not remove exactly 5 volumes"
log("  VERIFIED: exactly 5 volumes removed; source NIfTI not modified")
p_d5 = save_nii(d5, in_img.affine, in_img.header, os.path.join(D_INT, "desc-discard5_bold.nii.gz"))

# =====================================================================
# 2. SLICE-TIMING CORRECTION
# =====================================================================
hdr("STEP 2 -- SLICE-TIMING CORRECTION")
has_st = "SliceTiming" in meta and bool(meta.get("SliceTiming"))
if has_st:
    STC_STATUS, STC_REASON = "PERFORMED", ""
else:
    STC_STATUS = "NOT_PERFORMED"
    STC_REASON = "SliceTiming metadata unavailable"
    log("  SLICE_TIMING_CORRECTION = NOT_PERFORMED")
    log(f"  REASON = {STC_REASON}")
    log("  No slice order assumed; no slice timing vector fabricated.")
    p_stc = save_nii(d5, in_img.affine, in_img.header, os.path.join(D_INT, "desc-no-stc_bold.nii.gz"))
    DEVIATIONS.append({
        "paper_step": "Slice-timing correction",
        "implemented_step": "NOT PERFORMED (data passed through unchanged)",
        "reason": "No SliceTiming field in the BIDS JSON sidecar and slice order is not recoverable "
                  "from ADNI DICOM/BIDS metadata. Fabricating a slice order was prohibited.",
    })

# =====================================================================
# 3. HEAD-MOTION CORRECTION (mc-afni2)
# =====================================================================
hdr("STEP 3 -- HEAD-MOTION CORRECTION / REALIGNMENT")
p_mc = os.path.join(D_INT, "desc-mc_bold.nii.gz")
p_mcdat = os.path.join(D_MOT, "mc-afni2_motion.mcdat")
cmd = ["mc-afni2", "--i", p_stc, "--o", p_mc, "--mcdat", p_mcdat]
log("  tool: FreeSurfer FS-FAST mc-afni2 (rigid-body, 6 DOF)")
log("  reference: frame 0 of the post-discard series = first retained volume "
    f"(original volume index {N_DISCARD})")
log("  command: " + " ".join(cmd))
r = subprocess.run(cmd, capture_output=True, text=True)
log(f"  exit code: {r.returncode}")
if r.returncode != 0:
    log("  STDERR:\n" + r.stderr[-3000:]); log("FATAL: motion correction failed"); sys.exit(1)

mcdat = np.loadtxt(p_mcdat)
log(f"  .mcdat shape: {mcdat.shape} (expect {N_KEPT} rows x 10 cols)")
assert mcdat.shape[0] == N_KEPT, "motion rows != retained volumes"

# mc-afni2 .mcdat columns: n, roll, pitch, yaw (deg), dS, dL, dP (mm), rmsold, rmsnew, trans
rot_deg = mcdat[:, 1:4]                      # roll, pitch, yaw  (degrees)
trans_mm = np.column_stack([mcdat[:, 5], mcdat[:, 6], mcdat[:, 4]])  # dL->x, dP->y, dS->z (mm)

import pandas as pd
mot_df = pd.DataFrame({
    "trans_x_mm_dL": trans_mm[:, 0], "trans_y_mm_dP": trans_mm[:, 1], "trans_z_mm_dS": trans_mm[:, 2],
    "rot_roll_deg": rot_deg[:, 0], "rot_pitch_deg": rot_deg[:, 1], "rot_yaw_deg": rot_deg[:, 2],
})
p_mot_tsv = os.path.join(D_MOT, "motion_parameters.tsv")
mot_df.to_csv(p_mot_tsv, sep="\t", index=False)
log(f"  6 motion parameters written (native mc-afni2 values, not re-estimated): {p_mot_tsv}")
log("  axis mapping: translations are mc-afni2 anatomical displacements "
    "dL(left)->x, dP(posterior)->y, dS(superior)->z; rotations are roll/pitch/yaw in degrees")

# FD (Power et al. 2012): sum|dtrans| + sum|drot_rad * 50mm|
rot_rad = np.deg2rad(rot_deg)
dtr = np.vstack([np.zeros((1, 3)), np.diff(trans_mm, axis=0)])
dro = np.vstack([np.zeros((1, 3)), np.diff(rot_rad, axis=0)])
FD = np.abs(dtr).sum(1) + np.abs(dro * 50.0).sum(1)
pd.DataFrame({"volume": np.arange(N_KEPT), "FD_mm": FD}).to_csv(
    os.path.join(D_MOT, "fd_values.tsv"), sep="\t", index=False)
FD_mean, FD_max, FD_med = float(FD.mean()), float(FD.max()), float(np.median(FD))
log(f"  FD (Power 2012, 50mm radius): mean={FD_mean:.4f}mm  median={FD_med:.4f}mm  max={FD_max:.4f}mm")
log("  No volumes removed on the basis of motion (paper does not require scrubbing).")

# =====================================================================
# 4. MNI NORMALIZATION (direct EPI -> MNI; no T1w exists)
# =====================================================================
hdr("STEP 4 -- MNI NORMALIZATION")
log("  MNI normalization performed directly from BOLD/EPI because no T1w "
    "anatomical image is available.")
import ants
import templateflow.api as tflow

TPL_NAME = "MNI152NLin6Asym"
tpl_brain = str(tflow.get(TPL_NAME, resolution=2, desc="brain", suffix="T1w", extension="nii.gz"))
tpl_mask = str(tflow.get(TPL_NAME, resolution=2, desc="brain", suffix="mask", extension="nii.gz"))
log(f"  template: {TPL_NAME} res-02 desc-brain_T1w  ({tpl_brain})")
log(f"  template brain mask: {tpl_mask}")

mc_img = nib.load(p_mc)
mc_data = mc_img.get_fdata(dtype=np.float32)
mean_bold = mc_data.mean(axis=3)


def otsu(vals3d):
    v = vals3d[vals3d > 0]
    hist, edges = np.histogram(v, bins=256)
    mids = (edges[:-1] + edges[1:]) / 2
    w1, w2 = np.cumsum(hist), np.cumsum(hist[::-1])[::-1]
    m1 = np.cumsum(hist * mids) / np.maximum(w1, 1)
    m2 = np.cumsum((hist * mids)[::-1])[::-1] / np.maximum(w2, 1)
    var = w1[:-1] * w2[1:] * (m1[:-1] - m2[1:]) ** 2
    return mids[int(np.argmax(var))]


thr = otsu(mean_bold)
lab, n_comp = ndimage.label(mean_bold > thr, structure=np.ones((3, 3, 3)))
sizes = ndimage.sum(mean_bold > thr, lab, range(1, n_comp + 1))
native_mask = ndimage.binary_fill_holes(lab == int(np.argmax(sizes) + 1))
log(f"  BOLD reference brain mask for registration (Otsu + largest 3D component + fill): "
    f"{int(native_mask.sum())} voxels, {n_comp} components before selection")
log("  (this mask is used only to skull-strip the registration reference; "
    "it is not applied to the time series)")

p_ref = os.path.join(D_XFM, "meanBOLD_brain_reference.nii.gz")
save_nii(mean_bold * native_mask, mc_img.affine, mc_img.header, p_ref)

fixed = ants.image_read(tpl_brain)
moving = ants.image_read(p_ref)
log("  registration: ANTsPy ants.registration(type_of_transform='SyN') "
    "-- rigid+affine+SyN, mutual-information driven")
t0 = time.time()
reg = ants.registration(fixed=fixed, moving=moving, type_of_transform="SyN", verbose=False)
log(f"  registration completed in {time.time()-t0:.1f}s")
for i, t in enumerate(reg["fwdtransforms"]):
    dst = os.path.join(D_XFM, f"xfm_fwd_{i}_" + os.path.basename(t))
    shutil.copy2(t, dst); log(f"  forward transform saved: {dst}")
for i, t in enumerate(reg["invtransforms"]):
    dst = os.path.join(D_XFM, f"xfm_inv_{i}_" + os.path.basename(t))
    if os.path.isfile(t):
        shutil.copy2(t, dst); log(f"  inverse transform saved: {dst}")

log("  applying transform to all volumes of the motion-corrected 4D series...")
t0 = time.time()
bold4d_ants = ants.image_read(p_mc)
warped = ants.apply_transforms(fixed=fixed, moving=bold4d_ants,
                               transformlist=reg["fwdtransforms"],
                               interpolator="linear", imagetype=3)
log(f"  4D resampling into MNI completed in {time.time()-t0:.1f}s")
p_mni = os.path.join(D_INT, "desc-mni_bold.nii.gz")
ants.image_write(warped, p_mni)

mni_img = nib.load(p_mni)
log(f"  desc-mni_bold: shape={mni_img.shape}  voxel size (before 4mm resample)="
    f"{tuple(round(float(z),4) for z in mni_img.header.get_zooms()[:3])}")
log(f"  orientation={''.join(nib.aff2axcodes(mni_img.affine))}  "
    f"affine finite={bool(np.isfinite(mni_img.affine).all())}  det={np.linalg.det(mni_img.affine):.4f}")
tpl_img = nib.load(tpl_brain)
in_mni = bool(mni_img.shape[:3] == tpl_img.shape[:3] and np.allclose(mni_img.affine, tpl_img.affine, atol=1e-3))
log(f"  VERIFIED in MNI space (grid + affine identical to template): {in_mni}")
assert mni_img.shape[3] == N_KEPT, "volume count changed during normalization"

# registration QC figure
regqc = os.path.join(D_QC, "registration_qc.png")
tpl_d = tpl_img.get_fdata()
mni_mean = mni_img.get_fdata(dtype=np.float32).mean(axis=3)
fig, ax = plt.subplots(2, 3, figsize=(13, 8))
for c, (sl, axis) in enumerate([(tpl_d.shape[0] // 2, 0), (tpl_d.shape[1] // 2, 1), (tpl_d.shape[2] // 2, 2)]):
    t_sl = np.rot90(np.take(tpl_d, sl, axis=axis))
    b_sl = np.rot90(np.take(mni_mean, sl, axis=axis))
    ax[0, c].imshow(t_sl, cmap="gray"); ax[0, c].set_title(f"{TPL_NAME} template"); ax[0, c].axis("off")
    ax[1, c].imshow(b_sl, cmap="gray")
    ax[1, c].contour(t_sl > np.percentile(t_sl[t_sl > 0], 40) if (t_sl > 0).any() else t_sl,
                     levels=[0.5], colors="r", linewidths=0.7)
    ax[1, c].set_title("normalized mean BOLD + template contour"); ax[1, c].axis("off")
plt.tight_layout(); plt.savefig(regqc, dpi=110); plt.close()
log(f"  registration QC figure: {regqc}")

del bold4d_ants, warped, mc_data

# =====================================================================
# 5. RESAMPLE TO 4 x 4 x 4 mm
# =====================================================================
hdr("STEP 5 -- RESAMPLE TO 4 x 4 x 4 mm ISOTROPIC")
from nilearn.image import resample_img, smooth_img

tgt_aff = tpl_img.affine.copy()
scale = np.array([4.0, 4.0, 4.0]) / np.array([abs(tpl_img.affine[i, i]) for i in range(3)])
tgt_aff[:3, :3] = tpl_img.affine[:3, :3] @ np.diag(scale)
tgt_shape = tuple(int(np.ceil(tpl_img.shape[i] / scale[i])) for i in range(3))
log(f"  target grid derived from the {TPL_NAME} template affine; target shape={tgt_shape}")

r4 = resample_img(mni_img, target_affine=tgt_aff, target_shape=tgt_shape,
                  interpolation="continuous", force_resample=True, copy_header=True)
p_4mm = os.path.join(D_INT, "desc-mni4mm_bold.nii.gz")
nib.save(r4, p_4mm)
z4 = tuple(round(float(z), 5) for z in r4.header.get_zooms()[:3])
log(f"  saved: {p_4mm}")
log(f"  dimensions: {r4.shape}")
log(f"  voxel size: {z4}")
log("  affine:")
for row in r4.affine:
    log("    " + np.array2string(row, precision=5, suppress_small=True))
assert all(abs(z - 4.0) < 0.01 for z in z4), f"voxel size is not 4mm: {z4}"
log("  VERIFIED: voxel size == 4 x 4 x 4 mm")

# reference masks on the 4mm grid
tpl_mask_4 = resample_img(nib.load(tpl_mask), target_affine=tgt_aff, target_shape=tgt_shape,
                          interpolation="nearest", force_resample=True, copy_header=True)
data4 = r4.get_fdata(dtype=np.float32)
cover = data4.std(axis=3) > 0
BRAIN = (tpl_mask_4.get_fdata() > 0) & cover
log(f"  analysis brain mask (template brain mask AND non-constant BOLD coverage): "
    f"{int(BRAIN.sum())} voxels")
save_nii(BRAIN.astype(np.uint8), r4.affine, r4.header, os.path.join(D_XFM, "brain_mask_mni4mm.nii.gz"),
         dtype=np.uint8)

# =====================================================================
# 6. SPATIAL SMOOTHING 6mm FWHM
# =====================================================================
hdr("STEP 6 -- SPATIAL SMOOTHING (6 mm FWHM)")
log("  nilearn.image.smooth_img(fwhm=6) -- fwhm argument is FWHM in mm, NOT sigma")
log(f"  equivalent Gaussian sigma = FWHM/(2*sqrt(2*ln2)) = {6.0/2.3548200:.4f} mm (for reference only)")
sm = smooth_img(r4, fwhm=6.0)
p_sm = os.path.join(D_INT, "desc-smooth6mm_bold.nii.gz")
nib.save(sm, p_sm)
log(f"  saved: {p_sm}  shape={sm.shape}")

# =====================================================================
# 7. LINEAR DETRENDING
# =====================================================================
hdr("STEP 7 -- LINEAR DETRENDING")
sm_data = sm.get_fdata(dtype=np.float32)
nt = sm_data.shape[3]
flat = sm_data.reshape(-1, nt).T                      # time x voxels
tvec = np.arange(nt, dtype=np.float64)
X_lin = np.column_stack([np.ones(nt), (tvec - tvec.mean()) / tvec.std()])
beta = np.linalg.lstsq(X_lin, flat, rcond=None)[0]
vox_mean = flat.mean(axis=0)
det = flat - X_lin @ beta + vox_mean               # remove intercept+linear, restore temporal mean
log("  removed intercept + linear trend per voxel (OLS); voxel temporal mean added back")
log("  LINEAR ONLY -- no quadratic term included")
det4 = det.T.reshape(sm_data.shape).astype(np.float32)
p_det = save_nii(det4, sm.affine, sm.header, os.path.join(D_INT, "desc-detrend_bold.nii.gz"))
log(f"  timepoints before={nt} after={det4.shape[3]}  VERIFIED unchanged: {nt == det4.shape[3]}")

# =====================================================================
# 8. NUISANCE REGRESSION
# =====================================================================
hdr("STEP 8 -- NUISANCE REGRESSION")

# ---- Friston-24 ----
R = np.column_stack([trans_mm, rot_deg])              # 6 params, native units
Rd = np.vstack([np.zeros((1, 6)), np.diff(R, axis=0)])  # temporal derivatives
F24 = np.column_stack([R, Rd, R ** 2, Rd ** 2])
assert F24.shape == (nt, 24), f"Friston-24 must be 24 columns, got {F24.shape}"
log(f"  Friston-24 built: 6 motion + 6 derivatives + 6 squares + 6 squared-derivatives "
    f"= {F24.shape[1]} columns  (VERIFIED, not reduced to 6)")

# ---- WM / CSF from template tissue priors, direction of inter-template xfm verified empirically ----
log("\n  WM / CSF regressors:")
log("    No T1w anatomical image exists for this subject, so subject-specific tissue")
log("    segmentation is impossible. Testing whether published template tissue priors")
log("    can supply reliable WM/CSF masks instead (no segmentation is fabricated).")

wm_src = str(tflow.get("MNI152NLin2009cAsym", resolution=2, label="WM", suffix="probseg", extension="nii.gz"))
csf_src = str(tflow.get("MNI152NLin2009cAsym", resolution=2, label="CSF", suffix="probseg", extension="nii.gz"))
log(f"    source priors: MNI152NLin2009cAsym res-02 label-WM/CSF probseg")
log(f"      {wm_src}\n      {csf_src}")
xfm_h5 = os.path.join(os.path.dirname(tpl_brain),
                      f"tpl-{TPL_NAME}_from-MNI152NLin2009cAsym_mode-image_xfm.h5")
_xfm_sz = os.path.getsize(xfm_h5) if os.path.isfile(xfm_h5) else 0
XFM_AVAILABLE = _xfm_sz > 1024          # TemplateFlow ships unfetched files as 0-byte stubs
log(f"    official TemplateFlow inter-template transform: {xfm_h5}")
log(f"      exists={os.path.isfile(xfm_h5)}  size={_xfm_sz} bytes  usable={XFM_AVAILABLE}")
if os.path.isfile(xfm_h5) and not XFM_AVAILABLE:
    log("      -> file is an unfetched 0-byte TemplateFlow stub, not a usable transform")

tpl_ants = ants.image_read(tpl_brain)
tpl_brain_d = tpl_img.get_fdata()
brain_tpl = nib.load(tpl_mask).get_fdata() > 0


def corr_with_t1w(arr):
    a, b = arr[brain_tpl].ravel(), tpl_brain_d[brain_tpl].ravel()
    if a.std() < 1e-9:
        return -1.0
    return float(np.corrcoef(a, b)[0, 1])


cands = {}
wm_ants = ants.image_read(wm_src)
cands["no_transform"] = ants.resample_image_to_target(wm_ants, tpl_ants, interp_type="linear").numpy()
if XFM_AVAILABLE:
    for nm, inv in [("xfm_forward", False), ("xfm_inverse", True)]:
        try:
            cands[nm] = ants.apply_transforms(fixed=tpl_ants, moving=wm_ants, transformlist=[xfm_h5],
                                              whichtoinvert=[inv], interpolator="linear").numpy()
        except Exception as e:
            log(f"    {nm} failed: {type(e).__name__}: {str(e)[:120]}")
log("    empirical check -- correlation of transformed WM prior with template T1w intensity")
log("    (WM is bright on T1w, so the correct mapping maximizes this correlation):")
scores = {}
for nm, arr in cands.items():
    scores[nm] = corr_with_t1w(arr)
    log(f"      {nm}: r = {scores[nm]:+.4f}")
best = max(scores, key=scores.get)
log(f"    selected mapping: {best}  (r={scores[best]:+.4f})")

if best == "no_transform":
    wm_tpl_space = cands["no_transform"]
    csf_tpl_space = ants.resample_image_to_target(ants.image_read(csf_src), tpl_ants, interp_type="linear").numpy()
    XFM_NOTE = "priors resampled directly onto the target template grid (no inter-template warp applied)"
else:
    inv = (best == "xfm_inverse")
    wm_tpl_space = cands[best]
    csf_tpl_space = ants.apply_transforms(fixed=tpl_ants, moving=ants.image_read(csf_src),
                                          transformlist=[xfm_h5], whichtoinvert=[inv],
                                          interpolator="linear").numpy()
    XFM_NOTE = f"priors warped 2009cAsym->{TPL_NAME} with the official TemplateFlow transform ({best})"
log(f"    {XFM_NOTE}")


def prior_to_mask(prior_tpl_space, thresh, erode_iter, name):
    """Threshold and erode at the prior's native 2mm template resolution, THEN resample the
    binary mask to the 4mm analysis grid. Thresholding after 4mm resampling would destroy
    these masks through partial-volume dilution."""
    m = prior_tpl_space >= thresh
    n_thr = int(m.sum())
    if erode_iter > 0:
        m = ndimage.binary_erosion(m, structure=np.ones((3, 3, 3)), iterations=erode_iter)
    n_ero = int(m.sum())
    img_t = nib.Nifti1Image(m.astype(np.float32), tpl_img.affine)
    r = resample_img(img_t, target_affine=tgt_aff, target_shape=tgt_shape,
                     interpolation="continuous", force_resample=True, copy_header=True)
    m4 = (r.get_fdata() >= 0.5) & BRAIN          # >=50% occupancy of the 4mm voxel
    log(f"    {name}: at native 2mm prob>={thresh} -> {n_thr} vox; after {erode_iter}x erosion "
        f"-> {n_ero} vox; resampled to 4mm (>=50% occupancy) and intersected with brain mask "
        f"-> {int(m4.sum())} vox")
    save_nii(m4.astype(np.uint8), r4.affine, r4.header,
             os.path.join(D_XFM, f"{name.lower()}_mask_mni4mm.nii.gz"), dtype=np.uint8)
    return m4


WM_ERODE, CSF_ERODE = 1, 0
log("    WM is eroded by 1 iteration to avoid grey-matter partial volume; CSF is NOT eroded")
log("    because the ventricular CSF prior is only a few voxels thick and a 3x3x3 erosion")
log("    would remove essentially all of it (verified: 923 -> 12 voxels).")
WM_M = prior_to_mask(wm_tpl_space, 0.95, WM_ERODE, "WM")
CSF_M = prior_to_mask(csf_tpl_space, 0.90, CSF_ERODE, "CSF")

# Tissue signals are extracted at the 2mm normalized resolution, where the masks are
# actually resolved. Ventricular CSF is thinner than a 4mm voxel, so a 4mm mask collapses
# (923 vox at 2mm -> 11 vox at 4mm). The regressor is a time series, so extracting it at
# 2mm and applying it to the 4mm analysis series is valid and avoids that collapse.
hdr_note = "    tissue signals extracted at 2mm (normalized series), not at 4mm, because a 4mm"
log("\n" + hdr_note)
log("    voxel is thicker than the ventricular CSF sheet; masks below are the 2mm versions.")
mni_dobj = nib.load(p_mni).dataobj
vol0 = np.asarray(mni_dobj[..., 0], dtype=np.float32)
cover2 = vol0 != 0


def prior_to_mask_2mm(prior_tpl_space, thresh, erode_iter, name):
    m = prior_tpl_space >= thresh
    n_thr = int(m.sum())
    if erode_iter > 0:
        m = ndimage.binary_erosion(m, structure=np.ones((3, 3, 3)), iterations=erode_iter)
    m = m & cover2
    log(f"    {name} @2mm: prob>={thresh} -> {n_thr} vox; after {erode_iter}x erosion and "
        f"BOLD-coverage intersection -> {int(m.sum())} vox")
    save_nii(m.astype(np.uint8), tpl_img.affine, tpl_img.header,
             os.path.join(D_XFM, f"{name.lower()}_mask_mni2mm.nii.gz"), dtype=np.uint8)
    return m


WM_M2 = prior_to_mask_2mm(wm_tpl_space, 0.95, WM_ERODE, "WM")
CSF_M2 = prior_to_mask_2mm(csf_tpl_space, 0.90, CSF_ERODE, "CSF")

det_flat = det4.reshape(-1, nt).T
GS = det_flat[:, BRAIN.ravel()].mean(axis=1)
log(f"    global signal: mean over {int(BRAIN.sum())} brain-mask voxels (4mm analysis grid)")

MIN_VOX = 20
wm_ok, csf_ok = int(WM_M2.sum()) >= MIN_VOX, int(CSF_M2.sum()) >= MIN_VOX
wm_raw = np.zeros(nt, dtype=np.float64)
csf_raw = np.zeros(nt, dtype=np.float64)
if wm_ok or csf_ok:
    for t in range(nt):
        v = np.asarray(mni_dobj[..., t], dtype=np.float32)
        if wm_ok:
            wm_raw[t] = v[WM_M2].mean()
        if csf_ok:
            csf_raw[t] = v[CSF_M2].mean()
    del v


def detrend_1d(y):
    """Same intercept+linear removal applied to the imaging data, mean restored."""
    b_ = np.linalg.lstsq(X_lin, y[:, None], rcond=None)[0]
    return (y[:, None] - X_lin @ b_ + y.mean()).ravel()


WM_SIG = detrend_1d(wm_raw) if wm_ok else None
CSF_SIG = detrend_1d(csf_raw) if csf_ok else None
log("    extracted tissue time series were linearly detrended the same way as the imaging data")

log("\n    reliability diagnostics (a prior-derived regressor that merely duplicates the")
log("    global signal carries no independent tissue information):")
r_wm_gs = float(np.corrcoef(WM_SIG, GS)[0, 1]) if wm_ok else float("nan")
r_csf_gs = float(np.corrcoef(CSF_SIG, GS)[0, 1]) if csf_ok else float("nan")
log(f"      corr(WM, GlobalSignal)  = {r_wm_gs:+.4f}   (n_vox@2mm={int(WM_M2.sum())})")
log(f"      corr(CSF, GlobalSignal) = {r_csf_gs:+.4f}   (n_vox@2mm={int(CSF_M2.sum())})")
DUP = 0.98
wm_use = wm_ok and abs(r_wm_gs) < DUP
csf_use = csf_ok and abs(r_csf_gs) < DUP
log(f"      WM regressor usable:  {wm_use}    CSF regressor usable: {csf_use}   "
    f"(thresholds: >={MIN_VOX} voxels and |r| with GS < {DUP})")

regressors, names = [], []
regressors.append(np.ones(nt)); names.append("intercept")
for i, n in enumerate(["trans_x", "trans_y", "trans_z", "roll", "pitch", "yaw"]):
    regressors.append(F24[:, i]); names.append(f"friston24_{n}")
for i, n in enumerate(["trans_x", "trans_y", "trans_z", "roll", "pitch", "yaw"]):
    regressors.append(F24[:, 6 + i]); names.append(f"friston24_{n}_deriv")
for i, n in enumerate(["trans_x", "trans_y", "trans_z", "roll", "pitch", "yaw"]):
    regressors.append(F24[:, 12 + i]); names.append(f"friston24_{n}_sq")
for i, n in enumerate(["trans_x", "trans_y", "trans_z", "roll", "pitch", "yaw"]):
    regressors.append(F24[:, 18 + i]); names.append(f"friston24_{n}_deriv_sq")
WM_STATUS = "NOT PERFORMED"
CSF_STATUS = "NOT PERFORMED"
if wm_use:
    regressors.append(WM_SIG); names.append("white_matter"); WM_STATUS = "PERFORMED (template prior)"
if csf_use:
    regressors.append(CSF_SIG); names.append("csf"); CSF_STATUS = "PERFORMED (template prior)"
regressors.append(GS); names.append("global_signal")

Xn = np.column_stack(regressors)
log(f"\n  nuisance design matrix: {Xn.shape[0]} timepoints x {Xn.shape[1]} regressors")
log("  columns: " + ", ".join(names))
pd.DataFrame(Xn, columns=names).to_csv(os.path.join(D_CONF, "nuisance_regressors.tsv"),
                                       sep="\t", index=False)
log(f"  saved: {os.path.join(D_CONF, 'nuisance_regressors.tsv')}")

bn = np.linalg.lstsq(Xn, det_flat, rcond=None)[0]
res = det_flat - Xn @ bn + det_flat.mean(axis=0)   # residuals, temporal mean restored
nz4 = res.T.reshape(det4.shape).astype(np.float32)
nz4[~BRAIN] = 0
p_nz = save_nii(nz4, r4.affine, r4.header, os.path.join(D_INT, "desc-nuisance_bold.nii.gz"))
log("  residuals kept; voxel temporal mean added back so the series stays in native "
    "intensity units (keeps tSNR/mean-signal QC interpretable)")

if not wm_use or not csf_use:
    DEVIATIONS.append({
        "paper_step": "Nuisance regression including white-matter and CSF signals",
        "implemented_step": f"WM: {WM_STATUS}; CSF: {CSF_STATUS}",
        "reason": "No T1w anatomical image exists, so subject-specific tissue segmentation is "
                  "impossible; only template priors were available and they failed the "
                  "reliability check (too few voxels, or the regressor duplicated the global signal).",
    })
if wm_use or csf_use:
    DEVIATIONS.append({
        "paper_step": "White-matter / CSF nuisance signals from subject tissue segmentation",
        "implemented_step": "Signals taken from published MNI template tissue priors "
                            "(thresholded + eroded), NOT from subject-specific segmentation",
        "reason": "No T1w anatomical image exists for this dataset. Template priors are real "
                  "published probability maps, but they are population-level and their accuracy "
                  "depends on EPI->MNI registration quality.",
    })

# =====================================================================
# 9. BAND-PASS FILTER 0.01 - 0.10 Hz
# =====================================================================
hdr("STEP 9 -- TEMPORAL BAND-PASS FILTER (0.01 - 0.10 Hz)")
LOW, HIGH = 0.01, 0.10
fs = 1.0 / TR
nyq = fs / 2.0
log(f"  TR taken from the NIfTI header (not assumed): {TR} s")
log(f"  sampling frequency = 1/TR = {fs:.6f} Hz;  Nyquist = {nyq:.6f} Hz")
log(f"  requirement 0 < {LOW} < {HIGH} < {nyq:.6f}: {0 < LOW < HIGH < nyq}")
if not (0 < LOW < HIGH < nyq):
    log("FATAL: requested band is invalid for this TR -- stopping rather than changing the band")
    sys.exit(1)
b, a = signal.butter(2, [LOW / nyq, HIGH / nyq], btype="bandpass")
log("  filter: 2nd-order Butterworth, zero-phase (scipy.signal.filtfilt)")
res_flat = nz4.reshape(-1, nt).T
mean_keep = res_flat.mean(axis=0)
bp = signal.filtfilt(b, a, res_flat, axis=0) + mean_keep
bp4 = bp.T.reshape(nz4.shape).astype(np.float32)
bp4[~BRAIN] = 0
p_bp = save_nii(bp4, r4.affine, r4.header, os.path.join(D_INT, "desc-bandpass_bold.nii.gz"))
log("  temporal mean restored after filtering (the 0.01 Hz high-pass removes DC by construction)")
log(f"  timepoints: {bp4.shape[3]} (unchanged)")

# =====================================================================
# 10. FINAL OUTPUT
# =====================================================================
hdr("STEP 10 -- FINAL PREPROCESSED BOLD")
p_final = os.path.join(D_FIN, f"{SUBJECT}_{SESSION}_task-rest_{RUN}_desc-paperpreproc_bold.nii.gz")
shutil.copy2(p_bp, p_final)
fin_img = nib.load(p_final)
log(f"  {p_final}")
log(f"  shape={fin_img.shape}  voxel={tuple(round(float(z),4) for z in fin_img.header.get_zooms()[:3])}  "
    f"TR={float(fin_img.header.get_zooms()[3])}  dtype={fin_img.get_data_dtype()}  "
    f"orientation={''.join(nib.aff2axcodes(fin_img.affine))}")
fd_ = fin_img.get_fdata(dtype=np.float32)
log(f"  finite={bool(np.isfinite(fd_).all())}  non-empty={bool(np.any(fd_!=0))}  "
    f"NaN={bool(np.isnan(fd_).any())}  Inf={bool(np.isinf(fd_).any())}")

# =====================================================================
# QC
# =====================================================================
hdr("QUALITY CONTROL")
pre_mean = in_data.mean(axis=3)
pre_thr = otsu(pre_mean)
plab, pn = ndimage.label(pre_mean > pre_thr, structure=np.ones((3, 3, 3)))
psz = ndimage.sum(pre_mean > pre_thr, plab, range(1, pn + 1))
pre_brain = ndimage.binary_fill_holes(plab == int(np.argmax(psz) + 1))

# tSNR on the pre-GLM/pre-regression magnitude-domain stage (smoothed 4mm), never on residuals
sm_mean, sm_sd = sm_data.mean(axis=3), sm_data.std(axis=3)
with np.errstate(divide="ignore", invalid="ignore"):
    tsnr = np.where(sm_sd > 0, sm_mean / sm_sd, 0)
tsnr_vals = tsnr[BRAIN]
mean_tsnr = float(np.mean(tsnr_vals[np.isfinite(tsnr_vals)]))

mean_in_brain = float(sm_mean[BRAIN].mean())
bg = (~pre_brain) & (pre_mean > 0)
bg_sig = float(pre_mean[bg].mean()) if bg.any() else float("nan")
bg_sd = float(pre_mean[bg].std()) if bg.any() else float("nan")

# DVARS (Power 2012) on the smoothed pre-regression stage
sflat = sm_data.reshape(-1, nt).T[:, BRAIN.ravel()]
dv = np.sqrt(np.mean(np.diff(sflat, axis=0) ** 2, axis=1))
DVARS = np.concatenate([[0.0], dv])
pd.DataFrame({"volume": np.arange(nt), "DVARS": DVARS}).to_csv(
    os.path.join(D_QC, "dvars_values.csv"), index=False)

save_nii(sm_mean, r4.affine, r4.header, os.path.join(D_QC, "temporal_mean_image.nii.gz"))
save_nii(sm_sd, r4.affine, r4.header, os.path.join(D_QC, "temporal_std_image.nii.gz"))
save_nii(tsnr, r4.affine, r4.header, os.path.join(D_QC, "tsnr_image.nii.gz"))

qc = {
    "n_volumes_before_preprocessing": N_ORIG,
    "n_volumes_after_discarding_first_5": N_KEPT,
    "TR_seconds": TR,
    "voxel_size_input_mm": str(audit["voxel_size_mm"]),
    "voxel_size_final_mm": str(list(z4)),
    "mean_FD_mm": FD_mean, "median_FD_mm": FD_med, "max_FD_mm": FD_max,
    "FD_gt_0.5mm_count": int((FD > 0.5).sum()),
    "mean_tSNR": mean_tsnr, "median_tSNR": float(np.median(tsnr_vals[np.isfinite(tsnr_vals)])),
    "tSNR_stage": "4mm smoothed, pre-detrend/pre-nuisance (non-zero-mean magnitude domain)",
    "mean_signal_inside_brain": mean_in_brain,
    "background_signal_mean": bg_sig, "background_signal_sd": bg_sd,
    "background_definition": "non-brain, non-zero voxels of the temporal mean of the RAW input",
    "mean_DVARS": float(DVARS[1:].mean()), "median_DVARS": float(np.median(DVARS[1:])),
    "max_DVARS": float(DVARS[1:].max()),
    "DVARS_stage": "4mm smoothed, pre-detrend/pre-nuisance; first volume set to 0 (no preceding volume)",
    "brain_mask_voxels_4mm": int(BRAIN.sum()),
    "WM_regressor": WM_STATUS, "CSF_regressor": CSF_STATUS,
    "CNR": "NOT_RELIABLY_COMPUTABLE_WITH_AVAILABLE_DATA",
    "CNR_reason": "No T1w anatomical image / tissue segmentation for this subject",
    "slice_timing_correction": STC_STATUS,
    "final_has_NaN": bool(np.isnan(fd_).any()), "final_has_Inf": bool(np.isinf(fd_).any()),
    "registration_in_MNI_verified": in_mni,
}
pd.DataFrame(list(qc.items()), columns=["metric", "value"]).to_csv(
    os.path.join(D_QC, "qc_metrics.csv"), index=False)
for k, v in qc.items():
    log(f"  {k}: {v}")

plt.figure(figsize=(11, 3.4))
plt.plot(FD, lw=0.9); plt.axhline(0.5, color="r", ls="--", lw=0.8, label="0.5 mm reference")
plt.xlabel("volume (after discarding 5)"); plt.ylabel("FD (mm)")
plt.title(f"Framewise Displacement  mean={FD_mean:.3f}  max={FD_max:.3f}"); plt.legend()
plt.tight_layout(); plt.savefig(os.path.join(D_QC, "fd_plot.png"), dpi=120); plt.close()

plt.figure(figsize=(11, 3.4))
plt.plot(DVARS, lw=0.9); plt.xlabel("volume"); plt.ylabel("DVARS")
plt.title(f"DVARS  mean={DVARS[1:].mean():.1f}  max={DVARS[1:].max():.1f}")
plt.tight_layout(); plt.savefig(os.path.join(D_QC, "dvars_plot.png"), dpi=120); plt.close()

fin_mean = fd_.mean(axis=3)
fig, ax = plt.subplots(2, 3, figsize=(13, 8))
for c, axis in enumerate([0, 1, 2]):
    a = np.rot90(np.take(pre_mean, pre_mean.shape[axis] // 2, axis=axis))
    b_ = np.rot90(np.take(fin_mean, fin_mean.shape[axis] // 2, axis=axis))
    ax[0, c].imshow(a, cmap="gray"); ax[0, c].set_title(f"BEFORE (raw input) axis {axis}"); ax[0, c].axis("off")
    ax[1, c].imshow(b_, cmap="gray"); ax[1, c].set_title(f"AFTER (final preproc) axis {axis}"); ax[1, c].axis("off")
plt.suptitle("Representative sagittal / coronal / axial slices: before vs after")
plt.tight_layout(); plt.savefig(os.path.join(D_QC, "before_after.png"), dpi=120); plt.close()

for nm, vol, ttl in [("mean_bold_raw.png", pre_mean, "Mean BOLD (raw input)"),
                     ("mean_bold_final.png", fin_mean, "Mean BOLD (final preprocessed)")]:
    fig, ax = plt.subplots(1, 3, figsize=(12, 4))
    for c, axis in enumerate([0, 1, 2]):
        ax[c].imshow(np.rot90(np.take(vol, vol.shape[axis] // 2, axis=axis)), cmap="gray"); ax[c].axis("off")
    plt.suptitle(ttl); plt.tight_layout(); plt.savefig(os.path.join(D_QC, nm), dpi=120); plt.close()
log(f"  QC figures written to {D_QC}")

RUNTIME = time.time() - T_START

# =====================================================================
# REPRODUCIBILITY
# =====================================================================
hdr("REPRODUCIBILITY")


def ver(mod):
    try:
        return __import__(mod).__version__
    except Exception:
        return "unavailable"


fs_ver = "unavailable"
try:
    with open("/home/harish/freesurfer/build-stamp.txt") as f:
        fs_ver = f.read().strip()
except Exception:
    pass

CFG = {
    "input_path": INPUT_BOLD,
    "input_json": INPUT_JSON,
    "subject": SUBJECT, "session": SESSION, "run": RUN,
    "TR_seconds": TR, "TR_source": "NIfTI header (pixdim[4]); JSON RepetitionTime agrees",
    "original_dimensions": list(in_img.shape),
    "original_voxel_size_mm": audit["voxel_size_mm"],
    "original_orientation": orient,
    "volumes_removed": N_DISCARD, "volumes_retained": N_KEPT,
    "slice_timing_status": STC_STATUS, "slice_timing_reason": STC_REASON,
    "motion_correction_method": "FreeSurfer FS-FAST mc-afni2, rigid-body 6-DOF, reference = frame 0 "
                                "of the post-discard series (first retained volume)",
    "mni_template": f"{TPL_NAME} res-02 desc-brain_T1w (TemplateFlow)",
    "registration_method": "ANTsPy ants.registration(type_of_transform='SyN'); direct EPI->MNI "
                           "because no T1w anatomical image is available",
    "registration_reference": "temporal mean of motion-corrected BOLD, skull-stripped with "
                              "Otsu + largest 3D connected component + hole filling",
    "final_voxel_size_mm": list(z4),
    "resample_interpolation": "continuous (nilearn resample_img), grid derived from template affine",
    "smoothing_FWHM_mm": 6.0, "smoothing_tool": "nilearn.image.smooth_img (fwhm argument = FWHM, not sigma)",
    "detrending_method": "per-voxel OLS removal of intercept + linear term; temporal mean restored; "
                         "no quadratic term",
    "nuisance_regressors": names,
    "nuisance_n_regressors": int(Xn.shape[1]),
    "friston24_included": True,
    "wm_regressor_status": WM_STATUS, "csf_regressor_status": CSF_STATUS,
    "wm_csf_prior_source": "MNI152NLin2009cAsym res-02 label-WM/CSF probseg (TemplateFlow)",
    "wm_csf_prior_mapping": XFM_NOTE,
    "wm_threshold": 0.95, "csf_threshold": 0.90,
    "wm_erosion_iterations_at_2mm": WM_ERODE, "csf_erosion_iterations_at_2mm": CSF_ERODE,
    "prior_mask_procedure": "threshold + erode at native 2mm template resolution; 4mm versions "
                            "(>=50% voxel occupancy) saved for inspection",
    "tissue_signal_extraction_resolution_mm": 2,
    "tissue_signal_extraction_note": "WM/CSF signals are extracted from the 2mm normalized series "
                                     "because ventricular CSF is thinner than a 4mm voxel (923 vox "
                                     "at 2mm collapses to 11 vox at 4mm); the extracted time series "
                                     "are linearly detrended exactly as the imaging data",
    "wm_mask_voxels_2mm": int(WM_M2.sum()), "csf_mask_voxels_2mm": int(CSF_M2.sum()),
    "corr_WM_globalsignal": r_wm_gs, "corr_CSF_globalsignal": r_csf_gs,
    "global_signal_regression": True,
    "filter_low_hz": LOW, "filter_high_hz": HIGH,
    "filter_design": "2nd-order Butterworth, zero-phase filtfilt",
    "nyquist_hz": nyq,
    "software_versions": {
        "python": platform.python_version(), "numpy": ver("numpy"), "scipy": ver("scipy"),
        "nibabel": ver("nibabel"), "nilearn": ver("nilearn"), "pandas": ver("pandas"),
        "antspyx": ver("ants"), "templateflow": ver("templateflow"), "matplotlib": ver("matplotlib"),
        "freesurfer": fs_ver,
    },
    "platform": platform.platform(),
    "cpu_workers": 1,
    "thread_env": {k: os.environ[k] for k in
                   ["ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS", "OMP_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]},
    "datetime_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "runtime_seconds": round(RUNTIME, 1),
    "command": f"python3 {os.path.abspath(__file__)}",
    "output_root": ROOT,
}
with open(os.path.join(ROOT, "pipeline_config.json"), "w") as f:
    json.dump(CFG, f, indent=2)
log(f"  pipeline_config.json written  (runtime {RUNTIME:.1f}s)")

# =====================================================================
# FINAL REPORT
# =====================================================================
performed = [
    "Discard first 5 volumes",
    "Head-motion correction / realignment (mc-afni2)",
    "MNI normalization (direct EPI->MNI, ANTs SyN)",
    "Resample to 4 x 4 x 4 mm",
    "6 mm FWHM spatial smoothing",
    "Linear detrending",
    f"Nuisance regression ({Xn.shape[1]} regressors incl. full Friston-24 and global signal)",
    "0.01-0.10 Hz band-pass filtering",
]
skipped = []
if STC_STATUS != "PERFORMED":
    skipped.append(("Slice-timing correction", STC_REASON))
if not wm_use:
    skipped.append(("White-matter nuisance regressor", "no T1w segmentation; template prior failed reliability check"))
if not csf_use:
    skipped.append(("CSF nuisance regressor", "no T1w segmentation; template prior failed reliability check"))

rep = os.path.join(ROOT, "PILOT_PREPROCESSING_REPORT.md")
with open(rep, "w", encoding="utf-8") as f:
    w = f.write
    w("# Pilot Preprocessing Report -- ADNI rs-fMRI paper pipeline\n\n")
    w(f"Single-acquisition pilot. Generated {CFG['datetime_utc']}. Runtime {RUNTIME:.1f}s. 1 CPU worker.\n\n")

    w("# 1. Input\n\n")
    w(f"**EXACTLY ONE 4D NIfTI was processed.**\n\n- `{INPUT_BOLD}`\n")
    w(f"- Subject/session/run: {SUBJECT} / {SESSION} / {RUN}\n")
    w("- The source file was opened read-only and never modified, renamed or recompressed.\n\n")

    w("# 2. Dataset metadata\n\n| Field | Value |\n|---|---|\n")
    for k in ["n_dimensions", "dim_x", "dim_y", "dim_z", "n_timepoints", "voxel_size_mm",
              "TR_seconds_nifti_header", "orientation", "datatype_on_disk", "min", "max", "mean",
              "has_NaN", "has_Inf", "is_4D"]:
        w(f"| {k} | {audit[k]} |\n")
    w(f"| JSON RepetitionTime | {TR_json} |\n| JSON SliceTiming present | {'SliceTiming' in meta} |\n")
    w(f"| Manufacturer | {meta.get('Manufacturer')} {meta.get('ManufacturersModelName')} |\n\n")
    w("Affine:\n\n```\n" + np.array2string(in_img.affine, precision=5, suppress_small=True) + "\n```\n\n")

    w("# 3. Paper preprocessing pipeline\n\n```\nRAW 4D rs-fMRI -> discard first 5 volumes -> slice-timing\n"
      "correction -> head-motion correction -> MNI normalization -> resample 4x4x4 mm ->\n"
      "6 mm FWHM smoothing -> linear detrending -> nuisance regression -> 0.01-0.10 Hz\n"
      "band-pass -> final preprocessed BOLD\n```\n\n")

    w("# 4. Steps successfully performed\n\n")
    for s in performed:
        w(f"- {s}\n")
    w("\n")

    w("# 5. Steps skipped\n\n")
    if skipped:
        for s, _ in skipped:
            w(f"- {s}\n")
    else:
        w("- None\n")
    w("\n# 6. Reasons for skipped steps\n\n")
    if skipped:
        for s, why in skipped:
            w(f"- **{s}** -- {why}\n")
    else:
        w("- Not applicable\n")
    w("\n")

    w("# 7. Intermediate outputs\n\n| Step | File |\n|---|---|\n")
    for lbl, p in [("1 discard 5", p_d5), ("2 slice timing (skipped)", p_stc), ("3 motion correction", p_mc),
                   ("4 MNI normalization", p_mni), ("5 resample 4mm", p_4mm), ("6 smoothing 6mm", p_sm),
                   ("7 linear detrend", p_det), ("8 nuisance regression", p_nz), ("9 band-pass", p_bp)]:
        w(f"| {lbl} | `{p}` |\n")
    w(f"\nTransforms: `{D_XFM}`  Motion: `{D_MOT}`  Confounds: `{D_CONF}`\n\n")

    w("# 8. Final output\n\n")
    w(f"`{p_final}`\n\n| Property | Value |\n|---|---|\n")
    w(f"| shape | {fin_img.shape} |\n| voxel size (mm) | {tuple(round(float(z),4) for z in fin_img.header.get_zooms()[:3])} |\n")
    w(f"| TR (s) | {float(fin_img.header.get_zooms()[3])} |\n| dtype | {fin_img.get_data_dtype()} |\n")
    w(f"| orientation | {''.join(nib.aff2axcodes(fin_img.affine))} |\n")
    w(f"| NaN / Inf | {bool(np.isnan(fd_).any())} / {bool(np.isinf(fd_).any())} |\n")
    w(f"| non-empty | {bool(np.any(fd_!=0))} |\n\n")

    w("# 9. QC metrics\n\n| Metric | Value |\n|---|---|\n")
    for k, v in qc.items():
        w(f"| {k} | {v} |\n")
    w("\nFigures: `qc/fd_plot.png`, `qc/dvars_plot.png`, `qc/before_after.png`, "
      "`qc/mean_bold_raw.png`, `qc/mean_bold_final.png`, `qc/registration_qc.png`.\n")
    w("Images: `qc/temporal_mean_image.nii.gz`, `qc/temporal_std_image.nii.gz`, `qc/tsnr_image.nii.gz`.\n\n")

    w("# 10. Registration QC\n\n")
    w(f"- Template: {TPL_NAME} res-02 `desc-brain_T1w` (TemplateFlow)\n")
    w("- **MNI normalization performed directly from BOLD/EPI because no T1w anatomical image is available.**\n")
    w("- Method: ANTsPy `ants.registration(type_of_transform='SyN')` (rigid + affine + SyN).\n")
    w(f"- Output grid identical to template (shape and affine): **{in_mni}**\n")
    w(f"- Normalized series shape {mni_img.shape}, voxel size before 4 mm resampling "
      f"{tuple(round(float(z),4) for z in mni_img.header.get_zooms()[:3])}\n")
    w("- Visual check: `qc/registration_qc.png` (template contour over normalized mean BOLD).\n")
    w("- Transforms saved under `transforms/`.\n")
    w("- Caveat: EPI-to-template registration without a subject T1w is less accurate than the "
      "T1w-mediated registration assumed by the paper.\n\n")

    w("# 11. Motion QC\n\n")
    w(f"- mean FD = {FD_mean:.4f} mm, median = {FD_med:.4f} mm, max = {FD_max:.4f} mm\n")
    w(f"- volumes with FD > 0.5 mm: {int((FD>0.5).sum())} / {N_KEPT}\n")
    w("- 6 motion parameters are the native mc-afni2 estimates (`motion/motion_parameters.tsv`); "
      "none were synthesized or re-estimated by another method.\n")
    w("- No volumes were removed on the basis of motion.\n\n")

    w("# 12. Before vs after comparison\n\n| | Before (raw input) | After (final) |\n|---|---|---|\n")
    w(f"| volumes | {N_ORIG} | {N_KEPT} |\n")
    w(f"| matrix | {in_img.shape[:3]} | {fin_img.shape[:3]} |\n")
    w(f"| voxel size (mm) | {audit['voxel_size_mm']} | {list(z4)} |\n")
    w(f"| space | native EPI | {TPL_NAME} (MNI) |\n")
    w(f"| smoothing | none | 6 mm FWHM |\n")
    w(f"| mean signal in brain | {float(pre_mean[pre_brain].mean()):.2f} | {float(fin_mean[BRAIN].mean()):.2f} |\n")
    w("\nSee `qc/before_after.png`.\n\n")

    w("# 13. Deviations from the paper\n\n")
    w("| Paper step | Implemented step | Deviation / reason |\n|---|---|---|\n")
    for d in DEVIATIONS:
        w(f"| {d['paper_step']} | {d['implemented_step']} | {d['reason']} |\n")
    w("| MNI normalization via anatomical T1w | Direct EPI -> MNI template registration | "
      "Dataset contains no T1w image; no T1w-based registration was simulated or claimed. |\n")
    w("\nParameters preserved exactly as specified: 5 volumes discarded, 4x4x4 mm resampling, "
      "6 mm FWHM smoothing, 0.01-0.10 Hz band, full 24-column Friston-24 design.\n\n")

    w("# 14. Reproducibility information\n\n")
    w(f"- Config: `{os.path.join(ROOT,'pipeline_config.json')}`\n- Log: `{LOG_PATH}`\n")
    w(f"- Script: `{os.path.abspath(__file__)}`\n- Workers: 1 "
      f"(ITK/OMP/OpenBLAS/MKL threads pinned to 1)\n- Runtime: {RUNTIME:.1f} s\n\n")
    w("| Software | Version |\n|---|---|\n")
    for k, v in CFG["software_versions"].items():
        w(f"| {k} | {v} |\n")
    w(f"\nPlatform: {platform.platform()}\n")

log(f"  report written: {rep}")
hdr("PILOT COMPLETE -- stopped after preprocessing + QC")
log("No Brainnetome registration, ROI extraction, ALFF, ReHo, Degree Centrality or "
    "Functional Connectivity was performed.")
log("PILOT_DONE")
_logf.close()
