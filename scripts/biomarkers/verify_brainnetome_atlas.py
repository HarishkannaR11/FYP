"""
RECOVERED from session history (original location test/biomarkers/sub-019S4549/
was deleted in a cleanup pass before this preservation step could copy it
directly; recreated verbatim from conversation context, not fabricated).

Part 2 ONLY: verify the manually-downloaded Brainnetome-246 atlas.
Does NOT perform registration, ROI extraction, or any biomarker computation.
Does NOT modify the original downloaded atlas file.
"""
import os
import re
import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ATLAS_PATH = "/mnt/c/Users/krish/FYP/atlases/Brainnetome246/BN_Atlas_246_2mm.nii.gz"
LUT_PATH = "/mnt/c/Users/krish/FYP/atlases/Brainnetome246/BN_Atlas_246_LUT.txt"
MNI_TEMPLATE_PATH = "/mnt/c/Users/krish/FYP/test/biomarkers/sub-019S4549/registration/MNI152_T1_2mm_template.nii.gz"

OUT_DIR = "/mnt/c/Users/krish/FYP/test/biomarkers/sub-019S4549"
REPORT_PATH = os.path.join(OUT_DIR, "brainnetome_atlas_verification.txt")
QC_PNG = os.path.join(OUT_DIR, "brainnetome_atlas_qc.png")

lines = []
results = {}


def log(msg=""):
    print(msg)
    lines.append(str(msg))


def main():
    log("=== BRAINNETOME-246 ATLAS VERIFICATION (Part 2 ONLY) ===\n")

    log("--- 1. Atlas file existence/readability ---")
    log(f"Atlas path: {ATLAS_PATH}")
    atlas_ok = os.path.exists(ATLAS_PATH)
    log(f"Exists: {atlas_ok}")
    if not atlas_ok:
        log("STOP: atlas file not found.")
        results["atlas_file"] = "FAIL"
        write_report()
        return
    try:
        img = nib.load(ATLAS_PATH)
        data = img.get_fdata()
        log("Readable via NiBabel: YES")
        results["atlas_file"] = "PASS"
    except Exception as e:
        log(f"Readable via NiBabel: NO -- {e}")
        results["atlas_file"] = "FAIL"
        write_report()
        return
    log("")

    log("--- 2-5. Header information ---")
    shape = img.shape
    zooms = img.header.get_zooms()
    affine = img.affine
    orient = "".join(nib.aff2axcodes(affine))
    log(f"Dimensions: {shape}")
    log(f"Voxel size: {zooms}")
    log(f"Affine:\n{affine}")
    log(f"Orientation: {orient}\n")

    log("--- 6-8. ROI label analysis (NOT just max(label)) ---")
    unique_labels = np.unique(data)
    nonzero_labels = unique_labels[unique_labels != 0]
    nonzero_labels_int = sorted(int(round(v)) for v in nonzero_labels)
    n_unique_nonzero = len(nonzero_labels_int)
    max_label = max(nonzero_labels_int) if nonzero_labels_int else 0
    min_label = min(nonzero_labels_int) if nonzero_labels_int else 0

    log(f"Number of DISTINCT nonzero label VALUES actually present in the volume: {n_unique_nonzero}")
    log(f"Min label value present: {min_label}   Max label value present: {max_label}")
    log(f"max(label) == 246: {max_label == 246}  (NOTE: this alone is NOT sufficient evidence of 246 ROIs)")

    expected_ids = set(range(1, 247))
    present_ids = set(nonzero_labels_int)
    missing_from_volume = sorted(expected_ids - present_ids)
    unexpected_in_volume = sorted(present_ids - expected_ids)

    log(f"IDs 1-246 present in volume but missing (0 voxels for that label): {missing_from_volume}")
    log(f"IDs present in volume outside the expected 1-246 range: {unexpected_in_volume}")

    rois_246_confirmed = (n_unique_nonzero == 246 and not missing_from_volume and not unexpected_in_volume)
    log(f"\nEXACTLY 246 valid ROI labels actually present as nonzero voxels: {rois_246_confirmed}")
    results["246_rois"] = "PASS" if rois_246_confirmed else "FAIL"
    log("")

    label_voxel_counts = {}
    for lbl in nonzero_labels_int:
        label_voxel_counts[lbl] = int((data == lbl).sum())
    tiny_rois = [(lbl, c) for lbl, c in label_voxel_counts.items() if c < 5]
    log(f"ROIs with fewer than 5 voxels (worth flagging, not necessarily invalid): "
        f"{tiny_rois if tiny_rois else 'none'}\n")

    log("--- 9-12. LUT (label lookup table) verification ---")
    log(f"LUT path: {LUT_PATH}")
    lut_exists = os.path.exists(LUT_PATH)
    log(f"Exists: {lut_exists}")
    if not lut_exists:
        log("STOP: LUT file not found.")
        results["lut_file"] = "FAIL"
        write_report()
        return

    lut_entries = {}
    duplicate_ids = []
    malformed_lines = []
    with open(LUT_PATH) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            parts = re.split(r"\s+", line)
            if len(parts) < 2:
                malformed_lines.append((lineno, line))
                continue
            try:
                roi_id = int(parts[0])
            except ValueError:
                malformed_lines.append((lineno, line))
                continue
            roi_label = parts[1]
            if roi_id in lut_entries:
                duplicate_ids.append(roi_id)
            lut_entries[roi_id] = roi_label

    log(f"Total LUT entries parsed: {len(lut_entries)}")
    log(f"Malformed lines: {malformed_lines if malformed_lines else 'none'}")
    log(f"Duplicate ROI IDs in LUT: {duplicate_ids if duplicate_ids else 'none'}")
    results["lut_file"] = "PASS" if not malformed_lines and not duplicate_ids else "FAIL"

    lut_ids_excl_bg = set(lut_entries.keys()) - {0}
    log(f"LUT ROI IDs (excluding ID 0/background): {len(lut_ids_excl_bg)} entries")
    log(f"LUT ROI ID range: {min(lut_ids_excl_bg) if lut_ids_excl_bg else None} - "
        f"{max(lut_ids_excl_bg) if lut_ids_excl_bg else None}")

    missing_from_lut = sorted(expected_ids - lut_ids_excl_bg)
    extra_in_lut = sorted(lut_ids_excl_bg - expected_ids)
    log(f"IDs 1-246 missing from LUT: {missing_from_lut if missing_from_lut else 'none'}")
    log(f"LUT IDs outside expected 1-246 range (excl. background 0): {extra_in_lut if extra_in_lut else 'none'}")
    log("")

    log("--- LUT <-> atlas volume consistency ---")
    ids_in_volume_not_in_lut = sorted(present_ids - lut_ids_excl_bg)
    ids_in_lut_not_in_volume = sorted(lut_ids_excl_bg - present_ids)
    log(f"ROI IDs present in atlas VOLUME but missing from LUT: "
        f"{ids_in_volume_not_in_lut if ids_in_volume_not_in_lut else 'none'}")
    log(f"ROI IDs present in LUT but with ZERO voxels in atlas volume: "
        f"{ids_in_lut_not_in_volume if ids_in_lut_not_in_volume else 'none'}")

    lut_atlas_consistent = not ids_in_volume_not_in_lut and not ids_in_lut_not_in_volume
    log(f"LUT <-> atlas fully consistent: {lut_atlas_consistent}\n")
    results["lut_atlas_consistency"] = "PASS" if lut_atlas_consistent else "FAIL"

    log("Sample ROI ID -> label mappings (first 5, for manual sanity check):")
    for rid in sorted(lut_entries.keys())[:6]:
        log(f"  {rid}: {lut_entries[rid]}")
    log("")

    log("--- 13-14. MNI space / template compatibility ---")
    log("Brainnetome atlas is documented by its publisher (atlas.brainnetome.org) as distributed")
    log("in MNI152 space. This verification checks header GEOMETRY compatibility with the MNI")
    log("template used in this pilot -- it does NOT independently re-derive/confirm true")
    log("anatomical MNI correspondence beyond trusting the publisher's stated space.\n")

    mni_available = os.path.exists(MNI_TEMPLATE_PATH)
    if mni_available:
        mni_img = nib.load(MNI_TEMPLATE_PATH)
        mni_shape = mni_img.shape
        mni_zooms = mni_img.header.get_zooms()
        mni_affine = mni_img.affine
        mni_orient = "".join(nib.aff2axcodes(mni_affine))

        log(f"MNI template (from prior pilot step): {MNI_TEMPLATE_PATH}")
        log(f"  dims={mni_shape} voxel={mni_zooms} orientation={mni_orient}")
        log(f"Brainnetome atlas: dims={shape} voxel={zooms} orientation={orient}\n")

        dims_match = tuple(shape) == tuple(mni_shape)
        voxel_match = tuple(round(float(v), 2) for v in zooms) == tuple(round(float(v), 2) for v in mni_zooms)
        orient_match = orient == mni_orient
        affine_close = np.allclose(affine, mni_affine, atol=0.5)

        log(f"Dimensions match MNI template exactly: {dims_match}")
        log(f"Voxel size matches MNI template: {voxel_match}")
        log(f"Orientation matches MNI template: {orient_match}")
        log(f"Affine matrices close (atol=0.5mm): {affine_close}")

        compatible = dims_match and voxel_match and orient_match and affine_close
        log(f"\nDIRECTLY grid-compatible with MNI template (no resampling needed): {compatible}")
        results["mni_space"] = "PASS" if orient_match else "UNCERTAIN"
        results["mni_template_compat"] = "PASS" if compatible else "FAIL"
    else:
        log(f"MNI template file from prior step not found at {MNI_TEMPLATE_PATH} -- cannot directly compare.")
        results["mni_space"] = "UNCERTAIN"
        results["mni_template_compat"] = "FAIL"
        compatible = False
    log("")

    log("--- 15. Interpolation requirement if resampling is ever needed ---")
    log("The atlas contains DISCRETE integer ROI labels (categorical data), not continuous")
    log("intensities. If resampling to a different grid is ever required, NEAREST-NEIGHBOR")
    log("interpolation MUST be used -- linear/cubic interpolation would blend adjacent ROI IDs")
    log("into invalid intermediate label values and corrupt the parcellation.\n")

    log("--- QC image ---")
    x, y, z = shape
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    views = [
        (data[x // 2, :, :].T, "sagittal"),
        (data[:, y // 2, :].T, "coronal"),
        (data[:, :, z // 2].T, "axial"),
    ]
    for ax, (sl, name) in zip(axes, views):
        ax.imshow(sl, cmap="nipy_spectral", origin="lower", interpolation="nearest")
        ax.set_title(f"Brainnetome-246 ({name}, label colormap)", fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(QC_PNG, dpi=150)
    log(f"Saved: {QC_PNG}\n")

    overall = all(v == "PASS" for v in [results.get("atlas_file"), results.get("lut_file"),
                                          results.get("246_rois"), results.get("lut_atlas_consistency"),
                                          results.get("mni_template_compat")])
    results["overall"] = "PASS" if overall else "FAIL"

    log("=== FINAL VERIFICATION RESULTS ===")
    log(f"Atlas file: {results.get('atlas_file')}")
    log(f"LUT file: {results.get('lut_file')}")
    log(f"246 ROIs confirmed: {results.get('246_rois')}")
    log(f"LUT <-> atlas consistency: {results.get('lut_atlas_consistency')}")
    log(f"MNI space: {results.get('mni_space')}")
    log(f"MNI template compatibility: {results.get('mni_template_compat')}")
    log(f"Overall atlas verification: {results.get('overall')}")

    write_report()


def write_report():
    with open(REPORT_PATH, "w") as f:
        f.write("\n".join(lines))
    print(f"\nReport written to {REPORT_PATH}")


if __name__ == "__main__":
    main()
