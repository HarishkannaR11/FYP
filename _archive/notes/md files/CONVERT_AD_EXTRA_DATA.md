# Task: Convert ADNI AD_EXTRA_DATA DICOM Archives to NIfTI (Conversion Only)

This specification defines the conversion-and-validation pipeline tailored for the AD (Alzheimer's Disease) resting-state fMRI dataset located in `RAW_DATA_EXTRA/AD_EXTRA_DATA_35.zip`.

This is a **conversion-and-validation job only**. Do not preprocess, register, smooth, filter, parcellate, extract features, or train anything.

---

### Current Execution Status & Progress

* **Status**: **STOPPED / CANCELLED BY USER** (Raw data conversion cancelled; preprocessed data already available).
* **Script**: `scripts/data_prep/convert_ad_extra_data.py` (Removed per cleanup instruction).
* **Tool**: `dcm2niix` v1.0.20260724 (Specification retained for future reference).
* **Archive**: `RAW_DATA_EXTRA/` and `AD_EXTRA_DATA_35.zip` (Deleted per user instruction).
* **Outputs**: All temporary staging and intermediate NIfTI outputs removed.

---

### 1. Inputs

* **Archive Location**: `RAW_DATA_EXTRA/AD_EXTRA_DATA_35.zip` (2.01 GB, 243,840 DICOM files, 35 series)
  * *Important Rule*: Do not modify or delete this archive.
* **Internal Archive Layout**: Standard ADNI hierarchy:
  ```text
  ADNI/<SUBJECT_ID>/<SeriesDescription>/<YYYY-MM-DD_HH_MM_SS.0>/<ImageUID>/*.dcm
  ```
  *Examples*:
  * `ADNI/018_S_4696/Resting_State_fMRI/2012-11-01_11_17_01.0/I343571/*.dcm` (6,720 files, 140 volumes)
  * `ADNI/018_S_4696/Extended_Resting_State_fMRI/2014-04-29_11_06_39.0/I422892/*.dcm` (9,600 files, 200 volumes)

---

### 2. Series Selection & Audit Rules (AD Dataset)

* **Eligibility Criteria**:
  * `stage == "AD"`
  * `n_dicom >= 1000`
  * Valid series structure
* **Dataset Characteristics (Catalog of 35 Series)**:
  * Total series in archive: **35 series** across 15 AD subjects (`002_S_5018`, `006_S_4153`, `006_S_4192`, `006_S_4867`, `013_S_5071`, `018_S_4696`, `018_S_4733`, `019_S_4252`, `019_S_4477`, `019_S_4549`, `019_S_5012`, `019_S_5019`, `053_S_5070`, `130_S_4641`, `130_S_4660`, `130_S_4730`, `130_S_4971`, `130_S_4982`, `130_S_4984`, `130_S_4990`, `130_S_5006`, `130_S_5059`).
  * **32 standard series**: 6,720 DICOM files each ($6720 / 48 = 140$ volumes).
  * **3 extended series**: 9,600 DICOM files each ($9600 / 48 = 200$ volumes).
  * Total DICOM count: $32 \times 6720 + 3 \times 9600 = 243,840$ files.
* **Suspicious Check**:
  * Mark a series `SUSPICIOUS` if `n_dicom / 48` is not a whole number.
  * In this AD archive, all 35 series evaluate to exact whole numbers (140 or 200 volumes). If any series deviates during processing, convert it but flag it with status `SUSPICIOUS` rather than plain `OK`.
* **Duplicates & Multi-series Handling**:
  * Note: Subject `019_S_4252` contains two distinct series acquired at the same timestamp (`2013-10-17_12_37_35.0`): `I396527` and `I396530`. Both must be differentiated by image UID / series number and checked for filename collisions.

---

### 3. Tool Specifications

* **Tool**: `dcm2niix`
  * Version check: `dcm2niix -v`. Required version is verified before batch processing.
* **Invocation Pattern**:
  ```bash
  dcm2niix -z y -b y -f "%n_%d_%t_%s" -o "<OUT_DIR>" "<EXTRACTED_SERIES_DIR>"
  ```
  * `-z y`: Gzip compressed output (`.nii.gz`)
  * `-b y`: Generate BIDS JSON sidecar (`.json`)
  * `-f "%n_%d_%t_%s"`: Output filename format containing subject/patient name, series description, acquisition time, and series number.

---

### 4. Disk Management & Sequencing

* **Strict Rule**: **Do NOT extract the archive wholesale** (avoid inflating ~10 GB of loose DICOM files at once).
* **Sequential Processing (Single Worker)**:
  1. Inspect archive metadata and pick the next series.
  2. Extract only that series' DICOMs into a temporary staging directory (`_staging/AD/...`).
  3. Execute `dcm2niix` on the staging folder.
  4. Move generated `.nii.gz` and `.json` outputs to destination.
  5. Run validation checks on the produced files.
  6. Clean up and delete the temporary staging directory.
  7. Append record to `conversion_state.jsonl`.
  8. Proceed to next series.
  * No multiprocessing or threading.

---

### 5. Output Layout

* Target directory structure:
  ```text
  NIfTI_extra/AD/<ADNI_SUBJECT>/<dcm2niix-generated name>.nii.gz
  NIfTI_extra/AD/<ADNI_SUBJECT>/<dcm2niix-generated name>.json
  ```
* **Collision Policy**:
  * Never overwrite an existing file.
  * If a target filename already exists, log it as a collision and skip overwriting.

---

### 6. Acquisition Length & Volume Retention

* Full acquisition length must be preserved. **Do NOT truncate**.
* Standard series: 140 volumes.
* Extended protocol: 200 volumes.
* Native volume count must be verified and logged for every series.

---

### 7. Validation Checklist (Every Output)

Every converted NIfTI file must pass the following validation steps:
1. **Readable**: Loads without error using `nibabel`.
2. **Dimensionality**: Is 4D with time dimension (4th dimension) $\ge 2$.
3. **Voxel Size**: Voxel sizes are finite, positive, and within range 0.5–8.0 mm.
4. **Affine Matrix**: Affine matrix is finite and non-singular ($\det(\text{affine}) \neq 0$).
5. **Data Validity**: Data array is non-empty, contains no NaNs or Infs, and is not constant (std > 0).
6. **JSON Sidecar**: Parses cleanly as JSON and contains `RepetitionTime`.
7. **Volume Count Consistency**: Number of volumes matches expected `n_dicom / 48`.
8. **Logging Warnings**: All `stdout` and `stderr` output from `dcm2niix` captured verbatim.
* *Do not fabricate or interpolate missing metadata* (e.g. `SliceTiming` is absent in this dataset and must not be synthesized).

---

### 8. Resumability & State Tracking

* Maintain state log at `conversion_state.jsonl` (one JSON line per series):
  ```json
  {
    "stage": "AD",
    "adni_subject": "<SUBJECT>",
    "image_uid": "<IMAGE_UID>",
    "description": "<DESCRIPTION>",
    "acq_datetime": "<DATETIME>",
    "n_dicom_listed": 6720,
    "n_dicom_extracted": 6720,
    "expected_volumes": 140,
    "suspicious": false,
    "dcm2niix_returncode": 0,
    "dcm2niix_warnings": [],
    "outputs": ["NIfTI_extra/AD/..."],
    "collisions": [],
    "validation": { ... },
    "status": "SUCCESS",
    "seconds": 12.4
  }
  ```
* **Resumability logic**:
  * On restart, skip a series only if its recorded status in `conversion_state.jsonl` is successful **AND** its output files actually exist on disk.

---

### 9. Final Reporting

At pipeline completion, output summary statistics:
* Total series found
* Successful conversions
* Failed conversions (with reasons)
* Suspicious series flagged
* Total volumes converted across all series
* Detailed list of output files
