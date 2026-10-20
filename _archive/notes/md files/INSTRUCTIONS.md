# Repository Instructions for AI Assistant Sessions

> **Note for AI Assistant**: Read this file at the start of every session when requested by the user. These instructions are mandatory across all operations, data processing, pipelines, audits, and analysis tasks in this repository.

---

## Core Operational Principles

### 1. Mandatory Task-Specific Markdown (`.md`) Creation
* **Every process gets an MD document**: Whenever any workflow, conversion pipeline, data preparation, audit, or analysis task is initiated, the assistant **must immediately create a dedicated markdown file** inside the `md files/` directory.
* **Customize to Actual Folder Structure**: The documentation must never use hypothetical or generic paths. It must reflect the user's exact file and folder structure, archive names, and dataset specifications.
* **Location Rule**: All process documentation, operational runbooks, and task tracking files **must reside strictly inside the `md files/` folder**.

---

### 2. Real-Time Simultaneous Updates
* **Keep documentation live**: The assistant must **simultaneously update** the relevant markdown file whenever:
  * A series, subject, or batch completes.
  * A tool version or dependency is verified or changed.
  * A script or configuration is created or modified.
  * An error, collision, or suspicious condition is detected.
  * Any execution milestone is reached.
* **Progress Logging**: Every running process must maintain an explicit, live progress table or checklist in its corresponding `.md` file with timestamps, metrics, and outcomes.

---

### 3. Pipeline & Data Conversion Integrity Standards
* **Conversion Only Boundary**: When instructed to perform conversion/validation jobs, strictly limit work to conversion and validation. Do **not** apply preprocessing, smoothing, filtering, parcellation, registration, or feature extraction unless explicitly requested.
* **Zero Archive Tampering**: Never modify, delete, or recompress raw archives (`.zip`, `.tar.gz`, etc.).
* **Non-Destructive Extraction**:
  * Never extract multi-gigabyte archives wholesale to avoid disk overflow.
  * Extract sequentially, one series or subject at a time, to a designated temporary staging directory (`_staging/...`).
  * Process, validate, move outputs, clean up staging, and then proceed to the next unit.
* **Collision Safety**: Never overwrite existing output files. If an output filename already exists, log it as a collision and skip overwriting.
* **Full Acquisition Retention**: Never truncate temporal dimensions or volume counts without explicit instructions. Both standard (e.g. 140 vols) and extended protocols (e.g. 200 vols) must be preserved in their entirety.

---

### 4. Tool & Environment Verification
* **Exact Tool Versions**: Always check and verify tool versions (e.g. `dcm2niix -v`) prior to batch processing. Stop immediately and report if versions do not match the expected specification.
* **Deterministic Logging**: Always maintain a structured JSON lines file (e.g. `conversion_state.jsonl`) recording per-item metadata, return codes, verbatim warnings, output file paths, and execution times.
* **Resumability Verification**: On session start or pipeline restart, skip an item **only** if its logged status is successful **AND** its output files actually exist on disk. Never skip based solely on log records.

---

### 5. Standard Verification Checklist for NIfTI Outputs
Every NIfTI file generated must be verified against:
1. Readable by `nibabel`.
2. Verified 4D with temporal dimension $\ge 2$.
3. Spatial voxel sizes are finite, positive, and in a physiological range (0.5–8.0 mm).
4. Affine matrix is finite and non-singular ($\det(\text{affine}) \neq 0$).
5. Data array contains no NaNs or Infs, and is not constant ($\sigma > 0$).
6. Companion BIDS JSON sidecar parses validly and contains mandatory fields (`RepetitionTime`).
7. Volume count matches expected DICOM calculation (`n_dicom / slices_per_volume`).
8. No fabricated metadata (e.g. never invent `SliceTiming` when genuinely absent).

---

### 6. Starting a New Session
When the user starts a new conversation and says:
> *"Read the instruction file"* or *"Follow the session instructions"*

The AI Assistant must:
1. Open and review `md files/INSTRUCTIONS.md`.
2. Inspect the `md files/` folder for existing process files to understand active or previous workflows.
3. Check state files (e.g. `conversion_state.jsonl`) and verify disk outputs to resume seamlessly without duplicating work.
4. Confirm readiness and present the current status to the user.
