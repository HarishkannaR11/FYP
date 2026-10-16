# Reproducibility

Only commands verified against this repository appear below. Preprocessing
and feature extraction were **not** re-executed to produce this document.

---

## 1. Environment

Two environments are used, deliberately:

| Runs on | Used for |
|---|---|
| **Windows** (Python 3.13.3) | statistics, heatmaps, cohort selection, model code, audits |
| **WSL2 Ubuntu** (Python 3.12.3, venv) | DICOM→NIfTI, preprocessing, biomarker extraction, normalisation |

The neuroimaging stack (FreeSurfer, ANTsPy, nipype) lives only in WSL. Several
scripts hardcode `/mnt/c/...` paths and **will silently discover zero inputs
if run from Windows** — `normalize_within_subject.py` is the known case.

### WSL environment

```bash
FREESURFER_HOME=/home/harish/freesurfer
FSFAST_HOME=/home/harish/freesurfer/fsfast
SUBJECTS_DIR=$FREESURFER_HOME/subjects
source $FREESURFER_HOME/SetUpFreeSurfer.sh
source ~/fyp-neuro-env/bin/activate
```

Verified versions in that venv:

```
nipype 1.12.0 · antspyx 0.6.3 · nibabel 5.4.2
nilearn 0.14.1 · numpy 2.3.5 · scipy 1.15.3 · templateflow 25.1.2
```

These match the `software_versions` recorded in each acquisition's
provenance JSON.

### Windows environment

```
python 3.13.3 · numpy 2.2.2 · scipy · pandas 2.3.3
nibabel 5.4.2 · matplotlib · torch 2.11.0+cpu · pytest 9.0.3
```

### External tools

| Tool | Location | Version |
|---|---|---|
| dcm2niix | `tools/dcm2niix_win/dcm2niix.exe` | **v1.0.20260724** |
| FreeSurfer / FS-FAST | `/home/harish/freesurfer` (WSL) | 7.4.1 |
| `mc-afni2` | `$FSFAST_HOME/bin/mc-afni2` | — |

**Use exactly dcm2niix v1.0.20260724.** A different version changes the JSON
sidecars and breaks consistency with the existing 167 acquisitions.

### Paths

| Item | Path |
|---|---|
| Project root | `C:\Users\krish\FYP` (`/mnt/c/Users/krish/FYP` in WSL) |
| Atlas | `atlases/Brainnetome246/BN_Atlas_246_2mm.nii.gz` |
| Atlas LUT | `atlases/Brainnetome246/BN_Atlas_246_LUT.txt` |
| BIDS | `BIDS/sub-*/ses-*/func/` |
| Preprocessed | `derivatives/fsfast/<GROUP>/<sub>/<ses>/<run>/` |
| WSL scratch | `/home/harish/fyp_work/fsfast_4mm6mm` |

Template files are fetched by **TemplateFlow** on first use and cached at
`~/.cache/templateflow`.

---

## 2. Resource constraints

These are not optional — they were the cause of repeated failures.

- **One worker everywhere.** `n_procs = 1`; no multiprocessing, threading or
  GPU parallelism. Set `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
  MKL_NUM_THREADS=1 ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS=1`.
- **Per-acquisition processing on the WSL-native filesystem**, copying only
  final outputs to `/mnt/c`. Sustained small-granularity array I/O against
  DrvFs produces `[Errno 5] Input/output error`.
- **Windows commit charge** has repeatedly reached 95–99%, causing
  `MemoryError` on 1 MB reads, `ModuleNotFoundError` on pandas, `uv_spawn`
  failures and a dcm2niix access violation (`0xC0000005`). WSL has its own
  memory pool and is largely unaffected.
- **Disk**: the wrappers refuse to start below 900 MB free WSL memory or
  2 GB free on C:.

---

## 3. Commands

### Audit the dataset

```bash
# Windows
python scripts/audit/final_6group_audit.py        # regenerates FINAL_6GROUP_AUDIT/
```

Note: this parses failure reasons from `derivatives/fsfast/<GROUP>/logs/
preprocessing.log`. If a log has been truncated, failures will be
under-reported — `PROCESSING_CHECKPOINT.json` is authoritative.

### Inventory a new DICOM archive (dry run, extracts nothing)

```bash
python scripts/data_prep/stage_dicom_inventory.py \
    --stage MCI --archive MCI_ersfMRI.zip --target 20
```

Standard-library only, so it runs under memory pressure when pandas cannot
import.

### Convert DICOM → NIfTI

```bash
python scripts/data_prep/convert_extra_dicom.py \
    --stages MCI \
    --dedup derivatives/EXTRA_DATA_INVENTORY/extra_series_dedup_MCI.csv \
    --state derivatives/EXTRA_DATA_INVENTORY/conversion_state_MCI.jsonl
```

One series at a time: extract → convert → validate → delete staging. Peak
disk ≈ 150 MB rather than the full archive. Resumable; re-validates outputs
on disk before skipping.

### BIDS-ify new acquisitions

```bash
python scripts/data_prep/bidsify_new_acquisitions.py --dry-run
python scripts/data_prep/bidsify_new_acquisitions.py --apply
```

Copies (never moves), never overwrites, appends to `participants.tsv` only
for new subjects.

### Preprocess

```bash
# WSL, via the resilient wrapper
bash scripts/setup/retry_production_4mm6mm.sh          # all six groups
bash scripts/setup/retry_mci_additions_4mm6mm.sh       # additions only
```

Direct invocation:

```bash
python3 pipeline/fsfast_production_4mm6mm.py <GROUP>
# GROUP in: AD CN_Final EMCI LMCI MCI SMC_Final
```

Optional scoping to an additions manifest, leaving defaults intact:

```bash
export FSFAST_MANIFEST_OVERRIDE=mci_additions/acquisition_manifest_final.csv
export FSFAST_EXPECTED_N=2
```

Without those two variables the script reproduces the original production run
exactly.

### Extract biomarkers (WSL)

```bash
python3 scripts/biomarkers/biomarker_dataset_pipeline.py discover   # dry run
python3 scripts/biomarkers/biomarker_dataset_pipeline.py run
python3 scripts/biomarkers/biomarker_dataset_pipeline.py audit
```

Eligibility comes from `PROCESSING_CHECKPOINT.json`, not from directory
existence.

### Normalise (WSL only)

```bash
python3 scripts/biomarkers/normalize_within_subject.py
```

Running this from Windows discovers **0 acquisitions** because of hardcoded
`/mnt/c` paths. It reports the discrepancy rather than failing silently.

### Build the cohort and model-ready arrays

```bash
python scripts/data_prep/build_phase1_dataset.py            # audit + plan
python scripts/data_prep/build_phase1_dataset.py --copy     # execute
python scripts/model/build_transformer_inputs.py
```

### Statistics and heatmaps

```bash
V=derivatives/INTRA_INTER_CLASS_HEATMAPS/v2_cohort116

python scripts/audit/one_vs_rest_checkpointed.py $V --verify-only   # prove the
                                                                   # fast statistic
python scripts/audit/one_vs_rest_checkpointed.py $V                 # 24 tests
python scripts/audit/plot_overall_significance.py $V                # 4x6 figure
python scripts/audit/roi_stage_significance.py $V                   # 246x6 figures
python scripts/audit/pairwise_stage_statistics.py $V/pairwise       # 246x15
python scripts/audit/analyze_intra_inter_class.py                   # intra/inter
```

### Model (architecture only — nothing is trained)

```bash
python -m pytest scripts/model/test_multi_attention_transformer.py -v
python scripts/model/multi_attention_transformer.py       # smoke test
python scripts/model/smoke_test_real_data.py              # real cohort features
```

---

## 4. Seeds and permutation configuration

| Analysis | Seed | Permutations |
|---|---|---:|
| One-vs-rest multivariate (4×6) | `20261013` | 9,999 |
| Intra- vs inter-class | `0` | 5,000 |
| Reliability null sampling | `0` | 3,000 |

The one-vs-rest script creates a **fresh RNG per test**, so any single test is
reproducible from (seed, biomarker, stage) alone — required for checkpointing
to be meaningful. Non-permutation analyses are deterministic.

With N = 9,999 and the `+1` convention, the smallest attainable p is
**1 × 10⁻⁴**.

---

## 5. Expected outputs

```
derivatives/
├── fsfast/<GROUP>/<sub>/<ses>/<run>/        preprocessed BOLD, QC, biomarkers
├── fsfast/FINAL_6GROUP_AUDIT/               9 CSVs + report
├── fsfast/PROCESSING_CHECKPOINT.json        authoritative completed/failed
├── biomarkers_normalized/<GROUP>/...        mALFF, mReHo, DC_z, Fisher-z FC
├── clinical_cleaned/                        master table + per-stage + audit
├── phase1_selected_20/                      116-subject cohort + manifests
├── model_ready/                             X_regional, X_fc, y, subjects
└── INTRA_INTER_CLASS_HEATMAPS/v2_cohort116/
    ├── figures/                             5 PNG + 5 PDF
    ├── pairwise/                            18 CSVs, 30 FC matrices, 8 figures
    ├── one_vs_rest_*.csv                    statistics, p, q
    ├── one_vs_rest_checkpoint.jsonl         resumable state
    ├── *_roi_stage_*.csv                    16 CSVs
    ├── HEATMAP_README.md                    corrected
    ├── HEATMAP_README.v1_uncorrected.md     preserved original
    └── FINAL_STATISTICAL_AUDIT.md
```

---

## 6. Validating outputs

```bash
# cohort is exactly 116 subjects, one acquisition each
python -c "
import numpy as np
s=np.load('derivatives/model_ready/subjects.npy'); y=np.load('derivatives/model_ready/y.npy')
print(len(s), len(set(s)), 'one-per-subject:', len(s)==len(set(s)))
print({k:int((y==i).sum()) for i,k in enumerate(['CN','SMC','EMCI','MCI','LMCI','AD'])})"
```

```bash
# all 24 tests completed, one cohort, one seed
python -c "
import csv
r=list(csv.DictReader(open('derivatives/INTRA_INTER_CLASS_HEATMAPS/v2_cohort116/one_vs_rest_statistics.csv')))
print(len(r), all(x['status']=='COMPLETED' for x in r),
      {x['cohort_fingerprint'] for x in r}, {x['seed'] for x in r})"
```

Expected: `116 116 one-per-subject: True`, and `24 True
{'6ab68ae6e80f61c2'} {'20261013'}`.

Verify figures open and are non-empty by checking the PNG magic bytes
(`\x89PNG`) and PDF header (`%PDF-`); all 10 were confirmed valid.

---

## 7. Common errors and recovery

| Symptom | Cause | Recovery |
|---|---|---|
| `discovered 0 acquisitions` | WSL-path script run from Windows | run it inside WSL |
| `ModuleNotFoundError: nilearn` | Windows Python lacks the neuro stack | run inside the WSL venv |
| `EOFError: Ran out of input` from nipype | corrupted cache pickle | wrappers detect and delete, then resume |
| `[Errno 5] Input/output error` | DrvFs write truncation on `/mnt/c` | stage on WSL-native disk |
| dcm2niix returns `3221225477` | access violation under low memory | free memory and retry; same data converts cleanly |
| `MemoryError` on small reads | Windows commit charge ≈ 99% | close applications; WSL jobs are unaffected |
| `uv_spawn` failure | out of commit charge | free memory before launching |
| `ast.parse` / `SyntaxError` after a scripted patch | patch wrote a literal `\n` | restore from `.bak` and re-apply |

Wrappers in `scripts/setup/` distinguish a **Python traceback** (stop for
review) from a **WSL crash** (retry), and treat `SyntaxError`,
`ImportError` and `AssertionError` as fatal rather than retryable.

---

## 8. What was not re-run for this documentation

DICOM conversion, preprocessing, biomarker extraction and normalisation were
**not** re-executed. All figures quoted here are the existing outputs. No
dataset, feature, statistical result, checkpoint or figure was modified while
writing these documents.
