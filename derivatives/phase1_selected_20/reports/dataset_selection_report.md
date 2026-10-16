# Phase-1 Selected Dataset: 20 Acquisitions per Stage

Output: `derivatives/phase1_selected_20/`

Built by `scripts/data_prep/build_phase1_dataset.py` from acquisitions
that already had valid preprocessing and biomarker outputs. Nothing was
converted, preprocessed or re-extracted. Source trees were opened
read-only.

## 1. Source inventory

| Stage | Candidates | Valid unique | Distinct subjects |
|---|---:|---:|---:|
| CN | 27 | 27 | 26 |
| SMC | 31 | 31 | 21 |
| EMCI | 29 | 29 | 26 |
| MCI | 34 | 34 | 16 |
| LMCI | 21 | 21 | 20 |
| AD | 25 | 25 | 20 |

## 2. Classification

- `VALID_UNIQUE`: 167

No acquisition was invalid, missing biomarkers, or contained
non-finite values. No duplicate acquisition ids were found.

## 3. Selection

| Stage | Selected | Target | Distinct subjects | Repeat acquisitions |
|---|---:|---:|---:|---:|
| CN | 20 | 20 | 20 | 0 |
| SMC | 20 | 20 | 20 | 0 |
| EMCI | 20 | 20 | 20 | 0 |
| MCI | 16 | 20 | 16 | 0 |
| LMCI | 20 | 20 | 20 | 0 |
| AD | 20 | 20 | 20 | 0 |
| **Total** | **116** | **120** | **116** | **0** |

### Selection rule

```
Deterministic, label-blind selection. One acquisition per subject.

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
```

## 4. Verification

| Stage | Selected | Verified OK | Subjects |
|---|---:|---:|---:|
| CN | 20 | 20 | 20 |
| SMC | 20 | 20 | 20 |
| EMCI | 20 | 20 | 20 |
| MCI | 16 | 16 | 16 |
| LMCI | 20 | 20 | 20 |
| AD | 20 | 20 | 20 |

**Total verified: 116 / 120**

Each verified acquisition has: a preprocessed BOLD image, all four
normalised features (mALFF, mReHo, DC_z, Fisher-z FC), all raw
biomarker arrays, 246 ROI features, a 246x246 FC matrix, and no
NaN or infinite values.

No verification problems.

## 5. Copy integrity

- Files copied: 96
- Already present and identical: 5482
- Hash mismatches: 0
- Collisions not overwritten: 0
- Total bytes: 1,974,837,755

Every destination file's SHA-256 was recomputed after copying and
compared with its source. Mapping: `manifests/copy_mapping.csv`.

## 6. Notes

- `FC_Strength` is present in the copied `biomarkers/` trees for
  traceability but is **not** a model feature: it equals DC/245
  exactly (max difference 6.939e-18, r = 1.000000000000).
- Stage directories use canonical names; `CN_Final` and `SMC_Final`
  map to `CN` and `SMC`. Original group labels are retained in
  `manifests/acquisition_inventory.csv` as `source_group`.
- All 120 acquisitions come from 116 distinct
  subjects, not 120: see section 3.
- Selection used only subject/session/run ordering. No QC metric,
  biomarker value or stage label influenced it.
