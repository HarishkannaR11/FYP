# Data Selection

How 175 raw ADNI acquisitions became a 116-subject analysis cohort, and why
each subject was kept or dropped.

Every count was verified against the files in this repository on 2026-10-20.
Nothing here is estimated.

---

## 1. The full chain

```
                      ADNI / LONI Image & Data Archive
                  resting-state fMRI only — no T1w structural
                                   │
                                   ▼
                        ┌─────────────────────┐
                        │  175 acquisitions   │  raw, audited
                        └──────────┬──────────┘
                                   │
                    ─ 2            │  PRE-EXCLUSION
                                   │  truncated scan, byte-identical duplicate
                                   ▼
                        ┌─────────────────────┐
                        │  173 eligible       │
                        └──────────┬──────────┘
                                   │
                    ─ 8            │  PREPROCESSING FAILURE
                                   │  TR 6.02 s → Nyquist 0.083 Hz,
                                   │  below the 0.10 Hz band-pass cutoff
                                   ▼
                        ┌─────────────────────┐
                        │  165 preprocessed   │
                        └──────────┬──────────┘
                                   │
                    + 2            │  LATER ADDITION
                                   │  two MCI subjects, separate download
                                   ▼
                        ┌─────────────────────┐
                        │  167 acquisitions   │
                        │  129 distinct       │  ADNI is longitudinal:
                        │      subjects       │  many subjects, several visits
                        └──────────┬──────────┘
                                   │
                      0 invalid    │  BIOMARKER VALIDATION
                      0 missing    │  all 167 pass
                      0 non-finite │
                      0 duplicate  │
                                   ▼
                        ┌─────────────────────┐
                        │  167 VALID_UNIQUE   │
                        └──────────┬──────────┘
                                   │
                   ─ 51            │  COHORT SELECTION
                                   │  one acquisition per subject,
                                   │  capped at 20 per stage
                                   ▼
        ┌──────────────────────────────────────────────────────┐
        │   CN 20   SMC 20   EMCI 20   MCI 16   LMCI 20  AD 20  │
        │                   116 subjects                        │
        │        1 acquisition each · 0 repeat acquisitions     │
        └──────────────────────────┬───────────────────────────┘
                                   │
                  ┌────────────────┴────────────────┐
                  ▼                                 ▼
        ┌──────────────────┐              ┌──────────────────┐
        │  train pool 81   │              │   test 35        │
        │  5 stratified    │              │   LOCKED         │
        │  folds, 16–17 ea │              │                  │
        └──────────────────┘              └──────────────────┘
                     split fingerprint 6f478fbcf56364e2
```

---

## 2. Every exclusion, by name

### 2.1 Pre-exclusions — 2 acquisitions

Source: `dataset_exclusions.csv`.

| Acquisition | Reason |
|---|---|
| `CN_Final/sub-012S4026/ses-01/run-01` | 7 volumes — truncated, unusable |
| `LMCI/sub-006S4363/ses-01/run-02` | byte-identical duplicate of `run-01` |

Neither subject was lost: `sub-006S4363` kept `run-01`.

### 2.2 Preprocessing failures — 8 acquisitions

Source: `dataset_failures.csv`. All eight share one cause and **all eight come
from site 002.**

| Group | Acquisition | TR | Nyquist |
|---|---|---:|---:|
| CN_Final | `sub-002S0413/ses-01/run-01` | 6.0216 s | 0.0830 Hz |
| CN_Final | `sub-002S0685/ses-01/run-01` | 6.0216 s | 0.0830 Hz |
| EMCI | `sub-002S2043/ses-01/run-01` | 6.0216 s | 0.0830 Hz |
| LMCI | `sub-002S4171/ses-01/run-01` | 6.0216 s | 0.0830 Hz |
| MCI | `sub-002S0729/ses-01/run-01` | 6.0216 s | 0.0830 Hz |
| MCI | `sub-002S1155/ses-02/run-01` | 6.0216 s | 0.0830 Hz |
| MCI | `sub-002S4799/ses-01/run-01` | 6.0216 s | 0.0830 Hz |
| SMC_Final | `sub-002S5256/ses-02/run-01` | 6.0235 s | 0.0830 Hz |

The band-pass filter runs to 0.10 Hz. At TR ≈ 6.02 s the Nyquist frequency
`1/(2·TR)` is 0.083 Hz — **below** that cutoff, so the filter is not defined.
The pipeline refused these rather than silently narrowing the band, which
would have given them a different filter from every other acquisition.

This is a slow-TR protocol at one site, not a quality problem with the
subjects. Several of them appear in the final cohort through a different
visit — `sub-002S0729`, `sub-002S1155` and `sub-002S4799` are all in the
116.

### 2.3 Later addition — 2 acquisitions

`007_S_2394` and `100_S_0296`, both MCI, converted separately from
`NIfTI_extra` (image ids `I680900` and `I422066`). They are absent from
`audit/bids_conversion/bids_mapping.csv` and from every ADNI assessment table
in the project, which is why they lack medical history downstream.

---

## 3. The selection rule

From 167 valid acquisitions across 129 subjects, 116 were kept.

```
Deterministic, label-blind. One acquisition per subject.

Per stage:
  1. consider only VALID_UNIQUE acquisitions
  2. sort by (subject, session, run) ascending — lexicographic,
     stable, independent of any biomarker value or outcome
  3. take the first acquisition of each distinct subject, in that
     order, until 20 are held
```

**Why one acquisition per subject.** ADNI is longitudinal. Keeping several
visits from one person would leak that person across cross-validation folds
and inflate every score. MCI is the extreme case: **34 acquisitions from only
16 people.**

**Why label-blind matters.** The sort key is identifiers only. No biomarker
value, quality metric or diagnosis influences which visit is kept, so the
selection cannot manufacture a group difference that is reported later as a
finding.

**Why MCI is 16, not 20.** Only 16 distinct MCI subjects had valid
preprocessed acquisitions. The target was not met, and was **not** topped up
with repeat scans from the same 16 people — that would have broken the
one-per-subject rule for the sake of a round number.

| Stage | Valid acquisitions | Distinct subjects | Selected | Target |
|---|---:|---:|---:|---:|
| CN | 27 | 26 | 20 | 20 |
| SMC | 31 | 21 | 20 | 20 |
| EMCI | 29 | 26 | 20 | 20 |
| MCI | 34 | **16** | **16** | 20 |
| LMCI | 21 | 20 | 20 | 20 |
| AD | 25 | 20 | 20 | 20 |
| **Total** | **167** | **129** | **116** | 120 |

Implemented by `scripts/data_prep/build_phase1_dataset.py`. Source trees were
opened read-only; nothing was converted, preprocessed or re-extracted.
Manifests in `derivatives/phase1_selected_20/manifests/`.

---

## 4. Train / test split

| Partition | n | Use |
|---|---:|---|
| train pool | 81 | model fitting, 5 stratified CV folds of 16–17 |
| test | 35 | held out |

Split at **subject** level, stratified by stage, recorded in
`derivatives/model_ready/splits/split_assignment.csv` with fingerprint
`6f478fbcf56364e2`.

Test composition: CN 6, SMC 6, EMCI 6, MCI 5, LMCI 6, AD 6.

**The test set has been evaluated more than once.**
`final_heldout_results.json` records one evaluation of the frozen DC-only
model; a later feature-selection study ran three further configurations on the
same 35 subjects. Four evaluations total. Report all four and keep the
pre-declared primary as the headline — selecting the best of the four would
turn the held-out set into a tuning set.

---

## 5. What the selection does and does not protect against

**Protects against:**

- *Subject leakage* — one acquisition per subject, split at subject level.
- *Selection bias on the outcome* — the rule never reads a biomarker or label.
- *Silent parameter drift* — the 8 slow-TR acquisitions were refused rather
  than filtered differently.
- *Duplicate inflation* — byte-identical duplicate removed; 0 duplicate
  acquisition ids remain.

**Does not protect against:**

- *Small n.* 116 subjects across 6 classes is roughly 19 per class, and each
  CV fold trains on about 65 subjects. Every feature space in this project is
  p ≫ n.
- *Site effects.* All 8 preprocessing failures came from site 002, so site and
  protocol are not evenly distributed across the cohort, and site was not
  modelled.
- *Class imbalance.* MCI at 16 against 20 elsewhere, and falling to 14 when
  medical history is required.
- *The ordinal classes being clinically adjacent.* CN/SMC and EMCI/MCI/LMCI
  are separated by criteria that rs-fMRI may not resolve at all — see
  [LIMITATIONS_AND_FUTURE_WORK.md](LIMITATIONS_AND_FUTURE_WORK.md).

---

## 6. Reproducing this

| Step | Script | Output |
|---|---|---|
| Audit and QC | `scripts/audit/final_6group_audit.py` | `derivatives/fsfast/FINAL_6GROUP_AUDIT/` |
| Cohort selection | `scripts/data_prep/build_phase1_dataset.py` | `derivatives/phase1_selected_20/` |
| Feature matrices | `scripts/data_prep/` | `derivatives/model_ready/` |
| Train/test split | `scripts/model/make_splits.py` | `splits/split_assignment.csv` |
| Clinical join | `scripts/comorbidity/build_model_clinical.py` | `comorbidity_features/model_clinical_116.csv` |

Related: [DATASET_AND_COHORT.md](DATASET_AND_COHORT.md) ·
[PREPROCESSING_PIPELINE.md](PREPROCESSING_PIPELINE.md) ·
[REPRODUCIBILITY.md](REPRODUCIBILITY.md) ·
`derivatives/phase1_selected_20/README.md`
