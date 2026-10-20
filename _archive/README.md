# Archive

Files moved out of the project root on 2026-10-20 to make it readable.
**Nothing was deleted.** Everything here can be moved back with `mv`.

Only items that nothing in `docs/`, `scripts/`, `pipeline/` or `README.md`
referenced were moved. One exception is noted below.

```
_archive/
├── logs/          44  run logs from every pipeline stage
├── scratch/        5  shell accidents and regenerable files
├── notes/          4  superseded documents and submissions
├── validation/     1  the old test/ directory
└── derivatives/  87M  superseded analysis outputs
```

## `logs/` — 44 files, 275 KB

Run records for conversion, preprocessing, biomarker extraction,
normalisation, auditing and the statistical passes: `*.log`, `*.err`, `*.out`.

These are **provenance, not junk**. They record what was run and what it
printed, including the failures. Keep them — they are the evidence behind
the counts in [docs/DATA_SELECTION.md](../docs/DATA_SELECTION.md), such as
the 8 slow-TR preprocessing refusals.

## `scratch/` — 5 items

| item | what it is |
|---|---|
| `tatus` | a mistyped `git status >` redirect |
| `how :3:.gitignore` | another stray redirect |
| `wsl_depcheck.sh`, `.out` | one-off WSL dependency check; output empty |
| `Untitled-1.txt` | a scratch note on the FreeSurfer wrapper design |
| `__pycache__/` | compiled Python, regenerates on import |

Safe to delete. Kept only because deleting was not asked for.

## `notes/` — 4 items

| item | what it is |
|---|---|
| `README.v1_backup.md` | the previous root README. **`README.md` links here** — the link was updated when this moved |
| `md files/` | `CONVERT_AD_EXTRA_DATA.md`, `INSTRUCTIONS.md` — conversion instructions written for a second machine |
| `old18_reference.txt` | one-vs-rest statistics from the earlier 18-subject-per-stage cohort, superseded by the 116-subject results |
| `SecondReviewLaTeX.zip` | the second-review submission source |

## `validation/` — the old `test/`

Preprocessing validation records for the AD group, plus a cleanup report.
Real provenance; moved only because `test/` at the project root reads like a
test suite, which it is not.

## `derivatives/` — 87 MB of superseded analysis

| item | size | why it moved |
|---|---:|---|
| `_QUARANTINE_wrong_pipeline/` | 54 M | output of a pipeline run that was already quarantined; referenced nowhere |
| `v1_heatmaps_114subj/` | 23 M | the **114-subject** intra/inter-class analysis — `figures/`, `matrices/`, `pairwise/`, `statistical_tests.csv`, `intra_class_summary.csv`, `subject_stage_mapping.csv`, `analysis_log.txt`. Superseded by `v2_cohort116/`, which has its own complete `pairwise/` (58 files, same count) computed on the 116-subject cohort |
| `pscore_other_variants/` | 10 M | 58 intermediate P-score figure designs from the iterations that produced `pscore_figures/png/` |
| `comorbidity_idasearch_only/` | 160 K | the first clinical build, from the undated `idaSearch` export only — no CDR sum-of-boxes, not scan-matched. Superseded by `model_clinical_116.csv` |
| `FINAL_6GROUP_AUDIT_backup_165/` | 94 K | audit snapshot from the 165-acquisition era, before the two MCI additions; every file differs from the current audit |

`v1_heatmaps_114subj/subject_stage_mapping.csv` is worth flagging: it has
**114 rows, not 116**, and predates the two MCI additions. Anything joined
against it silently drops `sub-007S2394` and `sub-100S0296`. Use
`derivatives/model_ready/subjects.npy` instead.

The only remaining reference to any of this is a docstring in
`scripts/audit/pairwise_stage_statistics.py` naming
`INTRA_INTER_CLASS_HEATMAPS/pairwise/` as its **output** path, so re-running
the script simply recreates it.

---

## Deliberately **not** archived

| kept in root | why |
|---|---|
| `pilot_preprocessing/`, `pilot_preprocessing_2mm4mm/` | 2.6 GB of pilot derivatives, but referenced by 8 files including `scripts/audit/qc_audit_original_pilot.py`, `qc_audit_pilot_2mm4mm.py` and `scripts/setup/retry_pilot_2mm4mm.sh`. Moving them breaks those scripts. Archive them only together with a path update. |
| `BIDS/`, `NIfTI_/`, `NIfTI_extra/`, `raw_data_extra/` | source imaging — read-only, never to be moved or altered |
| `Comorbidity/` | ADNI clinical source tables |
| `tools/` | third-party binaries; gitignored rather than moved, since the path is referenced by the conversion scripts |
| `derivatives/clinical_cleaned/` | **stale** — built 2026-10-08, before the LMCI tables arrived, so its LMCI rows are blank and it covers 94/116. But `scripts/model/audit_clinical_feasibility.py` and `audit_pipeline.py` read it, so moving it breaks them. Use `derivatives/comorbidity_features/model_clinical_116.csv` instead |
| `derivatives/STAGE_HEATMAPS/`, `ATLAS_FIGURES/`, `PREPROCESSING_FIGURES/`, `REVIEW2_FINDINGS/` | all referenced by docs or scripts |

## Restoring

```bash
mv _archive/notes/README.v1_backup.md .     # then revert the README link
mv _archive/logs/*.log .
```
