# Clinical and comorbidity data for this cohort

Copy of `derivatives/comorbidity_features/`, placed here so the 116-subject
cohort and its clinical data sit together. **That directory is the original** —
rebuild there, then re-copy. Do not edit these in place.

Nothing under the stage folders (`AD/`, `CN/`, `EMCI/`, `LMCI/`, `MCI/`,
`SMC/`) was read, written or altered.

## Use this one

**`model_clinical_116.csv`** — 116 rows, 50 columns.

Row order is exactly `derivatives/model_ready/subjects.npy`, so **row *i* here
is row *i* of `X_regional.npy`, `X_fc.npy` and `y.npy`**. No join needed.

| file | what it is |
|---|---|
| `model_clinical_116.csv` | **the table** — cognitive, mood, genetics, 19 comorbidity flags |
| `model_clinical_report.txt` | coverage, scan-match gaps, prevalence, rejections |
| `stage_subject_mapping_116.csv` | join table: cohort folder ⟷ BIDS id ⟷ ADNI id |
The earlier idaSearch-only build was moved to
`_archive/derivatives/comorbidity_idasearch_only/`. It is not scan-matched
and has no CDR sum-of-boxes — do not use it.

Built by `scripts/comorbidity/build_model_clinical.py`.

## Where it comes from

| block | source | scan-matched |
|---|---|---|
| MMSE, CDR global, CDR-SB | `Comorbidity/Tables_*.zip` → CDR / MMSE | **yes**, median 0 d, max 77 d |
| GDS | `Comorbidity/Tables_*.zip` → GDSCALE | **yes**, median 0 d |
| 19 comorbidity flags | `Tables_*.zip` → MEDHIST (= `new_comorbidity/*_comorbidity.csv`) | yes, median 209 d |
| APOE genotype | `Comorbidity/APOERES_01Sep2026.csv` | n/a |
| 2 MCI subjects only | `new_comorbidity/gdscale-cdr-mmse/idaSearch_*.csv` | **no** — see below |

Every row records its own source and match method in `*_source` and `*_match`.

## Coverage — not uniform, check before filtering

| field | available | missing |
|---|---|---|
| MMSE | **116/116** | — |
| CDR global | **116/116** | — |
| GDS | 115/116 | `130_S_4990` (AD) |
| APOE genotype | 115/116 | `019_S_5180` (SMC) |
| CDR sum-of-boxes | 114/116 | `007_S_2394`, `100_S_0296` (MCI) |
| comorbidity flags | 114/116 | `007_S_2394`, `100_S_0296` (MCI) |
| everything | **113/116** | |

Usable *n* depends on which fields you take:

| fields | n |
|---|---|
| MMSE + CDR global | **116** |
| + GDS | 115 |
| + APOE | 114 |
| + comorbidity flags | 113 |
| + CDR-SB | 112 |

## Three things to know

**`-4` means missing.** It is ADNI's missing code and appears throughout the
sources. It is stored blank here. Reading it as 0 would invert the prevalence
of several conditions. Blank is never "no disease" — nothing is imputed.

**Two subjects are not scan-matched.** `007_S_2394` and `100_S_0296` are absent
from every zip table, so their MMSE / CDR / GDS come from the idaSearch export,
which has no date. Their `*_match` column says so. They have no CDR-SB and no
comorbidity flags. The other 114 are matched to the visit nearest the scan.

**MMSE, CDR global and CDR-SB partly define the label.** ADNI's diagnostic
criteria use their cut-offs to assign the stage being predicted; CDR-SB is the
finest-grained form of that criterion. Global CDR alone reaches 0.386 balanced
accuracy against the rs-fMRI model's 0.307 — that measures the enrolment rule,
not the biology. Use them as a baseline, not as imaging-model features.
**`GDS_total`, the 19 `MH_*` flags and `APOE4_count` are not part of the
criteria** and are the honest covariates here.
