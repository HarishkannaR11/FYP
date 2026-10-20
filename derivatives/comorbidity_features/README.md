# Comorbidity Features — 116-Subject Cohort

Clinical, genetic and comorbidity data mapped onto the analysis cohort.

**Use `model_clinical_116.csv`.** 116 rows, 50 columns, in
`derivatives/model_ready/subjects.npy` row order — row *i* here is row *i* of
`X_regional.npy`, `X_fc.npy` and `y.npy`. No join needed.

| file | rows | what it is |
|---|---:|---|
| **`model_clinical_116.csv`** | 116 | **the table** — cognitive, mood, genetics, 19 comorbidity flags |
| `model_clinical_report.txt` | — | coverage, scan-match gaps, prevalence, rejections |
| `stage_subject_mapping_116.csv` | 116 | join table: cohort folder ⟷ BIDS id ⟷ ADNI id ⟷ source file |
| `clinical_visits_116.csv` | 670 | every clinical visit, unaggregated |
| `clinical_subject_116.csv` | 116 | per-subject baseline / last / median / min / max |
| `cohort_comorbidity_116.csv` | 116 | earlier merge, idaSearch scores + MEDHIST |
| `clinical_qc_report.txt`, `cohort_comorbidity_report.txt` | — | build logs for the two earlier passes |

The last three predate `model_clinical_116.csv` and are kept for traceability.
They are **not scan-matched** and have no CDR sum-of-boxes. Do not model from
them.

Built by `scripts/comorbidity/build_model_clinical.py`. Sources are read-only.

---

## The comorbidity block

This is what the project's *comorbidity-aware* claim rests on. **21 features,
112 subjects**, none of which enters ADNI's diagnostic criteria:

```
19 × MH_*  +  GDS_total  +  APOE4_count
```

| | columns |
|---|---|
| comorbidity | `MH_psychiatric`, `MH_neurologic_non_AD`, `MH_head_eyes_ears_nose_throat`, `MH_cardiovascular`, `MH_respiratory`, `MH_hepatic`, `MH_dermatologic_connective_tissue`, `MH_musculoskeletal`, `MH_endocrine_metabolic`, `MH_gastrointestinal`, `MH_hematopoietic_lymphatic`, `MH_renal_genitourinary`, `MH_allergies_drug_sensitivities`, `MH_alcohol_abuse`, `MH_drug_abuse`, `MH_smoking`, `MH_malignancy`, `MH_major_surgical_procedures`, `MH_other` |
| mood | `GDS_total` |
| genetics | `APOE4_count` (0/1/2), `APOE_genotype` |

5-fold CV balanced accuracy on 112 subjects, chance 0.167:

| features | balanced accuracy |
|---|---|
| 19 comorbidity flags alone | 0.258 ± 0.096 |
| APOE4 count alone | 0.258 ± 0.049 |
| GDS alone | 0.194 ± 0.049 |
| **the whole block** | **0.264 ± 0.082** |
| *rs-fMRI imaging model, for comparison* | *0.307 CV / 0.200 held-out* |

Above chance, but weak — the ±0.082 spread overlaps chance on some folds.
`APOE4_count` alone matches all 19 flags combined, from one variable.

Of the 19 flags, roughly 6–8 carry usable variance. Five are too rare at this
*n* (drug abuse 2, alcohol abuse 4, hematopoietic 10, hepatic 11, malignancy
25) and three are near-constant at ~75% (major surgical procedures;
head/eyes/ears/nose/throat; musculoskeletal).

---

## Do not use MMSE or CDR as model features

`MMSE_total`, `CDR_global` and `CDR_SOB` are present in the table, but they
are **inputs to the ADNI criteria that assign the stage being predicted**.
Global CDR alone reaches **0.386** balanced accuracy against the imaging
model's 0.307, and the CDR × stage cross-tab simply reproduces the enrolment
rule (CDR 0 → CN/SMC, 0.5 → the MCI band, 1 → AD).

They belong in a baseline table — "Global CDR alone reaches 0.386 while our
imaging model reaches 0.307" is an honest and useful framing — not in the
feature set.

Note that MMSE and CDR are **not comorbidities**. They are diagnostic
instruments. The comorbidity block above is unaffected by this problem.

---

## Coverage

| field | available | missing |
|---|---|---|
| MMSE, CDR global | **116/116** | — |
| GDS | 115/116 | `130_S_4990` (AD) |
| APOE genotype | 115/116 | `019_S_5180` (SMC) |
| CDR sum-of-boxes | 114/116 | `007_S_2394`, `100_S_0296` (MCI) |
| 19 comorbidity flags | 114/116 | `007_S_2394`, `100_S_0296` (MCI) |
| the comorbidity block (21 features) | **112/116** | the four above |
| everything | 113/116 | |

Filter on `complete`, `has_cognitive`, `has_mood` or `has_medhist` rather
than re-deriving.

---

## Provenance and cleaning

| block | source | scan-matched |
|---|---|---|
| MMSE, CDR, CDR-SB | `Comorbidity/Tables_*.zip` → CDR / MMSE | **yes** — median 0 d, max 77 d |
| GDS | `Tables_*.zip` → GDSCALE | **yes** — median 0 d |
| 19 flags | `Tables_*.zip` → MEDHIST | yes — median 209 d (screening visit) |
| APOE | `Comorbidity/APOERES_01Sep2026.csv` | n/a |
| 2 MCI subjects only | `new_comorbidity/gdscale-cdr-mmse/idaSearch_*.csv` | **no** — undated export |

Every row records its own `*_source`, `*_match`, `*_visit_date` and signed
`*_days_from_MRI`.

**`-4` and `-1` are ADNI missing codes** and are stored blank, never 0.
Reading them as 0 would invert the prevalence of several conditions. A blank
is never "no disease" — nothing is imputed. Totals are range-checked; 0 values
were rejected. Medical-history labels come from `DATADIC_12Oct2026.csv`, not
inferred: `MH3HEAD` is *"Head, Eyes, Ears, Nose and Throat"*, **not** head
injury.

Under the ADNI Data Use Agreement these tables are not redistributable.
They regenerate from `scripts/comorbidity/build_model_clinical.py`.

See also: `../phase1_selected_20/comorbidity/SOURCE.md` ·
`../../docs/DATASET_AND_COHORT.md` §4
