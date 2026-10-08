# Clinical Data Audit Report

Generated (UTC): 2026-10-07T19:43:32.923124+00:00

Read-only with respect to every ADNI source file. No imputation, scaling,
feature selection or model training was performed.

## 1. Source files discovered

| Table | Rows | Cols | ID cols | Visit cols | Description |
|---|---:|---:|---|---|---|
| `AD_9_01_2026` | 30 | 12 | Subject | Visit | ADNI image-collection export: Subject, Group, Sex, Age, Visit, Acq Date |
| `APOERES_01Sep2026` | 3216 | 16 | PTID,RID | PHASE,VISCODE | APOE genotyping result; GENOTYPE as allele pair |
| `CN_Final_9_01_2026` | 30 | 12 | Subject | Visit | ADNI image-collection export: Subject, Group, Sex, Age, Visit, Acq Date |
| `EMCI_9_01_2026` | 30 | 12 | Subject | Visit | ADNI image-collection export: Subject, Group, Sex, Age, Visit, Acq Date |
| `MCI_9_01_2026` | 35 | 12 | Subject | Visit | ADNI image-collection export: Subject, Group, Sex, Age, Visit, Acq Date |
| `SMC_Final_9_01_2026` | 32 | 12 | Subject | Visit | ADNI image-collection export: Subject, Group, Sex, Age, Visit, Acq Date |
| `AD::CDR` | 80 | 25 | PTID,RID | PHASE,VISCODE,VISCODE2 | Clinical Dementia Rating (longitudinal); CDGLOBAL and CDRSB |
| `AD::MMSE` | 78 | 58 | PTID,RID | PHASE,VISCODE,VISCODE2 | Mini-Mental State Examination (longitudinal); MMSCORE 0-30 |
| `AD::GDSCALE` | 80 | 32 | PTID,RID | PHASE,VISCODE,VISCODE2 | Geriatric Depression Scale (longitudinal); GDTOTAL 0-15 |
| `AD::MEDHIST` | 25 | 39 | PTID,RID | PHASE,VISCODE,VISCODE2 | Medical history checklist, screening visit; MH* category flags |
| `AD::APOERES` | 25 | 16 | PTID,RID | PHASE,VISCODE | APOE genotyping result; GENOTYPE as allele pair |
| `AD::Functional_MRI_Images` | 30 | 19 | subject_id |  | Imaging index: image_id, fmri_date, fmri_visit, scanner |
| `CN::CDR` | 228 | 25 | PTID,RID | PHASE,VISCODE,VISCODE2 | Clinical Dementia Rating (longitudinal); CDGLOBAL and CDRSB |
| `CN::MMSE` | 212 | 58 | PTID,RID | PHASE,VISCODE,VISCODE2 | Mini-Mental State Examination (longitudinal); MMSCORE 0-30 |
| `CN::GDSCALE` | 222 | 32 | PTID,RID | PHASE,VISCODE,VISCODE2 | Geriatric Depression Scale (longitudinal); GDTOTAL 0-15 |
| `CN::MEDHIST` | 39 | 39 | PTID,RID | PHASE,VISCODE,VISCODE2 | Medical history checklist, screening visit; MH* category flags |
| `CN::APOERES` | 29 | 16 | PTID,RID | PHASE,VISCODE | APOE genotyping result; GENOTYPE as allele pair |
| `CN::Functional_MRI_Images` | 30 | 19 | subject_id |  | Imaging index: image_id, fmri_date, fmri_visit, scanner |
| `EMCI::APOERES` | 27 | 16 | PTID,RID | PHASE,VISCODE | APOE genotyping result; GENOTYPE as allele pair |
| `EMCI::Functional_MRI_Images` | 30 | 19 | subject_id |  | Imaging index: image_id, fmri_date, fmri_visit, scanner |
| `EMCI::CDR` | 162 | 25 | PTID,RID | PHASE,VISCODE,VISCODE2 | Clinical Dementia Rating (longitudinal); CDGLOBAL and CDRSB |
| `EMCI::MMSE` | 154 | 58 | PTID,RID | PHASE,VISCODE,VISCODE2 | Mini-Mental State Examination (longitudinal); MMSCORE 0-30 |
| `EMCI::GDSCALE` | 161 | 32 | PTID,RID | PHASE,VISCODE,VISCODE2 | Geriatric Depression Scale (longitudinal); GDTOTAL 0-15 |
| `EMCI::MEDHIST` | 36 | 39 | PTID,RID | PHASE,VISCODE,VISCODE2 | Medical history checklist, screening visit; MH* category flags |
| `MCI::GDSCALE` | 147 | 32 | PTID,RID | PHASE,VISCODE,VISCODE2 | Geriatric Depression Scale (longitudinal); GDTOTAL 0-15 |
| `MCI::MEDHIST` | 27 | 39 | PTID,RID | PHASE,VISCODE,VISCODE2 | Medical history checklist, screening visit; MH* category flags |
| `MCI::MMSE` | 158 | 58 | PTID,RID | PHASE,VISCODE,VISCODE2 | Mini-Mental State Examination (longitudinal); MMSCORE 0-30 |
| `MCI::CDR` | 159 | 25 | PTID,RID | PHASE,VISCODE,VISCODE2 | Clinical Dementia Rating (longitudinal); CDGLOBAL and CDRSB |
| `MCI::APOERES` | 14 | 16 | PTID,RID | PHASE,VISCODE | APOE genotyping result; GENOTYPE as allele pair |
| `MCI::Functional_MRI_Images` | 35 | 19 | subject_id |  | Imaging index: image_id, fmri_date, fmri_visit, scanner |
| `SMC::CDR` | 74 | 25 | PTID,RID | PHASE,VISCODE,VISCODE2 | Clinical Dementia Rating (longitudinal); CDGLOBAL and CDRSB |
| `SMC::MMSE` | 73 | 58 | PTID,RID | PHASE,VISCODE,VISCODE2 | Mini-Mental State Examination (longitudinal); MMSCORE 0-30 |
| `SMC::GDSCALE` | 74 | 32 | PTID,RID | PHASE,VISCODE,VISCODE2 | Geriatric Depression Scale (longitudinal); GDTOTAL 0-15 |
| `SMC::MEDHIST` | 21 | 39 | PTID,RID | PHASE,VISCODE,VISCODE2 | Medical history checklist, screening visit; MH* category flags |
| `SMC::APOERES` | 20 | 16 | PTID,RID | PHASE,VISCODE | APOE genotyping result; GENOTYPE as allele pair |
| `SMC::Functional_MRI_Images` | 32 | 19 | subject_id |  | Imaging index: image_id, fmri_date, fmri_visit, scanner |
| `participants` | 131 | 2 | participant_id |  | Project BIDS participants file: authoritative group label |

## 2-4. Subjects, visits, variables

- Project cohort: **127 subjects** (authoritative stage from the imaging pipeline and `participants.tsv`)
- Source tables discovered: **37**
- Clinical visit records for cohort subjects: **2441**
- Subjects with at least one clinical visit record: **127 / 127**
- Variables catalogued: **1023**
- Variables with a decoded meaning: **202**
- Variables flagged potentially useful: **131**

## 5. Decoding

No ADNI codebook or data dictionary is bundled with these files. Decoding
was therefore restricted to:

- **APOE genotype -> e4 allele count**: counted directly from the genotype
  string itself (e.g. `3/4` -> 1). Derived from the value, not a lookup.
- **VISCODE**: `sc` screening, `bl` baseline, `mNN` month NN, `vNN` study
  visit. Anything else -> `UNKNOWN_CODE`.
- **Medical-history flags**: 1 present, 0 absent, negative sentinel ->
  missing. This follows ADNI convention and the observed value set; it is
  recorded as a derivation rule, not a codebook lookup.
- **Sex**: `M`/`F` as exported.

Anything not covered above is left raw and marked `UNKNOWN_CODE`.

## 6. Missing-value handling

Sentinel codes considered: [-1, -4, -9]. A sentinel was only treated as
missing where the column also contains non-negative values, so genuinely
signed measures are not corrupted. Missing is represented as `NaN`.
**No imputation was performed.** Imputation must be fitted on training
subjects only, after the subject-level split.

## 7. Duplicates and conflicts

- Conflicting duplicate values found: **116**
- None were auto-resolved; see `duplicate_conflict_report.csv`.

## 8. Visit matching

Visit-dependent scores were matched to the assessment closest in date
to each subject's earliest MRI acquisition. The gap in days is retained
per subject in `<score>_days_from_MRI`, and the method used is recorded
in `<score>_match_method`, so no selection is silent.

MRI acquisition dates come from the `Functional_MRI_Images` tables and
the ADNI image-collection exports. The preprocessing provenance paths
carry a usable timestamp for only 25 of 165 acquisitions, because most
source files were renamed to BIDS style, so they are a supplement only.

| Score | Date-matched | Fallback (first record) | Median gap (d) | p90 gap (d) | Max gap (d) |
|---|---:|---:|---:|---:|---:|
| CDR_global | 107 | 0 | 0 | 22 | 77 |
| CDR_SB | 107 | 0 | 0 | 22 | 77 |
| MMSE_total | 107 | 0 | 0 | 22 | 77 |
| GDS_total | 107 | 0 | 0 | 23 | 189 |

Any row whose `_match_method` reads `FALLBACK` had no usable
assessment date and took the first available record. Those values are
flagged, not hidden, and should be treated with caution.

## 9. Comorbidity derivation

| Output variable | Source column | Rule |
|---|---|---|
| `APOE_genotype_raw / APOE4_allele_count / APOE4_carrier` | APOERES_01Sep2026.csv :: GENOTYPE | allele count = number of '4' characters in the genotype pair |
| `Sex_raw, Sex_decoded, Age_years` | *_9_01_2026.csv :: Sex / Age / Acq Date | earliest listed acquisition per subject |
| `CDR_global` | CDR :: CDGLOBAL | value from the assessment closest in date to the subject's earliest MRI acquisition; gap in days recorded |
| `CDR_SB` | CDR :: CDRSB | value from the assessment closest in date to the subject's earliest MRI acquisition; gap in days recorded |
| `MMSE_total` | MMSE :: MMSCORE | value from the assessment closest in date to the subject's earliest MRI acquisition; gap in days recorded |
| `GDS_total` | GDSCALE :: GDTOTAL | value from the assessment closest in date to the subject's earliest MRI acquisition; gap in days recorded |
| `MHPSYCH_clean / MHPSYCH_decoded` | MEDHIST :: MHPSYCH | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention, no codebook shipped with the source |
| `MH2NEURL_clean / MH2NEURL_decoded` | MEDHIST :: MH2NEURL | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention, no codebook shipped with the source |
| `MH3HEAD_clean / MH3HEAD_decoded` | MEDHIST :: MH3HEAD | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention, no codebook shipped with the source |
| `MH4CARD_clean / MH4CARD_decoded` | MEDHIST :: MH4CARD | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention, no codebook shipped with the source |
| `MH5RESP_clean / MH5RESP_decoded` | MEDHIST :: MH5RESP | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention, no codebook shipped with the source |
| `MH6HEPAT_clean / MH6HEPAT_decoded` | MEDHIST :: MH6HEPAT | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention, no codebook shipped with the source |
| `MH7DERM_clean / MH7DERM_decoded` | MEDHIST :: MH7DERM | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention, no codebook shipped with the source |
| `MH8MUSCL_clean / MH8MUSCL_decoded` | MEDHIST :: MH8MUSCL | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention, no codebook shipped with the source |
| `MH9ENDO_clean / MH9ENDO_decoded` | MEDHIST :: MH9ENDO | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention, no codebook shipped with the source |
| `MH10GAST_clean / MH10GAST_decoded` | MEDHIST :: MH10GAST | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention, no codebook shipped with the source |
| `MH11HEMA_clean / MH11HEMA_decoded` | MEDHIST :: MH11HEMA | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention, no codebook shipped with the source |
| `MH12RENA_clean / MH12RENA_decoded` | MEDHIST :: MH12RENA | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention, no codebook shipped with the source |
| `MH13ALLE_clean / MH13ALLE_decoded` | MEDHIST :: MH13ALLE | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention, no codebook shipped with the source |
| `MH14ALCH_clean / MH14ALCH_decoded` | MEDHIST :: MH14ALCH | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention, no codebook shipped with the source |
| `MH15DRUG_clean / MH15DRUG_decoded` | MEDHIST :: MH15DRUG | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention, no codebook shipped with the source |
| `MH16SMOK_clean / MH16SMOK_decoded` | MEDHIST :: MH16SMOK | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention, no codebook shipped with the source |
| `MH17MALI_clean / MH17MALI_decoded` | MEDHIST :: MH17MALI | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention, no codebook shipped with the source |
| `MH18SURG_clean / MH18SURG_decoded` | MEDHIST :: MH18SURG | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention, no codebook shipped with the source |
| `MH19OTHR_clean / MH19OTHR_decoded` | MEDHIST :: MH19OTHR | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention, no codebook shipped with the source |
| `cardiovascular_history` | MEDHIST :: MH4CARD | BROAD category; includes hypertension but is NOT a hypertension-specific flag |
| `endocrine_metabolic_history` | MEDHIST :: MH9ENDO | BROAD category; includes diabetes but is NOT a diabetes-specific flag |

> **Naming caution.** `MH4CARD` and `MH9ENDO` are broad ADNI category
> flags. They are exported here as `cardiovascular_history` and
> `endocrine_metabolic_history`. They are **not** hypertension-specific or
> diabetes-specific, and must not be relabelled as such.

## 10. Stage-wise subject counts

| Stage | Subjects | Expected | Match |
|---|---:|---:|---|
| CN | 26 | 26 | yes |
| SMC | 21 | 21 | yes |
| EMCI | 26 | 26 | yes |
| MCI | 14 | 14 | yes |
| LMCI | 20 | 20 | yes |
| AD | 20 | 20 | yes |
| **Total** | **127** | **127** | |

## 11. Coverage of the master table

| Variable | Present / 127 | % |
|---|---:|---:|
| `APOE_genotype_raw` | 126 | 99% |
| `APOE4_allele_count` | 126 | 99% |
| `APOE4_carrier` | 126 | 99% |
| `APOE_source` | 127 | 100% |
| `Sex_raw` | 107 | 84% |
| `Sex_decoded` | 107 | 84% |
| `Age_years` | 107 | 84% |
| `Age_source_visit` | 107 | 84% |
| `Age_source_acq_date` | 107 | 84% |
| `MRI_date_used_for_matching` | 107 | 84% |
| `MRI_date_source` | 107 | 84% |
| `CDR_global` | 107 | 84% |
| `CDR_global_visit` | 107 | 84% |
| `CDR_global_days_from_MRI` | 107 | 84% |
| `CDR_global_match_method` | 107 | 84% |
| `CDR_SB` | 107 | 84% |
| `CDR_SB_visit` | 107 | 84% |
| `CDR_SB_days_from_MRI` | 107 | 84% |
| `CDR_SB_match_method` | 107 | 84% |
| `MMSE_total` | 107 | 84% |
| `MMSE_total_visit` | 107 | 84% |
| `MMSE_total_days_from_MRI` | 107 | 84% |
| `MMSE_total_match_method` | 107 | 84% |
| `GDS_total` | 107 | 84% |
| `GDS_total_visit` | 107 | 84% |
| `GDS_total_days_from_MRI` | 107 | 84% |
| `GDS_total_match_method` | 107 | 84% |
| `MHPSYCH_raw` | 107 | 84% |
| `MHPSYCH_clean` | 107 | 84% |
| `MHPSYCH_decoded` | 107 | 84% |
| `MH2NEURL_raw` | 107 | 84% |
| `MH2NEURL_clean` | 107 | 84% |
| `MH2NEURL_decoded` | 107 | 84% |
| `MH3HEAD_raw` | 107 | 84% |
| `MH3HEAD_clean` | 107 | 84% |
| `MH3HEAD_decoded` | 107 | 84% |
| `MH4CARD_raw` | 107 | 84% |
| `MH4CARD_clean` | 107 | 84% |
| `MH4CARD_decoded` | 107 | 84% |
| `MH5RESP_raw` | 107 | 84% |
| `MH5RESP_clean` | 107 | 84% |
| `MH5RESP_decoded` | 107 | 84% |
| `MH6HEPAT_raw` | 107 | 84% |
| `MH6HEPAT_clean` | 107 | 84% |
| `MH6HEPAT_decoded` | 107 | 84% |
| `MH7DERM_raw` | 107 | 84% |
| `MH7DERM_clean` | 107 | 84% |
| `MH7DERM_decoded` | 107 | 84% |
| `MH8MUSCL_raw` | 107 | 84% |
| `MH8MUSCL_clean` | 107 | 84% |
| `MH8MUSCL_decoded` | 107 | 84% |
| `MH9ENDO_raw` | 107 | 84% |
| `MH9ENDO_clean` | 107 | 84% |
| `MH9ENDO_decoded` | 107 | 84% |
| `MH10GAST_raw` | 107 | 84% |
| `MH10GAST_clean` | 107 | 84% |
| `MH10GAST_decoded` | 107 | 84% |
| `MH11HEMA_raw` | 107 | 84% |
| `MH11HEMA_clean` | 107 | 84% |
| `MH11HEMA_decoded` | 107 | 84% |
| `MH12RENA_raw` | 107 | 84% |
| `MH12RENA_clean` | 107 | 84% |
| `MH12RENA_decoded` | 107 | 84% |
| `MH13ALLE_raw` | 107 | 84% |
| `MH13ALLE_clean` | 107 | 84% |
| `MH13ALLE_decoded` | 107 | 84% |
| `MH14ALCH_raw` | 107 | 84% |
| `MH14ALCH_clean` | 107 | 84% |
| `MH14ALCH_decoded` | 107 | 84% |
| `MH15DRUG_raw` | 107 | 84% |
| `MH15DRUG_clean` | 107 | 84% |
| `MH15DRUG_decoded` | 107 | 84% |
| `MH16SMOK_raw` | 107 | 84% |
| `MH16SMOK_clean` | 107 | 84% |
| `MH16SMOK_decoded` | 107 | 84% |
| `MH17MALI_raw` | 107 | 84% |
| `MH17MALI_clean` | 107 | 84% |
| `MH17MALI_decoded` | 107 | 84% |
| `MH18SURG_raw` | 107 | 84% |
| `MH18SURG_clean` | 107 | 84% |
| `MH18SURG_decoded` | 107 | 84% |
| `MH19OTHR_raw` | 107 | 84% |
| `MH19OTHR_clean` | 107 | 84% |
| `MH19OTHR_decoded` | 107 | 84% |
| `cardiovascular_history` | 107 | 84% |
| `endocrine_metabolic_history` | 107 | 84% |

## 12. Unresolved issues

- 116 conflicting duplicate values require manual review

## 13. Final master table

`master_subject_clinical.csv`: **127 rows x 90 columns**

Stage tables in `stages/` carry the same columns, one file per stage.
