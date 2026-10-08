# Clinical Data Dictionary

Generated (UTC): 2026-10-07T19:43:32.915817+00:00

Every variable in `master_subject_clinical.csv`. No ADNI codebook ships
with the source files, so decoding is limited to mappings derivable from
the data itself; anything else is recorded as `UNKNOWN_CODE`.

| Variable | Meaning | Source file | Source column | Type | Coding | Missing means | Cleaning | ML-suitable | Visit-dependent |
|---|---|---|---|---|---|---|---|---|---|
| `subject_id` | project cohort field (authoritative) | derivatives/biomarkers_normalized + participants.tsv |  | object |  | NaN = not available | see audit report | no (identifier/provenance) | no |
| `group_dir` | project cohort field (authoritative) | derivatives/biomarkers_normalized + participants.tsv |  | object |  | NaN = not available | see audit report | no (identifier/provenance) | no |
| `stage` | project cohort field (authoritative) | derivatives/biomarkers_normalized + participants.tsv |  | object |  | NaN = not available | see audit report | no (identifier/provenance) | no |
| `stage_ordinal` | project cohort field (authoritative) | derivatives/biomarkers_normalized + participants.tsv |  | int64 |  | NaN = not available | see audit report | no (identifier/provenance) | no |
| `APOE_genotype_raw` |  | APOERES_01Sep2026.csv | GENOTYPE | object | allele count = number of '4' characters in the genotype pair | NaN = not available | see audit report | no (identifier/provenance) | no |
| `APOE4_allele_count` |  | APOERES_01Sep2026.csv | GENOTYPE | float64 | allele count = number of '4' characters in the genotype pair | NaN = not available | see audit report | yes | no |
| `APOE4_carrier` |  | APOERES_01Sep2026.csv | GENOTYPE | float64 | allele count = number of '4' characters in the genotype pair | NaN = not available | see audit report | yes | no |
| `APOE_source` |  |  |  | object |  | NaN = not available | see audit report | no (identifier/provenance) | no |
| `Sex_raw` | Biological sex as exported by ADNI (M/F) | *_9_01_2026.csv | Sex / Age / Acq Date | object | earliest listed acquisition per subject | NaN = not available | see audit report | yes | no |
| `Sex_decoded` | Biological sex as exported by ADNI (M/F) | *_9_01_2026.csv | Sex / Age / Acq Date | object | earliest listed acquisition per subject | NaN = not available | see audit report | yes | no |
| `Age_years` |  | *_9_01_2026.csv | Sex / Age / Acq Date | float64 | earliest listed acquisition per subject | NaN = not available | see audit report | yes | no |
| `Age_source_visit` |  |  |  | object |  | NaN = not available | see audit report | no (identifier/provenance) | yes |
| `Age_source_acq_date` |  |  |  | object |  | NaN = not available | see audit report | no (identifier/provenance) | no |
| `MRI_date_used_for_matching` |  |  |  | object |  | NaN = not available | see audit report | no (identifier/provenance) | no |
| `MRI_date_source` |  |  |  | object |  | NaN = not available | see audit report | no (identifier/provenance) | no |
| `CDR_global` |  | CDR | CDGLOBAL | float64 | value from the assessment closest in date to the subject's earliest MR | NaN = not available | see audit report | yes | yes |
| `CDR_global_visit` |  |  |  | object |  | NaN = not available | see audit report | no (identifier/provenance) | yes |
| `CDR_global_days_from_MRI` |  |  |  | float64 |  | NaN = not available | see audit report | no (identifier/provenance) | yes |
| `CDR_global_match_method` |  |  |  | object |  | NaN = not available | see audit report | no (identifier/provenance) | no |
| `CDR_SB` |  | CDR | CDRSB | float64 | value from the assessment closest in date to the subject's earliest MR | NaN = not available | see audit report | yes | yes |
| `CDR_SB_visit` |  |  |  | object |  | NaN = not available | see audit report | no (identifier/provenance) | yes |
| `CDR_SB_days_from_MRI` |  |  |  | float64 |  | NaN = not available | see audit report | no (identifier/provenance) | yes |
| `CDR_SB_match_method` |  |  |  | object |  | NaN = not available | see audit report | no (identifier/provenance) | no |
| `MMSE_total` |  | MMSE | MMSCORE | float64 | value from the assessment closest in date to the subject's earliest MR | NaN = not available | see audit report | yes | yes |
| `MMSE_total_visit` |  |  |  | object |  | NaN = not available | see audit report | no (identifier/provenance) | yes |
| `MMSE_total_days_from_MRI` |  |  |  | float64 |  | NaN = not available | see audit report | no (identifier/provenance) | yes |
| `MMSE_total_match_method` |  |  |  | object |  | NaN = not available | see audit report | no (identifier/provenance) | no |
| `GDS_total` |  | GDSCALE | GDTOTAL | float64 | value from the assessment closest in date to the subject's earliest MR | NaN = not available | see audit report | yes | yes |
| `GDS_total_visit` |  |  |  | object |  | NaN = not available | see audit report | no (identifier/provenance) | yes |
| `GDS_total_days_from_MRI` |  |  |  | float64 |  | NaN = not available | see audit report | no (identifier/provenance) | yes |
| `GDS_total_match_method` |  |  |  | object |  | NaN = not available | see audit report | no (identifier/provenance) | no |
| `MHPSYCH_raw` | Medical history: psychiatric | MEDHIST | MHPSYCH | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MHPSYCH_clean` | Medical history: psychiatric | MEDHIST | MHPSYCH | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MHPSYCH_decoded` | Medical history: psychiatric | MEDHIST | MHPSYCH | object | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH2NEURL_raw` | Medical history: neurological (non-AD) | MEDHIST | MH2NEURL | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH2NEURL_clean` | Medical history: neurological (non-AD) | MEDHIST | MH2NEURL | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH2NEURL_decoded` | Medical history: neurological (non-AD) | MEDHIST | MH2NEURL | object | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH3HEAD_raw` | Medical history: head trauma | MEDHIST | MH3HEAD | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH3HEAD_clean` | Medical history: head trauma | MEDHIST | MH3HEAD | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH3HEAD_decoded` | Medical history: head trauma | MEDHIST | MH3HEAD | object | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH4CARD_raw` | Medical history: cardiovascular (includes hypertension) | MEDHIST | MH4CARD | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH4CARD_clean` | Medical history: cardiovascular (includes hypertension) | MEDHIST | MH4CARD | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH4CARD_decoded` | Medical history: cardiovascular (includes hypertension) | MEDHIST | MH4CARD | object | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH5RESP_raw` | Medical history: respiratory | MEDHIST | MH5RESP | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH5RESP_clean` | Medical history: respiratory | MEDHIST | MH5RESP | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH5RESP_decoded` | Medical history: respiratory | MEDHIST | MH5RESP | object | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH6HEPAT_raw` | Medical history: hepatic | MEDHIST | MH6HEPAT | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH6HEPAT_clean` | Medical history: hepatic | MEDHIST | MH6HEPAT | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH6HEPAT_decoded` | Medical history: hepatic | MEDHIST | MH6HEPAT | object | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH7DERM_raw` | Medical history: dermatologic | MEDHIST | MH7DERM | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH7DERM_clean` | Medical history: dermatologic | MEDHIST | MH7DERM | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH7DERM_decoded` | Medical history: dermatologic | MEDHIST | MH7DERM | object | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH8MUSCL_raw` | Medical history: musculoskeletal | MEDHIST | MH8MUSCL | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH8MUSCL_clean` | Medical history: musculoskeletal | MEDHIST | MH8MUSCL | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH8MUSCL_decoded` | Medical history: musculoskeletal | MEDHIST | MH8MUSCL | object | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH9ENDO_raw` | Medical history: endocrine-metabolic (includes diabetes) | MEDHIST | MH9ENDO | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH9ENDO_clean` | Medical history: endocrine-metabolic (includes diabetes) | MEDHIST | MH9ENDO | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH9ENDO_decoded` | Medical history: endocrine-metabolic (includes diabetes) | MEDHIST | MH9ENDO | object | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH10GAST_raw` | Medical history: gastrointestinal | MEDHIST | MH10GAST | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH10GAST_clean` | Medical history: gastrointestinal | MEDHIST | MH10GAST | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH10GAST_decoded` | Medical history: gastrointestinal | MEDHIST | MH10GAST | object | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH11HEMA_raw` | Medical history: haematologic/lymphatic | MEDHIST | MH11HEMA | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH11HEMA_clean` | Medical history: haematologic/lymphatic | MEDHIST | MH11HEMA | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH11HEMA_decoded` | Medical history: haematologic/lymphatic | MEDHIST | MH11HEMA | object | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH12RENA_raw` | Medical history: renal-genitourinary | MEDHIST | MH12RENA | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH12RENA_clean` | Medical history: renal-genitourinary | MEDHIST | MH12RENA | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH12RENA_decoded` | Medical history: renal-genitourinary | MEDHIST | MH12RENA | object | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH13ALLE_raw` | Medical history: allergies/drug sensitivities | MEDHIST | MH13ALLE | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH13ALLE_clean` | Medical history: allergies/drug sensitivities | MEDHIST | MH13ALLE | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH13ALLE_decoded` | Medical history: allergies/drug sensitivities | MEDHIST | MH13ALLE | object | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH14ALCH_raw` | Medical history: alcohol abuse | MEDHIST | MH14ALCH | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH14ALCH_clean` | Medical history: alcohol abuse | MEDHIST | MH14ALCH | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH14ALCH_decoded` | Medical history: alcohol abuse | MEDHIST | MH14ALCH | object | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH15DRUG_raw` | Medical history: drug abuse | MEDHIST | MH15DRUG | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH15DRUG_clean` | Medical history: drug abuse | MEDHIST | MH15DRUG | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH15DRUG_decoded` | Medical history: drug abuse | MEDHIST | MH15DRUG | object | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH16SMOK_raw` | Medical history: smoking | MEDHIST | MH16SMOK | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH16SMOK_clean` | Medical history: smoking | MEDHIST | MH16SMOK | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH16SMOK_decoded` | Medical history: smoking | MEDHIST | MH16SMOK | object | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH17MALI_raw` | Medical history: malignancy | MEDHIST | MH17MALI | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH17MALI_clean` | Medical history: malignancy | MEDHIST | MH17MALI | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH17MALI_decoded` | Medical history: malignancy | MEDHIST | MH17MALI | object | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH18SURG_raw` | Medical history: major surgical procedures | MEDHIST | MH18SURG | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH18SURG_clean` | Medical history: major surgical procedures | MEDHIST | MH18SURG | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH18SURG_decoded` | Medical history: major surgical procedures | MEDHIST | MH18SURG | object | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH19OTHR_raw` | Medical history: other | MEDHIST | MH19OTHR | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH19OTHR_clean` | Medical history: other | MEDHIST | MH19OTHR | float64 | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `MH19OTHR_decoded` | Medical history: other | MEDHIST | MH19OTHR | object | 1 -> present, 0 -> absent, negative sentinel -> NaN; ADNI convention,  | NaN = not available | see audit report | yes | no |
| `cardiovascular_history` |  | MEDHIST | MH4CARD | float64 | BROAD category; includes hypertension but is NOT a hypertension-specif | NaN = not available | see audit report | yes | no |
| `endocrine_metabolic_history` |  | MEDHIST | MH9ENDO | float64 | BROAD category; includes diabetes but is NOT a diabetes-specific flag | NaN = not available | see audit report | yes | no |

## Missingness in the master table

| Variable | Present | Missing | % present |
|---|---:|---:|---:|
| `subject_id` | 127 | 0 | 100.0% |
| `group_dir` | 127 | 0 | 100.0% |
| `stage` | 127 | 0 | 100.0% |
| `stage_ordinal` | 127 | 0 | 100.0% |
| `APOE_genotype_raw` | 126 | 1 | 99.2% |
| `APOE4_allele_count` | 126 | 1 | 99.2% |
| `APOE4_carrier` | 126 | 1 | 99.2% |
| `APOE_source` | 127 | 0 | 100.0% |
| `Sex_raw` | 107 | 20 | 84.3% |
| `Sex_decoded` | 107 | 20 | 84.3% |
| `Age_years` | 107 | 20 | 84.3% |
| `Age_source_visit` | 107 | 20 | 84.3% |
| `Age_source_acq_date` | 107 | 20 | 84.3% |
| `MRI_date_used_for_matching` | 107 | 20 | 84.3% |
| `MRI_date_source` | 107 | 20 | 84.3% |
| `CDR_global` | 107 | 20 | 84.3% |
| `CDR_global_visit` | 107 | 20 | 84.3% |
| `CDR_global_days_from_MRI` | 107 | 20 | 84.3% |
| `CDR_global_match_method` | 107 | 20 | 84.3% |
| `CDR_SB` | 107 | 20 | 84.3% |
| `CDR_SB_visit` | 107 | 20 | 84.3% |
| `CDR_SB_days_from_MRI` | 107 | 20 | 84.3% |
| `CDR_SB_match_method` | 107 | 20 | 84.3% |
| `MMSE_total` | 107 | 20 | 84.3% |
| `MMSE_total_visit` | 107 | 20 | 84.3% |
| `MMSE_total_days_from_MRI` | 107 | 20 | 84.3% |
| `MMSE_total_match_method` | 107 | 20 | 84.3% |
| `GDS_total` | 107 | 20 | 84.3% |
| `GDS_total_visit` | 107 | 20 | 84.3% |
| `GDS_total_days_from_MRI` | 107 | 20 | 84.3% |
| `GDS_total_match_method` | 107 | 20 | 84.3% |
| `MHPSYCH_raw` | 107 | 20 | 84.3% |
| `MHPSYCH_clean` | 107 | 20 | 84.3% |
| `MHPSYCH_decoded` | 107 | 20 | 84.3% |
| `MH2NEURL_raw` | 107 | 20 | 84.3% |
| `MH2NEURL_clean` | 107 | 20 | 84.3% |
| `MH2NEURL_decoded` | 107 | 20 | 84.3% |
| `MH3HEAD_raw` | 107 | 20 | 84.3% |
| `MH3HEAD_clean` | 107 | 20 | 84.3% |
| `MH3HEAD_decoded` | 107 | 20 | 84.3% |
| `MH4CARD_raw` | 107 | 20 | 84.3% |
| `MH4CARD_clean` | 107 | 20 | 84.3% |
| `MH4CARD_decoded` | 107 | 20 | 84.3% |
| `MH5RESP_raw` | 107 | 20 | 84.3% |
| `MH5RESP_clean` | 107 | 20 | 84.3% |
| `MH5RESP_decoded` | 107 | 20 | 84.3% |
| `MH6HEPAT_raw` | 107 | 20 | 84.3% |
| `MH6HEPAT_clean` | 107 | 20 | 84.3% |
| `MH6HEPAT_decoded` | 107 | 20 | 84.3% |
| `MH7DERM_raw` | 107 | 20 | 84.3% |
| `MH7DERM_clean` | 107 | 20 | 84.3% |
| `MH7DERM_decoded` | 107 | 20 | 84.3% |
| `MH8MUSCL_raw` | 107 | 20 | 84.3% |
| `MH8MUSCL_clean` | 107 | 20 | 84.3% |
| `MH8MUSCL_decoded` | 107 | 20 | 84.3% |
| `MH9ENDO_raw` | 107 | 20 | 84.3% |
| `MH9ENDO_clean` | 107 | 20 | 84.3% |
| `MH9ENDO_decoded` | 107 | 20 | 84.3% |
| `MH10GAST_raw` | 107 | 20 | 84.3% |
| `MH10GAST_clean` | 107 | 20 | 84.3% |
| `MH10GAST_decoded` | 107 | 20 | 84.3% |
| `MH11HEMA_raw` | 107 | 20 | 84.3% |
| `MH11HEMA_clean` | 107 | 20 | 84.3% |
| `MH11HEMA_decoded` | 107 | 20 | 84.3% |
| `MH12RENA_raw` | 107 | 20 | 84.3% |
| `MH12RENA_clean` | 107 | 20 | 84.3% |
| `MH12RENA_decoded` | 107 | 20 | 84.3% |
| `MH13ALLE_raw` | 107 | 20 | 84.3% |
| `MH13ALLE_clean` | 107 | 20 | 84.3% |
| `MH13ALLE_decoded` | 107 | 20 | 84.3% |
| `MH14ALCH_raw` | 107 | 20 | 84.3% |
| `MH14ALCH_clean` | 107 | 20 | 84.3% |
| `MH14ALCH_decoded` | 107 | 20 | 84.3% |
| `MH15DRUG_raw` | 107 | 20 | 84.3% |
| `MH15DRUG_clean` | 107 | 20 | 84.3% |
| `MH15DRUG_decoded` | 107 | 20 | 84.3% |
| `MH16SMOK_raw` | 107 | 20 | 84.3% |
| `MH16SMOK_clean` | 107 | 20 | 84.3% |
| `MH16SMOK_decoded` | 107 | 20 | 84.3% |
| `MH17MALI_raw` | 107 | 20 | 84.3% |
| `MH17MALI_clean` | 107 | 20 | 84.3% |
| `MH17MALI_decoded` | 107 | 20 | 84.3% |
| `MH18SURG_raw` | 107 | 20 | 84.3% |
| `MH18SURG_clean` | 107 | 20 | 84.3% |
| `MH18SURG_decoded` | 107 | 20 | 84.3% |
| `MH19OTHR_raw` | 107 | 20 | 84.3% |
| `MH19OTHR_clean` | 107 | 20 | 84.3% |
| `MH19OTHR_decoded` | 107 | 20 | 84.3% |
| `cardiovascular_history` | 107 | 20 | 84.3% |
| `endocrine_metabolic_history` | 107 | 20 | 84.3% |
