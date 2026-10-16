# new_comorbidity — Description (revised)

Folder `Comorbidity/new_comorbidity/` holds 6 CSVs, one per stage. Each has only the comorbidity data (the **MEDHIST** table, initial medical history) of the subjects of that stage in `acquisition_inventory.csv`. Built by `scripts/clinical/build_new_comorbidity.py`; ZIPs are only read.

## What the ZIPs contain
Each `Tables_*.zip` (AD, CN, EMCI, MCI, SMC, and the new `Tables_12Oct2026_LMCI.zip`) holds six ADNI tables for the whole cohort: CDR, MMSE, GDSCALE, **MEDHIST**, APOERES, Functional_MRI_Images. The LMCI ZIP also has `DATADIC`. MEDHIST is the comorbidity table. It has yes/no flags for psychiatric, neurologic, cardiac, respiratory, hepatic, endocrine, GI, renal, alcohol, drug, smoking, malignancy and other history (`MHPSYCH`, `MH2NEURL` … `MH19OTHR`), plus `PTID`, `VISCODE`, `VISDATE`.

## Output files
Original columns and values, unchanged. Rows are in original order.

| File | Subjects found / in inventory | Rows |
|---|---|---|
| `AD_comorbidity.csv` | 20 / 20 | 20 |
| `CN_comorbidity.csv` | 20 / 20 | 24 |
| `EMCI_comorbidity.csv` | 20 / 20 | 25 |
| `LMCI_comorbidity.csv` | 20 / 20 | 20 |
| `MCI_comorbidity.csv` | 14 / 14 | 27 |
| `SMC_comorbidity.csv` | 20 / 20 | 20 |

Subjects come from `selected_acquisitions.csv`, which has no duplicate subjects. Each file was checked against its ZIP's MEDHIST table: header and rows are identical to the ZIP rows for these subjects, with none missing or extra. Some subjects have more than one MEDHIST row (one per ADNI phase), so rows can exceed subjects.
