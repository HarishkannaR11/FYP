# Clinical Features: MMSE, GDSCALE, Global CDR

Built from `Comorbidity/new_comorbidity/gdscale-cdr-mmse/idaSearch_10_20_2026_*.csv`
by `scripts/comorbidity/build_clinical_features.py`. Source files are read-only
and unmodified.

```
9,784 export rows
  ->   709 rows belonging to the 116 cohort subjects
  ->   670 visit rows after removing 39 exact duplicates
  ->   116 subject rows
```

| file | contents |
|---|---|
| `clinical_visits_116.csv` | 670 rows — every visit, every subject, cleaned |
| `clinical_subject_116.csv` | 116 rows — baseline / last / median / min / max / range per score |
| `clinical_qc_report.txt` | full run log: missingness, range checks, cross-checks |

**Cleaning applied:** blanks parsed as missing (never as 0); whitespace
stripped; 39 exact duplicate rows removed; rows sorted by subject then age;
every value range-checked against its instrument (MMSE 0–30, GDSCALE 0–15,
Global CDR ∈ {0, 0.5, 1, 2, 3}).

**Checks that passed:** all 116 subjects found; the export's `Research Group`
agrees with the cohort stage on all 709 rows; no value fell outside its
instrument range; no blank ages; no visit carrying conflicting scores.
Observed Global CDR values are exactly {0, 0.5, 1, 2}.

**`*_median` uses `median_low`, deliberately.** These are ordinal clinical
scales, so averaging the two middle visits would invent levels that do not
exist — MMSE 12.5 on an integer scale, Global CDR 0.25 or 1.5 when the defined
levels are 0/0.5/1/2/3. The first version of this table did exactly that and
produced 46 such values. `median_low` always returns a value actually
measured.

**Missing — not imputed, by design:**

| score | subjects with ≥1 value | missing entirely |
|---|---|---|
| MMSE | 113 / 116 | `018_S_5250`, `018_S_5262`, `019_S_5180` |
| GDSCALE | 112 / 116 | the same three plus `130_S_4990` |
| Global CDR | 113 / 116 | the same three |

---

## Problem 1: no value can be tied to the scan

The IDA export carries **Age but no study date**, so a visit cannot be matched
to the rs-fMRI scan the cohort used. 112 of 116 subjects have more than one
visit, and the values move between them:

| score | changes across visits in | largest within-subject swing |
|---|---|---|
| MMSE | 77 subjects | **12 points** |
| GDSCALE | 59 subjects | 6 points |
| Global CDR | 24 subjects | 1.0 — a full severity level |

Matching was attempted three ways and all three failed:

1. **Scan date ↔ age.** Scan dates exist in `audit/bids_conversion/bids_mapping.csv`,
   but converting a date to an age needs a date of birth, which ADNI does not
   release. A DOB can be fitted when a subject has several scans, but most have
   only one, which leaves the fit underdetermined — any age fits.
2. **Series description.** 103 of 116 subjects have several visits sharing the
   same description, so it does not single one out.
3. **Visit ordering.** The export lists ADNI visits the project never
   downloaded, so position *n* in one list is not position *n* in the other.

`baseline`, `last` and `median` are therefore all provided. **Pick one
deliberately and state which was used.** Do not let the choice drift between
analyses — on MMSE it is worth up to 12 points.

**The fix:** re-run the IDA search with a study date or visit-code column
included, then the scan date in `bids_mapping.csv` matches it directly. Worth
doing before this feature is used in anything final.

---

## Problem 2: these scores partly define the label

This is the more serious one. ADNI's diagnostic criteria use MMSE and CDR
cut-offs to **assign** the very stage the model predicts. The scores are not
independent measurements of the subject — they are inputs to the labelling
rule. The cohort shows it plainly:

**Global CDR (median) × cohort stage**

| stage | 0.0 | 0.5 | 1.0 |
|---|---|---|---|
| CN | **18** | 2 | |
| SMC | **12** | 4 | 1 |
| EMCI | 2 | **18** | |
| MCI | 3 | **12** | 1 |
| LMCI | | **18** | 2 |
| AD | | 6 | **13** |

CDR 0 → CN/SMC, CDR 0.5 → the whole MCI band, CDR 1 → AD. That is the
enrolment rule, read back.

**5-fold CV balanced accuracy, 112 subjects with all three scores (chance 0.167):**

| features | balanced accuracy |
|---|---|
| **Global CDR alone** | **0.386 ± 0.057** |
| all three clinical scores | 0.369 ± 0.075 |
| MMSE + Global CDR | 0.361 ± 0.082 |
| MMSE alone | 0.342 ± 0.068 |
| GDSCALE alone | 0.197 ± 0.027 |
| *rs-fMRI imaging model, for comparison* | *0.307 CV / 0.200 held-out* |

**One number from a clinical interview beats 246 brain regions of rs-fMRI.**
That is not a finding about the brain; it is a finding about how the labels
were made.

### What this means for use

- **Do not** add these to the imaging model and report the accuracy gain as a
  result. The gain is mostly the labelling rule returning to you, and a
  reviewer who knows ADNI will say so immediately.
- **Do** report the table above as a baseline. "Global CDR alone reaches 0.400
  while our imaging model reaches 0.307" is an honest, interesting result that
  frames what the imaging is up against.
- **GDSCALE is the exception.** Depression score is not part of the diagnostic
  cut-offs, so it is the one of the three that can be used as a genuine
  comorbidity covariate. On its own it reaches 0.242 — weak, but not circular.
- If the three are used together anyway, say in the write-up that MMSE and CDR
  are label-defining and that the resulting accuracy is not evidence the
  imaging carries signal.
