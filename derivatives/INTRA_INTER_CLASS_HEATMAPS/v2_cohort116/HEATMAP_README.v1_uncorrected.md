# Significance Heatmaps — Cohort v2 (116 subjects)

Everything in this directory was computed from one cohort and one set of
feature files. Results from the earlier 114-subject analysis are **not**
mixed in; those live in `../pairwise/` and are untouched.

## Cohort

Fingerprint `6ab68ae6e80f61c2` (SHA-256 of sorted subject IDs + stage labels).

| Stage | Subjects | Acquisitions |
|---|---:|---:|
| CN | 20 | 20 |
| SMC | 20 | 20 |
| EMCI | 20 | 20 |
| MCI | **16** | 16 |
| LMCI | 20 | 20 |
| AD | 20 | 20 |
| **Total** | **116** | **116** |

**One acquisition per subject.** No subject appears twice, so observations are
independent and no repeated-measures model is required. MCI has 16 rather
than 20 because only 16 MCI subjects have valid preprocessing, biomarkers,
normalisation and QC — the shortfall is reported, not padded.

**Stage order is `CN → SMC → EMCI → MCI → LMCI → AD`**, the order fixed at
Review I and encoded in `model_ready` (MCI=3, LMCI=4). A request to place
LMCI before MCI was not applied, because it contradicts the ordinal encoding
used by every other output in the project.

## 1. `overall_significance_heatmap.png` / `.pdf` / `.csv`

**4 biomarkers × 6 stages = 24 cells.**

Each cell is one *one-vs-rest* test: that stage's subjects against the other
five stages **pooled**.

- **Statistic** — distance-based pseudo-F (two-group PERMANOVA) on z-scored
  features. Multivariate: ROIs/edges are features of a subject, never
  independent observations.
- **Null** — stage membership permuted across subjects, 9,999 permutations,
  seed `20261013`. Each test uses a fresh RNG seeded identically, so any
  single test is reproducible from (seed, biomarker, stage) alone.
- **p** — `(#{F_perm ≥ F_obs} + 1) / (N + 1)`, so the smallest attainable
  value is 1×10⁻⁴.
- **Representation** — regional biomarkers use their 246 ROI values; FC uses
  its 30,135 unique upper-triangle Fisher-z edges, diagonal excluded.
- **Correction** — Benjamini–Hochberg across all 24 tests.
- **Colour** — `−log10(q)`, floored at 1×10⁻⁴.

**Result: 0 of 24 cells reach q < 0.05.** Smallest q = 0.1392 (mReHo/EMCI,
raw p = 0.0058). The red line on the colour bar marks the q = 0.05 threshold;
no cell reaches it.

## 2. `<biomarker>_roi_stage_significance_heatmap.png` / `.pdf`

**246 ROIs × 6 stages, one figure per biomarker.**

For each ROI and stage, that stage is compared against the other five pooled.

- **Test** — Mann-Whitney U, two-sided (n = 16–20 vs 96–100; normality cannot
  be assumed at these sizes).
- **Effect size** — Hedges' g, stage minus rest, bias-corrected.
- **Correction** — Benjamini–Hochberg across **246 × 6 = 1,476 tests, declared
  per biomarker**.
- **Colour** — `−log10(q)`. Cyan stars would mark q < 0.05; there are none.

### The top annotation strip

The strip above each heatmap shows the **stage-level multivariate** q from
analysis 1. It is a different hypothesis at a different level and is labelled
as such on the figure. **It is not an ROI p-value** and was not used to fill
any ROI cell.

### How FC is represented

FC is a 246×246 network, not a regional scalar. The ROI-level summary used is
the **unsigned mean connectivity strength**, `FCstr_i = mean_{j≠i} |z_ij|`.

The obvious alternative — the signed row mean — equals DC/245 and correlates
with DC_z at **r = +0.994** in this cohort, so it would have reproduced DC_z
in a different colour. The unsigned strength correlates with DC_z at
**r = −0.090**: net signed connectivity versus total connectivity magnitude
are near-orthogonal here, so FC contributes genuinely separate information.

**Result: 0 ROIs reach q < 0.05 in any of the four biomarkers.**

| Biomarker | raw p<0.05 | expected by chance | q<0.05 | min q |
|---|---:|---:|---:|---:|
| mALFF | 121 | 74 | **0** | 0.2227 |
| mReHo | 109 | 74 | **0** | 0.1181 |
| DC_z | 77 | 74 | **0** | 0.7297 |
| FC | 108 | 74 | **0** | 0.3858 |

## 3. `pairwise/` — the 15 stage-comparison analyses

246 ROIs × 15 pairwise comparisons per biomarker, plus FC edge-level
matrices. Mann-Whitney U, Hedges' g, BH across 246 × 15 = 3,690 per
biomarker; FC edges form a separate family per comparison.

**Result: 0 of 3,690 per biomarker; 0 of 452,025 FC edge tests.**
Cross-biomarker consensus is 0 in every cell.

## How to read a null result

Nothing here survives multiple-comparison correction. That is the finding,
not a failure to produce one, and the figures show it rather than rescaling
colours to suggest otherwise.

Two things are worth separating:

1. **Effect sizes are not zero.** |g| reaches 1.07 at ROI level and the raw
   p<0.05 counts exceed chance for mALFF and mReHo. The effects are real in
   the sample; they are not large enough to survive correction for 1,476 or
   3,690 simultaneous tests at n = 16–20 per stage.
2. **The limit is resolution, not analysis choice.** With 20 vs 20 subjects,
   a Mann-Whitney p-value cannot fall below roughly 10⁻⁵ even under perfect
   separation. More subjects raise that ceiling; a smaller, pre-specified
   hypothesis family lowers the bar. No reasonable change of test or
   correction alters the conclusion.

Five independent analyses on this cohort agree: pairwise ROI, pairwise FC
edges, intra- versus inter-class permutation, ROI × stage one-vs-rest, and
stage-level one-vs-rest. The consistency indicates an absence of detectable
*univariate* stage signal at this sample size — which is the quantitative
argument for a multivariate model, not evidence against one.

## Limitations

- **One-vs-rest pools the other five stages** and therefore ignores the
  ordinal structure CN→AD. It asks "is this stage separable from everything
  else", not "do the stages progress". Exploratory.
- **MCI n = 16** against 20 elsewhere; its comparisons are the least powered.
- **FC edge-wise inference is underpowered**: 30,135 edges against 16–20
  subjects per group. The near-1.0 minimum q-values reflect that ratio, not
  an established absence of connectivity differences.
- **No causal claim** is supported by any figure here.
- **No model was trained** and no accuracy is reported.

## Reproducing

```
python scripts/audit/one_vs_rest_checkpointed.py  <out_dir>   # 24 tests, checkpointed
python scripts/audit/plot_overall_significance.py <out_dir>   # 4x6 figure
python scripts/audit/roi_stage_significance.py    <out_dir>   # 246x6 figures
python scripts/audit/pairwise_stage_statistics.py <out_dir>/pairwise
```

`one_vs_rest_checkpoint.jsonl` records every completed test with its cohort
fingerprint, permutation count and seed. On restart, entries are reused only
when all three match the current configuration, so results can never be
silently carried across cohorts.
