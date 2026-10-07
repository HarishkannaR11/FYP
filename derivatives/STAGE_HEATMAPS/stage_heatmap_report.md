# Stage-wise Biomarker Heatmaps

Generated (UTC): 2026-10-07T14:33:22.842785+00:00

## 1. Dataset summary

- Acquisitions: **165**
- Distinct subjects: **127**
- Source: `derivatives/biomarkers_normalized/` (mALFF, mReHo, DC_z) and `derivatives/fsfast/.../biomarkers/` (FC_Strength)
- No existing derivative was modified; this stage is additive only.

## 2. Subject-level aggregation method

Acquisitions are averaged **within subject first**, then subjects are averaged within stage, so every subject contributes exactly once to its stage. Averaging acquisitions directly would over-weight the 31 subjects who contributed more than one scan (one contributed four).

## 3. Subject count per stage

| Stage | Subjects | Acquisitions | Matches expected |
|---|---:|---:|---|
| CN | 26 | 27 | yes |
| SMC | 21 | 31 | yes |
| EMCI | 26 | 29 | yes |
| MCI | 14 | 32 | yes |
| LMCI | 20 | 21 | yes |
| AD | 20 | 25 | yes |
| **Total** | **127** | **165** | |

## 4. Stage order

`CN -> SMC -> EMCI -> MCI -> LMCI -> AD`

Fixed at Review I; ROIs and stages are never reordered or clustered.

## 5. Formula

```
subject_mean[r, subject] = mean of that subject's acquisition values for ROI r
stage_mean[r, stage]     = mean of subject_mean[r, subject] over that stage
z[r, s]                  = (stage_mean[r,s] - mean(stage_mean[r,:]))
                           / std(stage_mean[r,:])
```

## 6. Biomarker heatmaps generated

| Biomarker | Shape | NaN | Inf | Range |
|---|---|---:|---:|---|
| mALFF | 246 x 6 | 0 | 0 | 0.4138 to 1.9321 |
| mReHo | 246 x 6 | 0 | 0 | 0.7819 to 1.3360 |
| DC_z | 246 x 6 | 0 | 0 | -2.1378 to 1.3493 |
| FC_Strength | 246 x 6 | 0 | 0 | -2.1378 to 1.3493 |

## 7. Absolute-value heatmap

Sequential colormap (`viridis`); each cell is the stage-wise mean of that ROI in the biomarker's own units. Shows which ROIs carry large values, which is driven largely by anatomy rather than by stage.

## 8. Row-wise z-score heatmap

Diverging colormap (`RdBu_r`) centred at 0; colorbar labelled "Row-wise z-score". Each ROI is standardised across the six stages independently, so the figure shows how each ROI *changes across stages* rather than how large it is.

## 9. FC_Strength redundancy verification

Checked on the original (un-normalized) biomarker outputs, where the stated relation applies:

- Acquisitions checked: **165**
- `max |DC/245 - FC_Strength|` = **6.939e-18**
- Minimum Pearson r across acquisitions = **1.000000000000**
- Relation holds: **True**

On the resulting stage matrices (both within-subject z-scored):

- `max |DC_z - FC_Strength|` = **6.661e-16**
- Pearson r = **1.000000000000**

> FC_Strength is a deterministic rescaling of DC and is therefore not an independent biomarker feature. It is shown here only for visualization/comparison and is excluded from model input.

Consequently the `FC_Strength` heatmaps are visually identical to the `DC_z` heatmaps.

## 10. QC results

| Check | Result |
|---|---|
| Output shape (246 x 6) | PASS |
| No NaN | PASS |
| No Inf | PASS |
| No NaN/Inf in row-z | PASS |
| Exactly 246 ROIs | PASS |
| No duplicate ROI IDs | PASS |
| No missing ROI IDs | PASS |
| Exactly 6 stages, correct order | PASS |
| Subject-level aggregation used | PASS |
| Subject counts match expected | PASS |
| No subject in multiple stages | PASS |

## 11. Warnings and failures

None.

---

**Interpretation note.** These are descriptive, exploratory visualizations of stage-wise regional biomarker patterns. They do not demonstrate Alzheimer's disease progression, and no statistical testing, model training, accuracy or AUC computation was performed in this task.
