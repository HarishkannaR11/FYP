# P-Score Figures

```
pscore_figures/
├── png/              the 4 figures to use   (150 dpi)
├── pdf/              the same 4, vector
├── logs/             run log
└── other_variants/   earlier alternative designs, kept for reference
```

## The figures — `png/` and `pdf/`

`group_hierarchical_pscore_top_{mALFF,mReHo,DC_z,FC}`

Derived from `figures/group_hierarchical_<bm>.png`, but the quantity on
display is different: **every cell is a P-score**, not a biomarker value.

| element | what it is now |
|---|---|
| heatmap values | ROI-level **P-score** = −log10(FDR-adjusted q), group vs pooled rest; BH over 246 × 6 = 1,476 tests per biomarker |
| colour | diverging RdBu_r, **two-segment, per figure** — see below |
| ROI dendrogram (left) | **removed**; rows keep their clustered order, the tree is not drawn |
| group dendrogram (top) | correlation distance + average linkage, on the **P-score** profile |
| branch labels | `d` = merge distance; `q*` = selective permutation q |
| P-score row | group P-score and max ROI P at each leaf stem |

The tree clusters on **P-score profiles**, not on biomarker activity, so the
dendrogram and the cells below it encode the same quantity.

## The colour scale — read this before interpreting colour

The ramp is blue → pale → red, but it is **not** centred on zero, because a
P-score has no sign and no zero crossing. It is a two-segment linear map:

| anchor | meaning |
|---|---|
| deepest blue | P = 0 (q = 1.000), no evidence at all |
| pale middle | the **median P-score of that figure** |
| deepest red | the **99th percentile** of that figure; higher cells are clipped, shown by the arrow on the colourbar |

Per-figure anchors, from `logs/group_hierarchical_pscore_top.log`:

| biomarker | median (pale) | p99 (red) | cells clipped | true max |
|---|---|---|---|---|
| mALFF | 0.0266 | 0.2841 | 1 / 1476 | 0.6523 |
| mReHo | 0.0348 | 0.4985 | 9 / 1476 | 0.9278 |
| DC_z | 0.0087 | 0.1199 | 14 / 1476 | 0.1369 |
| FC | 0.0743 | 0.3271 | 4 / 1476 | 0.4137 |

Three consequences, all of which must be stated when the figure is shown:

1. **Pale is not neutral and not zero.** It is the typical ROI in that panel.
   Red means "above-median evidence *within this panel*" — nothing more.
2. **The panels are not comparable.** Each has its own anchors, so the same
   red is q ≈ 0.52 in mALFF and q ≈ 0.76 in DC_z.
3. **No colour on any panel reaches significance.** q = 0.05 needs a P-score
   of 1.301. The largest value anywhere in the four figures is 0.928, so the
   threshold is off-scale on all of them.

A linear 0 → max ramp was tried first and rejected: 97–99.9% of cells fall
below half the figure maximum, so every panel rendered as one flat blue block
with no visible structure.

## The two numbers in the P-score row

**`group P-score`** — the real statistic. −log10(q) from the **multivariate**
stage-versus-rest test: distance-based pseudo-F over all 246 ROIs jointly,
9,999 permutations, Benjamini–Hochberg across the 4 × 6 = 24 family.
Source: `../../one_vs_rest_statistics.csv`. One valid value per group.

**`max ROI P`** — descriptive only. The largest ROI-level P-score for that
group, i.e. the maximum of 246 already-corrected values from
`../../<bm>_roi_stage_qvalues.csv`. **Not itself a test.**

Per-group values: `../csv/group_hierarchical_pscore_top_annotations.csv`

## Result

| | |
|---|---|
| Groups with multivariate q < 0.05 | **0 of 24** |
| Largest group P-score | **0.856** |
| Largest max-ROI P | **0.928** |
| Significance threshold | **1.301** |

**Nothing reaches significance.** The heatmap shows biomarker *activity*; the
top row shows statistical *significance*. They are different quantities, which
the figure states explicitly.

Branch heights are merge distances, not p-values.

## `other_variants/` — 50 files

Earlier designs produced while converging on the above. Kept, not deleted, in
case any is wanted:

| set | what it was |
|---|---|
| `pscore_grouptree_*` | P-score heatmap, fixed 0–1.40 scale, no ROI dendrogram |
| `pscore_grouptree_dist_*` | + merge-distance branch labels |
| `pscore_grouptree_annotated_*` | + selective permutation q per branch |
| `pscore_grouptree_groupscore_*` | + per-group P-scores above the tree |
| `pscore_reference_style_*` | P-score cells, per-figure colour scale, ROI dendrogram restored |
| `pscore_hierarchical_combined_4panel` | four panels, P-score cells |
| `diagnostic_group_hierarchical_pscore_four_biomarkers` | four panels, 300 dpi |

Note: in `pscore_reference_style_*` the colour scale is **per figure**, so the
same red means a different q in each and the 1.301 threshold is off-scale.
Those are useful for seeing relative variation, not for any claim about
significance.

## Reproducing

```bash
python scripts/audit/group_hierarchical_with_pscore_top.py
```

PNG output is 150 dpi because the machine ran out of memory at 220 (commit
charge reached 99.7%). The PDFs are vector and unaffected. Raise `dpi` in the
script and re-run with memory free if higher-resolution rasters are needed.
