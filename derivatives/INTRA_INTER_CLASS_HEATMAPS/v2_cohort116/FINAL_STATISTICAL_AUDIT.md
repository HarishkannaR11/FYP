# Final Statistical and Data-Integrity Audit — Cohort v2 (116 subjects)

Audit of the significance heatmaps in this directory. Every check below was
executed against the files on disk, not inferred. One substantive error was
found and corrected; it is described in full.

Cohort fingerprint `6ab68ae6e80f61c2` · seed `20261013` · 9,999 permutations.

---

## Summary

| # | Check | Outcome |
|---|---|---|
| 1 | Stage ordering | **Discrepancy with the request, not with the data** |
| 2 | Mann-Whitney resolution claim | **ERROR FOUND — corrected** |
| 3 | FC ROI representation | Verified; one claim reworded |
| 4 | FDR families and CSV/plot agreement | Verified |
| 5 | Cohort, seed, permutations, no zero-fill | Verified |

---

## 1. Stage ordering

**The canonical order in this project is `CN → SMC → EMCI → MCI → LMCI → AD`.
It is NOT `CN → SMC → EMCI → LMCI → MCI → AD`.**

```
model_ready stage_order : ['CN','SMC','EMCI','MCI','LMCI','AD']
ordinal encoding        : CN=0  SMC=1  EMCI=2  MCI=3  LMCI=4  AD=5
```

Label-to-index agreement was checked in the data itself, not from the
metadata alone — counting rows by string label and by integer index:

| Stage | index | n (by label) | n (by index) | agree |
|---|---:|---:|---:|---|
| CN | 0 | 20 | 20 | yes |
| SMC | 1 | 20 | 20 | yes |
| EMCI | 2 | 20 | 20 | yes |
| MCI | 3 | 16 | 16 | yes |
| LMCI | 4 | 20 | 20 | yes |
| AD | 5 | 20 | 20 | yes |

**No plotted label disagrees with the data behind it.** Every figure and CSV
in this directory uses the column order `CN, SMC, EMCI, MCI, LMCI, AD`,
which is the order the integer labels encode. Verified in
`overall_significance_heatmap.csv` and the `*_roi_stage_*.csv` headers.

**The discrepancy is between the request and the project**, not within the
outputs. A task specification asked for LMCI before MCI, describing it as
the project's existing order. It is not: the ordinal encoding, the project
documentation and the Review II deck all place MCI before LMCI. The
project's order was used.

This cannot be fixed by relabelling. Swapping the labels alone would
mislabel the data; genuinely reordering would require changing the ordinal
encoding in `model_ready` and regenerating every downstream output, and
would break the ordinal head the model is designed around. **No change was
made.** If the clinical intent really is LMCI → MCI, that decision belongs
upstream in `build_transformer_inputs.py`, not in the plotting layer.

---

## 2. Mann-Whitney minimum p-value — ERROR FOUND AND CORRECTED

### The false claim

`HEATMAP_README.md` (now `HEATMAP_README.v1_uncorrected.md`) stated:

> "With 20 vs 20 subjects, a Mann-Whitney p-value cannot fall below roughly
> 10⁻⁵ even under perfect separation."

**This is false by six to seventeen orders of magnitude.**

### The correct values

Under complete separation the exact two-sided p is `2 / C(m+n, m)`:

| Group sizes | C(m+n, m) | exact min two-sided p | SciPy `auto` (asymptotic) |
|---|---:|---:|---:|
| 20 vs 20 | 137,846,528,820 | **1.45×10⁻¹¹** | 6.80×10⁻⁸ |
| 16 vs 20 | 7,307,872,110 | 2.74×10⁻¹⁰ | 3.82×10⁻⁷ |
| 20 vs 96 | 1.41×10²² | 1.42×10⁻²² | 2.34×10⁻¹² |
| 16 vs 100 | 1.74×10¹⁹ | 1.15×10⁻¹⁹ | 1.54×10⁻¹⁰ |

SciPy's `mannwhitneyu` with default `method='auto'` uses the **asymptotic
normal approximation** at these sample sizes, not the exact distribution.
That is the appropriate choice (the exact computation is intractable for
20 vs 96) and it is what produced every p-value in this directory. Even the
asymptotic floor, 6.8×10⁻⁸ at 20 vs 20, is far below any threshold in use.

### What this means for the conclusions

The null results stand, but **the stated reason for them was wrong.**

Observed minimum raw p-values, against the BH rank-1 threshold of
0.05 / 1476 = 3.39×10⁻⁵:

| Biomarker | min raw p | ratio to rank-1 threshold |
|---|---:|---:|
| mReHo | 9.65×10⁻⁵ | 2.8× too large |
| mALFF | 1.51×10⁻⁴ | 4.5× too large |
| FC | 6.17×10⁻⁴ | 18× too large |
| DC_z | 1.07×10⁻³ | 32× too large |

mReHo came within a factor of three. **The test had ample resolution; the
effects were not large enough.** The limiting factor is effect size and
sample size acting on the standard error — not a floor on attainable
p-values.

### Correction applied

- Original preserved verbatim as `HEATMAP_README.v1_uncorrected.md`.
- `HEATMAP_README.md` now withdraws the claim explicitly, gives the correct
  exact and asymptotic minima, and attributes the null result to effect size
  rather than resolution.
- No statistical output was altered. The error was in the written
  interpretation only; no p-value, q-value or figure depended on it.

---

## 3. FC ROI-level representation

### How it is computed

For each subject and each ROI *i*:

```
FCstr_i = mean over j != i of |z_ij|
```

Verified directly against the source matrices:

- Source `X_fc.npy` is `(116, 246, 246)`, float32, symmetric
  (`allclose(M, Mᵀ)` true), finite throughout.
- **Diagonal**: already exactly zero in the pipeline output
  (`max|diag| = 0.000`), and additionally excluded by an explicit
  off-diagonal mask. Each ROI therefore averages exactly **245 edges**.
- **Negative values**: 48.8% of upper-triangle edges are negative. They are
  retained and enter through their absolute value — not discarded, not
  clipped.
- **Missing edges**: none. All values finite; no imputation occurs.
- Independent recomputation for three subjects matched the stored array to
  `1.5×10⁻⁸` (float32 round-off).
- Stored summary: `(116, 246)`, all values ≥ 0, range 0.0943–0.9053.

### Why it is appropriate, and how it differs from DC_z

DC_z is the **signed** degree centrality — the net sum of a region's
connectivity, in which a strong positive and a strong negative edge cancel.
The unsigned mean |z| measures **total connectivity magnitude**, in which
both contribute.

These answer different questions. A region may be strongly embedded in the
connectome while having near-zero net degree, because its positive and
anticorrelated couplings offset. DC_z cannot distinguish that region from a
weakly connected one; unsigned strength can. Given that nearly half of all
edges here are negative, the distinction is substantive rather than
theoretical.

The signed row mean was **excluded** because it equals DC/245 exactly and
reproduces DC_z (r = 0.97–0.998 across sampled subjects) — including it
would have presented the same quantity twice under two names.

**Correction of a prior statement**: the low correlation between unsigned
strength and DC_z (r = −0.09 on average) establishes **non-redundancy
only**. It does not establish that unsigned strength is a better or more
valid measure. The justification for using it is the conceptual one above;
the correlation merely confirms the two are not duplicates. No claim of
scientific superiority is made or supported.

---

## 4. FDR correction

### Overall 4 × 6 family

- Family size **24**, as declared. Verified: `len(p) == 24`.
- BH recomputed independently from the stored raw p-values reproduces the
  stored q-values exactly: **max |difference| = 0.000×10⁰**.
- All 24 corrected **together as one family**, not per biomarker or per
  stage.

### ROI × stage families

Each regional biomarker declares 246 ROIs × 6 stages = **1,476** hypotheses,
corrected separately per biomarker.

| Biomarker | family size | == 1476 | BH recompute matches | max \|diff\| |
|---|---:|---|---|---:|
| mALFF | 1476 | yes | yes | 3.29×10⁻⁶ |
| mReHo | 1476 | yes | yes | 2.50×10⁻⁶ |
| DC_z | 1476 | yes | yes | 2.51×10⁻⁶ |
| FC | 1476 | yes | yes | 2.55×10⁻⁶ |

Residuals of ~10⁻⁶ are the CSV's six-significant-figure text rounding, not a
procedural difference.

### Plot / CSV agreement

Plotted values are read from the same arrays written to CSV within a single
process run; the CSV round-trip was re-read and re-corrected above with the
agreement shown. Raw p and adjusted q are stored in **separate files**
(`*_roi_stage_pvalues.csv`, `*_roi_stage_qvalues.csv`) and are never
interchanged.

### Top annotation strips

**Confirmed: the strips above each ROI × stage heatmap carry the stage-level
multivariate one-vs-rest q-values**, drawn from
`one_vs_rest_statistics.csv` — a different hypothesis at a different level
(whole-feature-vector separation, permutation-based). They are **not**
ROI-specific p-values, are plotted on their own axis with their own scale,
and are captioned on each figure as "NOT an ROI p-value". No ROI cell was
filled from them.

---

## 5. Final validation

| Check | Result |
|---|---|
| Cohort fingerprint, all 24 tests | `{'6ab68ae6e80f61c2'}` — single value |
| Random seed | `{'20261013'}` — single value |
| Permutations | `{'9999'}` — single value |
| All tests COMPLETED | yes, 24/24 |
| Missing results replaced by zero | **none** — 0 rows with empty or zero p |
| Per-stage n in results | CN 20, SMC 20, EMCI 20, **MCI 16**, LMCI 20, AD 20 |
| 114-subject results mixed in | **no** — those remain in `../pairwise/`; the fingerprint check in the checkpoint loader rejects any entry from a different cohort |
| Canonical stage order in all figures/CSVs | yes, `CN, SMC, EMCI, MCI, LMCI, AD` |
| Figures open and non-empty | 10/10 valid PNG/PDF, 0 bad |

---

## Findings and corrections

**One substantive error, corrected:** the Mann-Whitney resolution-floor claim
in `HEATMAP_README.md` (section 2). Written interpretation only; no computed
result depended on it. Original preserved as
`HEATMAP_README.v1_uncorrected.md`.

**One claim reworded:** the FC/DC_z correlation is now described as
establishing non-redundancy rather than implying superiority.

**One unresolved discrepancy, deliberately not "fixed":** a specification
requested LMCI before MCI. The project encodes MCI before LMCI. The project
order was retained and the conflict is documented rather than silently
resolved in either direction.

**No statistical output was regenerated or overwritten during this audit.**

---

## Unresolved limitations

- **MCI n = 16** against 20 elsewhere. Its comparisons are the least
  powered, and the shortfall against the 20-per-stage intent is real.
- **One-vs-rest pools the other five stages** and ignores the ordinal
  structure CN → AD. It tests separability, not progression.
- **FC edge-wise inference is underpowered**: 30,135 edges against 16–20
  subjects per group. The near-1.0 minimum q-values reflect that ratio.
  Zero significant edges is not evidence that no connectivity differences
  exist.
- **No causal claim** is supported by any output here.
- **No model has been trained** and no accuracy is reported anywhere in this
  directory.
- **Nothing reaches q < 0.05.** No result in this directory should be
  described as statistically significant.
