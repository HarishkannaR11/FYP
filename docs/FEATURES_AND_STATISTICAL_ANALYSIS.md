# Features and Statistical Analysis

Documents the biomarkers and every statistical procedure **as implemented**.
Results are in [RESULTS_AND_INTERPRETATION.md](RESULTS_AND_INTERPRETATION.md).

---

## 1. Parcellation

**Brainnetome-246 atlas** (Fan et al., 2016), `atlases/Brainnetome246/`:

- `BN_Atlas_246_2mm.nii.gz` — 246 labelled regions, 210 cortical + 36 subcortical
- `BN_Atlas_246_LUT.txt` — official lookup table

The atlas is reoriented LAS → RAS losslessly, then resampled onto **each
acquisition's own BOLD grid with nearest-neighbour interpolation only** —
never a continuous interpolation, which would invent fractional labels.

Verified coverage (`dataset_brainnetome_summary.csv`): **246/246 ROIs
present**, 0 zero-voxel, 0 below 10 voxels; voxel counts 10 (min), 68
(median), 186 (max). Because every acquisition registers onto the identical
template-derived 4 mm grid, this holds identically for all of them.

**ROI ordering** is the atlas's own index 1…246 and is preserved everywhere.
Odd/even indices alternate left/right hemisphere (ROI 1 = `A8m_L`,
ROI 2 = `A8m_R`), so ROIs `2k−1` and `2k` are homologues. Every statistical
CSV carries `ROI_ID` and `ROI_NAME` from the LUT.

A region's time series is the mean of its voxels, giving **T × 246** per
acquisition.

---

## 2. Biomarker definitions

### ALFF — amplitude of low-frequency fluctuation

Mean FFT amplitude within 0.01–0.10 Hz, computed voxel-wise then averaged
per ROI (Zang et al., 2007).

$$\mathrm{ALFF}_i = \frac{1}{|F|}\sum_{f \in F} |X_i(f)|, \qquad F = [0.01, 0.10]\,\mathrm{Hz}$$

*How strongly a region fluctuates.*

### ReHo — regional homogeneity

Kendall's coefficient of concordance *W* over a 27-voxel neighbourhood
(3×3×3, centre included), with boundary-adaptive *K* (Zang et al., 2004).

$$W = \frac{12\sum_{t}\left(R_t - \bar{R}\right)^2}{K^2 (T^3 - T)}$$

where $R_t$ is the summed rank at time *t* across the *K* neighbours.

*How synchronised a region's voxels are with each other.*

### FC — functional connectivity

Pearson correlation between every pair of ROI time series: a **246 × 246**
matrix, **signed and unthresholded**, diagonal retained in the raw output.

### DC — degree centrality

$$\mathrm{DC}_i = \sum_{j \neq i} \mathrm{FC}_{ij}$$

*How connected a region is to the rest of the brain, net of sign.*

### FC_Strength — excluded

$$\mathrm{FCS}_i = \frac{1}{245}\sum_{j\neq i}\mathrm{FC}_{ij} = \frac{\mathrm{DC}_i}{245}$$

Verified empirically: maximum absolute difference **6.939 × 10⁻¹⁸**, Pearson
r = **1.000000000000**. It is a rescaling of DC and carries no independent
information. **It is not used as a model feature.** The raw files are
retained for traceability only.

---

## 3. Within-acquisition normalisation

`scripts/biomarkers/normalize_within_subject.py`. Every statistic is computed
from **one acquisition's own 246 values** — no cross-subject statistic and no
label is involved, so the procedure is leakage-free by construction.

| Feature | Transform | Rationale |
|---|---|---|
| mALFF | $\mathrm{ALFF}_i / \overline{\mathrm{ALFF}}$ | ALFF is strictly positive; a ratio is the standard normalisation (Zou et al., 2008). A z-score would be inappropriate for a positive-only quantity. |
| mReHo | $\mathrm{ReHo}_i / \overline{\mathrm{ReHo}}$ | same reasoning |
| DC_z | $(\mathrm{DC}_i - \overline{\mathrm{DC}})/\sigma_{\mathrm{DC}}$ (ddof = 0) | DC is signed, so a z-score is well defined |
| FC | $\operatorname{artanh}\!\big(\mathrm{clip}(r, \pm 0.999999)\big)$, **diagonal zeroed first** | variance-stabilising; the diagonal is zeroed *before* `artanh` so `artanh(1) = ∞` is never produced |

Verified across all acquisitions: `mean(mALFF) = mean(mReHo) = 1.000000`,
`std(DC_z) = 1.000000`, **0 off-diagonal entries required clipping**, Fisher-z
diagonal exactly 0, 0 NaN/Inf in inputs or outputs.

---

## 4. Feature arrays

`derivatives/model_ready/`

```
X_regional.npy   (116, 246, 3)    float32   channel 0 = mALFF, 1 = mReHo, 2 = DC_z
X_fc.npy         (116, 246, 246)  float32   Fisher-z, symmetric, diagonal 0
y.npy            (116,)           int64     CN=0 SMC=1 EMCI=2 MCI=3 LMCI=4 AD=5
subjects.npy     (116,)           str
```

Row *n* of every array refers to the same acquisition; `index.csv` maps row →
acquisition → source directory.

Verified before writing: all finite, FC symmetric to 1e-5, FC diagonal zero
to 1e-6, no zero-variance acquisition, labels in range, one row per
acquisition.

### ROI-level FC summary

FC is a network, not a regional scalar. For analyses needing one value per
ROI, the summary used is **unsigned mean connectivity strength**:

$$\mathrm{FCstr}_i = \frac{1}{245}\sum_{j \neq i} \left| z_{ij} \right|$$

Verified: diagonal already exactly zero and additionally masked, so exactly
**245 edges** per ROI; 48.8% of edges are negative and enter via absolute
value; no missing edges; independent recomputation matched the stored array
to 1.5 × 10⁻⁸.

**Why not the signed row mean?** It equals DC/245 and correlates with DC_z at
**r = +0.994** — it would present DC_z twice under two names. The unsigned
version correlates at **r = −0.090**.

**What that correlation does and does not establish**: it establishes
**non-redundancy only**. It does not make unsigned strength a better or more
valid measure. The justification is conceptual — DC_z is *net signed*
connectivity, in which strong positive and negative edges cancel; unsigned
strength is *total magnitude*, in which both contribute. With nearly half of
all edges negative, a region can be heavily embedded in the connectome while
having near-zero net degree. The two answer different questions.

---

## 5. Statistical procedures

Three distinct analyses, never mixed. Each declares its own hypothesis family.

### A. Pairwise stage comparisons — 246 ROIs × 15 comparisons

`scripts/audit/pairwise_stage_statistics.py`

All 15 pairs in fixed order: CN–SMC, CN–EMCI, CN–MCI, CN–LMCI, CN–AD,
SMC–EMCI, SMC–MCI, SMC–LMCI, SMC–AD, EMCI–MCI, EMCI–LMCI, EMCI–AD,
MCI–LMCI, MCI–AD, LMCI–AD.

- **Test** — Mann-Whitney *U*, two-sided
- **Effect size** — Hedges' *g*, first stage minus second:

$$g = \frac{\bar{x}_1 - \bar{x}_2}{s_p}\left(1 - \frac{3}{4(n_1+n_2)-9}\right), \qquad s_p = \sqrt{\frac{(n_1-1)s_1^2+(n_2-1)s_2^2}{n_1+n_2-2}}$$

- Rank-biserial correlation also stored, as the effect size that *matches*
  the rank test — Hedges' *g* is reported for interpretability and is **not**
  the statistic the p-value comes from.
- **FDR family** — 246 × 15 = **3,690 tests per biomarker**
- **FC** — 30,135 upper-triangle edges, diagonal excluded, **a separate
  family per comparison**, never pooled with ROI tests

### B. ROI × stage one-vs-rest — 246 ROIs × 6 stages

`scripts/audit/roi_stage_significance.py`

For each ROI and stage, that stage's subjects versus the other five
**pooled**. A genuine per-stage design — **not** the 15 pairwise results
reshaped.

- **Test** — Mann-Whitney *U*, two-sided (n = 16–20 vs 96–100)
- **Effect size** — Hedges' *g*, stage minus rest
- **FDR family** — 246 × 6 = **1,476 tests per biomarker**

### C. Stage-level multivariate one-vs-rest — 4 × 6

`scripts/audit/one_vs_rest_checkpointed.py`

Each cell is one multivariate test of a stage against the other five pooled.

**Statistic** — distance-based pseudo-F, the two-group PERMANOVA statistic
on squared Euclidean distances over z-scored features:

$$F = \frac{SS_B / (k-1)}{SS_W / (n-k)}, \qquad k = 2$$

ROIs and edges are **features of a subject**, never independent
observations.

An algebraic identity is used for speed. For two groups:

$$SS_B = \frac{n_g n_r}{n}\lVert \bar{x}_g - \bar{x}_r \rVert^2, \qquad SS_W = SST - SS_B$$

with *SST* constant across permutations, so each permutation needs only the
stage group's row-sum. **This is exact, not an approximation** — verified
against the literal definition on random permutations of every biomarker:
maximum relative difference **8.2 × 10⁻¹⁶**. The script aborts if the
discrepancy exceeds 1e-9.

**Null** — stage membership permuted **across subjects**. Since each subject
contributes exactly one acquisition, permuting subject labels is exactly
permuting observations; no within-subject dependence exists to preserve.

$$p = \frac{\#\{F_{\text{perm}} \geq F_{\text{obs}}\} + 1}{N + 1}$$

The +1 makes the estimate exact-conservative and bounds it away from zero, so
with N = 9,999 the smallest attainable p is **1 × 10⁻⁴**.

- **Permutations** — 9,999
- **Seed** — `20261013`, a *fresh RNG per test*, so any single test is
  reproducible from (seed, biomarker, stage) alone
- **Representation** — regional biomarkers use 246 ROI values; FC uses its
  30,135 unique upper-triangle Fisher-z edges
- **FDR family** — all **24 tests together**

A parametric multivariate test (Hotelling's *T*²) is not usable here: with
p ≫ n the pooled covariance is singular. A distance-based statistic is well
defined regardless, which is why it is used for all four biomarkers rather
than switching methods between them.

### Multiple-comparison correction

Benjamini–Hochberg throughout. The *k*-th smallest of *m* p-values is
significant if

$$p_{(k)} \leq \frac{k}{m}\alpha, \qquad \alpha = 0.05$$

Each family is declared explicitly above and corrected independently. Raw
p-values and adjusted q-values are stored in **separate files** and are never
interchanged.

---

## 6. Heatmap conventions

| Figure | Rows × cols | Colour | Scale |
|---|---|---|---|
| `overall_significance_heatmap` | 4 × 6 | −log₁₀(q) | floor 1e-4 |
| `<biomarker>_roi_stage_significance_heatmap` | 246 × 6 | −log₁₀(q) | floor 1e-4 |
| `pairwise_<biomarker>` | 246 × 15 | Hedges' *g* (diverging, centred 0), plus binary significance panel | — |
| `cross_biomarker_consensus` | 246 × 15 | count 0–3 | — |

**The −log₁₀(q) floor is 1 × 10⁻⁴**, equal to the smallest attainable
permutation p-value. Without it, a q of exactly 0 would render as infinity.

**Top annotation strips** on the ROI × stage figures carry the *stage-level
multivariate* q-values from analysis C. They are a **different hypothesis at
a different level** — plotted on their own axis, with their own scale, and
captioned on every figure as "NOT an ROI p-value". No ROI cell was ever
filled from them.

**Cross-biomarker consensus** counts, per ROI and comparison, how many of
mALFF/mReHo/DC_z reach q < 0.05:

$$C(r,c) = \sum_{b} \mathbb{1}\left[q(r,c,b) < 0.05\right] \in \{0,1,2,3\}$$

This is a **descriptive count, not a combined p-value**. The three biomarkers
are correlated (6–11% shared variance), so agreement among them is **not**
independent replication.

---

## 7. Checkpointing and cohort fingerprinting

`one_vs_rest_checkpoint.jsonl` records each completed test atomically
(tmp file → `fsync` → `os.replace`), storing biomarker, stage, observed *F*,
raw p, permutation count, seed, feature count, status, error, and a **cohort
fingerprint**:

```
SHA-256( sorted subject IDs ‖ stage labels )[:16]  =  6ab68ae6e80f61c2
```

On restart, an entry is reused **only** if its fingerprint, permutation
count, seed and status all match the current configuration. Results from the
earlier 114-subject cohort are therefore structurally incapable of leaking
into the 116-subject analysis.

If any of the 24 tests is missing or failed, the plotting script **refuses to
render** rather than drawing a gap as zero.

---

## 8. Corrected statistical claim

An earlier version of `HEATMAP_README.md` stated that a Mann-Whitney
p-value "cannot fall below roughly 10⁻⁵" at these sample sizes and used that
to explain the null results. **This was false and has been withdrawn.**

Under complete separation the exact two-sided minimum is $2/\binom{m+n}{m}$:

| Sizes | Exact minimum | SciPy default (asymptotic) |
|---|---:|---:|
| 20 vs 20 | 1.45 × 10⁻¹¹ | 6.80 × 10⁻⁸ |
| 20 vs 96 | 1.42 × 10⁻²² | 2.34 × 10⁻¹² |

SciPy's `mannwhitneyu` with `method='auto'` uses the asymptotic normal
approximation at these sizes — appropriate, since the exact computation is
intractable for 20 vs 96 — and that is what produced every p-value here.
There is **no resolution floor near any threshold in use**.

The original file is preserved unmodified as
`HEATMAP_README.v1_uncorrected.md`; the correction is recorded in
[FINAL_STATISTICAL_AUDIT.md](../derivatives/INTRA_INTER_CLASS_HEATMAPS/v2_cohort116/FINAL_STATISTICAL_AUDIT.md) §2.

---

## References

- Fan, L. et al. (2016). The Human Brainnetome Atlas: A New Brain Atlas Based on Connectional Architecture. *Cerebral Cortex*, 26(8), 3508–3526.
- Zang, Y. et al. (2004). Regional homogeneity approach to fMRI data analysis. *NeuroImage*, 22(1), 394–400.
- Zang, Y.-F. et al. (2007). Altered baseline brain activity in children with ADHD revealed by resting-state functional MRI. *Brain and Development*, 29(2), 83–91.
- Zou, Q.-H. et al. (2008). An improved approach to detection of amplitude of low-frequency fluctuation (ALFF) for resting-state fMRI: fractional ALFF. *Journal of Neuroscience Methods*, 172(1), 137–141.
- Zuo, X.-N. et al. (2012). Network centrality in the human functional connectome. *Cerebral Cortex*, 22(8), 1862–1875.
- Benjamini, Y. & Hochberg, Y. (1995). Controlling the False Discovery Rate: A Practical and Powerful Approach to Multiple Testing. *Journal of the Royal Statistical Society: Series B*, 57(1), 289–300.
- Anderson, M. J. (2001). A new method for non-parametric multivariate analysis of variance. *Austral Ecology*, 26(1), 32–46.
- Friston, K. J. et al. (1996). Movement-related effects in fMRI time-series. *Magnetic Resonance in Medicine*, 35(3), 346–355.
- Power, J. D. et al. (2012). Spurious but systematic correlations in functional connectivity MRI networks arise from subject motion. *NeuroImage*, 59(3), 2142–2154.
- Avants, B. B. et al. (2008). Symmetric diffeomorphic image registration with cross-correlation. *Medical Image Analysis*, 12(1), 26–41.
