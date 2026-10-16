# Understanding the Diagnostic-Group Hierarchical Clustering

Written for the project review: the project guide, the review panel, and the
project team. It explains what the dendrogram heatmaps show, what they are
evidence of, and — equally important — what they are not evidence of.

Project: *Comorbidity-Aware Meta-Bayesian Multi-Attention Transformer for
Alzheimer's Disease Progression Prediction Using Resting-State fMRI Biomarkers.*

---

## 1. What hierarchical clustering is

Hierarchical clustering is a way of organising items by similarity, without
being told the answer in advance.

It works bottom-up. Start with every item alone. Find the two most similar
items and join them. Treat that pair as a single item. Repeat until everything
is joined into one tree. The tree is called a **dendrogram**.

Two decisions define the tree completely:

- **The distance metric** — how we measure "similar" between two items.
- **The linkage method** — how we measure distance between *groups* of items
  once they have been merged.

Nothing about disease, diagnosis or biology enters the algorithm. It only sees
numbers. This matters: the algorithm will *always* produce a tree, even from
pure noise. A tree is not a finding by itself.

## 2. What the rows and columns represent

In each of the four main figures (`group_hierarchical_<biomarker>.png`):

- **Each row is one brain region** — one of the 246 regions of the Brainnetome
  atlas. All 246 rows are plotted; every 10th is labelled with its ID and
  anatomical name.
- **Each column is one diagnostic group** — CN, SMC, EMCI, MCI, LMCI, AD.
- **Each cell is the average value of that biomarker, in that region, across
  the subjects in that group.** Groups contain 20, 20, 20, 16, 20 and 20
  subjects respectively.

So the matrix is 246 × 6. A column is one group's "regional fingerprint".

## 3. What the colours mean

Within each row, we convert the six group values to a **z-score across the six
groups**: how far above or below the row's own six-group average each group sits,
measured in standard deviations.

- **Red** — this group is **above** that region's six-group average.
- **Blue** — this group is **below** it.
- **White** — close to the average.

The comparison is always *within a row*. Red in one row and red in another do
not mean the same absolute value; they mean "high for that region".

The colour scale is **fixed at ±2.05 for all four biomarkers**, so panels can be
compared directly. That bound is not arbitrary: with six groups, a z-score
computed this way can mathematically never exceed 2.041.

**Colour does not mean significance.** A deep red cell is not a discovery. It
means that group's average is relatively high in that region — nothing more.

## 4. How the distance metric and linkage method build the tree

**For the six diagnostic groups**, we use **correlation distance**: two groups
are close when their 246-region profiles rise and fall *in the same pattern*,
regardless of overall magnitude. Distance = 1 − correlation, so identical
patterns give 0 and opposite patterns give 2.

We use correlation rather than plain distance because the scientific question is
"do these two groups show the same regional pattern?", not "are their numbers the
same size?". With 246 regions per group, the correlation is estimated from
plenty of data.

**For the 246 regions**, we use **Euclidean distance** — ordinary straight-line
distance in the 6-group space. Each region row has only six numbers, where a
correlation would be unstable.

**The linkage method is "average"** in both cases: the distance between two
clusters is the average distance between all pairs of their members. We chose it
over Ward's method because Ward assumes clusters are roughly spherical and
equally sized, which we have no reason to believe here.

**One honest caveat.** Because each row is centred across the six groups, the six
columns are mathematically constrained — their correlations are pushed downward
by about −0.2 on average. That is why the distances cluster near 1.0. It is a
property of the arithmetic, not a property of Alzheimer's disease, and it is why
we do not interpret the absolute distances.

## 5. How to read short and long branches

The **height** at which two items join is the distance between them.

- **A short branch** — the two joined early, so they are relatively similar.
- **A long branch** — they stayed separate until late, so they are relatively
  dissimilar.
- **Two groups joined at a low height** are more alike *in the chosen metric*
  than two joined higher up.

Three cautions:

1. **Height is a distance, never a p-value.** You cannot read significance off a
   dendrogram.
2. **Left-to-right order is not meaningful** beyond the branching structure. The
   algorithm may draw the same tree with branches swapped.
3. **Adjacent leaves are not necessarily adjacent disease stages.** The tree
   reorders the columns; the clinical order CN → SMC → EMCI → MCI → LMCI → AD is
   printed on every figure precisely so the two are not confused.

## 6. How this differs from the ROI clustering we already had

| | Existing Z-score figures | This analysis |
|---|---|---|
| What is clustered | the 246 **regions**, and the 116 **subjects** | the 6 **diagnostic groups** |
| Question answered | which regions behave alike across people | which diagnostic groups look alike |
| Matrix | 246 × 116 | 246 × 6 |

The earlier figures could not answer the guide's question, because diagnostic
group was only an annotation strip there — it was never the thing being
clustered. Here it is.

## 7. How this differs from the P-score clustering

| | P-score figures | This analysis |
|---|---|---|
| Cell value | −log10(FDR-adjusted q) | mean biomarker value |
| What that is | **strength of statistical evidence** | **the measurement itself** |
| Clusters | regions with similar *evidence* profiles | groups with similar *biological* profiles |
| Metric | Euclidean + Ward | correlation + average (groups) |

The P-score figures cluster regions by how strong their statistical evidence is.
Since **no test in this project reaches significance**, that surface is largely
noise, which the P-score README states directly. This new analysis clusters the
actual measured values instead, which is a more informative descriptive view —
though still descriptive.

## 8. What the dendrograms actually show

Taken from the generated figures, not assumed in advance:

| Biomarker | Leaf order (left to right) | Main branches |
|---|---|---|
| mALFF | EMCI, SMC, CN, AD, LMCI, MCI | {LMCI,AD}, {SMC,EMCI}, {CN,SMC,EMCI}, {MCI,LMCI,AD} |
| mReHo | AD, CN, MCI, SMC, LMCI, EMCI | {CN,AD}, {SMC,LMCI}, {SMC,MCI,LMCI} |
| DC_z | MCI, AD, LMCI, SMC, CN, EMCI | {CN,SMC}, {LMCI,AD}, {MCI,LMCI,AD}, {CN,SMC,EMCI} |
| FC | CN, MCI, AD, LMCI, SMC, EMCI | {CN,MCI}, {SMC,EMCI}, {LMCI,AD}, {SMC,EMCI,LMCI,AD} |

**The single most useful observation:** mALFF and DC_z both split the six groups
into something close to an **early block {CN, SMC, EMCI}** and a **late block
{MCI, LMCI, AD}**. That is the shape you would hope for if these biomarkers
tracked severity.

**But the stability testing says do not lean on it.** We re-ran the whole
clustering on 1,000 bootstrap resamples of subjects within each group:

- {CN, SMC, EMCI} recurs in only **25%** (mALFF) and **19%** (DC_z) of resamples.
- {MCI, LMCI, AD} recurs in **19%** and **17%**.
- The most reproducible branch anywhere is {CN, MCI} in FC at **72%**.
- The adjacent-severity pair **{LMCI, AD}** recurs in three of four biomarkers,
  but never above **46%**.

A branch that survives fewer than half of resamples is not something to build an
argument on. The early/late split is a **hypothesis worth testing with more
subjects**, not a result.

mReHo's tree is the odd one out — it pairs **CN with AD**, the two extremes of
the progression. With 36% stability, the most likely explanation is instability
at this sample size rather than biology.

## 9. Do the diagnostic groups cluster together?

At the **subject** level — which is the harder and more honest test — **no.**

In `subject_level_clustering.png`, subjects are clustered on their own feature
profiles with the diagnostic label used only as a colour strip. If the groups
were separable, the strip would show six contiguous blocks of colour. **It is
visibly interleaved throughout.**

Quantitatively: a subject's nearest neighbour shares their diagnostic group
**23.3%** of the time, against a chance baseline of **16.0%**. Slightly above
chance, nowhere near separation. AD is the worst at 15% — at chance. CN and LMCI
are best at 30%.

**Conclusion to report:** the group-average profiles differ enough to produce a
structured dendrogram, but **individual subjects do not separate by diagnosis in
this feature space.** That is consistent with the null FDR results and is the
honest state of the project.

## 10. Why this is descriptive, not proof

Four reasons, each independent:

1. **No hypothesis test was performed.** Clustering produces a tree; it does not
   compute a p-value or control any error rate.
2. **Clustering always produces a tree.** Run it on random numbers and you get a
   clean-looking dendrogram. Structure in the picture is not evidence of
   structure in the population.
3. **The bootstrap shows most branches are unstable.** Resampling the same
   subjects changes the tree. A result that changes when you resample the same
   data is not a result.
4. **Averaging hides variability.** Each column is a mean over 16–20 people.
   Group means can differ visibly while the individuals overlap almost
   completely — which is exactly what the subject-level figure shows.

## 11. How to report the null FDR findings

State them plainly and first. They are a finding, not a gap:

> Across all FDR-corrected analyses, no result reached statistical significance.
> 0 of 24 stage-level multivariate tests and 0 of 5,904 ROI × stage tests had
> q < 0.05. The smallest adjusted q-value was 0.139 (mReHo at EMCI). Earlier
> pairwise regional and connectivity analyses were likewise null under their own
> correction procedures.

Then the three qualifications that must accompany it:

- **This does not prove no biological difference exists.** Effect sizes reach
  |g| ≈ 1.1; they do not survive correction for thousands of simultaneous tests
  at 16–20 subjects per group. The limitation is statistical power.
- **This does not mean the features carry no predictive information.**
  Significance testing and prediction are different questions. A set of features
  can be individually non-significant and still jointly predictive.
- **This does not justify a complex model by itself.** Weak signal plus a small
  sample is precisely the regime where a flexible model fits noise. Any benefit
  must be demonstrated against simple baselines on held-out data.

## 12. What would be needed to claim useful diagnostic discrimination

Nothing in this folder supports such a claim. To earn it:

1. **More subjects.** ~19 per class against a 1,476-test correction family is the
   binding constraint. Expand **by subject**, not by scan — ADNI is longitudinal,
   so searching by diagnosis returns more visits, not more people.
2. **A held-out evaluation.** Subject-grouped nested cross-validation, with any
   cross-subject scaling fitted on training folds only.
3. **Simple baselines first.** Majority class, and ordinal logistic regression on
   a handful of features. Report these *before* any transformer result.
4. **Confound control.** Residual head motion still correlates with 129/246
   (mALFF) and 131/246 (mReHo) regions after nuisance regression, and one
   scanning site contributes 40% of the AD scans. Both need to be covariates or
   harmonised.
5. **Calibration, not just accuracy.** The Bayesian component's whole purpose is
   that borderline subjects receive wide uncertainty — that is a calibration
   claim and needs calibration metrics.
6. **External validation** on a non-ADNI cohort before any generalisation claim.
7. **The missing clinical data.** LMCI currently has **no** MMSE, CDR, GDS or
   medical-history records (0 of 20), so the comorbidity-aware component cannot
   yet be evaluated across all six stages.

---

## A short explanation to say aloud at the review

> "You asked for the hierarchical tree over the diagnostic groups, so we built
> it. For each biomarker we took the average value in all 246 Brainnetome
> regions for each of the six groups, which gives a 246-by-6 matrix, and we
> clustered the six groups by how similar their regional patterns are, using
> correlation distance and average linkage.
>
> The trees are interesting. In mALFF and DC_z the six groups split roughly into
> an early block — CN, SMC, EMCI — and a late block — MCI, LMCI, AD, which is
> the shape you would want if these measures tracked severity. LMCI and AD pair
> together in three of the four biomarkers.
>
> But we tested how stable those branches are by bootstrapping the subjects a
> thousand times, and most of them reappear in well under half the resamples.
> So we are presenting this as an exploratory observation, not a finding.
>
> We also checked whether individual subjects cluster by diagnosis, and they do
> not. A subject's nearest neighbour shares their diagnosis 23% of the time
> against a 16% chance baseline — above chance, but nowhere near separation.
>
> That is consistent with our statistics: nothing reaches significance after FDR
> correction. The smallest adjusted q-value is 0.139. We think that is a power
> limitation at 116 subjects rather than an absence of signal, and the next step
> is more subjects and the missing LMCI clinical tables rather than a more
> complex model.
>
> On the Transformer: the encoder is implemented and all 24 unit tests pass. It
> accepts our real data — a 116-by-246-by-3 regional tensor and a
> 116-by-246-by-246 connectivity tensor — and runs a forward pass with 196,374
> parameters. It has **not** been trained. There is no training loop, no
> optimiser, no data splitting and no checkpoint anywhere in the repository, so
> we have no accuracy to report and we are not claiming any."
