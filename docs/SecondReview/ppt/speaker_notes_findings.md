# Speaker notes: Findings 3 and 4, and the backups on motion, site and "why 246 regions"

For slides 22 and 23 and the backups 3/10 ("Residual Motion and Site Effects") and 6/10
("Why 246 Regions?"). Every number is reproduced by
`python3 scripts/audit/findings_checks.py`. None of this is a model result: no stage label is
used for Findings 3 and 4 or for the granularity check.

## Slide 22 - Finding 3: the biomarkers are not redundant (45 s)

"Do our biomarkers just repeat each other? Finding 1 showed that FC strength is exactly DC
divided by 245, so we removed it. For the three that remain, we compared their patterns across
the 246 regions, scan by scan. They share only 6 to 11 percent of their variance: they are 89
to 94 percent non-redundant. And ALFF and degree centrality go in opposite directions in every
one of the 165 scans: regions with strong local fluctuations tend to be less connected. That
supports using them as separate tasks in our design. Whether it helps staging we can only test
once the model exists."

If asked what "shared variance" is: for each scan, the squared correlation, across the 246
regions, between two biomarkers' maps; averaged over the 165 scans (mALFF-mReHo 11.3 %,
mALFF-DC_z 11.1 %, mReHo-DC_z 5.7 %). Low overlap means different information about the
brain, not information about disease.

## Slide 23 - Finding 4: the features are reproducible and plausible (60 s)

"Are the features reproducible? 31 people were scanned more than once: 69 scans, and 45 of the
46 same-person pairs are from different sessions. For ALFF, each scan's most similar scan is
the same person's other scan in 63 of 69 cases, against about 3 percent by chance. No stage
label is used. As a plausibility check we looked at left-right homologous regions, which are
known to be strongly connected: they are in 164 of our 165 scans, mean Fisher z 0.63 against
0.01 for all other pairs. One EMCI scan does not show it, and we will check it."

If asked "isn't that just the scanner?": partly it may be. A person's repeat scans usually share
a site and scanner (the site is the prefix of the ADNI subject ID), so this shows the features
are stable and sensible, not that they are free of scanner effects.

If asked "is 91 % an accuracy?": no. It is how often a scan's closest match is the same person.
It is not a disease classifier and uses no labels. Chance is 3 % (1 in 31 people); counted
exactly for nearest-neighbour matching it is 2.0 %.

The scan that lacks the homologous pattern: EMCI, sub-012S4849, ses-01, run-01 (mean
homologous Fisher z -0.003). It is also unusual in the QC table: tSNR 87 (7th percentile), mean
FD 0.44 mm (73rd percentile), and the widest spread of FC edge weights in the cohort (SD 0.50
against a median of 0.28). It has not been excluded from anything. Worth a look before
modelling.

## Backup 3/10 - Residual motion and site effects (30 s)

"After our 27 nuisance regressors, ALFF and ReHo still track head motion in more than half of
the regions: 129 and 131 of 246, against about 12 expected by chance. DC is less affected, 39.
So mean framewise displacement will go into the model as a covariate. Separately, ADNI is
multi-site: 14 sites, and 40 percent of our AD scans come from one site, 68 percent from two,
while no site gives more than 19 percent of the CN scans. So a CN-versus-AD difference is partly
a site difference; we need harmonisation or a site covariate before Module 4."

Caveats to volunteer: the motion test is Pearson r with each scan's mean FD, p < 0.05
uncorrected, over 164 scans (the artefact scan sub-018S4313 is excluded; with all 165 it is
135 / 126 / 43), and repeat scans of one person are not independent. The site shares are shares
of scans (by people: AD 40 % and 65 % of 20).

## Backup 6/10 - Why 246 regions? (30 s)

"We cannot yet say which regions matter for disease. We can ask whether 246 regions hold
information that coarser versions lose. Using ALFF and no stage label, a scan's closest match is
the same person 91.3 percent of the time with 246 regions, 87.0 with the 123 left-right merged
regions, and 84.1, 72.5 and 50.7 with coarser merges. Returns flatten: a random 100 regions
already give 88.3. Two regions share only about 4 percent of their variance, and 79 components are
needed for 95 percent. None of this is about disease. Only the trained model can say whether 246
is the right size."

Say what the merges are: consecutive atlas labels in order. Merging by 2 combines each left-right
pair; merging by 3, 6 and 41 combines neighbouring labels in index order. They are not other
established atlases. In this atlas the hippocampus and the amygdala each have 4 ROIs (2 per
hemisphere).

## What changed from `new_findings_slides.tex`

- The homologous-region control now uses all 165 scans and every pair (0.63 against 0.01). The
  original used the first 60 scans (27 CN, 31 SMC, 2 EMCI: no MCI, LMCI or AD) and 200 sampled
  pairs per scan (0.61 against 0.02). The t value (24.6) is left off: repeat scans of one person
  are not independent.
- "ID acc." is now "identification rate", and the slide says it is not a classifier.
- "Shared variance 5.0 %" is 4.0 % over all 165 scans (5.0 % holds only for the 69 repeat scans).
- "Parcellation 82 / 41 / 6" is "regions after merging consecutive labels", because they are not
  separate parcellations.
- "That is now measured, not assumed" (Gap G1) and "hemispheric asymmetry is real information"
  are not on the slides: the first overstates what a low overlap shows, and the second is not
  separable from the loss of half the features in the merge.
- Not added: "Why the Full 246-ROI Space Is Retained" (the compact slide 11 replaces it) and the
  user's two backup figures for it (slide-sized versions are backups 4/10 and 5/10).
