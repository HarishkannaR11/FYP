# Speaker notes: the full architecture slide

For slide 6 ("Full System Architecture"), right after the overview on slide 5. About 60-75
seconds. Point at each box as you go. Top row: built and checked. Bottom row: the design for
the next phase.

## Say this

"This is the whole system on one page. A scan goes in at the top left, and a stage with a
confidence comes out at the bottom right.

The top row is what we have built and checked.

Module 1 collects and checks the data: the brain scans and the clinical notes from ADNI.
175 scans were audited, 2 were dropped and 173 were eligible.

Module 2 cleans the scans. It removes head movement and noise and fits every brain onto the
same standard map. 165 scans came through.

Module 3 turns each scan into numbers. For each of 246 brain regions we get the four measures
you saw: how active it is, teamwork inside it, who it goes with, and how well linked it is.

Then the arrow goes down to the bottom row. This is the design for the next phase. Nothing in
it is trained yet.

Module 4 is the learning part. The four measures are four tasks. One shared model looks at all
246 regions at once, in three ways: the whole brain, close neighbours, and brain networks. The
learning loop trains it so that what it learns works for all four measures. Then we add the
patient's clinical facts.

Module 5 gives the answer: one of the six ordered stages, a probability for each, and how
sure it is.

Module 6 is the screen for the doctor. It is decision support, not a diagnosis.

Green is built and checked. Blue and dashed is design only."

## If the panel asks

- **Why two rows?** Only to fit the slide. Read the top row left to right, then the bottom row.
- **What exactly is built?** Modules 1 to 3: 165 scans processed and verified. Nothing in Modules
  4 to 6 is trained, and no accuracy, AUC or other result exists.
- **Where do the clinical facts come from?** Module 1 cleaned the ADNI tables: age, APOE
  epsilon-4, MMSE, CDR-SB, GDS, and cardiovascular and endocrine history. If asked about gaps:
  covariates exist for 107 of 127 subjects; the late-MCI group has APOE epsilon-4 only.
- **What is "meta-learning" here?** Each biomarker is a task. The shared encoder is trained so
  that a few update steps adapt it to any of them (MAML-style), which gives one
  biomarker-invariant representation, Z'.
- **What is "ordinal"?** The stages have an order. Confusing neighbouring stages is a smaller
  error than confusing distant ones, so the head uses ordered thresholds.
- **What is "Bayesian"?** The classifier weights have a distribution. Averaging over samples
  gives the stage probabilities and a measure of uncertainty (predictive entropy).
- **Where is the evaluation?** It belongs to Module 4: subject-grouped cross-validation with
  cross-subject scaling fitted on training folds only. It is not built.

The figure is `figures/architecture_full.tex` (standalone TikZ, compile with pdflatex). The same
file is Figure 4.1 in the report.
