# Invalid records, kept for transparency only

These 5 records (variant "small", seed 20261015, folds 0-4) were produced by a bug in
`run_transformer_variants.py`. The per-split nested-chosen feature setting was applied to every variant instead of
only to "chosen", so these runs changed two things at once: the model size and the feature setting. That breaks
the pre-registered one-change design.

They are excluded from every analysis. The bug was fixed: the chosen setting is now applied only when the variant is
"chosen". The variant was then rerun from the start.
