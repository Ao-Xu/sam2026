# Joint-selection records for Table G.7 and Figure G.3

This directory contains the current five-radius, five-split frozen-feature
experiment. It is separate from the older fixed-radius/bootstrap comparison
under `../raw/`. The selection protocol, commands, and environment are in
`../README.md` and `../run_upgrade.py`.

- `selection_manifest.json`: for each of 30 backbone/task/split combinations,
  the train, validation, and test indices; the full candidate validation-loss
  grid; and the selected configuration for each method.
- `fair_tuning_splits.csv`: all 120 independently evaluated selected fits,
  with validation and test metrics, solver residuals, and selected parameters.
- `fair_tuning_summary.csv`: 42 aggregate rows, including the 24 method/task
  rows used for Table G.7 and 18 paired difference rows.
- `metadata.json`: fixed grids, seeds, environment, and test-pool scope.
- `validation_curves.pdf/png` and `fair_tuning.tex`: generated Figure G.3 and
  Table G.7 material.
- `verification.json`: consistency checks for the supplied records.
- `paper_reference_summary.csv`: the previously reported Table G.7 aggregate
  values, retained so a fresh rerun can be compared with the manuscript.
- `paper_reference_selection_manifest.json`: a five-radius reference obtained
  by filtering the archived candidate-loss manifest to the radius grid that
  generated Table G.7 and reselecting the minimum validation loss. It provides
  an independent check of candidate losses and split indices.
- `reference_provenance.json`: SHA-256 of the archived manifest and the exact
  filtering rule used to construct that reference.

The eleven ridge values are `1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2,
3e-2, 1e-1, 3e-1, 1`; the five samplewise radii are `0, .25, .5, 1, 2`.
The five stratified splits use 240 training and 60 validation examples per
class. A separate fixed 200-example-per-task test pool is not used to choose
parameters. Split-level intervals do not measure uncertainty from drawing a
new test pool.
