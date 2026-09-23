# Protocol for Table G.7 and Figure G.3

This experiment uses the cached, frozen, unit-normalized ResNet-18 and
ViT-B/16 features in `../../deep_features/raw/`. For each backbone and each
CIFAR-10 pair (cat/dog, car/truck, plane/ship), it repeats selection over five
independent stratified splits. Seeds are `20270901` through `20270905`, plus
`1000 * pair_index` for the train/validation permutation. Each class contributes
240 fitting and 60 validation observations from the cached training pool. The
separate two-class test pool contains 200 examples and is fixed across splits.

Every fit independently minimizes its exact convex logistic objective in
float64 with external positive ridge. Samplewise selection evaluates all
`11 x 5 = 55` pairs of ridge and radius. The ridge grid is
`1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1`; radii are
`0, .25, .5, 1, 2`. Ordinary ERM evaluates the same eleven ridge values.
ERM plus scale minimizes validation log loss over positive output scales in
`[.05, 20]` for each ridge. The analytic match is computed at the fixed
reference setting `tau=.02, delta=1`; it is a mechanism control, not a
validation-tuned performance baseline. Ties use the first candidate in grid
order.

Hyperparameters are chosen only by validation log loss. Each selected
configuration is fitted afresh on the 480 fitting points and evaluated once
on the fixed held-out test pool. The CSV includes log loss, Brier score,
accuracy, score variance, solver iterations, and gradient residual.
Table G.7 reports the mean test log loss and a split-level normal interval
`mean +/- 1.96 * sample_sd / sqrt(5)`; paired method differences are computed
split by split. These intervals describe train/validation split sensitivity
conditional on the fixed test pool, feature map, and search grid.

`selection_manifest.json` includes every validation candidate, all split
indices, and every selected configuration. `fair_tuning_splits.csv` contains
the 120 selected-method evaluations; `fair_tuning_summary.csv` contains the
42 reported aggregate and paired-difference rows. `run_upgrade.py` regenerates
all three from the frozen-feature arrays. `verify_upgrade.py` checks their
internal consistency and numerical agreement with the archived Table G.7
summary. `plot_validation_curves.py` creates Figure G.3; `plot_fair_tuning.py`
creates the LaTeX table.
