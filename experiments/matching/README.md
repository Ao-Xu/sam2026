# Matching controls

Read protocol.md before inspecting results. Uses the original full normalized ResNet-18/ViT-B/16 feature caches in ../deep_features/raw and the original closed periodic kernel implementation in ../run_experiments.py.

Run from this directory with Python 3.12 and the existing torch/scipy environment:

```text
python run.py --population
python run.py
python analyze.py
python verify.py
```

The first command executes population quadrature and original Newton cross-checks. The second executes matched weak-signal controls, validation fits and all paired true-label bootstrap fits. These commands overwrite their own matching outputs; they do not alter original experiment data. Analysis regenerates vector figures in the local `figures/` directory and tables/CSV/summary in `results/`. The verifier checks independent gradients, split isolation, pairing, original solver agreement and selected independent CPU optimizer fits.

Formal objective counts: 112 population fits, 72 matched weak-signal heads, 36 validation heads and 900 bootstrap heads. Numerical cross-check fits are additional. All methods and six task/backbone settings remain in the outputs.

Files: raw retains fitted parameters, predictions, indices, bootstrap counts, validation scores, selected hyperparameters and residuals; results contains complete metrics, conditional intervals and verification.json. Variability conditions on the fixed test pool and selected hyperparameters. Positive scales do not change classification signs. The tuned baseline search is finite and fixed in advance, not a claim of globally optimal tuning.

All six validation-tuned ordinary logistic baselines obtain lower test log loss than the fixed-radius samplewise comparator. Matching controls are close at weak signals, while independently measured third harmonics have opposite local directions. These outcomes support response analysis without asserting samplewise predictive superiority.

## Five-split joint selection (Table G.7 and Figure G.3)

The `upgrade/` directory is a separate experiment. It contains the five split
indices for each backbone/task, every candidate validation loss, selected
parameters, per-split test results, and the aggregate table. Samplewise
selection uses eleven ridge values from `1e-5` to `1` crossed with the five
radii `0, .25, .5, 1, 2`; ordinary controls use the same ridge grid.
The test pool is fixed and is never used for selection. From this directory:

```text
python verify_upgrade.py
python plot_validation_curves.py
python plot_fair_tuning.py
```

`plot_validation_curves.py` regenerates Figure G.3 from candidate-level
validation losses, and `plot_fair_tuning.py` regenerates the LaTeX for
Table G.7 from the per-split summary. To rerun the entire selection experiment
from the included frozen-feature arrays with Python 3.12, PyTorch 2.7.1, and
SciPy, run `python run_upgrade.py --splits 5`. This overwrites only `upgrade/`.
The original `run.py` and its six fixed-setting bootstrap outputs are not the
source of Table G.7.

## Higher-order diagnostics

`run_cubic_residual.py` reproduces the periodic cubic-truncation check from the
appendix. From this directory, run

```text
python run_cubic_residual.py --data raw/periodic_1024.npz --out results/cubic_residual
python plot_cubic_residual.py --input results/cubic_residual/cubic_residuals.csv --output figures/cubic_residual_diagnostic.pdf
```

The first script reports the raw $O(\epsilon^5)$ remainders, the uncentered
matched difference divided by $\epsilon^3$, and the centered $O(\epsilon^2)$
error. `run_general_cubic_rbf.py` performs the deterministic nonperiodic RBF
check of the general cubic operator:

```text
python run_general_cubic_rbf.py --out results/general_cubic_rbf --figure figures/general_cubic_rbf.pdf
```

These are population-quadrature diagnostics with fixed representations; they
are not end-to-end neural-network training experiments. The RBF script reports
the node-kernel diagonal reconstruction error and the weighted feature trace
as checks that the weighted Gram eigenvectors have been mapped back to node
values correctly. The supplied `results/general_cubic_rbf_order32` directory
records the independent 32-by-32 quadrature check.
