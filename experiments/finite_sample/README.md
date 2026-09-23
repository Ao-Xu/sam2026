# Finite-sample decomposition and fixed real features

The frozen `protocol.md` specifies all conditions and pairing. This directory adds 4,000 optimized fits: 3,000 RBF fits, 400 bandwidth checks, and 600 real fixed-feature fits. Fits share data across specified conditions; they are not 4,000 independent datasets.

From this directory, using Python with NumPy, SciPy, scikit-learn, matplotlib:

```
python run.py rbf
python run.py real
python bandwidth.py
python analyze.py
python verify.py
```

The scripts set BLAS threads to one and import the supplied solver in `../generalization/run.py`. Rerunning data-generation scripts replaces their own raw output files. The raw records, split/preprocessing archive and summarized tables are already supplied. `verify.py` checks a separate L-BFGS fit, an unfiltered Gram covariance calculation, the exact decomposition identities, data split and training-only preprocessing.

`results/data_provenance.json` specifies the actual library version and data hashes. `DATA_LICENSE.md` credits the dataset. `figures/finite_sample_decomposition.pdf` is the manuscript figure.

At balanced labels, nonlinear and random-design bias estimates are distinguished from label Monte Carlo fluctuation. The individual covariance control-variate estimate is unbiased up to numerical approximation; ratios are not claimed unbiased. The reported paired interval for the raw second-moment ratio is not an interval for the control-variate ratio. Original-label real-data performance is separate from the randomized-label covariance mechanism. All real-data statements condition on the frozen preprocessing, source training pool and fixed test probes.
