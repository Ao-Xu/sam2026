# Reproducing the revised paper

These experiments solve the samplewise RKHS logistic objective with a fixed external ridge penalty. They do not implement the ICML 2025 Functional SAM algorithm. See PROTOCOL.md for the objective and numerical-error interpretation.

The recorded software environment used Python 3.7, NumPy 1.21.6, SciPy 1.1.0 and Matplotlib 2.2.3. Dependencies are NumPy, SciPy and Matplotlib; no GPU or external dataset is used. Raw files preserve the executed results. Run commands from this directory with the same numerical environment for closest reproduction; BLAS and floating-point differences can affect final digits.

```powershell
$env:OMP_NUM_THREADS="1"
$env:MKL_NUM_THREADS="1"
python -X utf8 run_experiments.py --family asymptotic --replicates 100 --sizes 64,128,256,512,1024
python -X utf8 run_experiments.py --family balanced --replicates 200 --sizes 128,512,1024
python -X utf8 run_experiments.py --family spectral
python -X utf8 run_experiments.py --family scalar
python -X utf8 analyze_results.py
```

Generation overwrites the corresponding family raw results. Back up raw/ if retaining the delivered run. The scripts resolve their output directory relative to their own location. Analysis alone regenerates figures and tables from existing raw files, checks counts and replicate identities, and takes only a few seconds.

Files:

- `run_experiments.py`: closed-form infinite-rank kernel, convex optimization, data generation and raw recording.
- `analyze_results.py`: bootstrap calculations, numerical comparisons and figures.
- `raw/config_*.json`: executed family configuration and NumPy version.
- `raw/*.jsonl` and `raw/scalar.json`: 6,675 recorded outcomes, including 630 deterministic population-quadrature fits and 45 scalar fits. Additional zero-radius baseline fits are reused within paired experiments.
- `results/summary.json`: figure/table values and numerical audit summaries.
- `results/slopes_table.tex`: finite-range slope table with bootstrap intervals.
- `figures/`: editable vector SVG, publication PDF and preview PNG.

All robust estimators are independently optimized. The theoretical first-order displacement is used only as a diagnostic comparison. Population quadrature is distinguished from randomly sampled empirical fits. The solver's retained-coordinate residual is not a full RKHS error certificate: PROTOCOL.md gives the additional bound for numerically omitted eigenmodes and states its exact-arithmetic limitation.

The finite-range slope intervals exclude the asymptotic exponents. The variance ratios are above one but differ detectably from their limiting values. These discrepancies are reported in the paper; the data are not presented as exact empirical verification of asymptotic limits.
# General-kernel extension

The original experiment commands below reproduce the periodic analysis. The additional `generalization` directory contains its own complete `PROTOCOL.md`, raw records, numerical summaries and vector figures. Run `python -X utf8 generalization/run.py`, then `python -X utf8 generalization/analyze.py` and `python -X utf8 generalization/verify.py` to reproduce the new nonuniform/mixed-signal and nonperiodic RBF evidence. This extension contains 2,088 fitted problems and 12 operator-reference records. Its pointwise covariance intervals quantify Monte Carlo uncertainty; finite-sample deviations from the asymptotic references are retained.
