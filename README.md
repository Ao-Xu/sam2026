# Code and recorded experiments

Code and recorded experiments for *Interpreting Samplewise Function Perturbations: Signal, Noise, and Nonlinear Structure*.

## Start here
Use Python 3.12. Install analysis dependencies with:
```
python -m pip install -r requirements-analysis.txt
python experiments/matching/verify_upgrade.py
python experiments/matching/plot_validation_curves.py
python experiments/matching/plot_fair_tuning.py
```
These commands check the supplied five-split joint-selection records and regenerate Figure G.3 and Table G.7 from recorded data, without training or downloading models. Outputs are in `experiments/matching/upgrade/`. The current main text has Figure 1 (`intro_overview.pdf`) and Figure 2 (`response_diagnostics.pdf`); their included PDFs are in the manuscript source package. `experiments/visual-composer/plot_dense_main.py` generates earlier exploratory panels named `matched_spectral_dense` and `deep_transfer_dense`, not the current numbered main-text figures. Times New Roman is used when installed; otherwise Matplotlib substitutes a serif font.

## Experiment map and commands
Run commands inside the indicated directory. Raw data and recorded results are included; rerunning generation overwrites that family's outputs.

| Directory | Evidence | Regenerate / analyze |
|---|---|---|
| experiments | Periodic asymptotics, balanced noise, spectral and scalar examples | See experiments/README.md for exact family sizes; python analyze_results.py analyzes supplied runs |
| experiments/generalization | Nonuniform periodic and nonperiodic kernels | python run.py; python analyze.py; python verify.py |
| experiments/finite_sample | Covariance decomposition and breast-cancer features | python run.py rbf; python run.py real; python bandwidth.py; python analyze.py; python verify.py |
| experiments/certificates | Interval certificates | python certify_witness.py; python certify_radius.py |
| experiments/deep_features | Frozen ResNet-18 and ViT-B/16 | python extract.py; python run.py; python analyze.py; python verify.py |
| experiments/matching | Analytic matching, fixed-setting controls, and five-split joint selection | python run.py --population; python run.py; python analyze.py; python verify.py; python run_upgrade.py --splits 5; python verify_upgrade.py; python plot_validation_curves.py; python plot_fair_tuning.py |
| experiments/visual-composer | Earlier exploratory matched-response and deep-feature panels | python plot_dense_main.py |

Each experiment directory contains its protocol and recorded outputs. The original experiments/README.md describes the original CPU study, not the hardware requirements of later extensions.

## Environments and downloads
The original CPU experiments ran with Python 3.7, NumPy 1.21.6, SciPy 1.1.0, and Matplotlib 2.2.3; recorded configurations preserve that environment. The analysis requirements above provide the modern environment used for the final main figures; legacy scripts may require their recorded environment for exact reproduction.

Deep-feature generation and matching fits use Python 3.12, torch 2.7.1, torchvision 0.22.1, and CUDA. Install the CUDA 12.8 builds using:
```
python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r experiments/deep_features/requirements.txt
```
The extraction scripts require a CUDA-capable GPU; the supplied features allow analysis without extraction. Supplied feature arrays and fitted heads allow inspection and analysis without extraction. Full extraction downloads CIFAR-10 and torchvision pretrained weights. Original images, model weights, wheels and download caches are excluded. Source attribution and dataset conditions are retained in each protocol, DATA_LICENSE.md and sources.json where available.

## Scope
These implementations optimize the paper's samplewise logistic objective; they are not implementations of shared SAM or ICML 2025 Functional SAM. Figures are based on recorded fits. Bootstrap uncertainty in the original fixed-setting comparison is conditional on the specified pools and selected parameters. The joint-selection supplement instead reports variation across five independent training/validation splits with a fixed test pool. See protocols for pairing and numerical tolerances.

## Packaging changes
Local wheel paths were replaced by installable package versions. The joint-selection supplement uses the five-radius grid `0, .25, .5, 1, 2` that produces Table G.7, with the expanded eleven-point ridge grid starting at `1e-5`. The fixed-setting files in `experiments/matching/raw/` remain a separate experiment. Historical manifest/verification files describe their original runs.

## Content anonymization

The distributed files omit local paths, personal identifiers, machine model names, precise runtime records, protocol calendar dates, interpreter build strings, and document creation metadata. Numerical arrays, labels, split indices, seeds, hyperparameters, solver residuals, scientific tables, dependency versions, and third-party attribution are preserved. Run `python verify_integrity.py` to check the published file set against `SHA256SUMS`.
