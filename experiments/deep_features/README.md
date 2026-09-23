# Frozen deep-feature supplement

The completed study evaluates exact samplewise function perturbations on frozen ImageNet-pretrained ResNet-18 and ViT-B/16 features. It is not shared SAM, Functional SAM, full-network fine-tuning, or a tuned-ERM performance comparison.

## Reproduce

Create a Python 3.12 environment. Install torch 2.7.1 and torchvision 0.22.1 from the CUDA12.8 wheel index, then the pinned analysis packages in requirements.txt. Extraction requires a CUDA-capable GPU with adequate memory.

Run `python extract.py`, `python run.py`, `python analyze.py`, then `python verify.py` in this directory. Extraction obtains the byte-identical CIFAR archive through download_cifar.py, verifies official MD5 and downloads official torchvision weights. Model evaluation uses deterministic weight-specific preprocessing. Existing complete feature/fit files are reused. Remove a specific cache only when deliberately reproducing that stage.

`protocol.md` was frozen before outcomes; its SHA256 is in sources.json. A development-only smoke test corrected an initial gradient-sign bug before any formal fit. No outcome-based rerun or setting selection occurred. All heads use float64 and full feature dimensions. No independent training-data randomness is attributed to repeated initialization of the same convex fit.

## Inspect

- `raw/data_indices.npz`: original fixed train/test indices and labels.
- `raw/*_features.npz`: raw and normalized features; fitting renormalizes raw features in float64.
- `raw/*_signal.npz`: all72 independent expected-label optima and linear references.
- `raw/*_noise.npz`: all800 fitted heads,100 paired label vectors per model, linearized fits, exact conditional references, design indices and probes.
- `raw/*_real_*.npz`: all720 fitted heads,30 paired bootstrap counts per task/backbone and test scores.
- `results/*.csv`: every declared setting and uncertainty; summary.json gives aggregate statistics.
- `results/deep_transfer_main.*`: compact main figure; deep_transfer.* gives per-backbone panels.
- `results/verification.json`: independent CPU optimizer, numerical-gradient and direct covariance checks.

The1592 heads are not1592 independent image datasets. Conditional noise intervals are all above one, yet all exclude linearized references despite less than1% relative ratio error. All six true-label endpoint comparisons lower logloss and increase score resampling variance; accuracy does not improve consistently. Pointwise intervals condition on the declared pools and do not measure new-test-set uncertainty. Images, model weights, environment binaries and download caches are excluded from the reproducibility archive.
