# Experimental protocol

Theoretical identity: samplewise RKHS logistic perturbation with external fixed ridge, not Singh2025 Functional SAM or practical parameter SAM. Main empirical experiments use the infinite-rank periodic kernel K(x,x')=[1+2 sum_{j>=1}cos(j(x-x'))/j^4]/[1+pi^4/45], evaluated in closed form. Diagonal is one. Computation uses the empirical representer problem with numerical rank filtering as specified below; the filtering is an approximation in floating-point linear algebra, not a finite-rank kernel model. Ridge tau=0.02 unless a stated ridge ablation changes it.

Completed-run requirements: 3,000 asymptotic records (5 sample sizes, 100 data replicates, 3 gamma values, 2 eta values); 3,000 balanced records (3 sizes, 200 replicates, 4 fixed radii plus one shrinking radius); 630 spectral population-quadrature fits (grids 256/512/1024, tau .005/.02/.08, k 1/3, epsilon .025/.05/.1/.2/.4, delta 0/.005/.02/.1/.25/.5/1); 45 scalar optima. analyze_results.py asserts complete counts and balanced paired replicate identities before output.

Independent Newton optimization solves average log(1+exp(delta-yf(x)))+tau||f||_H²/2. No theoretical-shift formula is used to generate robust fits. Strong convexity converts the full RKHS gradient norm into a distance bound by division by tau. Stored residuals are in the retained whitened coordinates. Eigenvalues at most max(lambda_max*1e-13,1e-13) are filtered numerically; with unit diagonal and bounded logistic score, their omitted-gradient contribution is at most sqrt(1e-13). Add this conservative term before using the distance bound. This accounts for rank filtering in exact PSD arithmetic, not arbitrary floating-point roundoff; the projected residual alone is not a full-space certificate.

Evidence families:

1. Shrinking-radius empirical asymptotics: independent data replicates, paired estimators, n grid and gamma grid; report sqrt(n) distance, delta-normalized distance, first-order prediction residual and empirical objective stationarity. Labels have conditional mean 0.5 cos(x).
2. Population spectral response: uniform numerical quadrature of both label outcomes; compute actual minimizers across epsilon and delta. Compare leading signal and cubic harmonic formulas, show separate coefficient signs; repeat quadrature resolution. This is controlled numerical approximation of population integrals, not statistical samples or a proof.
3. Balanced-label sampling: p=1/2, fixed radius covariance ratios and shrinking-radius absence of deterministic drift. Covariance is estimated from independently fitted estimators, not sampled from the theoretical Gaussian limit.
4. Objective/Singh distinction: scalar binary logistic model (constant kernel) independently solve samplewise and common-perturbation objectives, and compare ordinary optimum. Singh's affine update equality is algebraic, not an emulated neural-network benchmark.

Save code, full config, raw JSONL/CSV, summary JSON, and vector plots. Use only newly computed results. Old MobileNet and drift-constructed figures are not carried into the revised scientific evidence.

Method identity gate: exact samplewise logistic objective, ordinary ridge logistic baseline, exact scalar common-perturbation minimax. These are full mathematical models selected for the revised claims. Numerical quadrature is explicitly identified and checked against resolution. No truncated neural network is labeled a full-method benchmark.
