"""Validate the matched cubic operator on a nonperiodic RBF geometry.

This deterministic population experiment uses a product density on [-1,1]^2,
a normalized Gaussian kernel, and bounded q=(x1+0.5*x2)/1.5.  It compares
independently optimized population fits with the general cubic prediction
Delta_delta[q] and records the centered O(epsilon**2) error.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.linalg import eigh, solve
from scipy.special import expit


def quadrature(order):
    nodes, weights = leggauss(order)
    xx, yy = np.meshgrid(nodes, nodes, indexing="ij")
    wx, wy = np.meshgrid(weights, weights, indexing="ij")
    x = np.column_stack([xx.ravel(), yy.ravel()])
    density = (1.0 + 0.6 * x[:, 0]) * (1.0 + 0.4 * x[:, 1]) / 4.0
    return x, (wx * wy).ravel() * density


def rbf_kernel(x, z, bandwidth=0.7):
    sqdist = (
        np.sum(x * x, axis=1)[:, None]
        + np.sum(z * z, axis=1)[None, :]
        - 2.0 * x.dot(z.T)
    )
    return np.exp(-np.maximum(sqdist, 0.0) / (2.0 * bandwidth**2))


class PopulationModel:
    def __init__(self, kernel, weights):
        self.weights = np.asarray(weights, dtype=float)
        root_w = np.sqrt(self.weights)
        weighted_kernel = (root_w[:, None] * kernel) * root_w[None, :]
        eigenvalues, eigenvectors = eigh(weighted_kernel, check_finite=False)
        threshold = max(float(eigenvalues[-1]) * 1e-13, 1e-13)
        keep = eigenvalues > threshold
        self.eigenvalues = eigenvalues[keep]
        self.eigenvectors = eigenvectors[:, keep]
        # The eigenvectors diagonalize the weighted Gram matrix in weighted
        # L2 coordinates. Convert them back to values at the quadrature
        # nodes before using them as predictions.
        self.features = (
            self.eigenvectors / root_w[:, None]
        ) * np.sqrt(self.eigenvalues)[None, :]
        reconstructed_kernel = self.features.dot(self.features.T)
        self.node_kernel_diag_error = float(
            np.max(np.abs(np.diag(reconstructed_kernel) - np.diag(kernel)))
        )
        self.weighted_feature_trace = float(
            np.trace(self.features.T.dot(self.weights[:, None] * self.features))
        )
        self.discarded = int((~keep).sum())

    def _objective(self, coefficients, delta, tau, probabilities):
        f = self.features.dot(coefficients)
        plus = expit(delta - f)
        minus = expit(delta + f)
        value = self.weights.dot(
            probabilities * np.logaddexp(0.0, delta - f)
            + (1.0 - probabilities) * np.logaddexp(0.0, delta + f)
        )
        gradient_point = -probabilities * plus + (1.0 - probabilities) * minus
        gradient = self.features.T.dot(self.weights * gradient_point)
        curvature = probabilities * plus * (1.0 - plus)
        curvature += (1.0 - probabilities) * minus * (1.0 - minus)
        hessian = (self.features.T * (self.weights * curvature)).dot(self.features)
        value += 0.5 * tau * coefficients.dot(coefficients)
        gradient += tau * coefficients
        hessian += tau * np.eye(len(coefficients))
        return float(value), gradient, hessian, f

    def fit(self, delta, tau, probabilities, tolerance=1e-13):
        coefficients = np.zeros(self.features.shape[1])
        for iteration in range(100):
            value, gradient, hessian, _ = self._objective(
                coefficients, delta, tau, probabilities
            )
            if np.linalg.norm(gradient) <= tolerance:
                break
            direction = solve(hessian, gradient, assume_a="pos", check_finite=False)
            step = 1.0
            while step > 2.0**-30:
                candidate = coefficients - step * direction
                candidate_value = self._objective(
                    candidate, delta, tau, probabilities
                )[0]
                if candidate_value <= value - 1e-4 * step * gradient.dot(direction) + 1e-15:
                    break
                step *= 0.5
            coefficients = candidate
        value, gradient, _, function = self._objective(
            coefficients, delta, tau, probabilities
        )
        return coefficients, function, {
            "objective": float(value),
            "gradient_residual": float(np.linalg.norm(gradient)),
            "iterations": int(iteration + 1),
            "discarded_modes": self.discarded,
        }

    def linear_response(self, delta, tau, q):
        s = float(expit(delta))
        p = s * (1.0 - s)
        operator = (self.features.T * (self.weights * p)).dot(self.features)
        operator += tau * np.eye(self.features.shape[1])
        rhs = s * self.features.T.dot(self.weights * q)
        return solve(operator, rhs, assume_a="pos", check_finite=False)

    def cubic_response(self, delta, tau, q, linear_coefficients=None):
        s = float(expit(delta))
        p = s * (1.0 - s)
        if linear_coefficients is None:
            linear_coefficients = self.linear_response(delta, tau, q)
        h = self.features.dot(linear_coefficients)
        sigma2 = p * (1.0 - 2.0 * s)
        sigma3 = p * (1.0 - 6.0 * s + 6.0 * s * s)
        source = 0.5 * sigma2 * q * h**2 - (sigma3 / 6.0) * h**3
        operator = (self.features.T * (self.weights * p)).dot(self.features)
        operator += tau * np.eye(self.features.shape[1])
        rhs = self.features.T.dot(self.weights * source)
        return solve(operator, rhs, assume_a="pos", check_finite=False)


def slope(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = (x > 0.0) & (y > 0.0) & np.isfinite(y)
    return float(np.polyfit(np.log(x[keep]), np.log(y[keep]), 1)[0])


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_plot(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman"],
        "font.size": 8.5,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.7))
    colors = {0.5: "#315B83", 1.0: "#B44948"}
    for delta in (0.5, 1.0):
        subset = sorted(
            [row for row in rows if abs(row["delta"] - delta) < 1e-12],
            key=lambda row: row["epsilon"],
        )
        epsilon = np.array([row["epsilon"] for row in subset])
        observed = np.array([row["observed_delta_l2"] for row in subset])
        centered = np.array([row["centered_delta_l2_error"] for row in subset])
        predicted = subset[0]["predicted_delta_l2"]
        color = colors[delta]
        axes[0].semilogx(
            epsilon, observed, "o-", color=color, ms=3,
            label=rf"observed, $\delta={delta:g}$",
        )
        axes[0].axhline(
            predicted, color=color, ls=":", lw=0.9,
            label=rf"predicted, $\delta={delta:g}$",
        )
        axes[1].loglog(
            epsilon, centered, "o-", color=color, ms=3,
            label=rf"$\delta={delta:g}$",
        )
    axes[0].set(
        xlabel=r"Signal amplitude $\epsilon$",
        ylabel=r"Normalized difference norm",
        title=r"(a) Observed versus predicted $\Delta_\delta$",
    )
    axes[1].set(
        xlabel=r"Signal amplitude $\epsilon$",
        ylabel=r"Centered $\Delta$ error",
        title=r"(b) General-geometry remainder",
    )
    axes[0].legend(frameon=False, fontsize=6.5, loc="best")
    axes[1].legend(frameon=False, fontsize=7, loc="best")
    for axis in axes:
        axis.grid(alpha=0.18, which="both")
        axis.tick_params(labelsize=7.2)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(path, bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--order", type=int, default=24)
    parser.add_argument("--out", default="validation")
    parser.add_argument("--figure", default="figures/general_cubic_rbf.pdf")
    args = parser.parse_args()

    x, weights = quadrature(args.order)
    q = (x[:, 0] + 0.5 * x[:, 1]) / 1.5
    model = PopulationModel(rbf_kernel(x, x), weights)
    epsilon_grid = np.array([0.08, 0.12, 0.16, 0.24])
    tau = 0.02
    rows = []

    for delta in (0.5, 1.0):
        s = float(expit(delta))
        p = s * (1.0 - s)
        lam = tau / (4.0 * p)
        gamma = s / (2.0 * p)
        linear = model.linear_response(delta, tau, q)
        cubic = model.cubic_response(delta, tau, q, linear)
        control_linear = model.linear_response(0.0, lam, q)
        control_cubic = model.cubic_response(0.0, lam, q, control_linear)
        predicted_delta = model.features.dot(cubic - gamma * control_cubic)
        predicted_probe = float(weights.dot(q * predicted_delta))
        predicted_norm = float(np.sqrt(weights.dot(predicted_delta**2)))

        for epsilon in epsilon_grid:
            probabilities = (1.0 + epsilon * q) / 2.0
            _, samplewise, sw_info = model.fit(delta, tau, probabilities)
            _, control, ctl_info = model.fit(0.0, lam, probabilities)
            control *= gamma
            observed = (samplewise - control) / epsilon**3
            centered = observed - predicted_delta
            observed_probe = float(weights.dot(q * observed))
            rows.append({
                "delta": float(delta),
                "epsilon": float(epsilon),
                "tau": tau,
                "lambda_match": float(lam),
                "gamma_match": float(gamma),
                "predicted_delta_l2": predicted_norm,
                "observed_delta_l2": float(np.sqrt(weights.dot(observed**2))),
                "centered_delta_l2_error": float(np.sqrt(weights.dot(centered**2))),
                "predicted_probe": predicted_probe,
                "observed_probe": observed_probe,
                "probe_absolute_error": abs(observed_probe - predicted_probe),
                "samplewise_gradient_residual": sw_info["gradient_residual"],
                "control_gradient_residual": ctl_info["gradient_residual"],
            })

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "general_cubic_rbf.csv", rows)
    summary = []
    for delta in (0.5, 1.0):
        subset = [row for row in rows if abs(row["delta"] - delta) < 1e-12]
        summary.append({
            "delta": delta,
            "centered_l2_slope": slope(
                [row["epsilon"] for row in subset],
                [row["centered_delta_l2_error"] for row in subset],
            ),
            "uncentered_l2_slope": slope(
                [row["epsilon"] for row in subset],
                [row["observed_delta_l2"] for row in subset],
            ),
            "max_probe_absolute_error": max(
                row["probe_absolute_error"] for row in subset
            ),
            "max_solver_residual": max(
                max(row["samplewise_gradient_residual"],
                    row["control_gradient_residual"])
                for row in subset
            ),
            "predicted_delta_l2": subset[0]["predicted_delta_l2"],
        })
    (output / "general_cubic_rbf_summary.json").write_text(
        json.dumps({
            "quadrature_order": args.order,
            "quadrature_points": len(x),
            "bandwidth": 0.7,
            "density": "(1+0.6*x1)(1+0.4*x2)/4",
            "signal": "(x1+0.5*x2)/1.5",
            "tau": tau,
            "epsilon_grid": epsilon_grid.tolist(),
            "paired_parameter_settings": len(rows),
            "independently_optimized_objectives": 2 * len(rows),
            "feature_checks": {
                "node_kernel_diag_max_error": model.node_kernel_diag_error,
                "weighted_feature_trace": model.weighted_feature_trace,
            },
            "summary": summary,
        }, indent=2), encoding="utf-8"
    )
    make_plot(rows, Path(args.figure))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()


