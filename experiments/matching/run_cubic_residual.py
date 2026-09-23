"""Validate the cubic and matched-control population remainders.

The diagnostic is deterministic: it compares independently optimized periodic
population fits with their analytic cubic truncations. The script reports
both the uncentered normalized matched difference, which converges to
Delta_delta[q], and the centered error, which is predicted to be
O(epsilon**2) after division by epsilon**3.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.special import expit

PROJECT = Path(__file__).resolve().parents[1]


def kernel_eigenvalue(j: int) -> float:
    denominator = 1.0 + np.pi**4 / 45.0
    return 1.0 / denominator if j == 0 else 1.0 / (denominator * j**4)


def coefficients(tau: float, delta: float):
    s = float(expit(delta))
    p = s * (1.0 - s)
    sigma2 = p * (1.0 - 2.0 * s)
    sigma3 = p * (1.0 - 6.0 * s + 6.0 * s * s)
    mu1 = kernel_eigenvalue(1)
    mu3 = kernel_eigenvalue(3)
    a = mu1 * s / (tau + mu1 * p)
    source = sigma2 * a * a / 8.0 - sigma3 * a**3 / 24.0
    b1 = 3.0 * mu1 * source / (tau + mu1 * p)
    b3 = mu3 * source / (tau + mu3 * p)
    return a, b1, b3


def rms(values) -> float:
    return float(np.sqrt(np.mean(np.square(values))))


def slope(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = (x > 0) & (y > 0) & np.isfinite(y)
    if int(keep.sum()) < 2:
        return float("nan")
    return float(np.polyfit(np.log(x[keep]), np.log(y[keep]), 1)[0])


def write_csv(path: Path, rows) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def find_data(explicit: Optional[str]) -> Path:
    if explicit:
        candidate = Path(explicit).expanduser().resolve()
        if not candidate.exists():
            raise FileNotFoundError(candidate)
        return candidate
    candidates = (
        PROJECT.parent / "experiments" / "matching" / "raw" / "periodic_1024.npz",
        PROJECT / "experiments" / "matching" / "raw" / "periodic_1024.npz",
        PROJECT / "data" / "periodic_1024.npz",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "periodic_1024.npz was not found; pass it with --data."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", help="periodic population cache (.npz)")
    parser.add_argument("--out", default=str(PROJECT / "validation"),
                        help="directory for CSV, JSON, and slope summaries")
    args = parser.parse_args()

    data_path = find_data(args.data)
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = np.load(data_path)
    x, epsilon, radii = data["x"], data["epsilon"], data["radii"]
    fit, matched = data["fit"], data["matched"]
    tau = 0.02
    rows = []
    paths = []

    for radius_index, delta in enumerate(radii):
        a, b1, b3 = coefficients(tau, float(delta))
        raw_sw, raw_ctl, centered = [], [], []
        uncentered = []
        lam = tau / (4.0 * expit(delta) * (1.0 - expit(delta)))
        gamma = expit(delta) / (2.0 * expit(delta) * (1.0 - expit(delta)))
        a0, b10, b30 = coefficients(lam, 0.0)

        for epsilon_index, e in enumerate(epsilon):
            lead = e * a * np.cos(x)
            cubic = lead + e**3 * (b1 * np.cos(x) + b3 * np.cos(3.0 * x))
            lead0 = e * gamma * a0 * np.cos(x)
            cubic0 = lead0 + e**3 * gamma * (
                b10 * np.cos(x) + b30 * np.cos(3.0 * x)
            )
            f = fit[radius_index, :, epsilon_index]
            g = matched[radius_index, :, epsilon_index]

            raw_sw_error = rms(f - cubic)
            raw_ctl_error = rms(g - cubic0)
            predicted_delta = (cubic - lead) / e**3 - (cubic0 - lead0) / e**3
            observed_delta = (f - g) / e**3
            uncentered_error = rms(observed_delta)
            centered_error = rms(observed_delta - predicted_delta)
            raw_sw.append(raw_sw_error)
            raw_ctl.append(raw_ctl_error)
            uncentered.append(uncentered_error)
            centered.append(centered_error)

            sw_c3 = 2.0 * np.mean(((f - lead) / e**3) * np.cos(3.0 * x))
            rows.append({
                "radius": float(delta),
                "epsilon": float(e),
                "samplewise_cubic_truncation_remainder": raw_sw_error,
                "matched_cubic_truncation_remainder": raw_ctl_error,
                "matched_difference_normalized_uncentered": uncentered_error,
                "matched_difference_centered_remainder": centered_error,
                "predicted_matched_difference_norm": rms(predicted_delta),
                "samplewise_c3_residual": float(sw_c3 - b3),
                "predicted_samplewise_c3": float(b3),
                "predicted_matched_c3": float(gamma * b30),
                "lambda_match": float(lam),
                "gamma_match": float(gamma),
            })

        paths.append({
            "radius": float(delta),
            "samplewise_cubic_truncation_slope": slope(epsilon, raw_sw),
            "matched_cubic_truncation_slope": slope(epsilon, raw_ctl),
            "matched_difference_normalized_uncentered_slope": slope(epsilon, uncentered),
            "matched_difference_centered_slope": slope(epsilon, centered),
            "max_samplewise_cubic_truncation_remainder": float(max(raw_sw)),
            "max_centered_matched_difference_remainder": float(max(centered)),
        })

    write_csv(out / "cubic_residuals.csv", rows)
    (out / "cubic_residual_slopes.json").write_text(
        json.dumps(paths, indent=2), encoding="utf-8"
    )
    (out / "cubic_residual_metadata.json").write_text(
        json.dumps({
            "source": data_path.name,
            "tau": tau,
            "raw_remainder_definition": "fit minus complete linear-plus-cubic truncation",
            "matched_normalized_definition": "(samplewise-minus-control)/epsilon^3",
            "centered_definition": "normalized difference minus predicted Delta_delta[q]",
            "expected_raw_order": 5,
            "expected_uncentered_limit": "Delta_delta[q]",
            "expected_centered_order": 2,
            "quadrature": "deterministic 1024-point periodic population",
        }, indent=2), encoding="utf-8"
    )
    print(json.dumps(paths, indent=2))


if __name__ == "__main__":
    main()

