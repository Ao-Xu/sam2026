"""Audit the joint-selection records and compare a rerun with paper inputs."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent / "upgrade"
TAUS = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0]
DELTAS = [0.0, 0.25, 0.5, 1.0, 2.0]
METHODS = ("SW-tuned", "ERM-tuned", "ERM-calibrated", "Matched")


def read_csv(name):
    with (OUT / name).open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def key(record):
    return record["model"], record["pair"], int(record["split_index"])


def near(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


rows = read_csv("fair_tuning_splits.csv")
summary = read_csv("fair_tuning_summary.csv")
manifests = json.loads((OUT / "selection_manifest.json").read_text(encoding="utf-8"))
meta = json.loads((OUT / "metadata.json").read_text(encoding="utf-8"))
assert len(manifests) == 30 and len(rows) == 120 and len(summary) == 42
assert meta["n_splits"] == 5 and meta["output_rows"] == 120
assert meta["tau_grid"] == TAUS and meta["delta_grid"] == DELTAS
assert {r["method"] for r in rows} == set(METHODS)
assert len({(key(r), r["method"]) for r in rows}) == len(rows)
row_by_key = {(key(r), r["method"]): r for r in rows}
candidate_counts = {"SW-tuned": 55, "ERM-tuned": 11, "ERM-calibrated": 11}

for m in manifests:
    k = key(m)
    tr, va, te = (set(m[field]) for field in
                  ("train_indices", "validation_indices", "test_indices"))
    assert len(tr) == 480 and len(va) == 120 and len(te) == 200
    assert not tr & va
    # Train/validation and test indices address distinct cached arrays.
    assert min(tr | va) >= 0 and max(tr | va) < 3000
    assert min(te) >= 0 and max(te) < 1000
    assert m["tau_grid"] == TAUS and m["delta_grid"] == DELTAS
    assert set(m["selected"]) == set(METHODS)
    assert set(m["candidates"]) == set(candidate_counts)
    for method, n in candidate_counts.items():
        candidates = m["candidates"][method]
        assert len(candidates) == n
        best = min(candidates, key=lambda c: c["val_loss"])
        chosen = m["selected"][method]
        assert all(near(chosen[field], best[field], 1e-12)
                   for field in ("tau", "delta", "scale", "val_loss"))
    sw_grid = {(c["tau"], c["delta"]) for c in m["candidates"]["SW-tuned"]}
    assert sw_grid == {(tau, delta) for tau in TAUS for delta in DELTAS}
    for method in METHODS:
        r = row_by_key[(k, method)]
        chosen = m["selected"][method]
        assert int(r["split_seed"]) == int(m["split_seed"])
        assert int(r["n_train"]) == 480 and int(r["n_validation"]) == 120
        assert int(r["n_test"]) == 200
        assert float(r["solver_residual"]) < 1e-9
        assert float(r["tau"]) > 0 and float(r["scale"]) > 0
        assert all(near(r[field], chosen[field], 1e-12)
                   for field in ("tau", "delta", "scale"))
        assert int(r["candidate_count"]) == candidate_counts.get(method, 0)
        if method != "Matched":
            assert near(r["validation_log_loss"], chosen["val_loss"], 1e-8)

summary_by_key = {(r["model"], r["pair"], r["method"]): r for r in summary}
max_summary_difference = 0.0
for model in ("resnet18", "vit_b_16"):
    for pair in ("cat_dog", "car_truck", "plane_ship"):
        split_rows = [{method: row_by_key[((model, pair, split), method)]
                       for method in METHODS} for split in range(5)]
        for method in list(METHODS) + [f"{m}-minus-SW-tuned" for m in METHODS[1:]]:
            if "-minus-SW-tuned" in method:
                base = method.removesuffix("-minus-SW-tuned")
                values = np.array([float(z[base]["test_log_loss"]) -
                                   float(z["SW-tuned"]["test_log_loss"])
                                   for z in split_rows])
            else:
                values = np.array([float(z[method]["test_log_loss"])
                                   for z in split_rows])
            target = summary_by_key[(model, pair, method)]
            mean = float(values.mean())
            sd = float(values.std(ddof=1))
            half_width = 1.96 * sd / np.sqrt(5)
            for field, value in (("mean_test_log_loss", mean),
                                 ("sd_test_log_loss", sd),
                                 ("ci95_low", mean - half_width),
                                 ("ci95_high", mean + half_width)):
                difference = abs(float(target[field]) - value)
                max_summary_difference = max(max_summary_difference, difference)
                assert difference < 1e-10, (model, pair, method, field, difference)

reference_summary = read_csv("paper_reference_summary.csv")
assert len(reference_summary) == len(summary)
reference_by_key = {(r["model"], r["pair"], r["method"]): r
                    for r in reference_summary}
max_paper_summary_difference = 0.0
for k, row in summary_by_key.items():
    ref = reference_by_key[k]
    for field in ("mean_test_log_loss", "sd_test_log_loss", "ci95_low", "ci95_high"):
        max_paper_summary_difference = max(
            max_paper_summary_difference, abs(float(row[field]) - float(ref[field])))
assert max_paper_summary_difference < 1e-8

reference_manifests = json.loads(
    (OUT / "paper_reference_selection_manifest.json").read_text(encoding="utf-8"))
reference_by_key = {key(m): m for m in reference_manifests}
max_paper_candidate_loss_difference = 0.0
for m in manifests:
    ref = reference_by_key[key(m)]
    assert m["train_indices"] == ref["train_indices"]
    assert m["validation_indices"] == ref["validation_indices"]
    assert m["test_indices"] == ref["test_indices"]
    for method in candidate_counts:
        candidates, expected = m["candidates"][method], ref["candidates"][method]
        assert len(candidates) == len(expected)
        for observed, archived in zip(candidates, expected):
            assert all(near(observed[field], archived[field], 1e-12)
                       for field in ("tau", "delta", "scale"))
            max_paper_candidate_loss_difference = max(
                max_paper_candidate_loss_difference,
                abs(float(observed["val_loss"]) - float(archived["val_loss"])))
assert max_paper_candidate_loss_difference < 1e-8

report = {
    "status": "PASS",
    "split_rows": len(rows),
    "selection_manifests": len(manifests),
    "summary_rows": len(summary),
    "samplewise_candidates_per_split": candidate_counts["SW-tuned"],
    "max_solver_residual": max(float(r["solver_residual"]) for r in rows),
    "max_internal_summary_difference": max_summary_difference,
    "max_paper_summary_difference": max_paper_summary_difference,
    "max_paper_candidate_loss_difference": max_paper_candidate_loss_difference,
    "split_disjointness": "PASS",
    "selection_vs_selected_fit": "PASS",
    "test_pool": "fixed separate cached array; no test-loss selection",
}
(OUT / "verification.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
