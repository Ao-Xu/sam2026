"""Fair-tuning and split-robustness supplement.

All output is written to matching/upgrade and the original matching outputs are
left untouched. Run with the Python 3.12 environment used for deep features:
python run_upgrade.py --splits 5
"""
from __future__ import annotations
import argparse, csv, json, math, sys, time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import expit

R = Path(__file__).resolve().parent
ROOT = R.parent
OUT = R / "upgrade"
sys.path.insert(0, str(R))
import run as base
import torch

TAU_GRID = np.array([1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3,
                     1e-2, 3e-2, 1e-1, 3e-1, 1.0], float)
# The five-radius grid actually used for Table G.7.
DELTA_GRID = np.array([0.0, 0.25, 0.5, 1.0, 2.0], float)
PAIRS = ((3, 5, "cat_dog"), (1, 9, "car_truck"), (0, 8, "plane_ship"))
MODELS = ("resnet18", "vit_b_16")
SEEDS = (20270901, 20270902, 20270903, 20270904, 20270905)

def loss(score, y):
    return float(np.logaddexp(0.0, -y * score).mean())

def fit(x, y, tau, delta):
    yt = base.ten(y[:, None])
    w, rec = base.solve(x, yt, float(tau), float(delta), tol=1e-9)
    return w.detach().cpu().numpy()[:, 0], rec

def match(delta=1.0, tau=.02):
    s = float(expit(delta)); p = s * (1. - s)
    return tau / (4. * p), s / (2. * p)

def choose(xtr, xva, ytr, yva, xtest, ytest, seed):
    xv = xva.detach().cpu().numpy()
    cand = {"SW-tuned": [], "ERM-tuned": [], "ERM-calibrated": []}
    # Same expanded tau grid for every method; SW additionally searches radius.
    for tau in TAU_GRID:
        for delta in DELTA_GRID:
            w, rec = fit(xtr, ytr, tau, delta)
            cand["SW-tuned"].append(dict(tau=float(tau), delta=float(delta),
                scale=1., val_loss=loss(xv @ w, yva),
                iterations=rec["iterations"], residual=rec["residual"]))
    for tau in TAU_GRID:
        w, rec = fit(xtr, ytr, tau, 0.)
        sv = xv @ w
        cand["ERM-tuned"].append(dict(tau=float(tau), delta=0., scale=1.,
            val_loss=loss(sv, yva), iterations=rec["iterations"],
            residual=rec["residual"]))
        def objective(log_scale):
            return loss(sv * math.exp(float(log_scale)), yva)
        opt = minimize_scalar(objective, bounds=(math.log(.05), math.log(20.)),
                              method="bounded")
        cand["ERM-calibrated"].append(dict(tau=float(tau), delta=0.,
            scale=float(math.exp(opt.x)), val_loss=float(opt.fun),
            iterations=rec["iterations"], residual=rec["residual"]))
    selected = {k: min(v, key=lambda z: z["val_loss"]) for k, v in cand.items()}
    lam, scale = match()
    selected["Matched"] = dict(tau=float(lam), delta=0., scale=float(scale),
                               val_loss=None)
    xt = xtest.detach().cpu().numpy()
    rows = []
    for name, cfg in selected.items():
        w, rec = fit(xtr, ytr, cfg["tau"], cfg["delta"])
        sv = xv @ w * cfg["scale"]; st = xt @ w * cfg["scale"]
        pr = expit(st); truth = (ytest + 1.) / 2.
        rows.append(dict(split_seed=int(seed), method=name, tau=cfg["tau"],
            delta=cfg["delta"], scale=cfg["scale"],
            validation_log_loss=loss(sv, yva), test_log_loss=loss(st, ytest),
            test_brier=float(np.mean((pr - truth) ** 2)),
            test_accuracy=float(np.mean((st > 0.) == (ytest > 0.))),
            test_score_variance=float(np.var(st, ddof=1)),
            solver_iterations=int(rec["iterations"]),
            solver_residual=float(rec["residual"]), n_train=int(len(ytr)),
            n_validation=int(len(yva)), n_test=int(len(ytest)),
            candidate_count=int(len(cand.get(name, []))),
            selection_grid=("expanded_tau_x_delta" if name == "SW-tuned"
                            else "expanded_tau" if name != "Matched" else "analytic")))
    return rows, {"candidates": cand, "selected": selected}

def split_indices(labels, a, b, seed):
    rng = np.random.default_rng(seed); tr = []; va = []
    for c in (a, b):
        ix = np.flatnonzero(labels == c)
        if len(ix) < 300: raise RuntimeError(f"class {c} has only {len(ix)} rows")
        ix = rng.permutation(ix[:300]); tr.extend(ix[:240]); va.extend(ix[240:300])
    return np.asarray(tr, int), np.asarray(va, int)

def write_csv(path, rows):
    if not rows: return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def run(n_splits, models):
    OUT.mkdir(parents=True, exist_ok=True)
    labels = np.load(ROOT / "deep_features/raw/data_indices.npz")
    all_rows = []; manifests = []; t0 = time.time()
    for model in models:
        data = np.load(ROOT / f"deep_features/raw/{model}_features.npz")
        tr_raw = data["train_raw"].astype(np.float64)
        te_raw = data["test_raw"].astype(np.float64)
        tr_raw /= np.linalg.norm(tr_raw, axis=1, keepdims=True)
        te_raw /= np.linalg.norm(te_raw, axis=1, keepdims=True)
        xall = base.ten(tr_raw); xteall = base.ten(te_raw)
        for pi, (a, b, pair) in enumerate(PAIRS):
            test_ix = np.flatnonzero((labels["test_labels"] == a) |
                                     (labels["test_labels"] == b))
            xt = xteall[test_ix]
            yt = np.where(labels["test_labels"][test_ix] == b, 1., -1.)
            for si, seed in enumerate(SEEDS[:n_splits]):
                tr, va = split_indices(labels["train_labels"], a, b,
                                        int(seed + 1000 * pi))
                ytr = np.where(labels["train_labels"][tr] == b, 1., -1.)
                yva = np.where(labels["train_labels"][va] == b, 1., -1.)
                rows, sel = choose(xall[tr], xall[va], ytr, yva, xt, yt, seed)
                for row in rows:
                    row.update(model=model, pair=pair, class_a=int(a),
                               class_b=int(b), split_index=int(si))
                all_rows.extend(rows)
                manifests.append(dict(model=model, pair=pair, split_index=int(si),
                    split_seed=int(seed), train_indices=tr.tolist(),
                    validation_indices=va.tolist(), test_indices=test_ix.tolist(),
                    selected=sel["selected"], candidates=sel["candidates"],
                    tau_grid=TAU_GRID.tolist(),
                    delta_grid=DELTA_GRID.tolist()))
                print(f"[{model} {pair} split {si + 1}/{n_splits}] "
                      f"elapsed={time.time() - t0:.1f}s", flush=True)
    write_csv(OUT / "fair_tuning_splits.csv", all_rows)
    (OUT / "selection_manifest.json").write_text(json.dumps(manifests, indent=2),
                                                   encoding="utf-8")
    summary = []
    for model in models:
        for pair in (p[2] for p in PAIRS):
            subset = [r for r in all_rows if r["model"] == model and r["pair"] == pair]
            by_split = {}
            for r in subset: by_split.setdefault(r["split_index"], {})[r["method"]] = r
            for method in ("SW-tuned", "ERM-tuned", "ERM-calibrated", "Matched"):
                z = np.asarray([v[method]["test_log_loss"] for v in by_split.values()])
                sd = float(z.std(ddof=1)) if len(z) > 1 else 0.
                se = sd / max(len(z), 1) ** .5
                summary.append(dict(model=model, pair=pair, method=method,
                    n_splits=int(len(z)), mean_test_log_loss=float(z.mean()),
                    sd_test_log_loss=sd, ci95_low=float(z.mean() - 1.96 * se),
                    ci95_high=float(z.mean() + 1.96 * se)))
            for comp in ("ERM-tuned", "ERM-calibrated", "Matched"):
                z = np.asarray([v[comp]["test_log_loss"] - v["SW-tuned"]["test_log_loss"]
                                for v in by_split.values()])
                sd = float(z.std(ddof=1)) if len(z) > 1 else 0.
                se = sd / max(len(z), 1) ** .5
                summary.append(dict(model=model, pair=pair,
                    method=f"{comp}-minus-SW-tuned", n_splits=int(len(z)),
                    mean_test_log_loss=float(z.mean()), sd_test_log_loss=sd,
                    ci95_low=float(z.mean() - 1.96 * se),
                    ci95_high=float(z.mean() + 1.96 * se)))
    write_csv(OUT / "fair_tuning_summary.csv", summary)
    meta = dict(protocol="fair_tuning_repeated_splits_v1", n_splits=int(n_splits),
        models=list(models), seeds=list(SEEDS[:n_splits]), tau_grid=TAU_GRID.tolist(),
        delta_grid=DELTA_GRID.tolist(),
        test_pool="fixed CIFAR test pool; split-level uncertainty only",
        output_rows=len(all_rows), elapsed_seconds=time.time() - t0,
        torch_version=torch.__version__, cuda_available=bool(torch.cuda.is_available()))
    (OUT / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--model", choices=MODELS, action="append")
    args = ap.parse_args()
    if args.splits < 1 or args.splits > len(SEEDS):
        raise SystemExit(f"--splits must be in [1,{len(SEEDS)}]")
    run(args.splits, tuple(args.model) if args.model else MODELS)
