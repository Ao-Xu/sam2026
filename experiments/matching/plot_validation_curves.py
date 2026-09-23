"""Plot validation curves from the repeated-split selection manifest."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent
OUT = R / "upgrade"
man = json.loads((OUT / "selection_manifest.json").read_text(encoding="utf-8"))
models = ["resnet18", "vit_b_16"]
pairs = ["cat_dog", "car_truck", "plane_ship"]
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman"],
                     "font.size": 8, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.0), layout="constrained",
                         sharex=True)
colors = {"SW-tuned": "#315B83", "ERM-tuned": "#B44948",
          "ERM-calibrated": "#2F855A"}
for i, model in enumerate(models):
    for j, pair in enumerate(pairs):
        ax = axes[i, j]
        records = [m for m in man if m["model"] == model and m["pair"] == pair]
        for method in ("SW-tuned", "ERM-tuned", "ERM-calibrated"):
            vals = []
            taus = sorted({float(v["tau"]) for m in records
                           for v in m["candidates"][method]})
            for tau in taus:
                per_split = []
                for m in records:
                    cand = [v for v in m["candidates"][method]
                            if abs(float(v["tau"]) - tau) < 1e-14]
                    # SW has five deltas per tau; use the best validation loss
                    # at each tau, while ordinary controls have one candidate.
                    per_split.append(min(float(v["val_loss"]) for v in cand))
                vals.append((np.mean(per_split), np.std(per_split, ddof=1)))
            mean = np.array([v[0] for v in vals]); sd = np.array([v[1] for v in vals])
            tau = np.array(taus)
            ax.plot(tau, mean, "o-", ms=2.5, lw=.9, color=colors[method],
                    label=method.replace("-tuned", "").replace("ERM-", "ERM "))
            ax.fill_between(tau, mean - sd, mean + sd, color=colors[method],
                            alpha=.10, linewidth=0)
        ax.set_xscale("log")
        ax.grid(alpha=.16, which="both")
        ax.set_title(f"{model.replace('_', ' ')} / {pair}", fontsize=8)
        if i == 1: ax.set_xlabel(r"$\tau$")
        if j == 0: ax.set_ylabel("Validation log loss")
        if i == 0 and j == 2: ax.legend(frameon=False, fontsize=6, loc="best")
fig.savefig(OUT / "validation_curves.pdf")
fig.savefig(OUT / "validation_curves.png", dpi=220)
plt.close(fig)
