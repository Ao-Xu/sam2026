"""Plot the deterministic cubic residual diagnostics."""
import argparse
from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--input", default=str(PROJECT / "validation" / "cubic_residuals.csv"))
parser.add_argument("--output", default=str(PROJECT / "figures" / "cubic_residual_diagnostic.pdf"))
args = parser.parse_args()
OUT = Path(args.input).expanduser().resolve().parent
FIG = Path(args.output).expanduser().resolve()
FIG.parent.mkdir(parents=True, exist_ok=True)
rows = list(csv.DictReader(Path(args.input).expanduser().resolve().open(encoding="utf-8")))
for row in rows:
    for key, value in list(row.items()):
        if key not in ("radius", "epsilon"):
            row[key] = float(value)
    row["radius"] = float(row["radius"])
    row["epsilon"] = float(row["epsilon"])

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 8.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
})
fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.8))
colors = {0.02: "#315B83", 0.25: "#B44948"}

for delta in (0.02, 0.25):
    subset = [row for row in rows if abs(row["radius"] - delta) < 1e-10]
    subset.sort(key=lambda row: row["epsilon"])
    epsilon = np.array([row["epsilon"] for row in subset])
    color = colors[delta]
    label = rf"$\delta={delta:g}$"

    axes[0].loglog(
        epsilon,
        [row["samplewise_cubic_truncation_remainder"] for row in subset],
        "o-", color=color, ms=3, label=f"samplewise, {label}",
    )
    axes[0].loglog(
        epsilon,
        [row["matched_cubic_truncation_remainder"] for row in subset],
        "s--", color=color, ms=3, alpha=0.75,
        label=f"matched, {label}",
    )
    axes[1].semilogx(
        epsilon,
        [row["matched_difference_normalized_uncentered"] for row in subset],
        "o-", color=color, ms=3, label=label,
    )
    axes[1].axhline(
        subset[0]["predicted_matched_difference_norm"],
        color=color, ls=":", lw=0.9, alpha=0.85,
    )
    axes[2].loglog(
        epsilon,
        [row["matched_difference_centered_remainder"] for row in subset],
        "o-", color=color, ms=3, label=label,
    )

axes[0].set(
    xlabel=r"Signal amplitude $\epsilon$",
    ylabel=r"$L^2$ error",
    title=r"(a) Cubic truncation: $O(\epsilon^5)$",
)
axes[1].set(
    xlabel=r"Signal amplitude $\epsilon$",
    ylabel=r"$\|(f-g)/\epsilon^3\|_2$",
    title=r"(b) Uncentered limit: $\Delta_\delta[q]$",
)
axes[2].set(
    xlabel=r"Signal amplitude $\epsilon$",
    ylabel=r"Centered $L^2$ error",
    title=r"(c) Centered error: $O(\epsilon^2)$",
)
axes[0].legend(frameon=False, fontsize=6.5, loc="best")
axes[1].legend(frameon=False, fontsize=7, loc="best")
axes[2].legend(frameon=False, fontsize=7, loc="best")
for axis in axes:
    axis.grid(alpha=0.18, which="both")
    axis.tick_params(labelsize=7.2)

fig.tight_layout(w_pad=1.0)
fig.savefig(FIG, bbox_inches="tight")
fig.savefig(FIG.with_suffix(".png"), dpi=220, bbox_inches="tight")
plt.close(fig)



