"""Plot split-level fair-tuning results and write a compact manuscript table."""
from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "upgrade" / "fair_tuning_summary.csv"
FIG = ROOT / "upgrade" / "fair_tuning.pdf"
TABLE = ROOT / "upgrade" / "fair_tuning.tex"

rows = list(csv.DictReader(CSV.open(encoding="utf-8")))
methods = ["SW-tuned", "ERM-tuned", "ERM-calibrated", "Matched"]
labels = {
    "SW-tuned": "SW tuned",
    "ERM-tuned": "ERM tuned",
    "ERM-calibrated": "ERM + scale",
    "Matched": "analytic match",
}
models = ["resnet18", "vit_b_16"]
pairs = ["cat_dog", "car_truck", "plane_ship"]
names = {"cat_dog": "cat/dog", "car_truck": "car/truck", "plane_ship": "plane/ship"}

fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.45), sharey=False)
colors = ["#2166ac", "#4daf4a", "#e41a1c", "#777777"]
for ax, model in zip(axes, models):
    x = np.arange(len(pairs))
    width = 0.19
    for j, method in enumerate(methods):
        vals, errs = [], []
        for pair in pairs:
            r = next(z for z in rows if z["model"] == model and z["pair"] == pair and z["method"] == method)
            vals.append(float(r["mean_test_log_loss"]))
            errs.append(1.96 * float(r["sd_test_log_loss"]) / np.sqrt(float(r["n_splits"])))
        ax.errorbar(x + (j - 1.5) * width, vals, yerr=errs, fmt="o", ms=3.6,
                    lw=1.1, capsize=2.2, color=colors[j], label=labels[method])
    ax.set_xticks(x)
    ax.set_xticklabels([names[p] for p in pairs], rotation=20, ha="right")
    ax.set_title("ResNet-18" if model == "resnet18" else "ViT-B/16", fontsize=9)
    ax.grid(axis="y", alpha=.25, linewidth=.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=7.5)
    ax.set_ylabel("test log loss" if model == "resnet18" else "")
axes[0].legend(frameon=False, fontsize=6.6, ncol=2, loc="upper left")
fig.tight_layout(w_pad=1.0)
FIG.parent.mkdir(exist_ok=True)
fig.savefig(FIG, bbox_inches="tight")
plt.close(fig)

def fmt(x):
    return f"{float(x):.3f}"

TABLE.parent.mkdir(exist_ok=True)
with TABLE.open("w", encoding="utf-8") as f:
    f.write("\\begin{table}[t]\n\\centering\n")
    f.write("\\caption{Fair validation-selected comparison over five independent stratified splits. Entries are mean test log loss with split-level 95\\% normal intervals; the fixed test pool is never used for selection. The analytic match is a mechanism control, not a tuned performance baseline.}\\label{tab:fair-tuning}\n")
    f.write("\\resizebox{\\linewidth}{!}{%\n")
    f.write("\\begin{tabular}{llrrrr}\n\\toprule\nFeatures & Task & SW tuned & ERM tuned & ERM + scale & Analytic match\\\\\n\\midrule\n")
    for model in models:
        for pair in pairs:
            vals=[]
            for method in methods:
                r = next(z for z in rows if z["model"] == model and z["pair"] == pair and z["method"] == method)
                mean=float(r["mean_test_log_loss"]); half=1.96*float(r["sd_test_log_loss"])/np.sqrt(float(r["n_splits"]))
                vals.append(f"{mean:.3f}\\,$\\pm$\\,{half:.3f}")
            f.write(f"{'ResNet-18' if model=='resnet18' else 'ViT-B/16'} & {names[pair]} & " + " & ".join(vals) + "\\\\\n")
    f.write("\\bottomrule\n\\end{tabular}}\n\\end{table}\n")
print(FIG)
print(TABLE)
