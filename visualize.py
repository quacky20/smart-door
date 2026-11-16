import json
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

output_dir = "evaluation_plots"
os.makedirs(output_dir, exist_ok=True)

with open("evaluation_results.json", "r") as f:
  data = json.load(f)

# plotting style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 1.0,
}
)

per_class = data["per_class"]
classes = [k for k in data["labels"] if k != "Unknown"]

metrics_df = pd.DataFrame([
    {
        "Class": cls,
        "Support": per_class[cls]["support"],
        "Precision": per_class[cls]["precision"],
        "Recall": per_class[cls]["recall"],
        "F1-Score": per_class[cls]["f1"],
    }
    for cls in classes
])

# sorting by F1-Score
metrics_df = metrics_df.sort_values(by="F1-Score", ascending=False)

# ==========================================
# PLOT 1: Per-Class F1-Score (Horizontal Bar)
# ==========================================
plt.figure(figsize=(10, 8))
ax = sns.barplot(
    x="F1-Score", y="Class", data=metrics_df, hue="Class", palette="crest", legend=False
)

plt.title(
    "Smart Door Face Recognition: Per-Class F1-Score",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
plt.xlabel("F1-Score", fontsize=12)
plt.ylabel("Identity Class", fontsize=12)
plt.xlim(0, 1.05)

for p in ax.patches:
  width = p.get_width()
  if width > 0:
    ax.annotate(
        f"{width:.2f}",
        (width + 0.01, p.get_y() + p.get_height() / 2.0),
        ha="left",
        va="center",
        fontsize=9,
        color="black",
    )

sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "per_class_f1_scores.png"), dpi=300)
plt.close()

# ==========================================
# PLOT 2: Per-Class Metric Heatmap
# ==========================================
plt.figure(figsize=(8, 10))
heatmap_data = metrics_df.set_index("Class")[
    ["Precision", "Recall", "F1-Score"]
]

sns.heatmap(
    heatmap_data,
    annot=True,
    fmt=".2f",
    cmap="YlGnBu",
    cbar=True,
    linewidths=0.5,
    vmin=0,
    vmax=1,
)

plt.title(
    "Per-Class Performance Metrics Heatmap",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
plt.ylabel("Identity Class", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "per_class_heatmap.png"), dpi=300)
plt.close()

# ==========================================
# PLOT 3: Overall System Performance Summary
# ==========================================
summary_metrics = pd.DataFrame({
    "Metric": [
        "Overall Accuracy",
        "Macro Recall",
        "Macro F1",
        "Weighted F1",
        "Detection Rate",
    ],
    "Score": [
        data["accuracy"],
        data["recall_macro_known"],
        data["f1_macro_known"],
        data["f1_weighted_known"],
        data["detection_rate"],
    ],
})

plt.figure(figsize=(8, 5))
ax = sns.barplot(
    x="Score",
    y="Metric",
    data=summary_metrics,
    hue="Metric",
    palette="viridis",
    legend=False,
)

plt.title(
    "Smart Door Evaluation: Overall System Metrics",
    fontsize=14,
    fontweight="bold",
    pad=15,
)
plt.xlabel("Score Value", fontsize=12)
plt.ylabel("Metric Type", fontsize=12)
plt.xlim(0, 1.1)

for p in ax.patches:
  width = p.get_width()
  ax.annotate(
      f"{width:.1%}",
      (width + 0.02, p.get_y() + p.get_height() / 2.0),
      ha="left",
      va="center",
      fontsize=10,
      fontweight="bold",
  )

sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "overall_system_metrics.png"), dpi=300)
plt.close()

print(
    f"All plots have been successfully generated and saved inside the"
    f" '{output_dir}/' directory!"
)