
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "cleaned_iris_dataset.csv")
X = df[["sepal_length","sepal_width","petal_length","petal_width"]]
VIZ = ROOT / "visualizations"

# Petal relationship
fig, ax = plt.subplots()
for s in ["setosa","versicolor","virginica"]:
    sub = df[df.species == s]
    ax.scatter(sub.petal_length, sub.petal_width, label=s)
ax.set_title("Petal Length vs Petal Width by Species")
ax.set_xlabel("Petal length")
ax.set_ylabel("Petal width")
ax.legend()
fig.tight_layout()
fig.savefig(VIZ / "03_petal_relationship.png", dpi=180)
plt.close(fig)

# Correlation
corr = X.corr()
fig, ax = plt.subplots()
im = ax.imshow(corr, aspect="auto")
ax.set_xticks(range(4), X.columns, rotation=30, ha="right")
ax.set_yticks(range(4), X.columns)
ax.set_title("Feature Correlation Matrix")
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{corr.iloc[i,j]:.2f}", ha="center", va="center")
fig.colorbar(im, ax=ax)
fig.tight_layout()
fig.savefig(VIZ / "04_correlation_heatmap.png", dpi=180)
plt.close(fig)
print("Core EDA visualizations regenerated.")
