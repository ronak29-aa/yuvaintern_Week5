
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RAW = pd.read_csv(ROOT / "data" / "simulated_raw_iris_quality_issues.csv")
DF = pd.read_csv(ROOT / "data" / "cleaned_iris_dataset.csv")
VIZ = ROOT / "visualizations"

# Data-quality summary
summary = pd.DataFrame({
    "Stage": ["Before cleaning", "After cleaning"],
    "Missing_cells": [RAW.isna().sum().sum(), DF.isna().sum().sum()],
    "Duplicate_rows": [RAW.duplicated().sum(), DF.duplicated().sum()]
})
summary.to_csv(ROOT / "data" / "data_quality_summary.csv", index=False)

fig, ax = plt.subplots()
ax.bar(summary.Stage, summary.Missing_cells)
ax.set_title("Data Quality: Missing Values Before and After Cleaning")
ax.set_ylabel("Missing cells")
fig.tight_layout()
fig.savefig(VIZ / "01_data_quality.png", dpi=180)
plt.close(fig)
print(summary)
