
from pathlib import Path
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
df = pd.read_csv(ROOT / "data" / "cleaned_iris_dataset.csv")
groups = [df.loc[df.species == s, "petal_length"].values for s in ["setosa","versicolor","virginica"]]

anova = stats.f_oneway(*groups)
levene = stats.levene(*groups, center="median")

out = pd.DataFrame([{
    "ANOVA_F": anova.statistic,
    "ANOVA_p": anova.pvalue,
    "Levene_p": levene.pvalue
}])
out.to_csv(ROOT / "data" / "statistical_test_summary.csv", index=False)
print(out.to_string(index=False))
