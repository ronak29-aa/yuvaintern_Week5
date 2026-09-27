
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "cleaned_iris_dataset.csv"

df = pd.read_csv(DATA)
features = ["sepal_length","sepal_width","petal_length","petal_width"]
X, y = df[features], df["species"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, random_state=42))
    ]),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, random_state=42)
}

rows, cv_rows = [], []
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision_weighted": precision_score(y_test, pred, average="weighted"),
        "Recall_weighted": recall_score(y_test, pred, average="weighted"),
        "F1_weighted": f1_score(y_test, pred, average="weighted")
    })
    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy")
    cv_rows.append({"Model": name, "CV_Mean_Accuracy": scores.mean(), "CV_SD": scores.std()})

metrics = pd.DataFrame(rows)
cv = pd.DataFrame(cv_rows)
metrics.to_csv(ROOT / "data" / "model_metrics.csv", index=False)
cv.to_csv(ROOT / "data" / "cross_validation_results.csv", index=False)

lr = models["Logistic Regression"]
perm = permutation_importance(lr, X_test, y_test, n_repeats=20, random_state=42, scoring="accuracy")
pd.DataFrame({
    "Feature": features,
    "Importance_Mean": perm.importances_mean,
    "Importance_SD": perm.importances_std
}).sort_values("Importance_Mean", ascending=False).to_csv(
    ROOT / "data" / "permutation_importance.csv", index=False
)

print(metrics.to_string(index=False))
print("\nCross-validation:")
print(cv.to_string(index=False))
