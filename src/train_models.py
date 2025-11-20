import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

from preprocess import build_preprocessor

# Load dataset
df = pd.read_csv("data/preterm_dataset.csv")
target = "Pre-term"

X = df.drop(columns=[target])
y = df[target]

preprocessor, num_feats, cat_feats = build_preprocessor(df, target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Models
models = {
    "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss"),
    "DNN": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
}

results = {}

for name, model in models.items():
    from sklearn.pipeline import Pipeline
    clf = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)
    results[name] = report
    joblib.dump(clf, f"models/{name}.pkl")
    print(f"{name} trained and saved!")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"{name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.savefig(f"reports/{name}_cm.png")

# Save report
with open("reports/report.html", "w") as f:
    f.write("<h1>Model Reports</h1>")
    for name, report in results.items():
        f.write(f"<h2>{name}</h2>")
        f.write("<pre>" + str(report) + "</pre>")
 