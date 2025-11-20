import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import joblib

# 1. Load dataset
df = pd.read_csv("data/dummy_preterm_data.csv")  # path to new dataset

# 2. Separate features and target
X = df.drop(columns=["Preterm"])  # replace 'Preterm' with your target column name
y = df["Preterm"]

# 3. Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Train Random Forest
model_new = RandomForestClassifier(n_estimators=100, random_state=42)
model_new.fit(X_train, y_train)

# 5. Evaluate
y_pred = model_new.predict(X_test)

# Accuracy
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"Test Accuracy: {accuracy:.2f}%")

# Precision, Recall, F1-score
precision = precision_score(y_test, y_pred) * 100
recall = recall_score(y_test, y_pred) * 100
f1 = f1_score(y_test, y_pred) * 100

print(f"Precision: {precision:.2f}%")
print(f"Recall: {recall:.2f}%")
print(f"F1 Score: {f1:.2f}%")

# Detailed classification report in %
report = classification_report(y_test, y_pred, digits=4)
print("Classification Report:\n", report)

# 6. Save the trained model
joblib.dump(model_new, "models/RandomForest_New.pkl")
print("Model saved as models/RandomForest_New.pkl")
