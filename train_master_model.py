import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

df = pd.read_csv("master_dataset.csv")

X = df[[
    "diet_score",
    "depression_score",
    "financial_score",
    "academic_score"
]]

y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

labels = ["Low", "Moderate", "High"]

y_test_labels = [labels[i] for i in y_test]
y_pred_labels = [labels[i] for i in y_pred_test]

print("\nTest Accuracy:")
print(accuracy_score(y_test, y_pred_test))

print("\nTrain Accuracy:")
print(accuracy_score(y_train, y_pred_train))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_test))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_test))

print("\nSample Predictions:")
for i in range(10):
    print(f"Actual: {y_test_labels[i]}  |  Predicted: {y_pred_labels[i]}")

joblib.dump(model, "master_model.pkl")
joblib.dump(scaler, "master_scaler.pkl")

print("\nMaster model trained and saved successfully!")