import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ================= LOAD DATA =================
df = pd.read_csv("master_dataset.csv")

X = df[[
    "diet_score",
    "depression_score",
    "financial_score",
    "academic_score"
]]

y = df["target"]

# ================= SPLIT =================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ================= SCALING (VERY IMPORTANT) =================
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ================= MODEL =================
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# ================= PREDICTIONS =================
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

# ================= LABEL MAPPING =================
labels = ["Low", "Moderate", "High"]

y_test_labels = [labels[i] for i in y_test]
y_pred_labels = [labels[i] for i in y_pred_test]

# ================= EVALUATION =================
print("\n📊 Test Accuracy:")
print(accuracy_score(y_test, y_pred_test))

print("\n📊 Train Accuracy:")
print(accuracy_score(y_train, y_pred_train))

print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred_test))

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_test))

# ================= SAMPLE OUTPUT =================
print("\n🔍 Sample Predictions:")
for i in range(10):
    print(f"Actual: {y_test_labels[i]}  |  Predicted: {y_pred_labels[i]}")

# ================= SAVE =================
joblib.dump(model, "master_model.pkl")
joblib.dump(scaler, "master_scaler.pkl")

print("\n✅ Master model trained and saved successfully!")