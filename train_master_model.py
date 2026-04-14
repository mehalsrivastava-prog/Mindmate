import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ---------------- LOAD DATA ----------------
df = pd.read_csv("master_dataset.csv")

# ---------------- ADD NOISE (FIXED POSITION) ----------------
noise_fraction = 0.1 
num_samples = int(noise_fraction * len(df))

indices = np.random.choice(df.index, num_samples, replace=False)

# smarter noise (shift class instead of random)
df.loc[indices, "target"] = (
    df.loc[indices, "target"] + np.random.choice([-1, 1], size=num_samples)
) % 3

# ---------------- FEATURES & TARGET ----------------
X = df[[
    "diet_score",
    "depression_score",
    "financial_score",
    "academic_score"
]]

y = df["target"]

# ---------------- TRAIN TEST SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- SCALING ----------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------- MODEL ----------------
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

model.fit(X_train, y_train)

# ---------------- PREDICTIONS ----------------
y_pred_test = model.predict(X_test)
y_pred_train = model.predict(X_train)

labels = ["Low", "Moderate", "High"]

y_test_labels = [labels[i] for i in y_test]
y_pred_labels = [labels[i] for i in y_pred_test]

# ---------------- RESULTS ----------------
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

# ---------------- SAVE ----------------
joblib.dump(model, "master_model.pkl")
joblib.dump(scaler, "master_scaler.pkl")

print("\nMaster model trained and saved successfully!")