import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib

# ---------------- LOAD ----------------
df = pd.read_csv("data.csv")
df.columns = df.columns.str.strip()

# ---------------- SAMPLE ----------------
df = df.sample(1000, replace=True, random_state=42).reset_index(drop=True)

# ---------------- SELECT FEATURES ----------------
df = df[[
    "FWBscore",
    "FWB1_1",
    "FWB1_2",
    "FWB1_3",
    "SWB_1",
    "SWB_2",
    "SWB_3",
    "fpl"
]]

# ---------------- HANDLE MISSING ----------------
df.fillna(df.mean(numeric_only=True), inplace=True)

# ---------------- CREATE TARGET ----------------
df["Financial_Stress_Score"] = 100 - df["FWBscore"]

# ---------------- DROP ORIGINAL ----------------
df = df.drop("FWBscore", axis=1)

# ---------------- SPLIT ----------------
X = df.drop("Financial_Stress_Score", axis=1)
y = df["Financial_Stress_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL (LINEAR REGRESSION) ----------------
model = LinearRegression()

model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test)

print("\n📉 Mean Absolute Error (MAE):")
print(mean_absolute_error(y_test, y_pred))

print("\n📊 R2 Score:")
print(r2_score(y_test, y_pred))

# ---------------- OPTIONAL (UNDERSTANDING MODEL) ----------------
print("\n📌 Coefficients:")
for col, coef in zip(X.columns, model.coef_):
    print(f"{col}: {coef:.2f}")

print("\n📌 Intercept:")
print(model.intercept_)

# ---------------- CONVERT TO LABELS ----------------

def get_label(val):
    if val < 33:
        return 0   # Low
    elif val < 66:
        return 1   # Moderate
    else:
        return 2   # High

y_pred_labels = [get_label(v) for v in y_pred]
y_test_labels = [get_label(v) for v in y_test]
from sklearn.metrics import accuracy_score, classification_report

print("\n Classification Accuracy:")
print(accuracy_score(y_test_labels, y_pred_labels))

print("\n Classification Report:")
print(classification_report(y_test_labels, y_pred_labels))
# ---------------- SAVE ----------------
joblib.dump(model, "financial_stress_regressor.pkl")

print("\nLinear Regression model saved successfully!")