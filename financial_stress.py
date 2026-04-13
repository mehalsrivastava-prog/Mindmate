import pandas as pd
import os
from sklearn.metrics import mean_absolute_error, r2_score

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
from sklearn.model_selection import train_test_split

X = df.drop("Financial_Stress_Score", axis=1)
y = df["Financial_Stress_Score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL ----------------
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=120,
    max_depth=6,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test)

print("\n📉 Mean Absolute Error (MAE):")
print(mean_absolute_error(y_test, y_pred))

print("\n📊 R2 Score:")
print(r2_score(y_test, y_pred))

# ---------------- SAVE ----------------
import joblib

joblib.dump(model, "financial_stress_regressor.pkl")

print("\n✅ Financial Stress model saved successfully!")