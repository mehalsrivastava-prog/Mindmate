import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

# ---------------- LOAD ----------------
df = pd.read_csv("academic_stress.csv")

# ---------------- CLEAN ----------------
df.columns = df.columns.str.strip().str.lower()

# ---------------- MAP COLUMNS ----------------
col_map = {}

for col in df.columns:
    if "competition" in col:
        col_map["academic_competition"] = col
    elif "coping" in col:
        col_map["coping_strategy"] = col
    elif "environment" in col:
        col_map["study_environment"] = col
    elif "home" in col:
        col_map["home_pressure"] = col
    elif "peer" in col:
        col_map["peer_pressure"] = col
    elif "stress" in col:
        col_map["stress_level"] = col

# ---------------- CREATE CLEAN DF ----------------
df_clean = pd.DataFrame()

for new_col, old_col in col_map.items():
    df_clean[new_col] = df[old_col]

# ---------------- CONVERT RANGE ----------------
def convert_range(val):
    if isinstance(val, str) and "-" in val:
        try:
            a, b = val.split("-")
            return (float(a) + float(b)) / 2
        except:
            return np.nan
    try:
        return float(val)
    except:
        return np.nan

for col in ["academic_competition", "home_pressure", "peer_pressure", "stress_level"]:
    df_clean[col] = df_clean[col].apply(convert_range)

# ---------------- ONE HOT ENCODING ----------------
# Analyze becomes baseline automatically
df_clean = pd.get_dummies(
    df_clean,
    columns=["coping_strategy", "study_environment"],
    drop_first=True
)

# ---------------- HANDLE MISSING ----------------
df_clean.fillna(df_clean.mean(numeric_only=True), inplace=True)

# ---------------- FEATURES ----------------
X = df_clean.drop("stress_level", axis=1)
y = df_clean["stress_level"]

feature_order = X.columns.tolist()
joblib.dump(feature_order, "feature_order.pkl")

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL ----------------
model = LinearRegression()
model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test)

print("\n📊 Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2:", r2_score(y_test, y_pred))

# ---------------- FEATURE IMPORTANCE ----------------
print("\n📊 Feature Importance (Weights):")
for feature, coef in zip(X.columns, model.coef_):
    print(feature, ":", round(coef, 3))

# ---------------- SAVE ----------------
joblib.dump(model, "academic_stress_model.pkl")

print("\n✅ Model trained successfully!")