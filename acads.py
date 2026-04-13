import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

df = pd.read_csv("academic_stress.csv")

# ---------------- CLEAN COLUMN NAMES ----------------
df.columns = df.columns.str.strip().str.lower()

# ---------------- CLEAN COLUMN NAMES ----------------
df.columns = df.columns.str.strip().str.lower()

print("Available columns:", df.columns.tolist())

# ---------------- AUTO MAP COLUMNS ----------------
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

print("Mapped columns:", col_map)

# ---------------- CREATE CLEAN DATAFRAME ----------------
df_clean = pd.DataFrame()

for new_col, old_col in col_map.items():
    df_clean[new_col] = df[old_col]

# ---------------- CONVERT RANGE → NUMBER ----------------
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
    if col in df_clean.columns:
        df_clean[col] = df_clean[col].apply(convert_range)

# ---------------- ENCODE CATEGORICAL ----------------
le_coping = LabelEncoder()
le_env = LabelEncoder()

df_clean["coping_strategy"] = le_coping.fit_transform(df_clean["coping_strategy"].astype(str))
df_clean["study_environment"] = le_env.fit_transform(df_clean["study_environment"].astype(str))

# ---------------- HANDLE MISSING ----------------
df_clean.fillna(df_clean.mean(numeric_only=True), inplace=True)

# ---------------- FEATURES & TARGET ----------------
X = df_clean.drop("stress_level", axis=1)
y = df_clean["stress_level"]
feature_order = X.columns.tolist()
joblib.dump(feature_order, "feature_order.pkl")

# ---------------- SPLIT ----------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ---------------- MODEL ----------------
model = RandomForestRegressor(
    n_estimators=150,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# ---------------- EVALUATION ----------------
y_pred = model.predict(X_test)

print("\n📊 Model Performance:")
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# ---------------- SAVE ----------------
joblib.dump(model, "academic_stress_model.pkl")
joblib.dump(le_coping, "coping_encoder.pkl")
joblib.dump(le_env, "env_encoder.pkl")

print("\n✅ Model trained successfully!")