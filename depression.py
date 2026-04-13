import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

df = pd.read_csv("depression_survey.csv")

df.columns = df.columns.str.strip()
df.fillna(0, inplace=True)

for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

def depression_score(row):
    score = 0

    score += row["q1_sad"] * 1.5
    score += row["q2_interest"] * 1.5

    score += row["q3_sleep"]
    score += row["q4_energy"]
    score += row["q5_appetite"]
    score += row["q6_guilt"]
    score += row["q7_focus"]
    score += row["q8_movement"]

    score += row["q9_selfharm"] * 2.5

    return score

df["raw_score"] = df.apply(depression_score, axis=1)

df["depression_score"] = 10 * (df["raw_score"] - df["raw_score"].min()) / (
    df["raw_score"].max() - df["raw_score"].min()
)

def label(score):
    if score < 3:
        return "Low"
    elif score < 6:
        return "Moderate"
    else:
        return "High"

df["Depression_Label"] = df["depression_score"].apply(label)

print("📊 Label Distribution:")
print(df["Depression_Label"].value_counts())

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df["Depression_Label"] = le.fit_transform(df["Depression_Label"])

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

X = df.drop(["Depression_Label", "raw_score"], axis=1)
y = df["Depression_Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(
    n_estimators=120,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=4,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

print("\n📈 Test Accuracy:")
print(accuracy_score(y_test, y_test_pred))

print("\n📈 Train Accuracy:")
print(accuracy_score(y_train, y_train_pred))

print("\n📊 Classification Report:")
print(classification_report(y_test, y_test_pred, zero_division=0))


import pandas as pd

importance = pd.Series(model.feature_importances_, index=X.columns)
print("\n🔥 Feature Importance:")
print(importance.sort_values(ascending=False))

import joblib

joblib.dump(model, "depression_model.pkl")
joblib.dump(le, "depression_encoder.pkl")

print("\n✅ Depression model trained successfully!")