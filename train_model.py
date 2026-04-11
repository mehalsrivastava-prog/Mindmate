import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
import joblib

# ─── 1. Load Dataset ────────────────────────────────────────
df = pd.read_csv("Student Depression Dataset.csv")
df.columns = df.columns.str.strip()

# ─── 2. Clean ───────────────────────────────────────────────
df = df.dropna()

# ─── 3. Encode categorical ─────────────────────────────────
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == "object":
        df[col] = le.fit_transform(df[col])

# ─── 4. Feature Mapping ─────────────────────────────────────
def convert_sleep(val):
    val = str(val)
    if "Less" in val:
        return 4
    elif "5-6" in val:
        return 5.5
    elif "7-8" in val:
        return 7.5
    elif "More" in val:
        return 9
    return 6

df["Sleep Duration"] = df["Sleep Duration"].apply(convert_sleep)

df["age"] = df["Age"]
df["sleep"] = df["Sleep Duration"]
df["work_hours"] = df["Work/Study Hours"]

df["activity"] = df["Dietary Habits"] + df["CGPA"] * 0.3
df["social"] = df["Study Satisfaction"]+df["Financial Stress"]
df["stress_self"] = df["Academic Pressure"]* 1.3

# ─── 5. Stress Label ────────────────────────────────────────
def create_stress_label(row):
    score = 0

    if row["sleep"] < 5:
        score += 4
    elif row["sleep"] < 7:
        score += 2

    if row["work_hours"] > 10:
        score += 3
    elif row["work_hours"] > 8:
        score += 1

    score += row["stress_self"] * 1.5
    score += (10 - row["social"]) * 1.2
    score += (10 - row["activity"]) * 0.8

    if row["Depression"] == 1:
        score += 4

    return score

# Use percentiles 🔥
df["score"] = df.apply(create_stress_label, axis=1)
low_thresh = df["score"].quantile(0.30)
high_thresh = df["score"].quantile(0.70)

def assign_label(score):
    if score < low_thresh:
        return "Low"
    elif score < high_thresh:
        return "Medium"
    else:
        return "High"

df["label"] = df["score"].apply(assign_label)

# ─── 6. Features ────────────────────────────────────────────
X = df[["age", "sleep", "work_hours", "activity", "social", "stress_self"]]
y = df["label"]

# ─── 7. Split ───────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─── 8. Scaling ─────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─── 9. LOGISTIC REGRESSION 🔥 ──────────────────────────────
model = LogisticRegression(max_iter=2000, class_weight="balanced")

model.fit(X_train_scaled, y_train)

# ─── 10. Evaluate ───────────────────────────────────────────
y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)

print("\n🔥 Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# ─── 11. Save ───────────────────────────────────────────────
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Logistic Regression model saved!")
print(df["label"].value_counts())