"""
train_model.py
--------------
Generates a synthetic mental health dataset, trains a KNN classifier,
and saves the model + scaler to disk using joblib.
"""

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
import os

# ─── 1. Generate Synthetic Dataset ───────────────────────────────────────────
np.random.seed(42)
n_samples = 1000

# Features: age, sleep_hours, work_hours, activity, social, stress_self
data = {
    "age":         np.random.randint(18, 65, n_samples),
    "sleep":       np.round(np.random.uniform(3, 10, n_samples), 1),
    "work_hours":  np.random.randint(4, 16, n_samples),
    "activity":    np.round(np.random.uniform(0, 14, n_samples), 1),
    "social":      np.random.randint(1, 11, n_samples),
    "stress_self": np.random.randint(1, 11, n_samples),
}
df = pd.DataFrame(data)

# ─── 2. Create Labels (rule-based heuristic) ─────────────────────────────────
def label_stress(row):
    score = 0
    # Poor sleep raises stress
    if row["sleep"] < 5:
        score += 3
    elif row["sleep"] < 7:
        score += 1
    # Long work hours raise stress
    if row["work_hours"] > 12:
        score += 3
    elif row["work_hours"] > 9:
        score += 1
    # Low activity raises stress
    if row["activity"] < 2:
        score += 2
    # Low social interaction raises stress
    if row["social"] <= 3:
        score += 2
    elif row["social"] >= 7:
        score -= 1
    # Self-reported stress is the strongest signal
    score += int(row["stress_self"] * 0.8)
    # Older age adds slight vulnerability
    if row["age"] > 50:
        score += 1

    if score <= 5:
        return "Low"
    elif score <= 9:
        return "Medium"
    else:
        return "High"

df["label"] = df.apply(label_stress, axis=1)

print("Label distribution:")
print(df["label"].value_counts())

# ─── 3. Prepare Features & Labels ────────────────────────────────────────────
X = df[["age", "sleep", "work_hours", "activity", "social", "stress_self"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─── 4. Feature Scaling ───────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ─── 5. Train KNN Model ───────────────────────────────────────────────────────
# k=7 is a good default for balanced datasets of this size
knn = KNeighborsClassifier(n_neighbors=7, metric="euclidean")
knn.fit(X_train_scaled, y_train)

# ─── 6. Evaluate ──────────────────────────────────────────────────────────────
y_pred = knn.predict(X_test_scaled)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ─── 7. Save Model & Scaler ───────────────────────────────────────────────────
os.makedirs(os.path.dirname(__file__) if os.path.dirname(__file__) else ".", exist_ok=True)
joblib.dump(knn,    "model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("\n✅ model.pkl and scaler.pkl saved successfully!")
