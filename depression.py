# train_model.py

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# ─── 1. Generate Synthetic Dataset (BETTER) ─────────────────

np.random.seed(42)
n = 500

df = pd.DataFrame({
    "q1_sad": np.random.randint(0, 4, n),
    "q2_interest": np.random.randint(0, 4, n),
    "q3_sleep": np.random.randint(0, 4, n),
    "q4_energy": np.random.randint(0, 4, n),
    "q5_appetite": np.random.randint(0, 4, n),
    "q6_guilt": np.random.randint(0, 4, n),
    "q7_focus": np.random.randint(0, 4, n),
    "q8_movement": np.random.randint(0, 4, n),
    "q9_selfharm": np.random.randint(0, 4, n)
})

# ─── 2. Create Hidden True Score (with noise) ─────────────────

true_score = (
    df["q1_sad"] * 1.2 +
    df["q2_interest"] * 1.3 +
    df["q9_selfharm"] * 2.0 +
    df.sum(axis=1) * 0.5
)

# Add randomness (IMPORTANT 🔥)
true_score += np.random.normal(0, 2, size=n)

# ─── 3. Create Labels (NOT PERFECT anymore) ─────────────────

def label(score):
    if score < np.percentile(true_score, 33):
        return "Low"
    elif score < np.percentile(true_score, 66):
        return "Moderate"
    else:
        return "High"

df["Depression_Label"] = true_score.apply(label)

# ─── 4. Encode Labels ─────────────────

le = LabelEncoder()
df["Depression_Label"] = le.fit_transform(df["Depression_Label"])

# ─── 5. Train/Test Split ─────────────────

X = df.drop("Depression_Label", axis=1)
y = df["Depression_Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ─── 6. Train Model ─────────────────

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)

# ─── 7. Evaluate ─────────────────

y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

print("\n📈 Test Accuracy:")
print(accuracy_score(y_test, y_test_pred))

print("\n📈 Train Accuracy:")
print(accuracy_score(y_train, y_train_pred))

print("\n📊 Classification Report:")
print(classification_report(y_test, y_test_pred))

# ─── 8. Feature Importance ─────────────────

importance = pd.Series(model.feature_importances_, index=X.columns)
print("\n🔥 Feature Importance:")
print(importance.sort_values(ascending=False))

# ─── 9. Save Model ─────────────────

joblib.dump(model, "depression_model.pkl")
joblib.dump(le, "depression_encoder.pkl")

print("\n✅ Model trained successfully!")