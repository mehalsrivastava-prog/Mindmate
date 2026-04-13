# train_model.py (Improved Decision Tree)

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# ─── 1. Generate Dataset ─────────────────
np.random.seed(42)
n = 800   

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

true_score = (
    df["q1_sad"] * 1.5 +
    df["q2_interest"] * 1.5 +
    df["q9_selfharm"] * 3.0 +   
    df["q6_guilt"] * 1.2 +
    df.sum(axis=1) * 0.4
)

true_score += np.random.normal(0, 1.5, size=n)  

def label(score):
    if score < np.percentile(true_score, 33):
        return "Low"
    elif score < np.percentile(true_score, 66):
        return "Moderate"
    else:
        return "High"

df["Depression_Label"] = true_score.apply(label)

le = LabelEncoder()
df["Depression_Label"] = le.fit_transform(df["Depression_Label"])

X = df.drop("Depression_Label", axis=1)
y = df["Depression_Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = DecisionTreeClassifier(
    max_depth=6,              
    min_samples_split=20,     
    min_samples_leaf=10,      
    class_weight="balanced",  
    random_state=42
)

model.fit(X_train, y_train)

y_test_pred = model.predict(X_test)
y_train_pred = model.predict(X_train)

print("\nTest Accuracy:")
print(accuracy_score(y_test, y_test_pred))

print("\nTrain Accuracy:")
print(accuracy_score(y_train, y_train_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred))

joblib.dump(model, "depression_model.pkl")
joblib.dump(le, "depression_encoder.pkl")

print("\nDecision Tree trained!")