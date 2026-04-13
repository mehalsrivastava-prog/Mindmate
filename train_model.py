import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GridSearchCV
import joblib

# ─── 1. Load Dataset ────────────────────────────────────────
try:
    df = pd.read_csv("Student Depression Dataset.csv")
except FileNotFoundError:
    print(" Student Depression Dataset.csv not found. Creating synthetic dataset...")
    # Create synthetic dataset for demonstration
    np.random.seed(42)
    n_samples = 1000
    
    data = {
        'Age': np.random.randint(18, 25, n_samples),
        'Gender': np.random.choice(['Male', 'Female'], n_samples),
        'City': np.random.choice(['Mumbai', 'Delhi', 'Bangalore', 'Chennai', 'Kolkata'], n_samples),
        'Profession': np.random.choice(['Student'], n_samples),
        'Academic Pressure': np.random.randint(1, 6, n_samples),
        'Work Pressure': np.random.randint(1, 6, n_samples),
        'CGPA': np.random.uniform(5.0, 10.0, n_samples),
        'Study Satisfaction': np.random.randint(1, 6, n_samples),
        'Job Satisfaction': np.random.randint(1, 6, n_samples),
        'Sleep Duration': np.random.choice(['Less than 5 hours', '5-6 hours', '7-8 hours', 'More than 8 hours'], n_samples),
        'Dietary Habits': np.random.choice(['Healthy', 'Moderate', 'Unhealthy'], n_samples),
        'Degree': np.random.choice(['B.Tech', 'M.Tech', 'BCA', 'MCA', 'B.Sc', 'M.Sc'], n_samples),
        'Have you ever had suicidal thoughts ?': np.random.choice(['Yes', 'No'], n_samples),
        'Work/Study Hours': np.random.randint(1, 12, n_samples),
        'Financial Stress': np.random.randint(1, 6, n_samples),
        'Family History of Mental Illness': np.random.choice(['Yes', 'No'], n_samples),
        'Depression': np.random.randint(0, 2, n_samples)
    }
    
    df = pd.DataFrame(data)
    print(f" Created synthetic dataset with {n_samples} samples")
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

# ─── 9. BASE MODEL FOR COMPARISON
base_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight="balanced")
base_model.fit(X_train_scaled, y_train)
base_pred = base_model.predict(X_test_scaled)
base_accuracy = accuracy_score(y_test, base_pred)

# ─── 10. HYPERPARAMETER TUNING
print("\n" + "="*50)
print(" HYPERPARAMETER TUNING")
print("="*50)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42, class_weight="balanced")
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                          cv=3, n_jobs=-1, verbose=1, scoring='accuracy')

grid_search.fit(X_train_scaled, y_train)

# Best model
model = grid_search.best_estimator_
print(f"\nBest parameters: {grid_search.best_params_}")

# ─── 11. PERFORMANCE COMPARISON
print("\n" + "="*50)
print(" PERFORMANCE COMPARISON")
print("="*50)

# Base model performance
base_pred = base_model.predict(X_test_scaled)
base_accuracy = accuracy_score(y_test, base_pred)

# Tuned model performance
y_pred = model.predict(X_test_scaled)
tuned_accuracy = accuracy_score(y_test, y_pred)

print(f"Base Model Accuracy: {base_accuracy:.3f}")
print(f"Tuned Model Accuracy: {tuned_accuracy:.3f}")
print(f"Improvement: {(tuned_accuracy - base_accuracy):.3f} ({((tuned_accuracy - base_accuracy) / base_accuracy * 100):.1f}%)")

print("\nClassification Report (Tuned Model):\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix (Tuned Model):\n", confusion_matrix(y_test, y_pred))

# ─── 12. Save ───────────────────────────────────────────────
joblib.dump(model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n Optimized Random Forest model saved!")
print(df["label"].value_counts())

# ─── 13. Test compatibility with predict.py
print("\n" + "="*50)
print(" TESTING COMPATIBILITY WITH PREDICT.PY")
print("="*50)

# Load the saved model and scaler
test_model = joblib.load("model.pkl")
test_scaler = joblib.load("scaler.pkl")

# Sample input (same format as predict.py expects)
sample_input = {
    "age": 20,
    "sleep": 7,
    "work_hours": 8,
    "activity": 5,
    "social": 6,
    "stress_self": 4
}

# Create features array in same order as predict.py
test_features = np.array([[
    sample_input["age"],
    sample_input["sleep"],
    sample_input["work_hours"],
    sample_input["activity"],
    sample_input["social"],
    sample_input["stress_self"]
]])

# Scale features (same as predict.py)
test_features_scaled = test_scaler.transform(test_features)

# Make prediction (same as predict.py)
test_prediction = test_model.predict(test_features_scaled)[0]
test_proba = test_model.predict_proba(test_features_scaled)[0]
test_confidence = int(max(test_proba) * 100)

print(f"Sample input: {sample_input}")
print(f"Prediction: {test_prediction}")
print(f"Confidence: {test_confidence}%")
print(" Model and scaler are compatible with predict.py! ")
print("="*50)