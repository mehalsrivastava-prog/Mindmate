import joblib
import pandas as pd
import sys, json

# ---------------- INPUT ----------------
input_data = json.loads(sys.argv[1])

# ---------------- LOAD MODEL ----------------
master_model = joblib.load("master_model.pkl")
scaler = joblib.load("master_scaler.pkl")

# ---------------- MAP INPUTS ----------------

# DIET
diet_map = {
    "Unhealthy": 0,
    "Moderate": 0.5,
    "Healthy": 1
}
diet_score = diet_map.get(input_data["diet"], 0.5)

# DEPRESSION
dep_map = {
    "Low": 0,
    "Moderate": 0.5,
    "High": 1
}
depression_score = dep_map.get(input_data["depression"], 0.5)

# FINANCIAL (already numeric)
financial_score = float(input_data["financial"]) / 100

# ACADEMIC
acad_map = {
    "Low": 0,
    "Moderate": 0.5,
    "High": 1
}
academic_score = acad_map.get(input_data["academic"], 0.5)

# ---------------- MASTER INPUT ----------------
master_input = pd.DataFrame([{
    "diet_score": diet_score,
    "depression_score": depression_score,
    "financial_score": financial_score,
    "academic_score": academic_score
}])

# ---------------- SCALE ----------------
master_input_scaled = scaler.transform(master_input)

# ---------------- PREDICT ----------------
pred = master_model.predict(master_input_scaled)[0]
proba = master_model.predict_proba(master_input_scaled)[0]

labels = ["Low", "Moderate", "High"]

# ---------------- OUTPUT ----------------
print(json.dumps({
    "prediction": labels[pred],
    "confidence": round(max(proba), 2)
}))